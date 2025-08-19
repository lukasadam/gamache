import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import nbinom

from gamache.tl.backend import (
    IRLSBackend,
    apply_lineage_weights,
    block_diag,
    build_block_design,
    estimate_alpha_pearson,
    irls_weights,
    mu_eta,
    nb2_logpmf,
    penalty_matrix,
    pseudo_response,
)

# -------------------------
# Helpers / tiny generators
# -------------------------


def rng(seed=0):
    return np.random.default_rng(seed)


def simulate_nb2(mu, alpha, r):
    """
    Simulate NB2 counts with mean mu and dispersion alpha.
    NB2 -> r = 1/alpha, p = r / (r + mu)
    """
    r0 = 1.0 / alpha
    p = r0 / (r0 + mu)
    return r.negative_binomial(r0, p).astype(float)  # float for downstream math


# -----------
# Unit tests
# -----------


def test_nb2_logpmf_matches_scipy():
    r = rng(1)
    y = r.integers(0, 30, size=50).astype(float)
    mu = r.uniform(0.1, 10.0, size=50)
    alpha = 0.3
    r0 = 1.0 / alpha
    p = r0 / (r0 + mu)
    logpmf_ref = nbinom.logpmf(y, n=r0, p=p)
    logpmf_ours = nb2_logpmf(y, mu, alpha)
    assert_allclose(logpmf_ours, logpmf_ref, rtol=1e-10, atol=1e-10)


def test_mu_eta_and_shapes():
    eta = np.array([-2.0, 0.0, 3.0])
    out = mu_eta(eta)
    assert_allclose(out, np.exp(eta))
    assert out.shape == eta.shape


def test_irls_weights_limits():
    mu = np.array([0.1, 1.0, 10.0])
    # As alpha -> 0, w -> mu (with log link)
    w_small_alpha = irls_weights(mu, alpha=1e-12)
    assert_allclose(w_small_alpha, mu, rtol=1e-6, atol=1e-8)
    # Larger alpha shrinks weights toward 0
    w_big_alpha = irls_weights(mu, alpha=10.0)
    assert np.all(w_big_alpha < w_small_alpha)


def test_pseudo_response_formula():
    r = rng(2)
    eta = r.normal(size=20)
    mu = mu_eta(eta)
    y = r.poisson(mu)
    z = pseudo_response(eta, y, mu)
    assert_allclose(z, eta + (y - mu) / (mu + 1e-12))


def test_estimate_alpha_pearson_recovers_dispersion():
    r = rng(3)
    n = 2000
    mu = r.uniform(0.5, 5.0, size=n)
    alpha_true = 0.2
    y = simulate_nb2(mu, alpha_true, r)
    # Unweighted
    alpha_hat = estimate_alpha_pearson(y, mu)
    # This is a MoM—allow a fairly loose tolerance
    assert 0.05 < alpha_hat < 0.5

    # Weighted should still be reasonable
    w = r.uniform(0.2, 1.5, size=n)
    alpha_hat_w = estimate_alpha_pearson(y, mu, weights=w)
    assert 0.05 < alpha_hat_w < 0.5


def test_penalty_matrix_properties():
    k = 6
    K1 = penalty_matrix(k, order=1)
    K2 = penalty_matrix(k, order=2)
    assert K1.shape == (k, k)
    assert K2.shape == (k, k)

    # PSD checks: x^T K x >= 0
    r = rng(0)
    x = r.normal(size=k)
    for K in (K1, K2):
        val = float(x.T @ K @ x)
        assert val >= -1e-10

    # Order-1: constants have zero penalty
    ones = np.ones(k)
    assert_allclose(ones @ K1 @ ones, 0.0, atol=1e-10)
    # Order-2: linear trends have zero penalty
    lin = np.arange(k, dtype=float)
    assert_allclose(lin @ K2 @ lin, 0.0, atol=1e-10)

    # order < 1 returns zeros
    K0 = penalty_matrix(k, order=0)
    assert_allclose(K0, 0.0)


def test_block_diag():
    A = np.eye(2)
    B = 2 * np.eye(3)
    C = 3 * np.eye(1)
    BD = block_diag([A, B, C])
    assert BD.shape == (6, 6)
    # Check blocks are in the right places
    assert_allclose(BD[:2, :2], A)
    assert_allclose(BD[2:5, 2:5], B)
    assert_allclose(BD[5:, 5:], C)
    # Off-diagonal empty
    assert np.allclose(BD[:2, 2:], 0.0)


def test_apply_lineage_weights_none_returns_copy():
    X1 = np.ones((4, 3))
    X2 = 2 * np.ones((4, 2))
    out = apply_lineage_weights([X1, X2], lineage_weights=None)
    assert len(out) == 2
    assert_allclose(out[0], X1)
    assert_allclose(out[1], X2)


def test_apply_lineage_weights_scales_columns_per_lineage():
    r = rng(4)
    n = 5
    X1 = r.normal(size=(n, 2))
    X2 = r.normal(size=(n, 3))
    lw = r.uniform(0.0, 1.0, size=(n, 2))  # 2 lineages
    Xw1, Xw2 = apply_lineage_weights([X1, X2], lw)
    # Each block l uses sqrt(w[:, l]) row-wise
    assert_allclose(Xw1, X1 * np.sqrt(lw[:, [0]]))
    assert_allclose(Xw2, X2 * np.sqrt(lw[:, [1]]))


def test_build_block_design_shapes_and_slices():
    r = rng(5)
    n = 10
    X1 = r.normal(size=(n, 3))
    X2 = r.normal(size=(n, 2))
    lambdas = [0.1, 0.5]
    X, S, sl = build_block_design([X1, X2], lambdas, lineage_weights=None, penalty_order=2)
    assert X.shape == (n, 5)
    assert S.shape == (5, 5)
    assert (
        isinstance(sl, list)
        and sl[0].start == 0
        and sl[0].stop == 3
        and sl[1].start == 3
        and sl[1].stop == 5
    )
    # Basic PSD check on S
    z = r.normal(size=5)
    assert float(z.T @ S @ z) >= -1e-10


# -----------------------
# IRLSBackend with a mock
# -----------------------


def _ridge_pirls_solver(X, z, w, S):
    """
    Simple penalized WLS solver used to mock solve_penalized_wls:
      beta = (X^T W X + S)^(-1) X^T W z
      cov  = (X^T W X + S)^(-1)
    """
    W = np.diag(w)
    XtWX = X.T @ W @ X
    XtWz = X.T @ W @ z
    A = XtWX + S
    # regularize if singular
    try:
        beta = np.linalg.solve(A, XtWz)
        cov = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        A2 = A + 1e-8 * np.eye(A.shape[0])
        beta = np.linalg.solve(A2, XtWz)
        cov = np.linalg.inv(A2)
    return beta, cov


def test_irls_backend_converges_and_recovers_params(monkeypatch):
    # Patch the module-under-test's reference to solve_penalized_wls
    # IMPORTANT: patch the symbol where it is USED (your module), not where it's defined.
    import gamache.tl.irls as mut

    monkeypatch.setattr(mut, "solve_penalized_wls", _ridge_pirls_solver, raising=True)

    r = rng(6)
    n, p = 600, 3
    X = np.c_[np.ones(n), r.normal(size=(n, p - 1))]  # include intercept
    beta_true = np.array([1.0, 0.8, -0.4])
    offset = np.zeros(n)
    mu = np.exp(offset + X @ beta_true)
    alpha_true = 0.2
    y = simulate_nb2(mu, alpha_true, r)

    # small ridge penalty (keeps matrix invertible)
    S = 1e-6 * np.eye(p)

    res = IRLSBackend(max_iter=100, tol=1e-6, alpha_init=0.1, return_cov=True).fit(
        y=y, X=X, S=S, offset=offset
    )

    # Sanity checks
    assert res.beta.shape == (p,)
    assert np.isfinite(res.alpha) and res.alpha > 0
    assert 0 <= res.edf <= p + 1e-6
    assert res.cov is not None and res.cov.shape == (p, p)
    assert "converged_mu" in res.diagnostics

    # Parameter recovery (loose bounds—IRLS + MoM dispersion are approximate)
    assert_allclose(res.beta, beta_true, atol=0.35)
    assert 0.05 < res.alpha < 0.5


def test_irls_backend_all_zero_obs_weights_gives_edf_zero(monkeypatch):
    import gamache.tl.irls as mut

    monkeypatch.setattr(mut, "solve_penalized_wls", _ridge_pirls_solver, raising=True)

    n, p = 50, 2
    X = np.c_[np.ones(n), np.linspace(-1, 1, n)]
    offset = np.zeros(n)
    y = np.zeros(n)  # arbitrary; weights will zero everything
    S = 1e-6 * np.eye(p)

    res = IRLSBackend().fit(y=y, X=X, S=S, offset=offset, obs_weights=np.zeros(n))
    assert res.edf == 0.0
    assert_allclose(res.beta, np.zeros(p), atol=1e-12)
