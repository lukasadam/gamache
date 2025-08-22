<img  src="docs/assets/images/gamache_logo.png" alt="gamache logo" width="250">

[![gamache CI/CD (uv)](https://github.com/lukasadam/gamache/actions/workflows/python-ci-cd.yml/badge.svg)](https://github.com/lukasadam/gamache/actions/workflows/python-ci-cd.yml)
[![codecov](https://codecov.io/gh/lukasadam/gamache/graph/badge.svg?token=079DFV3CXJ)](https://codecov.io/gh/lukasadam/gamache)
[![Docs](https://img.shields.io/badge/docs-mkdocs--material-blue)](https://lukasadam.github.io/gamache/)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![Ruff](https://img.shields.io/badge/lint-ruff-1f6feb)](https://docs.astral.sh/ruff/)
[![Bandit](https://img.shields.io/badge/security-bandit-f58549)](https://bandit.readthedocs.io)

# Gamache

**gamache** (`"ga-mash"` | /ɡəˈmɑːʃ/) — a lightweight toolkit for **GAM-based pseudotime testing and visualization** in Python.

> The name is inspired by the smooth chocolate glaze. Just as ganache blends ingredients into a seamless emulsion, **Gamache** blends generalized additive models with pseudotime analysis to create smooth, interpretable gene expression trends.

## Motivation

In the Python single-cell ecosystem, existing pseudotime tools such as CellRank [CellRank](https://cellrank.readthedocs.io/en/latest/) and scFates [scFates](https://scfates.readthedocs.io/en/latest/) focus on advanced trajectory inference methods (e.g. Markov chains, principal graphs). While powerful, they don’t provide the minimal pseudotime testing workflow offered by [tradeSeq](https://github.com/statOmics/tradeSeq) in R: fitting GAMs on a single lineage, testing for association with pseudotime, and visualizing gene expression trends directly.

Gamache aims to fill this gap by providing a **lightweight, tradeSeq-style workflow** for pseudotime testing and visualization around `AnnData`.

---

## Features

- Generalized Additive Model (GAM) fitting utilities
- Association testing / Start vs. End testing along pseudotime
- Plotting helpers

## Roadmap

- Clustering of variables features
- Between lineages DE testing (diffEnd test, pattern test, earlyDE test)

## Project Status

🚧 **Gamache** is under active development. APIs may change without notice, and breaking changes are expected as the project evolves.

Feedback and issues are very welcome at this stage!

## Installation

Using **uv** (recommended for speed):

```bash
uv pip install git+https://github.com/lukasadam/gamache.git@main
```

or plain pip:

```bash
pip install git+https://github.com/lukasadam/gamache.git@main
```

## ⚡ Quickstart

```python
import numpy as np
import scanpy as sc
import gamache as gm

# Fit NB-GAM
model = gm.tl.fit_gam(adata)

# Test association for each gene with pseudotime
# H0: all smooth coefficients == 0
results_df = model.test_all(test="association")

# Filter results to significant genes
results_df = results_df[results_df["qvalue"] < 0.05].set_index("gene")

# Plot fit for multiple genes as a heatmap
gm.pl.plot_gene_heatmap(adata, list(results_df.index), model=model)
```

## Documentation

- User & API docs (MkDocs): **https://gamache.readthedocs.io/en/latest/**

## Contributing

Contributions are very welcome! 🎉 You can help by reporting issues, improving documentation, or submitting pull requests.

### Development Setup
```bash
# Run everything out of the box
make bootstrap

# Run linting
make lint

# Run tests
make test
```

### Pull Requests

- Please keep PRs focused and small when possible.
- Run lint tests before submitting.
- Update documentation for any user-facing changes.
- Include a clear description of what the change does and why it’s needed.
