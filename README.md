<img  src="docs/assets/gamache_logo.png" alt="gamache logo" width="300">

[![gamache CI/CD (uv)](https://github.com/lukasadam/gamache/actions/workflows/python-ci-cd.yml/badge.svg)](https://github.com/lukasadam/gamache/actions/workflows/python-ci-cd.yml)
[![Coverage](https://codecov.io/gh/lukasadam/gamache/branch/develop/graph/badge.svg)](https://codecov.io/gh/lukasadam/gamache)
[![Docs](https://img.shields.io/badge/docs-mkdocs--material-blue)](https://lukasadam.github.io/gamache/)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![Ruff](https://img.shields.io/badge/lint-ruff-1f6feb)](https://docs.astral.sh/ruff/)
[![Bandit](https://img.shields.io/badge/security-bandit-f58549)](https://bandit.readthedocs.io)

# Gamache

**gamache** (`"ga-mash"` | /ɡəˈmɑːʃ/) — a lightweight toolkit for **GAM-based pseudotime testing and visualization** in Python.

## Motivation

In the Python single-cell ecosystem, existing pseudotime tools such as CellRank [CellRank](https://cellrank.readthedocs.io/en/latest/) and scFates [scFates](https://scfates.readthedocs.io/en/latest/) focus on advanced trajectory inference methods (e.g. Markov chains, principal graphs). While powerful, they don’t provide the minimal pseudotime testing workflow offered by [tradeSeq](https://github.com/statOmics/tradeSeq) in R: fitting GAMs on a single lineage, testing for association with pseudotime, and visualizing gene expression trends directly.

Gamache aims to fill this gap by providing a **lightweight, tradeSeq-style workflow** for pseudotime testing and visualization around `AnnData`.

---

## Features

- Generalized Additive Model (GAM) fitting utilities
- Association testing along pseudotime / trajectories
- Block-design construction with block-diagonal penalties
- IRLS-based fitting backend
- Plotting helpers

## Installation

Using **uv** (recommended):

```bash
uv pip install -e .
```

or plain pip:

```bash
pip install -e .
```

## Quickstart

```python
import numpy as np
import scanpy as sc
import gamache as gm

adata = sc.read("paul15_endo.h5ad")
adata = adata.raw.to_adata()

# Fit GAM
model = gm.tl.fit_gam(adata, backend="irls")

# Perform association testing
model.association_test(gene="Hba-a2")
```

## Documentation

- User & API docs (MkDocs): **https://<ORG>.github.io/gamache/**
- Notebooks: `docs/notebooks/` and `notebooks/`

## Development

```bash
# The repo comes with all packages out of the box
make bootstrap
```

## Contributing

Issues and PRs are welcome! Please run lint, tests, and update docs for user-facing changes.

## License

Add a LICENSE file (e.g., MIT) to clarify usage and contributions.
