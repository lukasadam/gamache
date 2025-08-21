# Usage

Basic usage example:

```python
import numpy as np
import scanpy as sc
import gamache as gm

adata = sc.read("paul15_endo.h5ad")

# Make sure to use the raw counts 
# for the fitting step
adata = adata.raw.to_adata()

# Make sure that genes are detected 
sc.pp.filter_genes(adata, min_cells=50)

# Fit GAM (negative binomial with default settings)
model = gm.tl.fit_gam(adata)

# Assess goodness of fit for each gene
gof_df = model.deviance_explained()

# Test association for each gene with pseudotime
# H0: all smooth coefficients == 0
results_df = model.test_all(test="association")

# Plot fit for a single gene and color by grouping
gm.pl.plot_gene_fit(model, "Fam132a", color="paul15_clusters")

# Plot fit for multiple genes as a heatmap
gm.pl.plot_gene_heatmap(adata, ["Fam132a"], model=model)
```