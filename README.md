# Getting started with omicslog

**omicslog** adds automatic provenance tracking to [AnnData](https://anndata.readthedocs.io) objects. Once logging is started, every change you make to the object (adding or removing annotations, filtering cells or genes, storing embeddings or new layers) is recorded in an operation log stored inside the object itself, in `adata.uns["_omicslog"]`.

Each log entry has three fields:

| Field | Description |
|---|---|
| `Time` | When the operation happened (`YYYY-MM-DD HH:MM:SS`). |
| `Operation` | The part of the object that changed: `obs`, `var`, `obsm`, `varm`, `obsp`, `varp`, `layers`, or `subset`. |
| `Message` | What happened, for example `'cell_type' added` or `removed 73 cell(s) (73%), 27 cell(s) remaining`. |

Because the log lives in `.uns`, it is saved together with your data when you write the object to `.h5ad` or zarr, so the history of an analysis travels with its results.

## Installation

```bash
pip install omicslog
```

## Importing packages


```python
import omicslog
from omicslog import log_start
import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix
```

## Creating an example dataset

We start from a plain AnnData object with 100 cells and 2,000 genes of random counts.

> **Note:** The example data is adapted from the [AnnData getting-started tutorial](https://anndata.readthedocs.io/en/latest/tutorials/notebooks/getting-started.html).


```python
counts = csr_matrix(np.random.poisson(1, size=(100, 2000)), dtype=np.float32)
adata = ad.AnnData(counts)

adata.obs_names = [f"Cell_{i:d}" for i in range(adata.n_obs)]
adata.var_names = [f"Gene_{i:d}" for i in range(adata.n_vars)]
adata
```




    AnnData object with n_obs × n_vars = 100 × 2000



## Starting the log

`log_start()` returns a logged copy of an existing AnnData object; the original `adata` is left untouched. The new object behaves exactly like an AnnData and works with the usual tools (for example scanpy), but it starts with an empty operation log in `.uns["_omicslog"]`. When printed with `print(logdata)`, it also shows the log below the usual AnnData summary.


```python
logdata = log_start(adata)

logdata.uns["_omicslog"]
```




    Empty DataFrame
    Columns: [Time, Operation, Message]
    Index: []



## Adding cell annotations (`.obs`)

Adding, updating, or deleting a column in `.obs` or `.var` creates a log entry.


```python
ct = np.random.choice(["B", "T", "Monocyte"], size=(logdata.n_obs,))
logdata.obs["cell_type"] = pd.Categorical(ct)  # Categoricals are preferred for efficiency

logdata.uns["_omicslog"]
```




                      Time Operation            Message
    0  2026-11-18 12:12:49       obs  'cell_type' added



## Filtering cells (`.obs`)

Slicing a logged object returns a new logged object. Its log contains the full history of the parent plus an entry with the number and percentage of cells removed. The parent's log is not modified.


```python
logdata = logdata[logdata.obs.cell_type == "B"]

logdata.uns["_omicslog"]
```




                      Time Operation  \
    0  2026-11-18 12:12:49       obs   
    1  2026-11-18 12:12:49    subset   
    
                                              Message  
    0                               'cell_type' added  
    1  removed 64 cell(s) (64%), 36 cell(s) remaining  



## Filtering genes (`.var`)

Filtering genes works the same way and records how many genes were removed.


```python
logdata = logdata[:,logdata.var_names.str.endswith("1")]

logdata.uns["_omicslog"]
```




                      Time Operation  \
    0  2026-11-18 12:12:49       obs   
    1  2026-11-18 12:12:49    subset   
    2  2026-11-18 12:12:49    subset   
    
                                                 Message  
    0                                  'cell_type' added  
    1     removed 64 cell(s) (64%), 36 cell(s) remaining  
    2  removed 1800 gene(s) (90%), 200 gene(s) remaining  



## Adding embeddings and multidimensional annotations (`.obsm`, `.varm`)

`.obsm` stores per-cell matrices, such as PCA or UMAP coordinates, and `.varm` stores per-gene matrices, such as PCA loadings. Adding or removing an entry in either of them is logged, and so are pairwise matrices in `.obsp` and `.varp` (for example neighbour graphs).


```python
logdata.obsm["X_umap"] = np.random.normal(0, 1, size=(logdata.n_obs, 2))
logdata.varm["gene_stuff"] = np.random.normal(0, 1, size=(logdata.n_vars, 5))

logdata.uns["_omicslog"]
```




                      Time Operation  \
    0  2026-11-18 12:12:49       obs   
    1  2026-11-18 12:12:49    subset   
    2  2026-11-18 12:12:49    subset   
    3  2026-11-18 12:12:49      obsm   
    4  2026-11-18 12:12:49      varm   
    
                                                 Message  
    0                                  'cell_type' added  
    1     removed 64 cell(s) (64%), 36 cell(s) remaining  
    2  removed 1800 gene(s) (90%), 200 gene(s) remaining  
    3                                     'X_umap' added  
    4                                 'gene_stuff' added  



## Adding layers

Layers hold alternative versions of the expression matrix, such as normalised or log-transformed counts. Each new layer is logged.


```python
logdata.layers["log_transformed"] = np.log1p(logdata.X)

logdata.uns["_omicslog"]
```




                      Time Operation  \
    0  2026-11-18 12:12:49       obs   
    1  2026-11-18 12:12:49    subset   
    2  2026-11-18 12:12:49    subset   
    3  2026-11-18 12:12:49      obsm   
    4  2026-11-18 12:12:49      varm   
    5  2026-11-18 12:12:49    layers   
    
                                                 Message  
    0                                  'cell_type' added  
    1     removed 64 cell(s) (64%), 36 cell(s) remaining  
    2  removed 1800 gene(s) (90%), 200 gene(s) remaining  
    3                                     'X_umap' added  
    4                                 'gene_stuff' added  
    5                            'log_transformed' added  



## Accessing the log

The log is a regular pandas DataFrame, so it can be filtered, exported, or included in a report. `operation_log()` returns a copy of it, which you can modify without changing the log stored in the object.


```python
log = logdata.operation_log()
log[log["Operation"] == "subset"]
```




                      Time Operation  \
    1  2026-11-18 12:12:49    subset   
    2  2026-11-18 12:12:49    subset   
    
                                                 Message  
    1     removed 64 cell(s) (64%), 36 cell(s) remaining  
    2  removed 1800 gene(s) (90%), 200 gene(s) remaining  



## Import and export

`write_h5ad()` (and `write_zarr()`) save the log together with the data. To keep logging after loading, read the file with `omicslog.read_h5ad()` (or `omicslog.read_zarr()`): it returns a logged object whose new entries are added to the saved history.

Reading the file with `anndata.read_h5ad()` also works, but returns a plain AnnData. The saved log is still available in `.uns["_omicslog"]`, but new changes are not logged.


```python
logdata.write_h5ad("test_omicslog.h5ad")
```


```python
new_adata = omicslog.read_h5ad("test_omicslog.h5ad")
new_adata.obs["batch"] = "batch_1"  # logging continues after loading

new_adata.uns["_omicslog"]
```




                      Time Operation  \
    0  2026-11-18 12:12:49       obs   
    1  2026-11-18 12:12:49    subset   
    2  2026-11-18 12:12:49    subset   
    3  2026-11-18 12:12:49      obsm   
    4  2026-11-18 12:12:49      varm   
    5  2026-11-18 12:12:49    layers   
    6  2026-11-18 12:12:49       obs   
    
                                                 Message  
    0                                  'cell_type' added  
    1     removed 64 cell(s) (64%), 36 cell(s) remaining  
    2  removed 1800 gene(s) (90%), 200 gene(s) remaining  
    3                                     'X_umap' added  
    4                                 'gene_stuff' added  
    5                            'log_transformed' added  
    6                                      'batch' added  


