import pickle
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

import omicslog
from omicslog import LoggedAnnDataStandalone, log_start


def log_of(adata) -> pd.DataFrame:
    return adata.uns["_omicslog"]


def last_message(adata) -> str:
    return log_of(adata).iloc[-1]["Message"]


@pytest.fixture
def base():
    rng = np.random.default_rng(0)
    adata = ad.AnnData(csr_matrix(rng.poisson(1, (100, 50)).astype(np.float32)))
    adata.obs_names = [f"Cell_{i}" for i in range(100)]
    adata.var_names = [f"Gene_{i}" for i in range(50)]
    return adata


@pytest.fixture
def logged(base):
    adata = log_start(base)
    rng = np.random.default_rng(1)
    adata.obs["cell_type"] = pd.Categorical(rng.choice(["B", "T", "Mono"], adata.n_obs))
    return adata


# --- log_start ------------------------------------------------------------------


def test_log_start_returns_logged_object_with_empty_log(base):
    adata = log_start(base)
    assert isinstance(adata, LoggedAnnDataStandalone)
    assert list(log_of(adata).columns) == ["Time", "Operation", "Message"]
    assert log_of(adata).empty


def test_log_start_does_not_modify_original(base, logged):
    assert "cell_type" not in base.obs
    assert "_omicslog" not in base.uns


def test_log_start_is_idempotent(logged):
    assert log_start(logged) is logged


def test_uncopyable_uns_entry_is_kept_with_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adata = log_start(ad.AnnData(np.ones((2, 2)), uns={"gen": (x for x in [])}))
    assert "gen" in adata.uns
    assert any("deep-copied" in str(w.message) for w in caught)


# --- obs / var ----------------------------------------------------------------


def test_obs_column_added_is_logged(logged):
    row = log_of(logged).iloc[-1]
    assert (row["Operation"], row["Message"]) == ("obs", "'cell_type' added")


def test_obs_column_updated_is_logged(logged):
    logged.obs["cell_type"] = "B"
    assert last_message(logged) == "'cell_type' updated"


def test_var_column_added_is_logged(logged):
    logged.var["highly_variable"] = True
    row = log_of(logged).iloc[-1]
    assert (row["Operation"], row["Message"]) == ("var", "'highly_variable' added")


def test_obs_column_deleted_is_logged(logged):
    logged.obs["tmp"] = 1
    del logged.obs["tmp"]
    assert "tmp" not in logged.obs
    assert last_message(logged) == "'tmp' removed"


def test_loc_assignment_persists(logged):
    first = logged.obs_names[0]
    logged.obs.loc[first, "cell_type"] = "T"
    assert logged.obs.loc[first, "cell_type"] == "T"


def test_inplace_rename_persists(logged):
    logged.obs["old"] = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # pandas >= 3.1 deprecates inplace=True
        logged.obs.rename(columns={"old": "new"}, inplace=True)
    assert "new" in logged.obs.columns
    assert "old" not in logged.obs.columns


def test_obs_reassignment_keeps_logging(logged):
    logged.obs = pd.DataFrame(index=logged.obs_names)
    logged.obs["z"] = 0
    assert last_message(logged) == "'z' added"


# --- dict-like components -------------------------------------------------------


def test_layers_obsm_varm_are_logged(logged):
    logged.layers["log"] = np.log1p(logged.X)
    logged.obsm["X_umap"] = np.zeros((logged.n_obs, 2))
    logged.varm["loadings"] = np.zeros((logged.n_vars, 3))
    assert log_of(logged)["Operation"].tolist()[-3:] == ["layers", "obsm", "varm"]
    assert log_of(logged)["Message"].tolist()[-3:] == [
        "'log' added",
        "'X_umap' added",
        "'loadings' added",
    ]


def test_layer_deleted_is_logged(logged):
    logged.layers["log"] = np.log1p(logged.X)
    del logged.layers["log"]
    assert "log" not in logged.layers
    assert last_message(logged) == "'log' removed"


# --- subsetting -----------------------------------------------------------------


def test_obs_subset_returns_logged_object(logged):
    mask = logged.obs.cell_type == "B"
    subset = logged[mask]
    assert isinstance(subset, LoggedAnnDataStandalone)
    assert subset.n_obs == mask.sum()
    assert "cell(s) remaining" in last_message(subset)


def test_subset_does_not_extend_parent_log(logged):
    n_before = len(log_of(logged))
    subset = logged[logged.obs.cell_type == "B"]
    assert len(log_of(logged)) == n_before
    assert len(log_of(subset)) == n_before + 1


def test_var_subset_is_logged(logged):
    subset = logged[:, logged.var_names.str.endswith("1")]
    assert subset.n_vars == 5
    assert last_message(subset) == "removed 45 gene(s) (90%), 5 gene(s) remaining"


def test_inplace_subset_is_logged(logged):
    logged._inplace_subset_obs(slice(0, 10))
    assert logged.n_obs == 10
    assert "cell(s) remaining" in last_message(logged)


# --- operation_log / repr -------------------------------------------------------


def test_operation_log_returns_dataframe_copy(logged):
    log = logged.operation_log()
    assert isinstance(log, pd.DataFrame)
    assert len(log) == len(log_of(logged))
    log.loc[0, "Message"] = "changed"
    assert log_of(logged).loc[0, "Message"] != "changed"


def test_repr_includes_operation_log(logged):
    text = repr(logged)
    assert "Operation log:" in text
    assert "obs: 'cell_type' added" in text


# --- copy / pickle --------------------------------------------------------------


def test_copy_keeps_type_and_is_independent(logged):
    copied = logged.copy()
    assert isinstance(copied, LoggedAnnDataStandalone)
    copied.obs["new"] = 1
    assert "new" not in logged.obs
    assert last_message(copied) == "'new' added"
    assert last_message(logged) != "'new' added"


def test_pickle_roundtrip(logged):
    restored = pickle.loads(pickle.dumps(logged))
    assert log_of(restored).values.tolist() == log_of(logged).values.tolist()


# --- export / import ------------------------------------------------------------


@pytest.fixture
def rich(base):
    """Logged object exercising every string-bearing component that gets written."""
    base.raw = base.copy()
    adata = log_start(base)
    rng = np.random.default_rng(2)
    adata.obs["cell_type"] = pd.Categorical(rng.choice(["B", "T", "Mono"], adata.n_obs))
    adata.obs["label"] = [f"l{i}" for i in range(adata.n_obs)]
    adata.obs["maybe"] = pd.array([None] + ["x"] * (adata.n_obs - 1), dtype="string")
    adata.obsm["meta"] = pd.DataFrame({"s": ["q"] * adata.n_obs}, index=adata.obs_names)
    adata = adata[adata.obs.cell_type != "B"]
    adata = adata[:, adata.var_names.str.endswith("1")]
    adata.layers["log"] = np.log1p(adata.X)
    adata.uns["nested"] = {"df": pd.DataFrame({"k": ["v"]}), "n": 1}
    return adata


IO_FORMATS = [
    pytest.param("write_h5ad", omicslog.read_h5ad, "x.h5ad", id="h5ad"),
    pytest.param("write_zarr", omicslog.read_zarr, "x.zarr", id="zarr"),
]


@pytest.mark.parametrize(("write", "read", "name"), IO_FORMATS)
def test_roundtrip_preserves_data_and_log(rich, tmp_path, write, read, name):
    assert ad.settings.allow_write_nullable_strings is False  # works with defaults
    path = tmp_path / name
    getattr(rich, write)(path)
    back = read(path)

    assert isinstance(back, LoggedAnnDataStandalone)
    assert log_of(back).values.tolist() == log_of(rich).values.tolist()
    assert list(back.obs_names) == list(rich.obs_names)
    assert list(back.var_names) == list(rich.var_names)
    assert back.obs["label"].tolist() == rich.obs["label"].tolist()
    assert list(back.obs.cell_type.cat.categories) == list(rich.obs.cell_type.cat.categories)
    assert pd.isna(back.obs["maybe"].iloc[0])
    assert back.raw is not None
    assert "log" in back.layers
    assert back.obsm["meta"]["s"].iloc[0] == "q"
    assert back.uns["nested"]["df"]["k"].iloc[0] == "v"


@pytest.mark.parametrize(("write", "read", "name"), IO_FORMATS)
def test_logging_continues_after_reload(rich, tmp_path, write, read, name):
    path = tmp_path / name
    getattr(rich, write)(path)
    back = read(path)
    back.obs["after"] = 1
    assert last_message(back) == "'after' added"
    assert len(log_of(back)) == len(log_of(rich)) + 1


def test_second_roundtrip(rich, tmp_path):
    rich.write_h5ad(tmp_path / "first.h5ad")
    back = omicslog.read_h5ad(tmp_path / "first.h5ad")
    back.obs["after"] = 1
    back.write_h5ad(tmp_path / "second.h5ad")
    again = omicslog.read_h5ad(tmp_path / "second.h5ad")
    assert last_message(again) == "'after' added"


def test_plain_anndata_reader_keeps_log(rich, tmp_path):
    rich.write_h5ad(tmp_path / "x.h5ad")
    back = ad.read_h5ad(tmp_path / "x.h5ad")
    assert log_of(back).values.tolist() == log_of(rich).values.tolist()


def test_write_does_not_modify_in_memory_object(rich, tmp_path):
    index_dtype = rich.obs.index.dtype
    log_dtypes = log_of(rich).dtypes.tolist()
    rich.write_h5ad(tmp_path / "x.h5ad")
    assert rich.obs.index.dtype == index_dtype
    assert log_of(rich).dtypes.tolist() == log_dtypes
