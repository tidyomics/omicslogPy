from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import anndata as ad
import pandas as pd

try:  # accessor references only exist in newer anndata releases
    from anndata.acc import AdRef, RefAcc

    _ACCESSOR_TYPES: tuple[type, ...] = (AdRef, RefAcc)
except ImportError:
    _ACCESSOR_TYPES = ()

LOG_KEY = "_omicslog"
LOG_COLUMNS = ["Time", "Operation", "Message"]

def _safe_deepcopy_dict(d: dict) -> dict:
    result = {}
    for k, v in d.items():
        try:
            result[k] = deepcopy(v)
        except Exception:
            warnings.warn(
                f"uns[{k!r}] could not be deep-copied; the new object shares it with the original.",
                stacklevel=3,
            )
            result[k] = v
    return result


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _format_log_message(operation: str, message: str, ts: str | None = None) -> list[str]:
    stamp = ts or _timestamp()
    return [stamp, operation, message]


def _ensure_log_container(adata: ad.AnnData) -> pd.DataFrame:
    current = adata.uns.get(LOG_KEY)
    if not isinstance(current, pd.DataFrame):
        adata.uns[LOG_KEY] = pd.DataFrame(columns=LOG_COLUMNS)
    return adata.uns[LOG_KEY]


def _append_log_messages(adata: ad.AnnData, messages: list[list[str]]) -> None:
    if not messages:
        return
    container = _ensure_log_container(adata)
    new_rows = pd.DataFrame(messages, columns=LOG_COLUMNS)
    adata.uns[LOG_KEY] = pd.concat([container, new_rows], ignore_index=True)

def _inherit_and_append(
    parent: ad.AnnData,
    child: ad.AnnData,
    messages: list[list[str]],
) -> None:
    child.uns[LOG_KEY] = _ensure_log_container(parent)
    _append_log_messages(child, messages)

def _parent_set(obj, attr: str, value) -> None:
    """Set an attribute via the first parent class that defines it.
    Works for both plain properties (.fset) and custom descriptors (.__set__).
    """
    for base in type(obj).__mro__[1:]:
        if attr in base.__dict__:
            base.__dict__[attr].__set__(obj, value)
            return
    object.__setattr__(obj, attr, value)

class _LoggingDataFrame(pd.DataFrame):
    """DataFrame subclass that logs column mutations to the owning AnnData.

    Passes anndata's `isinstance(_, pd.DataFrame)` accessor checks (unlike a
    generic proxy) while still intercepting __setitem__/__delitem__.

    The instance is stored as the AnnData's real ``_obs``/``_var``, so every
    mutation (``.loc``, ``inplace=True``, ...) persists; only column
    assignment and deletion are logged.
    """
    _metadata = ["_owner", "_label"]

    @property
    def _constructor(self):
        # derived frames (slices, ops) are plain DataFrames, not logged
        return pd.DataFrame

    def __setitem__(self, key, value):
        owner = object.__getattribute__(self, "_owner")
        label = object.__getattribute__(self, "_label")
        existing = self.columns
        verb = "updated" if (isinstance(key, str) and key in existing) else "added"
        super().__setitem__(key, value)
        if isinstance(key, str):
            _append_log_messages(owner, [_format_log_message(label, f"'{key}' {verb}")])

    def __delitem__(self, key):
        owner = object.__getattribute__(self, "_owner")
        label = object.__getattribute__(self, "_label")
        super().__delitem__(key)
        _append_log_messages(owner, [_format_log_message(label, f"'{key}' removed")])


def _attached_frame(owner: ad.AnnData, label: str) -> pd.DataFrame:
    """Return the owner's real obs/var, converting it in place to a logging frame."""
    df = getattr(ad.AnnData, label).fget(owner)
    if owner.is_view or not isinstance(df, pd.DataFrame):
        return df
    if isinstance(df, _LoggingDataFrame) and getattr(df, "_owner", None) is owner:
        return df
    logged = _LoggingDataFrame(df)  # shares data under pandas copy-on-write
    object.__setattr__(logged, "_owner", owner)
    object.__setattr__(logged, "_label", label)
    object.__setattr__(owner, f"_{label}", logged)
    return logged


def _plain_frame(df):
    return pd.DataFrame(df) if isinstance(df, _LoggingDataFrame) else df


def _object_strings(values):
    """Turn pandas' nullable/`str` string data into object strings; leave the rest alone."""
    if isinstance(values.dtype, pd.StringDtype):
        return values.astype(object)
    if isinstance(values.dtype, pd.CategoricalDtype) and isinstance(
        values.dtype.categories.dtype, pd.StringDtype
    ):
        dtype = pd.CategoricalDtype(
            values.dtype.categories.astype(object), ordered=values.dtype.ordered
        )
        return values.astype(dtype)
    return values


def _writable(obj):
    """Copy of `obj` that every anndata version can write.

    With pandas >= 3 strings default to `StringDtype`, which anndata only writes
    when `allow_write_nullable_strings` is enabled, producing files that
    anndata < 0.11 cannot read. Object strings use the classic encoding instead.
    The caller's data is never modified.
    """
    if isinstance(obj, pd.DataFrame):
        df = pd.DataFrame(obj).copy(deep=False)
        df.index = _object_strings(df.index)
        df.columns = _object_strings(df.columns)
        for i in range(df.shape[1]):
            col = df.iloc[:, i]
            converted = _object_strings(col)
            if converted is not col:
                df.isetitem(i, converted)
        return df
    if isinstance(obj, dict):
        return {k: _writable(v) for k, v in obj.items()}
    return obj

class _LoggingProxy:
    """
    Transparent proxy for dict-like AnnData components (layers, obsm, varm, ...).
    Intercepts __setitem__ and __delitem__ to log mutations automatically.
    """

    def __init__(self, wrapped, owner: "LoggedAnnDataStandalone", label: str):
        object.__setattr__(self, "_w", wrapped)
        object.__setattr__(self, "_owner", owner)
        object.__setattr__(self, "_label", label)

    def __setitem__(self, key: str, value) -> None:
        w = object.__getattribute__(self, "_w")
        owner = object.__getattribute__(self, "_owner")
        label = object.__getattribute__(self, "_label")
        verb = "updated" if key in w else "added"
        w[key] = value
        _append_log_messages(owner, [_format_log_message(label, f"'{key}' {verb}")])

    def __delitem__(self, key: str) -> None:
        w = object.__getattribute__(self, "_w")
        owner = object.__getattribute__(self, "_owner")
        label = object.__getattribute__(self, "_label")
        del w[key]
        _append_log_messages(owner, [_format_log_message(label, f"'{key}' removed")])

    def __getitem__(self, key):
        return object.__getattribute__(self, "_w")[key]

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_w"), name)

    def __contains__(self, key):
        return key in object.__getattribute__(self, "_w")

    def __iter__(self):
        return iter(object.__getattribute__(self, "_w"))

    def __len__(self):
        return len(object.__getattribute__(self, "_w"))

    def __repr__(self):
        return repr(object.__getattribute__(self, "_w"))


@dataclass
class AnnDataSnapshot:
    """Captures the full state of an AnnData object for diffing."""
    n_obs: int
    n_vars: int
    obs_cols: list[str] = field(default_factory=list)
    var_cols: list[str] = field(default_factory=list)
    layers: list[str] = field(default_factory=list)
    obsm: list[str] = field(default_factory=list)
    varm: list[str] = field(default_factory=list)
    obsp: list[str] = field(default_factory=list)
    varp: list[str] = field(default_factory=list)

    @classmethod
    def from_anndata(cls, adata: ad.AnnData) -> "AnnDataSnapshot":
        return cls(
            n_obs=adata.n_obs,
            n_vars=adata.n_vars,
            obs_cols=list(adata.obs.columns),
            var_cols=list(adata.var.columns),
            layers=[k for k in adata.layers.keys() if k is not None],
            obsm=list(adata.obsm.keys()),
            varm=list(adata.varm.keys()),
            obsp=list(adata.obsp.keys()),
            varp=list(adata.varp.keys()),
        )


def _diff_keys(
    pre: list[str],
    post: list[str],
    label: str,
    operation: str,
    ts: str,
) -> list[list[str]]:
    msgs = []
    for k in sorted(set(pre) - set(post)):
        msgs.append(_format_log_message(operation, f"{label} removed: '{k}'", ts))
    for k in sorted(set(post) - set(pre)):
        msgs.append(_format_log_message(operation, f"{label} added: '{k}'", ts))
    return msgs


def _subset_messages(
    pre: AnnDataSnapshot,
    post: AnnDataSnapshot,
    operation: str = "subset",
) -> list[list[str]]:
    msgs: list[list[str]] = []
    ts = _timestamp()

    if pre.n_vars != post.n_vars:
        removed = pre.n_vars - post.n_vars
        pct = round((removed / pre.n_vars) * 100) if pre.n_vars else 0
        msgs.append(_format_log_message(
            operation,
            f"removed {removed} gene(s) ({pct}%), {post.n_vars} gene(s) remaining",
            ts,
        ))

    if pre.n_obs != post.n_obs:
        removed = pre.n_obs - post.n_obs
        pct = round((removed / pre.n_obs) * 100) if pre.n_obs else 0
        msgs.append(_format_log_message(
            operation,
            f"removed {removed} cell(s) ({pct}%), {post.n_obs} cell(s) remaining",
            ts,
        ))

    msgs += _diff_keys(pre.obs_cols, post.obs_cols, "obs column", operation, ts)
    msgs += _diff_keys(pre.var_cols, post.var_cols, "var column", operation, ts)
    msgs += _diff_keys(pre.layers,   post.layers,   "layer",      operation, ts)
    msgs += _diff_keys(pre.obsm,     post.obsm,     "obsm",       operation, ts)
    msgs += _diff_keys(pre.varm,     post.varm,     "varm",       operation, ts)
    msgs += _diff_keys(pre.obsp,     post.obsp,     "obsp",       operation, ts)
    msgs += _diff_keys(pre.varp,     post.varp,     "varp",       operation, ts)

    return msgs


class LoggedAnnDataStandalone(ad.AnnData):
    """Standalone subclass strategy with local logging helpers and message style."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        _ensure_log_container(self)

    @classmethod
    def _safe_component_copy(cls, value):
        return value.copy() if hasattr(value, "copy") else deepcopy(value)

    @classmethod
    def from_anndata(cls, adata: ad.AnnData) -> "LoggedAnnDataStandalone":
        if isinstance(adata, cls):
            _ensure_log_container(adata)
            return adata

        kwargs: dict[str, Any] = {
            "X": cls._safe_component_copy(adata.X) if adata.X is not None else None,
            "obs": adata.obs.copy(),
            "var": adata.var.copy(),
            "uns": _safe_deepcopy_dict(dict(adata.uns)),
            "obsm": {k: cls._safe_component_copy(v) for k, v in adata.obsm.items()},
            "varm": {k: cls._safe_component_copy(v) for k, v in adata.varm.items()},
            "layers": {
                k: cls._safe_component_copy(v)
                for k, v in adata.layers.items()
                if k is not None
            },
            "obsp": {k: cls._safe_component_copy(v) for k, v in adata.obsp.items()},
            "varp": {k: cls._safe_component_copy(v) for k, v in adata.varp.items()},
        }
        
        if adata.raw is not None:
            kwargs["raw"] = {
                "X": cls._safe_component_copy(adata.raw.X),
                "var": adata.raw.var.copy(),
                "varm": {k: cls._safe_component_copy(v) for k, v in adata.raw.varm.items()},
            }

        logged = cls(**kwargs)
        _ensure_log_container(logged)
        return logged

    # --- proxied properties: each needs a getter AND a setter ---

    @property
    def layers(self):
        return _LoggingProxy(super().layers, self, "layers")

    @layers.setter
    def layers(self, value):
        _parent_set(self, "layers", value)

    @property
    def obsm(self):
        return _LoggingProxy(super().obsm, self, "obsm")

    @obsm.setter
    def obsm(self, value):
        _parent_set(self, "obsm", value)

    @property
    def varm(self):
        return _LoggingProxy(super().varm, self, "varm")

    @varm.setter
    def varm(self, value):
        _parent_set(self, "varm", value)

    @property
    def obsp(self):
        return _LoggingProxy(super().obsp, self, "obsp")

    @obsp.setter
    def obsp(self, value):
        _parent_set(self, "obsp", value)

    @property
    def varp(self):
        return _LoggingProxy(super().varp, self, "varp")

    @varp.setter
    def varp(self, value):
        _parent_set(self, "varp", value)

    @property
    def obs(self):
        return _attached_frame(self, "obs")

    @obs.setter
    def obs(self, value):
        ad.AnnData.obs.fset(self, _plain_frame(value))

    @property
    def var(self):
        return _attached_frame(self, "var")

    @var.setter
    def var(self, value):
        ad.AnnData.var.fset(self, _plain_frame(value))

    def _unwrap(self) -> ad.AnnData:
        """Plain AnnData view of the real underlying data, without the logging proxies.

        anndata's writers use `type(elem)` to look up a serializer, and don't
        recognize `_LoggingProxy`, so the proxies must be unwrapped before writing.
        String data is also converted to a portable form (see `_writable`).
        """
        raw = None
        if self.raw is not None:
            raw = {
                "X": self.raw.X,
                "var": _writable(self.raw.var),
                "varm": _writable(dict(self.raw.varm)),
            }
        return ad.AnnData(
            X=super().X,
            obs=_writable(super().obs),
            var=_writable(super().var),
            uns=_writable(dict(self.uns)),
            obsm=_writable(dict(super().obsm)),
            varm=_writable(dict(super().varm)),
            layers=dict(super().layers),
            obsp=dict(super().obsp),
            varp=dict(super().varp),
            raw=raw,
        )

    def write_h5ad(self, filename=None, **kwargs):
        return self._unwrap().write_h5ad(filename, **kwargs)

    write = write_h5ad

    def write_loom(self, filename, **kwargs):
        return self._unwrap().write_loom(filename, **kwargs)

    def write_zarr(self, store, **kwargs):
        return self._unwrap().write_zarr(store, **kwargs)

    def write_csvs(self, dirname, **kwargs):
        return self._unwrap().write_csvs(dirname, **kwargs)

    # --- snapshot & subsetting ---

    def _snapshot(self) -> AnnDataSnapshot:
        return AnnDataSnapshot.from_anndata(self)

    def __getitem__(self, index):
        if _ACCESSOR_TYPES and isinstance(index, _ACCESSOR_TYPES):
            return super().__getitem__(index)   # returns a column array, not an AnnData
        pre = self._snapshot()
        result = super().__getitem__(index)
        if not isinstance(result, ad.AnnData):
            return result
        logged_result = self.from_anndata(result)
        msgs = _subset_messages(pre, logged_result._snapshot(), operation="subset")
        _inherit_and_append(self, logged_result, msgs)
        return logged_result

    def copy(self, filename=None):
        # anndata builds copies with a hard-coded `AnnData(...)`; re-wrap so the
        # copy keeps logging. AnnData(other) reuses the already-copied arrays.
        result = super().copy(filename=filename)
        if not isinstance(result, LoggedAnnDataStandalone):
            result = type(self)(result)
        return result

    def _operation_log_block(self) -> str:
        logs = self.uns.get(LOG_KEY, [])
        if isinstance(logs, pd.DataFrame):
            if logs.empty:
                return ""
            rows = logs.apply(lambda r: f"[{r['Time']}] {r['Operation']}: {r['Message']}", axis=1)
            return "\n\nOperation log:\n" + "\n".join(rows)
        if not logs:
            return ""
        return "\n\nOperation log:\n" + "\n".join(str(x) for x in logs)

    def __repr__(self) -> str:
        return super().__repr__() + self._operation_log_block()

    def __str__(self) -> str:
        return self.__repr__()

    def operation_log(self) -> pd.DataFrame:
        """Return a copy of the operation log as a DataFrame (Time, Operation, Message)."""
        return _ensure_log_container(self).copy()



def log_start(adata: ad.AnnData) -> LoggedAnnDataStandalone:
    """Start logging the operations performed on an AnnData object.

    Returns a logged copy of ``adata`` that behaves like a regular AnnData object.
    From then on, structural changes are recorded automatically in
    ``.uns["_omicslog"]``: adding, updating, or removing entries in ``.obs``,
    ``.var``, ``.obsm``, ``.varm``, ``.obsp``, ``.varp``, and ``.layers``, and
    subsetting cells or genes.

    Parameters
    ----------
    adata
        The AnnData object to log. It is copied and left unmodified. If it already
        holds an operation log (for example, a file saved by omicslog and read with
        :func:`anndata.read_h5ad`), new entries are added to that log.

    Returns
    -------
    LoggedAnnDataStandalone
        A logged copy of ``adata``. If ``adata`` is already a logged object, it is
        returned as is, without copying.

    Examples
    --------
    >>> import anndata as ad
    >>> import numpy as np
    >>> from omicslog import log_start
    >>> logdata = log_start(ad.AnnData(np.ones((3, 2))))
    >>> logdata.obs["group"] = ["a", "b", "a"]
    >>> logdata.uns["_omicslog"]["Message"].tolist()
    ["'group' added"]
    >>> subset = logdata[logdata.obs["group"] == "a"]
    >>> subset.uns["_omicslog"]["Message"].iloc[-1]
    'removed 1 cell(s) (33%), 2 cell(s) remaining'
    """
    return LoggedAnnDataStandalone.from_anndata(adata)

def read_h5ad(filename, **kwargs) -> LoggedAnnDataStandalone:
    """Read an `.h5ad` file and keep logging, continuing any saved operation log."""
    return LoggedAnnDataStandalone.from_anndata(ad.read_h5ad(filename, **kwargs))

def read_zarr(store, **kwargs) -> LoggedAnnDataStandalone:
    """Read a zarr store and keep logging, continuing any saved operation log."""
    return LoggedAnnDataStandalone.from_anndata(ad.read_zarr(store, **kwargs))
