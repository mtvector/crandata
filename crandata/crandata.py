import h5py
import numpy as np
import xarray as xr
import pandas as pd
# from anndata import AnnData

try:
    import sparse  # for sparse multidimensional arrays
except ImportError:
    print("no sparse")
    sparse = None

# -------------------------
# Sparse helpers
# -------------------------
def write_sparse_array(f, group_name: str, sparse_array):
    """
    Write a sparse.COO array to an HDF5 group.
    Stores three datasets: "data", "coords", and "shape",
    and sets attributes "sparse" and "dtype".
    """
    grp = f.create_group(group_name)
    grp.create_dataset("data", data=sparse_array.data, compression="gzip")
    grp.create_dataset("coords", data=sparse_array.coords, compression="gzip")
    grp.create_dataset("shape", data=np.array(sparse_array.shape), compression="gzip")
    grp.attrs["sparse"] = True
    grp.attrs["dtype"] = str(sparse_array.dtype)

def read_sparse_array(f, group_name: str):
    """
    Read a sparse.COO array from an HDF5 group.
    """
    grp = f[group_name]
    data = grp["data"][()]
    coords = grp["coords"][()]
    shape = tuple(grp["shape"][()])
    return sparse.COO(coords, data, shape=shape)

def _to_dataarray(X, dim_names):
    if not isinstance(X, xr.DataArray):
        X = np.asarray(X)
        X = xr.DataArray(X, dims=dim_names)
    return X

# -------------------------
# Lazy loading helper for backed arrays
# -------------------------
class LazyH5Array:
    """
    A simple lazy loader that wraps an HDF5 dataset.
    When indexed, it reopens the file, retrieves the requested slice, then closes the file.
    """
    def __init__(self, filename: str, dataset_name: str, shape, dtype, chunks=None):
        self.filename = filename
        self.dataset_name = dataset_name
        self.shape = shape
        self.dtype = dtype
        self.chunks = chunks

    def __getitem__(self, key):
        with h5py.File(self.filename, "r") as f:
            data = f[self.dataset_name][key]
        return data

    def __array__(self):
        return np.array(self[:])

# -------------------------
# Helpers for saving/loading xarray DataArrays
# -------------------------
def _save_xarray(f, name: str, xarr: xr.DataArray):
    """
    Save an xarray DataArray to an HDF5 dataset.
    Uses sparse routines if the underlying data is sparse.
    """
    try:
        underlying = xarr.variable.data
    except Exception:
        underlying = None
    if sparse is not None and underlying is not None and isinstance(underlying, sparse.COO):
        write_sparse_array(f, name, underlying)
    else:
        f.create_dataset(name, data=xarr.values, compression="gzip")
    ds = f[name]
    ds.attrs["dims"] = ",".join(xarr.dims)
    for dim in xarr.dims:
        if dim in xarr.coords:
            ds.attrs[f"coord_{dim}"] = np.array(xarr.coords[dim].values)

def _load_xarray(f, name: str, backed: bool) -> xr.DataArray:
    dset = f[name]
    dims = dset.attrs["dims"]
    if isinstance(dims, bytes):
        dims = dims.decode("utf-8")
    dims = dims.split(",")
    coords = {}
    for dim in dims:
        attr_name = f"coord_{dim}"
        if attr_name in dset.attrs:
            coords[dim] = dset.attrs[attr_name]
    if backed:
        lazy_obj = LazyH5Array(f.filename, name, shape=dset.shape, dtype=dset.dtype,
                                chunks=getattr(dset, "chunks", None))
        xarr = xr.DataArray(lazy_obj, dims=dims, coords=coords)
        xarr.attrs["_lazy_obj"] = lazy_obj
        return xarr
    else:
        data = dset[()]
        return xr.DataArray(data, dims=dims, coords=coords)

# -------------------------
# Helpers for saving/loading DataFrames
# -------------------------
def _save_dataframe(f, name: str, df: pd.DataFrame):
    grp = f.create_group(name)
    # Save index as string
    index_data = np.array(df.index.astype(str), dtype="S")
    grp.create_dataset("index", data=index_data, compression="gzip")
    # Store the index name (if any) as an attribute
    if df.index.name is not None:
        grp.attrs["index_name"] = df.index.name.encode("utf-8")
    else:
        grp.attrs["index_name"] = b""
    grp.attrs["columns"] = np.array(df.columns.astype(str), dtype="S")
    for col in df.columns:
        grp.create_dataset(col, data=df[col].values, compression="gzip")

def _load_dataframe(f, name: str) -> pd.DataFrame:
    grp = f[name]
    index_raw = grp["index"][()]
    # Decode index: if an element is a byte string, decode it
    index = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in index_raw]
    columns_raw = grp.attrs["columns"]
    # Decode column names
    columns = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in columns_raw]
    data = {}
    for col in columns:
        col_data = grp[col][()]
        # If the data array's dtype indicates byte strings, decode each element.
        if col_data.dtype.kind in ('S', 'O'):
            decoded = []
            for item in col_data:
                if isinstance(item, bytes):
                    decoded.append(item.decode("utf-8"))
                else:
                    decoded.append(item)
            data[col] = np.array(decoded)
        else:
            data[col] = col_data
    df = pd.DataFrame(data, index=index)
    # Restore the index name if it was stored.
    if "index_name" in grp.attrs:
        index_name = grp.attrs["index_name"]
        if index_name and isinstance(index_name, bytes):
            df.index.name = index_name.decode("utf-8")
        elif index_name:
            df.index.name = index_name
    return df

# -------------------------
# Generic attribute saving/loading helpers
# -------------------------
def _save_generic_attr(f, name: str, attr):
    """
    Save an attribute to HDF5.
    - If attr is an xarray DataArray, call _save_xarray.
    - If attr is a pandas DataFrame, call _save_dataframe.
    - If attr is a dict (exactly type dict), create a group and recursively save.
    - Otherwise, convert attr to a numpy array.
    """
    if type(attr) is dict:
        grp = f.create_group(name)
        for key, subattr in attr.items():
            _save_generic_attr(grp, key, subattr)
    elif isinstance(attr, xr.DataArray):
        _save_xarray(f, name, attr)
    elif isinstance(attr, pd.DataFrame):
        _save_dataframe(f, name, attr)
    else:
        arr = np.asarray(attr)
        if arr.ndim == 0:
            f.create_dataset(name, data=arr)
        else:
            f.create_dataset(name, data=arr, compression="gzip")

def _load_generic_attr(f, name: str, backed: bool) -> any:
    obj = f[name]
    if isinstance(obj, h5py.Dataset):
        if "dims" in obj.attrs:
            return _load_xarray(f, name, backed)
        else:
            return obj[()]
    elif isinstance(obj, h5py.Group):
        # If the group is marked as sparse, load it as an xarray DataArray.
        if "sparse" in obj.attrs and obj.attrs["sparse"]:
            dims = obj.attrs["dims"]
            if isinstance(dims, bytes):
                dims = dims.decode("utf-8")
            dims = dims.split(",")
            coords = {}
            for dim in dims:
                attr_name = f"coord_{dim}"
                if attr_name in obj.attrs:
                    coords[dim] = obj.attrs[attr_name]
            # Load the sparse array from the group
            sparse_arr = read_sparse_array(f, f"{name}")
            return xr.DataArray(sparse_arr, dims=dims, coords=coords)
        # If the group has an attribute "columns", assume it's a DataFrame.
        elif "columns" in obj.attrs:
            return _load_dataframe(f, name)
        else:
            out = {}
            for key in obj:
                out[key] = _load_generic_attr(obj, key, backed)
            return out
    else:
        raise ValueError(f"Unsupported HDF5 object type for attribute {name}")

def index_dim(xarr, dim, key):
    if isinstance(key, (int, slice, np.integer)) or \
       (isinstance(key, (list, tuple)) and all(isinstance(x, (int, np.integer)) for x in key)):
        return xarr.isel({dim: key})
    else:
        return xarr.sel({dim: key})

class _XWrapper:
    def __init__(self, xarray_obj):
        self._x = xarray_obj

    @property
    def data(self):
        # If a lazy object was stored in the xarray, return that.
        if hasattr(self._x, "attrs") and "_lazy_obj" in self._x.attrs:
            return self._x.attrs["_lazy_obj"]
        return self._x.data

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) < self._x.ndim:
            key = key + (slice(None),) * (self._x.ndim - len(key))
        result = self._x
        for dim, k in zip(self._x.dims, key):
            result = index_dim(result, dim, k)
        return result

    def __setitem__(self, key, value):
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) < self._x.ndim:
            key = key + (slice(None),) * (self._x.ndim - len(key))
        if all(isinstance(k, (int, slice, np.integer)) for k in key):
            self._x = self._x.copy(data=self._x.isel(dict(zip(self._x.dims, key))).data)
        else:
            result = self._x
            for dim, k in zip(self._x.dims, key):
                result = index_dim(result, dim, k)
            self._x = result

    def __array__(self):
        # Use our custom data property to avoid forcing evaluation.
        return np.asarray(self.data)

    def __repr__(self):
        return repr(self._x)

    def __getattr__(self, name):
        return getattr(self._x, name)

# -------------------------
# Standardized CrAnData class
# -------------------------
class CrAnData:#(AnnData):
    def __init__(self, X, obs=None, var=None, uns=None,
                 obsm=None, varm=None, layers=None,
                 varp=None, obsp=None, filename: str = None):
        # try: #Keep incase we need to pretend we're AnnData
        #     super().__init__(None)
        # except Exception:
        #     print("Bypassing AnnData.__init__")
        if isinstance(X, xr.DataArray):
            self._X = X
        else:
            arr = np.asarray(X)
            extra = arr.ndim - 2
            dims = ["obs", "var"] + [f"dim_{i}" for i in range(extra)]
            self._X = xr.DataArray(arr, dims=dims)
        n_obs, n_var = self._X.sizes["obs"], self._X.sizes["var"]

        if obs is None:
            self._obs = pd.DataFrame(index=[str(i) for i in range(n_obs)])
        else:
            self._obs = pd.DataFrame(obs)
            if len(self._obs) != n_obs:
                raise ValueError("Length of obs does not match number of observations in X.")
        if var is None:
            self._var = pd.DataFrame(index=[str(i) for i in range(n_var)])
        else:
            self._var = pd.DataFrame(var)
            if len(self._var) != n_var:
                raise ValueError("Length of var does not match number of variables in X.")

        self._X = self._X.assign_coords(obs=("obs", np.array(self._obs.index)),
                                         var=("var", np.array(self._var.index)))
        # uns, layers, obsm, varm, varp, and obsp are stored using our generic routines.
        self.uns = uns if uns is not None else {}
        self.layers = layers if layers is not None else {}
        self.obsm = obsm if obsm is not None else {}
        self.varm = varm if varm is not None else {}
        self.varp = varp if varp is not None else {}
        self.obsp = obsp if obsp is not None else None
        self.filename = filename

    @property
    def X(self):
        return _XWrapper(self._X)

    @X.setter
    def X(self, value):
        self._X = _to_dataarray(np.asarray(value), self._X.dims)
        self._X = self._X.assign_coords(obs=("obs", np.array(self._obs.index)),
                                         var=("var", np.array(self._var.index)))

    @property
    def obs(self):
        return self._obs

    @obs.setter
    def obs(self, value):
        self._obs = pd.DataFrame(value)

    @property
    def var(self):
        return self._var

    @var.setter
    def var(self, value):
        self._var = pd.DataFrame(value)

    @property
    def obsm(self):
        return self._obsm

    @obsm.setter
    def obsm(self, value):
        self._obsm = value

    @property
    def varm(self):
        return self._varm

    @varm.setter
    def varm(self, value):
        self._varm = value

    @property
    def varp(self):
        return self._varp

    @varp.setter
    def varp(self, value):
        self._varp = value

    @property
    def obsp(self):
        return self._obsp

    @obsp.setter
    def obsp(self, value):
        self._obsp = value

    @property
    def shape(self):
        return self._X.shape
        
    @property
    def obs_names(self):
        return self.obs.index

    @property
    def var_names(self):
        return self.var.index
        
    @property
    def n_obs(self):
        return len(self.obs.index)

    @property
    def n_vars(self):
        return len(self.var.index)
    
    def copy(self):
        return CrAnData(
            self._X.copy(),
            obs=self._obs.copy(),
            var=self._var.copy(),
            uns=self.uns.copy(),
            obsm={k: v.copy() for k, v in self.obsm.items()},
            varm={k: v.copy() for k, v in self.varm.items()},
            layers=self.layers.copy(),
            varp={k: v.copy() for k, v in self.varp.items()} if self.varp is not None else {},
            obsp=self.obsp.copy() if self.obsp is not None else None,
            filename=self.filename,
        )

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key, slice(None))
        if len(key) != 2:
            raise IndexError("Only two indices (obs, var) are supported for slicing.")
        obs_key, var_key = key
        new_X = self._X.isel(obs=obs_key, var=var_key)
        new_obs = self._obs.iloc[obs_key]
        new_var = self._var.iloc[var_key]
        new_uns = self.uns.copy()
        new_layers = self.layers.copy()
        new_obsm = {k: v.isel(obs=obs_key) for k, v in self.obsm.items()}
        new_varm = {k: v.isel(var=var_key) for k, v in self.varm.items()}
        new_varp = {}
        for k, v in self.varp.items():
            new_varp[k] = v.isel(var_0=var_key, var_1=var_key)
        new_obsp = self.obsp
        if new_obsp is not None:
            new_obsp = new_obsp.isel(obs_0=obs_key, obs_1=obs_key)
        return CrAnData(new_X, obs=new_obs, var=new_var, uns=new_uns,
                        layers=new_layers, obsm=new_obsm, varm=new_varm,
                        varp=new_varp, obsp=new_obsp, filename=self.filename)

    def _subset_df(self, df, key):
        if isinstance(key, (int, np.integer)):
            return df.iloc[[key]]
        elif isinstance(key, slice):
            return df.iloc[key]
        else:
            return df.loc[key]

    def __repr__(self):
        n_obs = self._X.sizes["obs"]
        n_var = self._X.sizes["var"]
        return f"CrAnData object with {n_obs} observations and {n_var} variables"

    def to_h5(self, h5_path: str):
        """
        Save all attributes of CrAnData in a standardized way.
        """
        attr_dict = {
            "X": self._X,
            "obs": self._obs,
            "var": self._var,
            "uns": self.uns,
            "layers": self.layers,
            "obsm": self.obsm,
            "varm": self.varm,
            "varp": self.varp,
            "obsp": self.obsp,
        }
        with h5py.File(h5_path, "w") as f:
            for key, value in attr_dict.items():
                if value is None:
                    continue
                _save_generic_attr(f, key, value)

    @classmethod
    def from_h5(cls, h5_path: str, backed: list[str] = ["X", "varp", "varm"]):
        attr_dict = {}
        with h5py.File(h5_path, "r") as f:
            for key in f:
                attr_dict[key] = _load_generic_attr(f, key, key in backed)
        # Ensure that layers (and similar mapping-type attributes) are dictionaries.
        if not isinstance(attr_dict.get("layers", {}), dict):
            attr_dict["layers"] = {}
        if not isinstance(attr_dict.get("obsm", {}), dict):
            attr_dict["obsm"] = {}
        if not isinstance(attr_dict.get("varm", {}), dict):
            attr_dict["varm"] = {}
        if not isinstance(attr_dict.get("varp", {}), dict):
            attr_dict["varp"] = {}
        if not isinstance(attr_dict.get("obsp", {}), dict):
            attr_dict["obsp"] = {}
    
        return cls(X=attr_dict["X"],
                   obs=attr_dict["obs"],
                   var=attr_dict["var"],
                   uns=attr_dict.get("uns", {}),
                   layers=attr_dict.get("layers", {}),
                   obsm=attr_dict.get("obsm", {}),
                   varm=attr_dict.get("varm", {}),
                   varp=attr_dict.get("varp", {}),
                   obsp=attr_dict.get("obsp", None))
