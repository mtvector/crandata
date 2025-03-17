import os
import json
import tempfile
import numpy as np
import xarray as xr
import h5py
import copy

try:
    import sparse  # for sparse multidimensional arrays
except ImportError:
    print("no sparse")
    sparse = None

# -------------------------
# Sparse helpers
# -------------------------
def ensure_row_major(sparse_array: sparse.COO) -> sparse.COO:
    """
    Ensure the sparse.COO array is in row-major order along its first axis.
    """
    if not np.all(np.diff(sparse_array.coords[0]) >= 0):
        order = np.argsort(sparse_array.coords[0])
        new_coords = sparse_array.coords[:, order]
        new_data = sparse_array.data[order]
        return sparse.COO(new_coords, new_data, shape=sparse_array.shape)
    return sparse_array

def write_sparse_array(f, group_name: str, sparse_array):
    """
    Write a sparse.COO array to an HDF5 group after ensuring row-major order.
    """
    sparse_array = ensure_row_major(sparse_array)
    grp = f.create_group(group_name)
    grp.create_dataset("data", data=sparse_array.data, compression="gzip")
    grp.create_dataset("coords", data=sparse_array.coords, compression="gzip")
    grp.create_dataset("shape", data=np.array(sparse_array.shape), compression="gzip")
    grp.attrs["sparse"] = True
    grp.attrs["dtype"] = str(sparse_array.dtype)

def read_sparse_array(f, group_name: str):
    grp = f[group_name]
    data = grp["data"][()]
    coords = grp["coords"][()]
    shape = tuple(grp["shape"][()])
    return sparse.COO(coords, data, shape=shape)

def _to_dataarray(X, dim_names):
    if isinstance(X, xr.DataArray):
        return X
    if sparse is not None and isinstance(X, sparse.COO):
        return xr.DataArray(X, dims=dim_names)
    X_arr = np.asarray(X)
    return xr.DataArray(X_arr, dims=dim_names)

# -------------------------
# Generic save/load helpers
# -------------------------
def _decode_if_needed(arr):
    if isinstance(arr, np.ndarray) and arr.dtype.kind in ('S', 'O'):
        # Decode each element if it is a bytes object.
        return np.array([x.decode('utf-8') if isinstance(x, bytes) else x for x in arr])
    return arr

def _save_xarray(f, name: str, xarr: xr.DataArray, chunk_sizes: dict[str, int] | None = None):
    if chunk_sizes is not None:
        cs = tuple(
            min(chunk_sizes.get(dim, xarr.sizes[dim]), xarr.sizes[dim])
            for dim in xarr.dims
        )
    else:
        cs = None
    try:
        underlying = xarr.variable.data
    except Exception:
        underlying = None
    if sparse is not None and underlying is not None and isinstance(underlying, sparse.COO):
        underlying = ensure_row_major(underlying)
        write_sparse_array(f, name, underlying)
    else:
        # If the data is a Unicode string array, convert it to a list so h5py can handle it.
        if xarr.values.dtype.kind == 'U':
            data_to_save = xarr.values.tolist()
            str_dtype = h5py.string_dtype(encoding='utf-8')
            f.create_dataset(name, data=data_to_save, dtype=str_dtype, chunks=cs, compression="gzip")
        else:
            f.create_dataset(name, data=xarr.values, chunks=cs, compression="gzip")
    ds = f[name]
    ds.attrs["dims"] = ",".join(xarr.dims)

def _load_xarray(f, name: str, backed: bool) -> xr.DataArray:
    dset = f[name]
    dims = dset.attrs["dims"]
    if isinstance(dims, bytes):
        dims = dims.decode("utf-8")
    dims = dims.split(",")
    if backed:
        lazy_obj = LazyH5Array(f.filename, name, shape=dset.shape, dtype=dset.dtype,
                                chunks=getattr(dset, "chunks", None))
        xarr = xr.DataArray(lazy_obj, dims=dims)
        xarr.attrs["_lazy_obj"] = lazy_obj
        return xarr
    else:
        data = dset[()]
        return xr.DataArray(data, dims=dims)

def _save_generic_attr(f, name: str, attr):
    if isinstance(attr, dict):
        grp = f.create_group(name)
        for key, subattr in attr.items():
            _save_generic_attr(grp, key, subattr)
    elif isinstance(attr, xr.DataArray):
        _save_xarray(f, name, attr)
    else:
        arr = np.asarray(attr)
        if arr.dtype.kind == 'U':  # Unicode strings
            str_dtype = h5py.string_dtype(encoding='utf-8')
            if arr.ndim == 0:
                f.create_dataset(name, data=str(arr), dtype=str_dtype)
            else:
                f.create_dataset(name, data=arr.tolist(), dtype=str_dtype, compression="gzip")
        else:
            if arr.ndim == 0:
                f.create_dataset(name, data=arr)
            else:
                f.create_dataset(name, data=arr, compression="gzip")

def _load_generic_attr(f, name: str, backed: bool) -> any:
    obj = f[name]
    if isinstance(obj, h5py.Dataset):
        if "dims" in obj.attrs:
            data = _load_xarray(f, name, backed)
        else:
            data = obj[()]
        return _decode_if_needed(data)
    elif isinstance(obj, h5py.Group):
        if "sparse" in obj.attrs and obj.attrs["sparse"]:
            dims = obj.attrs["dims"]
            if isinstance(dims, bytes):
                dims = dims.decode("utf-8")
            dims = dims.split(",")
            coords = {}
            for dim in dims:
                attr_name = f"coord_{dim}"
                if attr_name in obj.attrs:
                    coords[dim] = _decode_if_needed(obj.attrs[attr_name])
            sparse_arr = read_sparse_array(f, f"{name}")
            return xr.DataArray(sparse_arr, dims=dims, coords=coords)
        else:
            out = {}
            for key in obj:
                out[key] = _load_generic_attr(obj, key, backed)
            return out
    else:
        raise ValueError(f"Unsupported HDF5 object type for attribute {name}")

# -------------------------
# Lazy loading wrappers
# -------------------------
class LazyH5Array:
    """
    A lazy loader for an HDF5 dataset.
    Each indexing call opens the file, retrieves the requested slice, then closes the file.
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

class _XWrapper:
    """
    A simple wrapper for an xarray DataArray to allow lazy evaluation.
    """
    def __init__(self, xarray_obj):
        self._x = xarray_obj

    @property
    def data(self):
        if hasattr(self._x, "attrs") and "_lazy_obj" in self._x.attrs:
            return self._x.attrs["_lazy_obj"]
        return self._x.data

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (self._x.ndim - len(key))
        dims = self._x.dims[:len(key)]
        indexers = {dim: key[i] for i, dim in enumerate(dims)}
        return self._x.isel(**indexers)

    def __array__(self):
        return np.asarray(self.data)

    def __repr__(self):
        return repr(self._x)

    def __getattr__(self, name):
        return getattr(self._x, name)

# -------------------------
# The new fully generic CrAnData class with to_memory() and slicing prevention
# -------------------------
class CrAnData:
    """
    A fully generic container for genomic data.
    
    All properties are stored in an internal dictionary. The primary property is used for slicing.
    Arrays are expected to have named axes (or names are automatically generated) and coordinate
    labels may be assigned via the axis_indices dictionary.
    
    Slicing applies indices to the leftmost dimensions (with trailing slices added) for every xarray property.
    Hierarchical keys are allowed and new properties can be added after initialization.
    
    If arrays are stored on disk (via to_h5/from_h5), they are loaded lazily.
    The to_memory() method fully loads all lazy arrays into memory.
    Further slicing is disallowed on a backed CrAnData that has already been sliced.
    
    Default parameters mimic AnnData:
        CrAnData(X=None, obs=None, var=None, uns=None, obsm=None, varm=None, layers=None)
    Additional optional parameters:
        primary_key, global_axis_order, axis_indices, properties_config.
    """
    def __init__(self, 
                 X=None, obs=None, var=None, uns=None, obsm=None, varm=None, layers=None,
                 primary_key=None, global_axis_order=None, axis_indices=None, properties_config=None,
                 **kwargs):
        # Build the dictionary using AnnData-like default keys.
        self._data = {}
        self._data["X"] = X
        self._data["obs"] = obs
        self._data["var"] = var
        self._data["uns"] = uns
        self._data["obsm"] = obsm
        self._data["varm"] = varm
        self._data["layers"] = layers
        self._data.update(kwargs)
        
        self.global_axis_order = global_axis_order
        self.axis_indices = axis_indices if axis_indices is not None else {}
        # We no longer convert to DataFrame so set convert_to_dataframe to False.
        self._properties_config = {
            "obs": {"axis": "obs", "convert_to_dataframe": False, "default": None},
            "var": {"axis": "var", "convert_to_dataframe": False, "default": None},
            "uns": {"axis": None, "convert_to_dataframe": False, "default": {}},
            "obsm": {"axis": None, "convert_to_dataframe": False, "default": {}},
            "varm": {"axis": None, "convert_to_dataframe": False, "default": {}},
            "layers": {"axis": None, "convert_to_dataframe": False, "default": {}}
        }
        if properties_config is not None:
            self._properties_config.update(properties_config)
        
        self._primary = primary_key if primary_key is not None else "X"
        
        for key, val in self._data.items():
            if isinstance(val, (np.ndarray, list)):
                nd = np.asarray(val).ndim
                if self.global_axis_order is not None and len(self.global_axis_order) <= nd:
                    dims = self.global_axis_order + [f"dim_{i}" for i in range(len(self.global_axis_order), nd)]
                else:
                    dims = [f"dim_{i}" for i in range(nd)]
                arr = _to_dataarray(val, dims)
                new_coords = {}
                for d in dims:
                    if d not in arr.coords and d in self.axis_indices:
                        new_coords[d] = np.array(self.axis_indices[d])
                if new_coords:
                    arr = arr.assign_coords(**new_coords)
                self._data[key] = arr
        
        if self.global_axis_order is not None and isinstance(self._data[self._primary], xr.DataArray):
            current_dims = list(self._data[self._primary].dims)
            new_order = [d for d in self.global_axis_order if d in current_dims]
            remaining = [d for d in current_dims if d not in self.global_axis_order]
            self._data[self._primary] = self._data[self._primary].transpose(*new_order, *remaining)
        
        if "obs" not in self._data:
            self._data["obs"] = None
        if "var" not in self._data:
            self._data["var"] = None
        
        self._is_sliced = False
        self.filename = None
        
        for key in self._data.keys():
            self._create_dynamic_property(key, self._properties_config.get(key, {}).get("convert_to_dataframe", False))
        
        if isinstance(self._data[self._primary], xr.DataArray):
            for d in self._data[self._primary].dims:
                prop_names = f"{d}_names"
                def axis_names_getter(self, axis=d):
                    if axis in self._data[self._primary].coords:
                        return self._data[self._primary].coords[axis].values
                    elif axis in self.axis_indices:
                        return np.array(self.axis_indices[axis])
                    else:
                        size = self._data[self._primary].sizes.get(axis, None)
                        return np.arange(size) if size is not None else None
                setattr(self.__class__, prop_names, property(axis_names_getter))
                prop_size = f"n_{d}"
                def n_axis_getter(self, axis=d):
                    return self._data[self._primary].sizes.get(axis, None)
                setattr(self.__class__, prop_size, property(n_axis_getter))
        
        self.primary_key = self._primary
        
        self._propagate_missing_coordinates()

    def _create_dynamic_property(self, prop_name, convert_to_dataframe: bool):
        def getter(self):
            return self._data.get(prop_name, None)
        def setter(self, value):
            self._data[prop_name] = value
        setattr(self.__class__, prop_name, property(getter, setter))

    def _propagate_missing_coordinates(self):
        """
        For each axis in global_axis_order, if at least one property has nonempty coordinates,
        then assign those coordinates to any property that has that axis but is missing coordinates.
        """
        if self.global_axis_order is None:
            return
        default_coords = {}
        for axis in self.global_axis_order:
            for key, val in self._data.items():
                if isinstance(val, xr.DataArray) and axis in val.coords:
                    coord = np.asarray(val.coords[axis])
                    if coord.size > 0:
                        default_coords[axis] = coord
                        break
        for key, val in self._data.items():
            if isinstance(val, xr.DataArray) and any(ax in val.dims for ax in self.global_axis_order):
                new_coords = {}
                for axis in self.global_axis_order:
                    if axis in val.dims:
                        if (axis not in val.coords) or (np.asarray(val.coords[axis]).size == 0):
                            if axis in default_coords:
                                new_coords[axis] = default_coords[axis]
                if new_coords:
                    self._data[key] = val.assign_coords(**new_coords)

    def add_property(self, key, value, dims=None, axis_indices=None, config=None):
        """
        Add (or update) a property after initialization.
        """
        if isinstance(value, (np.ndarray, list)):
            nd = np.asarray(value).ndim
            if dims is None:
                dims = [f"dim_{i}" for i in range(nd)]
            arr = _to_dataarray(value, dims)
            if axis_indices is not None:
                new_coords = {}
                for d in dims:
                    if d not in arr.coords and d in axis_indices:
                        new_coords[d] = np.array(axis_indices[d])
                if new_coords:
                    arr = arr.assign_coords(**new_coords)
            self._data[key] = arr
        else:
            self._data[key] = value
        if config is not None:
            self._properties_config[key] = config
        self._create_dynamic_property(key, self._properties_config.get(key, {}).get("convert_to_dataframe", False))
        self._propagate_missing_coordinates()

    @property
    def primary(self):
        return _XWrapper(self._data[self._primary])

    @primary.setter
    def primary(self, value):
        if isinstance(value, xr.DataArray):
            self._data[self._primary] = value
        else:
            dims = self._data[self._primary].dims
            self._data[self._primary] = _to_dataarray(np.asarray(value), dims)

    @property
    def shape(self):
        return self._data[self._primary].shape

    def copy(self):
        return copy.deepcopy(self)

    def __getitem__(self, key):
        """
        Slice the primary array using the provided indices.
        Further slicing is disallowed on a backed CrAnData that has already been sliced.
        """
        if (hasattr(self._data[self._primary], "attrs") and "_lazy_obj" in self._data[self._primary].attrs
            and self._is_sliced):
            raise ValueError("Cannot further slice into a backed CrAnData object that has already been sliced.")
        if not isinstance(key, tuple):
            key = (key,)
        new_data = {}
        for prop, val in self._data.items():
            if isinstance(val, xr.DataArray):
                nd = val.ndim
                key_padded = key + (slice(None),) * (nd - len(key)) if len(key) < nd else key[:nd]
                indexers = {dim: key_padded[i] for i, dim in enumerate(val.dims[:len(key_padded)])}
                new_data[prop] = val.isel(**indexers)
            else:
                new_data[prop] = val
        new_obj = CrAnData(primary_key=self._primary, global_axis_order=self.global_axis_order,
                           axis_indices=self.axis_indices, properties_config=self._properties_config,
                           **new_data)
        new_obj._is_sliced = True
        return new_obj

    def __repr__(self):
        lines = []
        primary_arr = self._data[self._primary]
        axes_info = []
        for d in primary_arr.dims:
            size = primary_arr.sizes[d]
            backed = "backed" if (hasattr(primary_arr, "attrs") and "_lazy_obj" in primary_arr.attrs) else "in-memory"
            axes_info.append(f"{d}={size} ({backed})")
        lines.append(f"CrAnData (primary: '{self._primary}') with axes: " + ", ".join(axes_info))
        for key, val in self._data.items():
            if isinstance(val, xr.DataArray):
                dims_str = ", ".join([f"{d}={val.sizes[d]}" for d in val.dims])
                backed = "backed" if (hasattr(val, "attrs") and "_lazy_obj" in val.attrs) else "in-memory"
                lines.append(f"  {key}: [{dims_str}] ({backed})")
            else:
                lines.append(f"  {key}: {val}")
        return "\n".join(lines)

    def to_h5(self, h5_path: str, chunk_sizes: dict[str, int] | None = None):
        with h5py.File(h5_path, "w") as f:
            for key, value in self._data.items():
                if value is None:
                    continue
                if isinstance(value, xr.DataArray):
                    _save_xarray(f, key, value, chunk_sizes=chunk_sizes)
                else:
                    _save_generic_attr(f, key, value)
            f.attrs["global_axis_order"] = (
                ",".join(self.global_axis_order) if self.global_axis_order else ""
            )
            f.attrs["primary_key"] = self._primary
            f.attrs["filename"] = self.filename if self.filename is not None else ""
            f.attrs["properties_config"] = json.dumps(self._properties_config)

    @classmethod
    def from_h5(cls, h5_path: str, backed: list[str] = None):
        data = {}
        with h5py.File(h5_path, "r") as f:
            for key in f:
                data[key] = _load_generic_attr(f, key, backed=(key in backed if backed else False))
            global_axis_order_str = f.attrs.get("global_axis_order", "")
            global_axis_order = global_axis_order_str.split(",") if global_axis_order_str else None
            primary_key = f.attrs.get("primary_key", "")
            properties_config = json.loads(f.attrs.get("properties_config", "{}"))
            filename = f.attrs.get("filename", "")
        return cls(primary_key=primary_key, global_axis_order=global_axis_order,
                   axis_indices={}, properties_config=properties_config, **data)

    def to_memory(self):
        """
        Fully load all lazy (backed) xarray properties into memory.
        """
        for key, val in self._data.items():
            if isinstance(val, xr.DataArray) and hasattr(val, "attrs") and "_lazy_obj" in val.attrs:
                self._data[key] = xr.DataArray(val.values, dims=val.dims, coords=val.coords)
