# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import copy
import importlib
import json
import numbers
import os
import io
import sys
import types
import weakref
from ast import literal_eval
from functools import partial
from inspect import isclass, _findclass
from warnings import warn
import numpy as np
import pandas.errors
from deepdiff.diff import DeepDiff
from packaging.version import Version
from pandapower import __version__
import networkx
import numpy
import pandas as pd
from networkx.readwrite import json_graph
from numpy import ndarray, generic, equal, isnan, allclose, any as anynp

try:
    import psycopg2
    import psycopg2.errors
    import psycopg2.extras
    PSYCOPG2_INSTALLED = True
except ImportError:
    psycopg2 = None
    PSYCOPG2_INSTALLED = False
try:
    from pandas.testing import assert_series_equal, assert_frame_equal
except ImportError:
    from pandas.util.testing import assert_series_equal, assert_frame_equal
try:
    from cryptography.fernet import Fernet
    cryptography_INSTALLED = True
except ImportError:
    cryptography_INSTALLED = False
try:
    import hashlib
    hashlib_INSTALLED = True
except ImportError:
    hashlib_INSTALLED = False
try:
    import base64
    base64_INSTALLED = True
except ImportError:
    base64_INSTALLED = False
try:
    import zlib
    zlib_INSTALLED = True
except ImportError:
    zlib_INSTALLED = False

from pandapower.auxiliary import pandapowerNet, get_free_id, soft_dependency_error, _preserve_dtypes
from pandapower.create import create_empty_network

from functools import singledispatch

try:
    import fiona
    import fiona.crs
    import geopandas

    GEOPANDAS_INSTALLED = True
except ImportError:
    GEOPANDAS_INSTALLED = False

try:
    import shapely.geometry

    SHAPELY_INSTALLED = True
except (ImportError, OSError):
    SHAPELY_INSTALLED = False

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def coords_to_df(value, geotype="line"):
    columns = ["x", "y", "coords"] if geotype == "bus" else ["coords"]
    geo = pd.DataFrame(columns=columns, index=value.index)
    if "coords" in value.columns and any(~value.coords.isnull()):
        k = max(len(v) for v in value.coords.values)
        v = numpy.empty((len(value), k * 2))
        v.fill(numpy.nan)
        for i, idx in enumerate(value.index):
            # get coords and convert them to x1, y1, x2, y2...
            coords = value.at[idx, 'coords']
            if coords is None:
                continue
            v[i, :len(coords) * 2] = numpy.array(coords).flatten()
        geo = pd.DataFrame(v, index=value.index)
        geo.columns = ["%s%i" % (w, i) for i in range(k) for w in "xy"]
    if geotype == "bus":
        geo["x"] = value["x"].values
        geo["y"] = value["y"].values
    return geo


def to_dict_of_dfs(net, include_results=False, include_std_types=True, include_parameters=True,
                   include_empty_tables=True):
    dodfs = dict()
    dtypes = []
    parameters = dict()  # pd.DataFrame(columns=["parameter"])
    for item, value in net.items():
        # dont save internal variables and results (if not explicitely specified)
        if item.startswith("_") or (item.startswith("res") and not include_results):
            continue
        elif item == "std_types":
            if not include_std_types:
                continue
            for t in net.std_types.keys():  # which are ["line", "trafo", "trafo3w", "fuse"]
                if net.std_types[t]:  # avoid empty excel sheets for std_types if empty
                    type_df = pd.DataFrame(net.std_types[t]).T
                    if t == "fuse":
                        for c in type_df.columns:
                            type_df[c] = type_df[c].apply(lambda x: str(x) if isinstance(x, list) else x)
                    dodfs["%s_std_types" % t] = type_df
            continue
        elif item == "profiles":
            for t in net.profiles.keys():  # which could be e.g. "sgen", "gen", "load", ...
                if net.profiles[t].shape[0]:  # avoid empty excel sheets for std_types if empty
                    dodfs["%s_profiles" % t] = pd.DataFrame(net.profiles[t])
            continue
        elif item == "user_pf_options":
            if len(value) > 0:
                dodfs["user_pf_options"] = pd.DataFrame(value, index=[0])
            continue
        elif isinstance(value, (int, float, bool, str, numbers.Number)):
            # attributes of primitive types are just stored in a DataFrame "parameters"
            parameters[item] = net[item]
            continue
        elif not isinstance(value, pd.DataFrame):
            logger.warning("Could not serialize net.%s" % item)
            continue

        # value is pandas DataFrame
        if not include_empty_tables and value.empty:
            continue

        if item == "bus_geodata":
            geo = coords_to_df(value, geotype="bus")
            if GEOPANDAS_INSTALLED and isinstance(value, geopandas.GeoDataFrame):
                geo["geometry"] = [s.to_wkt() for s in net.bus_geodata.geometry.values]
            dodfs[item] = geo
        elif item == "line_geodata":
            geo = coords_to_df(value, geotype="line")
            if GEOPANDAS_INSTALLED and isinstance(value, geopandas.GeoDataFrame):
                geo["geometry"] = [s.to_wkt() for s in net.line_geodata.geometry.values]
            dodfs[item] = geo
        elif "object" in value.columns:
            columns = [c for c in value.columns if c != "object"]
            tab = value[columns].copy()
            tab["object"] = value["object"].apply(lambda x: json.dumps(x, cls=PPJSONEncoder,
                                                                       indent=2))
            tab = tab[value.columns]
            if "recycle" in tab.columns:
                tab["recycle"] = tab["recycle"].apply(json.dumps)
            dodfs[item] = tab
        else:
            dodfs[item] = value
        # save dtypes
        for column, dtype in value.dtypes.items():
            dtypes.append((item, column, str(dtype)))
    dodfs["dtypes"] = pd.DataFrame(dtypes, columns=["element", "column", "dtype"])
    if include_parameters and len(parameters) > 0:
        dodfs["parameters"] = pd.DataFrame(parameters, index=[0])
    return dodfs


def dicts_to_pandas(json_dict):
    warn("This function is deprecated and will be removed in a future release.\r\n"
         "Please resave your grid using the current pandapower version.", DeprecationWarning)
    pd_dict = dict()
    for k in sorted(json_dict.keys()):
        if isinstance(json_dict[k], dict):
            pd_dict[k] = pd.DataFrame.from_dict(json_dict[k], orient="columns")
            if pd_dict[k].shape[0] == 0:  # skip empty dataframes
                continue
            if pd_dict[k].index[0].isdigit():
                pd_dict[k].set_index(pd_dict[k].index.astype(numpy.int64), inplace=True)
        else:
            raise UserWarning("The network is an old version or corrupt. "
                              "Try to use the old load function")
    return pd_dict


def df_to_coords(net, item, table):
    # converts dataframe to coords in net
    num_points = len(table.columns) // 2
    net[item] = pd.DataFrame(index=table.index, columns=net[item].columns)
    if item == "bus_geodata":
        num_points -= 1
        net[item].loc[:, ['x', 'y']] = table.loc[:, ['x', 'y']]

    for i in table.index:
        coords = table.loc[i]
        # for i, coords in table.iterrows():
        coord = [(coords["x%u" % nr], coords["y%u" % nr]) for nr in range(num_points)
                 if pd.notnull(coords["x%u" % nr])]
        if len(coord):
            net[item].loc[i, "coords"] = coord


def from_dict_of_dfs(dodfs, net=None):
    if net is None:
        net = create_empty_network()
    for item, table in dodfs.items():
        if item == "dtypes":
            continue
        elif item == "parameters":
            for c in dodfs["parameters"].columns:
                net[c] = dodfs["parameters"].at[0, c]
                if c == "name" and pd.isnull(net[c]):
                    net[c] = ''
            continue
        elif item in ["line_geodata", "bus_geodata"]:
            table = table.rename_axis(net[item].index.name)
            df_to_coords(net, item, table)
        elif item.endswith("_std_types"):
            # when loaded from Excel, the lists in the DataFrame cells are strings -> we want to convert them back
            # to lists here. There is probably a better way to deal with it.
            if item.startswith("fuse"):
                for c in table.columns:
                    table[c] = table[c].apply(lambda x: json.loads(x) if isinstance(x, str) and x.startswith("[") else x)
            net["std_types"][item[:-10]] = table.T.to_dict()
            continue  # don't go into try..except
        elif item.endswith("_profiles"):
            if "profiles" not in net.keys():
                net["profiles"] = dict()
            table = table.rename_axis(None)
            net["profiles"][item[:-9]] = table
            continue  # don't go into try..except
        elif item == "user_pf_options":
            net['user_pf_options'] = {c: v for c, v in zip(table.columns, table.values[0])}
            continue  # don't go into try..except
        else:
            for json_column in ("object", "recycle"):
                if json_column in table.columns:
                    table[json_column] = table[json_column].apply(
                        lambda x: json.loads(x, cls=PPJSONDecoder))
            if not isinstance(table.index, pd.MultiIndex):
                table = table.rename_axis(net[item].index.name)
            net[item] = table
        # set the index to be Int
        try:
            net[item].set_index(net[item].index.astype(np.int64), inplace=True)
        except (TypeError, ValueError):
            # TypeError or ValueError: if not int index (e.g. str)
            pass
    if "dtypes" in dodfs:
        restore_all_dtypes(net, dodfs["dtypes"])
    elif "dtypes" in net:
        restore_all_dtypes(net, net["dtypes"])
    return net


def restore_all_dtypes(net, dtypes):
    for _, v in dtypes.iterrows():
        try:
            if v["dtype"] == "object":
                c = net[v.element][v.column]
                net[v.element][v.column] = numpy.where(c.isnull(), None, c)
                # net[v.element][v.column] = net[v.element][v.column].fillna(value=None)
            net[v.element][v.column] = net[v.element][v.column].astype(v["dtype"])
        except KeyError:
            pass
        except pandas.errors.IntCastingNaNError:
            logger.info(f"could not convert dtype in {v.element}: {v.column}")


def to_dict_with_coord_transform(net, point_geo_columns, line_geo_columns):
    save_net = dict()
    for key, item in net.items():
        if hasattr(item, "columns") and "geometry" in item.columns:
            # we convert shapely-objects to primitive data-types on a deepcopy
            item = copy.deepcopy(item)
            if key in point_geo_columns and not isinstance(item.geometry.values[0], tuple):
                item["geometry"] = item.geometry.apply(lambda x: (x.x, x.y))
            elif key in line_geo_columns and not isinstance(item.geometry.values[0], list):
                item["geometry"] = item.geometry.apply(lambda x: list(x.coords))

        save_net[key] = {"DF": item.to_dict("split"),
                         "dtypes": {col: dt for col, dt in zip(item.columns, item.dtypes)}} \
            if isinstance(item, pd.DataFrame) else item
    return save_net


def get_raw_data_from_pickle(filename):
    def read(f):
        return pd.read_pickle(f)

    if hasattr(filename, 'read'):
        net = read(filename)
    elif not os.path.isfile(filename):
        raise UserWarning("File %s does not exist!!" % filename)
    else:
        with open(filename, "rb") as f:
            net = read(f)
    return net


def transform_net_with_df_and_geo(net, point_geo_columns, line_geo_columns):
    try:
        epsg = net.gis_epsg_code
    except AttributeError:
        epsg = None

    for key, item in net.items():
        if isinstance(item, dict) and "DF" in item:
            df_dict = item["DF"]
            if "columns" in df_dict:
                # make sure the index is Int
                try:
                    df_index = pd.Index(df_dict['index'], dtype=numpy.int64)
                except TypeError:
                    df_index = df_dict['index']
                if GEOPANDAS_INSTALLED and "geometry" in df_dict["columns"] \
                        and epsg is not None:
                    # convert primitive data-types to shapely-objects
                    if key in point_geo_columns:
                        data = {"x": [row[0] for row in df_dict["data"]],
                                "y": [row[1] for row in df_dict["data"]]}
                        geo = [shapely.geometry.Point(row[2][0], row[2][1])
                               for row in df_dict["data"]]
                    elif key in line_geo_columns:
                        data = {"coords": [row[0] for row in df_dict["data"]]}
                        geo = [shapely.geometry.LineString(row[1]) for row in df_dict["data"]]

                    net[key] = geopandas.GeoDataFrame(data, crs=f"epsg:{epsg}", geometry=geo,
                                                      index=df_index)
                else:
                    net[key] = pd.DataFrame(columns=df_dict["columns"], index=df_index,
                                            data=df_dict["data"])
            else:
                net[key] = pd.DataFrame.from_dict(df_dict)
                if "columns" in item:
                    net[key] = net[key].reindex(item["columns"], axis=1)

            if "dtypes" in item:
                if "columns" in df_dict and "geometry" in df_dict["columns"]:
                    pass
                else:
                    try:
                        # only works with pandas 0.19 or newer
                        net[key] = net[key].astype(item["dtypes"])
                    except:
                        # works with pandas <0.19
                        for column in net[key].columns:
                            net[key][column] = net[key][column].astype(item["dtypes"][column])


def isinstance_partial(obj, cls):
    # this function shall make sure that for the given classes, no default string functions are
    # used, but the registered ones (to_serializable registry)
    if isinstance(obj, (pandapowerNet, tuple, numpy.floating)):
        return False
    return isinstance(obj, cls)


def check_net_version(net):
    if Version(net["format_version"]) > Version(__version__):
        logger.warning("pandapowerNet-version is newer than your pandapower version. Please update"
                       " pandapower `pip install --upgrade pandapower`.")


class PPJSONEncoder(json.JSONEncoder):
    def __init__(self, isinstance_func=isinstance_partial, **kwargs):
        super(PPJSONEncoder, self).__init__(**kwargs)
        self.isinstance_func = isinstance_func

    def iterencode(self, o, _one_shot=False):
        """Encode the given object and yield each string
        representation as available.

        For example::

            for chunk in JSONEncoder().iterencode(bigobject):
                mysocket.write(chunk)

        """
        if self.check_circular:
            markers = {}
        else:
            markers = None
        if self.ensure_ascii:
            _encoder = json.encoder.encode_basestring_ascii
        else:
            _encoder = json.encoder.encode_basestring

        def floatstr(o, allow_nan=self.allow_nan, _repr=float.__repr__, _inf=json.encoder.INFINITY,
                     _neginf=-json.encoder.INFINITY):
            # Check for specials.  Note that this type of test is processor
            # and/or platform-specific, so do tests which don't depend on the
            # internals.

            if o != o:
                text = 'NaN'
            elif o == _inf:
                text = 'Infinity'
            elif o == _neginf:
                text = '-Infinity'
            else:
                return _repr(o)

            if not allow_nan:
                raise ValueError(
                    "Out of range float values are not JSON compliant: " + repr(o))

            return text

        _iterencode = json.encoder._make_iterencode(
            markers, self.default, _encoder, self.indent, floatstr,
            self.key_separator, self.item_separator, self.sort_keys,
            self.skipkeys, _one_shot, isinstance=self.isinstance_func)
        return _iterencode(o, 0)

    def default(self, o):
        try:
            s = to_serializable(o)
        except TypeError:
            # Let the base class default method raise the TypeError
            return json.JSONEncoder.default(self, o)
        else:
            return s


class FromSerializable:
    def __init__(self):
        self.class_name = 'class_name'
        self.module_name = 'module_name'
        self.registry = {}

    def __get__(self, instance, owner):
        if instance is None:
            return self
        class_module = getattr(instance, self.class_name), getattr(instance, self.module_name)
        if class_module not in self.registry:
            _class = (class_module[0], '')
            _module = ('', class_module[1])
            if (_class in self.registry) and (_module in self.registry):
                logger.error('the saved object %s is ambiguous. There are at least two possibilites'
                             ' to decode the object' % class_module)
            elif _class in self.registry:
                class_module = _class
            elif _module in self.registry:
                class_module = _module
            else:
                class_module = ('', '')
        method = self.registry[class_module]
        return method.__get__(instance, owner)

    def register(self, class_name='', module_name=''):
        def decorator(method):
            self.registry[(class_name, module_name)] = method
            return method

        return decorator


class FromSerializableRegistry():
    from_serializable = FromSerializable()
    class_name = ''
    module_name = ''

    def __init__(self, obj, d, pp_hook_funct):
        self.obj = obj
        self.d = d
        self.pp_hook = pp_hook_funct

    @from_serializable.register(class_name='Series', module_name='pandas.core.series')
    def Series(self):
        is_multiindex = self.d.pop('is_multiindex', False)
        index_name = self.d.pop('index_name', None)
        index_names = self.d.pop('index_names', None)
        ser = pd.read_json(self.obj, precise_float=True, **self.d)

        # restore index name and Multiindex
        if index_name is not None:
            ser.index.name = index_name
        if is_multiindex:
            try:
                if len(ser) == 0:
                    ser.index = pd.MultiIndex.from_tuples([], names=index_names, dtype=np.int64)
                else:
                    ser.index = pd.MultiIndex.from_tuples(pd.Series(ser.index).apply(literal_eval).tolist())
            except:
                logger.warning("Converting index to multiindex failed.")
            else:
                if index_names is not None:
                    ser.index.names = index_names

        return ser

    @from_serializable.register(class_name='DataFrame', module_name='pandas.core.frame')
    def DataFrame(self):
        is_multiindex = self.d.pop('is_multiindex', False)
        is_multicolumn = self.d.pop('is_multicolumn', False)
        index_name = self.d.pop('index_name', None)
        index_names = self.d.pop('index_names', None)
        column_name = self.d.pop('column_name', None)
        column_names = self.d.pop('column_names', None)

        obj = self.obj
        if type(obj) == str and (not os.path.isabs(obj) or not obj.endswith('.json')):
            obj = io.StringIO(obj)

        df = pd.read_json(obj, precise_float=True, convert_axes=False, **self.d)

        if not df.shape[0] or self.d.get("orient", False) == "columns":
            try:
                df.set_index(df.index.astype(numpy.int64), inplace=True)
            except (ValueError, TypeError, AttributeError):
                logger.debug("failed setting index to int")
        if self.d.get("orient", False) == "columns":
            try:
                df.columns = df.columns.astype(numpy.int64)
            except (ValueError, TypeError, AttributeError):
                logger.debug("failed setting columns to int")

        # restore index name and Multiindex
        if index_name is not None:
            df.index.name = index_name
        if column_name is not None:
            df.columns.name = column_name
        if is_multiindex:
            try:
                if len(df) == 0:
                    df.index = pd.MultiIndex.from_frame(pd.DataFrame(columns=index_names, dtype=np.int64))
                else:
                    df.index = pd.MultiIndex.from_tuples(pd.Series(df.index).apply(literal_eval).tolist())
                # slower alternative code:
                # df.index = pd.MultiIndex.from_tuples([literal_eval(idx) for idx in df.index])
            except:
                logger.warning("Converting index to multiindex failed.")
            else:
                if index_names is not None:
                    df.index.names = index_names
        if is_multicolumn:
            try:
                if len(df) == 0:
                    df.columns = pd.MultiIndex.from_frame(pd.DataFrame(columns=column_names, dtype=np.int64))
                else:
                    df.columns = pd.MultiIndex.from_tuples(pd.Series(df.columns).apply(literal_eval).tolist())
            except:
                logger.warning("Converting columns to multiindex failed.")
            else:
                if column_names is not None:
                    df.columns.names = column_names

        # recreate jsoned objects
        for col in ('object', 'controller'):  # "controller" for backwards compatibility
            if (col in df.columns):
                df[col] = df[col].apply(self.pp_hook)
        return df

    @from_serializable.register(class_name='pandapowerNet', module_name='pandapower.auxiliary')#,
                                # empty_dict_like_object=None)
    def pandapowerNet(self):
        if isinstance(self.obj, str):  # backwards compatibility
            from pandapower import from_json_string
            return from_json_string(self.obj)
        else:
            if self.empty_dict_like_object is None:
                net = create_empty_network()
            else:
                net = self.empty_dict_like_object
            net.update(self.obj)
            return net

    @from_serializable.register(class_name="MultiGraph", module_name="networkx")
    def networkx(self):
        mg = json_graph.adjacency_graph(self.obj, attrs={'id': 'json_id', 'key': 'json_key'})
        edges = list()
        for (n1, n2, e) in mg.edges:
            attr = {k: v for k, v in mg.get_edge_data(n1, n2, key=e).items() if
                    k not in ("json_id", "json_key")}
            attr["key"] = e
            edges.append((n1, n2, attr))
        mg.clear_edges()
        for n1, n2, ed in edges:
            mg.add_edge(n1, n2, **ed)
        return mg

    @from_serializable.register(class_name="method")
    def method(self):
        logger.warning('deserializing of method not implemented')
        # class_ = getattr(module, obj) # doesn't work
        return self.obj

    @from_serializable.register(class_name='function')
    def function(self):
        module = importlib.import_module(self.module_name)
        if not hasattr(module, self.obj):  # in case a function is a lambda or is not defined
            raise UserWarning('Could not find the definition of the function %s in the module %s' %
                              (self.obj, module.__name__))
        class_ = getattr(module, self.obj)  # works
        return class_

    @from_serializable.register()
    def rest(self):
        module = importlib.import_module(self.module_name)
        class_ = getattr(module, self.class_name)
        if isclass(class_) and issubclass(class_, JSONSerializableClass):
            if isinstance(self.obj, str):
                self.obj = json.loads(self.obj, cls=PPJSONDecoder,
                                      object_hook=pp_hook)
                # backwards compatibility
            if "net" in self.obj:
                del self.obj["net"]
            return class_.from_dict(self.obj)
        else:
            # for non-pp objects, e.g. tuple
            try:
                return class_(self.obj, **self.d)
            except ValueError:
                data = json.loads(self.obj)
                df = pd.DataFrame(columns=self.d["columns"])
                for d in data["features"]:
                    idx = int(d["id"])
                    for prop, val in d["properties"].items():
                        df.at[idx, prop] = val
                    # for geom, val in d["geometry"].items():
                    #     df.at[idx, geom] = val
                return df

    if GEOPANDAS_INSTALLED:
        @from_serializable.register(class_name='GeoDataFrame', module_name='geopandas.geodataframe')
        def GeoDataFrame(self):
            df = geopandas.GeoDataFrame.from_features(fiona.Collection(self.obj), crs=self.d['crs'])
            if "id" in df:
                df.set_index(df['id'].values.astype(numpy.int64), inplace=True)
            else:
                df.set_index(df.index.values.astype(numpy.int64), inplace=True)
            # coords column is not handled properly when using from_features
            if 'coords' in df:
                # df['coords'] = df.coords.apply(json.loads)
                valid_coords = ~pd.isnull(df.coords)
                df.loc[valid_coords, 'coords'] = df.loc[valid_coords, "coords"].apply(json.loads)
            df = df.reindex(columns=self.d['columns'])

            # df.astype changes geodataframe to dataframe -> _preserve_dtypes fixes it
            _preserve_dtypes(df, dtypes=self.d["dtype"])
            return df

    if SHAPELY_INSTALLED:
        @from_serializable.register(module_name='shapely')
        def shapely(self):
            return shapely.geometry.shape(self.obj)


class PPJSONDecoder(json.JSONDecoder):
    def __init__(self, **kwargs):
        # net = pandapowerNet.__new__(pandapowerNet)
        #        net = create_empty_network()
        deserialize_pandas = kwargs.pop('deserialize_pandas', True)
        empty_dict_like_object = kwargs.pop('empty_dict_like_object', None)
        registry_class = kwargs.pop("registry_class", FromSerializableRegistry)
        super_kwargs = {"object_hook": partial(pp_hook,
                                               deserialize_pandas=deserialize_pandas,
                                               empty_dict_like_object=empty_dict_like_object,
                                               registry_class=registry_class)}
        super_kwargs.update(kwargs)
        super().__init__(**super_kwargs)


def pp_hook(d, deserialize_pandas=True, empty_dict_like_object=None,
            registry_class=FromSerializableRegistry):
    try:
        if '_module' in d and '_class' in d:
            if 'pandas' in d['_module'] and not deserialize_pandas:
                return json.dumps(d)
            elif "_object" in d:
                obj = d.pop('_object')
            elif "_state" in d:
                obj = d['_state']
                if '_init' in obj:
                    del obj['_init']
                return obj  # backwards compatibility
            else:
                # obj = {"_init": d, "_state": dict()}  # backwards compatibility
                obj = {key: val for key, val in d.items() if key not in ['_module', '_class']}
            fs = registry_class(obj, d, pp_hook)
            fs.class_name = d.pop('_class', '')
            fs.module_name = d.pop('_module', '')
            fs.empty_dict_like_object = empty_dict_like_object
            return fs.from_serializable()
        else:
            return d
    except TypeError:
        logger.debug('Loading your grid raised a TypeError. %s raised this exception' % d)
        return d


def encrypt_string(s, key, compress=True):
    missing_packages = numpy.array(["cryptography", "hashlib", "base64"])[~numpy.array([
        cryptography_INSTALLED, hashlib_INSTALLED, base64_INSTALLED])]
    if len(missing_packages):
        soft_dependency_error(str(sys._getframe().f_code.co_name) + "()", missing_packages)
    key_base = hashlib.sha256(key.encode())
    key = base64.urlsafe_b64encode(key_base.digest())
    cipher_suite = Fernet(key)

    s = s.encode()
    if compress:
        import zlib
        s = zlib.compress(s)
    s = cipher_suite.encrypt(s)
    s = s.decode()
    return s


def decrypt_string(s, key):
    missing_packages = numpy.array(["cryptography", "hashlib", "base64"])[~numpy.array([
        cryptography_INSTALLED, hashlib_INSTALLED, base64_INSTALLED])]
    if len(missing_packages):
        soft_dependency_error(str(sys._getframe().f_code.co_name) + "()", missing_packages)
    key_base = hashlib.sha256(key.encode())
    key = base64.urlsafe_b64encode(key_base.digest())
    cipher_suite = Fernet(key)

    s = s.encode()
    s = cipher_suite.decrypt(s)
    if zlib_INSTALLED:
        s = zlib.decompress(s)
    s = s.decode()
    return s


class JSONSerializableClass(object):
    json_excludes = ["self", "__class__"]

    def __init__(self, **kwargs):
        pass

    def to_json(self):
        """
        Each controller should have this method implemented. The resulting json string should be
        readable by the controller's from_json function and by the function add_ctrl_from_json in
        control_handler.
        """
        return json.dumps(self.to_dict(), cls=PPJSONEncoder)

    def to_dict(self):
        def consider_callable(value):
            if callable(value) and value.__class__ in (types.MethodType, types.FunctionType):
                if value.__class__ == types.MethodType and _findclass(value) is not None:
                    return with_signature(value, value.__name__, obj_module=_findclass(value))
                return with_signature(value, value.__name__)
            return value

        d = {key: consider_callable(val) for key, val in self.__dict__.items()
             if key not in self.json_excludes}
        return d

    def add_to_net(self, net, element, index=None, column="object", overwrite=False,
                   preserve_dtypes=False, fill_dict=None):
        if element not in net:
            net[element] = pd.DataFrame(columns=[column])
        if index is None:
            index = get_free_id(net[element])
        if index in net[element].index.values:
            obj = net[element].object.at[index]
            if overwrite or not isinstance(obj, JSONSerializableClass):
                logger.info("Updating %s with index %s" % (element, index))
            else:
                raise UserWarning("%s with index %s already exists" % (element, index))

        dtypes = None
        if preserve_dtypes:
            dtypes = net[element].dtypes

        if fill_dict is not None:
            for k, v in fill_dict.items():
                net[element].at[index, k] = v
        net[element].at[index, column] = self

        if preserve_dtypes:
            _preserve_dtypes(net[element], dtypes)

        return index

    def equals(self, other):
        # todo: can this method be removed?
        warn("JSONSerializableClass: the attribute 'equals' is deprecated "
             "and will be removed in the future. Use the '__eq__' method instead, "
             "by directly comparing the objects 'a == b'. "
             "To check if two variables point to the same object, use 'a is b'", DeprecationWarning)

        logger.warning("JSONSerializableClass: the attribute 'equals' is deprecated "
                       "and will be removed in the future. Use the '__eq__' method instead, "
                       "by directly comparing the objects 'a == b'. "
                       "To check if two variables point to the same object, use 'a is b'")

        class UnequalityFound(Exception):
            pass

        def check_equality(obj1, obj2):
            if isinstance(obj1, (ndarray, generic)) or isinstance(obj2, (ndarray, generic)):
                unequal = True
                if equal(obj1, obj2):
                    unequal = False
                elif anynp(isnan(obj1)):
                    if allclose(obj1, obj2, atol=0, rtol=0, equal_nan=True):
                        unequal = False
                if unequal:
                    raise UnequalityFound
            elif not isinstance(obj2, type(obj1)):
                raise UnequalityFound
            elif isinstance(obj1, pandapowerNet):
                pass
            elif isinstance(obj1, pd.DataFrame):
                if len(obj1) > 0:
                    try:
                        assert_frame_equal(obj1, obj2)
                    except:
                        raise UnequalityFound
            elif isinstance(obj2, pd.Series):
                if len(obj1) > 0:
                    try:
                        assert_series_equal(obj1, obj2)
                    except:
                        raise UnequalityFound
            elif isinstance(obj1, dict):
                check_dictionary_equality(obj1, obj2)
            elif obj1 != obj1 and obj2 != obj2:
                pass
            elif callable(obj1):
                check_callable_equality(obj1, obj2)
            elif obj1 != obj2:
                try:
                    if not (isnan(obj1) and isnan(obj2)):
                        raise UnequalityFound
                except:
                    raise UnequalityFound

        def check_dictionary_equality(obj1, obj2):
            if set(obj1.keys()) != set(obj2.keys()):
                raise UnequalityFound
            for key in obj1.keys():
                if key != "_init":
                    check_equality(obj1[key], obj2[key])

        def check_callable_equality(obj1, obj2):
            if isinstance(obj1, weakref.ref) and isinstance(obj2, weakref.ref):
                return
            if str(obj1) != str(obj2):
                raise UnequalityFound

        if isinstance(other, self.__class__):
            try:
                check_equality(self.__dict__, other.__dict__)
                return True
            except UnequalityFound:
                return False
        else:
            return False

    @classmethod
    def from_dict(cls, d):
        obj = JSONSerializableClass.__new__(cls)
        obj.__dict__.update(d)
        return obj

    @classmethod
    def from_json(cls, json_string):
        d = json.loads(json_string, cls=PPJSONDecoder)
        return cls.from_dict(d)

    def __eq__(self, other):
        """
        comparing class name and attributes instead of class object address directly.
        This allows more flexibility,
        e.g. when the class definition is moved to a different module.
        Checking isinstance(other, self.__class__) for an early return without calling DeepDiff.
        There is still a risk that the implementation details of the methods can differ
        if the classes are from different modules.
        The comparison is based on comparing dictionaries of the classes.
        To this end, the dictionary comparison library deepdiff is used for recursive comparison.
        """
        if self.__class__.__name__ != other.__class__.__name__:
            return False
        else:
            d = DeepDiff(self.__dict__, other.__dict__, ignore_nan_inequality=True,
                         significant_digits=6, math_epsilon=1e-6, ignore_private_variables=False)
            return len(d) == 0

    def __hash__(self):
        # for now we use the address of the object for hash, but we can change it in the future
        # to be based on the attributes e.g. with DeepHash or similar
        return hash(id(self))


def with_signature(obj, val, obj_module=None, obj_class=None):
    if obj_module is None:
        obj_module = obj.__module__.__str__()
    if obj_class is None:
        obj_class = obj.__class__.__name__
    d = {'_module': obj_module, '_class': obj_class, '_object': val}
    if hasattr(obj, 'dtype'):
        d.update({'dtype': str(obj.dtype)})
    return d


@singledispatch
def to_serializable(obj):
    logger.debug('standard case')
    return str(obj)


@to_serializable.register(pandapowerNet)
def json_pandapowernet(obj):
    logger.debug('pandapowerNet')
    net_dict = {k: item for k, item in obj.items() if not k.startswith("_")}
    for k, item in net_dict.items():
        if isinstance(item, str) and '_module' in item:
            net_dict[k] = json.loads(item)
    d = with_signature(obj, net_dict)
    return d


@to_serializable.register(pd.DataFrame)
def json_dataframe(obj):
    logger.debug('DataFrame')
    orient = "split" if not isinstance(obj.index, pd.MultiIndex) and not isinstance(
        obj.columns, pd.MultiIndex) else "columns"
    json_string = obj.to_json(orient=orient, default_handler=to_serializable, double_precision=15)
    d = with_signature(obj, json_string)
    d['orient'] = orient
    if len(obj.columns) > 0 and isinstance(obj.columns[0], str):
        d['dtype'] = obj.dtypes.astype('str').to_dict()

    # store index name (to covert by pandas with orient "split" or "columns" (and totally not with
    # Multiindex))
    if obj.index.name is not None:
        d['index_name'] = obj.index.name
    if obj.columns.name is not None:
        d['column_name'] = obj.columns.name
    if isinstance(obj.index, pd.MultiIndex) and set(obj.index.names) != {None}:
        d['index_names'] = obj.index.names
    if isinstance(obj.columns, pd.MultiIndex) and set(obj.columns.names) != {None}:
        d['column_names'] = obj.columns.names

    # store info that index is of type Multiindex originally
    d['is_multiindex'] = isinstance(obj.index, pd.MultiIndex)
    d['is_multicolumn'] = isinstance(obj.columns, pd.MultiIndex)

    return d


if GEOPANDAS_INSTALLED:
    @to_serializable.register(geopandas.GeoDataFrame)
    def json_geodataframe(obj):
        logger.debug('GeoDataFrame')
        d = with_signature(obj, obj.to_json())
        d.update({'dtype': obj.dtypes.astype('str').to_dict(),
                  'crs': obj.crs, 'columns': obj.columns})
        return d


@to_serializable.register(pd.Series)
def json_series(obj):
    logger.debug('Series')
    orient = "split" if not isinstance(obj.index, pd.MultiIndex) else "index"
    d = with_signature(obj, obj.to_json(orient=orient, default_handler=to_serializable,
                                        double_precision=15))
    d.update({'dtype': str(obj.dtypes), 'orient': orient, 'typ': 'series'})

    # store index name (to covert by pandas with orient "split" or "columns" (and totally not with
    # Multiindex))
    if obj.index.name is not None:
        d['index_name'] = obj.index.name
    if isinstance(obj.index, pd.MultiIndex) and set(obj.index.names) != {None}:
        d['index_names'] = obj.index.names

    # store info that index is of type Multiindex originally
    d['is_multiindex'] = isinstance(obj.index, pd.MultiIndex)

    return d


@to_serializable.register(numpy.ndarray)
def json_array(obj):
    logger.debug("ndarray")
    d = with_signature(obj, list(obj), obj_module='numpy', obj_class='array')
    return d


@to_serializable.register(numpy.integer)
def json_npint(obj):
    logger.debug("integer")
    d = with_signature(obj, int(obj), obj_module="numpy")
    d.pop('dtype')
    return d


@to_serializable.register(numpy.floating)
def json_npfloat(obj):
    logger.debug("floating")
    if numpy.isnan(obj):
        d = with_signature(obj, str(obj), obj_module="numpy")
    else:
        d = with_signature(obj, float(obj), obj_module="numpy")
    d.pop('dtype')
    return d


@to_serializable.register(numpy.bool_)
def json_npbool(obj):
    logger.debug("boolean")
    d = with_signature(obj, "true" if obj else "false", obj_module="numpy")
    d.pop('dtype')
    return d


@to_serializable.register(numbers.Number)
def json_num(obj):
    logger.debug("numbers.Number")
    return str(obj)


@to_serializable.register(complex)
def json_complex(obj):
    logger.debug("complex")
    d = with_signature(obj, str(obj), obj_module='builtins', obj_class='complex')
    d.pop('dtype')
    return d


@to_serializable.register(pd.Index)
def json_pdindex(obj):
    logger.debug("pd.Index")
    return with_signature(obj, list(obj), obj_module='pandas')


@to_serializable.register(bool)
def json_bool(obj):
    logger.debug("bool")
    return "true" if obj else "false"


@to_serializable.register(tuple)
def json_tuple(obj):
    logger.debug("tuple")
    d = with_signature(obj, list(obj), obj_module='builtins', obj_class='tuple')
    return d


@to_serializable.register(set)
def json_set(obj):
    logger.debug("set")
    d = with_signature(obj, list(obj), obj_module='builtins', obj_class='set')
    return d


@to_serializable.register(frozenset)
def json_frozenset(obj):
    logger.debug("frozenset")
    d = with_signature(obj, list(obj), obj_module='builtins', obj_class='frozenset')
    return d


@to_serializable.register(networkx.Graph)
def json_networkx(obj):
    logger.debug("nx graph")
    json_string = json_graph.adjacency_data(obj, attrs={'id': 'json_id', 'key': 'json_key'})
    d = with_signature(obj, json_string, obj_module="networkx")
    return d


@to_serializable.register(JSONSerializableClass)
def controller_to_serializable(obj):
    logger.debug('JSONSerializableClass')
    d = with_signature(obj, obj.to_json())
    return d


def mkdirs_if_not_existent(dir_to_create):
    already_exist = os.path.isdir(dir_to_create)
    os.makedirs(dir_to_create, exist_ok=True)
    return ~already_exist


if SHAPELY_INSTALLED:
    @to_serializable.register(shapely.geometry.LineString)
    def json_linestring(obj):
        logger.debug("shapely linestring")
        json_string = shapely.geometry.mapping(obj)
        d = with_signature(obj, json_string, obj_module="shapely")
        return d


    @to_serializable.register(shapely.geometry.Point)
    def json_point(obj):
        logger.debug("shapely Point")
        json_string = shapely.geometry.mapping(obj)
        d = with_signature(obj, json_string, obj_module="shapely")
        return d


    @to_serializable.register(shapely.geometry.Polygon)
    def json_polygon(obj):
        logger.debug("shapely Polygon")
        json_string = shapely.geometry.mapping(obj)
        d = with_signature(obj, json_string, obj_module="shapely")
        return d
