# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pandas as pd
from pandapower.create import create_empty_network
from pandapower.auxiliary import pandapowerNet
import numpy
import numbers
import json
import copy
import networkx
from networkx.readwrite import json_graph
import importlib
from numpy import ndarray
from warnings import warn

try:
    from functools import singledispatch
except ImportError:
    # Python 2.7
    from singledispatch import singledispatch

try:
    import fiona
    import geopandas as gpd

    GEOPANDAS_INSTALLED = True
except ImportError:
    GEOPANDAS_INSTALLED = False


try:
    import shapely.geometry
    SHAPELY_INSTALLED = True
except ImportError:
    SHAPELY_INSTALLED = False
    

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def coords_to_df(value, geotype="line"):
    geo = pd.DataFrame()
    for i in value.index:
        # get bus x y to save
        if geotype == "bus":
            geo.loc[i, "x"] = value.at[i, 'x']
            geo.loc[i, "y"] = value.at[i, 'y']
        # get coords and convert them to x1, y1, x2, y2...
        coords = value.at[i, 'coords']

        if isinstance(coords, list) or isinstance(coords, ndarray):
            for nr, (x, y) in enumerate(coords):
                geo.loc[i, "x%u" % nr] = x
                geo.loc[i, "y%u" % nr] = y
        elif pd.isnull(coords):
            continue
        else:
            logger.error("unkown format for coords for value {}".format(value))
            raise ValueError("coords unkown format")
    return geo


def to_dict_of_dfs(net, include_results=False, fallback_to_pickle=True):
    dodfs = dict()
    dodfs["dtypes"] = collect_all_dtypes_df(net)
    dodfs["parameters"] = pd.DataFrame(columns=["parameter"])
    for item, value in net.items():
        # dont save internal variables and results (if not explicitely specified)
        if item.startswith("_") or (item.startswith("res") and not include_results):
            continue
        elif item == "std_types":
            for t in net.std_types.keys():  # which are ["line", "trafo", "trafo3w"]
                dodfs["%s_std_types" % t] = pd.DataFrame(net.std_types[t]).T
        elif item == "user_pf_options":
            if len(value) > 0:
                dodfs["user_pf_options"] = pd.DataFrame(value, index=[0])
        elif isinstance(value, (int, float, bool, str)):
            # attributes of primitive types are just stored in a DataFrame "parameters"
            dodfs["parameters"].loc[item] = net[item]
        elif not isinstance(value, pd.DataFrame) and \
                (GEOPANDAS_INSTALLED and not isinstance(value, gpd.GeoDataFrame)):
            logger.warning("Could not serialize net.%s" % item)
        elif item == "bus_geodata":
            geo = coords_to_df(value, geotype="bus")
            dodfs[item] = geo
        elif item == "line_geodata":
            geo = coords_to_df(value)
            dodfs[item] = geo
        else:
            dodfs[item] = value
    return dodfs


def collect_all_dtypes_df(net):
    dtypes = []
    for element, table in net.items():
        if not hasattr(table, "dtypes"):
            continue
        for item, dtype in table.dtypes.iteritems():
            dtypes.append((element, item, str(dtype)))
    return pd.DataFrame(dtypes, columns=["element", "column", "dtype"])


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
    return net


def from_dict_of_dfs(dodfs):
    net = create_empty_network()
    for p, v in dodfs["parameters"].iterrows():
        net[p] = v.parameter
    for item, table in dodfs.items():
        if item in ("parameters", "dtypes"):
            continue
        elif item == "line_geodata":
            net = df_to_coords(net, item, table)
        elif item == "bus_geodata":
            net = df_to_coords(net, item, table)
        elif item.endswith("_std_types"):
            net["std_types"][item[:-10]] = table.T.to_dict()
            continue  # don't go into try..except
        elif item == "user_pf_options":
            net['user_pf_options'] = {c: v for c, v in zip(table.columns, table.values[0])}
            continue  # don't go into try..except
        else:
            net[item] = table
        # set the index to be Int64Index
        try:
            net[item].set_index(net[item].index.astype(numpy.int64), inplace=True)
        except TypeError:
            # TypeError: if not int64 index (e.g. str)
            pass
    restore_all_dtypes(net, dodfs["dtypes"])
    return net


def restore_all_dtypes(net, dtypes):
    for _, v in dtypes.iterrows():
        try:
            net[v.element][v.column] = net[v.element][v.column].astype(v["dtype"])
        except KeyError:
            pass


from json.encoder import _make_iterencode
from json.encoder import *


class PPJSONEncoder(json.JSONEncoder):

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
            _encoder = encode_basestring_ascii
        else:
            _encoder = encode_basestring

        def floatstr(o, allow_nan=self.allow_nan,
                     _repr=float.__repr__, _inf=INFINITY, _neginf=-INFINITY):
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
                    "Out of range float values are not JSON compliant: " +
                    repr(o))

            return text

        _iterencode = _make_iterencode(
            markers, self.default, _encoder, self.indent, floatstr,
            self.key_separator, self.item_separator, self.sort_keys,
            self.skipkeys, _one_shot, isinstance=isinstance_partial)
        return _iterencode(o, 0)

    def default(self, o):
        try:
            s = to_serializable(o)
        except TypeError:
            # Let the base class default method raise the TypeError
            return json.JSONEncoder.default(self, o)
        else:
            return s


def isinstance_partial(obj, cls):
    if isinstance(obj, (pandapowerNet, tuple)):
        return False
    return isinstance(obj, cls)


class PPJSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super(PPJSONDecoder, self).__init__(object_hook=pp_hook, *args, **kwargs)


def pp_hook(d):
    if '_module' in d.keys() and '_class' in d.keys():
        class_name = d.pop('_class')
        module_name = d.pop('_module')
        obj = d.pop('_object')
        keys = copy.deepcopy(list(d.keys()))
        for key in keys:
            if isinstance(d[key], dict):
                d[key] = pp_hook(d[key])

        if class_name == 'Series':
            return pd.read_json(obj, precise_float=True, **d)
        elif class_name == "DataFrame":
            df = pd.read_json(obj, precise_float=True, **d)
            try:
                df.set_index(df.index.astype(numpy.int64), inplace=True)
            except (ValueError, TypeError, AttributeError):
                logger.debug("failed setting int64 index")
            return df
        elif GEOPANDAS_INSTALLED and class_name == 'GeoDataFrame':
            df = gpd.GeoDataFrame.from_features(fiona.Collection(obj), crs=d['crs'])
            if "id" in df:
                df.set_index(df['id'].values.astype(numpy.int64), inplace=True)
            # coords column is not handled properly when using from_features
            if 'coords' in df:
                # df['coords'] = df.coords.apply(json.loads)
                valid_coords = ~pd.isnull(df.coords)
                df.loc[valid_coords, 'coords'] = df.loc[valid_coords, "coords"].apply(json.loads)
            df = df.reindex(columns=d['columns'])
            return df
        elif SHAPELY_INSTALLED and module_name == "shapely":
            return shapely.geometry.shape(obj)
        elif class_name == "pandapowerNet":
            from pandapower import from_json_string
            return from_json_string(obj)
        elif module_name == "networkx":
            return json_graph.adjacency_graph(obj, attrs={'id': 'json_id', 'key': 'json_key'})
        else:
            module = importlib.import_module(module_name)
            class_ = getattr(module, class_name)
            return class_(obj, **d)
    else:
        return d


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
def json_net(obj):
    from pandapower.file_io import to_json_string
    d = with_signature(obj, to_json_string(obj))
    return d


@to_serializable.register(pd.DataFrame)
def json_dataframe(obj):
    logger.debug('DataFrame')
    d = with_signature(obj, obj.to_json(orient='split',
                                        default_handler=to_serializable, double_precision=15))
    d.update({'dtype': obj.dtypes.astype('str').to_dict(), 'orient': 'split'})
    return d


if GEOPANDAS_INSTALLED:
    @to_serializable.register(gpd.GeoDataFrame)
    def json_geodataframe(obj):
        logger.debug('GeoDataFrame')
        d = with_signature(obj, obj.to_json())
        d.update({'dtype': obj.dtypes.astype('str').to_dict(),
                  'crs': obj.crs, 'columns': obj.columns})
        return d


@to_serializable.register(pd.Series)
def json_series(obj):
    logger.debug('Series')
    d = with_signature(obj, obj.to_json(orient='split', default_handler=to_serializable))
    d.update({'dtype': str(obj.dtypes), 'orient': 'split', 'typ': 'series'})
    return d


@to_serializable.register(numpy.ndarray)
def json_array(obj):
    logger.debug("ndarray")
    d = with_signature(obj, list(obj), obj_module='numpy', obj_class='array')
    return d


@to_serializable.register(numpy.integer)
def json_npint(obj):
    logger.debug("integer")
    return int(obj)


@to_serializable.register(numpy.floating)
def json_npfloat(obj):
    logger.debug("floating")
    return float(obj)


@to_serializable.register(numbers.Number)
def json_num(obj):
    logger.debug("numbers.Number")
    return str(obj)


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