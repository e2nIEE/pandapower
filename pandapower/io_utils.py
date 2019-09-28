# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pandas as pd
from pandas.util.testing import assert_series_equal, assert_frame_equal
from pandapower.create import create_empty_network
from pandapower.auxiliary import pandapowerNet
import numpy
import numbers
import json
import copy
import networkx
from networkx.readwrite import json_graph
import importlib
from numpy import ndarray, generic, equal, isnan, allclose, any as anynp
from warnings import warn
from inspect import isclass, signature
import os

try:
    from functools import singledispatch
except ImportError:
    # Python 2.7
    from singledispatch import singledispatch

try:
    import fiona
    import geopandas

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
    columns = ["x", "y", "coords"] if geotype == "bus" else ["coords"]
    geo = pd.DataFrame(columns=columns, index=value.index)
    if any(~value.coords.isnull()):
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


def to_dict_of_dfs(net, include_results=False, fallback_to_pickle=True, include_empty_tables=True):
    dodfs = dict()
    dtypes = []
    dodfs["parameters"] = dict()  # pd.DataFrame(columns=["parameter"])
    for item, value in net.items():
        # dont save internal variables and results (if not explicitely specified)
        if item.startswith("_") or (item.startswith("res") and not include_results):
            continue
        elif item == "std_types":
            for t in net.std_types.keys():  # which are ["line", "trafo", "trafo3w"]
                dodfs["%s_std_types" % t] = pd.DataFrame(net.std_types[t]).T
            continue
        elif item == "user_pf_options":
            if len(value) > 0:
                dodfs["user_pf_options"] = pd.DataFrame(value, index=[0])
            continue
        elif isinstance(value, (int, float, bool, str)):
            # attributes of primitive types are just stored in a DataFrame "parameters"
            dodfs["parameters"][item] = net[item]
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
        else:
            dodfs[item] = value
        # save dtypes 
        for column, dtype in value.dtypes.iteritems():
            dtypes.append((item, column, str(dtype)))
    dodfs["dtypes"] = pd.DataFrame(dtypes, columns=["element", "column", "dtype"])
    dodfs["parameters"] = pd.DataFrame(dodfs["parameters"], index=[0])
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


def from_dict_of_dfs(dodfs):
    sheets = ('line_std_types', 'trafo_std_types', 'trafo3w_std_types')
    types = ('line', 'trafo', 'trafo3w')
    # creates each std_types independently
    add_stdtypes = tuple(t for sh, t in zip(sheets, types) if sh in dodfs.keys())

    net = create_empty_network(add_stdtypes=add_stdtypes)
    for c in dodfs["parameters"].columns:
        net[c] = dodfs["parameters"].at[0, c]
    for item, table in dodfs.items():
        if item in ("parameters", "dtypes"):
            continue
        elif item in ["line_geodata", "bus_geodata"]:
            df_to_coords(net, item, table)
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
            if v["dtype"] == "object":
                c = net[v.element][v.column]
                net[v.element][v.column] = numpy.where(c.isnull(), None, c)
                # net[v.element][v.column] = net[v.element][v.column].fillna(value=None)
            net[v.element][v.column] = net[v.element][v.column].astype(v["dtype"])
        except KeyError:
            pass


def isinstance_partial(obj, cls):
    if isinstance(obj, (pandapowerNet, tuple)):
        return False
    return isinstance(obj, cls)


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


class PPJSONDecoder(json.JSONDecoder):
    def __init__(self, **kwargs):
        args = {"object_hook": pp_hook}
        args.update(kwargs)
        super().__init__(**args)


def pp_hook(d):
    if '_module' in d and '_class' in d:
        if "_object" in d:
            obj = d.pop('_object')
        elif "_init" in d:
            return d  # backwards compatibility
        else:
            obj = {"_init": d, "_state": dict()}  # backwards compatibility
        class_name = d.pop('_class')
        module_name = d.pop('_module')
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
            df = geopandas.GeoDataFrame.from_features(fiona.Collection(obj), crs=d['crs'])
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
            if isinstance(obj, str):  # backwards compatibility
                from pandapower import from_json_string
                return from_json_string(obj)
            else:
                net = create_empty_network()
                net.update(obj)
                return net
        elif module_name == "networkx":
            return json_graph.adjacency_graph(obj, attrs={'id': 'json_id', 'key': 'json_key'})
        else:
            module = importlib.import_module(module_name)
            class_ = getattr(module, class_name)
            if isclass(class_) and issubclass(class_, JSONSerializableClass):
                needs_net = "net" in signature(class_.__init__).parameters.keys()
                if needs_net and not hasattr(pp_hook, "net"):
                    return json.dumps(
                        {"_object": obj, "_class": class_name, "_module": module_name})
                else:
                    if isinstance(obj, str):
                        obj = json.loads(obj, cls=PPJSONDecoder)
                    if needs_net:
                        obj["_init"]["net"] = pp_hook.net
                    if "add_to_net" in obj["_init"]:
                        obj["_init"]["add_to_net"] = False
                    return class_.from_dict(obj)
            else:
                return class_(obj, **d)
    else:
        return d


class JSONSerializableClass(object):
    json_excludes = ["net", "self", "__class__"]
    
    def __init__(self, table=None, overwrite=True):
        self._init = dict()

    def update_initialized(self, parameters):
        """
        Saves all parameters as object attributes
        """
        if "kwargs" in parameters:
            self._init.update(parameters.pop("kwargs"))
        for excluded in self.json_excludes:
            if excluded in parameters:
                del parameters[excluded]
        self._init.update(parameters)

    def to_json(self):
        """
        Each controller should have this method implemented. The resulting json string should be
        readable by the controller's from_json function and by the function add_ctrl_from_json in
        control_handler.
        """
        return json.dumps(self.to_dict(), cls=PPJSONEncoder)

    def to_dict(self):
        init_parameters = signature(self.__init__).parameters.keys()
        d = {'_module': self.__module__.__str__(), '_class': self.__class__.__name__,
             "_init": {key: val for key, val in self._init.items() if
                       key in init_parameters and key not in self.json_excludes},
             "has_net": "net" in init_parameters,
             "_state": {key: val for key, val in self.__dict__.items() if key not in
                        self.json_excludes and not callable(val)}}
        return d

    def add_to_net(self, table, index, column="object", overwrite=True):
        if table not in self.net:
            self.net[table] = pd.DataFrame(columns=[column])
        if index in self.net[table].index:
            obj = self.net[table].object.at[index]
            obj_is_dict = isinstance(obj, dict)
            if not obj_is_dict:
                if overwrite:
                    logger.info("Updating %s with index %s" % (table, index))
                else:
                    raise UserWarning("%s with index %s already exists" % (table, index))
        self.net[table].at[index, column] = self

    def __eq__(self, other):

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
        state = d["_state"]
        init = d["_init"]
        if "index" in state:
            init["index"] = state["index"]
        obj = cls(**init)
        obj.__dict__.update(state)
        return obj

    @classmethod
    def from_json(cls, json_string):
        d = json.loads(json_string, cls=PPJSONDecoder)
        return cls.from_dict(d)


def restore_jsoned_objects(net, obj_hook=pp_hook):
    obj_hook.net = net
    restore_columns = [(element, "object") for element, table in net.items() \
                       if isinstance(table, pd.DataFrame) and "object" in table.columns]
    if "controller" in net and "controller" in net["controller"]:
        restore_columns.append(("controller", "controller"))
    for element, column in restore_columns:
        for i, c in zip(net[element].index, net[element][column].values):
            try:
                logger.debug("loading %s with index %s"%(element, str(i)))
                net[element][column].at[i] = obj_hook(c)
            except Exception as e:
                logger.warning("did not load %s with index %s: %s"%(element, str(i), e))
    del obj_hook.net


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
    net_dict = {k: item for k, item in obj.items() if not k.startswith("_")}
    d = with_signature(obj, net_dict)
    return d


@to_serializable.register(pd.DataFrame)
def json_dataframe(obj):
    logger.debug('DataFrame')
    d = with_signature(obj, obj.to_json(orient='split',
                                        default_handler=to_serializable, double_precision=15))
    d.update({'dtype': obj.dtypes.astype('str').to_dict(), 'orient': 'split'})
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


@to_serializable.register(JSONSerializableClass)
def controller_to_serializable(obj):
    logger.debug('JSONSerializableClass')
    d = with_signature(obj, obj.to_json())
    return d


def mkdirs_if_not_existent(dir):
    if os.path.isdir(dir) == False:
        os.makedirs(dir)
        return True
    return False


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
