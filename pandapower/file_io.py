# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import copy
import json
import os
import pickle
import sys
from packaging import version
from warnings import warn

try:
    from fiona.crs import from_epsg
    from geopandas import GeoDataFrame, GeoSeries
    from shapely.geometry import Point, LineString

    GEOPANDAS_INSTALLED = True
except ImportError:
    GEOPANDAS_INSTALLED = False

import pandas as pd

import numpy

from pandapower.auxiliary import pandapowerNet
from pandapower.create import create_empty_network
from pandapower.convert_format import convert_format
from pandapower.io_utils import to_dict_of_dfs, from_dict_of_dfs, PPJSONEncoder, PPJSONDecoder


def to_pickle(net, filename):
    """
    Saves a pandapower Network with the pickle library.

    INPUT:
        **net** (dict) - The pandapower format network

        **filename** (string) - The absolute or relative path to the output file or an writable file-like objectxs

    EXAMPLE:

        >>> pp.to_pickle(net, os.path.join("C:", "example_folder", "example1.p"))  # absolute path
        >>> pp.to_pickle(net, "example2.p")  # relative path

    """
    if hasattr(filename, 'write'):
        pickle.dump(dict(net), filename, protocol=2)
        return
    if not filename.endswith(".p"):
        raise Exception("Please use .p to save pandapower networks!")
    save_net = dict()
    for key, item in net.items():
        if hasattr(item, "columns") and "geometry" in item.columns:
            # we convert shapely-objects to primitive data-types on a deepcopy
            item = copy.deepcopy(item)
            if key == "bus_geodata" and not isinstance(item.geometry.values[0], tuple):
                item["geometry"] = item.geometry.apply(lambda x: (x.x, x.y))
            elif key == "line_geodata" and not isinstance(item.geometry.values[0], list):
                item["geometry"] = item.geometry.apply(lambda x: list(x.coords))

        save_net[key] = {"DF": item.to_dict("split"), "dtypes": {col: dt
                                                                 for col, dt in
                                                                 zip(item.columns, item.dtypes)}} \
            if isinstance(item, pd.DataFrame) else item

    with open(filename, "wb") as f:
        pickle.dump(save_net, f, protocol=2)  # use protocol 2 for py2 / py3 compatibility


def to_excel(net, filename, include_empty_tables=False, include_results=True):
    """
    Saves a pandapower Network to an excel file.

    INPUT:
        **net** (dict) - The pandapower format network

        **filename** (string) - The absolute or relative path to the output file

    OPTIONAL:
        **include_empty_tables** (bool, False) - empty element tables are saved as excel sheet

        **include_results** (bool, True) - results are included in the excel sheet

    EXAMPLE:

        >>> pp.to_excel(net, os.path.join("C:", "example_folder", "example1.xlsx"))  # absolute path
        >>> pp.to_excel(net, "example2.xlsx")  # relative path

    """
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    dict_net = to_dict_of_dfs(net, include_results=include_results)
    for item, table in dict_net.items():
        table.to_excel(writer, sheet_name=item)
    writer.save()


def to_json_string(net):
    """
        Returns a pandapower Network in JSON format. The index columns of all pandas DataFrames will
        be saved in ascending order. net elements which name begins with "_" (internal elements)
        will not be saved. Std types will also not be saved.

        INPUT:
            **net** (dict) - The pandapower format network

            **filename** (string) - The absolute or relative path to the input file.

        EXAMPLE:

             >>> json = pp.to_json_string(net)

    """
    json_string = "{"
    for k in sorted(net.keys()):
        if k[0] == "_":
            continue
        json_string += '"%s":%s,' % (k, json.dumps(net[k], cls=PPJSONEncoder, indent=4))
    json_string = json_string[:-1] + "}\n"
    return json_string


def to_json(net, filename=None):
    """
        Saves a pandapower Network in JSON format. The index columns of all pandas DataFrames will
        be saved in ascending order. net elements which name begins with "_" (internal elements)
        will not be saved. Std types will also not be saved.

        INPUT:
            **net** (dict) - The pandapower format network

            **filename** (string or file) - The absolute or relative path to the output file or file-like object

        EXAMPLE:

             >>> pp.to_json(net, "example.json")

    """
    if hasattr(filename, 'write'):
        json.dump(net, fp=filename, cls=PPJSONEncoder, indent=4)
    else:
        with open(filename, "w") as fp:
            json.dump(net, fp=fp, cls=PPJSONEncoder, indent=4)


def to_sql(net, con, include_results=True):
    dodfs = to_dict_of_dfs(net, include_results=include_results)
    for name, data in dodfs.items():
        data.to_sql(name, con, if_exists="replace")


def to_sqlite(net, filename, include_results=True):
    import sqlite3
    conn = sqlite3.connect(filename)
    to_sql(net, conn, include_results)
    conn.close()


def from_pickle(filename, convert=True):
    """
    Load a pandapower format Network from pickle file

    INPUT:
        **filename** (string or file) - The absolute or relative path to the input file or file-like object

    OUTPUT:
        **net** (dict) - The pandapower format network

    EXAMPLE:

        >>> net1 = pp.from_pickle(os.path.join("C:", "example_folder", "example1.p")) #absolute path
        >>> net2 = pp.from_pickle("example2.p") #relative path

    """

    def read(f):
        if sys.version_info >= (3, 0):
            return pickle.load(f, encoding='latin1')
        else:
            return pickle.load(f)

    if hasattr(filename, 'read'):
        net = read(filename)
    elif not os.path.isfile(filename):
        raise UserWarning("File %s does not exist!!" % filename)
    else:
        with open(filename, "rb") as f:
            net = read(f)
    net = pandapowerNet(net)

    try:
        epsg = net.gis_epsg_code
    except AttributeError:
        epsg = None

    for key, item in net.items():
        if isinstance(item, dict) and "DF" in item:
            df_dict = item["DF"]
            if "columns" in df_dict:
                # make sure the index is Int64Index
                try:
                    df_index = pd.Int64Index(df_dict['index'])
                except TypeError:
                    df_index = df_dict['index']
                if GEOPANDAS_INSTALLED and "geometry" in df_dict["columns"] \
                        and epsg is not None:
                    # convert primitive data-types to shapely-objects
                    if key == "bus_geodata":
                        data = {"x": [row[0] for row in df_dict["data"]],
                                "y": [row[1] for row in df_dict["data"]]}
                        geo = [Point(row[2][0], row[2][1]) for row in df_dict["data"]]
                    elif key == "line_geodata":
                        data = {"coords": [row[0] for row in df_dict["data"]]}
                        geo = [LineString(row[1]) for row in df_dict["data"]]

                    net[key] = GeoDataFrame(data, crs=from_epsg(epsg), geometry=geo,
                                            index=df_index)
                else:
                    net[key] = pd.DataFrame(columns=df_dict["columns"], index=df_index,
                                            data=df_dict["data"])
            else:
                net[key] = pd.DataFrame.from_dict(df_dict)
                if "columns" in item:
                    if version.parse(pd.__version__) < version.parse("0.21"):
                        net[key] = net[key].reindex_axis(item["columns"], axis=1)
                    else:
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
    if convert:
        convert_format(net)
    return net


def from_excel(filename, convert=True):
    """
    Load a pandapower network from an excel file

    INPUT:
        **filename** (string) - The absolute or relative path to the input file.

    OUTPUT:
        **convert** (bool) - use the convert format function to

        **net** (dict) - The pandapower format network

    EXAMPLE:

        >>> net1 = pp.from_excel(os.path.join("C:", "example_folder", "example1.xlsx")) #absolute path
        >>> net2 = pp.from_excel("example2.xlsx") #relative path

    """

    if not os.path.isfile(filename):
        raise UserWarning("File %s does not exist!" % filename)
    pd_version = version.parse(pd.__version__)
    if pd_version < version.parse("0.21"):
        xls = pd.ExcelFile(filename).parse(sheetname=None)
    elif pd_version < version.parse("0.24"):
        xls = pd.ExcelFile(filename).parse(sheet_name=None)
    else:
        xls = pd.ExcelFile(filename).parse(sheet_name=None, index_col=0)

    try:
        net = from_dict_of_dfs(xls)
    except:
        net = _from_excel_old(xls)
    if convert:
        convert_format(net)
    return net


def _from_excel_old(xls):
    par = xls["parameters"]["parameters"]
    name = None if pd.isnull(par.at["name"]) else par.at["name"]
    net = create_empty_network(name=name, f_hz=par.at["f_hz"])

    for item, table in xls.items():
        if item == "parameters":
            continue
        elif item.endswith("std_types"):
            item = item.split("_")[0]
            for std_type, tab in table.iterrows():
                net.std_types[item][std_type] = dict(tab)
        elif item == "line_geodata":
            points = int(len(table.columns) / 2)
            for i, coords in table.iterrows():
                coord = [(coords["x%u" % nr], coords["y%u" % nr]) for nr in range(points)
                         if pd.notnull(coords["x%u" % nr])]
                net.line_geodata.loc[i, "coords"] = coord
        else:
            net[item] = table
    return net


def from_json(filename, convert=True):
    """
    Load a pandapower network from a JSON file.
    The index of the returned network is not necessarily in the same order as the original network.
    Index columns of all pandas DataFrames are sorted in ascending order.

    INPUT:
        **filename** (string or file) - The absolute or relative path to the input file or file-like object

    OUTPUT:
        **convert** (bool) - use the convert format function to

        **net** (dict) - The pandapower format network

    EXAMPLE:

        >>> net = pp.from_json("example.json")

    """
    if hasattr(filename, 'read'):
        net = json.load(filename, cls=PPJSONDecoder)
    elif not os.path.isfile(filename):
        raise UserWarning("File %s does not exist!!" % filename)
    else:
        with open(filename) as fp:
            net = json.load(fp, cls=PPJSONDecoder)
            # this can be removed in the future
            # now net is saved with "_module", "_class", "_object"..., so json.load already returns
            # pandapowerNet. Older files don't have it yet, and are loaded as dict.
            # After some time, this part can be removed.
            if not isinstance(net, pandapowerNet):
                warn("This net is saved in older format, which will not be supported in future.\r\n"
                     "Please resave your grid using the current pandapower version.",
                     DeprecationWarning)
                net = from_json_dict(net)
    if convert:
        convert_format(net)
    return net


def from_json_string(json_string, convert=False):
    """
    Load a pandapower network from a JSON string.
    The index of the returned network is not necessarily in the same order as the original network.
    Index columns of all pandas DataFrames are sorted in ascending order.

    INPUT:
        **json_string** (string) - The json string representation of the network

    OUTPUT:
        **convert** (bool) - use the convert format function to

        **net** (dict) - The pandapower format network

    EXAMPLE:

        >>> net = pp.from_json_string(json_str)

    """
    data = json.loads(json_string, cls=PPJSONDecoder)
    net = from_json_dict(data)

    if convert:
        convert_format(net)
    return net


def from_json_dict(json_dict):
    """
    Load a pandapower network from a JSON string.
    The index of the returned network is not necessarily in the same order as the original network.
    Index columns of all pandas DataFrames are sorted in ascending order.

    INPUT:
        **json_dict** (json) - The json object representation of the network

    OUTPUT:
        **convert** (bool) - use the convert format function to

        **net** (dict) - The pandapower format network

    EXAMPLE:

        >>> net = pp.pp.from_json_dict(json.loads(json_str))

    """
    name = json_dict["name"] if "name" in json_dict else None
    f_hz = json_dict["f_hz"] if "f_hz" in json_dict else 50
    net = create_empty_network(name=name, f_hz=f_hz)
    if "parameters" in json_dict:
        for par, value in json_dict["parameters"]["parameter"].items():
            net[par] = value

    for key in sorted(json_dict.keys()):
        if key == 'dtypes':
            continue
        if key in net and isinstance(net[key], pd.DataFrame) and isinstance(json_dict[key], dict) \
             or key == "piecewise_linear_cost" or key == "polynomial_cost":
            net[key] = pd.DataFrame.from_dict(json_dict[key], orient="columns")
            net[key].set_index(net[key].index.astype(numpy.int64), inplace=True)
        else:
            net[key] = json_dict[key]
    return net


def from_sql(con):
    cursor = con.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    dodfs = dict()
    for t, in cursor.fetchall():
        table = pd.read_sql_query("SELECT * FROM %s" % t, con, index_col="index")
        table.index.name = None
        dodfs[t] = table
    net = from_dict_of_dfs(dodfs)
    return net


def from_sqlite(filename, netname=""):
    import sqlite3
    con = sqlite3.connect(filename)
    net = from_sql(con)
    con.close()
    return net
