# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import json
import os
import pickle
from warnings import warn

import numpy
import pandas as pd
from packaging import version

import pandapower.io_utils as io_utils
from pandapower.auxiliary import pandapowerNet
from pandapower.convert_format import convert_format
from pandapower.create import create_empty_network


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
    save_net = io_utils.to_dict_with_coord_transform(net, ["bus_geodata"], ["line_geodata"])

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
    dict_net = io_utils.to_dict_of_dfs(net, include_results=include_results,
                                       include_empty_tables=include_empty_tables)
    for item, table in dict_net.items():
        table.to_excel(writer, sheet_name=item)
    writer.save()


def to_json(net, filename=None, encryption_key=None):
    """
        Saves a pandapower Network in JSON format. The index columns of all pandas DataFrames will
        be saved in ascending order. net elements which name begins with "_" (internal elements)
        will not be saved. Std types will also not be saved.

        INPUT:
            **net** (dict) - The pandapower format network

            **filename** (string or file, None) - The absolute or relative path to the output file
                                                  or a file-like object,
                                                  if 'None' the function returns a json string

            **encrytion_key** (string, None) - If given, the pandapower network is stored as an
                                               encrypted json string


        EXAMPLE:

             >>> pp.to_json(net, "example.json")

    """
    json_string = json.dumps(net, cls=io_utils.PPJSONEncoder, indent=2)
    if encryption_key is not None:
        json_string = io_utils.encrypt_string(json_string, encryption_key)

    if filename is None:
        return json_string

    if hasattr(filename, 'write'):
        filename.write(json_string)
    else:
        with open(filename, "w") as fp:
            fp.write(json_string)


def to_sql(net, con, include_results=True):
    dodfs = io_utils.to_dict_of_dfs(net, include_results=include_results)
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

        **convert** (bool, True) - If True, converts the format of the net loaded from pickle from the older
            version of pandapower to the newer version format

    OUTPUT:
        **net** (dict) - The pandapower format network

    EXAMPLE:

        >>> net1 = pp.from_pickle(os.path.join("C:", "example_folder", "example1.p")) #absolute path
        >>> net2 = pp.from_pickle("example2.p") #relative path

    """

    net = pandapowerNet(io_utils.get_raw_data_from_pickle(filename))
    io_utils.transform_net_with_df_and_geo(net, ["bus_geodata"], ["line_geodata"])

    if convert:
        convert_format(net)
    return net


def from_excel(filename, convert=True):
    """
    Load a pandapower network from an excel file

    INPUT:
        **filename** (string) - The absolute or relative path to the input file.

        **convert** (bool, True) - If True, converts the format of the net loaded from excel from
            the older version of pandapower to the newer version format

    OUTPUT:
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
        xls = pd.read_excel(filename, sheet_name=None, index_col=0, engine="openpyxl")
        
    try:
        net = io_utils.from_dict_of_dfs(xls)
    except:
        net = _from_excel_old(xls)
    if convert:
        convert_format(net)
    return net


def _from_excel_old(xls):
    par = xls["parameters"]["parameter"]
    name = None if pd.isnull(par.at["name"]) else par.at["name"]
    net = create_empty_network(name=name, f_hz=par.at["f_hz"])
    net.update(par)
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


def from_json(filename, convert=True, encryption_key=None):
    """
    Load a pandapower network from a JSON file.
    The index of the returned network is not necessarily in the same order as the original network.
    Index columns of all pandas DataFrames are sorted in ascending order.

    INPUT:
        **filename** (string or file) - The absolute or relative path to the input file or file-like object

        **convert** (bool, True) - If True, converts the format of the net loaded from json from the older
            version of pandapower to the newer version format

        **encrytion_key** (string, "") - If given, key to decrypt an encrypted pandapower network

    OUTPUT:
        **net** (dict) - The pandapower format network

    EXAMPLE:

        >>> net = pp.from_json("example.json")

    """
    if hasattr(filename, 'read'):
        json_string = filename.read()
    elif not os.path.isfile(filename):
        raise UserWarning("File {} does not exist!!".format(filename))
    else:
        with open(filename, "r") as fp:
            json_string = fp.read()

    return from_json_string(json_string, convert=convert, encryption_key=encryption_key)


def from_json_string(json_string, convert=False, encryption_key=None):
    """
    Load a pandapower network from a JSON string.
    The index of the returned network is not necessarily in the same order as the original network.
    Index columns of all pandas DataFrames are sorted in ascending order.

    INPUT:
        **json_string** (string) - The json string representation of the network

        **convert** (bool, False) - If True, converts the format of the net loaded from json_string from the
            older version of pandapower to the newer version format

        **encrytion_key** (string, "") - If given, key to decrypt an encrypted json_string

    OUTPUT:
        **net** (dict) - The pandapower format network

    EXAMPLE:

        >>> net = pp.from_json_string(json_str)

    """
    if encryption_key is not None:
        json_string = io_utils.decrypt_string(json_string, encryption_key)

    net = json.loads(json_string, cls=io_utils.PPJSONDecoder)
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


def from_json_dict(json_dict):
    """
    Load a pandapower network from a JSON string.
    The index of the returned network is not necessarily in the same order as the original network.
    Index columns of all pandas DataFrames are sorted in ascending order.

    INPUT:
        **json_dict** (json) - The json object representation of the network

    OUTPUT:
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
    net = io_utils.from_dict_of_dfs(dodfs)
    return net


def from_sqlite(filename, netname=""):
    import sqlite3
    con = sqlite3.connect(filename)
    net = from_sql(con)
    con.close()
    return net
