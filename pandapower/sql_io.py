# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import numpy as np
import pandas as pd

from pandapower.create import create_empty_network

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def to_dict_of_dfs(net, include_results=False, create_dtype_df=True):
    dodfs = {}
    dodfs["parameters"] = pd.DataFrame(columns=["parameter"])
    for item, table in net.items():
        # dont save internal variables and results (if not explicitely specified)
        if item.startswith("_") or (item.startswith("res") and not include_results):
            continue
        # attributes of "primitive" types are just stored in a DataFrame "parameters"
        elif isinstance(table, (int, float, bool, str)):
            dodfs["parameters"].loc[item] = net[item]
        elif item == "std_types":
            for t in ["line", "trafo", "trafo3w"]:
                dodfs["%s_std_types" % t] = pd.DataFrame(net.std_types[t]).T
        elif type(table) != pd.DataFrame:
            logger.warning("Attribute net.%s could not be saved !" % item)
            continue
        elif item == "bus_geodata":
            dodfs[item] = pd.DataFrame(table[["x", "y"]])
        elif item == "line_geodata":
            geo = pd.DataFrame()
            for i, coord in table.iterrows():
                for nr, (x, y) in enumerate(coord.coords):
                    geo.loc[i, "x%u" % nr] = x
                    geo.loc[i, "y%u" % nr] = y
            dodfs[item] = geo    
        else:
            dodfs[item] = table
    return dodfs


def from_dict_of_dfs(dodfs):
    net = create_empty_network()
    for p, v in dodfs["parameters"].iterrows():
        net[p] = v.parameter
    for item, table in dodfs.items():
        if item == "parameters":
            continue
        elif item == "line_geodata":
            points = len(table.columns) // 2
            for i, coords in table.iterrows():
                coord = [(coords["x%u" % nr], coords["y%u" % nr]) for nr in range(points)
                         if pd.notnull(coords["x%u" % nr])]
                net.line_geodata.loc[i, "coords"] = coord
        elif item.endswith("_std_types"):
            net["std_types"][item[:-10]] = table.T.to_dict()
        else:
            net[item] = table
    return net


def collect_all_dtypes_df(net):
    dtypes = []
    for element, table in net.items():
        if not hasattr(table, "dtypes"):
            continue
        for item, dtype in table.dtypes.iteritems():
            dtypes.append((element, item, str(dtype)))
    return pd.DataFrame(dtypes, columns=["element", "column", "dtype"])


def restore_all_dtypes(net, dtdf):
    for _, v in dtdf.iterrows():
        net[v.element][v.column] = net[v.element][v.column].astype(v["dtype"])
    

def to_sql(net, con, include_empty_tables=False, include_results=True):
    dodfs = to_dict_of_dfs(net, include_results=include_results)
    dodfs["dtypes"] = collect_all_dtypes_df(net)
    for name, data in dodfs.items():
        data.to_sql(name, con, if_exists="replace")


def from_sql(con):
    cursor = con.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    dodfs = dict()
    for t, in cursor.fetchall():
        table = pd.read_sql_query("SELECT * FROM %s" % t, con, index_col="index")
        table.index.name = None
        dodfs[t] = table
    net = from_dict_of_dfs(dodfs)
    restore_all_dtypes(net, dodfs["dtypes"])
    return net


def to_sqlite(net, filename):
    import sqlite3
    conn = sqlite3.connect(filename)
    to_sql(net, conn)
    conn.close()


def from_sqlite(filename, netname=""):
    import sqlite3
    con = sqlite3.connect(filename)
    net = from_sql(con)
    con.close()
    return net


if __name__ == "__main__":
    import pandapower.networks
    from pandapower.test.toolbox import assert_net_equal
    net = networks.case9241pegase()
    to_sqlite(net, "test.db")
    n = from_sqlite("test.db")
    assert_net_equal(net, n)


                