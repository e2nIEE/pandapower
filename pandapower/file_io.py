# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

import os
import pickle
import pandas as pd
import sys
import numbers
import json
import numpy
from pandapower.toolbox import convert_format
from pandapower.create import create_empty_network
from pandapower.auxiliary import pandapowerNet


def to_pickle(net, filename):
    """
    Saves a pandapower Network with the pickle library.

    INPUT:
        **net** (dict) - The pandapower format network

        **filename** (string) - The absolute or relative path to the input file.

    EXAMPLE:

        >>> pp.to_pickle(net, os.path.join("C:", "example_folder", "example1.p"))  # absolute path
        >>> pp.to_pickle(net, "example2.p")  # relative path

    """
    if not filename.endswith(".p"):
        raise Exception("Please use .p to save pandapower networks!")
    with open(filename, "wb") as f:
        pickle.dump(dict(net), f, protocol=2) #use protocol 2 for py2 / py3 compatibility


def to_excel(net, filename, include_empty_tables=False, include_results=True):
    """
    Saves a pandapower Network to an excel file.

    INPUT:
        **net** (dict) - The pandapower format network

        **filename** (string) - The absolute or relative path to the input file.

    OPTIONAL:
        **include_empty_tables** (bool, False) - empty element tables are saved as excel sheet

        **include_results** (bool, True) - results are included in the excel sheet

    EXAMPLE:

        >>> pp.to_excel(net, os.path.join("C:", "example_folder", "example1.xlsx"))  # absolute path
        >>> pp.to_excel(net, "example2.xlsx")  # relative path

    """
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    for item, table in net.items():
        if item == "bus_geodata":
            table = pd.DataFrame(table[["x", "y"]])
        if type(table) != pd.DataFrame or item.startswith("_"):
            continue
        elif item.startswith("res"):
            if include_results and len(table) > 0:
                table.to_excel(writer, sheet_name=item)
        elif item == "line_geodata":
            geo = pd.DataFrame(index=table.index)
            for i, coord in table.iterrows():
                for nr, (x, y) in enumerate(coord.coords):
                    geo.loc[i, "x%u" % nr] = x
                    geo.loc[i, "y%u" % nr] = y
            geo.to_excel(writer, sheet_name=item)
        elif len(table) > 0 or include_empty_tables:
            table.to_excel(writer, sheet_name=item)
    parameters = pd.DataFrame(index=["name", "f_hz", "version"], columns=["parameters"],
                              data=[net.name, net.f_hz, net.version])
    pd.DataFrame(net.std_types["line"]).T.to_excel(writer, sheet_name="line_std_types")
    pd.DataFrame(net.std_types["trafo"]).T.to_excel(writer, sheet_name="trafo_std_types")
    pd.DataFrame(net.std_types["trafo3w"]).T.to_excel(writer, sheet_name="trafo3w_std_types")
    parameters.to_excel(writer, sheet_name="parameters")
    writer.save()


def to_json(net, filename):
    """
        Saves a pandapower Network in JSON format. The index columns of all pandas DataFrames will
        be saved in ascending order. net elements which name begins with "_" (internal elements)
        will not be saved. Std types will also not be saved.

        INPUT:
            **net** (dict) - The pandapower format network

            **filename** (string) - The absolute or relative path to the input file.

        EXAMPLE:

             >>> pp.to_pickle(net, "example.json")

    """
    json_string = "{"
    for k in sorted(net.keys()):
        if k[0] == "_":
            continue
        if isinstance(net[k], pd.DataFrame):
            if len(net[k]) == 0:  # do not bother saving empty data frames
                continue
            json_string += '"%s":%s,' % (k, net[k].to_json(orient="columns"))
        elif isinstance(net[k], numpy.ndarray):
            json_string += k + ":" + json.dumps(net[k].tolist()) + ","
        elif isinstance(net[k], dict):
            json_string += '"%s":%s,' % (k, json.dumps(net[k]))
        elif isinstance(net[k], bool):
            json_string += '"%s":%s,' % (k, "true" if net[k] else "false")
        elif isinstance(net[k], str):
            json_string += '"%s":"%s",' % (k, net[k])
        elif isinstance(net[k], numbers.Number):
            json_string += '"%s":%s,' % (k, net[k])
        elif net[k] is None:
            json_string += '"%s":null,' % k
        else:
            raise UserWarning("could not detect type of %s" % k)
    with open(filename, "w") as text_file:
        text_file.write(json_string[:-1] + "}\n")


def from_pickle(filename, convert=True):
    """
    Load a pandapower format Network from pickle file

    INPUT:
        **filename** (string) - The absolute or relative path to the input file.

    OUTPUT:
        **net** (dict) - The pandapower format network

    EXAMPLE:

        >>> net1 = pp.from_pickle(os.path.join("C:", "example_folder", "example1.p")) #absolute path
        >>> net2 = pp.from_pickle("example2.p") #relative path

    """
    if not os.path.isfile(filename):
        raise UserWarning("File %s does not exist!!" % filename)
    with open(filename, "rb") as f:
        if sys.version_info >= (3,0):
            net = pickle.load(f, encoding='latin1')  # with encoding in python 3
        else:
            net = pickle.load(f)  # without encoding in python 2
    net = pandapowerNet(net)
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
    xls = pd.ExcelFile(filename).parse(sheetname=None)
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
#    net.line.geodata.coords.
    if convert:
        convert_format(net)
    return net


def from_json(filename, convert=True):
    """
    Load a pandapower network from a JSON file.
    The index of the returned network is not necessarily in the same order as the original network.
    Index columns of all pandas DataFrames are sorted in ascending order.

    INPUT:
        **filename** (string) - The absolute or relative path to the input file.

    OUTPUT:
        **convert** (bool) - use the convert format function to

        **net** (dict) - The pandapower format network

    EXAMPLE:

        >>> net = pp.from_json("example.json")

    """
    with open(filename) as data_file:
        data = json.load(data_file)
        net = create_empty_network(name=data["name"], f_hz=data["f_hz"])

        # checks if field exists in empty network and if yes, matches data type
        def check_equal_type(name):
            if name in net:
                if isinstance(net[name], type(data[name])):
                    return True
                elif isinstance(net[name], pd.DataFrame) and isinstance(data[name], dict):
                    return True
                else:
                    return False
            return True

        for k in sorted(data.keys()):
            if not check_equal_type(k):
                raise UserWarning("Different data type for existing pandapower field")
            if isinstance(data[k], dict):
                if isinstance(net[k], pd.DataFrame):
                    columns = net[k].columns
                    net[k] = pd.DataFrame.from_dict(data[k], orient="columns")
                    net[k].set_index(net[k].index.astype(numpy.int64), inplace=True)
                    net[k] = net[k][columns]
                else:
                    net[k] = data[k]
            else:
                net[k] = data[k]
        if convert:
            convert_format(net)
        return net
    return None
