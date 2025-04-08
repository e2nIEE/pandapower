# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pickle

import os
import sys
import json
from warnings import warn
import numpy
import pandas as pd
from packaging.version import Version


try:
    import xlsxwriter
    xlsxwriter_INSTALLED = True
except ImportError:
    xlsxwriter_INSTALLED = False
try:
    import openpyxl
    openpyxl_INSTALLED = True
except ImportError:
    openpyxl_INSTALLED = False

from pandapower._version import __version__ as pp_version
from pandapower.auxiliary import soft_dependency_error
from pandapower.auxiliary import pandapowerNet
from pandapower.std_types import basic_std_types
from pandapower.create import create_empty_network
from pandapower.convert_format import convert_format
from pandapower.io_utils import to_dict_with_coord_transform, to_dict_of_dfs, PPJSONEncoder, encrypt_string, \
    get_raw_data_from_pickle, transform_net_with_df_and_geo, check_net_version, from_dict_of_dfs, decrypt_string, \
    PPJSONDecoder

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def to_pickle(net, filename):
    """
    Saves a pandapower Network with the pickle library.

    INPUT:
        **net** (dict) - The pandapower format network

        **filename** (string) - The absolute or relative path to the output file or an writable
        file-like objectxs

    EXAMPLE:

        >>> from pandapower import to_pickle
        >>> to_pickle(net, os.path.join("C:", "example_folder", "example1.p"))  # absolute path
        >>> to_pickle(net, "example2.p")  # relative path

    """
    if hasattr(filename, 'write'):
        pickle.dump(dict(net), filename, protocol=2)
        return
    if not filename.endswith(".p"):
        raise Exception("Please use .p to save pandapower networks!")
    save_net = to_dict_with_coord_transform(net, ["bus_geodata"], ["line_geodata"])

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

        >>> from pandapower import to_excel
        >>> to_excel(net, os.path.join("C:", "example_folder", "example1.xlsx"))  # absolute path
        >>> to_excel(net, "example2.xlsx")  # relative path

    """
    if not xlsxwriter_INSTALLED:
        soft_dependency_error(str(sys._getframe().f_code.co_name)+"()", "xlsxwriter")
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    dict_net = to_dict_of_dfs(net, include_results=include_results,
                                       include_empty_tables=include_empty_tables)
    for item, table in dict_net.items():
        table.to_excel(writer, sheet_name=item)

    try:
        writer.save()
    except AttributeError:
        writer._save()


def to_json(net, filename=None, encryption_key=None, store_index_names=None):
    """
        Saves a pandapower Network in JSON format. The index columns of all pandas DataFrames will
        be saved in ascending order. net elements which name begins with "_" (internal elements)
        will not be saved. Std types will also not be saved.

        INPUT:
            **net** (dict) - The pandapower format network

            **filename** (string or file, None) - The absolute or relative path to the output file
            or a file-like object, if 'None' the function returns a json string

            **encrytion_key** (string, None) - If given, the pandapower network is stored as an
            encrypted json string

        EXAMPLE:
             >>> from pandapower.file_io import to_json
             >>> to_json(net, "example.json")

    """
    # --- store index names
    if store_index_names is not None:
        msg = "The input parameter 'store_index_names' of function 'to_json()' is deprecated."
        if Version(pp_version) < Version("2.15"):
            warn(msg)
        else:
            raise DeprecationWarning(msg)

    json_string = json.dumps(net, cls=PPJSONEncoder, indent=2)
    if encryption_key is not None:
        json_string = encrypt_string(json_string, encryption_key)

    if filename is None:
        return json_string

    if hasattr(filename, 'write'):
        filename.write(json_string)
    else:
        with open(filename, "w") as fp:
            fp.write(json_string)


def from_pickle(filename, convert=True):
    """
    Load a pandapower format Network from pickle file

    INPUT:
        **filename** (string or file) - The absolute or relative path to the input file or
        file-like object

        **convert** (bool, True) - If True, converts the format of the net loaded from pickle
        from the older version of pandapower to the newer version format

    OUTPUT:
        **net** (dict) - The pandapower format network

    EXAMPLE:

        >>> from pandapower import from_pickle
        >>> net1 = from_pickle(os.path.join("C:", "example_folder", "example1.p")) #absolute path
        >>> net2 = from_pickle("example2.p") #relative path

    """

    net = pandapowerNet(get_raw_data_from_pickle(filename))
    transform_net_with_df_and_geo(net, ["bus_geodata"], ["line_geodata"])

    if convert:
        convert_format(net)

        # compare pandapowerNet-format_version and package-version
        check_net_version(net)
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

        >>> from pandapower import from_excel
        >>> net1 = from_excel(os.path.join("C:", "example_folder", "example1.xlsx"))
        >>> net2 = from_excel("example2.xlsx") #relative path

    """

    if not os.path.isfile(filename):
        raise UserWarning("File %s does not exist!" % filename)
    if not openpyxl_INSTALLED:
        soft_dependency_error(str(sys._getframe().f_code.co_name)+"()", "openpyxl")
    xls = pd.read_excel(filename, sheet_name=None, index_col=0, engine="openpyxl")

    try:
        net = from_dict_of_dfs(xls)
    except:
        net = _from_excel_old(xls)
    if convert:
        convert_format(net)

        # compare pandapowerNet-format_version and package-version
        check_net_version(net)
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


def from_json(filename, convert=True, encryption_key=None, elements_to_deserialize=None,
              keep_serialized_elements=True, add_basic_std_types=False, replace_elements=None,
              empty_dict_like_object=None, ignore_unknown_objects=False):
    """
    Load a pandapower network from a JSON file.
    The index of the returned network is not necessarily in the same order as the original network.
    Index columns of all pandas DataFrames are sorted in ascending order.

    INPUT:
        **filename** (string or file) - The absolute or relative path to the input file or
        file-like object

        **convert** (bool, True) - If True, converts the format of the net loaded from json
        from the older version of pandapower to the newer version format

        **encrytion_key** (string, "") - If given, key to decrypt an encrypted pandapower network

        **elements_to_deserialize** (list, None) - Deserialize only certain pandapower elements.
        If None all elements are deserialized.

        **keep_serialized_elements** (bool, True) - Keep serialized elements if given.
        Default: Serialized elements are kept.

        **add_basic_std_types** (bool, False) - Add missing standard-types from pandapower standard
        type library.

        **replace_elements** (dict, None) - Keys are replaced by values found in json string.
        Both key and value are supposed to be strings.

        **empty_dict_like_object** (dict/pandapower.pandapowerNet/..., None) - If None,
        the output of pandapower.create_empty_network() is used as an empty element to be filled by
        the data of the json string. Give another dict-like object to start filling that alternative
        object with the json data.

        **ignore_unknown_objects** (bool, False) - If set to True, ignore any objects that cannot be
         deserialized instead of raising an error

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

    return from_json_string(
        json_string,
        convert=convert,
        encryption_key=encryption_key,
        elements_to_deserialize=elements_to_deserialize,
        keep_serialized_elements=keep_serialized_elements,
        add_basic_std_types=add_basic_std_types,
        replace_elements=replace_elements,
        empty_dict_like_object=empty_dict_like_object,
        ignore_unknown_objects=ignore_unknown_objects)


def from_json_string(json_string, convert=False, encryption_key=None, elements_to_deserialize=None,
                     keep_serialized_elements=True, add_basic_std_types=False,
                     replace_elements=None, empty_dict_like_object=None, ignore_unknown_objects=False):
    """
    Load a pandapower network from a JSON string.
    The index of the returned network is not necessarily in the same order as the original network.
    Index columns of all pandas DataFrames are sorted in ascending order.

    INPUT:
        **json_string** (string) - The json string representation of the network

        **convert** (bool, False) - If True, converts the format of the net loaded from json_string
        from the older version of pandapower to the newer version format

        **encrytion_key** (string, "") - If given, key to decrypt an encrypted json_string

        **elements_to_deserialize** (list, None) - Deserialize only certain pandapower elements.
            If None all elements are deserialized.

        **keep_serialized_elements** (bool, True) - Keep serialized elements if given.
            Default: Serialized elements are kept.

        **add_basic_std_types** (bool, False) - Add missing standard-types from pandapower standard
        type library.

        **replace_elements** (dict, None) - Keys are replaced by values found in json string.
        Both key and value are supposed to be strings.

        **empty_dict_like_object** (dict/pandapower.pandapowerNet/..., None) - If None,
        the output of pandapower.create_empty_network() is used as an empty element to be filled by
        the data of the json string. Give another dict-like object to start filling that alternative
        object with the json data.

        **ignore_unknown_objects** (bool, False) - If set to True, ignore any objects that cannot be
         deserialized instead of raising an error

    OUTPUT:
        **net** (dict) - The pandapower format network

    EXAMPLE:

        >>> from pandapower import from_json_string
        >>> net = from_json_string(json_str)

    """
    if replace_elements is not None:
        for k, v in replace_elements.items():
            json_string = json_string.replace(k, v)

    if encryption_key is not None:
        json_string = decrypt_string(json_string, encryption_key)

    if elements_to_deserialize is None:
        net = json.loads(json_string, cls=PPJSONDecoder,
                         empty_dict_like_object=empty_dict_like_object,
                         ignore_unknown_objects=ignore_unknown_objects)
    else:
        net = json.loads(json_string, cls=PPJSONDecoder, deserialize_pandas=False,
                         empty_dict_like_object=empty_dict_like_object,
                         ignore_unknown_objects=ignore_unknown_objects)
        net_dummy = create_empty_network()
        if ('version' not in net.keys()) | (Version(net.version) < Version('2.1.0')):
            raise UserWarning('table selection is only possible for nets above version 2.0.1. '
                              'Convert and save your net first.')
        if keep_serialized_elements:
            for key in elements_to_deserialize:
                net[key] = json.loads(net[key], cls=PPJSONDecoder)
        else:
            if (('version' not in net.keys()) or (net['version'] != net_dummy.version)) and \
                    not convert:
                raise UserWarning(
                    'The version of your net %s you are trying to load differs from the actual '
                    'pandapower version %s. Before you can load only distinct tables, convert '
                    'and save your net first or set convert to True!'
                    % (net['version'], net_dummy.version))
            for key in net.keys():
                if key in elements_to_deserialize:
                    net[key] = json.loads(net[key], cls=PPJSONDecoder)
                elif not isinstance(net[key], str):
                    continue
                elif 'pandas' in net[key]:
                    net[key] = net_dummy[key]

    # this can be removed in the future
    # now net is saved with "_module", "_class", "_object"..., so json.load already returns
    # pandapowerNet. Older files don't have it yet, and are loaded as dict.
    # After some time, this part can be removed.
    if isinstance(net, dict) and "bus" in net and not isinstance(net, pandapowerNet):
        warn("This net is saved in older format, which will not be supported in future.\r\n"
             "Please resave your grid using the current pandapower version.",
             DeprecationWarning)
        net = from_json_dict(net)

    if convert:
        convert_format(net, elements_to_deserialize=elements_to_deserialize)

        # compare pandapowerNet-format_version and package-version
        check_net_version(net)
    if add_basic_std_types:
        # get std-types and add only new keys ones
        for key, std_types in basic_std_types().items():
            net.std_types[key] = dict(std_types, **net.std_types[key])

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

        >>> from pandapower import from_json_dict
        >>> net = from_json_dict(json.loads(json_str))

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
