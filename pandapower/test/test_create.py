# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 14:04:18 2016

@author: JKupka
"""

import pandapower as pp
import inspect
import pandas
import numpy as np

default_dtypes = {"int64": 1,
                  "uint32": 1,
                  "float64": 1.5,
                  "bool": True,
                  "object": "teststr"
                  }

empty_network = pp.create_empty_network()


def get_initialized_args(fct, args):
    # Increment busIDs to avoid errors
    default_dtypes["uint32"] += 1
    default_dtypes["int64"] += 1

    # Get argument datatype and initialize key
    init_args = {}
    for arg in args:
        dt = arg_datatype_by_name(arg)
        init_args[arg] = default_dtypes.setdefault(dt, None)

    # Set nan for unavailiable datatypes
    nans = ["min_vm_pu", "max_vm_pu", "min_p_kw", "max_p_kw", "cost_per_kw", "cost_per_kvar",
            "max_loading_percent", "nr_busses", "svsc_max_mva", "svsc_min_mva", "rx_min", "rx_max"]
    for ia in init_args:
        if ia in nans:
            init_args[ia] = np.nan

    # Set net
    init_args["net"] = net

    # Set std_types
    if "std_type" in init_args:
        if fc.find("line") != -1:
            init_args["std_type"] = "NAYY 4x50 SE"
        elif fc.find("transformer3w") != -1:
            init_args["std_type"] = "63/25/38 MVA 110/20/10 kV"
        else:
            init_args["std_type"] = "0.4 MVA 10/0.4 kV"

    # Set element types
    if "et" in init_args:
        init_args["et"] = "b"

    return init_args


def arg_datatype_by_name(arg=""):
    net = empty_network

    tables = net.keys()
    for table in tables:
        if type(net[table]) == pandas.core.frame.DataFrame:
            columns = net[table].columns
            for column in columns:
                if column == arg:
                    return "%s" % net[table][column].dtype
    return None


def get_allowed_functions(obj, forbidden):
    fcts = dir(obj)

    allowed_fcts = []

    for fct in fcts:
        if fct.find("create") != -1 and fct not in forbidden:
            allowed_fcts.append(fct)

    return allowed_fcts
    

def check_datatypes(net):
    ref_net = empty_network
    
    tables = ["bus", "load", "sgen", "gen", "ext_grid", "line", "trafo", "trafo3w", "switch", 
              "shunt", "impedance", "ward", "xward"]
              
    for table in tables:
        for column in net[table].columns:
            if column in ref_net[table]:
                try:
                    assert net[table][column].dtype == ref_net[table][column].dtype
                except AssertionError:
                    err_msg=("%s.%s should be %s, is %s" % (table, column, \
                    ref_net[table][column].dtype, net[table][column].dtype))
                    print(err_msg)
                    #raise AssertionError(err_msg)
                    

# Get list of createfunctions
net = pp.create_empty_network()

# Create default busses
pp.create_busses(net, 20)

forbidden_fcts = ["create_empty_network"]
fcts = get_allowed_functions(pp.create, forbidden_fcts)
print(fcts)

for fc in fcts:
    fc_callable = getattr(pp.create, fc)
    args = inspect.signature(fc_callable).parameters
    initialized_args = get_initialized_args(fc, args)
    print("%s %s" % (fc, initialized_args))
    print("")
    print(fc)
    ret = fc_callable(**initialized_args)
    assert net
    print("Return: %s" % ret)
    print("################################")
    
check_datatypes(net)
    

