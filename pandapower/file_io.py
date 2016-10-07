# -*- coding: utf-8 -*-
"""
Created on Wed Jun 04 16:08:24 2014

@author: TDess
"""
import os
import pickle
import pandapower as pp
import pandas as pd

def to_hdf5(net, filename, complevel=1, complib="zlib", save_res=False):
    raise  Exception('to_hdf5 is deprecated. Use to_pickle instead')


def from_hdf5(filename):
    # Load HDF5 File
    raise  Exception('from_hdf5 is deprecated. If you need to open a hdf5 File you may go back in GIT. However, to save and load files, to_pickle and from_pickle should be used.')
    
def to_pickle(net, filename):
    """
    Saves a Pandapower Network with the pickle library.

    INPUT:
        **net** (dict) - The Pandapower format network

        **filename** (string) - The absolute or relative path to the input file.

    EXAMPLE:
    
        >>> pp.to_pickle(net, os.path.join("C:", "example_folder", "example1.p"))  # absolute path
        >>> pp.to_pickle(net, "example2.p")  # relative path

    """
    if not filename.endswith(".p"):
        raise Exception("Please use .p to save pandapower networks!")
    with open(filename, "wb") as f:
        pickle.dump(dict(net), f, protocol=2)

def to_excel(net, filename, include_empty_tables=False, include_results=True):
    """
    Saves a Pandapower Network to an excel file.

    INPUT:
        **net** (dict) - The Pandapower format network

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
        if type(table) != pd.DataFrame or item.startswith("_"):
            continue
        elif item.startswith("res"):
            if include_results and len(table) > 0:
                table.to_excel(writer, sheet_name=item)            
        elif len(table) > 0 or include_empty_tables:
            table.to_excel(writer, sheet_name=item)
    parameters = pd.DataFrame(index=["name", "f_hz", "version"], columns=["parameters"],
                              data=[net.name, net.f_hz, net.version])
    pd.DataFrame(net.std_types["line"]).T.to_excel(writer, sheet_name="line_std_types")
    pd.DataFrame(net.std_types["trafo"]).T.to_excel(writer, sheet_name="trafo_std_types")
    pd.DataFrame(net.std_types["trafo3w"]).T.to_excel(writer, sheet_name="trafo3w_std_types")
    parameters.to_excel(writer, sheet_name="parameters")
    writer.save()    

def from_pickle(filename):
    """
    Load a Pandapower format Network from pickle file

    INPUT:
        **filename** (string) - The absolute or relative path to the input file.

    RETURN:

        **net** (dict) - The pandapower format network
        
    EXAMPLE:
    
        >>> net1 = pp.from_pickle(os.path.join("C:", "example_folder", "example1.p")) #absolute path
        >>> net2 = pp.from_pickle("example2.p") #relative path

    """
    
    if not os.path.isfile(filename):
        raise UserWarning("File %s does not exist!!" % filename)
    with open(filename, "rb") as f:
        try:
            net = pickle.load(f)
        except:
            net = pickle.load(f,encoding='latin1')
    net = pp.PandapowerNet(net)
    return pp.convert_format(net)

def from_excel(filename):
    """
    Load a Pandapower network from an excel file

    INPUT:
        **filename** (string) - The absolute or relative path to the input file.

    RETURN:

        **net** (dict) - The pandapower format network
        
    EXAMPLE:
    
        >>> net1 = pp.from_excel(os.path.join("C:", "example_folder", "example1.xlsx")) #absolute path
        >>> net2 = pp.from_excel("example2.xlsx") #relative path

    """
    xls = pd.ExcelFile(filename).parse(sheetname=None)
    par = xls["parameters"]["parameters"]
    name = None if pd.isnull(par.at["name"]) else par.at["name"]
    net = pp.create_empty_network(name=name, f_hz=par.at["f_hz"])
    
    for item, table in xls.items():
        if item == "parameter":
            continue    
        elif item.endswith("std_types"):
            item = item.split("_")[0]
            for std_type, tab in table.iterrows():
                net.std_types[item][std_type] = dict(tab)
        else:
            net[item] = table
    return pp.convert_format(net)
    
if __name__ == '__main__':
    import pandapower.networks as nw
    net = nw.create_kerber_dorfnetz()
    filename = 'pp_test.xlsx'
    pp.runpp(net)
    to_excel(net, filename, include_empty_tables=True, include_results=False)
    net2 = from_excel(filename)
