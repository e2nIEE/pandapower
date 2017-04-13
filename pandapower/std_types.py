# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import pandas as pd


def create_std_type(net, data, name, element="line", overwrite=True):
    """
    Creates type data in the type database. The parameters that are used for
    the loadflow have to be at least contained in data. These parameters are:
        - c_nf_per_km, r_ohm_per_km, x_ohm_per_km and max_i_ka (for lines)
        - sn_kva, vn_hv_kv, vn_lv_kv, vsc_percent, vscr_percent, pfe_kw, i0_percent, shift_degree* (for transformers)
        - sn_hv_kva, sn_mv_kva, sn_lv_kva, vn_hv_kv, vn_mv_kv, vn_lv_kv, vsc_hv_percent, vsc_mv_percent, vsc_lv_percent, vscr_hv_percent, vscr_mv_percent, vscr_lv_percent, pfe_kw, i0_percent, shift_mv_degree*, shift_lv_degree* (for 3-winding-transformers)
    additional parameters can be added and later loaded into pandapower with the function
    "parameter_from_std_type".

    \* only considered in loadflow if calculate_voltage_angles = True

    The standard type is saved into the pandapower library of the given network by default.

    INPUT:
        **net** - The pandapower network

        **data** - dictionary of standard type parameters

        **name** - name of the standard type as string

        **element** - "line", "trafo" or "trafo3w"

    EXAMPLE:

    >>> line_data = {"c_nf_per_km": 0, "r_ohm_per_km": 0.642, "x_ohm_per_km": 0.083, "max_i_ka": 0.142, "type": "cs", "q_mm2": 50}
    >>> pandapower.create_std_type(net, line_data, "NAYY 4×50 SE", element='line')
    """

    if type(data) != dict:
        raise UserWarning("type data has to be given as a dictionary of parameters")
    if element == "line":
        required = ["c_nf_per_km", "r_ohm_per_km", "x_ohm_per_km", "max_i_ka"]
    elif element == "trafo":
        required = ["sn_kva", "vn_hv_kv", "vn_lv_kv", "vsc_percent", "vscr_percent",
                    "pfe_kw", "i0_percent", "shift_degree"]
    elif element == "trafo3w":
        required = ["sn_hv_kva", "sn_mv_kva", "sn_lv_kva", "vn_hv_kv", "vn_mv_kv", "vn_lv_kv", "vsc_hv_percent",
                    "vsc_mv_percent", "vsc_lv_percent", "vscr_hv_percent", "vscr_mv_percent", "vscr_lv_percent",
                    "pfe_kw", "i0_percent", "shift_mv_degree", "shift_lv_degree"]
    else:
        raise ValueError("Unkown element type %s" % element)
    for par in required:
        if not par in data:
            raise UserWarning("%s is required as %s type parameter" % (par, element))
    library = net.std_types[element]
    if overwrite or not (name in library):
        library.update({name: data})


def create_std_types(net, data, element="line", overwrite=True):
    """
    Creates multiple standard types in the type database.

    INPUT:
        **net** - The pandapower network

        **data** - dictionary of standard type parameter sets

        **element** - "line", "trafo" or "trafo3w"

    EXAMPLE:

    >>> linetypes = {"typ1": {"r_ohm_per_km": 0.01, "x_ohm_per_km": 0.02, "c_nf_per_km": 10, "max_i_ka": 0.4, "type": "cs"},
    >>>              "typ2": {"r_ohm_per_km": 0.015, "x_ohm_per_km": 0.01, "c_nf_per_km": 30, "max_i_ka": 0.3, "type": "cs"}}
    >>> pp.create_std_types(net, data=linetypes, element="line")

    """
    for name, typdata in data.items():
        create_std_type(net, data=typdata, name=name, element=element, overwrite=overwrite)


def copy_std_types(to_net, from_net, element="line", overwrite=True):
    """
    Transfers all standard types of one network to another.

    INPUT:

        **to_net** - The pandapower network to which the standard types are copied

        **from_net** - The pandapower network from which the standard types are taken

        **element** - "line" or "trafo"

        **overwrite** - if True, overwrites standard types which already exist in to_net

    """
    for name, typdata in from_net.std_types[element].items():
        create_std_type(to_net, typdata, name, element=element, overwrite=overwrite)


def load_std_type(net, name, element="line"):
    """
    Loads linetype data from the linetypes data base. Issues a warning if
    linetype is unknown.

    INPUT:
        **net** - The pandapower network

        **name** - name of the standard type as string

        **element** - "line" or "trafo"

    OUTPUT:
        **typedata** - dictionary containing type data
    """
    library = net.std_types[element]
    if name in library:
        return library[name]
    else:
        raise UserWarning("Unknown standard %s type %s" % (element, name))


def std_type_exists(net, name, element="line"):
    """
    Checks if a standard type exists.

    INPUT:
        **net** - pandapower Network

        **name** - name of the standard type as string

        **element** - type of element ("line" or "trafo")

    OUTPUT:
        **exists** - True if standard type exists, False otherwise
    """
    library = net.std_types[element]
    return name in library


def delete_std_type(net, name, element="line"):
    """
    Deletes standard type parameters from database.

    INPUT:
        **net** - pandapower Network

        **name** - name of the standard type as string

        **element** - type of element ("line" or "trafo")

    """
    library = net.std_types[element]
    if name in library:
        del library[name]
    else:
        raise UserWarning("Unknown standard %s type %s" % (element, name))


def available_std_types(net, element="line"):
    """
    Returns all standard types available for this network as a table.

    INPUT:
        **net** - pandapower Network

        **element** - type of element ("line" or "trafo")

    OUTPUT:
        **typedata** - table of standard type parameters

    """
    return pd.DataFrame(net.std_types[element]).T


def parameter_from_std_type(net, parameter, element="line", fill=None):
    """
    Adds additional parameters, which are not included in the original pandapower datastructure
    but are available in the standard type database to the panadpower net.

    INPUT:
        **net** - pandapower network

        **parameter** - name of parameter as string

        **element** - type of element ("line" or "trafo")

        **fill** - fill-value that is assigned to all lines/trafos without
            a value for the parameter, either because the line/trafo has no type or because the
            type does not have a value for the parameter
    """
    if not parameter in net[element]:
        net[element][parameter] = fill
    for typ in net[element].std_type.unique():
        if pd.isnull(typ) or not std_type_exists(net, typ, element):
            continue
        typedata = load_std_type(net, name=typ, element=element)
        if parameter in typedata:
            util = net[element].loc[net[element].std_type == typ].index
            net[element].loc[util, parameter] = typedata[parameter]
    if fill is not None:
        net[element].loc[pd.isnull(net[element][parameter]).values, parameter] = fill


def change_std_type(net, eid, name, element="line"):
    """
    Changes the type of a given element in pandapower. Changes only parameter that are given
    for the type.

    INPUT:
        **net** - pandapower network

        **eid** - element index (either line or transformer index)

        **element** - type of element ("line" or "trafo")

        **name** - name of the new standard type

    """
    type_param = load_std_type(net, name, element)
    table = net[element]
    for column in table.columns:
        if column in type_param:
            table.at[eid, column] = type_param[column]
    table.at[eid, "std_type"] = name


def find_std_type_by_parameter(net, data, element="line", epsilon=0.):
    """
    Searches for a std_type that fits all values given in the data dictionary with the margin of
    epsilon.

    INPUT:
        **net** - pandapower network

        **data** - dictionary of standard type parameters

        **element** - type of element ("line" or "trafo")

        **epsilon** - tolerance margin for parameter comparison

    OUTPUT:
        **fitting_types** - list of fitting types or empty list
    """
    assert epsilon >= 0
    fitting_types = []
    for name, stp in net.std_types[element].items():
        for p, v in list(data.items()):
            if ((type(v) != float and stp[p] != v) or
                    (type(v) == float and abs(v - stp[p]) > epsilon)):
                break
        else:
            fitting_types.append(name)
    return fitting_types


def add_basic_std_types(net):
    if "std_types" not in net:
        net.std_types = {"line": {}, "trafo": {}, "trafo3w": {}}

    linetypes = {
        # Cables, all from S.744, Heuck: Elektrische Energieversorgung - Vierweg+Teubner 2013
        # another recommendable references for MV cables is Werth: Netzberechnung
        # mit Erzeugungsporfilen

        # Low Voltage
        "NAYY 4x50 SE":
        {"c_nf_per_km": 210,
         "r_ohm_per_km": 0.642,
         "x_ohm_per_km": 0.083,
         "max_i_ka": 0.142,
         "type": "cs",
         "q_mm2": 50},
        "NAYY 4x120 SE":
        {"c_nf_per_km": 264,
         "r_ohm_per_km": 0.225,
         "x_ohm_per_km": 0.080,
         "max_i_ka": 0.242,
         "type": "cs",
         "q_mm2": 120},
        "NAYY 4x150 SE":
        {"c_nf_per_km": 261,
         "r_ohm_per_km": 0.208,
         "x_ohm_per_km": 0.080,
         "max_i_ka": 0.270,
         "type": "cs",
         "q_mm2": 150},

        # Medium Voltage
        "NA2XS2Y 1x95 RM/25 12/20 kV":
        {"c_nf_per_km": 216,
            "r_ohm_per_km": 0.313,
            "x_ohm_per_km": 0.132,
            "max_i_ka": 0.252,
            "type": "cs",
            "q_mm2": 95},
        "NA2XS2Y 1x185 RM/25 12/20 kV":
        {"c_nf_per_km": 273,
            "r_ohm_per_km": 0.161,
            "x_ohm_per_km": 0.117,
            "max_i_ka": 0.362,
            "type": "cs",
            "q_mm2": 185},
        "NA2XS2Y 1x240 RM/25 12/20 kV":
        {"c_nf_per_km": 304,
            "r_ohm_per_km": 0.122,
            "x_ohm_per_km": 0.112,
            "max_i_ka": 0.421,
            "type": "cs",
            "q_mm2": 240},

        # High Voltage
        "N2XS(FL)2Y 1x120 RM/35 64/110 kV":
        {"c_nf_per_km": 112,
            "r_ohm_per_km": 0.153,
            "x_ohm_per_km": 0.166,
            "max_i_ka": 0.366,
            "type": "cs",
            "q_mm2": 120},
        "N2XS(FL)2Y 1x185 RM/35 64/110 kV":
        {"c_nf_per_km": 125,
            "r_ohm_per_km": 0.099,
            "x_ohm_per_km": 0.156,
            "max_i_ka": 0.457,
            "type": "cs",
            "q_mm2": 185},
        "N2XS(FL)2Y 1x240 RM/35 64/110 kV":
        {"c_nf_per_km": 135,
            "r_ohm_per_km": 0.075,
            "x_ohm_per_km": 0.149,
            "max_i_ka": 0.526,
            "type": "cs",
            "q_mm2": 240},
        "N2XS(FL)2Y 1x300 RM/35 64/110 kV":
        {"c_nf_per_km": 144,
            "r_ohm_per_km": 0.060,
            "x_ohm_per_km": 0.144,
            "max_i_ka": 0.588,
            "type": "cs",
            "q_mm2": 300},

        # Overhead Lines, all from S.742f, Heuck: Elektrische Energieversorgung -
        # Vierweg+Teubner 2013

        # Low Voltage
        "15-AL1/3-ST1A 0.4":
        {"c_nf_per_km": 11,
            "r_ohm_per_km": 1.8769,
            "x_ohm_per_km": 0.35,
            "max_i_ka": 0.105,
            "type": "ol",
            "q_mm2": 16},
        "24-AL1/4-ST1A 0.4":
        {"c_nf_per_km": 11.25,
            "r_ohm_per_km": 1.2012,
            "x_ohm_per_km": 0.335,
            "max_i_ka": 0.140,
            "type": "ol",
            "q_mm2": 24},
        "48-AL1/8-ST1A 0.4":
        {"c_nf_per_km": 12.2,
            "r_ohm_per_km": 0.5939,
            "x_ohm_per_km": 0.3,
            "max_i_ka": .210,
            "type": "ol",
            "q_mm2": 48},
        "94-AL1/15-ST1A 0.4":
        {"c_nf_per_km": 13.2,
            "r_ohm_per_km": 0.3060,
            "x_ohm_per_km": 0.29,
            "max_i_ka": 0.350,
            "type": "ol",
            "q_mm2": 94},

        # Medium Voltage
        "48-AL1/8-ST1A 10.0":
        {"c_nf_per_km": 10.1,
            "r_ohm_per_km": 0.5939,
            "x_ohm_per_km": 0.35,
            "max_i_ka": 0.210,
            "type": "ol",
            "q_mm2": 48},
        "94-AL1/15-ST1A 10.0":
        {"c_nf_per_km": 10.75,
            "r_ohm_per_km": 0.3060,
            "x_ohm_per_km": 0.33,
            "max_i_ka": 0.350,
            "type": "ol",
            "q_mm2": 94},
        "149-AL1/24-ST1A 10.0":
        {"c_nf_per_km": 11.25,
            "r_ohm_per_km": 0.1940,
            "x_ohm_per_km": 0.315,
            "max_i_ka": 0.470,
            "type": "ol",
            "q_mm2": 149},
        "48-AL1/8-ST1A 20.0":
        {"c_nf_per_km": 9.5,
         "r_ohm_per_km": 0.5939,
         "x_ohm_per_km": 0.372,
         "max_i_ka": 0.210,
         "type": "ol",
         "q_mm2": 48},
        "94-AL1/15-ST1A 20.0":
        {"c_nf_per_km": 10,
         "r_ohm_per_km": 0.3060,
         "x_ohm_per_km": 0.35,
         "max_i_ka": 0.350,
         "type": "ol",
         "q_mm2": 94},
        "149-AL1/24-ST1A 20.0":
        {"c_nf_per_km": 10.5,
         "r_ohm_per_km": 0.1940,
         "x_ohm_per_km": 0.337,
         "max_i_ka": 0.470,
         "type": "ol",
         "q_mm2": 149},
        "184-AL1/30-ST1A 20.0":
        {"c_nf_per_km": 10.75,
         "r_ohm_per_km": 0.1571,
         "x_ohm_per_km": 0.33,
         "max_i_ka": 0.535,
         "type": "ol",
         "q_mm2": 184},
        "243-AL1/39-ST1A 20.0":
        {"c_nf_per_km": 11,
         "r_ohm_per_km": 0.1188,
         "x_ohm_per_km": 0.32,
         "max_i_ka": 0.645,
         "type": "ol",
         "q_mm2": 243},

        # High Voltage
        "149-AL1/24-ST1A 110.0":
        {"c_nf_per_km": 8.75,
         "r_ohm_per_km": 0.1940,
         "x_ohm_per_km": 0.41,
         "max_i_ka": 0.470,
         "type": "ol",
         "q_mm2": 149},
        "184-AL1/30-ST1A 110.0":
        {"c_nf_per_km": 8.8,
         "r_ohm_per_km": 0.1571,
         "x_ohm_per_km": 0.4,
         "max_i_ka": 0.535,
         "type": "ol",
         "q_mm2": 184},

        "243-AL1/39-ST1A 110.0":
        {"c_nf_per_km": 9,
         "r_ohm_per_km": 0.1188,
         "x_ohm_per_km": 0.39,
         "max_i_ka": 0.645,
         "type": "ol",
         "q_mm2": 243},

        "305-AL1/39-ST1A 110.0":
        {"c_nf_per_km": 9.2,
         "r_ohm_per_km": 0.0949,
         "x_ohm_per_km": 0.38,
         "max_i_ka": 0.74,
         "type": "ol",
         "q_mm2": 305},

        # Transmission System
        # The following values of c and x are depend on the geometries of the  overhead line
        # Here it is assumed that for x the 220kV line uses twin conductors and the 380kV line uses
        # quad bundle conductor. The c values are estimated.
        "490-AL1/64-ST1A 220.0":
        {"c_nf_per_km": 10,
         "r_ohm_per_km": 0.059,
         "x_ohm_per_km": 0.285,
         "max_i_ka": 0.96,
         "type": "ol",
         "q_mm2": 490},

        "490-AL1/64-ST1A 380.0":
        {"c_nf_per_km": 11,
         "r_ohm_per_km": 0.059,
         "x_ohm_per_km": 0.253,
         "max_i_ka": 0.96,
         "type": "ol",
         "q_mm2": 490}
    }
    create_std_types(net, data=linetypes, element="line")

    trafotypes = {
        # generic HV/MV transformer
        # derived from Oswald - Transformatoren - Vorlesungsskript Elektrische Energieversorgung I
        # another recommendable references for distribution transformers is Werth:
        # Netzberechnung mit Erzeugungsprofilen
        "160 MVA 380/110 kV":
        {"i0_percent": 0.06,
         "pfe_kw": 60,
         "vscr_percent": 0.25,
         "sn_kva": 16e4,
         "vn_lv_kv": 110.0,
         "vn_hv_kv": 380.0,
         "vsc_percent": 12.2,
         "shift_degree": 0,
         "vector_group": "Yy0",
         "tp_side": "hv",
         "tp_mid": 0,
         "tp_min": -9,
         "tp_max": 9,
         "tp_st_degree": 0,
         "tp_st_percent": 1.5},

        "100 MVA 220/110 kV":
        {"i0_percent": 0.06,
         "pfe_kw": 55,
         "vscr_percent": 0.26,
         "sn_kva": 1e5,
         "vn_lv_kv": 110.0,
         "vn_hv_kv": 220.0,
         "vsc_percent": 12.0,
         "shift_degree": 0,
         "vector_group": "Yy0",
         "tp_side": "hv",
         "tp_mid": 0,
         "tp_min": -9,
         "tp_max": 9,
         "tp_st_degree": 0,
         "tp_st_percent": 1.5},

        # compare to IFT Ingenieurbüro data and Schlabbach book
        "63 MVA 110/20 kV":
        {"i0_percent": 0.086,
         "pfe_kw": 33,
         "vscr_percent": 0.322,
         "sn_kva": 63000,
         "vn_lv_kv": 20.0,
         "vn_hv_kv": 110.0,
         "vsc_percent": 11.2,
         "shift_degree": 150,
         "vector_group": "YNd5",
         "tp_side": "hv",
         "tp_mid": 0,
         "tp_min": -9,
         "tp_max": 9,
         "tp_st_degree": 0,
         "tp_st_percent": 1.5},

        "40 MVA 110/20 kV":
        {"i0_percent": 0.08,
         "pfe_kw": 31,
         "vscr_percent": 0.302,
         "sn_kva": 40000,
         "vn_lv_kv": 20.0,
         "vn_hv_kv": 110.0,
         "vsc_percent": 11.2,
         "shift_degree": 150,
         "vector_group": "YNd5",
         "tp_side": "hv",
         "tp_mid": 0,
         "tp_min": -9,
         "tp_max": 9,
         "tp_st_degree": 0,
         "tp_st_percent": 1.5},

        "25 MVA 110/20 kV":
        {"i0_percent": 0.071,
         "pfe_kw": 29,
         "vscr_percent": 0.282,
         "sn_kva": 25000,
         "vn_lv_kv": 20.0,
         "vn_hv_kv": 110.0,
         "vsc_percent": 11.2,
         "shift_degree": 150,
         "vector_group": "YNd5",
         "tp_side": "hv",
         "tp_mid": 0,
         "tp_min": -9,
         "tp_max": 9,
         "tp_st_degree": 0,
         "tp_st_percent": 1.5},

        "63 MVA 110/10 kV":
        {"sn_kva": 63000,
         "vn_hv_kv": 110,
         "vn_lv_kv": 10,
         "vsc_percent": 10.04,
         "vscr_percent": 0.31,
         "pfe_kw": 31.51,
         "i0_percent": 0.078,
         "shift_degree": 150,
         "vector_group": "YNd5",
         "tp_side": "hv",
         "tp_mid": 0,
         "tp_min": -9,
         "tp_max": 9,
         "tp_st_degree": 0,
         "tp_st_percent": 1.5},

        "40 MVA 110/10 kV":
        {"sn_kva": 40000,
         "vn_hv_kv": 110,
         "vn_lv_kv": 10,
         "vsc_percent": 10.04,
         "vscr_percent": 0.295,
         "pfe_kw": 30.45,
         "i0_percent": 0.076,
         "shift_degree": 150,
         "vector_group": "YNd5",
         "tp_side": "hv",
         "tp_mid": 0,
         "tp_min": -9,
         "tp_max": 9,
         "tp_st_degree": 0,
         "tp_st_percent": 1.5},

        "25 MVA 110/10 kV":
        {"sn_kva": 25000,
         "vn_hv_kv": 110,
         "vn_lv_kv": 10,
         "vsc_percent": 10.04,
         "vscr_percent": 0.276,
         "pfe_kw": 28.51,
         "i0_percent": 0.073,
         "shift_degree": 150,
         "vector_group": "YNd5",
         "tp_side": "hv",
         "tp_mid": 0,
         "tp_min": -9,
         "tp_max": 9,
         "tp_st_degree": 0,
         "tp_st_percent": 1.5},

        # Tafo20/0.4
        # 0.25 MVA 20/0.4 kV 0.45 Trafo Union
        "0.25 MVA 20/0.4 kV":
        {"sn_kva": 250,
         "vn_hv_kv": 20,
         "vn_lv_kv": 0.4,
         "vsc_percent": 6,
         "vscr_percent": 1.44,
         "pfe_kw": 0.8,
         "i0_percent": 0.32,
         "shift_degree": 150,
         "vector_group": "Yzn5",
         "tp_side": "hv",
         "tp_mid": 0,
         "tp_min": -2,
         "tp_max": 2,
         "tp_st_degree": 0,
         "tp_st_percent": 2.5},

        # 0.4 MVA 20/0.4 kV Trafo Union
        "0.4 MVA 20/0.4 kV":
        {"sn_kva": 400, "vn_hv_kv": 20, "vn_lv_kv": 0.4,
         "vsc_percent": 6,
         "vscr_percent": 1.425,
         "pfe_kw": 1.35,
         "i0_percent": 0.3375,
         "shift_degree": 150,
         "vector_group": "Dyn5",
         "tp_side": "hv",
         "tp_mid": 0,
         "tp_min": -2,
         "tp_max": 2,
         "tp_st_degree": 0,
         "tp_st_percent": 2.5},

        # 0.63 MVA 20/0.4 kV Trafo Union
        "0.63 MVA 20/0.4 kV":
        {"sn_kva": 630,
         "vn_hv_kv": 20,
         "vn_lv_kv": 0.4,
         "vsc_percent": 6,
         "vscr_percent": 1.206,
         "pfe_kw": 1.65,
         "i0_percent": 0.2619,
         "shift_degree": 150,
         "vector_group": "Dyn5",
         "tp_side": "hv",
         "tp_mid": 0,
         "tp_min": -2,
         "tp_max": 2,
         "tp_st_degree": 0,
         "tp_st_percent": 2.5},

        # Tafo10/0.4:
        # 0.25 MVA 10/0.4 kV 0.4 Trafo Union wnr
        "0.25 MVA 10/0.4 kV":
        {"sn_kva": 250,
         "vn_hv_kv": 10,
         "vn_lv_kv": 0.4,
         "vsc_percent": 4,
         "vscr_percent": 1.2,
         "pfe_kw": 0.6,
         "i0_percent": 0.24,
         "shift_degree": 150,
            "vector_group": "Dyn5",
         "tp_side": "hv",
         "tp_mid": 0,
         "tp_min": -2,
         "tp_max": 2,
         "tp_st_degree": 0,
         "tp_st_percent": 2.5},

        # 0.4 MVA 10/0.4 kV Trafo Union wnr
        "0.4 MVA 10/0.4 kV":
        {"sn_kva": 400,
         "vn_hv_kv": 10,
         "vn_lv_kv": 0.4,
         "vsc_percent": 4,
         "vscr_percent": 1.325,
         "pfe_kw": 0.95,
         "i0_percent": 0.2375,
         "shift_degree": 150,
         "vector_group": "Dyn5",
         "tp_side": "hv",
         "tp_mid": 0,
         "tp_min": -2,
         "tp_max": 2,
         "tp_st_degree": 0,
         "tp_st_percent": 2.5},

        # 0.63 MVA 10/0.4 kV Trafo Union wnr
        "0.63 MVA 10/0.4 kV":
        {"sn_kva": 630,
         "vn_hv_kv": 10,
         "vn_lv_kv": 0.4,
         "vsc_percent": 4,
         "vscr_percent": 1.0794,
         "pfe_kw": 1.18,
         "i0_percent": 0.1873,
         "shift_degree": 150,
         "vector_group": "Dyn5",
         "tp_side": "hv",
         "tp_mid": 0,
         "tp_min": -2,
         "tp_max": 2,
         "tp_st_degree": 0,
         "tp_st_percent": 2.5},
    }
    create_std_types(net, data=trafotypes, element="trafo")

    trafo3wtypes = {
        # trafo3w
        "63/25/38 MVA 110/20/10 kV":
        {"sn_hv_kva": 63000,
         "sn_mv_kva": 25000,
         "sn_lv_kva": 38000,
         "vn_hv_kv": 110,
         "vn_mv_kv": 20,
         "vn_lv_kv": 10,
         "vsc_hv_percent": 10.4,
         "vsc_mv_percent": 10.4,
         "vsc_lv_percent": 10.4,
         "vscr_hv_percent": 0.28,
         "vscr_mv_percent": 0.32,
         "vscr_lv_percent": 0.35,
         "pfe_kw": 35,
         "i0_percent": 0.89,
         "shift_mv_degree": 0,
         "shift_lv_degree": 0,
         "vector_group": "YN0yn0yn0",
         "tp_side": "hv",
         "tp_mid": 0,
         "tp_min": -10,
         "tp_max": 10,
         "tp_st_percent": 1.2},

        "63/25/38 MVA 110/10/10 kV":
        {"sn_hv_kva": 63000,
         "sn_mv_kva": 25000,
         "sn_lv_kva": 38000,
         "vn_hv_kv": 110,
         "vn_mv_kv": 10,
         "vn_lv_kv": 10,
         "vsc_hv_percent": 10.4,
         "vsc_mv_percent": 10.4,
         "vsc_lv_percent": 10.4,
         "vscr_hv_percent": 0.28,
         "vscr_mv_percent": 0.32,
         "vscr_lv_percent": 0.35,
         "pfe_kw": 35,
         "i0_percent": 0.89,
         "shift_mv_degree": 0,
         "shift_lv_degree": 0,
         "vector_group": "YN0yn0yn0",
         "tp_side": "hv",
         "tp_mid": 0,
         "tp_min": -10,
         "tp_max": 10,
         "tp_st_percent": 1.2}
    }
    create_std_types(net, data=trafo3wtypes, element="trafo3w")
    return linetypes, trafotypes, trafo3wtypes
