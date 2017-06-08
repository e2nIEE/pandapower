# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import pandas as pd
from numpy import nan, isnan, arange, dtype, zeros

from pandapower.auxiliary import pandapowerNet, get_free_id, _preserve_dtypes
from pandapower.results import reset_results
from pandapower.std_types import add_basic_std_types, load_std_type
from pandapower import __version__

def create_empty_network(name=None, f_hz=50., sn_kva=1e3):
    """
    This function initializes the pandapower datastructure.

    OPTIONAL:
        **f_hz** (float, 50.) - power system frequency in hertz

        **name** (string, None) - name for the network

    OUTPUT:
        **net** (attrdict) - PANDAPOWER attrdict with empty tables:

    EXAMPLE:
        net = create_empty_network()

    """
    net = pandapowerNet({
        # structure data
        "bus": [('name', dtype(object)),
                ('vn_kv', 'f8'),
                ('type', dtype(object)),
                ('zone', dtype(object)),
                ('in_service', 'bool'), ],
        "load": [("name", dtype(object)),
                 ("bus", "u4"),
                 ("p_kw", "f8"),
                 ("q_kvar", "f8"),
                 ("const_z_percent", "f8"),
                 ("const_i_percent", "f8"),
                 ("sn_kva", "f8"),
                 ("scaling", "f8"),
                 ("in_service", 'bool'),
                 ("type", dtype(object))],
        "sgen": [("name", dtype(object)),
                 ("bus", "i8"),
                 ("p_kw", "f8"),
                 ("q_kvar", "f8"),
                 ("sn_kva", "f8"),
                 ("scaling", "f8"),
                 ("in_service", 'bool'),
                 ("type", dtype(object))],
        "gen": [("name", dtype(object)),
                ("bus", "u4"),
                ("p_kw", "f8"),
                ("vm_pu", "f8"),
                ("sn_kva", "f8"),
                ("min_q_kvar", "f8"),
                ("max_q_kvar", "f8"),
                ("scaling", "f8"),
                ("in_service", 'bool'),
                ("type", dtype(object))],
        "switch": [("bus", "i8"),
                   ("element", "i8"),
                   ("et", dtype(object)),
                   ("type", dtype(object)),
                   ("closed", "bool"),
                   ("name", dtype(object))],
        "shunt": [("bus", "u4"),
                  ("name", dtype(object)),
                  ("q_kvar", "f8"),
                  ("p_kw", "f8"),
                  ("vn_kv", "f8"),
                  ("step", "u4"),
                  ("in_service", "bool")],
        "ext_grid": [("name", dtype(object)),
                     ("bus", "u4"),
                     ("vm_pu", "f8"),
                     ("va_degree", "f8"),
                     ("in_service", 'bool')],
        "line": [("name", dtype(object)),
                 ("std_type", dtype(object)),
                 ("from_bus", "u4"),
                 ("to_bus", "u4"),
                 ("length_km", "f8"),
                 ("r_ohm_per_km", "f8"),
                 ("x_ohm_per_km", "f8"),
                 ("c_nf_per_km", "f8"),
                 ("max_i_ka", "f8"),
                 ("df", "f8"),
                 ("parallel", "u4"),
                 ("type", dtype(object)),
                 ("in_service", 'bool')],
        "trafo": [("name", dtype(object)),
                  ("std_type", dtype(object)),
                  ("hv_bus", "u4"),
                  ("lv_bus", "u4"),
                  ("sn_kva", "f8"),
                  ("vn_hv_kv", "f8"),
                  ("vn_lv_kv", "f8"),
                  ("vsc_percent", "f8"),
                  ("vscr_percent", "f8"),
                  ("pfe_kw", "f8"),
                  ("i0_percent", "f8"),
                  ("shift_degree", "f8"),
                  ("tp_side", dtype(object)),
                  ("tp_mid", "i4"),
                  ("tp_min", "i4"),
                  ("tp_max", "i4"),
                  ("tp_st_percent", "f8"),
                  ("tp_st_degree", "f8"),
                  ("tp_pos", "i4"),
                  ("parallel", "u4"),
                  ("in_service", 'bool')],
        "trafo3w": [("name", dtype(object)),
                    ("std_type", dtype(object)),
                    ("hv_bus", "u4"),
                    ("mv_bus", "u4"),
                    ("lv_bus", "u4"),
                    ("sn_hv_kva", "u8"),
                    ("sn_mv_kva", "u8"),
                    ("sn_lv_kva", "u8"),
                    ("vn_hv_kv", "f8"),
                    ("vn_mv_kv", "f8"),
                    ("vn_lv_kv", "f8"),
                    ("vsc_hv_percent", "f8"),
                    ("vsc_mv_percent", "f8"),
                    ("vsc_lv_percent", "f8"),
                    ("vscr_hv_percent", "f8"),
                    ("vscr_mv_percent", "f8"),
                    ("vscr_lv_percent", "f8"),
                    ("pfe_kw", "f8"),
                    ("i0_percent", "f8"),
                    ("shift_mv_degree", "f8"),
                    ("shift_lv_degree", "f8"),
                    ("tp_side", dtype(object)),
                    ("tp_mid", "i4"),
                    ("tp_min", "i4"),
                    ("tp_max", "i4"),
                    ("tp_st_percent", "f8"),
                    ("tp_pos", "i4"),
                    ("in_service", 'bool')],
        "impedance": [("name", dtype(object)),
                      ("from_bus", "u4"),
                      ("to_bus", "u4"),
                      ("rft_pu", "f8"),
                      ("xft_pu", "f8"),
                      ("rtf_pu", "f8"),
                      ("xtf_pu", "f8"),
                      ("sn_kva", "f8"),
                      ("in_service", 'bool')],
        "dcline": [("name", dtype(object)),
                   ("from_bus", "u4"),
                   ("to_bus", "u4"),
                   ("p_kw", "f8"),
                   ("loss_percent", 'f8'),
                   ("loss_kw", 'f8'),
                   ("vm_from_pu", "f8"),
                   ("vm_to_pu", "f8"),
                   ("max_p_kw", "f8"),
                   ("min_q_from_kvar", "f8"),
                   ("min_q_to_kvar", "f8"),
                   ("max_q_from_kvar", "f8"),
                   ("max_q_to_kvar", "f8"),
                   ("in_service", 'bool')],
        "ward": [("name", dtype(object)),
                 ("bus", "u4"),
                 ("ps_kw", "f8"),
                 ("qs_kvar", "f8"),
                 ("qz_kvar", "f8"),
                 ("pz_kw", "f8"),
                 ("in_service", "bool")],
        "xward": [("name", dtype(object)),
                  ("bus", "u4"),
                  ("ps_kw", "f8"),
                  ("qs_kvar", "f8"),
                  ("qz_kvar", "f8"),
                  ("pz_kw", "f8"),
                  ("r_ohm", "f8"),
                  ("x_ohm", "f8"),
                  ("vm_pu", "f8"),
                  ("in_service", "bool")],
        "measurement": [("name", dtype(object)),
                        ("type", dtype(object)),
                        ("element_type", dtype(object)),
                        ("value", "f8"),
                        ("std_dev", "f8"),
                        ("bus", "u4"),
                        ("element", dtype(object))],
        "piecewise_linear_cost": [("type", dtype(object)),
                                  ("element", dtype(object)),
                                  ("element_type", dtype(object)),
                                  ("p", dtype(object)),
                                  ("f", dtype(object))],
        "polynomial_cost": [("type", dtype(object)),
                            ("element", dtype(object)),
                            ("element_type", dtype(object)),
                            ("c", dtype(object))],
        # geodata
        "line_geodata": [("coords", dtype(object))],
        "bus_geodata": [("x", "f8"), ("y", "f8")],


        # result tables
        "_empty_res_bus": [("vm_pu", "f8"),
                           ("va_degree", "f8"),
                           ("p_kw", "f8"),
                           ("q_kvar", "f8")],
        "_empty_res_ext_grid": [("p_kw", "f8"),
                                ("q_kvar", "f8")],
        "_empty_res_line": [("p_from_kw", "f8"),
                            ("q_from_kvar", "f8"),
                            ("p_to_kw", "f8"),
                            ("q_to_kvar", "f8"),
                            ("pl_kw", "f8"),
                            ("ql_kvar", "f8"),
                            ("i_from_ka", "f8"),
                            ("i_to_ka", "f8"),
                            ("i_ka", "f8"),
                            ("loading_percent", "f8")],
        "_empty_res_trafo": [("p_hv_kw", "f8"),
                             ("q_hv_kvar", "f8"),
                             ("p_lv_kw", "f8"),
                             ("q_lv_kvar", "f8"),
                             ("pl_kw", "f8"),
                             ("ql_kvar", "f8"),
                             ("i_hv_ka", "f8"),
                             ("i_lv_ka", "f8"),
                             ("loading_percent", "f8")],
        "_empty_res_trafo3w": [("p_hv_kw", "f8"),
                               ("q_hv_kvar", "f8"),
                               ("p_mv_kw", "f8"),
                               ("q_mv_kvar", "f8"),
                               ("p_lv_kw", "f8"),
                               ("q_lv_kvar", "f8"),
                               ("pl_kw", "f8"),
                               ("ql_kvar", "f8"),
                               ("i_hv_ka", "f8"),
                               ("i_mv_ka", "f8"),
                               ("i_lv_ka", "f8"),
                               ("loading_percent", "f8")],
        "_empty_res_load": [("p_kw", "f8"),
                            ("q_kvar", "f8")],
        "_empty_res_sgen": [("p_kw", "f8"),
                            ("q_kvar", "f8")],
        "_empty_res_gen": [("p_kw", "f8"),
                           ("q_kvar", "f8"),
                           ("va_degree", "f8"),
                           ("vm_pu", "f8")],
        "_empty_res_shunt": [("p_kw", "f8"),
                             ("q_kvar", "f8"),
                             ("vm_pu", "f8")],
        "_empty_res_impedance": [("p_from_kw", "f8"),
                                 ("q_from_kvar", "f8"),
                                 ("p_to_kw", "f8"),
                                 ("q_to_kvar", "f8"),
                                 ("pl_kw", "f8"),
                                 ("ql_kvar", "f8"),
                                 ("i_from_ka", "f8"),
                                 ("i_to_ka", "f8")],
        "_empty_res_dcline": [("p_from_kw", "f8"),
                              ("q_from_kvar", "f8"),
                              ("p_to_kw", "f8"),
                              ("q_to_kvar", "f8"),
                              ("pl_kw", "f8"),
                              ("vm_from_pu", "f8"),
                              ("va_from_degree", "f8"),
                              ("vm_to_pu", "f8"),
                              ("va_to_degree", "f8")],
        "_empty_res_ward": [("p_kw", "f8"),
                            ("q_kvar", "f8"),
                            ("vm_pu", "f8")],
        "_empty_res_xward": [("p_kw", "f8"),
                             ("q_kvar", "f8"),
                             ("vm_pu", "f8")],

        # internal
        "_ppc": None,
        "_is_elements": None,
        "_pd2ppc_lookups": {"bus": None,
                            "ext_grid": None,
                            "gen": None},
        "version": float(__version__[:3]),
        "converged": False,
        "name": name,
        "f_hz": f_hz,
        "sn_kva": sn_kva
    })
    for s in net:
        if isinstance(net[s], list):
            net[s] = pd.DataFrame(zeros(0, dtype=net[s]))
    add_basic_std_types(net)
    reset_results(net)
    return net


def create_bus(net, vn_kv, name=None, index=None, geodata=None, type="b",
               zone=None, in_service=True, max_vm_pu=nan,
               min_vm_pu=nan, **kwargs):
    """create_bus(net, vn_kv, name=None, index=None, geodata=None, type="b", \
                  zone=None, in_service=True, max_vm_pu=nan, min_vm_pu=nan)
    Adds one bus in table net["bus"].

    Busses are the nodes of the network that all other elements connect to.

    INPUT:
        **net** (pandapowerNet) - The pandapower network in which the element is created

    OPTIONAL:
        **name** (string, default None) - the name for this bus

        **index** (int, default None) - Force a specified ID if it is available. If None, the index one higher than the highest already existing index is selected.

        **vn_kv** (float) - The grid voltage level.

        **geodata** ((x,y)-tuple, default None) - coordinates used for plotting

        **type** (string, default "b") - Type of the bus. "n" - auxilary node,
        "b" - busbar, "m" - muff

        **zone** (string, None) - grid region

        **in_service** (boolean) - True for in_service or False for out of service

    OUTPUT:
        **index** (int) - The unique ID of the created element

    EXAMPLE:
        create_bus(net, name = "bus1")
    """
    if index and index in net["bus"].index:
        raise UserWarning("A bus with index %s already exists" % index)

    if index is None:
        index = get_free_id(net["bus"])

    # store dtypes
    dtypes = net.bus.dtypes

    net.bus.loc[index, ["name", "vn_kv", "type", "zone", "in_service"]] = \
        [name, vn_kv, type, zone, bool(in_service)]

    # and preserve dtypes
    _preserve_dtypes(net.bus, dtypes)

    if geodata:
        if len(geodata) != 2:
            raise UserWarning("geodata must be given as (x, y) tupel")
        net["bus_geodata"].loc[index, ["x", "y"]] = geodata

    if not isnan(min_vm_pu):
        if "min_vm_pu" not in net.bus.columns:
            net.bus.loc[:, "min_vm_pu"] = pd.Series()

        net.bus.loc[index, "min_vm_pu"] = float(min_vm_pu)

    if not isnan(max_vm_pu):
        if "max_vm_pu" not in net.bus.columns:
            net.bus.loc[:, "max_vm_pu"] = pd.Series()

        net.bus.loc[index, "max_vm_pu"] = float(max_vm_pu)

    return index


def create_buses(net, nr_buses, vn_kv, index=None, name=None, type="b", geodata=None,
                 zone=None, in_service=True, max_vm_pu=nan, min_vm_pu=nan):
    """create_buses(net, nr_buses, vn_kv, index=None, name=None, type="b", geodata=None, \
                    zone=None, in_service=True, max_vm_pu=nan, min_vm_pu=nan)
    Adds several buses in table net["bus"] at once.

    Busses are the nodal points of the network that all other elements connect to.

    Input:
        **net** (pandapowerNet) - The pandapower network in which the element is created

        **nr_buses** (int) - The number of buses that is created

    OPTIONAL:
        **name** (string, default None) - the name for this bus

        **index** (int, default None) - Force specified IDs if available. If None, the indeces higher than the highest already existing index are selected.

        **vn_kv** (float) - The grid voltage level.

        **geodata** ((x,y)-tuple, default None) - coordinates used for plotting

        **type** (string, default "b") - Type of the bus. "n" - auxilary node,
        "b" - busbar, "m" - muff

        **zone** (string, None) - grid region

        **in_service** (boolean) - True for in_service or False for out of service

    OUTPUT:
        **index** (int) - The unique indices ID of the created elements

    EXAMPLE:
        create_bus(net, name = "bus1")
    """
    if index:
        for idx in index:
            if idx in net.bus.index:
                raise UserWarning("A bus with index %s already exists" % index)
    else:
        bid = get_free_id(net["bus"])
        index = arange(bid, bid + nr_buses, 1)

    # TODO: not needed when concating anyways?
    # store dtypes
    # dtypes = net.bus.dtypes

    dd = pd.DataFrame(index=index, columns=net.bus.columns)
    dd["vn_kv"] = vn_kv
    dd["type"] = type
    dd["zone"] = zone
    dd["in_service"] = in_service
    dd["name"] = name
    net["bus"] = pd.concat([net["bus"], dd], axis=0).reindex_axis(net["bus"].columns, axis=1)

    # and preserve dtypes
    # _preserve_dtypes(net.bus, dtypes)

    if geodata:
        if len(geodata) != 2:
            raise UserWarning("geodata must be given as (x, y) tupel")
        net["bus_geodata"].loc[bid, ["x", "y"]] = geodata
    if not isnan(min_vm_pu):
        if "min_vm_pu" not in net.bus.columns:
            net.bus.loc[:, "min_vm_pu"] = pd.Series()

        net.bus.loc[index, "min_vm_pu"] = float(min_vm_pu)

    if not isnan(max_vm_pu):
        if "max_vm_pu" not in net.bus.columns:
            net.bus.loc[:, "max_vm_pu"] = pd.Series()

        net.bus.loc[index, "max_vm_pu"] = float(max_vm_pu)

    return index


def create_load(net, bus, p_kw, q_kvar=0, const_z_percent=0, const_i_percent=0, sn_kva=nan,
                name=None, scaling=1., index=None,
                in_service=True, type=None, max_p_kw=nan, min_p_kw=nan,
                max_q_kvar=nan, min_q_kvar=nan, controllable=nan):
    """create_load(net, bus, p_kw, q_kvar=0, const_z_percent=0, const_i_percent=0, sn_kva=nan, \
                   name=None, scaling=1., index=None, \
                   in_service=True, type=None, max_p_kw=nan, min_p_kw=nan, max_q_kvar=nan, \
                   min_q_kvar=nan, controllable=nan)
    Adds one load in table net["load"].

    All loads are modelled in the consumer system, meaning load is positive and generation is
    negative active power. Please pay attention to the correct signing of the reactive power as well.

    INPUT:
        **net** - The net within this load should be created

        **bus** (int) - The bus id to which the load is connected

    OPTIONAL:
        **p_kw** (float, default 0) - The real power of the load

        - postive value   -> load
        - negative value  -> generation

        **q_kvar** (float, default 0) - The reactive power of the load

        **const_z_percent** (float, default 0) - percentage of p_kw and q_kvar that will be associated to constant impedance load at rated voltage

        **const_i_percent** (float, default 0) - percentage of p_kw and q_kvar that will be associated to constant current load at rated voltage

        **sn_kva** (float, default None) - Nominal power of the load

        **name** (string, default None) - The name for this load

        **scaling** (float, default 1.) - An OPTIONAL scaling factor to be set customly

        **type** (string, None) -  type variable to classify the load

        **index** (int, None) - Force a specified ID if it is available. If None, the index one higher than the highest already existing index is selected.

        **in_service** (boolean) - True for in_service or False for out of service
        
        **max_p_kw** (float, default NaN) - Maximum active power load. Only respected for OPF
        
        **min_p_kw** (float, default NaN) - Minimum active power load. Only respected for OPF
        
        **max_q_kvar** (float, default NaN) - Maximum reactive power load. Only respected for OPF
        
        **min_q_kvar** (float, default NaN) - Minimum reactive power load. Only respected for OPF
        
        **controllable** (boolean, default NaN) - States, whether a load is controllable or not. Only respected for OPF

    OUTPUT:
        **index** (int) - The unique ID of the created element

    EXAMPLE:
        create_load(net, bus=0, p_kw=10., q_kvar=2.)

    """
    if bus not in net["bus"].index.values:
        raise UserWarning("Cannot attach to bus %s, bus does not exist" % bus)

    if index is None:
        index = get_free_id(net["load"])
    if index in net["load"].index:
        raise UserWarning("A load with the id %s already exists" % id)

    # store dtypes
    dtypes = net.load.dtypes

    net.load.loc[index, ["name", "bus", "p_kw", "const_z_percent", "const_i_percent", "scaling",
                         "q_kvar", "sn_kva", "in_service", "type"]] = \
        [name, bus, p_kw, const_z_percent, const_i_percent, scaling, q_kvar, sn_kva, bool(in_service), type]

    # and preserve dtypes
    _preserve_dtypes(net.load, dtypes)

    if not isnan(min_p_kw):
        if "min_p_kw" not in net.load.columns:
            net.load.loc[:, "min_p_kw"] = pd.Series()

        net.load.loc[index, "min_p_kw"] = float(min_p_kw)

    if not isnan(max_p_kw):
        if "max_p_kw" not in net.load.columns:
            net.load.loc[:, "max_p_kw"] = pd.Series()

        net.load.loc[index, "max_p_kw"] = float(max_p_kw)

    if not isnan(min_q_kvar):
        if "min_q_kvar" not in net.load.columns:
            net.load.loc[:, "min_q_kvar"] = pd.Series()

        net.load.loc[index, "min_q_kvar"] = float(min_q_kvar)

    if not isnan(max_q_kvar):
        if "max_q_kvar" not in net.load.columns:
            net.load.loc[:, "max_q_kvar"] = pd.Series()

        net.load.loc[index, "max_q_kvar"] = float(max_q_kvar)

    if not isnan(controllable):
        if "controllable" not in net.load.columns:
            net.load.loc[:, "controllable"] = pd.Series()

        net.load.loc[index, "controllable"] = bool(controllable)
    else:
        if "controllable" in net.load.columns:
            net.load.loc[index, "controllable"] = False

    return index

def create_load_from_cosphi(net, bus, sn_kva, cos_phi, mode, **kwargs):
    """
    Creates a load element from rated power and power factor cos(phi).

    INPUT:
        **net** - The net within this static generator should be created

        **bus** (int) - The bus id to which the load is connected

        **sn_kva** (float) - rated power of the generator

        **cos_phi** (float) - power factor cos_phi

        **mode** (str) - "ind" for inductive or "cap" for capacitive behaviour

        **kwargs are passed on to the create_load function

    OUTPUT:
        **index** (int) - The unique ID of the created load

    All elements are modeled from a consumer point of view. Active power will therefore always be
    positive, reactive power will be negative for inductive behaviour and positive for capacitive
    behaviour.
    """
    from pandapower.toolbox import pq_from_cosphi
    p_kw, q_kvar = pq_from_cosphi(sn_kva, cos_phi, qmode=mode, pmode="load")
    return create_load(net, bus, sn_kva=sn_kva, p_kw=p_kw, q_kvar=q_kvar, **kwargs)

def create_sgen(net, bus, p_kw, q_kvar=0, sn_kva=nan, name=None, index=None,
                scaling=1., type=None, in_service=True, max_p_kw=nan, min_p_kw=nan,
                max_q_kvar=nan, min_q_kvar=nan, controllable=nan, k=nan, rx=nan):
    """create_sgen(net, bus, p_kw, q_kvar=0, sn_kva=nan, name=None, index=None, \
                scaling=1., type=None, in_service=True, max_p_kw=nan, min_p_kw=nan, \
                max_q_kvar=nan, min_q_kvar=nan, controllable=nan, k=nan, rx=nan)
    Adds one static generator in table net["sgen"].

    Static generators are modelled as negative  PQ loads. This element is used to model generators
    with a constant active and reactive power feed-in. If you want to model a voltage controlled
    generator, use the generator element instead.

    All elements in the grid are modelled in the consumer system, including generators!
    If you want to model the generation of power, you have to assign a negative active power
    to the generator. Please pay attention to the correct signing of the
    reactive power as well.

    INPUT:
        **net** - The net within this static generator should be created

        **bus** (int) - The bus id to which the static generator is connected

        **p_kw** (float) - The real power of the static generator  (negative for generation!)

    OPTIONAL:

        **q_kvar** (float, default 0) - The reactive power of the sgen

        **sn_kva** (float, default None) - Nominal power of the sgen

        **name** (string, default None) - The name for this sgen

        **index** (int, None) - Force a specified ID if it is available. If None, the index one higher than the highest already existing index is selected.

        **scaling** (float, 1.) - An OPTIONAL scaling factor to be set customly

        **type** (string, None) -  type variable to classify the static generator

        **in_service** (boolean) - True for in_service or False for out of service

        **controllable** (bool, NaN) - Whether this generator is controllable by the optimal
        powerflow

    OUTPUT:
        **index** (int) - The unique ID of the created sgen

    EXAMPLE:
        create_sgen(net, 1, p_kw = -120)

    """
    if bus not in net["bus"].index.values:
        raise UserWarning("Cannot attach to bus %s, bus does not exist" % bus)

    if index is None:
        index = get_free_id(net["sgen"])

    if index in net["sgen"].index:
        raise UserWarning("A static generator with the id %s already exists" % index)

    # store dtypes
    dtypes = net.sgen.dtypes

    net.sgen.loc[index, ["name", "bus", "p_kw", "scaling",
                         "q_kvar", "sn_kva", "in_service", "type"]] = \
        [name, bus, p_kw, scaling, q_kvar, sn_kva, bool(in_service), type]

    # and preserve dtypes
    _preserve_dtypes(net.sgen, dtypes)

    if not isnan(min_p_kw):
        if "min_p_kw" not in net.sgen.columns:
            net.sgen.loc[:, "min_p_kw"] = pd.Series()

        net.sgen.loc[index, "min_p_kw"] = float(min_p_kw)

    if not isnan(max_p_kw):
        if "max_p_kw" not in net.sgen.columns:
            net.sgen.loc[:, "max_p_kw"] = pd.Series()

        net.sgen.loc[index, "max_p_kw"] = float(max_p_kw)

    if not isnan(min_q_kvar):
        if "min_q_kvar" not in net.sgen.columns:
            net.sgen.loc[:, "min_q_kvar"] = pd.Series()

        net.sgen.loc[index, "min_q_kvar"] = float(min_q_kvar)

    if not isnan(max_q_kvar):
        if "max_q_kvar" not in net.sgen.columns:
            net.sgen.loc[:, "max_q_kvar"] = pd.Series()

        net.sgen.loc[index, "max_q_kvar"] = float(max_q_kvar)

    if not isnan(controllable):
        if "controllable" not in net.sgen.columns:
            net.sgen.loc[:, "controllable"] = pd.Series()

        net.sgen.loc[index, "controllable"] = bool(controllable)
    else:
        if "controllable" in net.sgen.columns:
            net.sgen.loc[index, "controllable"] = False

    if not isnan(k):
        if "k" not in net.sgen.columns:
            net.sgen.loc[:, "k"] = pd.Series()

        net.sgen.loc[index, "k"] = float(k)

    if not isnan(rx):
        if "rx" not in net.sgen.columns:
            net.sgen.loc[:, "rx"] = pd.Series()

        net.sgen.loc[index, "rx"] = float(rx)

    return index

def create_sgen_from_cosphi(net, bus, sn_kva, cos_phi, mode, **kwargs):
    """
    Creates an sgen element from rated power and power factor cos(phi).

    INPUT:
        **net** - The net within this static generator should be created

        **bus** (int) - The bus id to which the static generator is connected

        **sn_kva** (float) - rated power of the generator

        **cos_phi** (float) - power factor cos_phi

        **mode** (str) - "ind" for inductive or "cap" for capacitive behaviour

    OUTPUT:
        **index** (int) - The unique ID of the created sgen

    All elements including generators are modeled from a consumer point of view. Active power
    will therefore always be negative, reactive power will be negative for inductive behaviour and
    positive for capacitive behaviour.
    """
    from pandapower.toolbox import pq_from_cosphi
    p_kw, q_kvar = pq_from_cosphi(sn_kva, cos_phi, qmode=mode, pmode="gen")
    return create_sgen(net, bus, sn_kva=sn_kva, p_kw=p_kw, q_kvar=q_kvar, **kwargs)


def create_gen(net, bus, p_kw, vm_pu=1., sn_kva=nan, name=None, index=None, max_q_kvar=nan,
               min_q_kvar=nan, min_p_kw=nan, max_p_kw=nan, scaling=1., type=None,
               controllable=nan, vn_kv=nan, xdss=nan, rdss=nan, cos_phi=nan, in_service=True):
    """create_gen(net, bus, p_kw, vm_pu=1., sn_kva=nan, name=None, index=None, max_q_kvar=nan, \
               min_q_kvar=nan, min_p_kw=nan, max_p_kw=nan, scaling=1., type=None, \
               controllable=nan, vn_kv=nan, xdss=nan, rdss=nan, cos_phi=nan, in_service=True)
    Adds a generator to the network.

    Generators are always modelled as voltage controlled PV nodes, which is why the input parameter
    is active power and a voltage set point. If you want to model a generator as PQ load with fixed
    reactive power and variable voltage, please use a static generator instead.

    INPUT:
        **net** - The net within this generator should be created

        **bus** (int) - The bus id to which the generator is connected

    OPTIONAL:
        **p_kw** (float, default 0) - The real power of the generator (negative for generation!)

        **vm_pu** (float, default 0) - The voltage set point of the generator.

        **sn_kva** (float, None) - Nominal power of the generator

        **name** (string, None) - The name for this generator

        **index** (int, None) - Force a specified ID if it is available. If None, the index one higher than the highest already existing index is selected.

        **scaling** (float, 1.0) - scaling factor which for the active power of the generator

        **type** (string, None) - type variable to classify generators

        **controllable** (bool, NaN) - Whether this generator is controllable by the optimal
        powerflow

        **vn_kv** (float, NaN) - Rated voltage of the generator for short-circuit calculation

        **xdss** (float, NaN) - Subtransient generator reactance for short-circuit calculation

        **rdss** (float, NaN) - Subtransient generator resistance for short-circuit calculation

        **cos_phi** (float, NaN) - Rated cosine phi of the generator for short-circuit calculation

        **in_service** (bool, True) - True for in_service or False for out of service

    OUTPUT:
        **index** (int) - The unique ID of the created generator

    EXAMPLE:
        create_gen(net, 1, p_kw = -120, vm_pu = 1.02)

    """
    if bus not in net["bus"].index.values:
        raise UserWarning("Cannot attach to bus %s, bus does not exist" % bus)

    if bus in net.ext_grid.bus.values:
        raise UserWarning(
            "There is already an external grid at bus %u, thus no other voltage controlling element (ext_grid, gen) is allowed at this bus." % bus)

#    if bus in net.gen.bus.values:
#        raise UserWarning(
#            "There is already a generator at bus %u, only one voltage controlling element (ext_grid, gen) is allowed per bus." % bus)

    if index is None:
        index = get_free_id(net["gen"])

    if index in net["gen"].index:
        raise UserWarning("A generator with the id %s already exists" % index)

    # store dtypes
    dtypes = net.gen.dtypes

    net.gen.loc[index, ["name", "bus", "p_kw", "vm_pu", "sn_kva",  "type", "in_service",
                        "scaling"]] = [name, bus, p_kw, vm_pu, sn_kva, type, bool(in_service),
                                       scaling]

    # and preserve dtypes
    _preserve_dtypes(net.gen, dtypes)

    if not isnan(min_p_kw):
        if "min_p_kw" not in net.gen.columns:
            net.gen.loc[:, "min_p_kw"] = pd.Series()
        net.gen.loc[index, "min_p_kw"] = float(min_p_kw)

    if not isnan(max_p_kw):
        if "max_p_kw" not in net.gen.columns:
            net.gen.loc[:, "max_p_kw"] = pd.Series()
        net.gen.loc[index, "max_p_kw"] = float(max_p_kw)

    if not isnan(min_q_kvar):
        if "min_q_kvar" not in net.gen.columns:
            net.gen.loc[:, "min_q_kvar"] = pd.Series()
        net.gen.loc[index, "min_q_kvar"] = float(min_q_kvar)

    if not isnan(max_q_kvar):
        if "max_q_kvar" not in net.gen.columns:
            net.gen.loc[:, "max_q_kvar"] = pd.Series()
        net.gen.loc[index, "max_q_kvar"] = float(max_q_kvar)

    if not isnan(controllable):
        if "controllable" not in net.gen.columns:
            net.gen.loc[:, "controllable"] = pd.Series(False)
        net.gen.loc[index, "controllable"] = bool(controllable)
    elif "controllable" in net.gen.columns:
            net.gen.loc[index, "controllable"] = False

    if not isnan(vn_kv):
        if "vn_kv" not in net.gen.columns:
            net.gen.loc[:, "vn_kv"] = pd.Series()
        net.gen.loc[index, "vn_kv"] = float(vn_kv)

    if not isnan(xdss):
        if "xdss" not in net.gen.columns:
            net.gen.loc[:, "xdss"] = pd.Series()
        net.gen.loc[index, "xdss"] = float(xdss)

    if not isnan(rdss):
        if "rdss" not in net.gen.columns:
            net.gen.loc[:, "rdss"] = pd.Series()
        net.gen.loc[index, "rdss"] = float(rdss)

    if not isnan(cos_phi):
        if "cos_phi" not in net.gen.columns:
            net.gen.loc[:, "cos_phi"] = pd.Series()
        net.gen.loc[index, "cos_phi"] = float(cos_phi)

    return index


def create_ext_grid(net, bus, vm_pu=1.0, va_degree=0., name=None, in_service=True,
                    s_sc_max_mva=nan, s_sc_min_mva=nan, rx_max=nan, rx_min=nan,
                    max_p_kw=nan, min_p_kw=nan, max_q_kvar=nan, min_q_kvar=nan,
                    index=None, **kwargs):
    """create_ext_grid(net, bus, vm_pu=1.0, va_degree=0., name=None, in_service=True,\
                    s_sc_max_mva=nan, s_sc_min_mva=nan, rx_max=nan, rx_min=nan,\
                    max_p_kw=nan, min_p_kw=nan, max_q_kvar=nan, min_q_kvar=nan,\
                    index=None)
    Creates an external grid connection.

    External grids represent the higher level power grid connection and are modelled as the slack
    bus in the power flow calculation.

    INPUT:
        **net** - pandapower network

        **bus** (int) - bus where the slack is connected

    OPTIONAL:
        **vm_pu** (float, default 1.0) - voltage at the slack node in per unit

        **va_degree** (float, default 0.) - voltage angle at the slack node in degrees*

        **name** (string, default None) - name of of the external grid

        **in_service** (boolean) - True for in_service or False for out of service

        **Sk_max** - maximal short circuit apparent power **

        **SK_min** - maximal short circuit apparent power **

        **RX_max** - maximal R/X-ratio **

        **RK_min** - minimal R/X-ratio **

        \* only considered in loadflow if calculate_voltage_angles = True

        \** only needed for short circuit calculations

    EXAMPLE:
        create_ext_grid(net, 1, voltage = 1.03)
    """
    if bus not in net["bus"].index.values:
        raise UserWarning("Cannot attach to bus %s, bus does not exist" % bus)

    if index and index in net["ext_grid"].index:
        raise UserWarning("An external grid with with index %s already exists" % index)

    if index is None:
        index = get_free_id(net["ext_grid"])

    if bus in net.ext_grid.bus.values:
        raise UserWarning(
            "There is already an external grid at bus %u, thus no other voltage controlling element (ext_grid, gen) is allowed at this bus." % bus)

    if bus in net.gen.bus.values:
        raise UserWarning(
            "There is already a generator at bus %u, thus no ext_grid is allowed at this bus." % bus)

        # store dtypes
    dtypes = net.ext_grid.dtypes

    net.ext_grid.loc[index, ["bus", "name", "vm_pu", "va_degree", "in_service"]] = \
        [bus, name, vm_pu, va_degree, bool(in_service)]

    if not isnan(s_sc_max_mva):
        if "s_sc_max_mva" not in net.ext_grid.columns:
            net.ext_grid.loc[:, "s_sc_max_mva"] = pd.Series()

        net.ext_grid.at[:, "s_sc_max_mva"] = float(s_sc_max_mva)

    if not isnan(s_sc_min_mva):
        if "s_sc_min_mva" not in net.ext_grid.columns:
            net.ext_grid.loc[:, "s_sc_min_mva"] = pd.Series()

        net.ext_grid.at[index, "s_sc_min_mva"] = float(s_sc_min_mva)

    if not isnan(rx_min):
        if "rx_min" not in net.ext_grid.columns:
            net.ext_grid.loc[:, "rx_min"] = pd.Series()

        net.ext_grid.at[index, "rx_min"] = float(rx_min)

    if not isnan(rx_max):
        if "rx_max" not in net.ext_grid.columns:
            net.ext_grid.loc[:, "rx_max"] = pd.Series()

        net.ext_grid.at[index, "rx_max"] = float(rx_max)

    if not isnan(min_p_kw):
        if "min_p_kw" not in net.ext_grid.columns:
            net.ext_grid.loc[:, "min_p_kw"] = pd.Series()

        net.ext_grid.loc[index, "min_p_kw"] = float(min_p_kw)

    if not isnan(max_p_kw):
        if "max_p_kw" not in net.ext_grid.columns:
            net.ext_grid.loc[:, "max_p_kw"] = pd.Series()

        net.ext_grid.loc[index, "max_p_kw"] = float(max_p_kw)

    if not isnan(min_q_kvar):
        if "min_q_kvar" not in net.ext_grid.columns:
            net.ext_grid.loc[:, "min_q_kvar"] = pd.Series()

        net.ext_grid.loc[index, "min_q_kvar"] = float(min_q_kvar)

    if not isnan(max_q_kvar):
        if "max_q_kvar" not in net.ext_grid.columns:
            net.ext_grid.loc[:, "max_q_kvar"] = pd.Series()

        net.ext_grid.loc[index, "max_q_kvar"] = float(max_q_kvar)

        # and preserve dtypes
    _preserve_dtypes(net.ext_grid, dtypes)
    return index


def create_line(net, from_bus, to_bus, length_km, std_type, name=None, index=None, geodata=None,
                df=1., parallel=1, in_service=True, max_loading_percent=nan):
    """ create_line(net, from_bus, to_bus, length_km, std_type, name=None, index=None, geodata=None,\
                df=1., parallel=1, in_service=True, max_loading_percent=nan)
    Creates a line element in net["line"]
    The line parameters are defined through the standard type library.


    INPUT:
        **net** - The net within this line should be created

        **from_bus** (int) - ID of the bus on one side which the line will be connected with

        **to_bus** (int) - ID of the bus on the other side which the line will be connected with

        **length_km** (float) - The line length in km

        **std_type** (string) - The linetype of a standard line pre-defined in standard_linetypes.

    OPTIONAL:
        **name** (string) - A custom name for this line

        **index** (int, None) - Force a specified ID if it is available. If None, the index one higher than the highest already existing index is selected.

        **geodata**
        (array, default None, shape= (,2L)) -
        The linegeodata of the line. The first row should be the coordinates
        of bus a and the last should be the coordinates of bus b. The points
        in the middle represent the bending points of the line

        **in_service** (boolean) - True for in_service or False for out of service

        **df** (float) - derating factor: maximal current of line in relation to nominal current of line (from 0 to 1)

        **parallel** (integer) - number of parallel line systems

        **max_loading_percent (float)** - maximum current loading (only needed for OPF)

    OUTPUT:
        **index** (int) - The unique ID of the created line

    EXAMPLE:
        create_line(net, "line1", from_bus = 0, to_bus = 1, length_km=0.1,  std_type="NAYY 4x50 SE")

    """

    # check if bus exist to attach the line to
    for b in [from_bus, to_bus]:
        if b not in net["bus"].index.values:
            raise UserWarning("Line %s tries to attach to non-existing bus %s"% (name, b))

    if index is None:
        index = get_free_id(net["line"])

    if index in net["line"].index:
        raise UserWarning("A line with index %s already exists" % index)

    v = {
        "name": name, "length_km": length_km, "from_bus": from_bus,
        "to_bus": to_bus, "in_service": bool(in_service), "std_type": std_type,
        "df": df, "parallel": parallel
    }

    lineparam = load_std_type(net, std_type, "line")
    v.update({
        "r_ohm_per_km": lineparam["r_ohm_per_km"],
        "x_ohm_per_km": lineparam["x_ohm_per_km"],
        "c_nf_per_km": lineparam["c_nf_per_km"],
        "max_i_ka": lineparam["max_i_ka"]
    })
    if "type" in lineparam:
        v.update({"type": lineparam["type"]})

    # store dtypes
    dtypes = net.line.dtypes

    net.line.loc[index, list(v.keys())] = list(v.values())

    # and preserve dtypes
    _preserve_dtypes(net.line, dtypes)

    if geodata is not None:
        net["line_geodata"].loc[index, "coords"] = geodata

    if not isnan(max_loading_percent):
        if "max_loading_percent" not in net.line.columns:
            net.line.loc[:, "max_loading_percent"] = pd.Series()

        net.line.loc[index, "max_loading_percent"] = float(max_loading_percent)

    return index


def create_line_from_parameters(net, from_bus, to_bus, length_km, r_ohm_per_km, x_ohm_per_km,
                                c_nf_per_km, max_i_ka, name=None, index=None, type=None,
                                geodata=None, in_service=True, df=1., parallel=1,
                                max_loading_percent=nan, **kwargs):

    """create_line_from_parameters(net, from_bus, to_bus, length_km, r_ohm_per_km, x_ohm_per_km, \
                                c_nf_per_km, max_i_ka, name=None, index=None, type=None, \
                                geodata=None, in_service=True, df=1., parallel=1, \
                                max_loading_percent=nan, **kwargs)
    Creates a line element in net["line"] from line parameters.

    INPUT:
        **net** - The net within this line should be created

        **from_bus** (int) - ID of the bus on one side which the line will be connected with

        **to_bus** (int) - ID of the bus on the other side which the line will be connected with

        **length_km** (float) - The line length in km

        **r_ohm_per_km** (float) - line resistance in ohm per km

        **x_ohm_per_km** (float) - line reactance in ohm per km

        **c_nf_per_km** (float) - line capacitance in nF per km

        **max_i_ka** (float) - maximum thermal current in kA


    OPTIONAL:
        **name** (string) - A custom name for this line

        **index** (int, None) - Force a specified ID if it is available. If None, the index one higher than the highest already existing index is selected.

        **in_service** (boolean) - True for in_service or False for out of service

        **type** (str) - type of line ("oh" for overhead line or "cs" for cable system)

        **df** (float) - derating factor: maximal current of line  in relation to nominal current of line (from 0 to 1)

        **parallel** (integer) - number of parallel line systems

        **geodata**
        (array, default None, shape= (,2L)) -
        The linegeodata of the line. The first row should be the coordinates
        of bus a and the last should be the coordinates of bus b. The points
        in the middle represent the bending points of the line

        **kwargs** - nothing to see here, go along

        **max_loading_percent (float)** - maximum current loading (only needed for OPF)

    OUTPUT:
        **index** (int) - The unique ID of the created line

    EXAMPLE:
        create_line_from_parameters(net, "line1", from_bus = 0, to_bus = 1, lenght_km=0.1,
        r_ohm_per_km = .01, x_ohm_per_km = 0.05, c_nf_per_km = 10,
        max_i_ka = 0.4)

    """

    # check if bus exist to attach the line to
    for b in [from_bus, to_bus]:
        if b not in net["bus"].index.values:
            raise UserWarning("Line %s tries to attach to non-existing bus %s"
                              % (name, b))

    if index is None:
        index = get_free_id(net["line"])

    if index in net["line"].index:
        raise UserWarning("A line with index %s already exists" % index)

    v = {
        "name": name, "length_km": length_km, "from_bus": from_bus,
        "to_bus": to_bus, "in_service": bool(in_service), "std_type": None,
        "df": df, "r_ohm_per_km": r_ohm_per_km, "x_ohm_per_km": x_ohm_per_km,
        "c_nf_per_km": c_nf_per_km, "max_i_ka": max_i_ka, "parallel": parallel, "type": type
    }

    # store dtypes
    dtypes = net.line.dtypes

    net.line.loc[index, list(v.keys())] = list(v.values())

    # and preserve dtypes
    _preserve_dtypes(net.line, dtypes)

    if geodata is not None:
        net["line_geodata"].loc[index, "coords"] = geodata

    if not isnan(max_loading_percent):
        if "max_loading_percent" not in net.line.columns:
            net.line.loc[:, "max_loading_percent"] = pd.Series()

        net.line.loc[index, "max_loading_percent"] = float(max_loading_percent)

    return index


def create_transformer(net, hv_bus, lv_bus, std_type, name=None, tp_pos=nan, in_service=True,
                       index=None, max_loading_percent=nan, parallel=1):
    """create_transformer(net, hv_bus, lv_bus, std_type, name=None, tp_pos=nan, in_service=True, \
                       index=None, max_loading_percent=nan, parallel=1)
    Creates a two-winding transformer in table net["trafo"].
    The trafo parameters are defined through the standard type library.

    INPUT:
        **net** - The net within this transformer should be created

        **hv_bus** (int) - The bus on the high-voltage side on which the transformer will be connected to

        **lv_bus** (int) - The bus on the low-voltage side on which the transformer will be connected to

        **std_type** -  The used standard type from the standard type library

    OPTIONAL:
        **name** (string, None) - A custom name for this transformer

        **tp_pos** (int, nan) - current tap position of the transformer. Defaults to the medium position (tp_mid)

        **in_service** (boolean, True) - True for in_service or False for out of service

        **index** (int, None) - Force a specified ID if it is available. If None, the index one higher than the highest already existing index is selected.

        **max_loading_percent (float)** - maximum current loading (only needed for OPF)

    OUTPUT:
        **index** (int) - The unique ID of the created transformer

    EXAMPLE:
        create_transformer(net, hv_bus = 0, lv_bus = 1, name = "trafo1", std_type = "0.4 MVA 10/0.4 kV")
    """

    # Check if bus exist to attach the trafo to
    for b in [hv_bus, lv_bus]:
        if b not in net["bus"].index.values:
            raise UserWarning("Trafo tries to attach to bus %s" % b)

    v = {
        "name": name, "hv_bus": hv_bus, "lv_bus": lv_bus,
        "in_service": bool(in_service), "std_type": std_type
    }
    ti = load_std_type(net, std_type, "trafo")

    if index is None:
        index = get_free_id(net["trafo"])

    if index in net["trafo"].index:
        raise UserWarning("A transformer with index %s already exists" % index)

    v.update({
        "sn_kva": ti["sn_kva"],
        "vn_hv_kv": ti["vn_hv_kv"],
        "vn_lv_kv": ti["vn_lv_kv"],
        "vsc_percent": ti["vsc_percent"],
        "vscr_percent": ti["vscr_percent"],
        "pfe_kw": ti["pfe_kw"],
        "i0_percent": ti["i0_percent"],
        "parallel": parallel,
        "shift_degree": ti["shift_degree"] if "shift_degree" in ti else 0
        })
    for tp in ("tp_mid", "tp_max", "tp_min", "tp_side", "tp_st_percent", "tp_st_degree"):
        if tp in ti:
            v.update({tp: ti[tp]})

    if ("tp_mid" in v) and (tp_pos is nan):
        v["tp_pos"] = v["tp_mid"]
    else:
        v["tp_pos"] = tp_pos
        if type(tp_pos) == float:
            net.trafo.tp_pos = net.trafo.tp_pos.astype(float)
    # store dtypes
    dtypes = net.trafo.dtypes

    net.trafo.loc[index, list(v.keys())] = list(v.values())

    if not isnan(max_loading_percent):
        if "max_loading_percent" not in net.trafo.columns:
            net.trafo.loc[:, "max_loading_percent"] = pd.Series()

        net.trafo.loc[index, "max_loading_percent"] = float(max_loading_percent)

    # and preserve dtypes
    _preserve_dtypes(net.trafo, dtypes)

    return index


def create_transformer_from_parameters(net, hv_bus, lv_bus, sn_kva, vn_hv_kv, vn_lv_kv,
                                       vscr_percent, vsc_percent, pfe_kw, i0_percent,
                                       shift_degree=0, tp_side=None, tp_mid=nan, tp_max=nan,
                                       tp_min=nan, tp_st_percent=nan, tp_st_degree=nan,
                                       tp_pos=nan, in_service=True, name=None, index=None,
                                       max_loading_percent=nan, parallel=1, **kwargs):

    """create_transformer_from_parameters(net, hv_bus, lv_bus, sn_kva, vn_hv_kv, vn_lv_kv, \
                                       vscr_percent, vsc_percent, pfe_kw, i0_percent, \
                                       shift_degree=0, tp_side=None, tp_mid=nan, tp_max=nan, \
                                       tp_min=nan, tp_st_percent=nan, tp_st_degree=nan, \
                                       tp_pos=nan, in_service=True, name=None, index=None, \
                                       max_loading_percent=nan, parallel=1, **kwargs)
    Creates a two-winding transformer in table net["trafo"].
    The trafo parameters are defined through the standard type library.

    INPUT:
        **net** - The net within this transformer should be created

        **hv_bus** (int) - The bus on the high-voltage side on which the transformer will be connected to

        **lv_bus** (int) - The bus on the low-voltage side on which the transformer will be connected to

        **sn_kva** (float) - rated apparent power

        **vn_hv_kv** (float) - rated voltage on high voltage side

        **vn_lv_kv** (float) - rated voltage on low voltage side

        **vscr_percent** (float) - real part of relative short-circuit voltage

        **vsc_percent** (float) - relative short-circuit voltage

        **pfe_kw** (float)  - iron losses in kW

        **i0_percent** (float) - open loop losses in percent of rated current

    OPTIONAL:
        **in_service** (boolean) - True for in_service or False for out of service

        **parallel** (integer) - number of parallel transformers

        **name** (string) - A custom name for this transformer

        **shift_degree** (float) - Angle shift over the transformer*

        **tp_side** (string) - position of tap changer ("hv", "lv")

        **tp_pos** (int, nan) - current tap position of the transformer. Defaults to the medium position (tp_mid)

        **tp_mid** (int, nan) - tap position where the transformer ratio is equal to the ration of the rated voltages

        **tp_max** (int, nan) - maximal allowed tap position

        **tp_min** (int, nan):  minimal allowed tap position

        **tp_st_percent** (int) - tap step in percent

        **index** (int, None) - Force a specified ID if it is available. If None, the index one higher than the highest already existing index is selected.

        **kwargs** - nothing to see here, go along

        \* only considered in loadflow if calculate_voltage_angles = True

        **max_loading_percent (float)** - maximum current loading (only needed for OPF)

    OUTPUT:
        **index** (int) - The unique ID of the created transformer

    EXAMPLE:
        create_transformer_from_parameters(net, hv_bus=0, lv_bus=1, name="trafo1", sn_kva=40, vn_hv_kv=110, vn_lv_kv=10, vsc_percent=10, vscr_percent=0.3, pfe_kw=30, i0_percent=0.1, shift_degree=30)
    """

    # Check if bus exist to attach the trafo to
    for b in [hv_bus, lv_bus]:
        if b not in net["bus"].index.values:
            raise UserWarning("Trafo tries to attach to bus %s" % b)

    if index is None:
        index = get_free_id(net["trafo"])

    if index in net["trafo"].index:
        raise UserWarning("A transformer with index %s already exists" % index)

    if tp_pos is nan:
        tp_pos = tp_mid
    v = {
        "name": name, "hv_bus": hv_bus, "lv_bus": lv_bus,
        "in_service": bool(in_service), "std_type": None, "sn_kva": sn_kva, "vn_hv_kv": vn_hv_kv,
        "vn_lv_kv": vn_lv_kv, "vsc_percent": vsc_percent, "vscr_percent": vscr_percent,
        "pfe_kw": pfe_kw, "i0_percent": i0_percent, "tp_mid": tp_mid,
        "tp_max": tp_max, "tp_min": tp_min, "shift_degree": shift_degree,
        "tp_side": tp_side, "tp_st_percent": tp_st_percent, "tp_st_degree": tp_st_degree,
        "parallel": parallel
    }

    if ("tp_mid" in v) and (tp_pos is nan):
        v["tp_pos"] = v["tp_mid"]
    else:
        v["tp_pos"] = tp_pos
        if type(tp_pos) == float:
            net.trafo.tp_pos = net.trafo.tp_pos.astype(float)

    # store dtypes
    dtypes = net.trafo.dtypes

    net.trafo.loc[index, list(v.keys())] = list(v.values())

    # and preserve dtypes
    _preserve_dtypes(net.trafo, dtypes)

    if not isnan(max_loading_percent):
        if "max_loading_percent" not in net.trafo.columns:
            net.trafo.loc[:, "max_loading_percent"] = pd.Series()

        net.trafo.loc[index, "max_loading_percent"] = float(max_loading_percent)

    return index


def create_transformer3w(net, hv_bus, mv_bus, lv_bus, std_type, name=None, tp_pos=nan,
                         in_service=True, index=None, max_loading_percent=nan):
    """create_transformer3w(net, hv_bus, mv_bus, lv_bus, std_type, name=None, tp_pos=nan, \
                         in_service=True, index=None, max_loading_percent=nan)
    Creates a three-winding transformer in table net["trafo3w"].
    The trafo parameters are defined through the standard type library.

    INPUT:
        **net** - The net within this transformer should be created

        **hv_bus** (int) - The bus on the high-voltage side on which the transformer will be connected to

        **mv_bus** (int) - The medium voltage bus on which the transformer will be connected to

        **lv_bus** (int) - The bus on the low-voltage side on which the transformer will be connected to

        **std_type** -  The used standard type from the standard type library

    OPTIONAL:
        **name** (string) - A custom name for this transformer

        **tp_pos** (int, nan) - current tap position of the transformer. Defaults to the medium position (tp_mid)

        **in_service** (boolean) - True for in_service or False for out of service

        **index** (int, None) - Force a specified ID if it is available. If None, the index one higher than the highest already existing index is selected.

        **max_loading_percent (float)** - maximum current loading (only needed for OPF)

    OUTPUT:
        **index** (int) - The unique ID of the created transformer

    EXAMPLE:
        create_transformer3w(net, hv_bus = 0, mv_bus = 1, lv_bus = 2, name = "trafo1", std_type = "63/25/38 MVA 110/20/10 kV")
    """

    # Check if bus exist to attach the trafo to
    for b in [hv_bus, mv_bus, lv_bus]:
        if b not in net["bus"].index.values:
            raise UserWarning("Trafo tries to attach to bus %s" % b)

    v = {
        "name": name, "hv_bus": hv_bus, "mv_bus": mv_bus, "lv_bus": lv_bus,
        "in_service": bool(in_service), "std_type": std_type
    }
    ti = load_std_type(net, std_type, "trafo3w")

    if index is None:
        index = get_free_id(net["trafo3w"])

    if index in net["trafo3w"].index:
        raise UserWarning("A three winding transformer with index %s already exists" % index)

    v.update({
        "sn_hv_kva": ti["sn_hv_kva"],
        "sn_mv_kva": ti["sn_mv_kva"],
        "sn_lv_kva": ti["sn_lv_kva"],
        "vn_hv_kv": ti["vn_hv_kv"],
        "vn_mv_kv": ti["vn_mv_kv"],
        "vn_lv_kv": ti["vn_lv_kv"],
        "vsc_hv_percent": ti["vsc_hv_percent"],
        "vsc_mv_percent": ti["vsc_mv_percent"],
        "vsc_lv_percent": ti["vsc_lv_percent"],
        "vscr_hv_percent": ti["vscr_hv_percent"],
        "vscr_mv_percent": ti["vscr_mv_percent"],
        "vscr_lv_percent": ti["vscr_lv_percent"],
        "pfe_kw": ti["pfe_kw"],
        "i0_percent": ti["i0_percent"],
        "shift_mv_degree": ti["shift_mv_degree"] if "shift_mv_degree" in ti else 0,
        "shift_lv_degree": ti["shift_lv_degree"] if "shift_lv_degree" in ti else 0
    })
    for tp in ("tp_mid", "tp_max", "tp_min", "tp_side", "tp_st_percent"):
        if tp in ti:
            v.update({tp: ti[tp]})

    if ("tp_mid" in v) and (tp_pos is nan):
        v["tp_pos"] = v["tp_mid"]
    else:
        v["tp_pos"] = tp_pos
        if type(tp_pos) == float:
            net.trafo3w.tp_pos = net.trafo3w.tp_pos.astype(float)

    dd = pd.DataFrame(v, index=[index])
    net["trafo3w"] = net["trafo3w"].append(dd).reindex_axis(net["trafo3w"].columns, axis=1)

    if not isnan(max_loading_percent):
        if "max_loading_percent" not in net.trafo3w.columns:
            net.trafo3w.loc[:, "max_loading_percent"] = pd.Series()

        net.trafo3w.loc[index, "max_loading_percent"] = float(max_loading_percent)

    return index


def create_transformer3w_from_parameters(net, hv_bus, mv_bus, lv_bus, vn_hv_kv, vn_mv_kv, vn_lv_kv,
                                         sn_hv_kva, sn_mv_kva, sn_lv_kva, vsc_hv_percent,
                                         vsc_mv_percent, vsc_lv_percent, vscr_hv_percent,
                                         vscr_mv_percent, vscr_lv_percent, pfe_kw, i0_percent,
                                         shift_mv_degree=0., shift_lv_degree=0., tp_side=None,
                                         tp_st_percent=nan, tp_pos=nan, tp_mid=nan, tp_max=nan,
                                         tp_min=nan, name=None, in_service=True, index=None,
                                         max_loading_percent=nan):
    """create_transformer3w_from_parameters(net, hv_bus, mv_bus, lv_bus, vn_hv_kv, vn_mv_kv, vn_lv_kv, \
                                         sn_hv_kva, sn_mv_kva, sn_lv_kva, vsc_hv_percent, \
                                         vsc_mv_percent, vsc_lv_percent, vscr_hv_percent, \
                                         vscr_mv_percent, vscr_lv_percent, pfe_kw, i0_percent,\
                                         shift_mv_degree=0., shift_lv_degree=0., tp_side=None, \
                                         tp_st_percent=nan, tp_pos=nan, tp_mid=nan, tp_max=nan, \
                                         tp_min=nan, name=None, in_service=True, index=None, \
                                         max_loading_percent=nan)
    Adds a three-winding transformer in table net["trafo3w"].

    Input:
        **net** (pandapowerNet) - The net within this transformer should be created

        **hv_bus** (int) - The bus on the high-voltage side on which the transformer will be connected to

        **mv_bus** (int) - The bus on the middle-voltage side on which the transformer will be connected to

        **lv_bus** (int) - The bus on the low-voltage side on which the transformer will be connected to

        **vn_hv_kv** (float) rated voltage on high voltage side

        **vn_mv_kv** (float) rated voltage on medium voltage side

        **vn_lv_kv** (float) rated voltage on low voltage side

        **sn_hv_kva** (float) - rated apparent power on high voltage side

        **sn_mv_kva** (float) - rated apparent power on medium voltage side

        **sn_lv_kva** (float) - rated apparent power on low voltage side

        **vsc_hv_percent** (float) - short circuit voltage from high to medium voltage

        **vsc_mv_percent** (float) - short circuit voltage from medium to low voltage

        **vsc_lv_percent** (float) - short circuit voltage from high to low voltage

        **vscr_hv_percent** (float) - real part of short circuit voltage from high to medium voltage

        **vscr_mv_percent** (float) - real part of short circuit voltage from medium to low voltage

        **vscr_lv_percent** (float) - real part of short circuit voltage from high to low voltage

        **pfe_kw** (float) - iron losses

        **i0_percent** (float) - open loop losses

    OPTIONAL:
        **shift_mv_degree** (float, 0) - angle shift to medium voltage side*

        **shift_lv_degree** (float, 0) - angle shift to low voltage side*

        **tp_st_percent** (float) - Tap step in percent

        **tp_side** (string, None) - "hv", "mv", "lv"

        **tp_mid** (int, nan) - default tap position

        **tp_min** (int, nan) - Minimum tap position

        **tp_max** (int, nan) - Maximum tap position

        **tp_pos** (int, nan) - current tap position of the transformer. Defaults to the medium position (tp_mid)

        **name** (string, None) - Name of the 3-winding transformer

        **in_service** (boolean, True) - True for in_service or False for out of service

        \* only considered in loadflow if calculate_voltage_angles = True
        \**The model currently only supports one tap-changer per 3W Transformer.

        **max_loading_percent (float)** - maximum current loading (only needed for OPF)

    OUTPUT:
        **trafo_id** - The unique trafo_id of the created 3W transformer

    Example:
        create_transformer3w_from_parameters(net, hv_bus=0, mv_bus=1, lv_bus=2, name="trafo1",
        sn_hv_kva=40, sn_mv_kva=20, sn_lv_kva=20, vn_hv_kv=110, vn_mv_kv=20, vn_lv_kv=10, vsc_hv_percent=10,
        vsc_mv_percent=11, vsc_lv_percent=12, vscr_hv_percent=0.3, vscr_mv_percent=0.31, vscr_lv_percent=0.32,
        pfe_kw=30, i0_percent=0.1, shift_mv_degree=30, shift_lv_degree=30)

    """

    # Check if bus exist to attach the trafo to
    for b in [hv_bus, mv_bus, lv_bus]:
        if b not in net["bus"].index.values:
            raise UserWarning("Trafo tries to attach to non-existent bus %s" % b)

    if index is None:
        index = get_free_id(net["trafo3w"])

    if index in net["trafo3w"].index:
        raise UserWarning("A three winding transformer with index %s already exists" % index)

    if tp_pos is nan:
        tp_pos = tp_mid

    # store dtypes
    dtypes = net.trafo3w.dtypes

    net.trafo3w.loc[index, ["lv_bus", "mv_bus", "hv_bus", "vn_hv_kv", "vn_mv_kv", "vn_lv_kv",
                            "sn_hv_kva", "sn_mv_kva", "sn_lv_kva", "vsc_hv_percent",
                            "vsc_mv_percent", "vsc_lv_percent", "vscr_hv_percent",
                            "vscr_mv_percent", "vscr_lv_percent", "pfe_kw", "i0_percent",
                            "shift_mv_degree", "shift_lv_degree", "tp_side", "tp_st_percent",
                            "tp_pos", "tp_mid", "tp_max", "tp_min", "in_service", "name", "std_type"
                            ]] = \
        [lv_bus, mv_bus, hv_bus, vn_hv_kv, vn_mv_kv, vn_lv_kv,
         sn_hv_kva, sn_mv_kva, sn_lv_kva, vsc_hv_percent, vsc_mv_percent,
         vsc_lv_percent, vscr_hv_percent, vscr_mv_percent, vscr_lv_percent,
         pfe_kw, i0_percent, shift_mv_degree, shift_lv_degree,
         tp_side, tp_st_percent, tp_pos, tp_mid, tp_max,
         tp_min, bool(in_service), name, None]

    # and preserve dtypes
    _preserve_dtypes(net.trafo3w, dtypes)

    if not isnan(max_loading_percent):
        if "max_loading_percent" not in net.trafo3w.columns:
            net.trafo3w.loc[:, "max_loading_percent"] = pd.Series()

        net.trafo3w.loc[index, "max_loading_percent"] = float(max_loading_percent)

    return index


def create_switch(net, bus, element, et, closed=True, type=None, name=None, index=None):
    """
    Adds a switch in the net["switch"] table.

    Switches can be either between to buses (bus-bus switch) or at the end of a line or transformer
    element (bus-elememnt switch).

    Two buses that are connected through a closed bus-bus switches are fused in the power flow if
    the switch es closed or separated if the switch is open.

    An element that is connected to a bus through a bus-element switch is connected to the bus
    if the switch is closed or disconnected if the switch is open.

    INPUT:
        **net** (pandapowerNet) - The net within this transformer should be created

        **bus** - The bus that the switch is connected to

        **element** - index of the element: bus id if et == "b", line id if et == "l", trafo id if et == "t"

        **et** - (string) element type: "l" = switch between bus and line, "t" = switch between
        bus and transformer, "t3" = switch between bus and 3-winding transformer, "b" = switch
        between two buses

        **closed** (boolean, True) - switch position: False = open, True = closed

        **type** (int, None) - indicates the type of switch: "LS" = Load Switch, "CB" = Circuit Breaker, "LBS" = Load Break Switch or "DS" = Disconnecting Switch

    OPTIONAL:
        **name** (string, default None) - The name for this switch

    OUTPUT:
        **sid** - The unique switch_id of the created switch

    EXAMPLE:
        create_switch(net, bus =  0, element = 1, et = 'b', type ="LS")

        create_switch(net, bus = 0, element = 1, et = 'l')

    """
    if bus not in net["bus"].index:
        raise UserWarning("Unknown bus index")
    if et == "l":
        elm_tab = 'line'
        if element not in net[elm_tab].index:
            raise UserWarning("Unknown line index")
        if (not net[elm_tab]["from_bus"].loc[element] == bus and
                not net[elm_tab]["to_bus"].loc[element] == bus):
            raise UserWarning("Line %s not connected to bus %s" % (element, bus))
    elif et == "t":
        elm_tab = 'trafo'
        if element not in net[elm_tab].index:
            raise UserWarning("Unknown bus index")
        if (not net[elm_tab]["hv_bus"].loc[element] == bus and
                not net[elm_tab]["lv_bus"].loc[element] == bus):
            raise UserWarning("Trafo %s not connected to bus %s" % (element, bus))
    elif et == "t3":
        raise NotImplemented("Switches for three winding transformers are not implemented")
#        elm_tab = 'trafo3w'
#        if element not in net[elm_tab].index:
#            raise UserWarning("Unknown trafo3w index")
#        if (not net[elm_tab]["hv_bus"].loc[element] == bus and
#                not net[elm_tab]["mv_bus"].loc[element] == bus and
#                not net[elm_tab]["lv_bus"].loc[element] == bus):
#            raise UserWarning("Trafo3w %s not connected to bus %s" % (element, bus))
    elif et == "b":
        if element not in net["bus"].index:
            raise UserWarning("Unknown bus index")
    else:
        raise UserWarning("Unknown element type")

    if index is None:
        index = get_free_id(net["switch"])
    if index in net["switch"].index:
        raise UserWarning("A switch with index %s already exists" % index)

    # store dtypes
    dtypes = net.switch.dtypes

    net.switch.loc[index, ["bus", "element", "et", "closed", "type", "name"]] = \
        [bus, element, et, closed, type, name]

    # and preserve dtypes
    _preserve_dtypes(net.switch, dtypes)

    return index


def create_shunt(net, bus, q_kvar, p_kw=0., vn_kv=None, step=1, name=None, in_service=True,
                 index=None):
    """
    Creates a shunt element

    INPUT:
        **net** (pandapowerNet) - The pandapower network in which the element is created

        **bus** - bus number of bus to whom the shunt is connected to

        **p_kw** - shunt active power in kW at v= 1.0 p.u.

        **q_kvar** - shunt susceptance in kVAr at v= 1.0 p.u.

    OPTIONAL:
        **vn_kv** (float, None) - rated voltage of the shunt. Defaults to rated voltage of connected bus

        **step** (int, 1) - step of shunt with which power values are multiplied

        **name** (str, None) - element name

        **in_service** (boolean, True) - True for in_service or False for out of service

        **index** (int, None) - Force a specified ID if it is available. If None, the index one higher than the highest already existing index is selected.

    OUTPUT:
        **index** (int) - The unique ID of the created shunt

    EXAMPLE:
        create_shunt(net, 0, 20)
    """
    if bus not in net["bus"].index.values:
        raise UserWarning("Cannot attach to bus %s, bus does not exist" % bus)

    if index is None:
        index = get_free_id(net["shunt"])

    if index in net["shunt"].index:
        raise UserWarning("A shunt with index %s already exists" % index)

    if vn_kv is None:
        vn_kv = net.bus.vn_kv.at[bus]
    # store dtypes
    dtypes = net.shunt.dtypes

    net.shunt.loc[index, ["bus", "name", "p_kw", "q_kvar", "vn_kv", "step", "in_service"]] = \
        [bus, name, p_kw, q_kvar, vn_kv, step, in_service]

    # and preserve dtypes
    _preserve_dtypes(net.shunt, dtypes)

    return index

def create_shunt_as_capacitor(net, bus, q_kvar, loss_factor, **kwargs):
    """
    Creates a shunt element representing a capacitor bank.

    INPUT:

        **net** (pandapowerNet) - The pandapower network in which the element is created

        **bus** - bus number of bus to whom the shunt is connected to

        **q_kvar** (float) - reactive power of the capacitor bank at rated voltage

        **loss_factor** (float) - loss factor tan(delta) of the capacitor bank

        **kwargs are passed to the create_shunt function


    OUTPUT:
        **index** (int) - The unique ID of the created shunt
    """
    q_kvar = -abs(q_kvar) #q is always negative for capacitor
    p_kw = abs(q_kvar*loss_factor) #p is always positive for active power losses
    return create_shunt(net, bus, q_kvar=q_kvar , p_kw=p_kw, **kwargs)



def create_impedance(net, from_bus, to_bus, rft_pu, xft_pu, sn_kva, rtf_pu=None, xtf_pu=None,
                     name=None, in_service=True, index=None):
    """
    Creates an per unit impedance element

    INPUT:
        **net** (pandapowerNet) - The pandapower network in which the element is created

        **from_bus** (int) - starting bus of the impedance

        **to_bus** (int) - ending bus of the impedance

        **r_pu** (float) - real part of the impedance in per unit

        **x_pu** (float) - imaginary part of the impedance in per unit

        **sn_kva** (float) - rated power of the impedance in kVA

    OUTPUT:

        impedance id
    """
    for b in [from_bus, to_bus]:
        if b not in net["bus"].index.values:
            raise UserWarning("Impedance %s tries to attach to non-existing bus %s"% (name, b))

    if index is None:
        index = get_free_id(net.impedance)

    if index in net["impedance"].index:
        raise UserWarning("An impedance with index %s already exists" % index)

        # store dtypes
    dtypes = net.impedance.dtypes
    if rtf_pu is None:
        rtf_pu = rft_pu
    if xtf_pu is None:
        xtf_pu = xft_pu
    net.impedance.loc[index, ["from_bus", "to_bus", "rft_pu", "xft_pu", "rtf_pu", "xtf_pu",
                              "name", "sn_kva", "in_service"]] = \
        [from_bus, to_bus, rft_pu, xft_pu, rtf_pu, xtf_pu, name, sn_kva, in_service]

    # and preserve dtypes
    _preserve_dtypes(net.impedance, dtypes)

    return index


def create_ward(net, bus, ps_kw, qs_kvar, pz_kw, qz_kvar, name=None, in_service=True, index=None):
    """
    Creates a ward equivalent.

    A ward equivalent is a combination of an impedance load and a PQ load.

    INPUT:
        **net** (pandapowernet) - The pandapower net within the element should be created

        **bus** (int) -  bus of the ward equivalent

        **ps_kw** (float) - active power of the PQ load

        **qs_kvar** (float) - reactive power of the PQ load

        **pz_kw** (float) - active power of the impedance load in kW at 1.pu voltage

        **qz_kvar** (float) - reactive power of the impedance load in kVar at 1.pu voltage

    OUTPUT:
        ward id
    """
    if bus not in net["bus"].index.values:
        raise UserWarning("Cannot attach to bus %s, bus does not exist" % bus)

    if index is None:
        index = get_free_id(net.ward)

    if index in net["ward"].index:
        raise UserWarning("A ward equivalent with index %s already exists" % index)

    # store dtypes
    dtypes = net.ward.dtypes

    net.ward.loc[index, ["bus", "ps_kw", "qs_kvar", "pz_kw", "qz_kvar", "name", "in_service"]] = \
        [bus, ps_kw, qs_kvar, pz_kw, qz_kvar, name, in_service]

    # and preserve dtypes
    _preserve_dtypes(net.ward, dtypes)

    return index


def create_xward(net, bus, ps_kw, qs_kvar, pz_kw, qz_kvar, r_ohm, x_ohm, vm_pu, in_service=True,
                 name=None, index=None):
    """
    Creates an extended ward equivalent.

    A ward equivalent is a combination of an impedance load, a PQ load and as voltage source with
    an internal impedance.

    INPUT:
        **net** - The pandapower net within the impedance should be created

        **bus** (int) -  bus of the ward equivalent

        **ps_kw** (float) - active power of the PQ load

        **qs_kvar** (float) - reactive power of the PQ load

        **pz_kw** (float) - active power of the impedance load in kW at 1.pu voltage

        **qz_kvar** (float) - reactive power of the impedance load in kVar at 1.pu voltage

        **vm_pu** (float)

    OUTPUT:
        xward id
    """
    if bus not in net["bus"].index.values:
        raise UserWarning("Cannot attach to bus %s, bus does not exist" % bus)

    if index is None:
        index = get_free_id(net.xward)

    if index in net["xward"].index:
        raise UserWarning("An extended ward equivalent with index %s already exists" % index)

    # store dtypes
    dtypes = net.xward.dtypes

    net.xward.loc[index, ["bus", "ps_kw", "qs_kvar", "pz_kw", "qz_kvar", "r_ohm", "x_ohm", "vm_pu",
                          "name", "in_service"]] = \
        [bus, ps_kw, qs_kvar, pz_kw, qz_kvar, r_ohm, x_ohm, vm_pu, name, in_service]

    # and preserve dtypes
    _preserve_dtypes(net.xward, dtypes)

    return index


def create_dcline(net, from_bus, to_bus, p_kw, loss_percent, loss_kw, vm_from_pu, vm_to_pu,
                  index=None, name=None, max_p_kw=nan, min_q_from_kvar=nan,
                  min_q_to_kvar=nan, max_q_from_kvar=nan, max_q_to_kvar=nan,
                  in_service=True):
    """create_dcline(net, from_bus, to_bus, p_kw, loss_percent, loss_kw, vm_from_pu, vm_to_pu, \
                  index=None, name=None, max_p_kw=nan, min_q_from_kvar=nan, \
                  min_q_to_kvar=nan, max_q_from_kvar=nan, max_q_to_kvar=nan, \
                  in_service=True)
    Creates a dc line.

    INPUT:
        **from_bus** (int) - ID of the bus on one side which the line will be connected with

        **to_bus** (int) - ID of the bus on the other side which the line will be connected with

        **p_kw** - (float) Measurement value. Units are "kW" for P, "kVar" for Q, "p.u." for V,
        "A" for I. Generation is a positive bus power injection, consumption negative.

        **loss_percent** - (float) Standard deviation in the same unit as the measurement.

        **loss_kw** - (int) Index of bus. Determines the position of the measurement for
        line/transformer measurements (bus == from_bus: measurement at from_bus;
        same for to_bus)

        **vm_from_pu** - (int, None) Index of measured element, if element_type is "line" or
        "transformer".

        **vm_to_pu** - (int, None) Index of measured element, if element_type is "line" or
        "transformer".

    OPTIONAL:
        **index** (int, None) - Force a specified ID if it is available. If None, the index one higher than the highest already existing index is selected.

        **name** (str, None) - A custom name for this dc line

        **in_service** (boolean) - True for in_service or False for out of service

    OUTPUT:
        **index** (int) - The unique ID of the created element

    EXAMPLE:
        create_dcline(net, from_bus=0, to_bus=1, p_kw=1e4, loss_percent=1.2, loss_kw=25, vm_from_pu=1.01, vm_to_pu=1.02)
    """
    for bus in [from_bus, to_bus]:
        if bus not in net["bus"].index.values:
            raise UserWarning("Cannot attach to bus %s, bus does not exist" % bus)

        if bus in net.ext_grid.bus.values:
            raise UserWarning("There is already an external grid at bus %u, only one voltage " +
                              "controlling element (ext_grid, gen) is allowed per bus." % bus)

        if bus in net.gen.bus.values:
            raise UserWarning("There is already a generator at bus %u, only one voltage " +
                              "controlling element (ext_grid, gen) is allowed per bus." % bus)

    if index is None:
        index = get_free_id(net["dcline"])

    if index in net["dcline"].index:
        raise UserWarning("A dcline with the id %s already exists" % index)

    # store dtypes
    dtypes = net.dcline.dtypes

    net.dcline.loc[index, ["name", "from_bus", "to_bus", "p_kw", "loss_percent", "loss_kw",
                           "vm_from_pu", "vm_to_pu",  "max_p_kw", "min_q_from_kvar",
                           "min_q_to_kvar", "max_q_from_kvar", "max_q_to_kvar", "in_service"]]\
        = [name, from_bus, to_bus, p_kw, loss_percent, loss_kw, vm_from_pu, vm_to_pu,
           max_p_kw, min_q_from_kvar, min_q_to_kvar, max_q_from_kvar, max_q_to_kvar,  in_service]

    # and preserve dtypes
    _preserve_dtypes(net.dcline, dtypes)

    return index


def create_measurement(net, type, element_type, value, std_dev, bus, element=None,
                       check_existing=True, index=None, name=None):
    """
    Creates a measurement, which is used by the estimation module. Possible types of measurements
    are: v, p, q, i

    INPUT:
        **type** (string) - Type of measurement. "v", "p", "q", "i" are possible.

        **element_type** (string) - Clarifies which element is measured. "bus", "line",
        "transformer" are possible.

        **value** (float) - Measurement value. Units are "kW" for P, "kVar" for Q, "p.u." for V,
        "A" for I. Generation is a positive bus power injection, consumption negative.

        **std_dev** (float) - Standard deviation in the same unit as the measurement.

        **bus** (int) - Index of bus. Determines the position of the measurement for
        line/transformer measurements (bus == from_bus: measurement at from_bus;
        same for to_bus)

        **element** (int, None) - Index of measured element, if element_type is "line" or
        "transformer".

    OPTIONAL:
        **check_existing** (bool) - Check for and replace existing measurements for this bus and
        type. Set it to false for performance improvements which can cause unsafe behaviour.

        **name** (str, None) - name of measurement.

    OUTPUT:
        (int) Index of measurement

    EXAMPLE:
        500 kW load measurement with 10 kW standard deviation on bus 0:
        create_measurement(net, "p", "bus", -500., 10., 0)
    """

    if bus not in net["bus"].index.values:
        raise UserWarning("Bus %s does not exist" % bus)

    if element is None and element_type in ["line", "transformer"]:
        raise UserWarning("The element type %s requires a value in 'element'" % element_type)

    if element is not None and element_type == "line" and element not in net["line"].index.values:
        raise UserWarning("Line %s does not exist" % element)

    if element is not None and element_type == "transformer" and element not in \
            net["trafo"].index.values:
        raise UserWarning("Transformer %s does not exist" % element)

    if index is None:
        index = get_free_id(net.measurement)

    if index in net["measurement"].index:
        raise UserWarning("A measurement with index %s already exists" % index)

    if type == "i" and element_type == "bus":
        raise UserWarning("Line current measurements cannot be placed at buses")

    if type == "v" and element_type in ["line", "transformer"]:
        raise UserWarning("Voltage measurements can only be placed at buses, not at %s"
                          % element_type)

    if check_existing:
        if element is None:
            existing = net.measurement[(net.measurement.type == type) &
                                       (net.measurement.bus == bus) &
                                       (pd.isnull(net.measurement.element))].index
        else:
            existing = net.measurement[(net.measurement.type == type) &
                                       (net.measurement.bus == bus) &
                                       (net.measurement.element == element)].index
        if len(existing) == 1:
            index = existing[0]
        elif len(existing) > 1:
            raise UserWarning("More than one measurement of this type exists")

    dtypes = net.measurement.dtypes
    net.measurement.loc[index] = [name, type.lower(), element_type, value, std_dev, bus, element]
    _preserve_dtypes(net.measurement, dtypes)
    return index


def create_piecewise_linear_cost(net, element, element_type, data_points, type="p", index=None):
    """
    Creates an entry for piecewise linear costs for an element. The currently supported elements are
     - Generator
     - External Grid
     - Static Generator
     - Load
     - Dcline

    INPUT:
        **element** (int) - ID of the element in the respective element table

        **element_type** (string) - Type of element ["gen", "sgen", "ext_grid", "load", "dcline"] are possible

        **data_points** - (numpy array) Numpy array containing n data points (see example)

    OPTIONAL:
        **type** - (string) - Type of cost ["p", "q"] are allowed

        **index** (int, index) - Force a specified ID if it is available. If None, the index one higher than the highest already existing index is selected.

    OUTPUT:
        **index** (int) - The unique ID of created cost entry

    EXAMPLE:
        create_piecewise_linear_cost(net, 0, "load", np.array([[0, 0], [75, 50], [150, 100]]))

    NOTE:
      costs for reactive power can only be quadratic, linear or constant. No higher grades supported.
    """

    if index is None:
        index = get_free_id(net["piecewise_linear_cost"])

    if index in net["piecewise_linear_cost"].index:
        raise UserWarning("A piecewise_linear_cost with the id %s already exists" % index)

    if not net["polynomial_cost"].loc[
            (net["polynomial_cost"].element_type == element_type) &
            (net["polynomial_cost"].element == element) &
            (net["polynomial_cost"].type == type)].empty:
        raise UserWarning("A polynomial_cost for %s with index %s already exists" %
                          (element_type, element))

    if not net["piecewise_linear_cost"].loc[
            (net["piecewise_linear_cost"].element_type == element_type) &
            (net["piecewise_linear_cost"].element == element) &
            (net["piecewise_linear_cost"].type == type)].empty:
        raise UserWarning("A piecewise_linear_cost for %s with index %s already exists" %
                          (element_type, element))

    p = data_points[:, 0]
    f = data_points[:, 1]

    if not (p[:-1] < p[1:]).all():
        raise ValueError("Piecewise linear costs need to be defined in ascending order: " +
                         "p0 < p1 < ... < pn")

    if element_type != 'dcline':
        if type == "p":
            if not (hasattr(net[element_type], "max_p_kw") and hasattr(net[element_type],
                                                                       "min_p_kw")):
                raise AttributeError("No operational constraints defined for controllable element!")
            if not (net[element_type].max_p_kw.at[element] <= max(p) and
                    net[element_type].min_p_kw.at[element] >= min(p)):
                raise ValueError("Cost function must be defined for whole power range of the "
                                 "generator")
        if type == "q":
            if not (hasattr(net[element_type], "max_q_kvar") or hasattr(net[element_type],
                                                                        "min_q_kvar")):
                raise AttributeError("No operational constraints defined!")
            if not (net[element_type].max_q_kvar.at[element] <= max(p) and net[
                    element_type].min_q_kvar.at[element] >= min(p)):
                raise ValueError("Cost function must be defined for whole power range of the "
                                 "generator")
    else:
        if type == "p":
            if not (hasattr(net[element_type], "max_p_kw")):
                raise AttributeError("No operational constraints defined for controllable element!")
            if not (net[element_type].max_p_kw.at[element] <= max(p)):
                raise ValueError("Cost function must be defined for whole power range of the "
                                 "generator")
        if type == "q":
            if not pd.Series([
                "max_q_to_kvar", "max_q_from_kvar", "min_q_to_kvar", "min_q_from_kvar"]).isin(
                 net[element_type].columns).all():
                raise AttributeError("No operational constraints defined!")
            if not (net[element_type].max_q_to_kvar.at[element] <= max(p) and
                    net[element_type].max_q_from_kvar.at[element] <= max(p) and
                    net[element_type].min_q_to_kvar.at[element] <= min(p) and
                    net[element_type].min_q_from_kvar.at[element] >= min(p)):
                raise ValueError("Cost function must be defined for whole power range of the "
                                 "generator")

    net.piecewise_linear_cost.loc[index, ["type", "element", "element_type"]] = \
        [type, element, element_type]

    net.piecewise_linear_cost.p.loc[index] = p.reshape((1, -1))
    net.piecewise_linear_cost.f.loc[index] = f.reshape((1, -1))

    return index


def create_polynomial_cost(net, element, element_type, coefficients, type="p", index=None):
    """
    Creates an entry for polynomial costs for an element. The currently supported elements are
     - Generator
     - External Grid
     - Static Generator
     - Load
     - Dcline

    INPUT:
        **element** (int) - ID of the element in the respective element table

        **element_type** (string) - Type of element ["gen", "sgen", "ext_grid", "load", "dcline"] are possible

        **data_points** - (numpy array) Numpy array containing n cost coefficients (see example)

        **type ** -"p" or "q"

    OPTIONAL:
        **type** - (string) - Type of cost ["p", "q"] are allowed

        **index** (int, None) - Force a specified ID if it is available. If None, the index one higher than the highest already existing index is selected.

    OUTPUT:
        **index** (int) - The unique ID of created cost entry

    EXAMPLE:
        create_polynomial_cost(net, 0, "gen", np.array([0, 1, 0]))
    """

    if index is None:
        index = get_free_id(net["polynomial_cost"])

    if index in net["polynomial_cost"].index:
        raise UserWarning("A polynomial_cost with the id %s already exists" % index)

    if not net["polynomial_cost"].loc[
            (net["polynomial_cost"].element_type == element_type) &
            (net["polynomial_cost"].element == element) &
            (net["polynomial_cost"].type == type)].empty:
        raise UserWarning("A polynomial_cost for %s with index %s already exists" %
                          (element_type, element))

    if not net["piecewise_linear_cost"].loc[
            (net["piecewise_linear_cost"].element_type == element_type) &
            (net["piecewise_linear_cost"].element == element) &
            (net["piecewise_linear_cost"].type == type)].empty:
        raise UserWarning("A piecewise_linear_cost for %s with index %s already exists" %
                          (element_type, element))
# =======
#     typecosts=  net["polynomial_cost"][net["polynomial_cost"].element_type == element_type]
#
#     if not typecosts[typecosts.type == type].loc[typecosts.element == element].empty:
#         raise UserWarning("A polynomial_cost for this element already exists")
# >>>>>>> Need to commit for merging my changes to make_objective with steffens changes.

    net.polynomial_cost.loc[index, ["type", "element", "element_type"]] = \
        [type, element, element_type]

    net.polynomial_cost.c.loc[index] = coefficients.reshape((1, -1))

    return index