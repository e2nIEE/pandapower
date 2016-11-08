# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

import numpy as np
import pandas as pd

from pandapower.std_types import add_basic_std_types, load_std_type
from pandapower.auxiliary import PandapowerNet, get_free_id, _preserve_dtypes
from pandapower.run import reset_results


def create_empty_network(name=None, f_hz=50.):
    """
    This function initializes the pandapower datastructure.

    OPTIONAL:

        **f_hz** (float, 50.) - power system frequency in hertz

        **name** (string, None) - name for the network

    RETURN:

        **net** (attrdict) - PANDAPOWER attrdict with empty tables:

            - bus
            - ext_grid
            - gen
            - impedance
            - line
            - load
            - sgen
            - shunt
            - trafo
            - trafo3w
            - ward
            - xward

    EXAMPLE:

        net = create_empty_network()

    """
    net = PandapowerNet({
        # structure data
        "bus": [('name', np.dtype(object)),
                ('vn_kv', 'f8'),
                ('type', np.dtype(object)),
                ('zone', np.dtype(object)),
                ('in_service', 'bool'), ],
        "load": [("name", np.dtype(object)),
                 ("bus", "u4"),
                 ("p_kw", "f8"),
                 ("q_kvar", "f8"),
                 ("sn_kva", "f8"),
                 ("scaling", "f8"),
                 ("in_service", 'bool'),
                 ("type", np.dtype(object))],
        "sgen": [("name", np.dtype(object)),
                 ("bus", "i8"),
                 ("p_kw", "f8"),
                 ("q_kvar", "f8"),
                 ("sn_kva", "f8"),
                 ("scaling", "f8"),
                 ("in_service", 'bool'),
                 ("type", np.dtype(object))],
        "gen": [("name", np.dtype(object)),
                ("bus", "u4"),
                ("p_kw", "f8"),
                ("vm_pu", "f8"),
                ("sn_kva", "f8"),
                ("min_q_kvar", "f8"),
                ("max_q_kvar", "f8"),
                ("scaling", "f8"),
                ("in_service", 'bool'),
                ("type", np.dtype(object))],
        "switch": [("bus", "i8"),
                   ("element", "i8"),
                   ("et", np.dtype(object)),
                   ("type", np.dtype(object)),
                   ("closed", "bool"),
                   ("name", np.dtype(object))],
        "shunt": [("bus", "u4"),
                  ("name", np.dtype(object)),
                  ("q_kvar", "f8"),
                  ("p_kw", "f8"),
                  ("in_service", "i8")],
        "ext_grid": [("name", np.dtype(object)),
                     ("bus", "u4"),
                     ("vm_pu", "f8"),
                     ("va_degree", "f8"),
                     ("in_service", 'bool')],
        "line": [("name", np.dtype(object)),
                 ("std_type", np.dtype(object)),
                 ("from_bus", "u4"),
                 ("to_bus", "u4"),
                 ("length_km", "f8"),
                 ("r_ohm_per_km", "f8"),
                 ("x_ohm_per_km", "f8"),
                 ("c_nf_per_km", "f8"),
                 ("imax_ka", "f8"),
                 ("df", "f8"),
                 ("parallel", "u4"),
                 ("type", np.dtype(object)),
                 ("in_service", 'bool')],
        "trafo": [("name", np.dtype(object)),
                  ("std_type", np.dtype(object)),
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
                  ("tp_side", np.dtype(object)),
                  ("tp_mid", "i4"),
                  ("tp_min", "i4"),
                  ("tp_max", "i4"),
                  ("tp_st_percent", "f8"),
                  ("tp_pos", "i4"),
                  ("in_service", 'bool')],
        "trafo3w": [("name", np.dtype(object)),
                    ("std_type", np.dtype(object)),
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
                    ("tp_side", np.dtype(object)),
                    ("tp_mid", "i4"),
                    ("tp_min", "i4"),
                    ("tp_max", "i4"),
                    ("tp_st_percent", "f8"),
                    ("tp_pos", "i4"),
                    ("in_service", 'bool')],
        "impedance": [("name", np.dtype(object)),
                      ("from_bus", "u4"),
                      ("to_bus", "u4"),
                      ("r_pu", "f8"),
                      ("x_pu", "f8"),
                      ("sn_kva", "f8"),
                      ("in_service", 'bool')],
        "ward": [("name", np.dtype(object)),
                 ("bus", "u4"),
                 ("ps_kw", "f8"),
                 ("qs_kvar", "f8"),
                 ("qz_kvar", "f8"),
                 ("pz_kw", "f8"),
                 ("in_service", "f8")],
        "xward": [("name", np.dtype(object)),
                  ("bus", "u4"),
                  ("ps_kw", "f8"),
                  ("qs_kvar", "f8"),
                  ("qz_kvar", "f8"),
                  ("pz_kw", "f8"),
                  ("r_ohm", "f8"),
                  ("x_ohm", "f8"),
                  ("vm_pu", "f8"),
                  ("in_service", "f8")],

        # geodata
        "line_geodata": [("coords", np.dtype(object))],
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
                           ("va_degree", "f8")],
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
        "_empty_res_ward": [("p_kw", "f8"),
                            ("q_kvar", "f8"),
                            ("vm_pu", "f8")],
        "_empty_res_xward": [("p_kw", "f8"),
                             ("q_kvar", "f8"),
                             ("vm_pu", "f8")],

        # internal
        "_ppc": None,
        "version": 1.0,
        "converged": False,
        "name": name,
        "f_hz": f_hz
    })
    for s in net:
        if isinstance(net[s], list):
            net[s] = pd.DataFrame(np.zeros(0, dtype=net[s]))
    add_basic_std_types(net)
    reset_results(net)
    return net


def create_bus(net, vn_kv, name=None, index=None, geodata=None, type="b",
               zone=None, in_service=True, max_vm_pu=np.nan,
               min_vm_pu=np.nan, **kwargs):
    """
    Adds one bus in table net["bus"].

    Busses are the nodes of the network that all other elements connect to.

    INPUT:
        **net** (PandapowerNet) - The pandapower network in which the element is created

    OPTIONAL:

        **name** (string, default None) - the name for this bus

        **index** (int, default None) - Force a specified ID if it is available

        **vn_kv** (float, default 0.4) - The grid voltage level.

        **busgeodata** ((x,y)-tuple, default None) - coordinates used for plotting

        **type** (string, default k) - Type of the bus. "n" - auxilary node,
        "b" - busbar, "m" - muff

        **zone** (string, None) - grid region

        **in_service** (boolean) - True for in_service or False for out of service

    OUTPUT:

        **eid** (int) - The index of the created element

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

    if not np.isnan(min_vm_pu):
        if "min_vm_pu" not in net.bus.columns:
            net.bus.loc[:, "min_vm_pu"] = pd.Series()

        net.bus.loc[index, "min_vm_pu"] = float(min_vm_pu)

    if not np.isnan(max_vm_pu):
        if "max_vm_pu" not in net.bus.columns:
            net.bus.loc[:, "max_vm_pu"] = pd.Series()

        net.bus.loc[index, "max_vm_pu"] = float(max_vm_pu)

    return index


def create_buses(net, nr_buses, vn_kv=0.4, index=None, name=None, type="b", geodata=None,
                 zone=None, in_service=True):
    """
    Adds several buses in table net["bus"] at once.

    Busses are the nodal points of the network that all other elements connect to.

    Input:

        **net** (PandapowerNet) - The pandapower network in which the element is created

        **nr_buses** (int) - The number of buses that is created

    OPTIONAL:

        **name** (string, default None) - the name for this bus

        **index** (int, default None) - Force a specified ID if it is available

        **vn_kv** (float, default 0.4) - The grid voltage level.

        **geodata** ((x,y)-tuple, default None) - coordinates used for plotting

        **type** (string, default k) - Type of the bus. "n" - auxilary node,
        "b" - busbar, "m" - muff

        **zone** (string, None) - grid region

        **in_service** (boolean) - True for in_service or False for out of service

    OUTPUT:

        **eid** (int) - The indeces of the created elements

    EXAMPLE:

        create_bus(net, name = "bus1")
    """
    if index:
        for idx in index:
            if idx in net.bus.index:
                raise UserWarning("A bus with index %s already exists" % index)
    else:
        bid = get_free_id(net["bus"])
        index = np.arange(bid, bid + nr_buses, 1)

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

    return index


def create_load(net, bus, p_kw, q_kvar=0, sn_kva=np.nan, name=None, scaling=1., index=None,
                in_service=True, type=None):
    """
    Adds one load in table net["load"].

    All loads are modelled in the consumer system, meaning load is positive and generation is
    negative active power. Please pay attention to the correct signing of the reactive power as well.

    INPUT:
        **net** - The net within this load should be created

        **bus** (int) - The bus id to which the load is connected

    OPTIONAL:

        **p_kw** (float, default 0) - The real power of the load

        **q_kvar** (float, default 0) - The reactive power of the load

        - postive value   -> load
        - negative value  -> generation

        **sn_kva** (float, default None) - Nominal power of the load

        **name** (string, default None) - The name for this load

        **scaling** (float, default 1.) - An OPTIONAL scaling factor to be set customly

        **type** (string, None) -  type variable to classify the load

        **index** (int, None) - Force the specified index. If None, the next highest available index
                                is used

        **in_service** (boolean) - True for in_service or False for out of service

    OUTPUT:

        **index** (int) - The index of the created element

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

    net.load.loc[index, ["name", "bus", "p_kw", "scaling",
                         "q_kvar", "sn_kva", "in_service", "type"]] = \
        [name, bus, p_kw, scaling, q_kvar, sn_kva, bool(in_service), type]

    # and preserve dtypes
    _preserve_dtypes(net.load, dtypes)

    return index


def create_sgen(net, bus, p_kw, q_kvar=0, sn_kva=np.nan, name=None, index=None,
                scaling=1., type=None, in_service=True, max_p_kw=np.nan, min_p_kw=np.nan,
                max_q_kvar=np.nan, min_q_kvar=np.nan, cost_per_kw=np.nan, cost_per_kvar=np.nan,
                controllable=False):
    """
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

    OPTIONAL:

        **p_kw** (float, default 0) - The real power of the static generator (negative for generation!)

        **q_kvar** (float, default 0) - The reactive power of the sgen

        **sn_kva** (float, default None) - Nominal power of the sgen

        **name** (string, default None) - The name for this sgen

        **index** (int, None) - Force the specified index. If None, the next highest available index
                                is used

        **scaling** (float, 1.) - An OPTIONAL scaling factor to be set customly

        **type** (string, None) -  type variable to classify the static generator

        **in_service** (boolean) - True for in_service or False for out of service

    OUTPUT:

        **index** - The unique id of the created sgen

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

    if not np.isnan(min_p_kw):
        if "min_p_kw" not in net.sgen.columns:
            net.sgen.loc[:, "min_p_kw"] = pd.Series()

        net.sgen.loc[index, "min_p_kw"] = float(min_p_kw)

    if not np.isnan(max_p_kw):
        if "max_p_kw" not in net.sgen.columns:
            net.sgen.loc[:, "max_p_kw"] = pd.Series()

        net.sgen.loc[index, "max_p_kw"] = float(max_p_kw)

    if not np.isnan(min_q_kvar):
        if "min_q_kvar" not in net.sgen.columns:
            net.sgen.loc[:, "min_q_kvar"] = pd.Series()

        net.sgen.loc[index, "min_q_kvar"] = float(min_q_kvar)

    if not np.isnan(max_q_kvar):
        if "max_q_kvar" not in net.sgen.columns:
            net.sgen.loc[:, "max_q_kvar"] = pd.Series()

        net.sgen.loc[index, "max_q_kvar"] = float(max_q_kvar)

    if not np.isnan(cost_per_kw):
        if "cost_per_kw" not in net.sgen.columns:
            net.sgen.loc[:, "cost_per_kw"] = pd.Series()

        net.sgen.loc[index, "cost_per_kw"] = float(cost_per_kw)

    if not np.isnan(cost_per_kvar):
        if "cost_per_kvar" not in net.sgen.columns:
            net.sgen.loc[:, "cost_per_kvar"] = pd.Series()

        net.sgen.loc[index, "cost_per_kvar"] = float(cost_per_kvar)

    if controllable:
        if "controllable" not in net.sgen.columns:
            net.sgen.loc[:, "controllable"] = pd.Series()

        net.sgen.loc[index, "controllable"] = bool(controllable)

    return index


def create_gen(net, bus, p_kw, vm_pu=1., sn_kva=np.nan, name=None, index=None, max_q_kvar=np.nan,
               min_q_kvar=np.nan, min_p_kw=np.nan, max_p_kw=np.nan, scaling=1., type=None,
               in_service=True, cost_per_kw=np.nan, cost_per_kvar=np.nan, controllable=False):
    """
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

        **index** (int, None) - Force the specified index. If None, the next highest available index
                                is used

        **scaling** (float, 1.0) - scaling factor which for the active power of the generator

        **type** (string, None) - type variable to classify generators

        **in_service** (boolean) - True for in_service or False for out of service

    OUTPUT:

        **index** - The unique id of the created generator

    EXAMPLE:

        create_gen(net, 1, p_kw = -120, vm_pu = 1.02)

    """
    if bus not in net["bus"].index.values:
        raise UserWarning("Cannot attach to bus %s, bus does not exist" % bus)

    if bus in net.ext_grid.bus.values:
        raise UserWarning(
            "There is already an external grid at bus %u, only one voltage controlling element (ext_grid, gen) is allowed per bus." % bus)

    if bus in net.gen.bus.values:
        raise UserWarning(
            "There is already a generator at bus %u, only one voltage controlling element (ext_grid, gen) is allowed per bus." % bus)

    if index is None:
        index = get_free_id(net["gen"])

    if index in net["gen"].index:
        raise UserWarning("A generator with the id %s already exists" % index)

    # store dtypes
    dtypes = net.gen.dtypes

    net.gen.loc[index, ["name", "bus", "p_kw", "vm_pu", "sn_kva",  "type", "in_service", "scaling"]]\
        = [name, bus, p_kw, vm_pu, sn_kva, type, bool(in_service), scaling]

    # and preserve dtypes
    _preserve_dtypes(net.gen, dtypes)

    if not np.isnan(min_p_kw):
        if "min_p_kw" not in net.gen.columns:
            net.gen.loc[:, "min_p_kw"] = pd.Series()

        net.gen.loc[index, "min_p_kw"] = float(min_p_kw)

    if not np.isnan(max_p_kw):
        if "max_p_kw" not in net.gen.columns:
            net.gen.loc[:, "max_p_kw"] = pd.Series()

        net.gen.loc[index, "max_p_kw"] = float(max_p_kw)

    if not np.isnan(min_q_kvar):
        if "min_q_kvar" not in net.gen.columns:
            net.gen.loc[:, "min_q_kvar"] = pd.Series()

        net.gen.loc[index, "min_q_kvar"] = float(min_q_kvar)

    if not np.isnan(max_q_kvar):
        if "max_q_kvar" not in net.gen.columns:
            net.gen.loc[:, "max_q_kvar"] = pd.Series()

        net.gen.loc[index, "max_q_kvar"] = float(max_q_kvar)

    if not np.isnan(cost_per_kw):
        if "cost_per_kw" not in net.gen.columns:
            net.gen.loc[:, "cost_per_kw"] = pd.Series()

        net.gen.loc[index, "cost_per_kw"] = float(cost_per_kw)

    if not np.isnan(cost_per_kvar):
        if "cost_per_kvar" not in net.gen.columns:
            net.gen.loc[:, "cost_per_kvar"] = pd.Series()

        net.gen.loc[index, "cost_per_kvar"] = float(cost_per_kvar)

    if controllable:
        if "controllable" not in net.gen.columns:
            net.gen.loc[:, "controllable"] = pd.Series()

        net.gen.loc[index, "controllable"] = bool(controllable)

    return index


def create_ext_grid(net, bus, vm_pu=1.0, va_degree=0., name=None, in_service=True,
                    s_sc_max_mva=np.nan, s_sc_min_mva=np.nan, rx_max=np.nan, rx_min=np.nan,
                    index=None, cost_per_kw=np.nan, cost_per_kvar=np.nan):
    """
    Creates an external grid connection.

    External grids represent the higher level power grid connection and are modelled as the slack
    bus in the power flow calculation.

    INPUT:
        **net** - pandapower network

        **bus** (int) - bus where the slack is connected

    OPTIONAL:

        **vm_pu** (float, default 1.0) - voltage at the slack node in per unit

        **va_degree** (float, default 0.) - name of of the external grid*

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
    if index and index in net["ext_grid"].index:
        raise UserWarning("An external grid with with index %s already exists" % index)

    if index is None:
        index = get_free_id(net["ext_grid"])

    if bus in net.ext_grid.bus.values:
        raise UserWarning(
            "There is already an external grid at bus %u, only one voltage controlling element (ext_grid, gen) is allowed per bus." % bus)

    if bus in net.gen.bus.values:
        raise UserWarning(
            "There is already a generator at bus %u, only one voltage controlling element (ext_grid, gen) is allowed per bus." % bus)

        # store dtypes
    dtypes = net.ext_grid.dtypes

    net.ext_grid.loc[index, ["bus", "name", "vm_pu", "va_degree", "in_service"]] = \
        [bus, name, vm_pu, va_degree, bool(in_service)]

    if not np.isnan(s_sc_max_mva):
        if "s_sc_max_mva" not in net.ext_grid.columns:
            net.ext_grid.loc[:, "s_sc_max_mva"] = pd.Series()

        net.ext_grid.at[:, "s_sc_max_mva"] = float(s_sc_max_mva)

    if not np.isnan(s_sc_min_mva):
        if "s_sc_min_mva" not in net.ext_grid.columns:
            net.ext_grid.loc[:, "s_sc_min_mva"] = pd.Series()

        net.ext_grid.at[index, "s_sc_min_mva"] = float(s_sc_min_mva)

    if not np.isnan(rx_min):
        if "rx_min" not in net.ext_grid.columns:
            net.ext_grid.loc[:, "rx_min"] = pd.Series()

        net.ext_grid.at[index, "rx_min"] = float(rx_min)

    if not np.isnan(rx_max):
        if "rx_max" not in net.ext_grid.columns:
            net.ext_grid.loc[:, "rx_max"] = pd.Series()

        net.ext_grid.at[index, "rx_max"] = float(rx_max)

    if not np.isnan(cost_per_kw):
        if "cost_per_kw" not in net.ext_grid.columns:
            net.ext_grid.loc[:, "cost_per_kw"] = pd.Series()

        net.ext_grid.loc[index, "cost_per_kw"] = float(cost_per_kw)

    if not np.isnan(cost_per_kvar):
        if "cost_per_kvar" not in net.ext_grid.columns:
            net.ext_grid.loc[:, "cost_per_kvar"] = pd.Series()

        net.ext_grid.loc[index, "cost_per_kvar"] = float(cost_per_kvar)

        # and preserve dtypes
    _preserve_dtypes(net.ext_grid, dtypes)
    return index


def create_line(net, from_bus, to_bus, length_km, std_type, name=None, index=None, geodata=None,
                df=1., parallel=1, in_service=True, max_loading_percent=np.nan):
    """
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

        **index** (int) - Force a specified ID if it is available

        **geodata**
        (np.array, default None, shape= (,2L)) -
        The linegeodata of the line. The first row should be the coordinates
        of bus a and the last should be the coordinates of bus b. The points
        in the middle represent the bending points of the line

        **in_service** (boolean) - True for in_service or False for out of service

        **df** (float) - derating factor: maximal current of line  in relation to nominal current of line (from 0 to 1)

        **parallel** (integer) - number of parallel line systems

    OUTPUT:

        **line_id** - The unique line_id of the created line

    EXAMPLE:

        create_line(net, "line1", from_bus = 0, to_bus = 1, length_km=0.1,  std_type="NAYY 4x50 SE")

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
        "to_bus": to_bus, "in_service": bool(in_service), "std_type": std_type,
        "df": df, "parallel": parallel
    }

    lineparam = load_std_type(net, std_type, "line")
    v.update({
        "r_ohm_per_km": lineparam["r_ohm_per_km"],
        "x_ohm_per_km": lineparam["x_ohm_per_km"],
        "c_nf_per_km": lineparam["c_nf_per_km"],
        "imax_ka": lineparam["imax_ka"]
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

    if not np.isnan(max_loading_percent):
        if "max_loading_percent" not in net.line.columns:
            net.line.loc[:, "max_loading_percent"] = pd.Series()

        net.line.loc[index, "max_loading_percent"] = float(max_loading_percent)

    return index


def create_line_from_parameters(net, from_bus, to_bus, length_km, r_ohm_per_km, x_ohm_per_km,
                                c_nf_per_km, imax_ka, name=None, index=None, type=None,
                                geodata=None, in_service=True, df=1., parallel=1,
                                max_loading_percent=np.nan, **kwargs):
    """
    Creates a line element in net["line"] from line parameters.

    INPUT:
        **net** - The net within this line should be created

        **from_bus** (int) - ID of the bus on one side which the line will be connected with

        **to_bus** (int) - ID of the bus on the other side which the line will be connected with

        **length_km** (float) - The line length in km

        **r_ohm_per_km** (float) - line resistance in ohm per km

        **x_ohm_per_km** (float) - line reactance in ohm per km

        **c_nf_per_km** (float) - line capacitance in nF per km

        **imax_ka** (float) - maximum thermal current in kA


    OPTIONAL:

        **name** (string) - A custom name for this line

        **index** (int) - Force a specified ID if it is available

        **in_service** (boolean) - True for in_service or False for out of service

        **type** (str) - type of line ("oh" for overhead line or "cs" for cable system)

        **df** (float) - derating factor: maximal current of line  in relation to nominal current of line (from 0 to 1)

        **parallel** (integer) - number of parallel line systems

        **geodata**
        (np.array, default None, shape= (,2L)) -
        The linegeodata of the line. The first row should be the coordinates
        of bus a and the last should be the coordinates of bus b. The points
        in the middle represent the bending points of the line

        **kwargs** - nothing to see here, go along

    OUTPUT:

        **line_id** - The unique line_id of the created line

    EXAMPLE:

        create_line_from_parameters(net, "line1", from_bus = 0, to_bus = 1, lenght_km=0.1,
        r_ohm_per_km = .01, x_ohm_per_km = 0.05, c_nf_per_km = 10,
        imax_ka = 0.4)

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
        "c_nf_per_km": c_nf_per_km, "imax_ka": imax_ka, "parallel": parallel, "type": type
    }

    # store dtypes
    dtypes = net.line.dtypes

    net.line.loc[index, list(v.keys())] = list(v.values())

    # and preserve dtypes
    _preserve_dtypes(net.line, dtypes)

    if geodata is not None:
        net["line_geodata"].loc[index, "coords"] = geodata

    if not np.isnan(max_loading_percent):
        if "max_loading_percent" not in net.line.columns:
            net.line.loc[:, "max_loading_percent"] = pd.Series()

        net.line.loc[index, "max_loading_percent"] = float(max_loading_percent)

    return index


def create_transformer(net, hv_bus, lv_bus, std_type, name=None, tp_pos=np.nan, in_service=True,
                       index=None, max_loading_percent=np.nan):
    """
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

        **index** (int) - Force a specified ID if it is available

    OUTPUT:

        **trafo_id** - The unique trafo_id of the created transformer

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
        "shift_degree": ti["shift_degree"] if "shift_degree" in ti else 0
    })
    for tp in ("tp_mid", "tp_max", "tp_min", "tp_side", "tp_st_percent"):
        if tp in ti:
            v.update({tp: ti[tp]})

    if ("tp_mid" in v) and (tp_pos is np.nan):
        v["tp_pos"] = v["tp_mid"]

    # store dtypes
    dtypes = net.trafo.dtypes

    net.trafo.loc[index, list(v.keys())] = list(v.values())

    if not np.isnan(max_loading_percent):
        if "max_loading_percent" not in net.trafo.columns:
            net.trafo.loc[:, "max_loading_percent"] = pd.Series()

        net.trafo.loc[index, "max_loading_percent"] = float(max_loading_percent)

    # and preserve dtypes
    _preserve_dtypes(net.trafo, dtypes)

    return index


def create_transformer_from_parameters(net, hv_bus, lv_bus, sn_kva, vn_hv_kv, vn_lv_kv, vscr_percent,
                                       vsc_percent, pfe_kw, i0_percent, shift_degree=0,
                                       tp_side=None, tp_mid=np.nan, tp_max=np.nan,
                                       tp_min=np.nan, tp_st_percent=np.nan, tp_pos=np.nan,
                                       in_service=True, name=None, index=None,
                                       max_loading_percent=np.nan, **kwargs):
    """
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

        **name** (string) - A custom name for this transformer

        **shift_degree** (float) - Angle shift over the transformer*

        **tp_side** (string) - position of tap changer ("hv", "lv")

        **tp_pos** (int, nan) - current tap position of the transformer. Defaults to the medium position (tp_mid)

        **tp_mid** (int, nan) - tap position where the transformer ratio is equal to the ration of the rated voltages

        **tp_max** (int, nan) - maximal allowed tap position

        **tp_min** (int, nan):  minimal allowed tap position

        **tp_st_percent** (int) - tap step in percent

        **index** (int) - Force a specified ID if it is available

        **kwargs** - nothing to see here, go along

        \* only considered in loadflow if calculate_voltage_angles = True

    OUTPUT:

        **trafo_id** - The unique trafo_id of the created transformer

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

    if tp_pos is np.nan:
        tp_pos = tp_mid
    v = {
        "name": name, "hv_bus": hv_bus, "lv_bus": lv_bus,
        "in_service": bool(in_service), "std_type": None, "sn_kva": sn_kva, "vn_hv_kv": vn_hv_kv,
        "vn_lv_kv": vn_lv_kv, "vsc_percent": vsc_percent, "vscr_percent": vscr_percent,
        "pfe_kw": pfe_kw, "i0_percent": i0_percent, "tp_pos": tp_pos, "tp_mid": tp_mid,
        "tp_max": tp_max, "tp_min": tp_min, "shift_degree": shift_degree,
        "tp_side": tp_side, "tp_st_percent": tp_st_percent
    }

    # store dtypes
    dtypes = net.trafo.dtypes

    net.trafo.loc[index, list(v.keys())] = list(v.values())

    # and preserve dtypes
    _preserve_dtypes(net.trafo, dtypes)

    if not np.isnan(max_loading_percent):
        if "max_loading_percent" not in net.trafo.columns:
            net.trafo.loc[:, "max_loading_percent"] = pd.Series()

        net.trafo.loc[index, "max_loading_percent"] = float(max_loading_percent)

    return index


def create_transformer3w(net, hv_bus, mv_bus, lv_bus, std_type, name=None, tp_pos=np.nan,
                         in_service=True, index=None):
    """
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

        **index** (int) - Force a specified ID if it is available

    OUTPUT:

        **trafo_id** - The unique trafo_id of the created transformer

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

    if ("tp_mid" in v) and (tp_pos is np.nan):
        v["tp_pos"] = v["tp_mid"]
    dd = pd.DataFrame(v, index=[index])
    net["trafo3w"] = net["trafo3w"].append(dd).reindex_axis(net["trafo3w"].columns, axis=1)

    return index


def create_transformer3w_from_parameters(net, hv_bus, mv_bus, lv_bus, vn_hv_kv, vn_mv_kv, vn_lv_kv,
                                         sn_hv_kva, sn_mv_kva, sn_lv_kva, vsc_hv_percent, vsc_mv_percent,
                                         vsc_lv_percent, vscr_hv_percent, vscr_mv_percent,
                                         vscr_lv_percent, pfe_kw, i0_percent, shift_mv_degree=0.,
                                         shift_lv_degree=0., tp_side=None, tp_st_percent=np.nan,
                                         tp_pos=np.nan, tp_mid=np.nan, tp_max=np.nan,
                                         tp_min=np.nan, name=None, in_service=True, index=None):
    """
    Adds a three-winding transformer in table net["trafo3w"].

    Input:
        **net** (PandapowerNet) - The net within this transformer should be created

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

        **tp_pos** (int, np.nan) - current tap position of the transformer. Defaults to the medium position (tp_mid)

        **name** (string, None) - Name of the 3-winding transformer

        **in_service** (boolean, True) - True for in_service or False for out of service

        \* only considered in loadflow if calculate_voltage_angles = True
        \**The model currently only supports one tap-changer per 3W Transformer.

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
        
    if tp_pos is np.nan:
        tp_pos = tp_mid
    
    # store dtypes
    dtypes = net.trafo3w.dtypes

    net.trafo3w.loc[index, ["lv_bus", "mv_bus", "hv_bus", "vn_hv_kv", "vn_mv_kv", "vn_lv_kv",
                            "sn_hv_kva", "sn_mv_kva", "sn_lv_kva", "vsc_hv_percent", "vsc_mv_percent",
                            "vsc_lv_percent", "vscr_hv_percent", "vscr_mv_percent", "vscr_lv_percent",
                            "pfe_kw", "i0_percent", "shift_mv_degree", "shift_lv_degree",
                            "tp_side", "tp_st_percent", "tp_pos", "tp_mid", "tp_max",
                            "tp_min", "in_service", "name"]] = \
        [lv_bus, mv_bus, hv_bus, vn_hv_kv, vn_mv_kv, vn_lv_kv,
         sn_hv_kva, sn_mv_kva, sn_lv_kva, vsc_hv_percent, vsc_mv_percent,
         vsc_lv_percent, vscr_hv_percent, vscr_mv_percent, vscr_lv_percent,
         pfe_kw, i0_percent, shift_mv_degree, shift_lv_degree,
         tp_side, tp_st_percent, tp_pos, tp_mid, tp_max,
         tp_min, bool(in_service), name]

    # and preserve dtypes
    _preserve_dtypes(net.trafo3w, dtypes)

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
        **net** (PandapowerNet) - The net within this transformer should be created

        **bus** - The bus that the switch is connected to

        **element** - index of the element: bus id if et == "b", line id if et == "l"

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
        elm_tab = 'trafo3w'
        if element not in net[elm_tab].index:
            raise UserWarning("Unknown trafo3w index")
        if (not net[elm_tab]["hv_bus"].loc[element] == bus and
                not net[elm_tab]["mv_bus"].loc[element] == bus and
                not net[elm_tab]["lv_bus"].loc[element] == bus):
            raise UserWarning("Trafo3w %s not connected to bus %s" % (element, bus))
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


def create_shunt(net, bus, q_kvar, p_kw=0., name=None, in_service=True, index=None):
    """
    Creates a shunt element

    INPUT:
        **net** (PandapowerNet) - The pandapower network in which the element is created

        **bus** - bus number of bus to whom the shunt is connected to

        **p_kw** - shunt active power in kW at v= 1.0 p.u.

        **q_kvar** - shunt susceptance in kVAr at v= 1.0 p.u.

    OPTIONAL:

        **name** (str, None) - element name

        **in_service** (boolean, True) - True for in_service or False for out of service

    OUTPUT:

        shunt id

    EXAMPLE:

        create_shunt(net, 0, 20)
    """
    if bus not in net["bus"].index.values:
        raise UserWarning("Cannot attach to bus %s, bus does not exist" % bus)

    if index is None:
        index = get_free_id(net["shunt"])

    if index in net["shunt"].index:
        raise UserWarning("A shunt with index %s already exists" % index)

    # store dtypes
    dtypes = net.shunt.dtypes

    net.shunt.loc[index, ["bus", "name", "p_kw", "q_kvar", "in_service"]] = \
        [bus, name, p_kw, q_kvar, in_service]

    # and preserve dtypes
    _preserve_dtypes(net.shunt, dtypes)

    return index


def create_impedance(net, from_bus, to_bus, r_pu, x_pu, sn_kva, name=None, in_service=True,
                     index=None):
    """
    Creates an per unit impedance element

    INPUT:
        **net** (PandapowerNet) - The pandapower network in which the element is created

        **from_bus** (int) - starting bus of the impedance

        **to_bus** (int) - ending bus of the impedance

        **r_pu** (float) - real part of the impedance in per unit

        **x_pu** (float) - imaginary part of the impedance in per unit

        **sn_kva** (float) - rated power of the impedance in kVA

    OUTPUT:

        impedance id
    """

    if index is None:
        index = get_free_id(net.impedance)

    if index in net["impedance"].index:
        raise UserWarning("An impedance with index %s already exists" % index)

        # store dtypes
    dtypes = net.impedance.dtypes

    net.impedance.loc[index, ["from_bus", "to_bus", "r_pu", "x_pu", "name",
                              "sn_kva", "in_service"]] = \
        [from_bus, to_bus, r_pu, x_pu, name, sn_kva, in_service]

    # and preserve dtypes
    _preserve_dtypes(net.impedance, dtypes)

    return index


def create_ward(net, bus, ps_kw, qs_kvar, pz_kw, qz_kvar, name=None, in_service=True, index=None):
    """
    Creates a ward equivalent.

    A ward equivalent is a combination of an impedance load and a PQ load.

    INPUT:
        **net** (Pandapowernet) - The pandapower net within the element should be created

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


if __name__ == "__main__":
    net = create_empty_network()
