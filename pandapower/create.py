# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pandas as pd
from numpy import nan, isnan, arange, dtype, zeros
from packaging import version

from pandapower.auxiliary import pandapowerNet, get_free_id, _preserve_dtypes
from pandapower.results import reset_results
from pandapower.std_types import add_basic_std_types, load_std_type
from pandapower import __version__


def create_empty_network(name="", f_hz=50., sn_mva=1, add_stdtypes=True):
    """
    This function initializes the pandapower datastructure.

    OPTIONAL:
        **f_hz** (float, 50.) - power system frequency in hertz

        **name** (string, None) - name for the network

        **sn_mva** (float, 1e3) - reference apparent power for per unit system

        **add_stdtypes** (boolean, True) - Includes standard types to net

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
                 ("p_mw", "f8"),
                 ("q_mvar", "f8"),
                 ("const_z_percent", "f8"),
                 ("const_i_percent", "f8"),
                 ("sn_mva", "f8"),
                 ("scaling", "f8"),
                 ("in_service", 'bool'),
                 ("type", dtype(object))],
        "sgen": [("name", dtype(object)),
                 ("bus", "i8"),
                 ("p_mw", "f8"),
                 ("q_mvar", "f8"),
                 ("sn_mva", "f8"),
                 ("scaling", "f8"),
                 ("in_service", 'bool'),
                 ("type", dtype(object)),
                 ("current_source", "bool")],
        "storage": [("name", dtype(object)),
                    ("bus", "i8"),
                    ("p_mw", "f8"),
                    ("q_mvar", "f8"),
                    ("sn_mva", "f8"),
                    ("soc_percent", "f8"),
                    ("min_e_mwh", "f8"),
                    ("max_e_mwh", "f8"),
                    ("scaling", "f8"),
                    ("in_service", 'bool'),
                    ("type", dtype(object))],
        "gen": [("name", dtype(object)),
                ("bus", "u4"),
                ("p_mw", "f8"),
                ("vm_pu", "f8"),
                ("sn_mva", "f8"),
                ("min_q_mvar", "f8"),
                ("max_q_mvar", "f8"),
                ("scaling", "f8"),
                ("slack", "bool"),
                ("in_service", 'bool'),
                ("type", dtype(object))],
        "switch": [("bus", "i8"),
                   ("element", "i8"),
                   ("et", dtype(object)),
                   ("type", dtype(object)),
                   ("closed", "bool"),
                   ("name", dtype(object)),
                   ("z_ohm", "f8")],
        "shunt": [("bus", "u4"),
                  ("name", dtype(object)),
                  ("q_mvar", "f8"),
                  ("p_mw", "f8"),
                  ("vn_kv", "f8"),
                  ("step", "u4"),
                  ("max_step", "u4"),
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
                 ("g_us_per_km", "f8"),
                 ("max_i_ka", "f8"),
                 ("df", "f8"),
                 ("parallel", "u4"),
                 ("type", dtype(object)),
                 ("in_service", 'bool')],
        "trafo": [("name", dtype(object)),
                  ("std_type", dtype(object)),
                  ("hv_bus", "u4"),
                  ("lv_bus", "u4"),
                  ("sn_mva", "f8"),
                  ("vn_hv_kv", "f8"),
                  ("vn_lv_kv", "f8"),
                  ("vk_percent", "f8"),
                  ("vkr_percent", "f8"),
                  ("pfe_kw", "f8"),
                  ("i0_percent", "f8"),
                  ("shift_degree", "f8"),
                  ("tap_side", dtype(object)),
                  ("tap_neutral", "i4"),
                  ("tap_min", "i4"),
                  ("tap_max", "i4"),
                  ("tap_step_percent", "f8"),
                  ("tap_step_degree", "f8"),
                  ("tap_pos", "i4"),
                  ("tap_phase_shifter", 'bool'),
                  ("parallel", "u4"),
                  ("df", "f8"),
                  ("in_service", 'bool')],
        "trafo3w": [("name", dtype(object)),
                    ("std_type", dtype(object)),
                    ("hv_bus", "u4"),
                    ("mv_bus", "u4"),
                    ("lv_bus", "u4"),
                    ("sn_hv_mva", "f8"),
                    ("sn_mv_mva", "f8"),
                    ("sn_lv_mva", "f8"),
                    ("vn_hv_kv", "f8"),
                    ("vn_mv_kv", "f8"),
                    ("vn_lv_kv", "f8"),
                    ("vk_hv_percent", "f8"),
                    ("vk_mv_percent", "f8"),
                    ("vk_lv_percent", "f8"),
                    ("vkr_hv_percent", "f8"),
                    ("vkr_mv_percent", "f8"),
                    ("vkr_lv_percent", "f8"),
                    ("pfe_kw", "f8"),
                    ("i0_percent", "f8"),
                    ("shift_mv_degree", "f8"),
                    ("shift_lv_degree", "f8"),
                    ("tap_side", dtype(object)),
                    ("tap_neutral", "i4"),
                    ("tap_min", "i4"),
                    ("tap_max", "i4"),
                    ("tap_step_percent", "f8"),
                    ("tap_step_degree", "f8"),
                    ("tap_pos", "i4"),
                    ("tap_at_star_point", 'bool'),
                    ("in_service", 'bool')],
        "impedance": [("name", dtype(object)),
                      ("from_bus", "u4"),
                      ("to_bus", "u4"),
                      ("rft_pu", "f8"),
                      ("xft_pu", "f8"),
                      ("rtf_pu", "f8"),
                      ("xtf_pu", "f8"),
                      ("sn_mva", "f8"),
                      ("in_service", 'bool')],
        "dcline": [("name", dtype(object)),
                   ("from_bus", "u4"),
                   ("to_bus", "u4"),
                   ("p_mw", "f8"),
                   ("loss_percent", 'f8'),
                   ("loss_mw", 'f8'),
                   ("vm_from_pu", "f8"),
                   ("vm_to_pu", "f8"),
                   ("max_p_mw", "f8"),
                   ("min_q_from_mvar", "f8"),
                   ("min_q_to_mvar", "f8"),
                   ("max_q_from_mvar", "f8"),
                   ("max_q_to_mvar", "f8"),
                   ("in_service", 'bool')],
        "ward": [("name", dtype(object)),
                 ("bus", "u4"),
                 ("ps_mw", "f8"),
                 ("qs_mvar", "f8"),
                 ("qz_mvar", "f8"),
                 ("pz_mw", "f8"),
                 ("in_service", "bool")],
        "xward": [("name", dtype(object)),
                  ("bus", "u4"),
                  ("ps_mw", "f8"),
                  ("qs_mvar", "f8"),
                  ("qz_mvar", "f8"),
                  ("pz_mw", "f8"),
                  ("r_ohm", "f8"),
                  ("x_ohm", "f8"),
                  ("vm_pu", "f8"),
                  ("in_service", "bool")],
        "measurement": [("name", dtype(object)),
                        ("measurement_type", dtype(object)),
                        ("element_type", dtype(object)),
                        ("element", "uint32"),
                        ("value", "float64"),
                        ("std_dev", "float64"),
                        ("side", dtype(object))],
        "pwl_cost": [("power_type", dtype(object)),
                     ("element", dtype(object)),
                     ("et", dtype(object)),
                     ("points", dtype(object))],
        "poly_cost": [("element", dtype(object)),
                      ("et", dtype(object)),
                      ("cp0_eur", dtype("f8")),
                      ("cp1_eur_per_mw", dtype("f8")),
                      ("cp2_eur_per_mw2", dtype("f8")),
                      ("cq0_eur", dtype("f8")),
                      ("cq1_eur_per_mvar", dtype("f8")),
                      ("cq2_eur_per_mvar2", dtype("f8"))
                      ],
        # geodata
        "line_geodata": [("coords", dtype(object))],
        "bus_geodata": [("x", "f8"), ("y", "f8"), ("coords", dtype(object))],

        # result tables
        "_empty_res_bus": [("vm_pu", "f8"),
                           ("va_degree", "f8"),
                           ("p_mw", "f8"),
                           ("q_mvar", "f8")],
        "_empty_res_ext_grid": [("p_mw", "f8"),
                                ("q_mvar", "f8")],
        "_empty_res_line": [("p_from_mw", "f8"),
                            ("q_from_mvar", "f8"),
                            ("p_to_mw", "f8"),
                            ("q_to_mvar", "f8"),
                            ("pl_mw", "f8"),
                            ("ql_mvar", "f8"),
                            ("i_from_ka", "f8"),
                            ("i_to_ka", "f8"),
                            ("i_ka", "f8"),
                            ("vm_from_pu", "f8"),
                            ("va_from_degree", "f8"),
                            ("vm_to_pu", "f8"),
                            ("va_to_degree", "f8"),
                            ("loading_percent", "f8")],
        "_empty_res_trafo": [("p_hv_mw", "f8"),
                             ("q_hv_mvar", "f8"),
                             ("p_lv_mw", "f8"),
                             ("q_lv_mvar", "f8"),
                             ("pl_mw", "f8"),
                             ("ql_mvar", "f8"),
                             ("i_hv_ka", "f8"),
                             ("i_lv_ka", "f8"),
                             ("vm_hv_pu", "f8"),
                             ("va_hv_degree", "f8"),
                             ("vm_lv_pu", "f8"),
                             ("va_lv_degree", "f8"),
                             ("loading_percent", "f8")],
        "_empty_res_trafo3w": [("p_hv_mw", "f8"),
                               ("q_hv_mvar", "f8"),
                               ("p_mv_mw", "f8"),
                               ("q_mv_mvar", "f8"),
                               ("p_lv_mw", "f8"),
                               ("q_lv_mvar", "f8"),
                               ("pl_mw", "f8"),
                               ("ql_mvar", "f8"),
                               ("i_hv_ka", "f8"),
                               ("i_mv_ka", "f8"),
                               ("i_lv_ka", "f8"),
                               ("vm_hv_pu", "f8"),
                               ("va_hv_degree", "f8"),
                               ("vm_mv_pu", "f8"),
                               ("va_mv_degree", "f8"),
                               ("vm_lv_pu", "f8"),
                               ("va_lv_degree", "f8"),
                               ("va_internal_degree", "f8"),
                               ("vm_internal_pu", "f8"),
                               ("loading_percent", "f8")],
        "_empty_res_load": [("p_mw", "f8"),
                            ("q_mvar", "f8")],
        "_empty_res_sgen": [("p_mw", "f8"),
                            ("q_mvar", "f8")],
        "_empty_res_storage": [("p_mw", "f8"),
                               ("q_mvar", "f8")],
        "_empty_res_gen": [("p_mw", "f8"),
                           ("q_mvar", "f8"),
                           ("va_degree", "f8"),
                           ("vm_pu", "f8")],
        "_empty_res_shunt": [("p_mw", "f8"),
                             ("q_mvar", "f8"),
                             ("vm_pu", "f8")],
        "_empty_res_impedance": [("p_from_mw", "f8"),
                                 ("q_from_mvar", "f8"),
                                 ("p_to_mw", "f8"),
                                 ("q_to_mvar", "f8"),
                                 ("pl_mw", "f8"),
                                 ("ql_mvar", "f8"),
                                 ("i_from_ka", "f8"),
                                 ("i_to_ka", "f8")],
        "_empty_res_dcline": [("p_from_mw", "f8"),
                              ("q_from_mvar", "f8"),
                              ("p_to_mw", "f8"),
                              ("q_to_mvar", "f8"),
                              ("pl_mw", "f8"),
                              ("vm_from_pu", "f8"),
                              ("va_from_degree", "f8"),
                              ("vm_to_pu", "f8"),
                              ("va_to_degree", "f8")],
        "_empty_res_ward": [("p_mw", "f8"),
                            ("q_mvar", "f8"),
                            ("vm_pu", "f8")],
        "_empty_res_xward": [("p_mw", "f8"),
                             ("q_mvar", "f8"),
                             ("vm_pu", "f8"),
                             ("va_internal_degree", "f8"),
                             ("vm_internal_pu", "f8")],

        # internal
        "_ppc": None,
        "_is_elements": None,
        "_pd2ppc_lookups": {"bus": None,
                            "ext_grid": None,
                            "gen": None},
        "version": __version__,
        "converged": False,
        "name": name,
        "f_hz": f_hz,
        "sn_mva": sn_mva
    })
    for s in net:
        if isinstance(net[s], list):
            net[s] = pd.DataFrame(zeros(0, dtype=net[s]), index=pd.Int64Index([]))
    if add_stdtypes:
        add_basic_std_types(net)
    else:
        net.std_types = {"line": {}, "trafo": {}, "trafo3w": {}}
    reset_results(net)
    net['user_pf_options'] = dict()
    return net


def create_bus(net, vn_kv, name=None, index=None, geodata=None, type="b",
               zone=None, in_service=True, max_vm_pu=nan,
               min_vm_pu=nan, coords=None, **kwargs):
    """
    Adds one bus in table net["bus"].

    Busses are the nodes of the network that all other elements connect to.

    INPUT:
        **net** (pandapowerNet) - The pandapower network in which the element is created

    OPTIONAL:
        **name** (string, default None) - the name for this bus

        **index** (int, default None) - Force a specified ID if it is available. If None, the \
            index one higher than the highest already existing index is selected.

        **vn_kv** (float) - The grid voltage level.

        **geodata** ((x,y)-tuple, default None) - coordinates used for plotting

        **type** (string, default "b") - Type of the bus. "n" - node,
        "b" - busbar, "m" - muff

        **zone** (string, None) - grid region

        **in_service** (boolean) - True for in_service or False for out of service

        **max_vm_pu** (float, NAN) - Maximum bus voltage in p.u. - necessary for OPF

        **min_vm_pu** (float, NAN) - Minimum bus voltage in p.u. - necessary for OPF

        **coords** (array, default None, shape= (,2L)) - busbar coordinates to plot the bus with multiple points.
            coords is typically a list of tuples (start and endpoint of the busbar) [(x1, y1), (x2, y2)]

    OUTPUT:
        **index** (int) - The unique ID of the created element

    EXAMPLE:
        create_bus(net, name = "bus1")
    """
    if index is not None and index in net["bus"].index:
        raise UserWarning("A bus with index %s already exists" % index)

    if index is None:
        index = get_free_id(net["bus"])

    # store dtypes
    dtypes = net.bus.dtypes

    net.bus.loc[index, ["name", "vn_kv", "type", "zone", "in_service"]] = \
        [name, vn_kv, type, zone, bool(in_service)]

    # and preserve dtypes
    _preserve_dtypes(net.bus, dtypes)

    if geodata is not None:
        if len(geodata) != 2:
            raise UserWarning("geodata must be given as (x, y) tupel")
        net["bus_geodata"].loc[index, ["x", "y"]] = geodata

    if coords is not None:
        net["bus_geodata"].loc[index, "coords"] = coords

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
                 zone=None, in_service=True, max_vm_pu=nan, min_vm_pu=nan, coords=None):
    """
    Adds several buses in table net["bus"] at once.

    Busses are the nodal points of the network that all other elements connect to.

    Input:
        **net** (pandapowerNet) - The pandapower network in which the element is created

        **nr_buses** (int) - The number of buses that is created

    OPTIONAL:
        **name** (string, default None) - the name for this bus

        **index** (int, default None) - Force specified IDs if available. If None, the indices \
            higher than the highest already existing index are selected.

        **vn_kv** (float) - The grid voltage level.

        **geodata** ((x,y)-tuple, default None) - coordinates used for plotting

        **type** (string, default "b") - Type of the bus. "n" - auxilary node,
        "b" - busbar, "m" - muff

        **zone** (string, None) - grid region

        **in_service** (boolean) - True for in_service or False for out of service

        **max_vm_pu** (float, NAN) - Maximum bus voltage in p.u. - necessary for OPF

        **min_vm_pu** (float, NAN) - Minimum bus voltage in p.u. - necessary for OPF

    OUTPUT:
        **index** (int) - The unique indices ID of the created elements

    EXAMPLE:
        create_bus(net, name = "bus1")
    """
    if index is not None:
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
    net["bus"] = net["bus"].append(dd)[net["bus"].columns.tolist()]
    # and preserve dtypes
    # _preserve_dtypes(net.bus, dtypes)

    if geodata is not None:
        # works with a 2-tuple or a matching array
        net.bus_geodata = net.bus_geodata.append(pd.DataFrame(index=index,
                                                              columns=net.bus_geodata.columns))
        net.bus_geodata.loc[index, ["x", "y"]] = geodata
    if coords is not None:
        net.bus_geodata = net.bus_geodata.append(pd.DataFrame(index=index,
                                                              columns=net.bus_geodata.columns))
        net["bus_geodata"].loc[index, "coords"] = coords

    if not isnan(min_vm_pu):
        if "min_vm_pu" not in net.bus.columns:
            net.bus.loc[:, "min_vm_pu"] = pd.Series()

        net.bus.loc[index, "min_vm_pu"] = float(min_vm_pu)

    if not isnan(max_vm_pu):
        if "max_vm_pu" not in net.bus.columns:
            net.bus.loc[:, "max_vm_pu"] = pd.Series()

        net.bus.loc[index, "max_vm_pu"] = float(max_vm_pu)

    return index


def create_load(net, bus, p_mw, q_mvar=0, const_z_percent=0, const_i_percent=0, sn_mva=nan,
                name=None, scaling=1., index=None,
                in_service=True, type=None, max_p_mw=nan, min_p_mw=nan,
                max_q_mvar=nan, min_q_mvar=nan, controllable=nan):
    """
    Adds one load in table net["load"].

    All loads are modelled in the consumer system, meaning load is positive and generation is
    negative active power. Please pay attention to the correct signing of the reactive power as
    well.

    INPUT:
        **net** - The net within this load should be created

        **bus** (int) - The bus id to which the load is connected

    OPTIONAL:
        **p_mw** (float, default 0) - The real power of the load

        - postive value   -> load
        - negative value  -> generation

        **q_mvar** (float, default 0) - The reactive power of the load

        **const_z_percent** (float, default 0) - percentage of p_mw and q_mvar that will be \
            associated to constant impedance load at rated voltage

        **const_i_percent** (float, default 0) - percentage of p_mw and q_mvar that will be \
            associated to constant current load at rated voltage

        **sn_mva** (float, default None) - Nominal power of the load

        **name** (string, default None) - The name for this load

        **scaling** (float, default 1.) - An OPTIONAL scaling factor to be set customly

        **type** (string, None) -  type variable to classify the load

        **index** (int, None) - Force a specified ID if it is available. If None, the index one \
            higher than the highest already existing index is selected.

        **in_service** (boolean) - True for in_service or False for out of service

        **max_p_mw** (float, default NaN) - Maximum active power load - necessary for controllable \
            loads in for OPF

        **min_p_mw** (float, default NaN) - Minimum active power load - necessary for controllable \
            loads in for OPF

        **max_q_mvar** (float, default NaN) - Maximum reactive power load - necessary for \
            controllable loads in for OPF

        **min_q_mvar** (float, default NaN) - Minimum reactive power load - necessary for \
            controllable loads in OPF

        **controllable** (boolean, default NaN) - States, whether a load is controllable or not. \
            Only respected for OPF

    OUTPUT:
        **index** (int) - The unique ID of the created element

    EXAMPLE:
        create_load(net, bus=0, p_mw=10., q_mvar=2.)

    """
    if bus not in net["bus"].index.values:
        raise UserWarning("Cannot attach to bus %s, bus does not exist" % bus)

    if index is None:
        index = get_free_id(net["load"])
    if index in net["load"].index:
        raise UserWarning("A load with the id %s already exists" % index)

    # store dtypes
    dtypes = net.load.dtypes

    net.load.loc[index, ["name", "bus", "p_mw", "const_z_percent", "const_i_percent", "scaling",
                         "q_mvar", "sn_mva", "in_service", "type"]] = \
        [name, bus, p_mw, const_z_percent, const_i_percent, scaling, q_mvar, sn_mva,
         bool(in_service), type]

    # and preserve dtypes
    _preserve_dtypes(net.load, dtypes)

    if not isnan(min_p_mw):
        if "min_p_mw" not in net.load.columns:
            net.load.loc[:, "min_p_mw"] = pd.Series()

        net.load.loc[index, "min_p_mw"] = float(min_p_mw)

    if not isnan(max_p_mw):
        if "max_p_mw" not in net.load.columns:
            net.load.loc[:, "max_p_mw"] = pd.Series()

        net.load.loc[index, "max_p_mw"] = float(max_p_mw)

    if not isnan(min_q_mvar):
        if "min_q_mvar" not in net.load.columns:
            net.load.loc[:, "min_q_mvar"] = pd.Series()

        net.load.loc[index, "min_q_mvar"] = float(min_q_mvar)

    if not isnan(max_q_mvar):
        if "max_q_mvar" not in net.load.columns:
            net.load.loc[:, "max_q_mvar"] = pd.Series()

        net.load.loc[index, "max_q_mvar"] = float(max_q_mvar)

    if not isnan(controllable):
        if "controllable" not in net.load.columns:
            net.load.loc[:, "controllable"] = pd.Series()

        net.load.loc[index, "controllable"] = bool(controllable)
    else:
        if "controllable" in net.load.columns:
            net.load.loc[index, "controllable"] = False

    return index


def create_load_from_cosphi(net, bus, sn_mva, cos_phi, mode, **kwargs):
    """
    Creates a load element from rated power and power factor cos(phi).

    INPUT:
        **net** - The net within this static generator should be created

        **bus** (int) - The bus id to which the load is connected

        **sn_mva** (float) - rated power of the load

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
    p_mw, q_mvar = pq_from_cosphi(sn_mva, cos_phi, qmode=mode, pmode="load")
    return create_load(net, bus, sn_mva=sn_mva, p_mw=p_mw, q_mvar=q_mvar, **kwargs)


def create_sgen(net, bus, p_mw, q_mvar=0, sn_mva=nan, name=None, index=None,
                scaling=1., type=None, in_service=True, max_p_mw=nan, min_p_mw=nan,
                max_q_mvar=nan, min_q_mvar=nan, controllable=nan, k=nan, rx=nan,
                current_source=True):
    """
    Adds one static generator in table net["sgen"].

    Static generators are modelled as positive and constant PQ power. This element is used to model generators
    with a constant active and reactive power feed-in. If you want to model a voltage controlled
    generator, use the generator element instead.

    gen, sgen and ext_grid in the grid are modelled in the generator system!
    If you want to model the generation of power, you have to assign a positive active power
    to the generator. Please pay attention to the correct signing of the
    reactive power as well (positive for injection and negative for consumption).

    INPUT:
        **net** - The net within this static generator should be created

        **bus** (int) - The bus id to which the static generator is connected

        **p_mw** (float) - The real power of the static generator  (positive for generation!)

    OPTIONAL:

        **q_mvar** (float, 0) - The reactive power of the sgen

        **sn_mva** (float, None) - Nominal power of the sgen

        **name** (string, None) - The name for this sgen

        **index** (int, None) - Force a specified ID if it is available. If None, the index one \
            higher than the highest already existing index is selected.

        **scaling** (float, 1.) - An OPTIONAL scaling factor to be set customly

        **type** (string, None) -  type variable to classify the static generator (no impact on \
            calculations)

        **in_service** (boolean) - True for in_service or False for out of service

        **max_p_mw** (float, NaN) - Maximum active power injection - necessary for \
            controllable sgens in OPF

        **min_p_mw** (float, NaN) - Minimum active power injection - necessary for \
            controllable sgens in OPF

        **max_q_mvar** (float, NaN) - Maximum reactive power injection - necessary for \
            controllable sgens in OPF

        **min_q_mvar** (float, NaN) - Minimum reactive power injection - necessary for \
            controllable sgens in OPF

        **controllable** (bool, NaN) - Whether this generator is controllable by the optimal
        powerflow

        **k** (float, NaN) - Ratio of nominal current to short circuit current

        **rx** (float, NaN) - R/X ratio for short circuit impedance. Only relevant if type is \
            specified as motor so that sgen is treated as asynchronous motor

        **current_source** (bool, True) - Model this sgen as a current source during short-\
            circuit calculations; useful in some cases, for example the simulation of full-\
            size converters per IEC 60909-0:2016.

    OUTPUT:
        **index** (int) - The unique ID of the created sgen

    EXAMPLE:
        create_sgen(net, 1, p_mw = -120)

    """
    if bus not in net["bus"].index.values:
        raise UserWarning("Cannot attach to bus %s, bus does not exist" % bus)

    if index is None:
        index = get_free_id(net["sgen"])

    if index in net["sgen"].index:
        raise UserWarning("A static generator with the id %s already exists" % index)

    # store dtypes
    dtypes = net.sgen.dtypes

    net.sgen.loc[index, ["name", "bus", "p_mw", "scaling",
                         "q_mvar", "sn_mva", "in_service", "type",
                         "current_source"]] = \
        [name, bus, p_mw, scaling, q_mvar, sn_mva, bool(in_service), type, current_source]

    # and preserve dtypes
    _preserve_dtypes(net.sgen, dtypes)

    if not isnan(min_p_mw):
        if "min_p_mw" not in net.sgen.columns:
            net.sgen.loc[:, "min_p_mw"] = pd.Series()

        net.sgen.loc[index, "min_p_mw"] = float(min_p_mw)

    if not isnan(max_p_mw):
        if "max_p_mw" not in net.sgen.columns:
            net.sgen.loc[:, "max_p_mw"] = pd.Series()

        net.sgen.loc[index, "max_p_mw"] = float(max_p_mw)

    if not isnan(min_q_mvar):
        if "min_q_mvar" not in net.sgen.columns:
            net.sgen.loc[:, "min_q_mvar"] = pd.Series()

        net.sgen.loc[index, "min_q_mvar"] = float(min_q_mvar)

    if not isnan(max_q_mvar):
        if "max_q_mvar" not in net.sgen.columns:
            net.sgen.loc[:, "max_q_mvar"] = pd.Series()

        net.sgen.loc[index, "max_q_mvar"] = float(max_q_mvar)

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


def create_sgen_from_cosphi(net, bus, sn_mva, cos_phi, mode, **kwargs):
    """
    Creates an sgen element from rated power and power factor cos(phi).

    INPUT:
        **net** - The net within this static generator should be created

        **bus** (int) - The bus id to which the static generator is connected

        **sn_mva** (float) - rated power of the generator

        **cos_phi** (float) - power factor cos_phi

        **mode** (str) - "ind" for inductive or "cap" for capacitive behaviour

    OUTPUT:
        **index** (int) - The unique ID of the created sgen

    gen, sgen, and ext_grid are modelled in the generator point of view. Active power
    will therefore be postive por generation, and reactive power will be negative for consumption behaviour and
    positive for generation behaviour.
    """
    from pandapower.toolbox import pq_from_cosphi
    p_mw, q_mvar = pq_from_cosphi(sn_mva, cos_phi, qmode=mode, pmode="gen")
    return create_sgen(net, bus, sn_mva=sn_mva, p_mw=p_mw, q_mvar=q_mvar, **kwargs)


def create_storage(net, bus, p_mw, max_e_mwh, q_mvar=0, sn_mva=nan, soc_percent=nan, min_e_mwh=0.0,
                   name=None, index=None, scaling=1., type=None, in_service=True, max_p_mw=nan,
                   min_p_mw=nan, max_q_mvar=nan, min_q_mvar=nan, controllable = nan):
    """
    Adds a storage to the network.

    In order to simulate a storage system it is possible to use sgens or loads to model the
    discharging or charging state. The power of a storage can be positive or negative, so the use
    of either a sgen or a load is (per definition of the elements) not correct.
    To overcome this issue, a storage element can be created.

    As pandapower is not a time dependend simulation tool and there is no time domain parameter in
    default power flow calculations, the state of charge (SOC) is not updated during any power flow
    calculation.
    The implementation of energy content related parameters in the storage element allows to create
    customized, time dependend simulations by running several power flow calculations and updating
    variables manually.

    INPUT:
        **net** - The net within this storage should be created

        **bus** (int) - The bus id to which the storage is connected

        **p_mw** (float) - The momentary real power of the storage \
            (positive for charging, negative for discharging)

        **max_e_mwh** (float) - The maximum energy content of the storage \
            (maximum charge level)

    OPTIONAL:
        **q_mvar** (float, default 0) - The reactive power of the storage

        **sn_mva** (float, default None) - Nominal power of the storage

        **soc_percent** (float, NaN) - The state of charge of the storage

        **min_e_mwh** (float, 0) - The minimum energy content of the storage \
            (minimum charge level)

        **name** (string, default None) - The name for this storage

        **index** (int, None) - Force a specified ID if it is available. If None, the index one \
            higher than the highest already existing index is selected.

        **scaling** (float, 1.) - An OPTIONAL scaling factor to be set customly

        **type** (string, None) -  type variable to classify the storage

        **in_service** (boolean) - True for in_service or False for out of service

        **max_p_mw** (float, NaN) - Maximum active power injection - necessary for a \
            controllable storage in OPF

        **min_p_mw** (float, NaN) - Minimum active power injection - necessary for a \
            controllable storage in OPF

        **max_q_mvar** (float, NaN) - Maximum reactive power injection - necessary for a \
            controllable storage in OPF

        **min_q_mvar** (float, NaN) - Minimum reactive power injection - necessary for a \
            controllable storage in OPF

        **controllable** (bool, NaN) - Whether this storage is controllable by the optimal
        powerflow

    OUTPUT:
        **index** (int) - The unique ID of the created storage

    EXAMPLE:
        create_storage(net, 1, p_mw = -30, max_e_mwh = 60, soc_percent = 1.0, min_e_mwh = 5)

    """
    if bus not in net["bus"].index.values:
        raise UserWarning("Cannot attach to bus %s, bus does not exist" % bus)

    if index is None:
        index = get_free_id(net["storage"])

    if index in net["storage"].index:
        raise UserWarning("A storage with the id %s already exists" % index)

    # store dtypes
    dtypes = net.storage.dtypes

    net.storage.loc[index, ["name", "bus", "p_mw", "q_mvar", "sn_mva", "scaling",
                            "soc_percent", "min_e_mwh", "max_e_mwh", "in_service", "type"]] = \
        [name, bus, p_mw, q_mvar, sn_mva, scaling,
         soc_percent, min_e_mwh, max_e_mwh, bool(in_service), type]

    # and preserve dtypes
    _preserve_dtypes(net.storage, dtypes)

    # check for OPF parameters and add columns to network table
    if not isnan(min_p_mw):
        if "min_p_mw" not in net.storage.columns:
            net.storage.loc[:, "min_p_mw"] = pd.Series()

        net.storage.loc[index, "min_p_mw"] = float(min_p_mw)

    if not isnan(max_p_mw):
        if "max_p_mw" not in net.storage.columns:
            net.storage.loc[:, "max_p_mw"] = pd.Series()

        net.storage.loc[index, "max_p_mw"] = float(max_p_mw)

    if not isnan(min_q_mvar):
        if "min_q_mvar" not in net.storage.columns:
            net.storage.loc[:, "min_q_mvar"] = pd.Series()

        net.storage.loc[index, "min_q_mvar"] = float(min_q_mvar)

    if not isnan(max_q_mvar):
        if "max_q_mvar" not in net.storage.columns:
            net.storage.loc[:, "max_q_mvar"] = pd.Series()

        net.storage.loc[index, "max_q_mvar"] = float(max_q_mvar)

    if not isnan(controllable):
        if "controllable" not in net.storage.columns:
            net.storage.loc[:, "controllable"] = pd.Series()

        net.storage.loc[index, "controllable"] = bool(controllable)
    else:
        if "controllable" in net.storage.columns:
            net.storage.loc[index, "controllable"] = False

    return index


def create_gen(net, bus, p_mw, vm_pu=1., sn_mva=nan, name=None, index=None, max_q_mvar=nan,
               min_q_mvar=nan, min_p_mw=nan, max_p_mw=nan, scaling=1., type=None, slack=False,
               controllable=nan, vn_kv=nan, xdss_pu=nan, rdss_pu=nan, cos_phi=nan, in_service=True):
    """
    Adds a generator to the network.

    Generators are always modelled as voltage controlled PV nodes, which is why the input parameter
    is active power and a voltage set point. If you want to model a generator as PQ load with fixed
    reactive power and variable voltage, please use a static generator instead.

    INPUT:
        **net** - The net within this generator should be created

        **bus** (int) - The bus id to which the generator is connected

    OPTIONAL:
        **p_mw** (float, default 0) - The real power of the generator (positive for generation!)

        **vm_pu** (float, default 0) - The voltage set point of the generator.

        **sn_mva** (float, None) - Nominal power of the generator

        **name** (string, None) - The name for this generator

        **index** (int, None) - Force a specified ID if it is available. If None, the index one \
            higher than the highest already existing index is selected.

        **scaling** (float, 1.0) - scaling factor which for the active power of the generator

        **type** (string, None) - type variable to classify generators

        **controllable** (bool, NaN) - Whether this generator is controllable by the optimal
        powerflow

        **vn_kv** (float, NaN) - Rated voltage of the generator for short-circuit calculation

        **xdss_pu** (float, NaN) - Subtransient generator reactance for short-circuit calculation

        **rdss_pu** (float, NaN) - Subtransient generator resistance for short-circuit calculation

        **cos_phi** (float, NaN) - Rated cosine phi of the generator for short-circuit calculation

        **in_service** (bool, True) - True for in_service or False for out of service

        **max_p_mw** (float, default NaN) - Maximum active power injection - necessary for OPF

        **min_p_mw** (float, default NaN) - Minimum active power injection - necessary for OPF

        **max_q_mvar** (float, default NaN) - Maximum reactive power injection - necessary for OPF

        **min_q_mvar** (float, default NaN) - Minimum reactive power injection - necessary for OPF

    OUTPUT:
        **index** (int) - The unique ID of the created generator

    EXAMPLE:
        create_gen(net, 1, p_mw = 120, vm_pu = 1.02)

    """
    if bus not in net["bus"].index.values:
        raise UserWarning("Cannot attach to bus %s, bus does not exist" % bus)

    if index is None:
        index = get_free_id(net["gen"])

    if index in net["gen"].index:
        raise UserWarning("A generator with the id %s already exists" % index)

    # store dtypes
    dtypes = net.gen.dtypes

    columns = ["name", "bus", "p_mw", "vm_pu", "sn_mva", "type", "slack", "in_service",
                        "scaling"]
    variables = [name, bus, p_mw, vm_pu, sn_mva, type, slack, bool(in_service), scaling]
    net.gen.loc[index, columns] = variables

    # and preserve dtypes
    _preserve_dtypes(net.gen, dtypes)

    if not isnan(min_p_mw):
        if "min_p_mw" not in net.gen.columns:
            net.gen.loc[:, "min_p_mw"] = pd.Series()
        net.gen.loc[index, "min_p_mw"] = float(min_p_mw)

    if not isnan(max_p_mw):
        if "max_p_mw" not in net.gen.columns:
            net.gen.loc[:, "max_p_mw"] = pd.Series()
        net.gen.loc[index, "max_p_mw"] = float(max_p_mw)

    if not isnan(min_q_mvar):
        if "min_q_mvar" not in net.gen.columns:
            net.gen.loc[:, "min_q_mvar"] = pd.Series()
        net.gen.loc[index, "min_q_mvar"] = float(min_q_mvar)

    if not isnan(max_q_mvar):
        if "max_q_mvar" not in net.gen.columns:
            net.gen.loc[:, "max_q_mvar"] = pd.Series()
        net.gen.loc[index, "max_q_mvar"] = float(max_q_mvar)

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

    if not isnan(xdss_pu):
        if "xdss_pu" not in net.gen.columns:
            net.gen.loc[:, "xdss_pu"] = pd.Series()
        if "rdss_pu" not in net.gen.columns:
            net.gen.loc[:, "rdss_pu"] = pd.Series()
        net.gen.loc[index, "xdss_pu"] = float(xdss_pu)

    if not isnan(rdss_pu):
        if "rdss_pu" not in net.gen.columns:
            net.gen.loc[:, "rdss_pu"] = pd.Series()
        net.gen.loc[index, "rdss_pu"] = float(rdss_pu)

    if not isnan(cos_phi):
        if "cos_phi" not in net.gen.columns:
            net.gen.loc[:, "cos_phi"] = pd.Series()
        net.gen.loc[index, "cos_phi"] = float(cos_phi)

    return index


def create_ext_grid(net, bus, vm_pu=1.0, va_degree=0., name=None, in_service=True,
                    s_sc_max_mva=nan, s_sc_min_mva=nan, rx_max=nan, rx_min=nan,
                    max_p_mw=nan, min_p_mw=nan, max_q_mvar=nan, min_q_mvar=nan,
                    index=None, **kwargs):
    """
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

        **s_sc_max_mva** (float, NaN) - maximal short circuit apparent power to calculate internal \
            impedance of ext_grid for short circuit calculations

        **s_sc_min_mva** (float, NaN) - minimal short circuit apparent power to calculate internal \
            impedance of ext_grid for short circuit calculations

        **rx_max** (float, NaN) - maximal R/X-ratio to calculate internal impedance of ext_grid \
            for short circuit calculations

        **rx_min** (float, NaN) - minimal R/X-ratio to calculate internal impedance of ext_grid \
            for short circuit calculations

        **max_p_mw** (float, NaN) - Maximum active power injection. Only respected for OPF

        **min_p_mw** (float, NaN) - Minimum active power injection. Only respected for OPF

        **max_q_mvar** (float, NaN) - Maximum reactive power injection. Only respected for OPF

        **min_q_mvar** (float, NaN) - Minimum reactive power injection. Only respected for OPF

        \* only considered in loadflow if calculate_voltage_angles = True

    EXAMPLE:
        create_ext_grid(net, 1, voltage = 1.03)
    """
    if bus not in net["bus"].index.values:
        raise UserWarning("Cannot attach to bus %s, bus does not exist" % bus)

    if index is not None and index in net["ext_grid"].index:
        raise UserWarning("An external grid with with index %s already exists" % index)

    if index is None:
        index = get_free_id(net["ext_grid"])

    # store dtypes
    dtypes = net.ext_grid.dtypes

    net.ext_grid.loc[index, ["bus", "name", "vm_pu", "va_degree", "in_service"]] = \
        [bus, name, vm_pu, va_degree, bool(in_service)]

    if not isnan(s_sc_max_mva):
        if "s_sc_max_mva" not in net.ext_grid.columns:
            net.ext_grid.loc[:, "s_sc_max_mva"] = pd.Series()

        net.ext_grid.at[index, "s_sc_max_mva"] = float(s_sc_max_mva)

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

    if not isnan(min_p_mw):
        if "min_p_mw" not in net.ext_grid.columns:
            net.ext_grid.loc[:, "min_p_mw"] = pd.Series()

        net.ext_grid.loc[index, "min_p_mw"] = float(min_p_mw)

    if not isnan(max_p_mw):
        if "max_p_mw" not in net.ext_grid.columns:
            net.ext_grid.loc[:, "max_p_mw"] = pd.Series()

        net.ext_grid.loc[index, "max_p_mw"] = float(max_p_mw)

    if not isnan(min_q_mvar):
        if "min_q_mvar" not in net.ext_grid.columns:
            net.ext_grid.loc[:, "min_q_mvar"] = pd.Series()

        net.ext_grid.loc[index, "min_q_mvar"] = float(min_q_mvar)

    if not isnan(max_q_mvar):
        if "max_q_mvar" not in net.ext_grid.columns:
            net.ext_grid.loc[:, "max_q_mvar"] = pd.Series()

        net.ext_grid.loc[index, "max_q_mvar"] = float(max_q_mvar)

        # and preserve dtypes
    _preserve_dtypes(net.ext_grid, dtypes)
    return index


def create_line(net, from_bus, to_bus, length_km, std_type, name=None, index=None, geodata=None,
                df=1., parallel=1, in_service=True, max_loading_percent=nan, alpha=None,
                temperature_degree_celsius=None):
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
        **name** (string, None) - A custom name for this line

        **index** (int, None) - Force a specified ID if it is available. If None, the index one \
            higher than the highest already existing index is selected.

        **geodata**
        (array, default None, shape= (,2L)) -
        The linegeodata of the line. The first row should be the coordinates
        of bus a and the last should be the coordinates of bus b. The points
        in the middle represent the bending points of the line

        **in_service** (boolean, True) - True for in_service or False for out of service

        **df** (float, 1) - derating factor: maximal current of line in relation to nominal current \
            of line (from 0 to 1)

        **parallel** (integer, 1) - number of parallel line systems

        **max_loading_percent (float)** - maximum current loading (only needed for OPF)

    OUTPUT:
        **index** (int) - The unique ID of the created line

    EXAMPLE:
        create_line(net, "line1", from_bus = 0, to_bus = 1, length_km=0.1,  std_type="NAYY 4x50 SE")

    """

    # check if bus exist to attach the line to
    for b in [from_bus, to_bus]:
        if b not in net["bus"].index.values:
            raise UserWarning("Line %s tries to attach to non-existing bus %s" % (name, b))

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
    v["g_us_per_km"] = lineparam["g_us_per_km"] if "g_us_per_km" in lineparam else 0.

    if "type" in lineparam:
        v["type"] = lineparam["type"]

    # if net.line column already has alpha, add it from std_type
    if "alpha" in net.line.columns and "alpha" in lineparam:
        v["alpha"] = lineparam["alpha"]

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

    if alpha is not None:
        if "alpha" not in net.line.columns:
            net.line.loc[:, "alpha"] = pd.Series()
        net.line.loc[index, "alpha"] = alpha

    if temperature_degree_celsius is not None:
        if "temperature_degree_celsius" not in net.line.columns:
            net.line.loc[:, "temperature_degree_celsius"] = pd.Series()
        net.line.loc[index, "temperature_degree_celsius"] = temperature_degree_celsius

    return index


def create_line_from_parameters(net, from_bus, to_bus, length_km, r_ohm_per_km, x_ohm_per_km,
                                c_nf_per_km, max_i_ka, name=None, index=None, type=None,
                                geodata=None, in_service=True, df=1., parallel=1, g_us_per_km=0.,
                                max_loading_percent=nan, alpha=None,
                                temperature_degree_celsius=None, **kwargs):
    """
    Creates a line element in net["line"] from line parameters.

    INPUT:
        **net** - The net within this line should be created

        **from_bus** (int) - ID of the bus on one side which the line will be connected with

        **to_bus** (int) - ID of the bus on the other side which the line will be connected with

        **length_km** (float) - The line length in km

        **r_ohm_per_km** (float) - line resistance in ohm per km

        **x_ohm_per_km** (float) - line reactance in ohm per km

        **c_nf_per_km** (float) - line capacitance in nano Farad per km

        **max_i_ka** (float) - maximum thermal current in kilo Ampere

    OPTIONAL:
        **name** (string, None) - A custom name for this line

        **index** (int, None) - Force a specified ID if it is available. If None, the index one \
            higher than the highest already existing index is selected.

        **in_service** (boolean, True) - True for in_service or False for out of service

        **type** (str, None) - type of line ("ol" for overhead line or "cs" for cable system)

        **df** (float, 1) - derating factor: maximal current of line in relation to nominal current \
            of line (from 0 to 1)

        **g_us_per_km** (float, 0) - dielectric conductance in micro Siemens per km

        **parallel** (integer, 1) - number of parallel line systems

        **geodata**
        (array, default None, shape= (,2L)) -
        The linegeodata of the line. The first row should be the coordinates
        of bus a and the last should be the coordinates of bus b. The points
        in the middle represent the bending points of the line

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
        "c_nf_per_km": c_nf_per_km, "max_i_ka": max_i_ka, "parallel": parallel, "type": type,
        "g_us_per_km": g_us_per_km
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

    if alpha is not None:
        if "alpha" not in net.line.columns:
            net.line.loc[:, "alpha"] = pd.Series()
        net.line.loc[index, "alpha"] = alpha

    if temperature_degree_celsius is not None:
        if "temperature_degree_celsius" not in net.line.columns:
            net.line.loc[:, "temperature_degree_celsius"] = pd.Series()
        net.line.loc[index, "temperature_degree_celsius"] = temperature_degree_celsius

    return index

def create_lines(net, from_buses, to_buses, length_km, std_type, name=None, index=None, geodata=None,
                df=1., parallel=1, in_service=True, max_loading_percent=nan):
    """ Convenience function for creating many lines at once. Parameters 'from_buses' and 'to_buses'
        must be arrays of equal length. Other parameters may be either arrays of the same length or
        single or values. In any case the line parameters are defined through a single standard
        type, so all lines have the same standard type.


        INPUT:
            **net** - The net within this line should be created

            **from_bus** (list of int) - ID of the bus on one side which the line will be connected with

            **to_bus** (list of int) - ID of the bus on the other side which the line will be connected with

            **length_km** (list of float) - The line length in km

            **std_type** (string) - The linetype of the lines.

        OPTIONAL:
            **name** (list of string, None) - A custom name for this line

            **index** (list of int, None) - Force a specified ID if it is available. If None, the index one \
                higher than the highest already existing index is selected.

            **geodata**
            (list of arrays, default None, shape of arrays (,2L)) -
            The linegeodata of the line. The first row should be the coordinates
            of bus a and the last should be the coordinates of bus b. The points
            in the middle represent the bending points of the line

            **in_service** (list of boolean, True) - True for in_service or False for out of service

            **df** (list of float, 1) - derating factor: maximal current of line in relation to nominal current \
                of line (from 0 to 1)

            **parallel** (list of integer, 1) - number of parallel line systems

            **max_loading_percent (list of float)** - maximum current loading (only needed for OPF)

        OUTPUT:
            **index** (list of int) - The unique ID of the created line

        EXAMPLE:
            create_line(net, "line1", from_bus = 0, to_bus = 1, length_km=0.1,  std_type="NAYY 4x50 SE")

    """

    nr_lines = len(from_buses)
    if index is not None:
        for idx in index:
            if idx in net.line.index:
                raise UserWarning("A line with index %s already exists" % index)
    else:
        lid = get_free_id(net["line"])
        index = arange(lid, lid + nr_lines, 1)

    dtypes = net.line.dtypes

    dd = pd.DataFrame(index=index, columns=net.line.columns)

    # user defined params
    dd["from_bus"] = from_buses
    dd["to_bus"] = to_buses
    dd["length_km"] = length_km
    dd["std_type"] = std_type

    # add std type data
    lineparam = load_std_type(net, std_type, "line")
    dd["r_ohm_per_km"] = lineparam["r_ohm_per_km"]
    dd["x_ohm_per_km"] = lineparam["x_ohm_per_km"]
    dd["c_nf_per_km"] = lineparam["c_nf_per_km"]
    dd["max_i_ka"] = lineparam["max_i_ka"]
    dd["g_us_per_km"] = lineparam["g_us_per_km"] if "g_us_per_km" in lineparam else 0.
    if "type" in lineparam:
        dd["type"] = lineparam["type"]

    # optional params
    dd["name"] = name
    dd["df"] = df
    dd["parallel"] = parallel
    dd["in_service"] = in_service

    # extend the lines by the frame we just created
    try:
        net["line"] = net["line"].append(dd, sort=False)
    except TypeError:
        # prior to pandas 0.23 there was no explicit parameter (instead it was standard behavior)
        net["line"] = net["line"].append(dd)


    if hasattr(max_loading_percent, "__iter__"):
        if "max_loading_percent" not in net.line.columns:
            net.line["max_loading_percent"] = pd.Series(index=net.line.index)
        net.line.loc[index, "max_loading_percent"] = [0 if isnan(ml) else float(ml) for ml in max_loading_percent]
    else:
        if not isnan(max_loading_percent):
            if "max_loading_percent" not in net.line.columns:
                net.line["max_loading_percent"] = pd.Series(index=net.line.index)
            net.line.loc[index, "max_loading_percent"] = max_loading_percent

    _preserve_dtypes(net.line, dtypes)

    if geodata is not None:
        dtypes = net.line_geodata.dtypes
        df = pd.DataFrame(index=index, columns=net.line_geodata.columns)
        # works with single or multiple lists of coordinates
        if len(geodata[0]) == 2 and not hasattr(geodata[0][0], "__iter__"):
            # geodata is a single list of coordinates
            df["coords"] = [geodata] * len(index)
        else:
            # geodata is multiple lists of coordinates
            df["coords"] = geodata

        if version.parse(pd.__version__) >= version.parse("0.23"):
            net.line_geodata = net.line_geodata.append(df, sort=False)
        else:
            # prior to pandas 0.23 there was no explicit parameter (instead it was standard behavior)
            net.line_geodata = net.line_geodata.append(df)

        _preserve_dtypes(net.line_geodata, dtypes)

    return index


def create_transformer(net, hv_bus, lv_bus, std_type, name=None, tap_pos=nan, in_service=True,
                       index=None, max_loading_percent=nan, parallel=1, df=1.):
    """
    Creates a two-winding transformer in table net["trafo"].
    The trafo parameters are defined through the standard type library.

    INPUT:
        **net** - The net within this transformer should be created

        **hv_bus** (int) - The bus on the high-voltage side on which the transformer will be \
            connected to

        **lv_bus** (int) - The bus on the low-voltage side on which the transformer will be \
            connected to

        **std_type** -  The used standard type from the standard type library

    OPTIONAL:
        **name** (string, None) - A custom name for this transformer

        **tap_pos** (int, nan) - current tap position of the transformer. Defaults to the medium \
            position (tap_neutral)

        **in_service** (boolean, True) - True for in_service or False for out of service

        **index** (int, None) - Force a specified ID if it is available. If None, the index one \
            higher than the highest already existing index is selected.

        **max_loading_percent (float)** - maximum current loading (only needed for OPF)

        **parallel** (integer) - number of parallel transformers

        **df** (float) - derating factor: maximal current of transformer in relation to nominal \
            current of transformer (from 0 to 1)

    OUTPUT:
        **index** (int) - The unique ID of the created transformer

    EXAMPLE:
        create_transformer(net, hv_bus = 0, lv_bus = 1, name = "trafo1", std_type = \
            "0.4 MVA 10/0.4 kV")
    """

    # Check if bus exist to attach the trafo to
    for b in [hv_bus, lv_bus]:
        if b not in net["bus"].index.values:
            raise UserWarning("Trafo tries to attach to bus %s" % b)

    if df <= 0:
        raise UserWarning("raiting factor df must be positive: df = %.3f" % df)

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
        "sn_mva": ti["sn_mva"],
        "vn_hv_kv": ti["vn_hv_kv"],
        "vn_lv_kv": ti["vn_lv_kv"],
        "vk_percent": ti["vk_percent"],
        "vkr_percent": ti["vkr_percent"],
        "pfe_kw": ti["pfe_kw"],
        "i0_percent": ti["i0_percent"],
        "parallel": parallel,
        "df": df,
        "shift_degree": ti["shift_degree"] if "shift_degree" in ti else 0,
        "tap_phase_shifter": ti["tap_phase_shifter"] if "tap_phase_shifter" in ti
                            and pd.notnull(ti["tap_phase_shifter"]) else False
    })
    for tp in ("tap_neutral", "tap_max", "tap_min", "tap_side", "tap_step_percent", "tap_step_degree"):
        if tp in ti:
            v.update({tp: ti[tp]})
    if ("tap_neutral" in v) and (tap_pos is nan):
        v["tap_pos"] = v["tap_neutral"]
    else:
        v["tap_pos"] = tap_pos
        if isinstance(tap_pos, float):
            net.trafo.tap_pos = net.trafo.tap_pos.astype(float)
    # store dtypes
    dtypes = net.trafo.dtypes

    net.trafo.loc[index, list(v.keys())] = list(v.values())

    if not isnan(max_loading_percent):
        if "max_loading_percent" not in net.trafo.columns:
            net.trafo.loc[:, "max_loading_percent"] = pd.Series()

        net.trafo.loc[index, "max_loading_percent"] = float(max_loading_percent)

    # tap_phase_shifter default False
    net.trafo.tap_phase_shifter.fillna(False, inplace=True)

    # and preserve dtypes
    _preserve_dtypes(net.trafo, dtypes)

    return index


def create_transformer_from_parameters(net, hv_bus, lv_bus, sn_mva, vn_hv_kv, vn_lv_kv,
                                       vkr_percent, vk_percent, pfe_kw, i0_percent,
                                       shift_degree=0, tap_side=None, tap_neutral=nan, tap_max=nan,
                                       tap_min=nan, tap_step_percent=nan, tap_step_degree=nan,
                                       tap_pos=nan, tap_phase_shifter=False, in_service=True,
                                       name=None, index=None, max_loading_percent=nan, parallel=1,
                                       df=1., **kwargs):
    """
    Creates a two-winding transformer in table net["trafo"].
    The trafo parameters are defined through the standard type library.

    INPUT:
        **net** - The net within this transformer should be created

        **hv_bus** (int) - The bus on the high-voltage side on which the transformer will be \
            connected to

        **lv_bus** (int) - The bus on the low-voltage side on which the transformer will be \
            connected to

        **sn_mva** (float) - rated apparent power

        **vn_hv_kv** (float) - rated voltage on high voltage side

        **vn_lv_kv** (float) - rated voltage on low voltage side

        **vkr_percent** (float) - real part of relative short-circuit voltage

        **vk_percent** (float) - relative short-circuit voltage

        **pfe_kw** (float)  - iron losses in kW

        **i0_percent** (float) - open loop losses in percent of rated current

    OPTIONAL:
        **in_service** (boolean) - True for in_service or False for out of service

        **parallel** (integer) - number of parallel transformers

        **name** (string) - A custom name for this transformer

        **shift_degree** (float) - Angle shift over the transformer*

        **tap_side** (string) - position of tap changer ("hv", "lv")

        **tap_pos** (int, nan) - current tap position of the transformer. Defaults to the medium \
            position (tap_neutral)

        **tap_neutral** (int, nan) - tap position where the transformer ratio is equal to the ration of \
            the rated voltages

        **tap_max** (int, nan) - maximal allowed tap position

        **tap_min** (int, nan):  minimal allowed tap position

        **tap_step_percent** (float) - tap step size for voltage magnitude in percent

        **tap_step_degree** (float) - tap step size for voltage angle in degree*

        **tap_phase_shifter** (bool) - whether the transformer is an ideal phase shifter*

        **index** (int, None) - Force a specified ID if it is available. If None, the index one \
            higher than the highest already existing index is selected.

        **max_loading_percent (float)** - maximum current loading (only needed for OPF)

        **df** (float) - derating factor: maximal current of transformer in relation to nominal \
            current of transformer (from 0 to 1)

        \* only considered in loadflow if calculate_voltage_angles = True

    OUTPUT:
        **index** (int) - The unique ID of the created transformer

    EXAMPLE:
        create_transformer_from_parameters(net, hv_bus=0, lv_bus=1, name="trafo1", sn_mva=40, \
            vn_hv_kv=110, vn_lv_kv=10, vk_percent=10, vkr_percent=0.3, pfe_kw=30, \
            i0_percent=0.1, shift_degree=30)
    """

    # Check if bus exist to attach the trafo to
    for b in [hv_bus, lv_bus]:
        if b not in net["bus"].index.values:
            raise UserWarning("Trafo tries to attach to bus %s" % b)

    if df <= 0:
        raise UserWarning("derating factor df must be positive: df = %.3f" % df)

    if index is None:
        index = get_free_id(net["trafo"])

    if index in net["trafo"].index:
        raise UserWarning("A transformer with index %s already exists" % index)

    if tap_pos is nan:
        tap_pos = tap_neutral
    v = {
        "name": name, "hv_bus": hv_bus, "lv_bus": lv_bus,
        "in_service": bool(in_service), "std_type": None, "sn_mva": sn_mva, "vn_hv_kv": vn_hv_kv,
        "vn_lv_kv": vn_lv_kv, "vk_percent": vk_percent, "vkr_percent": vkr_percent,
        "pfe_kw": pfe_kw, "i0_percent": i0_percent, "tap_neutral": tap_neutral,
        "tap_max": tap_max, "tap_min": tap_min, "shift_degree": shift_degree,
        "tap_side": tap_side, "tap_step_percent": tap_step_percent, "tap_step_degree": tap_step_degree,
        "tap_phase_shifter": tap_phase_shifter, "parallel": parallel, "df": df
    }

    if ("tap_neutral" in v) and (tap_pos is nan):
        v["tap_pos"] = v["tap_neutral"]
    else:
        v["tap_pos"] = tap_pos
        if type(tap_pos) == float:
            net.trafo.tap_pos = net.trafo.tap_pos.astype(float)

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


def create_transformer3w(net, hv_bus, mv_bus, lv_bus, std_type, name=None, tap_pos=nan,
                         in_service=True, index=None, max_loading_percent=nan,
                         tap_at_star_point=False):
    """
    Creates a three-winding transformer in table net["trafo3w"].
    The trafo parameters are defined through the standard type library.

    INPUT:
        **net** - The net within this transformer should be created

        **hv_bus** (int) - The bus on the high-voltage side on which the transformer will be \
            connected to

        **mv_bus** (int) - The medium voltage bus on which the transformer will be connected to

        **lv_bus** (int) - The bus on the low-voltage side on which the transformer will be \
            connected to

        **std_type** -  The used standard type from the standard type library

    OPTIONAL:
        **name** (string) - A custom name for this transformer

        **tap_pos** (int, nan) - current tap position of the transformer. Defaults to the medium \
            position (tap_neutral)

        **tap_at_star_point** (boolean) - Whether tap changer is located at the star point of the \
            3W-transformer or at the bus

        **in_service** (boolean) - True for in_service or False for out of service

        **index** (int, None) - Force a specified ID if it is available. If None, the index one \
            higher than the highest already existing index is selected.

        **max_loading_percent (float)** - maximum current loading (only needed for OPF)

        **tap_at_star_point (bool)** - whether tap changer is modelled at star point or at the bus

    OUTPUT:
        **index** (int) - The unique ID of the created transformer

    EXAMPLE:
        create_transformer3w(net, hv_bus = 0, mv_bus = 1, lv_bus = 2, name = "trafo1", std_type = \
            "63/25/38 MVA 110/20/10 kV")
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
        "sn_hv_mva": ti["sn_hv_mva"],
        "sn_mv_mva": ti["sn_mv_mva"],
        "sn_lv_mva": ti["sn_lv_mva"],
        "vn_hv_kv": ti["vn_hv_kv"],
        "vn_mv_kv": ti["vn_mv_kv"],
        "vn_lv_kv": ti["vn_lv_kv"],
        "vk_hv_percent": ti["vk_hv_percent"],
        "vk_mv_percent": ti["vk_mv_percent"],
        "vk_lv_percent": ti["vk_lv_percent"],
        "vkr_hv_percent": ti["vkr_hv_percent"],
        "vkr_mv_percent": ti["vkr_mv_percent"],
        "vkr_lv_percent": ti["vkr_lv_percent"],
        "pfe_kw": ti["pfe_kw"],
        "i0_percent": ti["i0_percent"],
        "shift_mv_degree": ti["shift_mv_degree"] if "shift_mv_degree" in ti else 0,
        "shift_lv_degree": ti["shift_lv_degree"] if "shift_lv_degree" in ti else 0,
        "tap_at_star_point": tap_at_star_point
    })
    for tp in ("tap_neutral", "tap_max", "tap_min", "tap_side", "tap_step_percent", "tap_step_degree"):
        if tp in ti:
            v.update({tp: ti[tp]})

    if ("tap_neutral" in v) and (tap_pos is nan):
        v["tap_pos"] = v["tap_neutral"]
    else:
        v["tap_pos"] = tap_pos
        if type(tap_pos) == float:
            net.trafo3w.tap_pos = net.trafo3w.tap_pos.astype(float)

    dd = pd.DataFrame(v, index=[index])
    if version.parse(pd.__version__) < version.parse("0.21"):
        net["trafo3w"] = net["trafo3w"].append(dd).reindex_axis(net["trafo3w"].columns, axis=1)
    else:
        net["trafo3w"] = net["trafo3w"].append(dd).reindex(net["trafo3w"].columns, axis=1)

    if not isnan(max_loading_percent):
        if "max_loading_percent" not in net.trafo3w.columns:
            net.trafo3w.loc[:, "max_loading_percent"] = pd.Series()

        net.trafo3w.loc[index, "max_loading_percent"] = float(max_loading_percent)

    return index


def create_transformer3w_from_parameters(net, hv_bus, mv_bus, lv_bus, vn_hv_kv, vn_mv_kv, vn_lv_kv,
                                         sn_hv_mva, sn_mv_mva, sn_lv_mva, vk_hv_percent,
                                         vk_mv_percent, vk_lv_percent, vkr_hv_percent,
                                         vkr_mv_percent, vkr_lv_percent, pfe_kw, i0_percent,
                                         shift_mv_degree=0., shift_lv_degree=0., tap_side=None,
                                         tap_step_percent=nan, tap_step_degree=nan, tap_pos=nan,
                                         tap_neutral=nan, tap_max=nan,
                                         tap_min=nan, name=None, in_service=True, index=None,
                                         max_loading_percent=nan, tap_at_star_point=False):
    """
    Adds a three-winding transformer in table net["trafo3w"].

    Input:
        **net** (pandapowerNet) - The net within this transformer should be created

        **hv_bus** (int) - The bus on the high-voltage side on which the transformer will be \
            connected to

        **mv_bus** (int) - The bus on the middle-voltage side on which the transformer will be \
            connected to

        **lv_bus** (int) - The bus on the low-voltage side on which the transformer will be \
            connected to

        **vn_hv_kv** (float) rated voltage on high voltage side

        **vn_mv_kv** (float) rated voltage on medium voltage side

        **vn_lv_kv** (float) rated voltage on low voltage side

        **sn_hv_mva** (float) - rated apparent power on high voltage side

        **sn_mv_mva** (float) - rated apparent power on medium voltage side

        **sn_lv_mva** (float) - rated apparent power on low voltage side

        **vk_hv_percent** (float) - short circuit voltage from high to medium voltage

        **vk_mv_percent** (float) - short circuit voltage from medium to low voltage

        **vk_lv_percent** (float) - short circuit voltage from high to low voltage

        **vkr_hv_percent** (float) - real part of short circuit voltage from high to medium voltage

        **vkr_mv_percent** (float) - real part of short circuit voltage from medium to low voltage

        **vkr_lv_percent** (float) - real part of short circuit voltage from high to low voltage

        **pfe_kw** (float) - iron losses in kW

        **i0_percent** (float) - open loop losses

    OPTIONAL:
        **shift_mv_degree** (float, 0) - angle shift to medium voltage side*

        **shift_lv_degree** (float, 0) - angle shift to low voltage side*

        **tap_step_percent** (float) - Tap step in percent

        **tap_step_degree** (float) - Tap phase shift angle in degrees

        **tap_side** (string, None) - "hv", "mv", "lv"

        **tap_neutral** (int, nan) - default tap position

        **tap_min** (int, nan) - Minimum tap position

        **tap_max** (int, nan) - Maximum tap position

        **tap_pos** (int, nan) - current tap position of the transformer. Defaults to the \
            medium position (tap_neutral)

        **tap_at_star_point** (boolean) - Whether tap changer is located at the star point of the \
            3W-transformer or at the bus

        **name** (string, None) - Name of the 3-winding transformer

        **in_service** (boolean, True) - True for in_service or False for out of service

        \* only considered in loadflow if calculate_voltage_angles = True
        \**The model currently only supports one tap-changer per 3W Transformer.

        **max_loading_percent (float)** - maximum current loading (only needed for OPF)

    OUTPUT:
        **trafo_id** - The unique trafo_id of the created 3W transformer

    Example:
        create_transformer3w_from_parameters(net, hv_bus=0, mv_bus=1, lv_bus=2, name="trafo1",
        sn_hv_mva=40, sn_mv_mva=20, sn_lv_mva=20, vn_hv_kv=110, vn_mv_kv=20, vn_lv_kv=10,
        vk_hv_percent=10,vk_mv_percent=11, vk_lv_percent=12, vkr_hv_percent=0.3,
        vkr_mv_percent=0.31, vkr_lv_percent=0.32, pfe_kw=30, i0_percent=0.1, shift_mv_degree=30,
        shift_lv_degree=30)

    """

    # Check if bus exist to attach the trafo to
    for b in [hv_bus, mv_bus, lv_bus]:
        if b not in net["bus"].index.values:
            raise UserWarning("Trafo tries to attach to non-existent bus %s" % b)

    if index is None:
        index = get_free_id(net["trafo3w"])

    if index in net["trafo3w"].index:
        raise UserWarning("A three winding transformer with index %s already exists" % index)

    if tap_pos is nan:
        tap_pos = tap_neutral

    # store dtypes
    dtypes = net.trafo3w.dtypes

    net.trafo3w.loc[index, ["lv_bus", "mv_bus", "hv_bus", "vn_hv_kv", "vn_mv_kv", "vn_lv_kv",
                            "sn_hv_mva", "sn_mv_mva", "sn_lv_mva", "vk_hv_percent",
                            "vk_mv_percent", "vk_lv_percent", "vkr_hv_percent",
                            "vkr_mv_percent", "vkr_lv_percent", "pfe_kw", "i0_percent",
                            "shift_mv_degree", "shift_lv_degree", "tap_side", "tap_step_percent",
                            "tap_step_degree", "tap_pos", "tap_neutral", "tap_max", "tap_min", "in_service",
                            "name", "std_type", "tap_at_star_point" ]] = \
        [lv_bus, mv_bus, hv_bus, vn_hv_kv, vn_mv_kv, vn_lv_kv,
         sn_hv_mva, sn_mv_mva, sn_lv_mva, vk_hv_percent, vk_mv_percent,
         vk_lv_percent, vkr_hv_percent, vkr_mv_percent, vkr_lv_percent,
         pfe_kw, i0_percent, shift_mv_degree, shift_lv_degree,
         tap_side, tap_step_percent, tap_step_degree, tap_pos, tap_neutral, tap_max,
         tap_min, bool(in_service), name, None, tap_at_star_point]

    # and preserve dtypes
    _preserve_dtypes(net.trafo3w, dtypes)

    if not isnan(max_loading_percent):
        if "max_loading_percent" not in net.trafo3w.columns:
            net.trafo3w.loc[:, "max_loading_percent"] = pd.Series()

        net.trafo3w.loc[index, "max_loading_percent"] = float(max_loading_percent)

    return index


def create_switch(net, bus, element, et, closed=True, type=None, name=None, index=None, z_ohm=0):
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

        **element** - index of the element: bus id if et == "b", line id if et == "l", trafo id if \
            et == "t"

        **et** - (string) element type: "l" = switch between bus and line, "t" = switch between
        bus and transformer, "b" = switch between two buses

    OPTIONAL:
        **closed** (boolean, True) - switch position: False = open, True = closed

        **type** (int, None) - indicates the type of switch: "LS" = Load Switch, "CB" = \
            Circuit Breaker, "LBS" = Load Break Switch or "DS" = Disconnecting Switch

        **z_ohm** (float, 0) - indicates the resistance of the switch, which has effect only on
            bus-bus switches, if sets to 0, the buses will be fused like before, if larger than
            0 a branch will be created for the switch which has also effects on the bus mapping

        **name** (string, default None) - The name for this switch

    OUTPUT:
        **sid** - The unique switch_id of the created switch

    EXAMPLE:
        create_switch(net, bus =  0, element = 1, et = 'b', type ="LS", z_ohm = 0.1)

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

    net.switch.loc[index, ["bus", "element", "et", "closed", "type", "name", "z_ohm"]] = \
        [bus, element, et, closed, type, name, z_ohm]

    # and preserve dtypes
    _preserve_dtypes(net.switch, dtypes)

    return index


def create_shunt(net, bus, q_mvar, p_mw=0., vn_kv=None, step=1, max_step=1, name=None,
                 in_service=True, index=None):
    """
    Creates a shunt element

    INPUT:
        **net** (pandapowerNet) - The pandapower network in which the element is created

        **bus** - bus number of bus to whom the shunt is connected to

        **p_mw** - shunt active power in kW at v= 1.0 p.u.

        **q_mvar** - shunt susceptance in kVAr at v= 1.0 p.u.

    OPTIONAL:
        **vn_kv** (float, None) - rated voltage of the shunt. Defaults to rated voltage of \
            connected bus

        **step** (int, 1) - step of shunt with which power values are multiplied

        **max_step** (boolean, True) - True for in_service or False for out of service

        **name** (str, None) - element name

        **in_service** (boolean, True) - True for in_service or False for out of service

        **index** (int, None) - Force a specified ID if it is available. If None, the index one \
            higher than the highest already existing index is selected.

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

    net.shunt.loc[index, ["bus", "name", "p_mw", "q_mvar", "vn_kv", "step", "max_step",
                          "in_service"]] = [bus, name, p_mw, q_mvar, vn_kv, step, max_step,
                                            in_service]

    # and preserve dtypes
    _preserve_dtypes(net.shunt, dtypes)

    return index


def create_shunt_as_capacitor(net, bus, q_mvar, loss_factor, **kwargs):
    """
    Creates a shunt element representing a capacitor bank.

    INPUT:

        **net** (pandapowerNet) - The pandapower network in which the element is created

        **bus** - bus number of bus to whom the shunt is connected to

        **q_mvar** (float) - reactive power of the capacitor bank at rated voltage

        **loss_factor** (float) - loss factor tan(delta) of the capacitor bank

        **kwargs are passed to the create_shunt function


    OUTPUT:
        **index** (int) - The unique ID of the created shunt
    """
    q_mvar = -abs(q_mvar)  # q is always negative for capacitor
    p_mw = abs(q_mvar * loss_factor)  # p is always positive for active power losses
    return create_shunt(net, bus, q_mvar=q_mvar, p_mw=p_mw, **kwargs)


def create_impedance(net, from_bus, to_bus, rft_pu, xft_pu, sn_mva, rtf_pu=None, xtf_pu=None,
                     name=None, in_service=True, index=None):
    """
    Creates an per unit impedance element

    INPUT:
        **net** (pandapowerNet) - The pandapower network in which the element is created

        **from_bus** (int) - starting bus of the impedance

        **to_bus** (int) - ending bus of the impedance

        **r_pu** (float) - real part of the impedance in per unit

        **x_pu** (float) - imaginary part of the impedance in per unit

        **sn_mva** (float) - rated power of the impedance in kVA

    OUTPUT:

        impedance id
    """
    for b in [from_bus, to_bus]:
        if b not in net["bus"].index.values:
            raise UserWarning("Impedance %s tries to attach to non-existing bus %s" % (name, b))

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
                              "name", "sn_mva", "in_service"]] = \
        [from_bus, to_bus, rft_pu, xft_pu, rtf_pu, xtf_pu, name, sn_mva, in_service]

    # and preserve dtypes
    _preserve_dtypes(net.impedance, dtypes)

    return index


def create_series_reactor_as_impedance(net, from_bus, to_bus, r_ohm, x_ohm, sn_mva,
                                       name=None, in_service=True, index=None):
    """
    Creates a series reactor as per-unit impedance
    :param net: (pandapowerNet) - The pandapower network in which the element is created
    :param from_bus: (int) - starting bus of the series reactor
    :param to_bus: (int) - ending bus of the series reactor
    :param r_ohm: (float) - real part of the impedance in Ohm
    :param x_ohm: (float) - imaginary part of the impedance in Ohm
    :param sn_mva: (float) - rated power of the series reactor in kVA
    :param vn_kv: (float) - rated voltage of the series reactor in kV
    :return: index of the created element
    """
    for b in [from_bus, to_bus]:
        if b not in net["bus"].index.values:
            raise UserWarning(
                "Series reactor %s tries to attach to non-existing bus %s" % (name, b))

    if net.bus.at[from_bus, 'vn_kv'] == net.bus.at[to_bus, 'vn_kv']:
        vn_kv = net.bus.at[from_bus, 'vn_kv']
    else:
        raise UserWarning('Unable to infer rated voltage vn_kv for series reactor %s due to '
                          'different rated voltages of from_bus %d (%.3f p.u.) and '
                          'to_bus %d (%.3f p.u.)' % (name, from_bus, net.bus.at[from_bus, 'vn_kv'],
                                                     to_bus, net.bus.at[to_bus, 'vn_kv']))

    base_z_ohm = vn_kv ** 2 / (sn_mva * 1e-3)
    rft_pu = r_ohm / base_z_ohm
    xft_pu = x_ohm / base_z_ohm

    index = create_impedance(net, from_bus=from_bus, to_bus=to_bus, rft_pu=rft_pu, xft_pu=xft_pu,
                             sn_mva=sn_mva, name=name, in_service=in_service,
                             index=index)
    return index


def create_ward(net, bus, ps_mw, qs_mvar, pz_mw, qz_mvar, name=None, in_service=True, index=None):
    """
    Creates a ward equivalent.

    A ward equivalent is a combination of an impedance load and a PQ load.

    INPUT:
        **net** (pandapowernet) - The pandapower net within the element should be created

        **bus** (int) -  bus of the ward equivalent

        **ps_mw** (float) - active power of the PQ load

        **qs_mvar** (float) - reactive power of the PQ load

        **pz_mw** (float) - active power of the impedance load in kW at 1.pu voltage

        **qz_mvar** (float) - reactive power of the impedance load in kVar at 1.pu voltage

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

    net.ward.loc[index, ["bus", "ps_mw", "qs_mvar", "pz_mw", "qz_mvar", "name", "in_service"]] = \
        [bus, ps_mw, qs_mvar, pz_mw, qz_mvar, name, in_service]

    # and preserve dtypes
    _preserve_dtypes(net.ward, dtypes)

    return index


def create_xward(net, bus, ps_mw, qs_mvar, pz_mw, qz_mvar, r_ohm, x_ohm, vm_pu, in_service=True,
                 name=None, index=None):
    """
    Creates an extended ward equivalent.

    A ward equivalent is a combination of an impedance load, a PQ load and as voltage source with
    an internal impedance.

    INPUT:
        **net** - The pandapower net within the impedance should be created

        **bus** (int) -  bus of the ward equivalent

        **ps_mw** (float) - active power of the PQ load

        **qs_mvar** (float) - reactive power of the PQ load

        **pz_mw** (float) - active power of the impedance load in kW at 1.pu voltage

        **qz_mvar** (float) - reactive power of the impedance load in kVar at 1.pu voltage

        **r_ohm** (float) - internal resistance of the voltage source

        **x_ohm** (float) - internal reactance of the voltage source

        **vm_pu** (float) - voltage magnitude at the additional PV-node

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

    net.xward.loc[index, ["bus", "ps_mw", "qs_mvar", "pz_mw", "qz_mvar", "r_ohm", "x_ohm", "vm_pu",
                          "name", "in_service"]] = \
        [bus, ps_mw, qs_mvar, pz_mw, qz_mvar, r_ohm, x_ohm, vm_pu, name, in_service]

    # and preserve dtypes
    _preserve_dtypes(net.xward, dtypes)

    return index


def create_dcline(net, from_bus, to_bus, p_mw, loss_percent, loss_mw, vm_from_pu, vm_to_pu,
                  index=None, name=None, max_p_mw=nan, min_q_from_mvar=nan,
                  min_q_to_mvar=nan, max_q_from_mvar=nan, max_q_to_mvar=nan,
                  in_service=True):
    """
    Creates a dc line.

    INPUT:
        **from_bus** (int) - ID of the bus on one side which the line will be connected with

        **to_bus** (int) - ID of the bus on the other side which the line will be connected with

        **p_mw** - (float) Active power transmitted from 'from_bus' to 'to_bus'

        **loss_percent** - (float) Relative transmission loss in percent of active power
            transmission

        **loss_mw** - (float) Total transmission loss in kW

        **vm_from_pu** - (float) Voltage setpoint at from bus

        **vm_to_pu** - (float) Voltage setpoint at to bus

    OPTIONAL:
        **index** (int, None) - Force a specified ID if it is available. If None, the index one \
            higher than the highest already existing index is selected.

        **name** (str, None) - A custom name for this dc line

        **in_service** (boolean) - True for in_service or False for out of service

        **max_p_mw** - Maximum active power flow. Only respected for OPF

        **min_q_from_mvar** - Minimum reactive power at from bus. Necessary for OPF

        **min_q_to_mvar** - Minimum reactive power at to bus. Necessary for OPF

        **max_q_from_mvar** - Maximum reactive power at from bus. Necessary for OPF

        **max_q_to_mvar ** - Maximum reactive power at to bus. Necessary for OPF

    OUTPUT:
        **index** (int) - The unique ID of the created element

    EXAMPLE:
        create_dcline(net, from_bus=0, to_bus=1, p_mw=1e4, loss_percent=1.2, loss_mw=25, \
            vm_from_pu=1.01, vm_to_pu=1.02)
    """
    for bus in [from_bus, to_bus]:
        if bus not in net["bus"].index.values:
            raise UserWarning("Cannot attach to bus %s, bus does not exist" % bus)

    if index is None:
        index = get_free_id(net["dcline"])

    if index in net["dcline"].index:
        raise UserWarning("A dcline with the id %s already exists" % index)

    # store dtypes
    dtypes = net.dcline.dtypes

    net.dcline.loc[index, ["name", "from_bus", "to_bus", "p_mw", "loss_percent", "loss_mw",
                           "vm_from_pu", "vm_to_pu", "max_p_mw", "min_q_from_mvar",
                           "min_q_to_mvar", "max_q_from_mvar", "max_q_to_mvar", "in_service"]] \
        = [name, from_bus, to_bus, p_mw, loss_percent, loss_mw, vm_from_pu, vm_to_pu,
           max_p_mw, min_q_from_mvar, min_q_to_mvar, max_q_from_mvar, max_q_to_mvar, in_service]

    # and preserve dtypes
    _preserve_dtypes(net.dcline, dtypes)

    return index


def create_measurement(net, meas_type, element_type, value, std_dev, element, side=None,
                       check_existing=True, index=None, name=None):
    """
    Creates a measurement, which is used by the estimation module. Possible types of measurements
    are: v, p, q, i

    INPUT:
        **meas_type** (string) - Type of measurement. "v", "p", "q", "i" are possible

        **element_type** (string) - Clarifies which element is measured. "bus", "line",
        "trafo", and "trafo3w" are possible

        **value** (float) - Measurement value. Units are "MW" for P, "MVar" for Q, "p.u." for V,
        "kA" for I. Generation is a positive bus power injection, consumption negative

        **std_dev** (float) - Standard deviation in the same unit as the measurement

        **element** (int) - Index of the measured element (either bus index, line index, trafo index, trafo3w index)

        **side** (int, string, default: None) - Only used for measured lines or transformers. Side defines at which end
        of the branch the measurement is gathered. For lines this may be "from", "to" to denote the side with the
        from_bus or to_bus. It can also the be index of the from_bus or to_bus. For transformers, it can be "hv", "mv"
        or "lv" or the corresponding bus index, respectively

    OPTIONAL:
        **check_existing** (bool, default: None) - Check for and replace existing measurements for this bus,
        type and element_type. Set it to false for performance improvements which can cause unsafe
        behaviour

        **index** (int, default: None) - Index of the measurement in the measurement table. Should not exist already.

        **name** (str, default: None) - Name of measurement

    OUTPUT:
        (int) Index of measurement

    EXAMPLES:
        2 MW load measurement with 0.05 MW standard deviation on bus 0:
        create_measurement(net, "p", "bus", 0, -2., 0.05.)

        4.5 MVar line measurement with 0.1 MVar standard deviation on the "to_bus" side of line 2
        create_measurement(net, "q", "line", 2, 4.5, 0.1, "to")
    """

    if meas_type not in ("v", "p", "q", "i"):
        raise UserWarning("Invalid measurement type ({})".format(meas_type))

    if side is None and element_type in ("line", "trafo"):
        raise UserWarning("The element type {} requires a value in 'element'".format(element_type))

    if meas_type == "v":
        element_type = "bus"

    if element_type not in ("bus", "line", "trafo", "trafo3w"):
        raise UserWarning("Invalid element type ({})".format(element_type))

    if element_type == "bus" and element not in net["bus"].index.values:
        raise UserWarning("Bus with index={} does not exist".format(element))

    if element is not None and element_type == "line" and element not in net["line"].index.values:
        raise UserWarning("Line with index={} does not exist".format(element))

    if element is not None and element_type == "trafo" and element not in \
            net["trafo"].index.values:
        raise UserWarning("Trafo with index={} does not exist".format(element))

    if element is not None and element_type == "trafo3w" and element not in \
            net["trafo3w"].index.values:
        raise UserWarning("Trafo3w with index={} does not exist".format(element))

    if index is None:
        index = get_free_id(net.measurement)

    if index in net["measurement"].index:
        raise UserWarning("A measurement with index={} already exists".format(index))

    if meas_type == "i" and element_type == "bus":
        raise UserWarning("Line current measurements cannot be placed at buses")

    if meas_type == "v" and element_type in ("line", "trafo", "trafo3w"):
        raise UserWarning("Voltage measurements can only be placed at buses, not at {}".format(element_type))

    if check_existing:
        if side is None:
            existing = net.measurement[(net.measurement.measurement_type == meas_type) &
                                       (net.measurement.element_type == element_type) &
                                       (net.measurement.element == element) &
                                       (pd.isnull(net.measurement.side))].index
        else:
            existing = net.measurement[(net.measurement.measurement_type == meas_type) &
                                       (net.measurement.element_type == element_type) &
                                       (net.measurement.element == element) &
                                       (net.measurement.side == side)].index
        if len(existing) == 1:
            index = existing[0]
        elif len(existing) > 1:
            raise UserWarning("More than one measurement of this type exists")

    dtypes = net.measurement.dtypes
    columns = ["name", "measurement_type", "element_type", "element", "value", "std_dev", "side"]
    net.measurement.loc[index, columns] =\
        [name, meas_type.lower(), element_type, element, value, std_dev, side]
    _preserve_dtypes(net.measurement, dtypes)
    return index


def create_pwl_cost(net, element, et, points, power_type="p", index=None):
    """
    Creates an entry for piecewise linear costs for an element. The currently supported elements are
     - Generator
     - External Grid
     - Static Generator
     - Load
     - Dcline
     - Storage

    INPUT:
        **element** (int) - ID of the element in the respective element table

        **et** (string) - element type, one of "gen", "sgen", "ext_grid", "load", "dcline", "storage"]

        **points** - (list) list of lists with [[p1, p2, c1], [p2, p3, c2], ...] where c(n) defines the costs between p(n) and p(n+1)

    OPTIONAL:
        **type** - (string) - Type of cost ["p", "q"] are allowed for active or reactive power

        **index** (int, index) - Force a specified ID if it is available. If None, the index one \
            higher than the highest already existing index is selected.

    OUTPUT:
        **index** (int) - The unique ID of created cost entry

    EXAMPLE:
        The cost function is given by the x-values p1 and p2 with the slope m between those points. The constant part
        b of a linear function y = m*x + b can be neglected for OPF purposes. The intervals have to be continuous (the
        starting point of an interval has to be equal to th end point of the previous interval).

        To create a gen with costs of 1€/MW between 0 and 20 MW and 2€/MW between 20 and 30:

        create_pwl_cost(net, 0, "gen", [[0, 20, 1], [20, 30, 2]])
    """

    if index is None:
        index = get_free_id(net["pwl_cost"])

    if index in net["pwl_cost"].index:
        raise UserWarning("A piecewise_linear_cost with the id %s already exists" % index)

    net.pwl_cost.loc[index, ["power_type", "element", "et"]] = \
        [power_type, element, et]
    net.pwl_cost.points.loc[index] = points
    return index

def create_poly_cost(net, element, et, cp1_eur_per_mw, cp0_eur=0, cq1_eur_per_mvar=0,
                           cq0_eur=0, cp2_eur_per_mw2=0, cq2_eur_per_mvar2=0, type="p", index=None):
    """
    Creates an entry for polynimoal costs for an element. The currently supported elements are
     - Generator
     - External Grid
     - Static Generator
     - Load
     - Dcline
     - Storage

    INPUT:
        **element** (int) - ID of the element in the respective element table

        **element_type** (string) - Type of element ["gen", "sgen", "ext_grid", "load", "dcline", "storage"] \
            are possible

        **cp1_eur_per_mw** (float) - Linear costs per MW

        **cp0_eur=0** (float) - Offset active power costs in euro

        **cq1_eur_per_mvar=0** (float) - Linear costs per Mvar

        **cq0_eur=0** (float) - Offset reactive power costs in euro

        **cp2_eur_per_mw2=0** (float) - Quadratic costs per MW

        **cq2_eur_per_mvar2=0** (float) - Quadratic costs per Mvar

    OPTIONAL:
        **type** - (string) - Type of cost ["p", "q"] are allowed

        **index** (int, index) - Force a specified ID if it is available. If None, the index one \
            higher than the highest already existing index is selected.

    OUTPUT:
        **index** (int) - The unique ID of created cost entry

    EXAMPLE:
        The polynomial cost function is given by the linear and quadratic cost coefficients.

        create_poly_cost(net, 0, "load", cp1_eur_per_mw = 0.1)
    """

    if index is None:
        index = get_free_id(net["poly_cost"])
    columns = ["element", "et", "cp0_eur", "cp1_eur_per_mw", "cq0_eur", "cq1_eur_per_mvar",
               "cp2_eur_per_mw2", "cq2_eur_per_mvar2"]
    variables = [element, et, cp0_eur, cp1_eur_per_mw, cq0_eur, cq1_eur_per_mvar,
                 cp2_eur_per_mw2, cq2_eur_per_mvar2]
    net.poly_cost.loc[index, columns] = variables
    return index

