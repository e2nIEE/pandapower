# -*- coding: utf-8 -*-

# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import numpy.core.numeric as ncn
from numpy import array,  zeros, isnan
from pandas import DataFrame
from pandapower.idx_bus import PV, REF, VA, VM, BUS_TYPE, NONE, VMAX, VMIN, PQ
from pandapower.idx_gen import QMIN, QMAX, PMIN, PMAX, GEN_STATUS, GEN_BUS, PG, VG, QG


def _build_gen_ppc(net, ppc):
    '''
    Takes the empty ppc network and fills it with the gen values. The gen
    datatype will be float afterwards.

    **INPUT**:
        **net** -The pandapower format network

        **ppc** - The PYPOWER format network to fill in values
    '''

    mode = net["_options"]["mode"]

    # if mode == power flow or short circuit...
    if mode == "pf" or mode == "sc":

        # get in service elements
        _is_elements = net["_is_elements"]
        eg_is_mask = _is_elements['ext_grid']
        gen_is_mask = _is_elements['gen']

        eg_end = np.sum(eg_is_mask)
        gen_end = eg_end + np.sum(gen_is_mask)
        xw_end = gen_end + len(net["xward"])

        # define default q limits
        q_lim_default = 1e9  # which is 1000 TW - should be enough for distribution grids.
        p_lim_default = 1e9

        _init_ppc_gen(ppc, xw_end, 0)
        if mode == "sc":
            return
        # add generator / pv data
        if gen_end > eg_end:
            _build_pp_gen(net, ppc, gen_is_mask, eg_end, gen_end, q_lim_default, p_lim_default)

        _build_pp_ext_grid(net, ppc, eg_is_mask, eg_end)

        # add extended ward pv node data
        if xw_end > gen_end:
            _build_pp_xward(net, ppc, gen_end, xw_end, q_lim_default)

    # if mode == optimal power flow...
    if mode == "opf":

        bus_lookup = net["_pd2ppc_lookups"]["bus"]
        calculate_voltage_angles = net["_options"]["calculate_voltage_angles"]

        if len(net.dcline) > 0:
            ppc["dcline"] = net.dcline[["loss_kw", "loss_percent"]].values
        # get in service elements
        _is_elements = net["_is_elements"]
        eg_is = net["ext_grid"][_is_elements['ext_grid']]
        gen_is = net["gen"][_is_elements['gen']]
        sg_is = net.sgen[(net.sgen.in_service & net.sgen.controllable) == True] \
            if "controllable" in net.sgen.columns else DataFrame()
        l_is = net.load[(net.load.in_service & net.load.controllable) == True] \
            if "controllable" in net.load.columns else DataFrame()
        stor_is = net.storage[(net.storage.in_service & net.storage.controllable) == True] \
            if "controllable" in net.storage.columns else DataFrame()

        _is_elements["sgen_controllable"] = sg_is
        _is_elements["load_controllable"] = l_is
        _is_elements["storage_controllable"] = stor_is
        eg_end = len(eg_is)
        gen_end = eg_end + len(gen_is)
        sg_end = gen_end + len(sg_is)
        l_end = sg_end + len(l_is)
        stor_end = l_end + len(stor_is)

        q_lim_default = 1e9  # which is 1000 TW - should be enough for distribution grids.
        p_lim_default = 1e9  # changes must be considered in check_opf_data
        delta = net["_options"]["delta"]

        # initialize generator matrix
        ppc["gen"] = zeros(shape=(stor_end, 21), dtype=float)
        ppc["gen"][:] = array([0, 0, 0, q_lim_default, -q_lim_default, 1., 1., 1, p_lim_default,
                                  -p_lim_default, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        # add sgens first so pv bus types won't be overwritten
        if sg_end > gen_end:
            gen_buses = bus_lookup[sg_is["bus"].values]

            ppc["gen"][gen_end:sg_end, GEN_BUS] = gen_buses
            ppc["gen"][gen_end:sg_end, PG] = - sg_is["p_kw"].values * 1e-3 * sg_is["scaling"].values
            ppc["gen"][gen_end:sg_end, QG] = sg_is["q_kvar"].values * 1e-3 * sg_is["scaling"].values

            # set bus values for generator buses
            ppc["bus"][gen_buses, BUS_TYPE] = PQ

            # set constraints for controllable sgens
            if "min_q_kvar" in sg_is.columns:
                ppc["gen"][gen_end:sg_end, QMAX] = - (sg_is["min_q_kvar"].values * 1e-3 - delta)
                max_q_kvar = ppc["gen"][gen_end:sg_end, [QMAX]]
                ncn.copyto(max_q_kvar, -q_lim_default, where=isnan(max_q_kvar))
                ppc["gen"][gen_end:sg_end, [QMAX]] = max_q_kvar

            if "max_q_kvar" in sg_is.columns:
                ppc["gen"][gen_end:sg_end, QMIN] = - (sg_is["max_q_kvar"].values * 1e-3 + delta)
                min_q_kvar = ppc["gen"][gen_end:sg_end, [QMIN]]
                ncn.copyto(min_q_kvar, q_lim_default, where=isnan(min_q_kvar))
                ppc["gen"][gen_end:sg_end, [QMIN]] = min_q_kvar

            if "max_p_kw" in sg_is.columns:
                ppc["gen"][gen_end:sg_end, PMIN] = - (sg_is["max_p_kw"].values * 1e-3 + delta)
                max_p_kw = ppc["gen"][gen_end:sg_end, [PMIN]]
                ncn.copyto(max_p_kw, -p_lim_default, where=isnan(max_p_kw))
                ppc["gen"][gen_end:sg_end, [PMIN]] = max_p_kw

            if "min_p_kw" in sg_is.columns:
                ppc["gen"][gen_end:sg_end, PMAX] = - (sg_is["min_p_kw"].values * 1e-3 - delta)
                min_p_kw = ppc["gen"][gen_end:sg_end, [PMAX]]
                ncn.copyto(min_p_kw, p_lim_default, where=isnan(min_p_kw))
                ppc["gen"][gen_end:sg_end, [PMAX]] = min_p_kw

        # add controllable loads
        if l_end > sg_end:
            load_buses = bus_lookup[l_is["bus"].values]

            ppc["gen"][sg_end:l_end, GEN_BUS] = load_buses
            ppc["gen"][sg_end:l_end, PG] = - l_is["p_kw"].values * 1e-3 * l_is["scaling"].values
            ppc["gen"][sg_end:l_end, QG] = l_is["q_kvar"].values * 1e-3 * l_is["scaling"].values

            # set bus values for controllable loads
            ppc["bus"][load_buses, BUS_TYPE] = PQ

            # set constraints for controllable loads
            if "min_q_kvar" in l_is.columns:
                ppc["gen"][sg_end:l_end, QMAX] = - (l_is["min_q_kvar"].values * 1e-3 - delta)
                max_q_kvar = ppc["gen"][sg_end:l_end, [QMAX]]
                ncn.copyto(max_q_kvar, -q_lim_default, where=isnan(max_q_kvar))
                ppc["gen"][sg_end:l_end, [QMAX]] = max_q_kvar

            if "max_q_kvar" in l_is.columns:
                ppc["gen"][sg_end:l_end, QMIN] = - (l_is["max_q_kvar"].values * 1e-3 + delta)
                min_q_kvar = ppc["gen"][sg_end:l_end, [QMIN]]
                ncn.copyto(min_q_kvar, q_lim_default, where=isnan(min_q_kvar))
                ppc["gen"][sg_end:l_end, [QMIN]] = min_q_kvar

            if "min_p_kw" in l_is.columns:
                ppc["gen"][sg_end:l_end, PMIN] = - (l_is["max_p_kw"].values * 1e-3 + delta)
                max_p_kw = ppc["gen"][sg_end:l_end, [PMIN]]
                ncn.copyto(max_p_kw, -p_lim_default, where=isnan(max_p_kw))
                ppc["gen"][sg_end:l_end, [PMIN]] = max_p_kw

            if "max_p_kw" in l_is.columns:
                ppc["gen"][sg_end:l_end, PMAX] = - (l_is["min_p_kw"].values * 1e-3 - delta)
                min_p_kw = ppc["gen"][sg_end:l_end, [PMAX]]
                ncn.copyto(min_p_kw, p_lim_default, where=isnan(min_p_kw))
                ppc["gen"][sg_end:l_end, [PMAX]] = min_p_kw

        # add controllable storages
        if stor_end > l_end:
            stor_buses = bus_lookup[stor_is["bus"].values]

            ppc["gen"][l_end:stor_end, GEN_BUS] = stor_buses
            ppc["gen"][l_end:stor_end, PG] = - stor_is["p_kw"].values * 1e-3 * stor_is["scaling"].values
            ppc["gen"][l_end:stor_end, QG] = stor_is["q_kvar"].values * 1e-3 * stor_is["scaling"].values

            # set bus values for generator buses
            ppc["bus"][stor_buses, BUS_TYPE] = PQ

            # set constraints for controllable sgens
            if "min_q_kvar" in stor_is.columns:
                ppc["gen"][l_end:stor_end, QMAX] = - (stor_is["min_q_kvar"].values * 1e-3 - delta)
                max_q_kvar = ppc["gen"][l_end:stor_end, [QMAX]]
                ncn.copyto(max_q_kvar, -q_lim_default, where=isnan(max_q_kvar))
                ppc["gen"][l_end:stor_end, [QMIN]] = max_q_kvar

            if "max_q_kvar" in stor_is.columns:
                ppc["gen"][l_end:stor_end, QMIN] = - (stor_is["max_q_kvar"].values * 1e-3 + delta)
                min_q_kvar = ppc["gen"][l_end:stor_end, [QMIN]]
                ncn.copyto(min_q_kvar, q_lim_default, where=isnan(min_q_kvar))
                ppc["gen"][l_end:stor_end, [QMIN]] = min_q_kvar

            if "max_p_kw" in stor_is.columns:
                ppc["gen"][l_end:stor_end, PMIN] = - (stor_is["max_p_kw"].values * 1e-3 + delta)
                max_p_kw = ppc["gen"][l_end:stor_end, [PMIN]]
                ncn.copyto(max_p_kw, -p_lim_default, where=isnan(max_p_kw))
                ppc["gen"][l_end:stor_end, [PMIN]] = max_p_kw

            if "min_p_kw" in stor_is.columns:
                ppc["gen"][l_end:stor_end, PMAX] = - (stor_is["min_p_kw"].values * 1e-3 - delta)
                min_p_kw = ppc["gen"][l_end:stor_end, [PMAX]]
                ncn.copyto(min_p_kw, p_lim_default, where=isnan(min_p_kw))
                ppc["gen"][l_end:stor_end, [PMAX]] = min_p_kw

        # add ext grid / slack data
        ppc["gen"][:eg_end, GEN_BUS] = bus_lookup[eg_is["bus"].values]
        ppc["gen"][:eg_end, VG] = eg_is["vm_pu"].values
        ppc["gen"][:eg_end, GEN_STATUS] = eg_is["in_service"].values
        if "max_p_kw" in eg_is.columns:
            ppc["gen"][:eg_end, PMIN] = - (eg_is["max_p_kw"].values * 1e-3 - delta)
            max_p_kw = ppc["gen"][:eg_end, [PMIN]]
            ncn.copyto(max_p_kw, -p_lim_default, where=isnan(max_p_kw))
            ppc["gen"][:eg_end, [PMIN]] = max_p_kw

        if "min_p_kw" in eg_is.columns:
            ppc["gen"][:eg_end, PMAX] = - (eg_is["min_p_kw"].values * 1e-3 + delta)
            min_p_kw = ppc["gen"][:eg_end, [PMAX]]
            ncn.copyto(min_p_kw, p_lim_default, where=isnan(min_p_kw))
            ppc["gen"][:eg_end, [PMAX]] = min_p_kw

        if "min_q_kvar" in eg_is.columns:
            ppc["gen"][:eg_end, QMAX] = - (eg_is["min_q_kvar"].values * 1e-3 - delta)
            max_q_kvar = ppc["gen"][:eg_end, [QMAX]]
            ncn.copyto(max_q_kvar, -q_lim_default, where=isnan(max_q_kvar))
            ppc["gen"][:eg_end, [QMAX]] = max_q_kvar

        if "max_q_kvar" in eg_is.columns:
            ppc["gen"][:eg_end, QMIN] = - (eg_is["max_q_kvar"].values * 1e-3 + delta)
            min_q_kvar = ppc["gen"][:eg_end, [QMIN]]
            ncn.copyto(min_q_kvar, q_lim_default, where=isnan(min_q_kvar))
            ppc["gen"][:eg_end, [QMIN]] = min_q_kvar

        # set bus values for external grid buses
        eg_buses = bus_lookup[eg_is["bus"].values]
        if calculate_voltage_angles:
            ppc["bus"][eg_buses, VA] = eg_is["va_degree"].values
        ppc["bus"][eg_buses, BUS_TYPE] = REF
        ppc["bus"][eg_buses, VM] = eg_is["vm_pu"].values

        # REF busses don't have flexible voltages by definition:
        ppc["bus"][eg_buses, VMAX] = ppc["bus"][ppc["bus"][:, BUS_TYPE] == REF, VM]
        ppc["bus"][eg_buses, VMIN] = ppc["bus"][ppc["bus"][:, BUS_TYPE] == REF, VM]

        # add generator / pv data
        if gen_end > eg_end:
            ppc["gen"][eg_end:gen_end, GEN_BUS] = bus_lookup[gen_is["bus"].values]
            ppc["gen"][eg_end:gen_end, PG] = - gen_is["p_kw"].values * 1e-3 * gen_is["scaling"].values
            ppc["gen"][eg_end:gen_end, VG] = gen_is["vm_pu"].values

            # set bus values for generator buses
            gen_buses = bus_lookup[gen_is["bus"].values]
            ppc["bus"][gen_buses, BUS_TYPE] = PV
            ppc["bus"][gen_buses, VM] = gen_is["vm_pu"].values

            # set constraints for PV generators
            _copy_q_limits_to_ppc(net, ppc, eg_end, gen_end, _is_elements['gen'])
            _copy_p_limits_to_ppc(net, ppc, eg_end, gen_end, _is_elements['gen'])

            _replace_nans_with_default_q_limits_in_ppc(ppc, eg_end, gen_end, q_lim_default)
            _replace_nans_with_default_p_limits_in_ppc(ppc, eg_end, gen_end, p_lim_default)


def _init_ppc_gen(ppc, xw_end, q_lim_default):
    # initialize generator matrix
    ppc["gen"] = np.zeros(shape=(xw_end, 21), dtype=float)
    ppc["gen"][:] = np.array([0, 0, 0, q_lim_default, -q_lim_default, 1.,
                              1., 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


def _build_pp_ext_grid(net, ppc, eg_is_mask, eg_end):
    calculate_voltage_angles = net["_options"]["calculate_voltage_angles"]
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    # add ext grid / slack data
    eg_buses = bus_lookup[net["ext_grid"]["bus"].values[eg_is_mask]]
    ppc["gen"][:eg_end, GEN_BUS] = eg_buses
    ppc["gen"][:eg_end, VG] = net["ext_grid"]["vm_pu"].values[eg_is_mask]
    ppc["gen"][:eg_end, GEN_STATUS] = True

    # set bus values for external grid buses
    if calculate_voltage_angles:
        ppc["bus"][eg_buses, VA] = net["ext_grid"]["va_degree"].values[eg_is_mask]
    ppc["bus"][eg_buses, BUS_TYPE] = REF
    # _build_gen_lookups(net, "ext_grid", 0, eg_end)


def _build_pp_gen(net, ppc, gen_is_mask, eg_end, gen_end, q_lim_default, p_lim_default):

    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    copy_constraints_to_ppc = net["_options"]["copy_constraints_to_ppc"]

    gen_buses = bus_lookup[net["gen"]["bus"].values[gen_is_mask]]
    gen_is_vm = net["gen"]["vm_pu"].values[gen_is_mask]
    ppc["gen"][eg_end:gen_end, GEN_BUS] = gen_buses
    ppc["gen"][eg_end:gen_end, PG] = - (net["gen"]["p_kw"].values[gen_is_mask] * 1e-3 *
                                        net["gen"]["scaling"].values[gen_is_mask])
    ppc["gen"][eg_end:gen_end, VG] = gen_is_vm

    # set bus values for generator buses

    ppc["bus"][gen_buses, BUS_TYPE] = PV
    ppc["bus"][gen_buses, VM] = gen_is_vm

    _copy_q_limits_to_ppc(net, ppc, eg_end, gen_end, gen_is_mask)
    _replace_nans_with_default_q_limits_in_ppc(ppc, eg_end, gen_end, q_lim_default)

    if copy_constraints_to_ppc:
        _copy_p_limits_to_ppc(net, ppc, eg_end, gen_end, gen_is_mask)
        _replace_nans_with_default_p_limits_in_ppc(ppc, eg_end, gen_end, p_lim_default)

    # _build_gen_lookups(net, "gen", eg_end, gen_end)


def _build_pp_xward(net, ppc, gen_end, xw_end, q_lim_default, update_lookup=True):
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    xw = net["xward"]
    xw_is = net["_is_elements"]['xward']
    if update_lookup:
        ppc["gen"][gen_end:xw_end, GEN_BUS] = bus_lookup[xw["ad_bus"].values]
    ppc["gen"][gen_end:xw_end, VG] = xw["vm_pu"].values
    ppc["gen"][gen_end:xw_end, GEN_STATUS] = xw_is
    ppc["gen"][gen_end:xw_end, QMIN] = -q_lim_default
    ppc["gen"][gen_end:xw_end, QMAX] = q_lim_default

    xward_buses = bus_lookup[net["xward"]["ad_bus"].values]
    ppc["bus"][xward_buses[xw_is], BUS_TYPE] = PV
    ppc["bus"][xward_buses[~xw_is], BUS_TYPE] = NONE
    ppc["bus"][xward_buses, VM] = net["xward"]["vm_pu"].values




def _update_gen_ppc(net, ppc):
    '''
    Takes the ppc network and updates the gen values from the values in net.

    **INPUT**:
        **net** -The pandapower format network

        **ppc** - The PYPOWER format network to fill in values
    '''
    # get options from net
    calculate_voltage_angles = net["_options"]["calculate_voltage_angles"]
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    # get in service elements
    _is_elements = net["_is_elements"]
    gen_is_mask = _is_elements['gen']
    # TODO maybe speed up things here, too
    eg_is = net["ext_grid"][_is_elements['ext_grid']]
    gen_is = net["gen"][_is_elements['gen']]

    eg_end = len(eg_is)
    gen_end = eg_end + len(gen_is)
    xw_end = gen_end + len(net["xward"])

    q_lim_default = 1e9  # which is 1000 TW - should be enough for distribution grids.

    # add ext grid / slack data
    ext_grid_lookup = net["_pd2ppc_lookups"]["ext_grid"]
    ext_grid_idx_ppc = ext_grid_lookup[eg_is.index]
    ppc["gen"][ext_grid_idx_ppc, VG] = eg_is["vm_pu"].values
    ppc["gen"][ext_grid_idx_ppc, GEN_STATUS] = eg_is["in_service"].values

    # set bus values for external grid buses
    if calculate_voltage_angles:
        # eg_buses = bus_lookup[eg_is["bus"].values]
        ppc["bus"][ext_grid_idx_ppc, VA] = eg_is["va_degree"].values

    # add generator / pv data
    if gen_end > eg_end:
        gen_lookup = net["_pd2ppc_lookups"]["gen"]
        gen_idx_ppc = gen_lookup[gen_is.index]
        ppc["gen"][gen_idx_ppc, PG] = - gen_is["p_kw"].values * 1e-3 * gen_is["scaling"].values
        ppc["gen"][gen_idx_ppc, VG] = gen_is["vm_pu"].values

        # set bus values for generator buses
        gen_buses = bus_lookup[gen_is["bus"].values]
        ppc["bus"][gen_buses, VM] = gen_is["vm_pu"].values

        _copy_q_limits_to_ppc(net, ppc, eg_end, gen_end, gen_is_mask)
        _replace_nans_with_default_q_limits_in_ppc(ppc, eg_end, gen_end, q_lim_default)

    # add extended ward pv node data
    if xw_end > gen_end:
        # ToDo: this must be tested in combination with recycle. Maybe the placement of the updated value in ppc["gen"]
        # ToDo: is wrong. -> I'll better raise en error
        raise NotImplementedError("xwards in combination with recycle is not properly implemented")
        # _build_pp_xward(net, ppc, gen_end, xw_end, q_lim_default,
        #                           update_lookup=False)


def _copy_q_limits_to_ppc(net, ppc, eg_end, gen_end, gen_is_mask):
    # Note: Pypower has generator reference system, pandapower uses load reference
    # system (max <-> min)

    delta = net["_options"]["delta"]

    if "max_q_kvar" in net["gen"].columns:
        ppc["gen"][eg_end:gen_end, QMIN] = -net["gen"]["max_q_kvar"].values[gen_is_mask] * 1e-3 - delta
    if "min_q_kvar" in net["gen"].columns:
        ppc["gen"][eg_end:gen_end, QMAX] = -net["gen"]["min_q_kvar"].values[gen_is_mask] * 1e-3 + delta


def _copy_p_limits_to_ppc(net, ppc, eg_end, gen_end, gen_is_mask):
    delta = net["_options"]["delta"]

    if "max_p_kw" in net["gen"].columns:
        ppc["gen"][eg_end:gen_end, PMIN] = -net["gen"]["max_p_kw"].values[gen_is_mask] * 1e-3 + delta
    if "min_p_kw" in net["gen"].columns:
        ppc["gen"][eg_end:gen_end, PMAX] = -net["gen"]["min_p_kw"].values[gen_is_mask] * 1e-3 - delta


def _replace_nans_with_default_q_limits_in_ppc(ppc, eg_end, gen_end, q_lim_default):
    # Note: Pypower has generator reference system, pandapower uses load reference system (max <-> min)
    max_q_kvar = ppc["gen"][eg_end:gen_end, [QMIN]]
    ncn.copyto(max_q_kvar, -q_lim_default, where=np.isnan(max_q_kvar))
    ppc["gen"][eg_end:gen_end, [QMIN]] = max_q_kvar

    min_q_kvar = ppc["gen"][eg_end:gen_end, [QMAX]]
    ncn.copyto(min_q_kvar, q_lim_default, where=np.isnan(min_q_kvar))
    ppc["gen"][eg_end:gen_end, [QMAX]] = min_q_kvar


def _replace_nans_with_default_p_limits_in_ppc(ppc, eg_end, gen_end, p_lim_default):
    # Note: Pypower has generator reference system, pandapower uses load reference system (max <-> min)
    max_p_kw = ppc["gen"][eg_end:gen_end, [PMIN]]
    ncn.copyto(max_p_kw, -p_lim_default, where=isnan(max_p_kw))
    ppc["gen"][eg_end:gen_end, [PMIN]] = max_p_kw

    min_p_kw = ppc["gen"][eg_end:gen_end, [PMAX]]
    ncn.copyto(min_p_kw, p_lim_default, where=isnan(min_p_kw))
    ppc["gen"][eg_end:gen_end, [PMAX]] = min_p_kw


def _check_voltage_setpoints_at_same_bus(ppc):
    # generator buses:
    gen_bus = ppc['gen'][:, GEN_BUS].astype(int)
    # generator setpoints:
    gen_vm = ppc['gen'][:, VG]
    if _different_values_at_one_bus(gen_bus, gen_vm):
        raise UserWarning("Generators with different voltage setpoints connected to the same bus")

def _check_voltage_angles_at_same_bus(net, ppc):
    gen_va = net.ext_grid.va_degree[net._is_elements["ext_grid"]].values
    eg_gens = net._pd2ppc_lookups["ext_grid"][net.ext_grid.index[net._is_elements["ext_grid"]]]
    gen_bus = ppc["gen"][eg_gens, GEN_BUS].astype(int)
    if _different_values_at_one_bus(gen_bus, gen_va):
        raise UserWarning("Ext grids with different voltage angle setpoints connected to the same bus")


def _different_values_at_one_bus(buses, values):
    """
    checks if there are different values in any of the

    """
    # buses with one or more generators and their index
    unique_bus, index_first_bus = np.unique(buses, return_index=True)

    # voltage setpoint lookup with the voltage of the first occurence of that bus
    first_values = -np.ones(buses.max() + 1)
    first_values[unique_bus] = values[index_first_bus]

    # generate voltage setpoints where all generators at the same bus
    # have the voltage of the first generator at that bus
    values_equal = first_values[buses]

    return not np.allclose(values, values_equal)
