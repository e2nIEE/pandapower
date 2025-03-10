# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np

from pandapower.pf.ppci_variables import bustypes
from pandapower.pypower.bustypes import bustypes_dc
from pandapower.pypower.idx_bus import PV, REF, VA, VM, BUS_TYPE, NONE, VMAX, VMIN, SL_FAC as SL_FAC_BUS
from pandapower.pypower.idx_bus_dc import DC_BUS_TYPE, DC_NONE
from pandapower.pypower.idx_gen import QMIN, QMAX, PMIN, PMAX, GEN_BUS, PG, VG, QG, MBASE, SL_FAC, gen_cols
from pandapower.pypower.idx_brch import F_BUS, T_BUS
from pandapower.auxiliary import _subnetworks, _sum_by_group
from pandapower.pypower.idx_ssc import SSC_BUS, SSC_SET_VM_PU, SSC_CONTROLLABLE
from pandapower.pypower.idx_vsc import VSC_MODE_AC, VSC_BUS, VSC_VALUE_AC, VSC_CONTROLLABLE, VSC_MODE_AC_V, \
    VSC_MODE_AC_SL

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def _build_gen_ppc(net, ppc):
    '''
    Takes the empty ppc network and fills it with the gen values. The gen
    datatype will be float afterwards.

    **INPUT**:
        **net** -The pandapower format network

        **ppc** - The PYPOWER format network to fill in values
    '''

    mode = net["_options"]["mode"]
    distributed_slack = net["_options"]["distributed_slack"]

    _is_elements = net["_is_elements"]
    gen_order = dict()
    f = 0
    for element in ["ext_grid", "gen"]:
        f = add_gen_order(gen_order, element, _is_elements, f)

    if mode == "opf":
        if len(net.dcline) > 0:
            ppc["dcline"] = net.dcline[["loss_mw", "loss_percent"]].values
        for element in ["sgen_controllable", "load_controllable", "storage_controllable"]:
            f = add_gen_order(gen_order, element, _is_elements, f)

    f = add_gen_order(gen_order, "xward", _is_elements, f)

    _init_ppc_gen(net, ppc, f)
    for element, (f, t) in gen_order.items():
        add_element_to_gen(net, ppc, element, f, t)
    net._gen_order = gen_order

    if distributed_slack:
        # we add the slack weights of the xward elements to the PQ bus and not the PV bus,
        # that is why we to treat the xward as a special case
        xward_pq_buses = _get_xward_pq_buses(net, ppc)
        gen_mask, xward_mask = _gen_xward_mask(net, ppc)
        _normalise_slack_weights(ppc, gen_mask, xward_mask, xward_pq_buses)


def add_gen_order(gen_order, element, _is_elements, f):
    if element in _is_elements and _is_elements[element].any():
        i = np.sum(_is_elements[element])
        gen_order[element] = (f, f + i)
        f += i
    return f


def _init_ppc_gen(net, ppc, nr_gens):
    # initialize generator matrix
    ppc["gen"] = np.zeros(shape=(nr_gens, gen_cols), dtype=np.float64)
    ppc["gen"][:] = np.array([0, 0, 0, 0, 0, 1.,
                              1., 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0])
    q_lim_default = net._options["p_lim_default"]
    p_lim_default = net._options["p_lim_default"]
    ppc["gen"][:, PMAX] = p_lim_default
    ppc["gen"][:, PMIN] = -p_lim_default
    ppc["gen"][:, QMAX] = q_lim_default
    ppc["gen"][:, QMIN] = -q_lim_default


def add_element_to_gen(net, ppc, element, f, t):
    if element == "ext_grid":
        _build_pp_ext_grid(net, ppc, f, t)
    elif element == "gen":
        _build_pp_gen(net, ppc, f, t)
    elif element == "sgen_controllable":
        _build_pp_pq_element(net, ppc, "sgen", f, t)
    elif element == "load_controllable":
        _build_pp_pq_element(net, ppc, "load", f, t, inverted=True)
    elif element == "storage_controllable":
        _build_pp_pq_element(net, ppc, "storage", f, t, inverted=True)
    elif element == "xward":
        _build_pp_xward(net, ppc, f, t)
    else:
        raise ValueError("Unknown element %s" % element)


def _build_pp_ext_grid(net, ppc, f, t):
    delta = net._options["delta"]
    eg_is = net._is_elements["ext_grid"]
    calculate_voltage_angles = net["_options"]["calculate_voltage_angles"]
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    # add ext grid / slack data
    eg_buses = bus_lookup[net["ext_grid"]["bus"].values[eg_is]]
    ppc["gen"][f:t, GEN_BUS] = eg_buses
    ppc["gen"][f:t, VG] = net["ext_grid"]["vm_pu"].values[eg_is]
    ppc["gen"][f:t, SL_FAC] = net["ext_grid"]["slack_weight"].values[eg_is]

    # set bus values for external grid buses
    if ppc.get("sequence", 1) == 1:
        if calculate_voltage_angles:
            ppc["bus"][eg_buses, VA] = net["ext_grid"]["va_degree"].values[eg_is]
        ppc["bus"][eg_buses, VM] = net["ext_grid"]["vm_pu"].values[eg_is]
    if net._options["mode"] == "opf":
        add_q_constraints(net, "ext_grid", eg_is, ppc, f, t, delta)
        add_p_constraints(net, "ext_grid", eg_is, ppc, f, t, delta)

        if "controllable" in net["ext_grid"]:
            # if we do and one of them is false, do this only for the ones, where it is false
            eg_constrained = net.ext_grid[eg_is][~net.ext_grid.controllable]
            if len(eg_constrained):
                eg_constrained_bus_ppc = [bus_lookup[egb] for egb in eg_constrained.bus.values]
                ppc["bus"][eg_constrained_bus_ppc, VMAX] = net["ext_grid"]["vm_pu"].values[eg_constrained.index] + delta
                ppc["bus"][eg_constrained_bus_ppc, VMIN] = net["ext_grid"]["vm_pu"].values[eg_constrained.index] - delta
        else:
            # if we dont:
            ppc["bus"][eg_buses, VMAX] = net["ext_grid"]["vm_pu"].values[eg_is] + delta
            ppc["bus"][eg_buses, VMIN] = net["ext_grid"]["vm_pu"].values[eg_is] - delta
    else:
        ppc["gen"][f:t, QMIN] = 0
        ppc["gen"][f:t, QMAX] = 0


def _check_gen_vm_limits(net, ppc, gen_buses, gen_is):
    # check vm_pu limit violation
    v_max_bound = ppc["bus"][gen_buses, VMAX] < net["gen"]["vm_pu"].values[gen_is]
    if np.any(v_max_bound):
        bound_gens = net["gen"].index.values[gen_is][v_max_bound]
        logger.warning("gen vm_pu > bus max_vm_pu for gens {}. "
                       "Setting bus limit for these gens.".format(bound_gens))

    v_min_bound = net["gen"]["vm_pu"].values[gen_is] < ppc["bus"][gen_buses, VMIN]
    if np.any(v_min_bound):
        bound_gens = net["gen"].index.values[gen_is][v_min_bound]
        logger.warning("gen vm_pu < bus min_vm_pu for gens {}. "
                       "Setting bus limit for these gens.".format(bound_gens))

    # check max_vm_pu / min_vm_pu limit violation
    if "max_vm_pu" in net["gen"].columns:
        v_max_bound = ppc["bus"][gen_buses, VMAX] < net["gen"]["max_vm_pu"].values[gen_is]
        if np.any(v_max_bound):
            bound_gens = net["gen"].index.values[gen_is][v_max_bound]
            logger.warning("gen max_vm_pu > bus max_vm_pu for gens {}. "
                           "Setting bus limit for these gens.".format(bound_gens))
            # set only vm of gens which do not violate the limits
            ppc["bus"][gen_buses[~v_max_bound], VMAX] = net["gen"]["max_vm_pu"].values[gen_is][~v_max_bound]
        else:
            # set vm of all gens
            ppc["bus"][gen_buses, VMAX] = net["gen"]["max_vm_pu"].values[gen_is]

    if "min_vm_pu" in net["gen"].columns:
        v_min_bound = net["gen"]["min_vm_pu"].values[gen_is] < ppc["bus"][gen_buses, VMIN]
        if np.any(v_min_bound):
            bound_gens = net["gen"].index.values[gen_is][v_min_bound]
            logger.warning("gen min_vm_pu < bus min_vm_pu for gens {}. "
                           "Setting bus limit for these gens.".format(bound_gens))
            # set only vm of gens which do not violate the limits
            ppc["bus"][gen_buses[~v_max_bound], VMIN] = net["gen"]["min_vm_pu"].values[gen_is][~v_min_bound]
        else:
            # set vm of all gens
            ppc["bus"][gen_buses, VMIN] = net["gen"]["min_vm_pu"].values[gen_is]
    return ppc


def _enforce_controllable_vm_pu_p_mw(net, ppc, gen_is, f, t):
    delta = net["_options"]["delta"]
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    controllable = net["gen"]["controllable"].values[gen_is]
    not_controllable = ~controllable.astype(bool)

    # if there are some non controllable gens -> set vm_pu and p_mw fixed
    if np.any(not_controllable):
        bus = net["gen"]["bus"].values[not_controllable]
        vm_pu = net["gen"]["vm_pu"].values[not_controllable]
        p_mw = net["gen"]["p_mw"].values[not_controllable]

        not_controllable_buses = bus_lookup[bus]
        ppc["bus"][not_controllable_buses, VMAX] = vm_pu + delta
        ppc["bus"][not_controllable_buses, VMIN] = vm_pu - delta

        not_controllable_gens = np.arange(f, t)[not_controllable]
        ppc["gen"][not_controllable_gens, PMIN] = p_mw - delta
        ppc["gen"][not_controllable_gens, PMAX] = p_mw + delta
    return ppc


def _build_pp_gen(net, ppc, f, t):
    delta = net["_options"]["delta"]
    gen_is = net._is_elements["gen"]
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    mode = net["_options"]["mode"]

    gen_buses = bus_lookup[net["gen"]["bus"].values[gen_is]]
    gen_is_vm = net["gen"]["vm_pu"].values[gen_is]
    ppc["gen"][f:t, GEN_BUS] = gen_buses
    ppc["gen"][f:t, PG] = (net["gen"]["p_mw"].values[gen_is] * net["gen"]["scaling"].values[gen_is])
    ppc["gen"][f:t, MBASE] = net["gen"]["sn_mva"].values[gen_is]
    ppc["gen"][f:t, SL_FAC] = net["gen"]["slack_weight"].values[gen_is]
    ppc["gen"][f:t, VG] = gen_is_vm

    # set bus values for generator buses
    ppc["bus"][gen_buses[ppc["bus"][gen_buses, BUS_TYPE] != REF], BUS_TYPE] = PV
    if mode != "se":
        ppc["bus"][gen_buses, VM] = gen_is_vm

    add_q_constraints(net, "gen", gen_is, ppc, f, t, delta)
    add_p_constraints(net, "gen", gen_is, ppc, f, t, delta)
    if mode == "opf":
        # this considers the vm limits for gens
        ppc = _check_gen_vm_limits(net, ppc, gen_buses, gen_is)
        if "controllable" in net.gen.columns:
            ppc = _enforce_controllable_vm_pu_p_mw(net, ppc, gen_is, f, t)


def _build_pp_xward(net, ppc, f, t, update_lookup=True):
    delta = net["_options"]["delta"]
    q_lim_default = net._options["q_lim_default"]
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    aux_buses = net["_pd2ppc_lookups"]["aux"]["xward"]
    xw = net["xward"]
    xw_is = net["_is_elements"]['xward']
    ppc["gen"][f:t, GEN_BUS] = bus_lookup[aux_buses[xw_is]]
    ppc["gen"][f:t, VG] = xw["vm_pu"][xw_is].values
    ppc["gen"][f:t, SL_FAC] = net["xward"]["slack_weight"].values[xw_is]
    ppc["gen"][f:t, PMIN] = - delta
    ppc["gen"][f:t, PMAX] = + delta
    ppc["gen"][f:t, QMIN] = -q_lim_default
    ppc["gen"][f:t, QMAX] = q_lim_default

    xward_buses = bus_lookup[aux_buses]
    ppc["bus"][xward_buses[xw_is], BUS_TYPE] = PV
    ppc["bus"][xward_buses[~xw_is], BUS_TYPE] = NONE
    ppc["bus"][xward_buses, VM] = net["xward"]["vm_pu"].values


def _build_pp_pq_element(net, ppc, element, f, t, inverted=False):
    delta = net._options["delta"]
    sign = -1 if inverted else 1
    is_element = net._is_elements["%s_controllable" % element]
    tab = net[element]
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    buses = bus_lookup[tab["bus"].values[is_element]]

    ppc["gen"][f:t, GEN_BUS] = buses
    if "sn_mva" in tab:
        ppc["gen"][f:t, MBASE] = tab["sn_mva"].values[is_element]
    ppc["gen"][f:t, PG] = sign * tab["p_mw"].values[is_element] * tab["scaling"].values[is_element]
    ppc["gen"][f:t, QG] = sign * tab["q_mvar"].values[is_element] * tab["scaling"].values[is_element]

    # set bus values for controllable loads
    #    ppc["bus"][buses, BUS_TYPE] = PQ
    add_q_constraints(net, element, is_element, ppc, f, t, delta, inverted)
    add_p_constraints(net, element, is_element, ppc, f, t, delta, inverted)


def add_q_constraints(net, element, is_element, ppc, f, t, delta, inverted=False):
    tab = net[element]
    if "min_q_mvar" in tab.columns:
        if inverted:
            ppc["gen"][f:t, QMAX] = -tab["min_q_mvar"].values[is_element] + delta
        else:
            ppc["gen"][f:t, QMIN] = tab["min_q_mvar"].values[is_element] - delta
    if "max_q_mvar" in tab.columns:
        if inverted:
            ppc["gen"][f:t, QMIN] = -tab["max_q_mvar"].values[is_element] - delta
        else:
            ppc["gen"][f:t, QMAX] = tab["max_q_mvar"].values[is_element] + delta


def add_p_constraints(net, element, is_element, ppc, f, t, delta, inverted=False):
    tab = net[element]
    if "min_p_mw" in tab.columns:
        if inverted:
            ppc["gen"][f:t, PMAX] = - tab["min_p_mw"].values[is_element] + delta
        else:
            ppc["gen"][f:t, PMIN] = tab["min_p_mw"].values[is_element] - delta
    if "max_p_mw" in tab.columns:
        if inverted:
            ppc["gen"][f:t, PMIN] = - tab["max_p_mw"].values[is_element] - delta
        else:
            ppc["gen"][f:t, PMAX] = tab["max_p_mw"].values[is_element] + delta


def _check_voltage_setpoints_at_same_bus(ppc):
    """
    Checks if voltage-controlling elements (generators, SSC, VSC) at the same bus have different setpoints.

    Given the grid data structure, this function verifies if any bus has voltage setpoints from different
    controlling elements that are inconsistent with each other.
    It raises a UserWarning if such discrepancies are found.

    Parameters:
    -----------
    ppc : dict
        The grid data structure, that contains grid data arrays

    Raises:
    -------
    UserWarning:
        If there are buses with voltage controlling elements that have different voltage setpoints.

    Notes:
    ------
    The function specifically checks for voltage setpoints discrepancies between:
    1. Generators
    2. Controllable SSCs
    3. VSCs with voltage control mode on the AC side and controllable state
    """
    # generator buses:
    gen_bus = ppc['gen'][:, GEN_BUS].astype(np.int64)
    # generator setpoints:
    gen_vm = ppc['gen'][:, VG]
    # ssc buses:
    ssc_relevant = np.flatnonzero(ppc['ssc'][:, SSC_CONTROLLABLE] == 1)
    ssc_bus = ppc['ssc'][ssc_relevant, SSC_BUS].astype(np.int64)
    # ssc setpoints:
    ssc_vm = ppc['ssc'][ssc_relevant, SSC_SET_VM_PU]
    # vsc buses:
    vsc_relevant = np.flatnonzero(((ppc['vsc'][:, VSC_MODE_AC] == VSC_MODE_AC_V) |
                                   (ppc['vsc'][:, VSC_MODE_AC] == VSC_MODE_AC_SL)) &
                                  (ppc['vsc'][:, VSC_CONTROLLABLE] == 1))
    vsc_bus = ppc['vsc'][vsc_relevant, VSC_BUS].astype(np.int64)
    # vsc setpoints:
    vsc_vm = ppc['vsc'][vsc_relevant, VSC_VALUE_AC]
    if _different_values_at_one_bus(np.concatenate([gen_bus, ssc_bus, vsc_bus]),
                                    np.concatenate([gen_vm, ssc_vm, vsc_vm])):
        raise UserWarning("Voltage controlling elements, i.e. generators, external grids, or DC lines, "
                          "at the same bus have different setpoints.")


def _check_voltage_angles_at_same_bus(net, ppc):
    if net._is_elements["ext_grid"].any():
        gen_va = net.ext_grid.va_degree.values[net._is_elements["ext_grid"]]
        eg_gens = net._pd2ppc_lookups["ext_grid"][net.ext_grid.index[net._is_elements["ext_grid"]]]
        gen_bus = ppc["gen"][eg_gens, GEN_BUS].astype(np.int64)
        if _different_values_at_one_bus(gen_bus, gen_va):
            raise UserWarning("Ext grids with different voltage angle setpoints connected to the same bus")


def _check_for_reference_bus(ppc):
    # todo implement VSC also as slack
    ref, _, _ = bustypes(ppc["bus"], ppc["gen"])
    # throw an error since no reference bus is defined
    if len(ref) == 0:
        raise UserWarning("No reference bus is available. Either add an ext_grid or a gen with slack=True")

    # todo test this
    bus_dc_type = ppc["bus_dc"][:, DC_BUS_TYPE]
    bus_dc_relevant = np.flatnonzero(bus_dc_type != DC_NONE)
    ref_dc, b2b_dc, _ = bustypes_dc(ppc["bus_dc"])
    # throw an error since no reference bus is defined
    if len(bus_dc_relevant) > 0 and len(ref_dc) == 0 and len(b2b_dc) == 0:
        raise UserWarning("No reference bus for the dc grid is available. Add a DC reference bus by setting the "
                          "DC control mode of at least one VSC converter to 'vm_pu'")


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


def _gen_xward_mask(net, ppc):
    gen_mask = ~np.isin(ppc['gen'][:, GEN_BUS], net["_pd2ppc_lookups"].get("aux", dict()).get("xward", []))
    xward_mask = np.isin(ppc['gen'][:, GEN_BUS], net["_pd2ppc_lookups"].get("aux", dict()).get("xward", []))
    return gen_mask, xward_mask


def _get_xward_pq_buses(net, ppc):
    # find the PQ and PV buses of the xwards; in build_branch.py the F_BUS is set to the PQ bus and T_BUS is set to the auxiliary PV bus
    ft = net["_pd2ppc_lookups"].get('branch', dict()).get("xward", [])
    if len(ft) > 0:
        f, t = ft
        xward_pq_buses = ppc['branch'][f:t, F_BUS].real.astype(np.int64)
        xward_pv_buses = ppc['branch'][f:t, T_BUS].real.astype(np.int64)
        # ignore the xward buses of xward elements that are out of service
        xward_pq_buses = np.setdiff1d(xward_pq_buses, xward_pq_buses[ppc['bus'][xward_pv_buses, BUS_TYPE] == NONE])
        return xward_pq_buses
    else:
        return np.array([], dtype=np.int64)


def _normalise_slack_weights(ppc, gen_mask, xward_mask, xward_pq_buses):
    """Unitise the slack contribution factors in each island to sum to 1."""
    subnets = _subnetworks(ppc)
    gen_buses = ppc['gen'][gen_mask, GEN_BUS].astype(np.int64)

    # it is possible that xward and gen are at the same bus (but not reasonable)
    if len(np.intersect1d(gen_buses, xward_pq_buses)):
        raise NotImplementedError("Found some of the xward PQ buses with slack weight > 0 that coincide with PV or SL buses."
                                  "This configuration is not supported.")

    gen_buses = np.r_[gen_buses, xward_pq_buses]
    slack_weights_gen = np.r_[ppc['gen'][gen_mask, SL_FAC], ppc['gen'][xward_mask, SL_FAC]].astype(np.float64)

    # only 1 ext_grid (reference bus) supported and all others are considered as PV buses,
    # 1 ext_grid is used as slack and others are converted to PV nodes internally;
    # calculate dist_slack for all SL and PV nodes that have non-zero slack weight:
    buses_with_slack_weights = ppc['gen'][ppc['gen'][:, SL_FAC] != 0, GEN_BUS].astype(np.int64)
    if np.sum(ppc['bus'][buses_with_slack_weights, BUS_TYPE] == REF) > 1:
        logger.info("Distributed slack calculation is implemented only for one reference type bus, "
                    "other reference buses will be converted to PV buses internally.")

    for subnet in subnets:
        subnet_gen_mask = np.isin(gen_buses, subnet)
        sum_slack_weights = np.sum(slack_weights_gen[subnet_gen_mask])
        if np.isclose(sum_slack_weights, 0):
            # ppc['gen'][subnet_gen_mask, SL_FAC] = 0
            raise ValueError("Distributed slack contribution factors in "
                             "island '%s' sum to zero." % str(subnet))
        elif sum_slack_weights < 0:
            raise ValueError("Distributed slack contribution factors in island '%s'" % str(subnet) +
                             " sum to negative value. Please correct the data.")
        else:
            # ppc['gen'][subnet_gen_mask, SL_FAC] /= sum_slack_weights
            slack_weights_gen /= sum_slack_weights
            buses, slack_weights_bus, _ = _sum_by_group(gen_buses[subnet_gen_mask], slack_weights_gen[subnet_gen_mask],
                                                        slack_weights_gen[subnet_gen_mask])
            ppc['bus'][buses, SL_FAC_BUS] = slack_weights_bus

    # raise NotImplementedError if there are several separate zones for distributed slack:
    if not np.isclose(sum(ppc['bus'][:, SL_FAC_BUS]), 1):
        raise NotImplementedError("Distributed slack calculation is not implemented for several separate zones at once, "
                                  "please calculate the zones separately.")
