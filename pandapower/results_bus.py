# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
from numpy import complex128
from pandapower.auxiliary import _sum_by_group, sequence_to_phase, _sum_by_group_nvals
from pandapower.pypower.idx_bus import VM, VA, PD, QD, LAM_P, LAM_Q, BASE_KV, NONE, BS

from pandapower.pypower.idx_gen import PG, QG
from pandapower.build_bus import _get_motor_pq, _get_symmetric_pq_of_unsymetric_element
from pandapower.pypower.idx_svc import SVC_THYRISTOR_FIRING_ANGLE, SVC_Q, SVC_X_PU

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def _set_buses_out_of_service(ppc):

    disco = np.where(ppc["bus"][:, 1] == NONE)[0]
    ppc["bus"][disco, VM] = np.nan
    ppc["bus"][disco, VA] = np.nan
    ppc["bus"][disco, PD] = 0
    ppc["bus"][disco, QD] = 0


def _get_bus_v_results(net, ppc, suffix=None):
    bus_idx = _get_bus_idx(net)

    res_table = "res_bus" if suffix is None else "res_bus%s" % suffix
    # always read the vm_pu from ppc, even for DC power flow - otherwise old values can persist
    net[res_table]["vm_pu"] = ppc["bus"][bus_idx][:, VM]
    # voltage angles
    net[res_table]["va_degree"] = ppc["bus"][bus_idx][:, VA]


def _get_bus_v_results_3ph(net, ppc0, ppc1, ppc2):
    ac = net["_options"]["ac"]
    V012_pu = _V012_from_ppc012(net, ppc0, ppc1, ppc2)
    # Uncomment for results in kV instead of pu
    # bus_base_kv = ppc0["bus"][:,BASE_KV]/np.sqrt(3)
    # V012_pu = V012_pu*bus_base_kv

    Vabc_pu = sequence_to_phase(V012_pu)

    if ac:
        net["res_bus_3ph"]["vm_a_pu"] = np.abs(Vabc_pu[0, :].flatten())
        net["res_bus_3ph"]["vm_b_pu"] = np.abs(Vabc_pu[1, :].flatten())
        net["res_bus_3ph"]["vm_c_pu"] = np.abs(Vabc_pu[2, :].flatten())
    # voltage angles
    net["res_bus_3ph"]["va_a_degree"] = np.angle(Vabc_pu[0, :].flatten())*180/np.pi
    net["res_bus_3ph"]["va_b_degree"] = np.angle(Vabc_pu[1, :].flatten())*180/np.pi
    net["res_bus_3ph"]["va_c_degree"] = np.angle(Vabc_pu[2, :].flatten())*180/np.pi
    net["res_bus_3ph"]["unbalance_percent"] = np.abs(V012_pu[2, :]/V012_pu[1, :])*100
    net["res_bus_3ph"].index = net["bus"].index


def _V012_from_ppc012(net, ppc0, ppc1, ppc2):
    bus_idx = _get_bus_idx(net)
    V012_pu = np.zeros((3, len(bus_idx)), dtype=complex128)
    V012_pu[0, :] = ppc0["bus"][bus_idx][:, VM] * np.exp(1j * np.deg2rad(ppc0["bus"][bus_idx][:, VA]))
    V012_pu[1, :] = ppc1["bus"][bus_idx][:, VM] * np.exp(1j * np.deg2rad(ppc1["bus"][bus_idx][:, VA]))
    V012_pu[2, :] = ppc2["bus"][bus_idx][:, VM] * np.exp(1j * np.deg2rad(ppc2["bus"][bus_idx][:, VA]))
    return V012_pu


def _get_bus_idx(net):
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    ppi = net["bus"].index.values
    bus_idx = bus_lookup[ppi]
    return bus_idx


def _get_opf_marginal_prices(net, ppc):
    bus_idx = _get_bus_idx(net)
    net["res_bus"]["lam_p"] = ppc["bus"][bus_idx][:, LAM_P]
    net["res_bus"]["lam_q"] = ppc["bus"][bus_idx][:, LAM_Q]


def _get_bus_results(net, ppc, bus_pq):
    ac = net["_options"]["ac"]
    mode = net["_options"]["mode"]

    # write sum of p and q values to bus
    net["res_bus"]["p_mw"].values[:] = bus_pq[:, 0]
    if ac:
        net["res_bus"]["q_mvar"].values[:] = bus_pq[:, 1]

    # opf variables
    if mode == "opf":
        _get_opf_marginal_prices(net, ppc)

    # update index in res bus bus
    net["res_bus"].index = net["bus"].index


def _get_bus_results_3ph(net, bus_pq):
    ac = net["_options"]["ac"]

    # write sum of p and q values to bus
    net["res_bus_3ph"]["p_a_mw"] = bus_pq[:, 0]
    net["res_bus_3ph"]["p_b_mw"] = bus_pq[:, 2]
    net["res_bus_3ph"]["p_c_mw"] = bus_pq[:, 4]
    if ac:
        net["res_bus_3ph"]["q_a_mvar"] = bus_pq[:, 1]
        net["res_bus_3ph"]["q_b_mvar"] = bus_pq[:, 3]
        net["res_bus_3ph"]["q_c_mvar"] = bus_pq[:, 5]

    # Todo: OPF

    # update index in res bus bus
    # net["res_bus"].index = net["bus"].index
    net["res_bus_3ph"].index = net["bus"].index


def write_voltage_dependend_load_results(net, p, q, b):
    l = net["load"]
    _is_elements = net["_is_elements"]

    if len(l) > 0:
        load_is = _is_elements["load"]
        scaling = l["scaling"].values
        bus_lookup = net["_pd2ppc_lookups"]["bus"]
        lidx = bus_lookup[l["bus"].values]

        voltage_depend_loads = net["_options"]["voltage_depend_loads"]

        cz = l["const_z_percent"].values / 100.
        ci = l["const_i_percent"].values / 100.
        cp = 1. - (cz + ci)

        # constant power
        pl = l["p_mw"].values * scaling * load_is * cp
        net["res_load"]["p_mw"] = pl
        p = np.hstack([p, pl])

        ql = l["q_mvar"].values * scaling * load_is * cp
        net["res_load"]["q_mvar"] = ql
        q = np.hstack([q, ql])

        b = np.hstack([b, l["bus"].values])

        if voltage_depend_loads:
            # constant impedance and constant current
            vm_l = net["_ppc"]["bus"][lidx, 7]
            volt_depend = ci * vm_l + cz * vm_l ** 2
            pl = l["p_mw"].values * scaling * load_is * volt_depend
            net["res_load"]["p_mw"] += pl
            p = np.hstack([p, pl])

            ql = l["q_mvar"].values * scaling * load_is * volt_depend
            net["res_load"]["q_mvar"] += ql
            q = np.hstack([q, ql])

            b = np.hstack([b, l["bus"].values])
        return p, q, b


def write_pq_results_to_element(net, ppc, element, suffix=None):
    """
    get p_mw and q_mvar for a specific pq element ("load", "sgen"...).
    This function basically writes values element table to res_element table
    :param net: pandapower net
    :param element: element name (str)
    :return:
    """
    # info from net
    _is_elements = net["_is_elements"]
    ac = net["_options"]["ac"]

    # info element
    el_data = net[element]
    res_ = "res_%s" % element
    if suffix is not None:
        res_ += "_%s" % suffix
    ctrl_ = "%s_controllable" % element

    is_controllable = False
    if ctrl_ in _is_elements:
        controlled_elements = net[element][net._is_elements[ctrl_]].index
        gen_idx = net._pd2ppc_lookups[ctrl_][controlled_elements]
        gen_sign = 1 if element == "sgen" else -1
        is_controllable = True

    if element == "motor":
        p_mw, q_mvar = _get_motor_pq(net)
        net[res_]["p_mw"].values[:] = p_mw
        net[res_]["q_mvar"].values[:] = q_mvar
        return net
    elif element.startswith("asymmetric"):
        p_mw, q_mvar = _get_symmetric_pq_of_unsymetric_element(net, element)
        net[res_]["p_mw"].values[:] = p_mw
        net[res_]["q_mvar"].values[:] = q_mvar
        return net

    # Wards and xwards have different names in their element table, but not in res table. Also no scaling -> Fix...
    p_mw = "ps_mw" if element in ["ward", "xward"] else "p_mw"
    q_mvar = "qs_mvar" if element in ["ward", "xward"] else "q_mvar"
    scaling = el_data["scaling"].values if element not in ["ward", "xward"] else 1.0

    element_in_service = _is_elements[element]

    # P result in mw to element
    net[res_]["p_mw"].values[:] = el_data[p_mw].values * scaling * element_in_service
    if is_controllable:
        net[res_]["p_mw"].loc[controlled_elements] = ppc["gen"][gen_idx, PG] * gen_sign

    # add results from distributed slack calculation for xwards from ppc instead of the element table
    if net._options['distributed_slack'] and element == "xward" and np.any(net.xward.loc[element_in_service, 'slack_weight'].values != 0):
        _extract_dist_slack_pq_results(net, ppc, element, res_)

    if ac:
        # Q result in mvar to element
        net[res_]["q_mvar"].values[:] = el_data[q_mvar].values * scaling * element_in_service
        if is_controllable:
            net[res_]["q_mvar"].loc[controlled_elements] = ppc["gen"][gen_idx, QG] * gen_sign
    return net


def _extract_dist_slack_pq_results(net, ppc, element, res_):
    node_elements = ['sgen', 'load', 'ward', 'xward', 'storage']
    for b in net[element].bus.values:
        connected = dict()
        for e in node_elements:
            conn = net[e].loc[net[e].in_service & (net[e].bus == b)].index.values
            if len(conn) > 0:
                connected.update({e: conn})

        p_bus = ppc['bus'][net._pd2ppc_lookups["bus"][b], PD]
        # first, count the total slack weights per bus, and obtain the variable part of the active power
        total_weight = 0
        for e, idx in connected.items():
            if 'slack_weight' in net[e].columns:
                elm_weight = net[e].loc[idx, 'slack_weight'].values
                if np.abs(elm_weight).sum() != 0:
                    total_weight += np.abs(elm_weight)
            p_bus -= net[e].loc[idx, 'ps_mw' if e in ["ward", "xward"] else "p_mw"].values.sum()

        # now, distribute the variable part of the active power among the dist_slack elements
        for e, idx in connected.items():
            if 'slack_weight' in net[e].columns:
                elm_weight = net[e].loc[idx, 'slack_weight'].values
                if np.abs(elm_weight).sum() != 0:
                    net[res_]["p_mw"] += p_bus * elm_weight / total_weight


def write_pq_results_to_element_3ph(net, element):
    """
    get p_mw and q_mvar for a specific pq element ("load", "sgen"...).
    This function basically writes values element table to res_element table

    :param net: pandapower net
    :param element: element name (str)
    :return:
    """

    # info from net
    _is_elements = net["_is_elements"]
    ac = net["_options"]["ac"]

    # info element
    el_data = net[element]
    res_ = "res_" + element+"_3ph"

    scaling = el_data["scaling"].values

    element_in_service = _is_elements[element]

    net[res_]["p_a_mw"] = list(
        (el_data["p_mw"].values / 3) * scaling * element_in_service) if element in ["load", "sgen"] \
        else list(el_data["p_a_mw"].values * scaling * element_in_service)

    net[res_]["p_b_mw"] = list(
        (el_data["p_mw"].values / 3) * scaling * element_in_service)if element in ["load", "sgen"] \
        else list(el_data["p_b_mw"].values * scaling * element_in_service)

    net[res_]["p_c_mw"] = list(
        (el_data["p_mw"].values / 3) * scaling * element_in_service) if element in ["load", "sgen"] \
        else list(el_data["p_c_mw"].values * scaling * element_in_service)
    if ac:
        # Q result in kvar to element
        net[res_]["q_a_mvar"] = list(
            (el_data["q_mvar"].values / 3) * scaling * element_in_service) if element in ["load", "sgen"] \
            else list(el_data["q_a_mvar"].values * scaling * element_in_service)

        net[res_]["q_b_mvar"] = list(
            (el_data["q_mvar"].values / 3) * scaling * element_in_service) if element in ["load", "sgen"] \
            else list(el_data["q_b_mvar"].values * scaling * element_in_service)

        net[res_]["q_c_mvar"] = list(
            (el_data["q_mvar"].values / 3) * scaling * element_in_service) if element in ["load", "sgen"] \
            else list(el_data["q_c_mvar"].values * scaling * element_in_service)

    # update index of result table
    net[res_].index = net[element].index
    return net


def get_p_q_b(net, element, suffix=None):
    ac = net["_options"]["ac"]
    res_ = "res_" + element
    if suffix != None:
        res_ += "_%s" % suffix

    # bus values are needed for stacking
    b = net[element]["bus"].values
    p = net[res_]["p_mw"]
    q = net[res_]["q_mvar"] if ac else np.zeros_like(p)
    return p, q, b


def get_p_q_b_3ph(net, element):
    ac = net["_options"]["ac"]
    res_ = "res_" + element+"_3ph"

    # bus values are needed for stacking
    b = net[element]["bus"].values
    pA = net[res_]["p_a_mw"]
    pB = net[res_]["p_b_mw"]
    pC = net[res_]["p_c_mw"]
    qA = net[res_]["q_a_mvar"] if ac else np.zeros_like(pA)
    qB = net[res_]["q_b_mvar"] if ac else np.zeros_like(pB)
    qC = net[res_]["q_c_mvar"] if ac else np.zeros_like(pC)
    return pA, qA, pB, qB, pC, qC, b


def _get_p_q_results(net, ppc, bus_lookup_aranged):
    bus_pq = np.zeros(shape=(len(net["bus"].index), 2), dtype=np.float64)
    b, p, q = np.array([]), np.array([]), np.array([])

    ac = net["_options"]["ac"]
    elements = ["load", "motor", "sgen", "storage", "ward", "xward",
                "asymmetric_load", "asymmetric_sgen"]

    if net["_options"]["voltage_depend_loads"] and ac:
        # voltage dependend loads need special treatment here

        p, q, b = write_voltage_dependend_load_results(net, p, q, b)
        elements.remove("load")

    for element in elements:
        if len(net[element]):
            write_pq_results_to_element(net, ppc, element)
            p_el, q_el, bus_el = get_p_q_b(net, element)
            if element.endswith("sgen"):
                p = np.hstack([p, -p_el])
                q = np.hstack([q, -q_el])
            else:
                p = np.hstack([p, p_el])
                q = np.hstack([q, q_el])
            b = np.hstack([b, bus_el])

    if not ac:
        q = np.zeros(len(p))

    # sum pq results from every element to be written to net['bus'] later on
    b_pp, vp, vq = _sum_by_group(b.astype(np.int64), p, q)
    b_ppc = bus_lookup_aranged[b_pp]
    bus_pq[b_ppc, 0] = vp
    bus_pq[b_ppc, 1] = vq
    return bus_pq


def _get_p_q_results_3ph(net, bus_lookup_aranged):
    # results to be filled (bus, p in kw, q in kvar)
    bus_pq = np.zeros(shape=(len(net["bus"].index), 6), dtype=float)
    b, pA, pB, pC, qA, qB, qC = np.array([]), np.array([]), np.array([]), np.array([]), \
        np.array([]), np.array([]), np.array([])

    ac = net["_options"]["ac"]
    # Todo: Voltage dependent loads
    elements = ["storage", "sgen", "load"]
    elements_3ph = ["asymmetric_load", "asymmetric_sgen"]
    for element in elements:
        sign = -1 if element in ['sgen', 'asymmetric_sgen'] else 1
        if len(net[element]):
            write_pq_results_to_element(net, net._ppc1, element, suffix="3ph")
            p_el, q_el, bus_el = get_p_q_b(net, element, suffix="3ph")
            pA = np.hstack([pA, sign * p_el/3])
            pB = np.hstack([pB, sign * p_el/3])
            pC = np.hstack([pC, sign * p_el/3])
            qA = np.hstack([qA, sign * q_el/3 if ac else np.zeros(len(p_el/3))])
            qB = np.hstack([qB, sign * q_el/3 if ac else np.zeros(len(p_el/3))])
            qC = np.hstack([qC, sign * q_el/3 if ac else np.zeros(len(p_el/3))])
            b = np.hstack([b, bus_el])
    for element in elements_3ph:
        sign = -1 if element in ['sgen', 'asymmetric_sgen'] else 1
        if len(net[element]):
            write_pq_results_to_element_3ph(net, element)
            p_el_A, q_el_A, p_el_B, q_el_B, p_el_C, q_el_C, bus_el = get_p_q_b_3ph(net, element)
            pA = np.hstack([pA, sign * p_el_A])
            pB = np.hstack([pB, sign * p_el_B])
            pC = np.hstack([pC, sign * p_el_C])
            qA = np.hstack([qA, sign * q_el_A if ac else np.zeros(len(p_el_A))])
            qB = np.hstack([qB, sign * q_el_B if ac else np.zeros(len(p_el_B))])
            qC = np.hstack([qC, sign * q_el_C if ac else np.zeros(len(p_el_C))])
            b = np.hstack([b, bus_el])

    # sum pq results from every element to be written to net['bus'] later on
    b_pp, vp_A, vq_A, vp_B, vq_B, vp_C, vq_C = _sum_by_group_nvals(b.astype(np.int64), pA, qA, pB, qB, pC, qC)
    b_ppc = bus_lookup_aranged[b_pp]
    bus_pq[b_ppc, 0] = vp_A
    bus_pq[b_ppc, 1] = vq_A
    bus_pq[b_ppc, 2] = vp_B
    bus_pq[b_ppc, 3] = vq_B
    bus_pq[b_ppc, 4] = vp_C
    bus_pq[b_ppc, 5] = vq_C
    return bus_pq


def _get_shunt_results(net, ppc, bus_lookup_aranged, bus_pq):
    ac = net["_options"]["ac"]

    b, p, q = np.array([]), np.array([]), np.array([])
    _is_elements = net["_is_elements"]
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    baseMVA = ppc["baseMVA"]
    baseZ = np.square(ppc["bus"][:, BASE_KV]) / baseMVA

    s = net["shunt"]
    if len(s) > 0:
        sidx = bus_lookup[s["bus"].values]
        shunt_is = _is_elements["shunt"]
        u_shunt = ppc["bus"][sidx, VM]
        step = s["step"]
        v_ratio = (ppc["bus"][sidx, BASE_KV] / net["shunt"]["vn_kv"].values) ** 2
        u_shunt = np.nan_to_num(u_shunt)
        p_shunt = u_shunt ** 2 * net["shunt"]["p_mw"].values * shunt_is * v_ratio * step
        net["res_shunt"]["p_mw"].values[:] = p_shunt
        p = np.hstack([p, p_shunt])
        if ac:
            net["res_shunt"]["vm_pu"].values[:] = u_shunt
            q_shunt = u_shunt ** 2 * net["shunt"]["q_mvar"].values * shunt_is * v_ratio * step
            net["res_shunt"]["q_mvar"].values[:] = q_shunt
            q = np.hstack([q, q_shunt])
        b = np.hstack([b, s["bus"].values])

    w = net["ward"]
    if len(w) > 0:
        widx = bus_lookup[w["bus"].values]
        ward_is = _is_elements["ward"]
        u_ward = ppc["bus"][widx, VM]
        u_ward = np.nan_to_num(u_ward)
        p_ward = u_ward ** 2 * net["ward"]["pz_mw"].values * ward_is
        net["res_ward"]["p_mw"].values[:] = net["res_ward"]["p_mw"].values + p_ward
        p = np.hstack([p, p_ward])
        if ac:
            net["res_ward"]["vm_pu"].values[:] = u_ward
            q_ward = u_ward ** 2 * net["ward"]["qz_mvar"].values * ward_is
            net["res_ward"]["q_mvar"].values[:] = net["res_ward"]["q_mvar"].values + q_ward
            q = np.hstack([q, q_ward])
        b = np.hstack([b, w["bus"].values])

    xw = net["xward"]
    if len(xw) > 0:
        widx = bus_lookup[xw["bus"].values]
        xward_is = _is_elements["xward"]
        u_xward = ppc["bus"][widx, VM]
        u_xward = np.nan_to_num(u_xward)
        p_xward = u_xward ** 2 * net["xward"]["pz_mw"].values * xward_is
        net["res_xward"]["p_mw"].values[:] = net["res_xward"]["p_mw"].values + p_xward
        p = np.hstack([p, p_xward])
        if ac:
            net["res_xward"]["vm_pu"].values[:] = u_xward
            q_xward = u_xward ** 2 * net["xward"]["qz_mvar"].values * xward_is
            net["res_xward"]["q_mvar"].values[:] = net["res_xward"]["q_mvar"].values + q_xward
            q = np.hstack([q, q_xward])
        b = np.hstack([b, xw["bus"].values])

    # svc = net["svc"]  # todo: uncomment this after PandaModels net also has this key
    svc = net.get("svc", np.array([]))
    if len(svc):
        svcidx = bus_lookup[svc["bus"].values]
        svc_is = _is_elements["svc"]
        net["res_svc"].loc[svc_is, "thyristor_firing_angle_degree"] = np.rad2deg(ppc["svc"][svc_is, SVC_THYRISTOR_FIRING_ANGLE])
        p = np.hstack([p, np.zeros_like(svc["bus"].values)])
        if ac:
            net["res_svc"].loc[svc_is, "vm_pu"] = ppc["bus"][svcidx[svc_is], VM]
            net["res_svc"].loc[svc_is, "va_degree"] = ppc["bus"][svcidx[svc_is], VA]
            q_svc = ppc["svc"][:, SVC_Q]
            net["res_svc"].loc[:, "q_mvar"] = q_svc  # write all because of zeros
            net["res_svc"].loc[svc_is, "x_ohm"] = ppc["svc"][svc_is, SVC_X_PU] * baseZ[svcidx[svc_is]]
            q = np.hstack([q, q_svc])
        b = np.hstack([b, svc["bus"].values])

    if not ac:
        q = np.zeros(len(p))
    b_pp, vp, vq = _sum_by_group(b.astype(np.int64), p, q)
    b_ppc = bus_lookup_aranged[b_pp]

    bus_pq[b_ppc, 0] += vp
    if ac:
        bus_pq[b_ppc, 1] += vq
