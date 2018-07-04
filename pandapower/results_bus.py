# -*- coding: utf-8 -*-

# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pandas as pd
import numpy as np
from numpy import zeros, array, float, hstack, invert, angle, complex128

from pandapower.auxiliary import _sum_by_group, sequence_to_phase, _sum_by_group_nvals
from pandapower.idx_bus import VM, VA, PD, QD, LAM_P, LAM_Q, BASE_KV, NONE
from pandapower.idx_gen import PG, QG


def _get_p_q_results_opf(net, ppc, bus_lookup_aranged):
    bus_pq = zeros(shape=(len(net["bus"].index), 2), dtype=float)
    b, p, q = array([]), array([]), array([])

    _is_elements = net["_is_elements"]

    l = net["load"]
    if len(l) > 0:
        load_is = _is_elements["load"]
        load_ctrl = (l.in_service & l.controllable).values
        scaling = l["scaling"].values
        pl = l["p_kw"].values * scaling * load_is * invert(load_ctrl)
        ql = l["q_kvar"].values * scaling * load_is * invert(load_ctrl)
        if any(load_ctrl):
            # get load index in ppc
            lidx_ppc = net._pd2ppc_lookups["load_controllable"][_is_elements["load_controllable"].index]
            pl[load_is & load_ctrl] = - ppc["gen"][lidx_ppc, PG] * 1000
            ql[load_is & load_ctrl] = - ppc["gen"][lidx_ppc, QG] * 1000

        net["res_load"]["p_kw"] = pl
        net["res_load"]["q_kvar"] = ql
        p = hstack([p, pl])
        q = hstack([q, ql])
        b = hstack([b, l["bus"].values])
        net["res_load"].index = net["load"].index

    sg = net["sgen"]
    if len(sg) > 0:
        sgen_is = _is_elements["sgen"]
        sgen_ctrl = (sg.in_service & sg.controllable).values
        scaling = sg["scaling"].values
        psg = sg["p_kw"].values * scaling * sgen_is * invert(sgen_ctrl)
        qsg = sg["q_kvar"].values * scaling * sgen_is * invert(sgen_ctrl)
        if any(sgen_ctrl):
            # get gen index in ppc
            gidx_ppc = net._pd2ppc_lookups["sgen_controllable"][_is_elements["sgen_controllable"].index]
            psg[sgen_is & sgen_ctrl] = - ppc["gen"][gidx_ppc, PG] * 1000
            qsg[sgen_is & sgen_ctrl] = - ppc["gen"][gidx_ppc, QG] * 1000

        net["res_sgen"]["p_kw"] = psg
        net["res_sgen"]["q_kvar"] = qsg
        q = hstack([q, qsg])
        p = hstack([p, psg])
        b = hstack([b, sg["bus"].values])
        net["res_sgen"].index = net["sgen"].index

    stor = net["storage"]
    if len(stor) > 0:
        stor_is = _is_elements["storage"]
        stor_ctrl = (stor.in_service & stor.controllable).values
        scaling = stor["scaling"].values
        pstor = stor["p_kw"].values * scaling * stor_is * invert(stor_ctrl)
        qstor = stor["q_kvar"].values * scaling * stor_is * invert(stor_ctrl)
        if any(stor_ctrl):
            # get storage index in ppc
            stidx_ppc = net._pd2ppc_lookups["storage_controllable"][_is_elements["storage_controllable"].index]
            pstor[stor_is & stor_ctrl] = - ppc["gen"][stidx_ppc, PG] * 1000
            qstor[stor_is & stor_ctrl] = - ppc["gen"][stidx_ppc, QG] * 1000

        net["res_storage"]["p_kw"] = pstor
        net["res_storage"]["q_kvar"] = qstor
        q = hstack([q, qstor])
        p = hstack([p, pstor])
        b = hstack([b, stor["bus"].values])
        net["res_storage"].index = net["storage"].index

    b_pp, vp, vq = _sum_by_group(b.astype(int), p, q)
    b_ppc = bus_lookup_aranged[b_pp]
    bus_pq[b_ppc, 0] = vp
    bus_pq[b_ppc, 1] = vq
    return bus_pq


def _set_buses_out_of_service(ppc):
    disco = np.where(ppc["bus"][:, 1] == NONE)[0]
    ppc["bus"][disco, VM] = np.nan
    ppc["bus"][disco, VA] = np.nan
    ppc["bus"][disco, PD] = 0
    ppc["bus"][disco, QD] = 0


def _get_bus_v_results(net, ppc):
    ac = net["_options"]["ac"]
    bus_idx = _get_bus_idx(net)
    if ac:
        net["res_bus"]["vm_pu"] = ppc["bus"][bus_idx][:, VM]
    # voltage angles
    net["res_bus"]["va_degree"] = ppc["bus"][bus_idx][:, VA]


def _get_bus_v_results_3ph(net, ppc0, ppc1, ppc2):
    ac = net["_options"]["ac"]
    bus_idx = _get_bus_idx(net)

    V012_pu = _V012_from_ppc012(net, ppc0, ppc1, ppc2)
    # Uncomment for results in kV instead of pu
    # bus_base_kv = ppc0["bus"][:,BASE_KV]/np.sqrt(3)
    # V012_pu = V012_pu*bus_base_kv

    Vabc_pu = sequence_to_phase(V012_pu)

    if ac:
        net["res_bus_3ph"]["vmA_pu"] = abs(Vabc_pu[0, :].getA1())
        net["res_bus_3ph"]["vmB_pu"] = abs(Vabc_pu[1, :].getA1())
        net["res_bus_3ph"]["vmC_pu"] = abs(Vabc_pu[2, :].getA1())
    # voltage angles
    net["res_bus_3ph"]["vaA_degree"] = angle(Vabc_pu[0, :].getA1())*180/np.pi
    net["res_bus_3ph"]["vaB_degree"] = angle(Vabc_pu[1, :].getA1())*180/np.pi
    net["res_bus_3ph"]["vaC_degree"] = angle(Vabc_pu[2, :].getA1())*180/np.pi
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
    net["res_bus"]["p_kw"] = bus_pq[:, 0]
    if ac:
        net["res_bus"]["q_kvar"] = bus_pq[:, 1]

    # opf variables
    if mode == "opf":
        _get_opf_marginal_prices(net, ppc)

    # update index in res bus bus
    net["res_bus"].index = net["bus"].index


def _get_bus_results_3ph(net, bus_pq):
    ac = net["_options"]["ac"]
    mode = net["_options"]["mode"]

    # write sum of p and q values to bus
    net["res_bus_3ph"]["pA_kw"] = bus_pq[:, 0]
    net["res_bus_3ph"]["pB_kw"] = bus_pq[:, 2]
    net["res_bus_3ph"]["pC_kw"] = bus_pq[:, 4]
    # net["res_bus"]["p_kw"] = np.sum(bus_pq[:,0])
    if ac:
        net["res_bus_3ph"]["qA_kvar"] = bus_pq[:, 1]
        net["res_bus_3ph"]["qB_kvar"] = bus_pq[:, 3]
        net["res_bus_3ph"]["qC_kvar"] = bus_pq[:, 5]

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
        pl = l["p_kw"].values * scaling * load_is * cp
        net["res_load"]["p_kw"] = pl
        p = np.hstack([p, pl])

        ql = l["q_kvar"].values * scaling * load_is * cp
        net["res_load"]["q_kvar"] = ql
        q = np.hstack([q, ql])

        b = np.hstack([b, l["bus"].values])

        if voltage_depend_loads:
            # constant impedance and constant current
            vm_l = net["_ppc"]["bus"][lidx,7]
            volt_depend = ci * vm_l + cz * vm_l ** 2
            pl = l["p_kw"].values * scaling * load_is * volt_depend
            net["res_load"]["p_kw"] += pl
            p = np.hstack([p, pl])

            ql = l["q_kvar"].values * scaling * load_is * volt_depend
            net["res_load"]["q_kvar"] += ql
            q = np.hstack([q, ql])

            b = np.hstack([b, l["bus"].values])

        net["res_load"].index = net["load"].index
        return p, q, b


def write_pq_results_to_element(net, element):
    """
    get p_kw and q_kvar for a specific pq element ("load", "sgen"...).
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
    res_ = "res_" + element

    # Wards and xwards have different names in their element table, but not in res table. Also no scaling -> Fix...
    p_kw = "ps_kw" if element in ["ward", "xward"] else "p_kw"
    q_kvar = "qs_kvar" if element in ["ward", "xward"] else "q_kvar"
    scaling = el_data["scaling"].values if element not in ["ward", "xward"] else 1.0

    element_in_service = _is_elements[element]

    # P result in kw to element
    net[res_]["p_kw"] = el_data[p_kw].values * scaling * element_in_service
    if ac:
        # Q result in kvar to element
        net[res_]["q_kvar"] = el_data[q_kvar].values * scaling * element_in_service

    # update index of result table
    net[res_].index = net[element].index
    return net


def write_pq_results_to_element_3ph(net, element):
    """
    get p_kw and q_kvar for a specific pq element ("load", "sgen"...).
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
    res_ = "res_" + element

    scaling = el_data["scaling"].values

    element_in_service = _is_elements[element]

    # P result in kw to element
    net[res_]["p_kw_A"] = pd.Series(el_data["p_kw_A"].values * scaling * element_in_service)
    net[res_]["p_kw_B"] = pd.Series(el_data["p_kw_B"].values * scaling * element_in_service)
    net[res_]["p_kw_C"] = pd.Series(el_data["p_kw_C"].values * scaling * element_in_service)
    if ac:
        # Q result in kvar to element
        net[res_]["q_kvar_A"] = pd.Series(el_data["q_kvar_A"].values * scaling * element_in_service)
        net[res_]["q_kvar_B"] = pd.Series(el_data["q_kvar_B"].values * scaling * element_in_service)
        net[res_]["q_kvar_C"] = pd.Series(el_data["q_kvar_C"].values * scaling * element_in_service)

    # update index of result table
    net[res_].index = net[element].index
    return net


def get_p_q_b(net, element):
    ac = net["_options"]["ac"]
    res_ = "res_" + element

    # bus values are needed for stacking
    b = net[element]["bus"].values
    p = net[res_]["p_kw"]
    q = net[res_]["q_kvar"] if ac else np.zeros_like(p)
    return p, q, b


def get_p_q_b_3ph(net, element):
    ac = net["_options"]["ac"]
    res_ = "res_" + element

    # bus values are needed for stacking
    b = net[element]["bus"].values
    pA = net[res_]["p_kw_A"]
    pB = net[res_]["p_kw_B"]
    pC = net[res_]["p_kw_C"]
    qA = net[res_]["q_kvar_A"] if ac else np.zeros_like(pA)
    qB = net[res_]["q_kvar_B"] if ac else np.zeros_like(pB)
    qC = net[res_]["q_kvar_C"] if ac else np.zeros_like(pC)
    return pA, pB, pC, qA, qB, qC, b


def _get_p_q_results(net, bus_lookup_aranged):
    # results to be filled (bus, p in kw, q in kvar)
    bus_pq = np.zeros(shape=(len(net["bus"].index), 2), dtype=np.float)
    b, p, q = np.array([]), np.array([]), np.array([])

    ac = net["_options"]["ac"]
    if net["_options"]["voltage_depend_loads"] and ac:
        # voltage dependend loads need special treatment here
        p, q, b = write_voltage_dependend_load_results(net, p, q, b)
        elements = ["sgen", "storage", "ward", "xward"]
    else:
        elements = ["load", "sgen", "storage", "ward", "xward"]

    for element in elements:
        if len(net[element]):
            write_pq_results_to_element(net, element)
            p_el, q_el, bus_el = get_p_q_b(net, element)
            p = np.hstack([p, p_el])
            q = np.hstack([q, q_el])
            b = np.hstack([b, bus_el])

    if not ac:
        q = np.zeros(len(p))

    # sum pq results from every element to be written to net['bus'] later on
    b_pp, vp, vq = _sum_by_group(b.astype(int), p, q)
    b_ppc = bus_lookup_aranged[b_pp]
    bus_pq[b_ppc, 0] = vp
    bus_pq[b_ppc, 1] = vq
    return bus_pq


def _get_p_q_results_3ph(net, bus_lookup_aranged):
    # results to be filled (bus, p in kw, q in kvar)
    bus_pq = np.zeros(shape=(len(net["bus"].index), 6), dtype=np.float)
    b, pA, pB, pC, qA, qB, qC = np.array([]), np.array([]), np.array([]), np.array([]), \
                                np.array([]), np.array([]), np.array([])

    ac = net["_options"]["ac"]
    # Todo: Voltage dependent loads
    elements = ["load", "sgen", "storage", "ward", "xward"]
    elements_3ph = ["load_3ph", "sgen_3ph"]

    for element in elements:
        if len(net[element]):
            write_pq_results_to_element(net, element)
            p_el, q_el, bus_el = get_p_q_b(net, element)
            pA = np.hstack([pA, p_el/3])
            pB = np.hstack([pB, p_el/3])
            pC = np.hstack([pC, p_el/3])
            qA = np.hstack([qA, q_el/3 if ac else np.zeros(len(p_el))])
            qB = np.hstack([qB, q_el/3 if ac else np.zeros(len(p_el))])
            qC = np.hstack([qC, q_el/3 if ac else np.zeros(len(p_el))])
            b = np.hstack([b, bus_el])
    for element in elements_3ph:
        if len(net[element]):
            write_pq_results_to_element_3ph(net, element)
            p_el_A, p_el_B, p_el_C, q_el_A, q_el_B, q_el_C, bus_el = get_p_q_b_3ph(net, element)
            pA = np.hstack([pA, p_el_A])
            qA = np.hstack([qA, q_el_A if ac else np.zeros(len(p_el_A))])
            pB = np.hstack([pB, p_el_B])
            qB = np.hstack([qB, q_el_B if ac else np.zeros(len(p_el_B))])
            pC = np.hstack([pC, p_el_C])
            qC = np.hstack([qC, q_el_C if ac else np.zeros(len(p_el_C))])
            b = np.hstack([b, bus_el])

    # sum pq results from every element to be written to net['bus'] later on
    b_pp, vp_A, vq_A, vp_B, vq_B, vp_C, vq_C = _sum_by_group_nvals(b.astype(int), pA, qA, pB, qB, pC, qC)
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

    s = net["shunt"]
    if len(s) > 0:
        sidx = bus_lookup[s["bus"].values]
        shunt_is = _is_elements["shunt"]
        u_shunt = ppc["bus"][sidx, VM]
        step = s["step"]
        v_ratio = (ppc["bus"][sidx, BASE_KV] / net["shunt"]["vn_kv"].values) ** 2
        u_shunt = np.nan_to_num(u_shunt)
        p_shunt = u_shunt ** 2 * net["shunt"]["p_kw"].values * shunt_is * v_ratio * step
        net["res_shunt"]["p_kw"] = p_shunt
        p = np.hstack([p, p_shunt])
        if ac:
            net["res_shunt"]["vm_pu"] = u_shunt
            q_shunt = u_shunt ** 2 * net["shunt"]["q_kvar"].values * shunt_is * v_ratio * step
            net["res_shunt"]["q_kvar"] = q_shunt
            q = np.hstack([q, q_shunt])
        b = np.hstack([b, s["bus"].values])
        net["res_shunt"].index = net["shunt"].index

    w = net["ward"]
    if len(w) > 0:
        widx = bus_lookup[w["bus"].values]
        ward_is = _is_elements["ward"]
        u_ward = ppc["bus"][widx, VM]
        u_ward = np.nan_to_num(u_ward)
        p_ward = u_ward ** 2 * net["ward"]["pz_kw"].values * ward_is
        net["res_ward"]["p_kw"] += p_ward
        p = np.hstack([p, p_ward])
        if ac:
            net["res_ward"]["vm_pu"] = u_ward
            q_ward = u_ward ** 2 * net["ward"]["qz_kvar"].values * ward_is
            net["res_ward"]["q_kvar"] += q_ward
            q = np.hstack([q, q_ward])
        b = np.hstack([b, w["bus"].values])
        net["res_ward"].index = net["ward"].index

    xw = net["xward"]
    if len(xw) > 0:
        widx = bus_lookup[xw["bus"].values]
        xward_is = _is_elements["xward"]
        u_xward = ppc["bus"][widx, VM]
        u_xward = np.nan_to_num(u_xward)
        p_xward = u_xward ** 2 * net["xward"]["pz_kw"].values * xward_is
        net["res_xward"]["p_kw"] += p_xward
        p = np.hstack([p, p_xward])
        if ac:
            net["res_xward"]["vm_pu"] = u_xward
            q_xward = u_xward ** 2 * net["xward"]["qz_kvar"].values * xward_is
            net["res_xward"]["q_kvar"] += q_xward
            q = np.hstack([q, q_xward])
        b = np.hstack([b, xw["bus"].values])
        net["res_xward"].index = net["xward"].index

    if not ac:
        q = np.zeros(len(p))
    b_pp, vp, vq = _sum_by_group(b.astype(int), p, q)
    b_ppc = bus_lookup_aranged[b_pp]

    bus_pq[b_ppc, 0] += vp
    if ac:
        bus_pq[b_ppc, 1] += vq
