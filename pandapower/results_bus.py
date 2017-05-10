# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import zeros, array, float, hstack, invert
from pandapower.idx_bus import VM, VA, PD, QD, LAM_P, LAM_Q, BASE_KV
from pandapower.idx_gen import PG, QG

from pandapower.auxiliary import _sum_by_group


def _get_p_q_results_opf(net, ppc, bus_lookup_aranged):
    bus_pq = zeros(shape=(len(net["bus"].index), 2), dtype=float)
    b, p, q = array([]), array([]), array([])

    _is_elements = net["_is_elements"]

    l = net["load"]
    if len(l) > 0:
        load_is = _is_elements["load"]
        load_ctrl = l["controllable"].values
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
        sgen_ctrl = sg["controllable"].values
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

    b_pp, vp, vq = _sum_by_group(b.astype(int), p, q)
    b_ppc = bus_lookup_aranged[b_pp]
    bus_pq[b_ppc, 0] = vp
    bus_pq[b_ppc, 1] = vq
    return bus_pq


def _set_buses_out_of_service(ppc):
    disco = np.where(ppc["bus"][:, 1] == 4)[0]
    ppc["bus"][disco, VM] = np.nan
    ppc["bus"][disco, VA] = np.nan
    ppc["bus"][disco, PD] = 0
    ppc["bus"][disco, QD] = 0


def _get_bus_v_results(net, ppc):
    ac = net["_options"]["ac"]

    ppi = net["bus"].index.values
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    bus_idx = bus_lookup[ppi]
    if ac:
        net["res_bus"]["vm_pu"] = ppc["bus"][bus_idx][:, VM]
    # voltage angles
    net["res_bus"]["va_degree"] = ppc["bus"][bus_idx][:, VA]

    net["res_bus"].index = net["bus"].index


def _get_bus_results(net, ppc, bus_pq):
    ac = net["_options"]["ac"]
    mode = net["_options"]["mode"]
    net["res_bus"]["p_kw"] = bus_pq[:, 0]
    if ac:
        net["res_bus"]["q_kvar"] = bus_pq[:, 1]

    ppi = net["bus"].index.values
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    bus_idx = bus_lookup[ppi]
    if ac:
        net["res_bus"]["vm_pu"] = ppc["bus"][bus_idx][:, VM]
    net["res_bus"].index = net["bus"].index
    # voltage angles
    net["res_bus"]["va_degree"] = ppc["bus"][bus_idx][:, VA]

    if mode == "opf":
        # marginal prices
        net["res_bus"]["lam_p"] = ppc["bus"][bus_idx][:, LAM_P]
        net["res_bus"]["lam_q"] = ppc["bus"][bus_idx][:, LAM_Q]


def _get_p_q_results(net,  bus_lookup_aranged):
    ac = net["_options"]["ac"]
    bus_pq = np.zeros(shape=(len(net["bus"].index), 2), dtype=np.float)
    b, p, q = np.array([]), np.array([]), np.array([])

    _is_elements = net["_is_elements"]

    l = net["load"]
    if len(l) > 0:
        load_is = _is_elements["load"]
        scaling = l["scaling"].values

        if ac:
            voltage_depend_loads = net["_options"]["voltage_depend_loads"]

            cz = l["const_z_percent"].values / 100. if voltage_depend_loads else 0.
            ci = l["const_i_percent"].values / 100. if voltage_depend_loads else 0.
            cp = 1 - (cz + ci)

            # constant power
            pl = l["p_kw"].values * scaling * load_is * cp
            net["res_load"]["p_kw"] = pl
            p = np.hstack([p, pl])

            ql = l["q_kvar"].values * scaling * load_is * cp
            net["res_load"]["q_kvar"] = ql
            q = np.hstack([q, ql])

            b = np.hstack([b, l["bus"].values])

            if voltage_depend_loads:
                v = (net.res_bus.vm_pu * np.exp(1j * np.deg2rad(net.res_bus.va_degree))).loc[l['bus'].values].fillna(0)
                # constant impedance
                sl = (l["p_kw"].values + 1j * l["q_kvar"].values) * scaling * load_is * cz * (v * np.conj(v))
                pl = sl.real
                net["res_load"]["p_kw"] += pl
                p = np.hstack([p, pl])

                ql = sl.imag
                net["res_load"]["q_kvar"] += ql
                q = np.hstack([q, ql])

                b = np.hstack([b, l["bus"].values])


                # constant current
                sl = (l["p_kw"].values + 1j * l["q_kvar"].values) * scaling * load_is * ci * v
                pl = sl.real
                net["res_load"]["p_kw"] += pl
                p = np.hstack([p, pl])

                ql = sl.imag
                net["res_load"]["q_kvar"] += ql
                q = np.hstack([q, ql])

                b = np.hstack([b, l["bus"].values])

        else:
            pl = l["p_kw"].values * scaling * load_is
            net["res_load"]["p_kw"] = pl
            p = np.hstack([p, pl])
            b = np.hstack([b, l["bus"].values])


        net["res_load"].index = net["load"].index

    sg = net["sgen"]
    if len(sg) > 0:
        sgen_is = _is_elements["sgen"]
        scaling = sg["scaling"].values
        psg = sg["p_kw"].values * scaling * sgen_is
        net["res_sgen"]["p_kw"] = psg
        p = np.hstack([p, psg])
        if ac:
            qsg = sg["q_kvar"].values * scaling * sgen_is
            net["res_sgen"]["q_kvar"] = qsg
            q = np.hstack([q, qsg])
        b = np.hstack([b, sg["bus"].values])
        net["res_sgen"].index = net["sgen"].index

    w = net["ward"]
    if len(w) > 0:
        ward_is = _is_elements["ward"]
        pw = w["ps_kw"].values * ward_is
        net["res_ward"]["p_kw"] = pw
        p = np.hstack([p, pw])
        if ac:
            qw = w["qs_kvar"].values * ward_is
            q = np.hstack([q, qw])
            net["res_ward"]["q_kvar"] = qw
        b = np.hstack([b, w["bus"].values])

    xw = net["xward"]
    if len(xw) > 0:
        xward_is = _is_elements["xward"]
        pxw = xw["ps_kw"].values * xward_is
        p = np.hstack([p, pxw])
        net["res_xward"]["p_kw"] = pxw
        if ac:
            qxw = xw["qs_kvar"].values * xward_is
            net["res_xward"]["q_kvar"] = qxw
            q = np.hstack([q, qxw])
        b = np.hstack([b, xw["bus"].values])
    if not ac:
        q = np.zeros(len(p))
    b_pp, vp, vq = _sum_by_group(b.astype(int), p, q)
    b_ppc = bus_lookup_aranged[b_pp]
    bus_pq[b_ppc, 0] = vp
    bus_pq[b_ppc, 1] = vq
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
        v_ratio = (ppc["bus"][sidx, BASE_KV] / net["shunt"]["vn_kv"].values)**2
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
