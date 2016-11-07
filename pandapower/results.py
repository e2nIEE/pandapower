# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

import numpy as np

from pypower.idx_brch import F_BUS, T_BUS, PF, QF, PT, QT
from pypower.idx_bus import BASE_KV, VM, VA, PD, QD
from pypower.idx_gen import PG, QG, GEN_BUS

from pandapower.auxiliary import get_indices, _sum_by_group

def _extract_results(net, ppc, is_elems, bus_lookup, trafo_loading, return_voltage_angles,
                     ac=True):
    # get in service elements
    bus_is = is_elems['bus']

    _set_buses_out_of_service(ppc)
    bus_pq = _get_p_q_results(net, ppc, bus_lookup, bus_is, ac)
    _get_shunt_results(net, ppc, bus_lookup, bus_pq, bus_is, ac)
    _get_branch_results(net, ppc, bus_lookup, bus_pq, trafo_loading, ac)
    _get_gen_results(net, ppc, is_elems, bus_lookup, bus_pq, return_voltage_angles, ac)
    _get_bus_results(net, ppc, bus_lookup, bus_pq, return_voltage_angles, ac)


def _set_buses_out_of_service(ppc):
    disco = np.where(ppc["bus"][:, 1] == 4)[0]
    ppc["bus"][disco, VM] = np.nan
    ppc["bus"][disco, VA] = np.nan
    ppc["bus"][disco, PD] = 0
    ppc["bus"][disco, QD] = 0


def _get_bus_results(net, ppc, bus_lookup, bus_pq, return_voltage_angles, ac=True):
    net["res_bus"]["p_kw"] = bus_pq[:, 0]
    if ac:
        net["res_bus"]["q_kvar"] = bus_pq[:, 1]

    ppi = net["bus"].index.values
    bus_idx = get_indices(ppi, bus_lookup)
    if ac:
        net["res_bus"]["vm_pu"] = ppc["bus"][bus_idx][:, VM]
    net["res_bus"].index = net["bus"].index
    if return_voltage_angles or not ac:
        net["res_bus"]["va_degree"] = ppc["bus"][bus_idx][:, VA]


def _get_branch_results(net, ppc, bus_lookup, pq_buses, trafo_loading, ac=True):
    """
    Extract the bus results and writes it in the Dataframe net.res_line and net.res_trafo.

    INPUT:

        **results** - the result of runpf loadflow calculation

        **p** - the dict to dump the "res_line" and "res_trafo" Dataframe

    """
    line_end = len(net["line"])
    trafo_end = line_end + len(net["trafo"])
    trafo3w_end = trafo_end + len(net["trafo3w"]) * 3
    impedance_end = trafo3w_end + len(net["impedance"])
    xward_end = impedance_end + len(net["xward"])

    i_ft, s_ft = _get_branch_flows(net, ppc)
    if line_end > 0:
        _get_line_results(net, ppc, i_ft, 0, line_end, ac)
        net["res_line"].index = net["line"].index

    if trafo_end > line_end:
        _get_trafo_results(net, ppc, trafo_loading, s_ft, i_ft, line_end, trafo_end, ac)
        net["res_trafo"].index = net["trafo"].index

    if trafo3w_end > trafo_end:
        _get_trafo3w_results(net, ppc, trafo_loading, s_ft, i_ft, trafo_end, trafo3w_end, ac)
        net["res_trafo3w"].index = net["trafo3w"].index

    if impedance_end > trafo3w_end:
        _get_impedance_results(net, ppc, i_ft, trafo3w_end, impedance_end, ac)
        net["res_impedance"].index = net["impedance"].index

    if xward_end > impedance_end:
        _get_xward_branch_results(net, ppc, bus_lookup, pq_buses, impedance_end, xward_end, ac)


def _get_gen_results(net, ppc, is_elems, bus_lookup, pq_bus, return_voltage_angles, ac=True):
    # get in service elements
    gen_is = is_elems['gen']
    eg_is = is_elems['eg']

    eg_end = len(net['ext_grid'])
    gen_end = eg_end + len(net['gen'])

    # get results for external grids
    # bus index of in service egs
    gidx = eg_is.bus.values
    n_res_eg = len(net['ext_grid'])
    # indices of in service gens in the ppc
    gidx_ppc = np.searchsorted(ppc['gen'][:, GEN_BUS], get_indices(eg_is["bus"], bus_lookup))
    # mask for indices of in service gens in net['res_gen']
    idx_eg = np.in1d(net['ext_grid'].bus, gidx)
    # read results from ppc for these buses
    p = np.zeros(n_res_eg)
    p[idx_eg] = -ppc["gen"][gidx_ppc, PG] * 1e3
    # store result in net['res']
    net["res_ext_grid"]["p_kw"] = p

    # if ac PF q results are also available
    if ac:
        q = np.zeros(n_res_eg)
        q[idx_eg] = -ppc["gen"][gidx_ppc, QG] * 1e3
        net["res_ext_grid"]["q_kvar"] = q

    # get bus values for pq_bus
    b = net['ext_grid'].bus.values
    # copy index for results
    net["res_ext_grid"].index = net['ext_grid'].index

    # get results for gens
    if gen_end > eg_end:

        # bus index of in service gens
        gidx = gen_is.bus.values
        n_res_gen = len(net['gen'])

        b = np.hstack([b, net['gen'].bus.values])

        # indices of in service gens in the ppc
        gidx_ppc = np.searchsorted(ppc['gen'][:, GEN_BUS], get_indices(gen_is["bus"], bus_lookup))
        # mask for indices of in service gens in net['res_gen']
        idx_gen = np.in1d(net['gen'].bus, gidx)

        # read results from ppc for these buses
        p_gen = np.zeros(n_res_gen)
        p_gen[idx_gen] = -ppc["gen"][gidx_ppc, PG] * 1e3
        if ac:
            q_gen = np.zeros(n_res_gen)
            q_gen[idx_gen] = -ppc["gen"][gidx_ppc, QG] * 1e3

        v_pu = np.zeros(n_res_gen)
        v_pu[idx_gen] = ppc["bus"][gidx_ppc][:, VM]
        net["res_gen"]["vm_pu"] = v_pu
        if return_voltage_angles:
            v_a = np.zeros(n_res_gen)
            v_a[idx_gen] = ppc["bus"][gidx_ppc][:, VA]
            net["res_gen"]["va_degree"] = v_a
        net["res_gen"].index = net['gen'].index

        # store result in net['res']
        p = np.hstack([p, p_gen])
        net["res_gen"]["p_kw"] = p_gen
        if ac:
            q = np.hstack([q, q_gen])
            net["res_gen"]["q_kvar"] = q_gen

    if not ac:
        q = np.zeros(len(p))
    b_sum, p_sum, q_sum = _sum_by_group(b, p, q)
    b = get_indices(b_sum, bus_lookup, fused_indices=False)
    pq_bus[b, 0] += p_sum
    pq_bus[b, 1] += q_sum


def _get_xward_branch_results(net, ppc, bus_lookup, pq_buses, f, t, ac=True):
    p_branch_xward = ppc["branch"][f:t, PF].real * 1e3
    net["res_xward"]["p_kw"] += p_branch_xward
    if ac:
        q_branch_xward = ppc["branch"][f:t, QF].real * 1e3
        net["res_xward"]["q_kvar"] += q_branch_xward
    else:
        q_branch_xward = np.zeros(len(p_branch_xward))
    b_pp, p, q = _sum_by_group(net["xward"]["bus"].values, p_branch_xward, q_branch_xward)
    b_ppc = get_indices(b_pp, bus_lookup, fused_indices=False)

    pq_buses[b_ppc, 0] += p
    pq_buses[b_ppc, 1] += q


def _get_branch_flows(net, ppc):
    br_idx = ppc["branch"][:, (F_BUS, T_BUS)].real.astype(int)
    u_ft = ppc["bus"][br_idx, 7] * ppc["bus"][br_idx, BASE_KV]
    s_ft = (np.sqrt(ppc["branch"][:, (PF, PT)].real**2 +
                    ppc["branch"][:, (QF, QT)].real**2) * 1e3)
    i_ft = s_ft * 1e-3 / u_ft / np.sqrt(3)
    return i_ft, s_ft


def _get_line_results(net, ppc, i_ft, f, t, ac=True):
    pf_kw = ppc["branch"][f:t, PF].real * 1e3
    qf_kvar = ppc["branch"][f:t, QF].real * 1e3
    net["res_line"]["p_from_kw"] = pf_kw
    if ac:
        net["res_line"]["q_from_kvar"] = qf_kvar

    pt_kw = ppc["branch"][f:t, PT].real * 1e3
    qt_kvar = ppc["branch"][f:t, QT].real * 1e3
    net["res_line"]["p_to_kw"] = pt_kw
    if ac:
        net["res_line"]["q_to_kvar"] = qt_kvar

    if ac:
        net["res_line"]["pl_kw"] = pf_kw + pt_kw
        net["res_line"]["ql_kvar"] = qf_kvar + qt_kvar

    i_ka = np.max(i_ft[f:t], axis=1)
    i_max = net["line"]["imax_ka"].values * net["line"]["df"].values * \
        net["line"]["parallel"].values

    net["res_line"]["i_ka"] = i_ka
    net["res_line"]["loading_percent"] = i_ka / i_max * 100


def _get_trafo_results(net, ppc, trafo_loading, s_ft, i_ft, f, t, ac=True):
    phv_kw = ppc["branch"][f:t, PF].real * 1e3
    plv_kw = ppc["branch"][f:t, PT].real * 1e3
    net["res_trafo"]["p_hv_kw"] = phv_kw
    net["res_trafo"]["p_lv_kw"] = plv_kw

    if ac:
        qhv_kvar = ppc["branch"][f:t, QF].real * 1e3
        qlv_kvar = ppc["branch"][f:t, QT].real * 1e3
        net["res_trafo"]["q_hv_kvar"] = qhv_kvar
        net["res_trafo"]["q_lv_kvar"] = qlv_kvar
        net["res_trafo"]["pl_kw"] = phv_kw + plv_kw
        net["res_trafo"]["ql_kvar"] = qhv_kvar + qlv_kvar

    net["res_trafo"]["i_hv_ka"] = i_ft[:, 0][f:t]
    net["res_trafo"]["i_lv_ka"] = i_ft[:, 1][f:t]
    if trafo_loading == "current":
        lds_trafo = i_ft[f:t] * net["trafo"][["vn_hv_kv", "vn_lv_kv"]].values * 1000. * np.sqrt(3) \
            / net["trafo"]["sn_kva"].values[:, np.newaxis] * 100.
        ld_trafo = np.max(lds_trafo, axis=1)
    elif trafo_loading == "power":
        ld_trafo = np.max(s_ft[f:t] / net["trafo"]["sn_kva"].values[:, np.newaxis] * 100., axis=1)
    else:
        raise ValueError(
            "Unknown transformer loading parameter %s - choose 'current' or 'power'" % trafo_loading)
    net["res_trafo"]["loading_percent"] = ld_trafo


def _get_trafo3w_results(net, ppc, trafo_loading, s_ft, i_ft, f, t, ac=True):
    hv = int(f + (t - f) / 3)
    mv = int(f + 2 * (t - f) / 3)
    lv = t

    phv_kw = ppc["branch"][f:hv, PF].real * 1e3
    pmv_kw = ppc["branch"][hv:mv, PT].real * 1e3
    plv_kw = ppc["branch"][mv:lv, PT].real * 1e3

    net["res_trafo3w"]["p_hv_kw"] = phv_kw
    net["res_trafo3w"]["p_mv_kw"] = pmv_kw
    net["res_trafo3w"]["p_lv_kw"] = plv_kw

    if ac:
        qhv_kvar = ppc["branch"][f:hv, QF].real * 1e3
        qmv_kvar = ppc["branch"][hv:mv, QT].real * 1e3
        qlv_kvar = ppc["branch"][mv:lv, QT].real * 1e3
        net["res_trafo3w"]["q_hv_kvar"] = qhv_kvar
        net["res_trafo3w"]["q_mv_kvar"] = qmv_kvar
        net["res_trafo3w"]["q_lv_kvar"] = qlv_kvar
        net["res_trafo3w"]["pl_kw"] = phv_kw + pmv_kw + plv_kw
        net["res_trafo3w"]["ql_kvar"] = qhv_kvar + qmv_kvar + qlv_kvar

    i_h = i_ft[:, 0][f:hv]
    i_m = i_ft[:, 1][hv:mv]
    i_l = i_ft[:, 1][mv:lv]
    net["res_trafo3w"]["i_hv_ka"] = i_h
    net["res_trafo3w"]["i_mv_ka"] = i_m
    net["res_trafo3w"]["i_lv_ka"] = i_l

    t3 = net["trafo3w"]
    if trafo_loading == "current":
        ld_h = i_h * t3["vn_hv_kv"].values * 1000. * np.sqrt(3) / t3["sn_hv_kva"].values * 100
        ld_m = i_m * t3["vn_mv_kv"].values * 1000. * np.sqrt(3) / t3["sn_mv_kva"].values * 100
        ld_l = i_l * t3["vn_lv_kv"].values * 1000. * np.sqrt(3) / t3["sn_lv_kva"].values * 100
        ld_trafo = np.max(np.vstack([ld_h, ld_m, ld_l]), axis=0)
    elif trafo_loading == "power":
        ld_h = s_ft[:, 0][f:hv] / t3["sn_hv_kva"] * 100.
        ld_m = s_ft[:, 1][hv:mv] / t3["sn_mv_kva"] * 100.
        ld_l = s_ft[:, 1][mv:lv] / t3["sn_lv_kva"] * 100.
        ld_trafo = np.max(np.vstack([ld_h, ld_m, ld_l]), axis=0)
    else:
        raise ValueError(
            "Unknown transformer loading parameter %s - choose 'current' or 'power'" % trafo_loading)
    net["res_trafo3w"]["loading_percent"] = ld_trafo


def _get_impedance_results(net, ppc, i_ft, f, t, ac=True):
    pf_kw = ppc["branch"][f:t, (PF)].real * 1e3
    pt_kw = ppc["branch"][f:t, (PT)].real * 1e3
    net["res_impedance"]["p_from_kw"] = pf_kw
    net["res_impedance"]["p_to_kw"] = pt_kw

    if ac:
        qf_kvar = ppc["branch"][f:t, (QF)].real * 1e3
        qt_kvar = ppc["branch"][f:t, (QT)].real * 1e3
        net["res_impedance"]["q_from_kvar"] = qf_kvar
        net["res_impedance"]["q_to_kvar"] = qt_kvar
        net["res_impedance"]["ql_kvar"] = qf_kvar + qt_kvar
        net["res_impedance"]["pl_kw"] = pf_kw + pt_kw

    net["res_impedance"]["i_from_ka"] = i_ft[f:t][:, 0]
    net["res_impedance"]["i_to_ka"] = i_ft[f:t][:, 1]


def _get_p_q_results(net, ppc, bus_lookup, bus_is, ac=True):
    bus_pq = np.zeros(shape=(len(net["bus"].index), 2), dtype=np.float)
    b, p, q = np.array([]), np.array([]), np.array([])

    l = net["load"]
    if len(l) > 0:
        load_is = np.in1d(l.bus.values, bus_is.index) \
            & l.in_service.values.astype(bool)
        scaling = l["scaling"].values
        pl = l["p_kw"].values * scaling * load_is
        net["res_load"]["p_kw"] = pl
        p = np.hstack([p, pl])
        if ac:
            ql = l["q_kvar"].values * scaling * load_is
            net["res_load"]["q_kvar"] = ql
            q = np.hstack([q, ql])
        b = np.hstack([b, l["bus"].values])
        net["res_load"].index = net["load"].index

    sg = net["sgen"]
    if len(sg) > 0:
        sgen_is = np.in1d(sg.bus.values, bus_is.index) \
            & sg.in_service.values.astype(bool)
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
        ward_is = np.in1d(w.bus.values, bus_is.index) \
            & w.in_service.values.astype(bool)
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
        xward_is = np.in1d(xw.bus.values, bus_is.index) \
            & xw.in_service.values.astype(bool)
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
    b_ppc = get_indices(b_pp, bus_lookup, fused_indices=False)
    bus_pq[b_ppc, 0] = vp
    bus_pq[b_ppc, 1] = vq
    return bus_pq


def _get_shunt_results(net, ppc, bus_lookup, bus_pq, bus_is, ac=True):
    b, p, q = np.array([]), np.array([]), np.array([])
    s = net["shunt"]
    if len(s) > 0:
        sidx = get_indices(s["bus"], bus_lookup)
        shunt_is = np.in1d(s.bus.values, bus_is.index) \
            & s.in_service.values.astype(bool)
        u_shunt = ppc["bus"][sidx, VM]
        u_shunt = np.nan_to_num(u_shunt)
        p_shunt = u_shunt**2 * net["shunt"]["p_kw"].values * shunt_is
        net["res_shunt"]["p_kw"] = p_shunt
        p = np.hstack([p, p_shunt])
        if ac:
            net["res_shunt"]["vm_pu"] = u_shunt
            q_shunt = u_shunt**2 * net["shunt"]["q_kvar"].values * shunt_is
            net["res_shunt"]["q_kvar"] = q_shunt
            q = np.hstack([q, q_shunt])
        b = np.hstack([b, s["bus"].values])
        net["res_shunt"].index = net["shunt"].index

    w = net["ward"]
    if len(w) > 0:
        widx = get_indices(w["bus"], bus_lookup)
        ward_is = np.in1d(w.bus.values, bus_is.index) \
            & w.in_service.values.astype(bool)
        u_ward = ppc["bus"][widx, VM]
        u_ward = np.nan_to_num(u_ward)
        p_ward = u_ward**2 * net["ward"]["pz_kw"].values * ward_is
        net["res_ward"]["p_kw"] += p_ward
        p = np.hstack([p, p_ward])
        if ac:
            net["res_ward"]["vm_pu"] = u_ward
            q_ward = u_ward**2 * net["ward"]["qz_kvar"].values * ward_is
            net["res_ward"]["q_kvar"] += q_ward
            q = np.hstack([q, q_ward])
        b = np.hstack([b, w["bus"].values])
        net["res_ward"].index = net["ward"].index

    xw = net["xward"]
    if len(xw) > 0:
        widx = get_indices(xw["bus"], bus_lookup)
        xward_is = np.in1d(xw.bus.values, bus_is.index) \
            & xw.in_service.values.astype(bool)
        u_xward = ppc["bus"][widx, VM]
        u_xward = np.nan_to_num(u_xward)
        p_xward = u_xward**2 * net["xward"]["pz_kw"].values * xward_is
        net["res_xward"]["p_kw"] += p_xward
        p = np.hstack([p, p_xward])
        if ac:
            net["res_xward"]["vm_pu"] = u_xward
            q_xward = u_xward**2 * net["xward"]["qz_kvar"].values * xward_is
            net["res_xward"]["q_kvar"] += q_xward
            q = np.hstack([q, q_xward])
        b = np.hstack([b, xw["bus"].values])
        net["res_xward"].index = net["xward"].index

    if not ac:
        q = np.zeros(len(p))
    b_pp, vp, vq = _sum_by_group(b.astype(int), p, q)
    b_ppc = get_indices(b_pp, bus_lookup, fused_indices=False)

    bus_pq[b_ppc, 0] += vp
    if ac:
        bus_pq[b_ppc, 1] += vq
