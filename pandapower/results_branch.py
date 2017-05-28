# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import numpy as np
import pandas as pd
from pandapower.idx_brch import F_BUS, T_BUS, PF, QF, PT, QT
from pandapower.idx_bus import BASE_KV

from pandapower.auxiliary import _sum_by_group


def _get_branch_results(net, ppc, bus_lookup_aranged, pq_buses):
    """
    Extract the bus results and writes it in the Dataframe net.res_line and net.res_trafo.

    INPUT:

        **results** - the result of runpf loadflow calculation

        **p** - the dict to dump the "res_line" and "res_trafo" Dataframe

    """
    i_ft, s_ft = _get_branch_flows(ppc)
    _get_line_results(net, ppc, i_ft)
    _get_trafo_results(net, ppc, s_ft, i_ft)
    _get_trafo3w_results(net, ppc, s_ft, i_ft)
    _get_impedance_results(net, ppc, i_ft)
    _get_xward_branch_results(net, ppc, bus_lookup_aranged, pq_buses)
    _get_switch_results(net, i_ft)


def _get_branch_flows(ppc):
    br_idx = ppc["branch"][:, (F_BUS, T_BUS)].real.astype(int)
    u_ft = ppc["bus"][br_idx, 7] * ppc["bus"][br_idx, BASE_KV]
    s_ft = (np.sqrt(ppc["branch"][:, (PF, PT)].real ** 2 +
                    ppc["branch"][:, (QF, QT)].real ** 2) * 1e3)
    i_ft = s_ft * 1e-3 / u_ft / np.sqrt(3)
    return i_ft, s_ft


def _get_line_results(net, ppc, i_ft):
    ac = net["_options"]["ac"]

    if not "line" in net._pd2ppc_lookups["branch"]:
        return
    f, t = net._pd2ppc_lookups["branch"]["line"]
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
    net["res_line"]["i_from_ka"] = i_ft[f:t][:, 0]
    net["res_line"]["i_to_ka"] = i_ft[f:t][:, 1]
    i_max = net["line"]["max_i_ka"].values * net["line"]["df"].values * \
            net["line"]["parallel"].values

    net["res_line"]["i_ka"] = i_ka
    net["res_line"]["loading_percent"] = i_ka / i_max * 100
    net["res_line"].index = net["line"].index


def _get_trafo_results(net, ppc, s_ft, i_ft):
    ac = net["_options"]["ac"]
    trafo_loading = net["_options"]["trafo_loading"]

    if not "trafo" in net._pd2ppc_lookups["branch"]:
        return
    f, t = net._pd2ppc_lookups["branch"]["trafo"]
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
    net["res_trafo"]["loading_percent"] = ld_trafo / net["trafo"]["parallel"].values
    net["res_trafo"].index = net["trafo"].index


def _get_trafo3w_results(net, ppc, s_ft, i_ft):
    trafo_loading = net["_options"]["trafo_loading"]
    ac = net["_options"]["ac"]

    if not "trafo3w" in net._pd2ppc_lookups["branch"]:
        return
    f, t = net._pd2ppc_lookups["branch"]["trafo3w"]
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
    net["res_trafo3w"].index = net["trafo3w"].index


def _get_impedance_results(net, ppc, i_ft):
    ac = net["_options"]["ac"]

    if not "impedance" in net._pd2ppc_lookups["branch"]:
        return
    f, t = net._pd2ppc_lookups["branch"]["impedance"]
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
    net["res_impedance"].index = net["impedance"].index


def _get_xward_branch_results(net, ppc, bus_lookup_aranged, pq_buses):
    ac = net["_options"]["ac"]

    if not "xward" in net._pd2ppc_lookups["branch"]:
        return
    f, t = net._pd2ppc_lookups["branch"]["xward"]
    p_branch_xward = ppc["branch"][f:t, PF].real * 1e3
    net["res_xward"]["p_kw"] += p_branch_xward
    if ac:
        q_branch_xward = ppc["branch"][f:t, QF].real * 1e3
        net["res_xward"]["q_kvar"] += q_branch_xward
    else:
        q_branch_xward = np.zeros(len(p_branch_xward))
    b_pp, p, q = _sum_by_group(net["xward"]["bus"].values, p_branch_xward, q_branch_xward)
    b_ppc = bus_lookup_aranged[b_pp]

    pq_buses[b_ppc, 0] += p
    pq_buses[b_ppc, 1] += q
    net["res_xward"].index = net["xward"].index


def _get_switch_results(net, i_ft):
    if not "switch" in net._pd2ppc_lookups["branch"]:
        return
    f, t = net._pd2ppc_lookups["branch"]["switch"]
    net["res_switch"] = pd.DataFrame(data=np.max(i_ft[f:t], axis=1), columns=["i_ka"],
                                     index=net.switch[net._closed_bb_switches].index)
