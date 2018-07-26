# -*- coding: utf-8 -*-

# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pandas as pd

from pandapower.auxiliary import _sum_by_group, I_from_SV_elementwise, sequence_to_phase, S_from_VI_elementwise
from pandapower.idx_brch import F_BUS, T_BUS, PF, QF, PT, QT
from pandapower.idx_bus import BASE_KV, VM, VA


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


def _get_branch_results_3ph(net, ppc0, ppc1, ppc2, bus_lookup_aranged, pq_buses):
    """
    Extract the bus results and writes it in the Dataframe net.res_line and net.res_trafo.

    INPUT:

        **results** - the result of runpf loadflow calculation

        **p** - the dict to dump the "res_line" and "res_trafo" Dataframe

    """
    I012_f, S012_f, V012_f, I012_t, S012_t, V012_t = _get_branch_flows_3ph(ppc0, ppc1, ppc2)
    _get_line_results_3ph(net, ppc0, ppc1, ppc2, I012_f, V012_f, I012_t, V012_t)
    # _get_trafo_results(net, ppc, s_ft, i_ft)
    # _get_trafo3w_results(net, ppc, s_ft, i_ft)
    # _get_impedance_results(net, ppc, i_ft)
    # _get_xward_branch_results(net, ppc, bus_lookup_aranged, pq_buses)
    # _get_switch_results(net, i_ft)


def _get_branch_flows(ppc):
    br_idx = ppc["branch"][:, (F_BUS, T_BUS)].real.astype(int)
    u_ft = ppc["bus"][br_idx, 7] * ppc["bus"][br_idx, BASE_KV]
    s_ft = (np.sqrt(ppc["branch"][:, (PF, PT)].real ** 2 +
                    ppc["branch"][:, (QF, QT)].real ** 2) * 1e3)
    i_ft = s_ft * 1e-3 / u_ft / np.sqrt(3)
    return i_ft, s_ft


def _get_branch_flows_3ph(ppc0, ppc1, ppc2):
    br_from_idx = ppc1["branch"][:, F_BUS].real.astype(int)
    br_to_idx = ppc1["branch"][:, T_BUS].real.astype(int)
    V012_f = np.matrix([(ppc["bus"][br_from_idx, VM] * ppc["bus"][br_from_idx, BASE_KV] *
                         np.exp(1j * np.deg2rad(ppc["bus"][br_from_idx, VA]))).flatten() for ppc in [ppc0, ppc1, ppc2]])
    V012_t = np.matrix([(ppc["bus"][br_to_idx, VM] * ppc["bus"][br_to_idx, BASE_KV] *
                         np.exp(1j * np.deg2rad(ppc["bus"][br_to_idx, VA]))).flatten() for ppc in [ppc0, ppc1, ppc2]])
    S012_f = np.matrix([((ppc["branch"][:, PF].real +
                    1j * ppc["branch"][:, QF].real) * 1e3)
                    for ppc in [ppc0, ppc1, ppc2]])
    S012_t = np.matrix([((ppc["branch"][:, PT].real +
                    1j * ppc["branch"][:, QT].real) * 1e3)
                    for ppc in [ppc0, ppc1, ppc2]])
    I012_f = I_from_SV_elementwise(S012_f * 1e-3, V012_f / np.sqrt(3))
    I012_t = I_from_SV_elementwise(S012_t * 1e-3, V012_t / np.sqrt(3))

    return I012_f, S012_f, V012_f, I012_t, S012_t, V012_t


def _get_line_results(net, ppc, i_ft):
    # create res_line_vals which are written to the pandas dataframe
    ac = net["_options"]["ac"]

    if not "line" in net._pd2ppc_lookups["branch"]:
        return
    f, t = net._pd2ppc_lookups["branch"]["line"]
    pf_kw = ppc["branch"][f:t, PF].real * 1e3
    q_from_kvar = ppc["branch"][f:t, QF].real * 1e3
    p_from_kw = pf_kw

    pt_kw = ppc["branch"][f:t, PT].real * 1e3
    q_to_kvar = ppc["branch"][f:t, QT].real * 1e3
    p_to_kw = pt_kw

    if ac:
        pl_kw = pf_kw + pt_kw
        ql_kvar = q_from_kvar + q_to_kvar
    else:
        pl_kw = np.zeros_like(pf_kw)
        ql_kvar = np.zeros_like(q_from_kvar)

    i_ka = np.max(i_ft[f:t], axis=1)
    i_from_ka = i_ft[f:t][:, 0]
    i_to_ka = i_ft[f:t][:, 1]
    line_df = net["line"]
    i_max = line_df["max_i_ka"].values * line_df["df"].values * line_df["parallel"].values

    # write to line
    res_line_df = net["res_line"]
    res_line_df["p_from_kw"].values[:] = p_from_kw
    res_line_df["q_from_kvar"].values[:] = q_from_kvar
    res_line_df["p_to_kw"].values[:] = p_to_kw
    res_line_df["q_to_kvar"].values[:] = q_to_kvar
    res_line_df["pl_kw"].values[:] = pl_kw
    res_line_df["ql_kvar"].values[:] = ql_kvar
    res_line_df["i_from_ka"].values[:] = i_from_ka
    res_line_df["i_to_ka"].values[:] = i_to_ka
    res_line_df["i_ka"].values[:] = i_ka
    res_line_df["loading_percent"].values[:] = i_ka / i_max * 100


def _get_line_results_3ph(net, ppc0, ppc1, ppc2, I012_f, V012_f, I012_t, V012_t):
    # create res_line_vals which are written to the pandas dataframe
    ac = net["_options"]["ac"]

    if not "line" in net._pd2ppc_lookups["branch"]:
        return

    f, t = net._pd2ppc_lookups["branch"]["line"]
    I012_from_ka = I012_f[:, f:t]
    I012_to_ka = I012_t[:, f:t]
    line_df = net["line"]
    i_max_phase = line_df["max_i_ka"].values * line_df["df"].values * line_df["parallel"].values

    Vabc_f, Vabc_t, Iabc_f, Iabc_t = [sequence_to_phase(X012) for X012 in [V012_f, V012_t, I012_f, I012_t]]
    Sabc_f, Sabc_t = [S_from_VI_elementwise(*Xabc_tup) * 1e3 / np.sqrt(3) for Xabc_tup in [(Vabc_f, Iabc_f), (Vabc_t, Iabc_t)]]
    # Todo: Check why the sqrt(3) is necessary in the previous line as opposed to _get_line_results()
    Pabcf_kw = Sabc_f.real[:, f:t]
    Qabcf_kvar = Sabc_f.imag[:, f:t]
    Pabct_kw = Sabc_t.real[:, f:t]
    Qabct_kvar = Sabc_t.imag[:, f:t]
    if ac:
        Pabcl_kw = Pabcf_kw + Pabct_kw
        Qabcl_kvar = Qabcf_kvar + Qabct_kvar
    else:
        Pabcl_kw = np.zeros_like(Pabcf_kw)
        Qabcl_kvar = np.zeros_like(Qabct_kvar)
    Iabc_f_ka = np.abs(sequence_to_phase(I012_from_ka))
    Iabc_t_ka = np.abs(sequence_to_phase(I012_to_ka))
    Iabc_ka = np.maximum.reduce([Iabc_t_ka, Iabc_f_ka])

    # write to line
    net["res_line_3ph"]["pA_from_kw"] = Pabcf_kw[0, :].A1
    net["res_line_3ph"]["pB_from_kw"] = Pabcf_kw[1, :].A1
    net["res_line_3ph"]["pC_from_kw"] = Pabcf_kw[2, :].A1
    net["res_line_3ph"]["qA_from_kvar"] = Qabcf_kvar[0, :].A1
    net["res_line_3ph"]["qB_from_kvar"] = Qabcf_kvar[1, :].A1
    net["res_line_3ph"]["qC_from_kvar"] = Qabcf_kvar[2, :].A1
    net["res_line_3ph"]["pA_to_kw"] = Pabct_kw[0, :].A1
    net["res_line_3ph"]["pB_to_kw"] = Pabct_kw[1, :].A1
    net["res_line_3ph"]["pC_to_kw"] = Pabct_kw[2, :].A1
    net["res_line_3ph"]["qA_to_kvar"] = Qabct_kvar[0, :].A1
    net["res_line_3ph"]["qB_to_kvar"] = Qabct_kvar[1, :].A1
    net["res_line_3ph"]["qC_to_kvar"] = Qabct_kvar[2, :].A1
    net["res_line_3ph"]["pAl_kw"] = Pabcl_kw[0, :].A1
    net["res_line_3ph"]["pBl_kw"] = Pabcl_kw[1, :].A1
    net["res_line_3ph"]["pCl_kw"] = Pabcl_kw[2, :].A1
    net["res_line_3ph"]["qAl_kvar"] = Qabcl_kvar[0, :].A1
    net["res_line_3ph"]["qBl_kvar"] = Qabcl_kvar[1, :].A1
    net["res_line_3ph"]["qCl_kvar"] = Qabcl_kvar[2, :].A1
    net["res_line_3ph"]["iA_from_ka"] = Iabc_f_ka[0, :].A1
    net["res_line_3ph"]["iB_from_ka"] = Iabc_f_ka[1, :].A1
    net["res_line_3ph"]["iC_from_ka"] = Iabc_f_ka[2, :].A1
    net["res_line_3ph"]["iA_to_ka"] = Iabc_t_ka[0, :].A1
    net["res_line_3ph"]["iB_to_ka"] = Iabc_t_ka[1, :].A1
    net["res_line_3ph"]["iC_to_ka"] = Iabc_t_ka[2, :].A1
    net["res_line_3ph"]["iA_ka"] = Iabc_ka[0, :]
    net["res_line_3ph"]["iB_ka"] = Iabc_ka[1, :]
    net["res_line_3ph"]["iC_ka"] = Iabc_ka[2, :]
    net["res_line_3ph"]["loading_percentA"] = Iabc_ka[0, :] / i_max_phase * 100
    net["res_line_3ph"]["loading_percentB"] = Iabc_ka[1, :] / i_max_phase * 100
    net["res_line_3ph"]["loading_percentC"] = Iabc_ka[2, :] / i_max_phase * 100
    net["res_line_3ph"]["loading_percent"] = Iabc_ka.max(axis=0) / i_max_phase * 100
    net["res_line_3ph"].index = net["line"].index


def _get_trafo_results(net, ppc, s_ft, i_ft):
    ac = net["_options"]["ac"]
    trafo_loading = net["_options"]["trafo_loading"]

    if not "trafo" in net._pd2ppc_lookups["branch"]:
        return
    f, t = net._pd2ppc_lookups["branch"]["trafo"]
    p_hv_kw = ppc["branch"][f:t, PF].real * 1e3
    p_lv_kw = ppc["branch"][f:t, PT].real * 1e3

    if ac:
        q_hv_kvar = ppc["branch"][f:t, QF].real * 1e3
        q_lv_kvar = ppc["branch"][f:t, QT].real * 1e3
        pl_kw = p_hv_kw + p_lv_kw
        ql_kvar = q_hv_kvar + q_lv_kvar
    else:
        q_hv_kvar = np.zeros_like(p_hv_kw)
        q_lv_kvar = np.zeros_like(p_lv_kw)
        pl_kw = np.zeros_like(p_lv_kw)
        ql_kvar = np.zeros_like(p_lv_kw)

    i_hv_ka = i_ft[:, 0][f:t]
    i_lv_ka = i_ft[:, 1][f:t]
    if trafo_loading == "current":
        trafo_df = net["trafo"]
        vns = np.vstack([trafo_df["vn_hv_kv"].values, trafo_df["vn_lv_kv"].values]).T
        lds_trafo = i_ft[f:t] * vns * 1000. * np.sqrt(3) \
                    / trafo_df["sn_kva"].values[:, np.newaxis] * 100.
        ld_trafo = np.max(lds_trafo, axis=1)
    elif trafo_loading == "power":
        ld_trafo = np.max(s_ft[f:t] / net["trafo"]["sn_kva"].values[:, np.newaxis] * 100., axis=1)
    else:
        raise ValueError(
            "Unknown transformer loading parameter %s - choose 'current' or 'power'" % trafo_loading)
    if any(net["trafo"]["df"].values <= 0):
        raise UserWarning('Transformer rating factor df must be positive. Transformers with false '
                          'rating factors: %s' % net["trafo"].query('df<=0').index.tolist())
    loading_percent = \
        ld_trafo / net["trafo"]["parallel"].values / net["trafo"]["df"].values

    # write results to trafo dataframe
    res_trafo_df = net["res_trafo"]
    res_trafo_df["p_hv_kw"].values[:] = p_hv_kw
    res_trafo_df["q_hv_kvar"].values[:] = q_hv_kvar
    res_trafo_df["p_lv_kw"].values[:] = p_lv_kw
    res_trafo_df["q_lv_kvar"].values[:] = q_lv_kvar
    res_trafo_df["pl_kw"].values[:] = pl_kw
    res_trafo_df["ql_kvar"].values[:] = ql_kvar
    res_trafo_df["i_hv_ka"].values[:] = i_hv_ka
    res_trafo_df["i_lv_ka"].values[:] = i_lv_ka
    res_trafo_df["loading_percent"].values[:] = loading_percent


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

    p_hv_kw = phv_kw
    p_mv_kw = pmv_kw
    p_lv_kw = plv_kw

    if ac:
        q_hv_kvar = ppc["branch"][f:hv, QF].real * 1e3
        q_mv_kvar = ppc["branch"][hv:mv, QT].real * 1e3
        q_lv_kvar = ppc["branch"][mv:lv, QT].real * 1e3
        pl_kw = phv_kw + pmv_kw + plv_kw
        ql_kvar = q_hv_kvar + q_mv_kvar + q_lv_kvar
    else:
        zeros = np.zeros_like(phv_kw)
        q_hv_kvar = zeros
        q_mv_kvar = zeros
        q_lv_kvar = zeros
        pl_kw = zeros
        ql_kvar = zeros

    i_h = i_ft[:, 0][f:hv]
    i_m = i_ft[:, 1][hv:mv]
    i_l = i_ft[:, 1][mv:lv]
    i_hv_ka = i_h
    i_mv_ka = i_m
    i_lv_ka = i_l

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
    loading_percent = ld_trafo

    # write results to trafo3w dataframe
    res_trafo3w_df = net["res_trafo3w"]
    res_trafo3w_df["p_hv_kw"].values[:] = p_hv_kw
    res_trafo3w_df["q_hv_kvar"].values[:] = q_hv_kvar
    res_trafo3w_df["p_mv_kw"].values[:] = p_mv_kw
    res_trafo3w_df["q_mv_kvar"].values[:] = q_mv_kvar
    res_trafo3w_df["p_lv_kw"].values[:] = p_lv_kw
    res_trafo3w_df["q_lv_kvar"].values[:] = q_lv_kvar
    res_trafo3w_df["pl_kw"].values[:] = pl_kw
    res_trafo3w_df["ql_kvar"].values[:] = ql_kvar
    res_trafo3w_df["i_hv_ka"].values[:] = i_hv_ka
    res_trafo3w_df["i_mv_ka"].values[:] = i_mv_ka
    res_trafo3w_df["i_lv_ka"].values[:] = i_lv_ka
    res_trafo3w_df["loading_percent"].values[:] = loading_percent


def _get_impedance_results(net, ppc, i_ft):
    ac = net["_options"]["ac"]

    if not "impedance" in net._pd2ppc_lookups["branch"]:
        return
    f, t = net._pd2ppc_lookups["branch"]["impedance"]
    pf_kw = ppc["branch"][f:t, (PF)].real * 1e3
    pt_kw = ppc["branch"][f:t, (PT)].real * 1e3
    p_from_kw = pf_kw
    p_to_kw = pt_kw

    if ac:
        q_from_kvar = ppc["branch"][f:t, (QF)].real * 1e3
        q_to_kvar = ppc["branch"][f:t, (QT)].real * 1e3
        ql_kvar = q_from_kvar + q_to_kvar
        pl_kw = pf_kw + pt_kw
    else:
        zeros = np.zeros_like(p_from_kw)
        # this looks like a pyramid
        q_from_kvar = zeros
        q_to_kvar = zeros
        ql_kvar = zeros
        pl_kw = zeros
        # zeros

    i_from_ka = i_ft[f:t][:, 0]
    i_to_ka = i_ft[f:t][:, 1]

    # write to impedance
    res_impediance_df = net["res_impedance"]
    res_impediance_df["p_from_kw"].values[:] = p_from_kw
    res_impediance_df["q_from_kvar"].values[:] = q_from_kvar
    res_impediance_df["p_to_kw"].values[:] = p_to_kw
    res_impediance_df["q_to_kvar"].values[:] = q_to_kvar
    res_impediance_df["pl_kw"].values[:] = pl_kw
    res_impediance_df["ql_kvar"].values[:] = ql_kvar
    res_impediance_df["i_from_ka"].values[:] = i_from_ka
    res_impediance_df["i_to_ka"].values[:] = i_to_ka


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
