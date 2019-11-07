from cmath import rect

from numpy import real, vectorize, deg2rad, maximum, sqrt

from pandapower import F_BUS, T_BUS
from pandapower.pf.pfsoln_numba import calc_branch_flows_batch
from pandapower.pypower.idx_bus import BASE_KV
from pandapower.results_branch import _get_trafo3w_lookups


def get_batch_line_results(net, i_abs):
    f, t = net._pd2ppc_lookups["branch"]["line"]
    line_df = net["line"]
    i_max = line_df["max_i_ka"].values * line_df["df"].values * line_df["parallel"].values
    i_ka = maximum(i_abs[0][:, f:t], i_abs[1][:, f:t])
    loading_percent = i_ka / i_max * 100
    i_from_ka = i_abs[0][:, :len(net.line)]
    i_to_ka = i_abs[1][:, :len(net.line)]
    return i_ka, i_from_ka, i_to_ka, loading_percent


def get_batch_trafo_results(net, i_abs, s_abs):
    if "trafo" not in net._pd2ppc_lookups["branch"]:
        return
    f, t = net._pd2ppc_lookups["branch"]["trafo"]

    i_ka = maximum(i_abs[0][:, f:t], i_abs[1][:, f:t])
    s_mva = maximum(s_abs[0][:, f:t], s_abs[1][:, f:t])
    sn_mva = net["trafo"]["sn_mva"].values

    trafo_loading = net["_options"]["trafo_loading"]

    if trafo_loading == "current":
        # get loading percent from rated current
        trafo_df = net["trafo"]
        s_hv = i_abs[0][:, f:t] * trafo_df["vn_hv_kv"].values * sqrt(3) / sn_mva * 100.
        s_lv = i_abs[1][:, f:t] * trafo_df["vn_lv_kv"].values * sqrt(3) / sn_mva * 100.
        ld_trafo = maximum(s_hv, s_lv)
    elif trafo_loading == "power":
        # get loading percent from rated power
        ld_trafo = s_mva / sn_mva * 100.
    else:
        raise ValueError(
            "Unknown transformer loading parameter %s - choose 'current' or 'power'" % trafo_loading)

    loading_percent = ld_trafo / net["trafo"]["parallel"].values / net["trafo"]["df"].values
    i_hv_ka = i_abs[0][:, f:t]
    i_lv_ka = i_abs[1][:, f:t]
    return i_ka, i_hv_ka, i_lv_ka, s_mva, loading_percent


def get_batch_trafo3w_results(net, i_abs, s_abs):
    if "trafo3w" not in net._pd2ppc_lookups["branch"]:
        return

    f, hv, mv, lv = _get_trafo3w_lookups(net)
    i_h = i_abs[0][:, f:hv]
    i_m = i_abs[1][:, hv:mv]
    i_l = i_abs[1][:, mv:lv]

    t3 = net["trafo3w"]
    trafo_loading = net["_options"]["trafo_loading"]
    if trafo_loading == "current":
        ld_h = i_h * t3["vn_hv_kv"].values * sqrt(3) / t3["sn_hv_mva"].values * 100
        ld_m = i_m * t3["vn_mv_kv"].values * sqrt(3) / t3["sn_mv_mva"].values * 100
        ld_l = i_l * t3["vn_lv_kv"].values * sqrt(3) / t3["sn_lv_mva"].values * 100
        ld_trafo = maximum(maximum(ld_h, ld_m), ld_l)
    elif trafo_loading == "power":
        ld_h = s_abs[0][:, f:hv] / t3["sn_hv_mva"].values * 100.
        ld_m = s_abs[1][:, hv:mv] / t3["sn_mv_mva"].values * 100.
        ld_l = s_abs[1][:, mv:lv] / t3["sn_lv_mva"].values * 100.
        ld_trafo = maximum(maximum(ld_h, ld_m), ld_l)
    else:
        raise ValueError(
            "Unknown transformer loading parameter %s - choose 'current' or 'power'" % trafo_loading)

    return i_h, i_m, i_l, ld_trafo


def v_to_i_s(net, vm, va):
    ppc = net["_ppc"]
    internal = ppc["internal"]
    Yf = internal["Yf"]
    Yt = internal["Yt"]
    V = polar_to_rad(vm, va)

    baseMVA = internal["baseMVA"]
    branch = internal["branch"]
    base_kv = internal["bus"][:, BASE_KV]
    f_bus = real(branch[:, F_BUS]).astype(int)
    t_bus = real(branch[:, T_BUS]).astype(int)

    # batch read
    Sb_f, sf_abs, if_abs = calc_branch_flows_batch(Yf.data, Yf.indptr, Yf.indices, V, baseMVA, Yf.shape[0],
                                                   f_bus, base_kv)
    Sb_t, st_abs, it_abs = calc_branch_flows_batch(Yt.data, Yt.indptr, Yt.indices, V, baseMVA, Yt.shape[0],
                                                   t_bus, base_kv)

    return (Sb_f, Sb_t), (sf_abs, st_abs), (if_abs, it_abs)


def polar_to_rad(vm, va):
    # get complex V matrix (input to batch branch flow function) from vm and va matrices
    nprect = vectorize(rect)
    return nprect(vm, deg2rad(va))
