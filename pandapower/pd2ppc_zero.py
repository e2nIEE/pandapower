# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import math
import numpy as np
from pandapower.auxiliary import _sum_by_group, get_values
from pandapower.pd2ppc import _pd2ppc, _ppc2ppci
from pandapower.idx_brch import BR_B, BR_R, BR_X, F_BUS, T_BUS
from pandapower.idx_bus import BASE_KV, BS, GS
from pandapower.build_branch import _calc_tap_from_dataframe, _transformer_correction_factor
from pandapower.shortcircuit.idx_bus import C_MAX


def _pd2ppc_zero(net):
    ppc_0, ppci_0 = _pd2ppc(net)
    _add_ext_grid_sc_impedance_zero(net, ppc_0)
    _calc_line_sc_impedance_zero(net, ppc_0)
    _add_trafo_sc_impedance_zero(net, ppc_0, trafo_df=None)
    ppci_0 = _ppc2ppci(ppc_0, ppci_0, net)
    return ppc_0, ppci_0


def _add_ext_grid_sc_impedance_zero(net, ppc):
    from pandapower.shortcircuit.idx_bus import C_MAX, C_MIN
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    case = net._options["case"]
    eg = net["ext_grid"][net._is_elements["ext_grid"]]
    if len(eg) == 0:
        return
    eg_buses = eg.bus.values
    eg_buses_ppc = bus_lookup[eg_buses]

    c = ppc["bus"][eg_buses_ppc,
                   C_MAX] if case == "max" else ppc["bus"][eg_buses_ppc, C_MIN]
    if not "s_sc_%s_mva" % case in eg:
        raise ValueError("short circuit apparent power s_sc_%s_mva needs to be specified for " % case +
                         "external grid")
    s_sc = eg["s_sc_%s_mva" % case].values
    if not "rx_%s" % case in eg:
        raise ValueError("short circuit R/X rate rx_%s needs to be specified for external grid" %
                         case)
    rx = eg["rx_%s" % case].values

    z_grid = c / s_sc
    x_grid = z_grid / np.sqrt(rx ** 2 + 1)
    r_grid = rx * x_grid
    eg["r"] = r_grid
    eg["x"] = x_grid

# ext_grid zero sequence impedance
    if case == "max":
        x0_grid = net.ext_grid["x0x_%s" % case] * x_grid
        r0_grid = net.ext_grid["r0x0_%s" % case] * x0_grid
    elif case == "min":
        x0_grid = net.ext_grid["x0x_%s" % case] * x_grid
        r0_grid = net.ext_grid["r0x0_%s" % case] * x0_grid
    y0_grid = 1 / (r0_grid + x0_grid * 1j)
    buses, gs, bs = _sum_by_group(eg_buses_ppc, y0_grid.real, y0_grid.imag)
    ppc["bus"][buses, GS] = gs
    ppc["bus"][buses, BS] = bs


def _calc_line_sc_impedance_zero(net, ppc):
    line = net["line"]
    branch_lookup = net["_pd2ppc_lookups"]["branch"]
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    length = line["length_km"].values
    parallel = line["parallel"].values
    fb = bus_lookup[line["from_bus"].values]
    baseR = np.square(ppc["bus"][fb, BASE_KV]) / net.sn_kva * 1e3

    if "line" in branch_lookup:
        f, t = branch_lookup["line"]
# line zero sequence impedance
        ppc["branch"][f:t, BR_R] = line["r0_ohm_per_km"].values * \
            length / baseR / parallel
        ppc["branch"][f:t, BR_X] = line["x0_ohm_per_km"].values * \
            length / baseR / parallel
        ppc["branch"][f:t, BR_B] = (
            2 * net["f_hz"] * math.pi * line["c0_nf_per_km"].values * 1e-9 * baseR * length * parallel)


def _add_trafo_sc_impedance_zero(net, ppc, trafo_df=None):
    if trafo_df is None:
        trafo_df = net["trafo"]
# checkout the collection index of non-ynyn trafos and ynyn-trafo
    idx_ynyn, idx_no_ynyn = _get_trafo_index(net, ppc, trafo_df)
#    print("ynyn:", idx_ynyn)
#    print("no_ynyn:", idx_no_ynyn)

# delete the non YNyn type trafo rows in ppc["branch"]
    ppc["branch"] = np.delete(ppc["branch"], idx_no_ynyn, 0)
    branch_lookup = net["_pd2ppc_lookups"]["branch"]
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    f, t = branch_lookup["line"]
    for idx in trafo_df.index:
        trafo_idx = trafo_df[trafo_df.index == idx]
#        bus_lookup = net["_pd2ppc_lookups"]["bus"]
#        vn_lv = get_values(ppc["bus"][:, BASE_KV], trafo_idx["lv_bus"].values, bus_lookup)
        vn_trafo_hv, vn_trafo_lv, shift = _calc_tap_from_dataframe(
            net, trafo_idx)
        trafo_buses_lv = trafo_idx["lv_bus"].values
        trafo_buses_hv = trafo_idx["hv_bus"].values
        trafo_type = trafo_idx.loc[idx, "vector_group"]
        vn_lv = ppc["bus"][trafo_buses_lv, BASE_KV]
# kt: transformer correction factor
        cmax = net._ppc["bus"][bus_lookup[int(trafo_idx["lv_bus"])], C_MAX]
        kt = _transformer_correction_factor(
            trafo_idx["vsc_percent"], trafo_idx["vscr_percent"], trafo_idx["sn_kva"], cmax)
# zero seq. transformer impedance without kt
        r0_trafo, x0_trafo = _calc_trafo_sc_impedance_zero(
            net, trafo_idx, vn_lv, vn_trafo_lv, net.sn_kva)
# zero sequence transformer magnetising impedance
        tap_lv = np.square(vn_trafo_lv / vn_lv) * net.sn_kva
        z_m = (trafo_idx.loc[idx, "vsc0_percent"] * trafo_idx.loc[idx,
                                                                  "mag0_percent"]) / 100. / trafo_idx.loc[idx, "sn_kva"] * tap_lv
        x_m = z_m / np.sqrt(trafo_idx.loc[idx, "mag0_rx"]**2 + 1)
        r_m = x_m * trafo_idx.loc[idx, "mag0_rx"]
        parallel = trafo_idx.loc[idx, "parallel"]
        r0_trafo_mag = r_m / parallel
        x0_trafo_mag = x_m / parallel

        if "Dyn" in trafo_type:
            #            r0_dyn, x0_dyn = _calc_trafo_sc_impedance_zero(net, trafo_idx, vn_lv, vn_trafo_lv, net.sn_kva)
            r0_dyn = kt * r0_trafo
            x0_dyn = kt * x0_trafo
            y0_dyn = 1 / (r0_dyn + x0_dyn * 1j)
            trafo_buses_ppc = bus_lookup[trafo_buses_lv]
            buses, gs, bs = _sum_by_group(
                trafo_buses_ppc, y0_dyn.real, y0_dyn.imag)
            ppc["bus"][buses, GS] += gs
            ppc["bus"][buses, BS] += bs

        elif "YNd" in trafo_type:
            #            r0_ynd, x0_ynd = _calc_trafo_sc_impedance_zero(net, trafo_idx, vn_lv, vn_trafo_lv, net.sn_kva)
            r0_ynd = r0_trafo * kt
            x0_ynd = x0_trafo * kt
            y0_ynd = 1 / (r0_ynd + x0_ynd * 1j)
            trafo_buses_ppc = bus_lookup[trafo_buses_hv]
            buses, gs, bs = _sum_by_group(
                trafo_buses_ppc, y0_ynd.real, y0_ynd.imag)
            ppc["bus"][buses, GS] += gs
            ppc["bus"][buses, BS] += bs

        elif "Yyn" in trafo_type:
            #            r0_trafo, x0_trafo = _calc_trafo_sc_impedance_zero(net, trafo_idx, vn_lv, vn_trafo_lv, net.sn_kva)
            #            r0_trafo_mag, x0_trafo_mag = _calc_trafo_mag_impedance_zero(net, trafo_idx, vn_lv, vn_trafo_lv, net.sn_kva)
            r0_yyn = r0_trafo * kt + r0_trafo_mag
            x0_yyn = x0_trafo * kt + x0_trafo_mag
            y0_yyn = 1 / (r0_yyn + x0_yyn * 1j)
            trafo_buses_ppc = bus_lookup[trafo_buses_lv]
            buses, gs, bs = _sum_by_group(
                trafo_buses_ppc, y0_yyn.real, y0_yyn.imag)
            ppc["bus"][buses, GS] += gs
            ppc["bus"][buses, BS] += bs

        elif "YNy" in trafo_type and "YNyn" not in trafo_type:
            #            r0_trafo, x0_trafo = _calc_trafo_sc_impedance_zero(net, trafo_idx, vn_lv, vn_trafo_lv, net.sn_kva)
            #            r0_trafo_mag, x0_trafo_mag = _calc_trafo_mag_impedance_zero(net, trafo_idx, vn_lv, vn_trafo_lv, net.sn_kva)
            r0_yny = r0_trafo * kt + r0_trafo_mag
            x0_yny = x0_trafo * kt + x0_trafo_mag
            y0_yny = 1 / (r0_yny + x0_yny * 1j)
            trafo_buses_ppc = bus_lookup[trafo_buses_hv]
            buses, gs, bs = _sum_by_group(
                trafo_buses_ppc, y0_yny.real, y0_yny.imag)
            ppc["bus"][buses, GS] += gs
            ppc["bus"][buses, BS] += bs

        elif "YNyn" in trafo_type:
            #            r0_ynyn, x0_ynyn = _calc_trafo_sc_impedance_zero(net, trafo_idx, vn_lv, vn_trafo_lv, net.sn_kva)
            r0_ynyn = kt * r0_trafo
            x0_ynyn = kt * x0_trafo
            tap_lv = np.square(vn_trafo_lv / vn_lv) * net.sn_kva
            z_m = (trafo_idx.loc[idx, "vsc0_percent"] * trafo_idx.loc[idx,
                                                                      "mag0_percent"]) / 100. / trafo_idx.loc[idx, "sn_kva"] * tap_lv
            x_m = z_m / np.sqrt(trafo_idx.loc[idx, "mag0_rx"]**2 + 1)
            r_m = x_m * trafo_idx.loc[idx, "mag0_rx"]
            parallel = trafo_idx.loc[idx, "parallel"]
            r0_trafo_mag = r_m / parallel
            x0_trafo_mag = x_m / parallel
# convert the t model to pi model

            z1 = trafo_idx.loc[idx, "si0_hv_partial"] * \
                (r0_ynyn + x0_ynyn * 1j)
            z2 = (1 - trafo_idx.loc[idx, "si0_hv_partial"]
                  ) * (r0_ynyn + x0_ynyn * 1j)
            z3 = r0_trafo_mag + x0_trafo_mag * 1j

            z_temp = z1 * z2 + z2 * z3 + z1 * z3
            za = z_temp / z2
            zb = z_temp / z1
            zc = z_temp / z3

            if trafo_idx.loc[idx, "si0_hv_partial"] == 0.5:
                ppc["branch"][t:t + 1, BR_R] = zc.real
                ppc["branch"][t:t + 1, BR_X] = zc.imag
                ppc["branch"][t:t + 1,
                              BR_B] = (2 / za).imag - (2 / za).real * 1j
                t += 1
            else:
                # add a shunt element parallel to zb if the leakage impedance distribution is unequal
                zs = (za * zb) / (za - zb)
                ys = 1 / zs

                ppc["branch"][t:t + 1, BR_R] = zc.real
                ppc["branch"][t:t + 1, BR_X] = zc.imag
                ppc["branch"][t:t + 1,
                              BR_B] = (2 / za).imag - (2 / za).real * 1j
                t += 1

                trafo_buses_ppc = bus_lookup[trafo_buses_lv]
                buses, gs, bs = _sum_by_group(
                    trafo_buses_ppc, ys.real, ys.imag)
                ppc["bus"][buses, GS] += gs
                ppc["bus"][buses, BS] += bs

        elif "Yy" in trafo_type and "Yyn" not in trafo_type:
            pass

        elif "Dy" in trafo_type and "Dyn" not in trafo_type:
            pass

        elif "Yd" in trafo_type:
            pass

        elif "Dd" in trafo_type:
            pass


# def _get_trafo_index(net, ppc, trafo_df):
#     """
#         idx_no_ynyn:find out the non YNyn type trafo index in ppc["branch"]
#         idx_ynyn: YNyn type trafo index
#     """
#     if trafo_df is None:
#         trafo_df = net["trafo"]
#     branch_lookup = net._pd2ppc_lookups["branch"]
#     f, t = branch_lookup["trafo"]
#     idx_no_ynyn = []
#     idx_ynyn = []
#     for idx in trafo_df.index:
#         trafo_idx = trafo_df[trafo_df.index == idx]
#         if "YNyn" in trafo_idx.loc[idx, "vector_group"]:
#             for i in list(range(f, t)):
#                 if trafo_idx["hv_bus"].values == ppc["branch"][i, F_BUS] and trafo_idx["lv_bus"].values == ppc["branch"][i, T_BUS]:
#                     idx_ynyn.append(i)
#         else:
#             for n in list(range(f, t)):
#                 if trafo_idx["hv_bus"].values == ppc["branch"][n, F_BUS] and trafo_idx["lv_bus"].values == ppc["branch"][n, T_BUS]:
#                     idx_no_ynyn.append(n)
#     return idx_ynyn, idx_no_ynyn

def _get_trafo_index(net, ppc, trafo_df):
    """
        idx_no_ynyn:find out the non YNyn type trafo index in ppc["branch"]
        idx_ynyn: YNyn type trafo index
    """
    if trafo_df is None:
        trafo_df = net["trafo"]
    branch_lookup = net._pd2ppc_lookups["branch"]
    bus_lookup = net._pd2ppc_lookups["bus"]
    f, t = branch_lookup["trafo"]
    idx_no_ynyn = []
    idx_ynyn = []
    for idx in trafo_df.index:
        trafo_idx = trafo_df[trafo_df.index == idx]
        if "YNyn" in trafo_idx.loc[idx, "vector_group"]:
            for i in list(range(f, t)):
                if bus_lookup[trafo_idx["hv_bus"].values] == ppc["branch"][i, F_BUS] and bus_lookup[trafo_idx["lv_bus"].values] == ppc["branch"][i, T_BUS]:
                    idx_ynyn.append(i)
        else:
            for n in list(range(f, t)):
                if bus_lookup[trafo_idx["hv_bus"].values] == ppc["branch"][n, F_BUS] and bus_lookup[trafo_idx["lv_bus"].values] == ppc["branch"][n, T_BUS]:
                    idx_no_ynyn.append(n)
    return idx_ynyn, idx_no_ynyn


def _calc_trafo_sc_impedance_zero(net, trafo_df, vn_lv, vn_trafo_lv, sn_kva):
    """
        calculates the zero sequence trafo impedance with correction factor
    """
    tap_lv = np.square(vn_trafo_lv / vn_lv) * \
        sn_kva  # adjust for low voltage side voltage converter
    sn_trafo_kva = trafo_df.sn_kva.values
    parallel = trafo_df["parallel"].values
#    mode = net["_options"]["mode"]
    z_sc = trafo_df["vsc0_percent"].values / 100. / sn_trafo_kva * tap_lv
    r_sc = trafo_df["vscr0_percent"].values / 100. / sn_trafo_kva * tap_lv
    z_sc = z_sc.astype(float)
    r_sc = r_sc.astype(float)
#    z_sc = np.float(z_sc)
#    r_sc = np.float(r_sc)
    x_sc = np.sign(z_sc) * np.sqrt(z_sc**2 - r_sc**2)
    return r_sc / parallel, x_sc / parallel

#    if mode == "sc":
#        if trafo_df.equals(net.trafo):
#            from pandapower.shortcircuit.idx_bus import C_MAX
#            bus_lookup = net._pd2ppc_lookups["bus"]
#            cmax = net._ppc["bus"][bus_lookup[net.trafo.lv_bus.values], C_MAX]
#            print("cmax=",cmax)
#            kt = _transformer_correction_factor(trafo_df["vsc_percent"], trafo_df["vscr_percent"], trafo_df["sn_kva"], cmax)
#            print("kt=",kt)
#            r_sc *= kt
#            x_sc *= kt
#    return r_sc / parallel, x_sc / parallel

# def _calc_trafo_mag_impedance_zero(net, trafo_df, vn_lv, vn_trafo_lv, sn_kva):
#    """
#        calculates the zero sequence trafo magnetising impedance
#    """
#    tap_lv = np.square(vn_trafo_lv / vn_lv) * sn_kva  # adjust for low voltage side voltage converter
#    sn_trafo_kva = trafo_df.sn_kva.values
#    parallel = trafo_df["parallel"].values
#    z_m = (trafo_df["vsc0_percent"].values * trafo_df["mag0_percent"].values) / 100. / sn_trafo_kva * tap_lv
#    z_m = np.float(z_m)
#    x_m = z_m / np.sqrt(trafo_df["mag0_rx"].astype(float)**2 + 1)
#    r_m = x_m * trafo_df["mag0_rx"]
#
#    return r_m / parallel, x_m / parallel
