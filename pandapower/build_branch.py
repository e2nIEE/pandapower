# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import copy
import math
from functools import partial

import numpy as np
import pandas as pd
from pandapower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT, BR_STATUS, RATE_A, \
                                BR_R_ASYM, BR_X_ASYM, branch_cols
from pandapower.idx_bus import BASE_KV, VM, VA, bus_cols
from pandapower.auxiliary import get_values


def _build_branch_ppc(net, ppc):
    """
    Takes the empty ppc network and fills it with the branch values. The branch
    datatype will be np.complex 128 afterwards.

    .. note:: The order of branches in the ppc is:
            1. Lines
            2. Transformers
            3. 3W Transformers (each 3W Transformer takes up three branches)
            4. Impedances
            5. Internal branch for extended ward

    **INPUT**:
        **net** -The pandapower format network

        **ppc** - The PYPOWER format network to fill in values

    """
    length = _initialize_branch_lookup(net)
    lookup = net._pd2ppc_lookups["branch"]
    mode = net._options["mode"]
    ppc["branch"] = np.zeros(shape=(length, branch_cols), dtype=np.complex128)
    if mode == "sc":
        from pandapower.shortcircuit.idx_brch import branch_cols_sc
        branch_sc = np.empty(shape=(length, branch_cols_sc), dtype=float)
        branch_sc.fill(np.nan)
        ppc["branch"] = np.hstack((ppc["branch"], branch_sc ))
    ppc["branch"][:, :13] = np.array([0, 0, 0, 0, 0, 250, 250, 250, 1, 0, 1, -360, 360])
    if "line" in lookup:
        f, t = lookup["line"]
        ppc["branch"][f:t, [F_BUS, T_BUS, BR_R, BR_X, BR_B,
                            BR_STATUS, RATE_A]] = _calc_line_parameter(net, ppc)
    if "trafo" in lookup:
        f, t = lookup["trafo"]
        ppc["branch"][f:t, [F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT, BR_STATUS,
                            RATE_A]] = _calc_trafo_parameter(net, ppc)
    if "trafo3w" in lookup:
        f, t = lookup["trafo3w"]
        ppc["branch"][f:t, [F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT, BR_STATUS, RATE_A]] = \
            _calc_trafo3w_parameter(net, ppc)
    if "impedance" in lookup:
        f, t = lookup["impedance"]
        ppc["branch"][f:t, [F_BUS, T_BUS, BR_R, BR_X, BR_R_ASYM, BR_X_ASYM, BR_STATUS]] = \
            _calc_impedance_parameter(net)
    if "xward" in lookup:
        f, t = lookup["xward"]
        ppc["branch"][f:t, [F_BUS, T_BUS, BR_R, BR_X, BR_STATUS]] = _calc_xward_parameter(net, ppc)

    if "switch" in lookup:
        f, t = lookup["switch"]
        ppc["branch"][f:t, [F_BUS, T_BUS, BR_R]] = _calc_switch_parameter(net, ppc)


def _initialize_branch_lookup(net):
    r_switch = net["_options"]["r_switch"]
    start = 0
    net._pd2ppc_lookups["branch"] = {}
    for element in ["line", "trafo", "trafo3w", "impedance", "xward"]:
        if len(net[element]) > 0:
            if element == "trafo3w":
                end = start + len(net[element]) * 3
            else:
                end = start + len(net[element])
            net._pd2ppc_lookups["branch"][element] = (start, end)
            start = end
    if r_switch > 0 and len(net._closed_bb_switches) > 0:
        end = start + net._closed_bb_switches.sum()
        net._pd2ppc_lookups["branch"]["switch"] = (start, end)
    return end


def _calc_trafo3w_parameter(net, ppc):
    copy_constraints_to_ppc = net["_options"]["copy_constraints_to_ppc"]
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    trafo_df = _trafo_df_from_trafo3w(net)
    net._equiv_trafo3w = trafo_df

    temp_para = np.zeros(shape=(len(trafo_df), 9), dtype=np.complex128)
    temp_para[:, 0] = bus_lookup[(trafo_df["hv_bus"].values).astype(int)]
    temp_para[:, 1] = bus_lookup[(trafo_df["lv_bus"].values).astype(int)]
    temp_para[:, 2:7] = _calc_branch_values_from_trafo_df(net, ppc, trafo_df)
    temp_para[:, 7] = trafo_df["in_service"].values
    if copy_constraints_to_ppc:
        max_load = trafo_df.max_loading_percent if "max_loading_percent" in trafo_df else 0
        temp_para[:, 8] = max_load / 100. * trafo_df.sn_kva / 1000.
    return temp_para


def _calc_line_parameter(net, ppc):
    """
    calculates the line parameter in per unit.

    **INPUT**:
        **net** -The pandapower format network

    **RETURN**:
        **t** - Temporary line parameter. Which is a complex128
                Nunmpy array. with the following order:
                0:bus_a; 1:bus_b; 2:r_pu; 3:x_pu; 4:b_pu
    """
    copy_constraints_to_ppc = net["_options"]["copy_constraints_to_ppc"]
    mode = net["_options"]["mode"]
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    line = net["line"]
    fb = bus_lookup[line["from_bus"].values]
    tb = bus_lookup[line["to_bus"].values]
    length = line["length_km"].values
    parallel = line["parallel"].values
    baseR = np.square(ppc["bus"][fb, BASE_KV]) / net.sn_kva * 1e3
    t = np.zeros(shape=(len(line.index), 7), dtype=np.complex128)

    t[:, 0] = fb
    t[:, 1] = tb

    t[:, 2] = line["r_ohm_per_km"].values * length / baseR / parallel
    t[:, 3] = line["x_ohm_per_km"].values * length / baseR / parallel
    if mode == "sc":
        if net["_options"]["case"] == "min":
            t[:, 2] *= _end_temperature_correction_factor(net)
    else:
        t[:, 4] = (2 * net.f_hz * math.pi * line["c_nf_per_km"].values * 1e-9 * baseR *
                   length * parallel)
    t[:, 5] = line["in_service"].values
    if copy_constraints_to_ppc:
        max_load = line.max_loading_percent.values if "max_loading_percent" in line else 0
        vr = net.bus.vn_kv.loc[line["from_bus"].values].values * np.sqrt(3)
        t[:, 6] = max_load / 100. * line.max_i_ka.values * line.df.values * parallel * vr
    return t


def _calc_trafo_parameter(net, ppc):
    '''
    Calculates the transformer parameter in per unit.

    **INPUT**:
        **net** - The pandapower format network

    **RETURN**:
        **temp_para** -
        Temporary transformer parameter. Which is a np.complex128
        Numpy array. with the following order:
        0:hv_bus; 1:lv_bus; 2:r_pu; 3:x_pu; 4:b_pu; 5:tab, 6:shift
    '''
    copy_constraints_to_ppc = net["_options"]["copy_constraints_to_ppc"]

    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    temp_para = np.zeros(shape=(len(net["trafo"].index), 9), dtype=np.complex128)
    trafo = net["trafo"]
    parallel = trafo["parallel"].values
    temp_para[:, 0] = bus_lookup[trafo["hv_bus"].values]
    temp_para[:, 1] = bus_lookup[trafo["lv_bus"].values]
    temp_para[:, 2:7] = _calc_branch_values_from_trafo_df(net, ppc)
    temp_para[:, 7] = trafo["in_service"].values
    if copy_constraints_to_ppc:
        max_load = trafo.max_loading_percent.values if "max_loading_percent" in trafo else 0
        temp_para[:, 8] = max_load / 100. * trafo.sn_kva.values / 1000. * parallel
    return temp_para


def _calc_branch_values_from_trafo_df(net, ppc, trafo_df=None):
    """
    Calculates the MAT/PYPOWER-branch-attributes from the pandapower trafo dataframe.

    PYPOWER and MATPOWER uses the PI-model to model transformers.
    This function calculates the resistance r, reactance x, complex susceptance c and the tap ratio
    according to the given parameters.

    .. warning:: This function returns the subsceptance b as a complex number
        **(-img + -re*i)**. MAT/PYPOWER is only intended to calculate the
        imaginary part of the subceptance. However, internally c is
        multiplied by i. By using subsceptance in this way, it is possible
        to consider the ferromagnetic loss of the coil. Which would
        otherwise be neglected.


    .. warning:: Tab switches effect calculation as following:
        On **high-voltage** side(=1) -> only **tab** gets adapted.
        On **low-voltage** side(=2) -> **tab, x, r** get adapted.
        This is consistent with Sincal.
        The Sincal method in this case is questionable.


    **INPUT**:
        **pd_trafo** - The pandapower format Transformer Dataframe.
                        The Transformer modell will only readfrom pd_net

    **RETURN**:
        **temp_para** - Temporary transformer parameter. Which is a complex128
                        Nunmpy array. with the following order:
                        0:r_pu; 1:x_pu; 2:b_pu; 3:tab;

    """
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    if trafo_df is None:
        trafo_df = net["trafo"]
    parallel = trafo_df["parallel"].values
    vn_lv = get_values(ppc["bus"][:, BASE_KV], trafo_df["lv_bus"].values, bus_lookup)
    ### Construct np.array to parse results in ###
    # 0:r_pu; 1:x_pu; 2:b_pu; 3:tab;
    temp_para = np.zeros(shape=(len(trafo_df), 5), dtype=np.complex128)
    vn_trafo_hv, vn_trafo_lv, shift = _calc_tap_from_dataframe(net, trafo_df, vn_lv)
    ratio = _calc_nominal_ratio_from_dataframe(ppc, trafo_df, vn_trafo_hv, vn_trafo_lv,
                                               bus_lookup)
    r, x, y = _calc_r_x_y_from_dataframe(net, trafo_df, vn_trafo_lv, vn_lv, net.sn_kva)
    temp_para[:, 0] = r / parallel
    temp_para[:, 1] = x / parallel
    temp_para[:, 2] = y * parallel
    temp_para[:, 3] = ratio
    temp_para[:, 4] = shift
    return temp_para


def _calc_r_x_y_from_dataframe(net, trafo_df, vn_trafo_lv, vn_lv, sn_kva):
    mode = net["_options"]["mode"]
    trafo_model = net["_options"]["trafo_model"]

    r, x = _calc_r_x_from_dataframe(trafo_df, vn_lv, vn_trafo_lv, sn_kva)
    if mode == "sc":
        y = 0
        if trafo_df.equals(net.trafo):
            from pandapower.shortcircuit.idx_bus import C_MAX
            bus_lookup = net._pd2ppc_lookups["bus"]
            cmax = net._ppc["bus"][bus_lookup[net.trafo.lv_bus.values], C_MAX]
            kt = _transformer_correction_factor(trafo_df.vsc_percent, trafo_df.vscr_percent,
                                                trafo_df.sn_kva, cmax)
            r *= kt
            x *= kt
    else:
        y = _calc_y_from_dataframe(trafo_df, vn_lv, vn_trafo_lv, sn_kva)
    if trafo_model == "pi":
        return r, x, y
    elif trafo_model == "t":
        return _wye_delta(r, x, y)
    else:
        raise ValueError("Unkonwn Transformer Model %s - valid values ar 'pi' or 't'" % trafo_model)


def _wye_delta(r, x, y):
    """
    20.05.2016 added by Lothar Löwer

    Calculate transformer Pi-Data based on T-Data

    """
    tidx = np.where(y != 0)
    za_star = (r[tidx] + x[tidx] * 1j) / 2
    zc_star = -1j / y[tidx]
    zSum_triangle = za_star * za_star + 2 * za_star * zc_star
    zab_triangle = zSum_triangle / zc_star
    zbc_triangle = zSum_triangle / za_star
    r[tidx] = zab_triangle.real
    x[tidx] = zab_triangle.imag
    y[tidx] = -2j / zbc_triangle
    return r, x, y


def _calc_y_from_dataframe(trafo_df, vn_lv, vn_trafo_lv, sn_kva):
    """
    Calculate the subsceptance y from the transformer dataframe.

    INPUT:

        **trafo** (Dataframe) - The dataframe in net.trafo
        which contains transformer calculation values.

    OUTPUT:
        **subsceptance** (1d array, np.complex128) - The subsceptance in pu in
        the form (-b_img, -b_real)
    """
    baseR = np.square(vn_lv) / sn_kva * 1e3

    ### Calculate subsceptance ###
    vnl_squared = trafo_df["vn_lv_kv"].values ** 2
    b_real = trafo_df["pfe_kw"].values / (1000. * vnl_squared) * baseR
    i0 = trafo_df["i0_percent"].values
    pfe = trafo_df["pfe_kw"].values
    sn = trafo_df["sn_kva"].values
    b_img = (i0 / 100. * sn / 1000.) ** 2 - (pfe / 1000.) ** 2

    b_img[b_img < 0] = 0
    b_img = np.sqrt(b_img) * baseR / vnl_squared
    y = - b_real * 1j - b_img * np.sign(i0)
    if "lv" in trafo_df["tp_side"].values:
        return y / np.square(vn_trafo_lv * vn_lv / trafo_df["vn_lv_kv"].values / vn_lv)
    else:
        return y


def _calc_tap_from_dataframe(net, trafo_df, vn_lv):
    """
    Adjust the nominal voltage vnh and vnl to the active tab position "tp_pos".
    If "side" is 1 (high-voltage side) the high voltage vnh is adjusted.
    If "side" is 2 (low-voltage side) the low voltage vnl is adjusted

    INPUT:
        **trafo** (Dataframe) - The dataframe in pd_net["structure"]["trafo"]
        which contains transformer calculation values.

    OUTPUT:
        **vn_hv_kv** (1d array, float) - The adusted high voltages

        **vn_lv_kv** (1d array, float) - The adjusted low voltages

    """
    calculate_voltage_angles = net["_options"]["calculate_voltage_angles"]
    mode = net["_options"]["mode"]
    # Changing Voltage on high-voltage side
    trafo_shift = trafo_df["shift_degree"].values.astype(float) if calculate_voltage_angles else \
        np.zeros(len(trafo_df))
    vnh = copy.copy(trafo_df["vn_hv_kv"].values.astype(float))
    vnl = copy.copy(trafo_df["vn_lv_kv"].values.astype(float))
    if mode == "sc":
        return vnh, vnl, trafo_shift

    tp_diff = trafo_df["tp_pos"].values - trafo_df["tp_mid"].values

    tap_os = np.isfinite(trafo_df["tp_pos"].values) & (trafo_df["tp_side"].values == "hv")
    if any(tap_os):
        os_steps = trafo_df["tp_st_percent"].values[tap_os]
        vnh[tap_os] *= np.ones((tap_os.sum()), dtype=np.float) + tp_diff[tap_os] * os_steps / 100.
        if calculate_voltage_angles and "tp_st_degree" in trafo_df:
            ps_os = tap_os & np.isfinite(trafo_df["tp_st_degree"].values)
            trafo_shift[ps_os] += tp_diff[ps_os] * trafo_df["tp_st_degree"].values[ps_os]

    # Changing Voltage on low-voltage side
    tap_us = np.isfinite(trafo_df["tp_pos"].values) & (trafo_df["tp_side"].values == "lv")
    if any(tap_us):
        us_steps = trafo_df["tp_st_percent"].values[tap_us]
        vnl[tap_us] *= np.ones((tap_us.sum()), dtype=np.float) + tp_diff[tap_us] * us_steps / 100.
        if calculate_voltage_angles and "tp_st_degree" in trafo_df:
            ps_us = tap_us & np.isfinite(trafo_df["tp_st_degree"].values)
            trafo_shift[ps_us] -= tp_diff[ps_us] * trafo_df["tp_st_degree"].values[ps_us]
    return vnh, vnl, trafo_shift


def _calc_r_x_from_dataframe(trafo_df, vn_lv, vn_trafo_lv, sn_kva):
    """
    Calculates (Vectorized) the resitance and reactance according to the
    transformer values

    """
    tap_lv = np.square(vn_trafo_lv / vn_lv) * sn_kva  # adjust for low voltage side voltage converter
    sn_trafo_kva = trafo_df.sn_kva.values
    z_sc = trafo_df["vsc_percent"].values / 100. / sn_trafo_kva * tap_lv
    r_sc = trafo_df["vscr_percent"].values / 100. / sn_trafo_kva * tap_lv
    x_sc = np.sign(z_sc) * np.sqrt(z_sc ** 2 - r_sc ** 2)
    return r_sc, x_sc


def _calc_nominal_ratio_from_dataframe(ppc, trafo_df, vn_hv_kv, vn_lv_kv, bus_lookup):
    """
    Calculates (Vectorized) the off nominal tap ratio::

                  (vn_hv_kv / vn_lv_kv) / (ub1_in_kv / ub2_in_kv)

    INPUT:
        **net** (Dataframe) - The net for which to calc the tap ratio.

        **vn_hv_kv** (1d array, float) - The adjusted nominal high voltages

        **vn_lv_kv** (1d array, float) - The adjusted nominal low voltages

    OUTPUT:
        **tab** (1d array, float) - The off-nominal tap ratio
    """
    # Calculating tab (trasformer off nominal turns ratio)
    tap_rat = vn_hv_kv / vn_lv_kv
    nom_rat = get_values(ppc["bus"][:, BASE_KV], trafo_df["hv_bus"].values, bus_lookup) / \
              get_values(ppc["bus"][:, BASE_KV], trafo_df["lv_bus"].values, bus_lookup)
    return tap_rat / nom_rat


def z_br_to_bus(z, s):
    return s[0] * np.array([z[0] / min(s[0], s[1]), z[1] /
                            min(s[1], s[2]), z[2] / min(s[0], s[2])])


def wye_delta(zbr_n, s):
    return .5 * s / s[0] * np.array([(zbr_n[0] + zbr_n[2] - zbr_n[1]),
                                     (zbr_n[1] + zbr_n[0] - zbr_n[2]),
                                     (zbr_n[2] + zbr_n[1] - zbr_n[0])])

def _trafo_df_from_trafo3w(net):
    mode = net._options["mode"]
    trafos2w = {}
    nr_trafos = len(net["trafo3w"])
    tap_variables = ("tp_pos", "tp_mid", "tp_max", "tp_min", "tp_st_percent")
    i = 0
    for _, ttab in net["trafo3w"].iterrows():
        vsc = np.array([ttab.vsc_hv_percent, ttab.vsc_mv_percent, ttab.vsc_lv_percent], dtype=float)
        vscr = np.array([ttab.vscr_hv_percent, ttab.vscr_mv_percent, ttab.vscr_lv_percent], dtype=float)
        sn = np.array([ttab.sn_hv_kva, ttab.sn_mv_kva, ttab.sn_lv_kva])
        vsc_2w_delta = z_br_to_bus(vsc, sn)
        vscr_2w_delta = z_br_to_bus(vscr, sn)
        if mode == "sc":
            kt = _transformer_correction_factor(vsc, vscr, sn, 1.1)
            vsc_2w_delta *= kt
            vscr_2w_delta *= kt
        vsc_2w = wye_delta(vsc_2w_delta, sn)
        vscr_2w = wye_delta(vscr_2w_delta, sn)
        taps = [dict((tv, np.nan) for tv in tap_variables) for _ in range(3)]
        for k in range(3):
            taps[k]["tp_side"] = None

        if pd.notnull(ttab.tp_side):
            if ttab.tp_side == "hv" or ttab.tp_side == 0:
                tp_trafo = 0
            elif ttab.tp_side == "mv":
                tp_trafo = 1
            elif ttab.tp_side == "lv":
                tp_trafo = 3
            for tv in tap_variables:
                taps[tp_trafo][tv] = ttab[tv]
            taps[tp_trafo]["tp_side"] = "hv" if tp_trafo == 0 else "lv"

        max_load = ttab.max_loading_percent if "max_loading_percent" in ttab else 0

        trafos2w[i] = {"hv_bus": ttab.hv_bus, "lv_bus": ttab.ad_bus, "sn_kva": ttab.sn_hv_kva,
                       "vn_hv_kv": ttab.vn_hv_kv, "vn_lv_kv": ttab.vn_hv_kv, "vscr_percent": vscr_2w[0],
                       "vsc_percent": vsc_2w[0], "pfe_kw": ttab.pfe_kw,
                       "i0_percent": ttab.i0_percent, "tp_side": taps[0]["tp_side"],
                       "tp_mid": taps[0]["tp_mid"], "tp_max": taps[0]["tp_max"],
                       "tp_min": taps[0]["tp_min"], "tp_pos": taps[0]["tp_pos"],
                       "tp_st_percent": taps[0]["tp_st_percent"], "parallel": 1,
                       "in_service": ttab.in_service, "shift_degree": 0, "max_loading_percent": max_load}
        trafos2w[i + nr_trafos] = {"hv_bus": ttab.ad_bus, "lv_bus": ttab.mv_bus,
                                   "sn_kva": ttab.sn_mv_kva, "vn_hv_kv": ttab.vn_hv_kv, "vn_lv_kv": ttab.vn_mv_kv,
                                   "vscr_percent": vscr_2w[1], "vsc_percent": vsc_2w[1], "pfe_kw": 0,
                                   "i0_percent": 0, "tp_side": taps[1]["tp_side"],
                                   "tp_mid": taps[1]["tp_mid"], "tp_max": taps[1]["tp_max"],
                                   "tp_min": taps[1]["tp_min"], "tp_pos": taps[1]["tp_pos"],
                                   "tp_st_percent": taps[1]["tp_st_percent"], "parallel": 1,
                                   "in_service": ttab.in_service, "shift_degree": ttab.shift_mv_degree,
                                   "max_loading_percent": max_load}
        trafos2w[i + 2 * nr_trafos] = {"hv_bus": ttab.ad_bus, "lv_bus": ttab.lv_bus,
                                       "sn_kva": ttab.sn_lv_kva,
                                       "vn_hv_kv": ttab.vn_hv_kv, "vn_lv_kv": ttab.vn_lv_kv, "vscr_percent": vscr_2w[2],
                                       "vsc_percent": vsc_2w[2], "pfe_kw": 0, "i0_percent": 0,
                                       "tp_side": taps[2]["tp_side"], "tp_mid": taps[2]["tp_mid"],
                                       "tp_max": taps[2]["tp_max"], "tp_min": taps[2]["tp_min"],
                                       "tp_pos": taps[2]["tp_pos"], "tp_st_percent": taps[2]["tp_st_percent"],
                                       "parallel": 1,
                                       "in_service": ttab.in_service, "shift_degree": ttab.shift_lv_degree,
                                       "max_loading_percent": max_load}
        i += 1
    trafo_df = pd.DataFrame(trafos2w).T
    for var in list(tap_variables) + ["i0_percent", "sn_kva", "vsc_percent", "vscr_percent",
                                      "vn_hv_kv", "vn_lv_kv", "pfe_kw", "max_loading_percent"]:
        trafo_df[var] = pd.to_numeric(trafo_df[var])
    return trafo_df


def _calc_impedance_parameter(net):
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    t = np.zeros(shape=(len(net["impedance"].index), 7), dtype=np.complex128)
    sn_impedance = net["impedance"]["sn_kva"].values
    sn_net = net.sn_kva
    rij = net["impedance"]["rft_pu"].values
    xij = net["impedance"]["xft_pu"].values
    rji = net["impedance"]["rtf_pu"].values
    xji = net["impedance"]["xtf_pu"].values
    t[:, 0] = bus_lookup[net["impedance"]["from_bus"].values]
    t[:, 1] = bus_lookup[net["impedance"]["to_bus"].values]
    t[:, 2] = rij / sn_impedance * sn_net
    t[:, 3] = xij / sn_impedance * sn_net
    t[:, 4] = (rji - rij) / sn_impedance * sn_net
    t[:, 5] = (xji - xij) / sn_impedance * sn_net
    t[:, 6] = net["impedance"]["in_service"].values
    return t


def _calc_xward_parameter(net, ppc):
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    baseR = np.square(get_values(ppc["bus"][:, BASE_KV], net["xward"]["bus"].values, bus_lookup)) / \
            net.sn_kva * 1e3
    t = np.zeros(shape=(len(net["xward"].index), 5), dtype=np.complex128)
    xw_is = net["_is_elements"]["xward"]
    t[:, 0] = bus_lookup[net["xward"]["bus"].values]
    t[:, 1] = bus_lookup[net["xward"]["ad_bus"].values]
    t[:, 2] = net["xward"]["r_ohm"] / baseR
    t[:, 3] = net["xward"]["x_ohm"] / baseR
    t[:, 4] = xw_is
    return t


def _gather_branch_switch_info(bus, branch_id, branch_type, net):
    # determine at which end the switch is located
    # 1 = to-bus/lv-bus; 0 = from-bus/hv-bus

    if branch_type == "l":
        branch_bus = net["line"]["to_bus"].at[branch_id]
        is_to_bus = int(branch_bus == bus)
        return is_to_bus, bus, net["line"].index.get_loc(branch_id)
    else:
        branch_bus = net["trafo"]["lv_bus"].at[branch_id]
        is_to_bus = int(branch_bus == bus)
        return is_to_bus, bus, net["trafo"].index.get_loc(branch_id)


def _switch_branches(net, ppc):
    from pandapower.shortcircuit.idx_bus import C_MIN, C_MAX
    """
    Updates the ppc["branch"] matrix with the changed from or to values
    according of the status of switches

    **INPUT**:
        **pd_net** - The pandapower format network

        **ppc** - The PYPOWER format network to fill in values
    """
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    connectivity_check = net["_options"]["check_connectivity"]
    mode = net._options["mode"]
    # get in service elements
    _is_elements = net["_is_elements"]
    lines_is = _is_elements['line']
    bus_is_idx = _is_elements['bus_is_idx']

    # opened bus line switches
    slidx = (net["switch"]["closed"].values == 0) \
            & (net["switch"]["et"].values == "l")

    # check if there are multiple opened switches at a line (-> set line out of service)
    sw_elem = net['switch'][slidx]["element"].values
    m = np.zeros_like(sw_elem, dtype=bool)
    m[np.unique(sw_elem, return_index=True)[1]] = True

    # if non unique elements are in sw_elem (= multiple opened bus line switches)
    if np.count_nonzero(m) < len(sw_elem):
        from_bus = lines_is.ix[sw_elem[~m]].from_bus.values
        to_bus = lines_is.ix[sw_elem[~m]].to_bus.values
        # check if branch is already out of service -> ignore switch
        from_bus = from_bus[~np.isnan(from_bus)].astype(int)
        to_bus = to_bus[~np.isnan(to_bus)].astype(int)

        # set branch in ppc out of service if from and to bus are at a line which is in service
        if not connectivity_check and from_bus.size and to_bus.size:
            # get from and to buses of these branches
            ppc_from = bus_lookup[from_bus]
            ppc_to = bus_lookup[to_bus]
            ppc_idx = np.in1d(ppc['branch'][:, 0], ppc_from) \
                      & np.in1d(ppc['branch'][:, 1], ppc_to)
            ppc["branch"][ppc_idx, BR_STATUS] = 0

            # drop from in service lines as well
            lines_is = lines_is.drop(sw_elem[~m])

    # opened switches at in service lines
    slidx = slidx \
            & (np.in1d(net["switch"]["element"].values, lines_is.index)) \
            & (np.in1d(net["switch"]["bus"].values, bus_is_idx))
    nlo = np.count_nonzero(slidx)

    stidx = (net.switch["closed"].values == 0) & (net.switch["et"].values == "t")
    nto = np.count_nonzero(stidx)

    if (nlo + nto) > 0:
        n_bus = len(ppc["bus"])

        if nlo:
            future_buses = [ppc["bus"]]
            line_switches = net["switch"].loc[slidx]

            # determine on which side the switch is located
            mapfunc = partial(_gather_branch_switch_info, branch_type="l", net=net)
            ls_info = list(map(mapfunc,
                               line_switches["bus"].values,
                               line_switches["element"].values))
            # we now have the following matrix
            # 0: 1 if switch is at to_bus, 0 else
            # 1: bus of the switch
            # 2: position of the line a switch is connected to
            ls_info = np.array(ls_info, dtype=int)

            # build new buses
            new_ls_buses = np.zeros(shape=(nlo, ppc["bus"].shape[1]), dtype=float)
            new_indices = np.arange(n_bus, n_bus + nlo)
            # the newly created buses
            new_ls_buses[:, :15] = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1.1, 0.9, 0, 0])
            new_ls_buses[:, 0] = new_indices
            new_ls_buses[:, BASE_KV] = get_values(ppc["bus"][:, BASE_KV], ls_info[:, 1], bus_lookup)
            #             set voltage of new buses to voltage on other branch end
            to_buses = ppc["branch"][ls_info[ls_info[:, 0].astype(bool), 2], 1].real.astype(int)
            from_buses = ppc["branch"][ls_info[np.logical_not(ls_info[:, 0]), 2], 0].real \
                .astype(int)

            if len(to_buses):
                ix = ls_info[:, 0] == 1
                new_ls_buses[ix, VM] = ppc["bus"][to_buses, VM]
                new_ls_buses[ix, VA] = ppc["bus"][to_buses, VA]
                if mode == "sc":
                    new_ls_buses[ix, C_MAX] = ppc["bus"][to_buses, C_MAX]
                    new_ls_buses[ix, C_MIN] = ppc["bus"][to_buses, C_MIN]

            if len(from_buses):
                ix = ls_info[:, 0] == 0
                new_ls_buses[ix, VM] = ppc["bus"][from_buses, VM]
                new_ls_buses[ix, VA] = ppc["bus"][from_buses, VA]
                if mode == "sc":
                    new_ls_buses[ix, C_MAX] = ppc["bus"][from_buses, C_MAX]
                    new_ls_buses[ix, C_MIN] = ppc["bus"][from_buses, C_MIN]

            future_buses.append(new_ls_buses)
            # re-route the end of lines to a new bus
            ppc["branch"][ls_info[ls_info[:, 0].astype(bool), 2], 1] = \
                new_indices[ls_info[:, 0].astype(bool)]
            ppc["branch"][ls_info[np.logical_not(ls_info[:, 0]), 2], 0] = \
                new_indices[np.logical_not(ls_info[:, 0])]

            ppc["bus"] = np.vstack(future_buses)

        if nto:
            future_buses = [ppc["bus"]]
            trafo_switches = net["switch"].loc[stidx]

            # determine on which side the switch is located
            mapfunc = partial(_gather_branch_switch_info, branch_type="t", net=net)
            ts_info = list(map(mapfunc,
                               trafo_switches["bus"].values,
                               trafo_switches["element"].values))
            # we now have the following matrix
            # 0: 1 if switch is at lv_bus, 0 else
            # 1: bus of the switch
            # 2: position of the trafo a switch is connected to
            ts_info = np.array(ts_info, dtype=int)

            # build new buses
            new_ts_buses = np.zeros(shape=(nto, ppc["bus"].shape[1]), dtype=float)
            new_indices = np.arange(n_bus + nlo, n_bus + nlo + nto)
            new_ts_buses[:, :15] = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1.1, 0.9, 0, 0])
            new_ts_buses[:, 0] = new_indices
            new_ts_buses[:, BASE_KV] = get_values(ppc["bus"][:, BASE_KV], ts_info[:, 1], bus_lookup)
            # set voltage of new buses to voltage on other branch end
            to_buses = ppc["branch"][ts_info[ts_info[:, 0].astype(bool), 2], 1].real.astype(int)
            from_buses = ppc["branch"][ts_info[np.logical_not(ts_info[:, 0]), 2], 0].real \
                .astype(int)

            # set newly created buses to voltage on other side of
            if len(to_buses):
                ix = ts_info[:, 0] == 1
                taps = ppc["branch"][ts_info[ts_info[:, 0].astype(bool), 2], VA].real
                shift = ppc["branch"][ts_info[ts_info[:, 0].astype(bool), 2], BASE_KV].real
                new_ts_buses[ix, VM] = ppc["bus"][to_buses, VM] * taps
                new_ts_buses[ix, VA] = ppc["bus"][to_buses, VA] + shift
                if mode == "sc":
                    new_ts_buses[ix, C_MAX] = ppc["bus"][to_buses, C_MAX]
                    new_ts_buses[ix, C_MIN] = 0.95#ppc["bus"][to_buses, C_MIN]
            if len(from_buses):
                ix = ts_info[:, 0] == 0
                taps = ppc["branch"][ts_info[np.logical_not(ts_info[:, 0]), 2], VA].real
                shift = ppc["branch"][ts_info[np.logical_not(ts_info[:, 0]), 2], BASE_KV].real
                new_ts_buses[ix, VM] = ppc["bus"][from_buses, VM] * taps
                new_ts_buses[ix, VA] = ppc["bus"][from_buses, VA] + shift
                if mode == "sc":
                    new_ts_buses[ix, C_MAX] = ppc["bus"][from_buses, C_MAX]
                    new_ts_buses[ix, C_MIN] = ppc["bus"][from_buses, C_MIN]
            future_buses.append(new_ts_buses)

            # re-route the hv/lv-side of the trafo to a new bus
            # (trafo entries follow line entries)
            at_lv_bus = ts_info[:, 0].astype(bool)
            at_hv_bus = ~at_lv_bus
            ppc["branch"][len(net.line) + ts_info[at_lv_bus, 2], 1] = \
                new_indices[at_lv_bus]
            ppc["branch"][len(net.line) + ts_info[at_hv_bus, 2], 0] = \
                new_indices[at_hv_bus]

            ppc["bus"] = np.vstack(future_buses)


def _branches_with_oos_buses(net, ppc):
    """
    Updates the ppc["branch"] matrix with the changed from or to values
    if the branch is connected to an out of service bus

    Adds auxiliary buses if branch is connected to an out of service bus
    Sets branch out of service if connected to two out of service buses

    **INPUT**:
        **n** - The pandapower format network

        **ppc** - The PYPOWER format network to fill in values
        **bus_is** - The in service buses
    """
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    # get in service elements
    _is_elements = net["_is_elements"]
    bus_is_idx = _is_elements['bus_is_idx']
    line_is = _is_elements['line']

    n_oos_buses = len(net['bus']) - len(bus_is_idx)

    # only filter lines at oos buses if oos buses exists
    if n_oos_buses > 0:
        n_bus = len(ppc["bus"])
        future_buses = [ppc["bus"]]
        # out of service buses
        bus_oos = np.setdiff1d(net['bus'].index.values, bus_is_idx)
        # from buses of line
        f_bus = line_is.from_bus.values
        t_bus = line_is.to_bus.values

        # determine on which side of the line the oos bus is located
        mask_from = np.in1d(f_bus, bus_oos)
        mask_to = np.in1d(t_bus, bus_oos)

        # determine if line is only connected to out of service buses -> set
        # branch in ppc out of service
        mask_and = mask_to & mask_from

        if np.any(mask_and):
            ppc["branch"][line_is[mask_and].index, BR_STATUS] = 0
            line_is = line_is[~mask_and]
            f_bus = f_bus[~mask_and]
            t_bus = t_bus[~mask_and]
            mask_from = mask_from[~mask_and]
            mask_to = mask_to[~mask_and]

        mask_or = mask_to | mask_from
        # check whether buses are connected to line
        oos_buses_at_lines = np.r_[f_bus[mask_from], t_bus[mask_to]]
        n_oos_buses_at_lines = len(oos_buses_at_lines)

        # only if oos_buses are at lines (they could be isolated as well)
        if n_oos_buses_at_lines > 0:
            ls_info = np.zeros((n_oos_buses_at_lines, 3), dtype=int)
            ls_info[:, 0] = mask_to[mask_or] & ~mask_from[mask_or]
            ls_info[:, 1] = oos_buses_at_lines
            ls_info[:, 2] = np.nonzero(np.in1d(net['line'].index, line_is.index[mask_or]))[0]

            # ls_info = list(map(mapfunc,
            #               line_switches["bus"].values,
            #               line_switches["element"].values))
            # we now have the following matrix
            # 0: 1 if switch is at to_bus, 0 else
            # 1: bus of the switch
            # 2: position of the line a switch is connected to
            # ls_info = np.array(ls_info, dtype=int)

            # build new buses
            new_ls_buses = np.zeros(shape=(n_oos_buses_at_lines, ppc["bus"].shape[1]), dtype=float)
            new_indices = np.arange(n_bus, n_bus + n_oos_buses_at_lines)
            # the newly created buses
            new_ls_buses[:, :15] = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1.1, 0.9, 0, 0])
            new_ls_buses[:, 0] = new_indices
            new_ls_buses[:, BASE_KV] = get_values(ppc["bus"][:, BASE_KV], ls_info[:, 1], bus_lookup)

            future_buses.append(new_ls_buses)

            # re-route the end of lines to a new bus
            ppc["branch"][ls_info[ls_info[:, 0].astype(bool), 2], 1] = \
                new_indices[ls_info[:, 0].astype(bool)]
            ppc["branch"][ls_info[np.logical_not(ls_info[:, 0]), 2], 0] = \
                new_indices[np.logical_not(ls_info[:, 0])]

            ppc["bus"] = np.vstack(future_buses)


def _update_trafo_trafo3w_ppc(net, ppc):
    """
    Updates the trafo and trafo3w values when reusing the ppc between two powerflows

    :param net: pandapower net
    :param ppc: pypower format
    :return: ppc with updates values
    """
    line_end = len(net["line"])
    trafo_end = line_end + len(net["trafo"])
    trafo3w_end = trafo_end + len(net["trafo3w"]) * 3

    if trafo_end > line_end:
        ppc["branch"][line_end:trafo_end,
        [F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT, BR_STATUS, RATE_A]] = \
            _calc_trafo_parameter(net, ppc)
    if trafo3w_end > trafo_end:
        ppc["branch"][trafo_end:trafo3w_end, [F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT, BR_STATUS]] = \
            _calc_trafo3w_parameter(net, ppc)


def _calc_switch_parameter(net, ppc):
    """
    calculates the line parameter in per unit.

    **INPUT**:
        **net** -The pandapower format network

    **RETURN**:
        **t** - Temporary line parameter. Which is a complex128
                Nunmpy array. with the following order:
                0:bus_a; 1:bus_b; 2:r_pu; 3:x_pu; 4:b_pu
    """
    r_switch = net["_options"]["r_switch"]
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    switch = net.switch[net._closed_bb_switches]
    fb = bus_lookup[switch["bus"].values]
    tb = bus_lookup[switch["element"].values]
    baseR = np.square(ppc["bus"][fb, BASE_KV]) / net.sn_kva * 1e3
    t = np.zeros(shape=(len(switch), 3), dtype=np.complex128)

    t[:, 0] = fb
    t[:, 1] = tb

    t[:, 2] = r_switch / baseR
    return t


def _end_temperature_correction_factor(net):
    if "endtemp_degree" not in net.line:
        raise UserWarning("Specify end temperature for lines in net.endtemp_degree")
    return (1 + .004 * (net.line.endtemp_degree.values.astype(float) - 20))  # formula from standard


def _transformer_correction_factor(vsc, vscr, sn, cmax):
    sn = sn / 1000.
    zt = vsc / 100 / sn
    rt = vscr / 100 / sn
    xt = np.sqrt(zt ** 2 - rt ** 2)
    kt = 0.95 * cmax / (1 + .6 * xt * sn)
    return kt
