# -*- coding: utf-8 -*-

# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import copy
import math
from functools import partial

import numpy as np
import pandas as pd

from pandapower.auxiliary import get_values
from pandapower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT, BR_STATUS, RATE_A, \
    BR_R_ASYM, BR_X_ASYM, branch_cols
from pandapower.idx_bus import BASE_KV, VM, VA


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
        ppc["branch"] = np.hstack((ppc["branch"], branch_sc))
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
    end = 0
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
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    trafo_df = _trafo_df_from_trafo3w(net)
    hv_bus = get_trafo_values(trafo_df, "hv_bus").astype(int)
    lv_bus = get_trafo_values(trafo_df, "lv_bus").astype(int)
    in_service = get_trafo_values(trafo_df, "in_service").astype(int)
    temp_para = np.zeros(shape=(len(hv_bus), 9), dtype=np.complex128)
    temp_para[:, 0] = bus_lookup[hv_bus]
    temp_para[:, 1] = bus_lookup[lv_bus]
    temp_para[:, 2:7] = _calc_branch_values_from_trafo_df(net, ppc, trafo_df)
    temp_para[:, 7] = in_service
    if net["_options"]["mode"] == "opf":
        if "max_loading_percent" in trafo_df:
            max_load = get_trafo_values(trafo_df, "max_loading_percent")
            sn_mva = get_trafo_values(trafo_df, "sn_mva")
            temp_para[:, 8] = max_load / 100. * sn_mva
        else:
            temp_para[:, 8] = np.nan
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
    mode = net["_options"]["mode"]
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    line = net["line"]
    fb = bus_lookup[line["from_bus"].values]
    tb = bus_lookup[line["to_bus"].values]
    length = line["length_km"].values
    parallel = line["parallel"].values
    baseR = np.square(ppc["bus"][fb, BASE_KV]) / net.sn_mva
    t = np.zeros(shape=(len(line.index), 7), dtype=np.complex128)

    t[:, 0] = fb
    t[:, 1] = tb

    t[:, 2] = line["r_ohm_per_km"].values * length / baseR / parallel
    t[:, 3] = line["x_ohm_per_km"].values * length / baseR / parallel
    if mode == "sc":
        if net["_options"]["case"] == "min":
            t[:, 2] *= _end_temperature_correction_factor(net)
    else:
        b = (2 * net.f_hz * math.pi * line["c_nf_per_km"].values * 1e-9 * baseR *
             length * parallel)
        g = line["g_us_per_km"].values * 1e-6 * baseR * length * parallel
        t[:, 4] = b - g * 1j
    t[:, 5] = line["in_service"].values
    if net._options["mode"] == "opf":
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

    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    temp_para = np.zeros(shape=(len(net["trafo"].index), 9), dtype=np.complex128)
    trafo = net["trafo"]
    parallel = trafo["parallel"].values
    temp_para[:, 0] = bus_lookup[trafo["hv_bus"].values]
    temp_para[:, 1] = bus_lookup[trafo["lv_bus"].values]
    temp_para[:, 2:7] = _calc_branch_values_from_trafo_df(net, ppc)
    temp_para[:, 7] = trafo["in_service"].values
    if any(trafo.df.values <= 0):
        raise UserWarning("Rating factor df must be positive. Transformers with false "
                          "rating factors: %s" % trafo.query('df<=0').index.tolist())
    if net._options["mode"] == "opf":
        max_load = trafo.max_loading_percent.values if "max_loading_percent" in trafo else 0
        temp_para[:, 8] = max_load / 100. * trafo.sn_mva.values * trafo.df.values * parallel
    return temp_para

def get_trafo_values(trafo_df, par):
    if isinstance(trafo_df, dict):
        return trafo_df[par]
    else:
        return trafo_df[par].values


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
    parallel = get_trafo_values(trafo_df, "parallel")
    vn_lv = get_values(ppc["bus"][:, BASE_KV], get_trafo_values(trafo_df, "lv_bus"), bus_lookup)
    ### Construct np.array to parse results in ###
    # 0:r_pu; 1:x_pu; 2:b_pu; 3:tab;
    temp_para = np.zeros(shape=(len(parallel), 5), dtype=np.complex128)
    vn_trafo_hv, vn_trafo_lv, shift = _calc_tap_from_dataframe(net, trafo_df)
    ratio = _calc_nominal_ratio_from_dataframe(ppc, trafo_df, vn_trafo_hv, vn_trafo_lv,
                                               bus_lookup)
    r, x, y = _calc_r_x_y_from_dataframe(net, trafo_df, vn_trafo_lv, vn_lv, net.sn_mva)
    temp_para[:, 0] = r / parallel
    temp_para[:, 1] = x / parallel
    temp_para[:, 2] = y * parallel
    temp_para[:, 3] = ratio
    temp_para[:, 4] = shift
    return temp_para


def _calc_r_x_y_from_dataframe(net, trafo_df, vn_trafo_lv, vn_lv, sn_mva):
    mode = net["_options"]["mode"]
    trafo_model = net["_options"]["trafo_model"]

    r, x = _calc_r_x_from_dataframe(trafo_df, vn_lv, vn_trafo_lv, sn_mva)
    if mode == "sc":
        y = 0
        if isinstance(trafo_df, pd.DataFrame): #2w trafo is dataframe, 3w trafo is dict
            from pandapower.shortcircuit.idx_bus import C_MAX
            bus_lookup = net._pd2ppc_lookups["bus"]
            cmax = net._ppc["bus"][bus_lookup[net.trafo.lv_bus.values], C_MAX]
            kt = _transformer_correction_factor(trafo_df.vsc_percent, trafo_df.vscr_percent,
                                                trafo_df.sn_mva, cmax)
            r *= kt
            x *= kt
    else:
        y = _calc_y_from_dataframe(trafo_df, vn_lv, vn_trafo_lv, sn_mva)
    if trafo_model == "pi":
        return r, x, y
    elif trafo_model == "t":
        return                                                                                                                                                                                              (r, x, y)
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


def _calc_y_from_dataframe(trafo_df, vn_lv, vn_trafo_lv, sn_mva):
    """
    Calculate the subsceptance y from the transformer dataframe.

    INPUT:

        **trafo** (Dataframe) - The dataframe in net.trafo
        which contains transformer calculation values.

    OUTPUT:
        **subsceptance** (1d array, np.complex128) - The subsceptance in pu in
        the form (-b_img, -b_real)
    """
    baseR = np.square(vn_lv) / sn_mva
    vn_lv_kv = get_trafo_values(trafo_df, "vn_lv_kv")
    pfe = get_trafo_values(trafo_df, "pfe_mw")

    ### Calculate subsceptance ###
    vnl_squared = vn_lv_kv ** 2
    b_real = pfe / vnl_squared * baseR
    i0 =  get_trafo_values(trafo_df, "i0_percent")
    sn = get_trafo_values(trafo_df, "sn_mva")
    b_img = (i0 / 100. * sn) ** 2 - pfe ** 2

    b_img[b_img < 0] = 0
    b_img = np.sqrt(b_img) * baseR / vnl_squared
    y = - b_real * 1j - b_img * np.sign(i0)
    if "lv" in get_trafo_values(trafo_df, "tp_side"):
        return y / np.square(vn_trafo_lv / vn_lv_kv)
    else:
        return y

def _calc_tap_from_dataframe(net, trafo_df):
    """
    Adjust the nominal voltage vnh and vnl to the active tab position "tp_pos".
    If "side" is 1 (high-voltage side) the high voltage vnh is adjusted.
    If "side" is 2 (low-voltage side) the low voltage vnl is adjusted

    INPUT:
        **net** - The pandapower format network

        **trafo** (Dataframe) - The dataframe in pd_net["structure"]["trafo"]
        which contains transformer calculation values.

    OUTPUT:
        **vn_hv_kv** (1d array, float) - The adusted high voltages

        **vn_lv_kv** (1d array, float) - The adjusted low voltages

        **trafo_shift** (1d array, float) - phase shift angle

    """
    calculate_voltage_angles = net["_options"]["calculate_voltage_angles"]
    mode = net["_options"]["mode"]
    vnh = copy.copy(get_trafo_values(trafo_df, "vn_hv_kv").astype(float))
    vnl = copy.copy(get_trafo_values(trafo_df, "vn_lv_kv").astype(float))
    trafo_shift = get_trafo_values(trafo_df, "shift_degree").astype(float) if calculate_voltage_angles else \
        np.zeros(len(vnh))
    if mode == "sc":
        return vnh, vnl, trafo_shift

    tp_pos = get_trafo_values(trafo_df, "tp_pos")
    tp_mid = get_trafo_values(trafo_df, "tp_mid")
    tp_diff =  tp_pos - tp_mid
    tp_phase_shifter = get_trafo_values(trafo_df, "tp_phase_shifter")
    tp_side = get_trafo_values(trafo_df, "tp_side")
    tp_st_percent = get_trafo_values(trafo_df, "tp_st_percent")
    tp_st_degree = get_trafo_values(trafo_df, "tp_st_degree")

    cos = lambda x: np.cos(np.deg2rad(x))
    sin = lambda x: np.sin(np.deg2rad(x))
    arctan = lambda x: np.rad2deg(np.arctan(x))

    for side, vn, direction in [("hv", vnh, 1), ("lv", vnl, -1)]:
        phase_shifters = tp_phase_shifter & (tp_side == side)
        tap_complex = np.isfinite(tp_st_percent) & np.isfinite(tp_pos) & (tp_side == side) & \
                       ~phase_shifters
        if np.any(tap_complex):
            tp_steps = tp_st_percent[tap_complex] * tp_diff[tap_complex] / 100
            tp_angles = np.nan_to_num(tp_st_degree[tap_complex])
            u1 = vn[tap_complex]
            du = u1 * np.nan_to_num(tp_steps)
            vn[tap_complex] = np.sqrt((u1 + du * cos(tp_angles)) ** 2 + (du * sin(tp_angles)) ** 2)
            trafo_shift[tap_complex] += (arctan(direction * du * sin(tp_angles) /
                                                (u1 + du * cos(tp_angles))))
        if np.any(phase_shifters):
            degree_is_set = np.nan_to_num(tp_st_degree[phase_shifters])!= 0
            percent_is_set = np.nan_to_num(tp_st_percent[phase_shifters]) !=0
            if any(degree_is_set & percent_is_set):
                raise UserWarning("Both tp_st_degree and tp_st_percent set for ideal phase shifter")
            trafo_shift[phase_shifters] += np.where(
                (degree_is_set),
                (direction * tp_diff[phase_shifters] * tp_st_degree[phase_shifters]),
                (direction * 2 * np.rad2deg(np.arcsin(tp_diff[phase_shifters] * \
                                                      tp_st_percent[phase_shifters]/100/2)))
                )
    return vnh, vnl, trafo_shift


def _calc_r_x_from_dataframe(trafo_df, vn_lv, vn_trafo_lv, sn_mva):
    """
    Calculates (Vectorized) the resitance and reactance according to the
    transformer values

    """
    vsc_percent = get_trafo_values(trafo_df, "vsc_percent")
    vscr_percent = get_trafo_values(trafo_df, "vscr_percent")
    tap_lv = np.square(vn_trafo_lv / vn_lv) * sn_mva  # adjust for low voltage side voltage converter
    sn_trafo_mva = get_trafo_values(trafo_df, "sn_mva")
    z_sc = vsc_percent / 100. / sn_trafo_mva * tap_lv
    r_sc = vscr_percent / 100. / sn_trafo_mva * tap_lv
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
    hv_bus = get_trafo_values(trafo_df, "hv_bus")
    lv_bus = get_trafo_values(trafo_df, "lv_bus")
    nom_rat = get_values(ppc["bus"][:, BASE_KV], hv_bus, bus_lookup) / \
              get_values(ppc["bus"][:, BASE_KV], lv_bus, bus_lookup)
    return tap_rat / nom_rat


def z_br_to_bus(z, s):
    return s[0] * np.array([z[0] / min(s[0], s[1]), z[1] /
                            min(s[1], s[2]), z[2] / min(s[0], s[2])])


def wye_delta(zbr_n, s):
    return .5 * s / s[0] * np.array([(zbr_n[0] + zbr_n[2] - zbr_n[1]),
                                     (zbr_n[1] + zbr_n[0] - zbr_n[2]),
                                     (zbr_n[2] + zbr_n[1] - zbr_n[0])])




def _calc_impedance_parameter(net):
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    t = np.zeros(shape=(len(net["impedance"].index), 7), dtype=np.complex128)
    sn_impedance = net["impedance"]["sn_mva"].values
    sn_net = net.sn_mva
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
            net.sn_mva
    t = np.zeros(shape=(len(net["xward"].index), 5), dtype=np.complex128)
    xw_is = net["_is_elements"]["xward"]
    t[:, 0] = bus_lookup[net["xward"]["bus"].values]
    t[:, 1] = bus_lookup[net._pd2ppc_lookups["aux"]["xward"]]
    t[:, 2] = net["xward"]["r_ohm"] / baseR
    t[:, 3] = net["xward"]["x_ohm"] / baseR
    t[:, 4] = xw_is
    return t


def _gather_branch_switch_info(bus, branch_id, branch_type, net):
    # determine at which end the switch is located
    # 1 = to-bus/lv-bus; 0 = from-bus/hv-bus
    branch_id = int(branch_id)
    lookup = net._pd2ppc_lookups["branch"]
    if branch_type == "l":
        branch_bus = net["line"]["to_bus"].at[branch_id]
        is_to_bus = int(branch_bus == bus)
        branch_idx = net["line"].index.get_loc(branch_id)
        return is_to_bus, bus, branch_idx
    elif branch_type == "t":
        branch_bus = net["trafo"]["lv_bus"].at[branch_id]
        is_to_bus = int(branch_bus == bus)
        branch_idx = lookup["trafo"][0] + net["trafo"].index.get_loc(branch_id)
        return is_to_bus, bus, branch_idx
    elif branch_type == "t3":
        f, t = lookup["trafo3w"]
        if net["trafo3w"]["hv_bus"].at[branch_id] == bus:
            is_to_bus = 0
            offset = 0
        elif net["trafo3w"]["mv_bus"].at[branch_id] == bus:
            is_to_bus = 1
            offset = (t - f)/3
        elif net["trafo3w"]["lv_bus"].at[branch_id] == bus:
            is_to_bus = 1
            offset = (t - f)/3*2
        branch_idx = lookup["trafo3w"][0] + net["trafo3w"].index.get_loc(branch_id) + offset
        return is_to_bus, bus,branch_idx

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
    bus_is_idx = _is_elements['bus_is_idx']
    lines_is_index = _is_elements["line_is_idx"]

    # opened bus line switches
    slidx = (net["switch"]["closed"].values == 0) \
            & (net["switch"]["et"].values == "l")

    # check if there are multiple opened switches at a line (-> set line out of service)
    sw_elem = net['switch'][slidx]["element"].values
    m = np.zeros_like(sw_elem, dtype=bool)
    m[np.unique(sw_elem, return_index=True)[1]] = True

    # if non unique elements are in sw_elem (= multiple opened bus line switches)
    if np.count_nonzero(m) < len(sw_elem):
        if 'line' not in _is_elements:
            get_is_lines(net)
        lines_is = _is_elements['line']
        lines_to_delete = [idx for idx in sw_elem[~m] if idx in lines_is.index]

        from_bus = lines_is.loc[lines_to_delete].from_bus.values
        to_bus = lines_is.loc[lines_to_delete].to_bus.values
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
            lines_is = lines_is.drop(lines_to_delete)
            _is_elements["line_is_idx"] = lines_is.index

    # opened switches at in service lines
    slidx = slidx \
            & (np.in1d(net["switch"]["element"].values, lines_is_index)) \
            & (np.in1d(net["switch"]["bus"].values, bus_is_idx))
    nlo = np.count_nonzero(slidx)

    stidx = (net.switch["closed"].values == 0) & (net.switch["et"].values == "t")
    st3idx = (net.switch["closed"].values == 0) & (net.switch["et"].values == "t3")
    nto =  np.count_nonzero(stidx) + np.count_nonzero(st3idx)

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
            # 2: branch index of the line a switch is connected to
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
            trafo3_switches = net["switch"].loc[st3idx]
            # determine on which side the switch is located
            t2s_info = list(map(partial(_gather_branch_switch_info, branch_type="t", net=net),
                               trafo_switches["bus"].values,
                               trafo_switches["element"].values))
            t3s_info = list(map(partial(_gather_branch_switch_info, branch_type="t3", net=net),
                               trafo3_switches["bus"].values,
                               trafo3_switches["element"].values))
            ts_info = t2s_info + t3s_info
            # we now have the following matrix
            # 0: 1 if switch is at lv_bus, 0 else
            # 1: bus of the switch
            # 2: branch index of the trafo a switch is connected to
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
                    new_ts_buses[ix, C_MIN] = 0.95  # ppc["bus"][to_buses, C_MIN]
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
            ppc["branch"][ts_info[at_lv_bus, 2], 1] = new_indices[at_lv_bus]
            ppc["branch"][ts_info[at_hv_bus, 2], 0] = new_indices[at_hv_bus]

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
    line_is_idx = _is_elements['line_is_idx']

    n_oos_buses = len(net['bus']) - len(bus_is_idx)

    # only filter lines at oos buses if oos buses exists
    if n_oos_buses > 0:
        n_bus = len(ppc["bus"])
        future_buses = [ppc["bus"]]
        # out of service buses
        bus_oos = np.setdiff1d(net['bus'].index.values, bus_is_idx)
        # from buses of line
        line_buses = net["line"][["from_bus", "to_bus"]].loc[line_is_idx].values
        f_bus = line_buses[:, 0]
        t_bus = line_buses[:, 1]

        # determine on which side of the line the oos bus is located
        mask_from = np.in1d(f_bus, bus_oos)
        mask_to = np.in1d(t_bus, bus_oos)

        mask_and = mask_to & mask_from
        if np.any(mask_and):
            mask_from[mask_and] = False
            mask_to[mask_and] = False

        # get lines that are connected to oos bus at exactly one side
        # buses that are connected to two oos buses will be removed by ext2int
        mask_or = mask_to | mask_from
        # check whether buses are connected to line
        oos_buses_at_lines = np.r_[f_bus[mask_from], t_bus[mask_to]]
        n_oos_buses_at_lines = len(oos_buses_at_lines)

        # only if oos_buses are at lines (they could be isolated as well)
        if n_oos_buses_at_lines > 0:
            ls_info = np.zeros((n_oos_buses_at_lines, 3), dtype=int)
            ls_info[:, 0] = mask_to[mask_or] & ~mask_from[mask_or]
            ls_info[:, 1] = oos_buses_at_lines
            ls_info[:, 2] = np.nonzero(np.in1d(net['line'].index, line_is_idx[mask_or]))[0]

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
    baseR = np.square(ppc["bus"][fb, BASE_KV]) / net.sn_mva
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
    zt = vsc / 100 / sn
    rt = vscr / 100 / sn
    xt = np.sqrt(zt ** 2 - rt ** 2)
    kt = 0.95 * cmax / (1 + .6 * xt * sn)
    return kt


def get_is_lines(net):
    _is_elements = net["_is_elements"]
    _is_elements["line"] = net["line"][net["line"]["in_service"].values.astype(bool)]

def _trafo_df_from_trafo3w(net):
    def get_nan(net):
        empty = np.zeros((len(net.trafo3w)*3))
        empty.fill(np.nan)
        return empty

    nr_trafos = len(net["trafo3w"])
    trafo2 = dict()
    mode = net._options["mode"]
    loss_location = net._options["trafo3w_losses"].lower()
    nr_trafos = len(net["trafo3w"])
    tap_variables = ("tp_side", "tp_pos", "tp_mid", "tp_max", "tp_min", "tp_st_percent",
                     "tp_st_degree", "tap_at_star_point")
    i = 0
    empty = get_nan(net)
    vsc = empty.copy()
    vscr = empty.copy()
    sn = empty.copy()
    t3 = net["trafo3w"]
    t3_variables = ("vsc_hv_percent", "vsc_mv_percent", "vsc_lv_percent", "vscr_hv_percent",
                    "vscr_mv_percent", "vscr_lv_percent", "sn_hv_mva", "sn_mv_mva", "sn_lv_mva")
    for i, (vsc_hv_percent, vsc_mv_percent, vsc_lv_percent, vscr_hv_percent, vscr_mv_percent,
            vscr_lv_percent, sn_hv_mva, sn_mv_mva, sn_lv_mva) \
            in enumerate(zip(*(t3[var].values for var in t3_variables))):
        vsc_3w = np.array([vsc_hv_percent, vsc_mv_percent, vsc_lv_percent], dtype=float)
        vscr_3w = np.array([vscr_hv_percent, vscr_mv_percent, vscr_lv_percent], dtype=float)
        sn_3w = np.array([sn_hv_mva, sn_mv_mva, sn_lv_mva])
        vsc_2w_delta = z_br_to_bus(vsc_3w, sn_3w)
        vscr_2w_delta = z_br_to_bus(vscr_3w, sn_3w)
        if mode == "sc":
            kt = _transformer_correction_factor(vsc_3w, vscr_3w, sn_3w, 1.1)
            vsc_2w_delta *= kt
            vscr_2w_delta *= kt
        vsci_2w_delta = np.sqrt(vsc_2w_delta ** 2 - vscr_2w_delta ** 2)
        vscr_2w = wye_delta(vscr_2w_delta, sn_3w)
        vsci_2w = wye_delta(vsci_2w_delta, sn_3w)
        vsc_2w = np.sign(vsci_2w) * np.sqrt(vsci_2w ** 2 + vscr_2w ** 2)
        vsc[i::nr_trafos] = vsc_2w
        vscr[i::nr_trafos] = vscr_2w
        sn[i::nr_trafos] = sn_3w
    trafo2["vscr_percent"] = vscr
    if any(vsc==0):
        raise UserWarning("Equivalent transformer with zero impedance!")
    trafo2["vsc_percent"] = vsc
    trafo2["sn_mva"] = sn
    tap_arrays = {var: np.array([None]*nr_trafos*3) if var == "tp_side" else  empty.copy()
                    for var in tap_variables if var != "tap_at_star_point"}
    for i, (tp_side, tp_pos, tp_mid, tp_max, tp_min, tp_st_percent, tp_st_degree, tap_at_star_point)\
            in enumerate(zip(*(t3[var].values for var in tap_variables))):
        if pd.notnull(tp_side):
            if tp_side == "hv" or tp_side == 0:
                offset = 0
            elif tp_side == "mv":
                offset = nr_trafos
            elif tp_side == "lv":
                offset = 2*nr_trafos
            tap_arrays["tp_side"][i+offset] = tp_side
            tap_arrays["tp_pos"][i+offset] = tp_pos
            tap_arrays["tp_mid"][i+offset] = tp_mid
            tap_arrays["tp_max"][i+offset] = tp_max
            tap_arrays["tp_min"][i+offset] = tp_min
            tap_arrays["tp_st_percent"][i+offset] = tp_st_percent
            tap_arrays["tp_st_degree"][i+offset] = tp_st_degree

            # consider where the tap is located - at the bus or at star point of the 3W-transformer
            if not tap_at_star_point:
                tap_arrays["tp_side"][i+offset] = "hv" if offset == 0 else "lv"
            else:
                tap_arrays["tp_side"][i+offset] = "lv" if offset == 0 else "hv"
                tap_arrays["tp_st_degree"][i+offset] += 180

    for column, values in tap_arrays.items():
        trafo2[column] = values
    zeros = np.zeros(len(net.trafo3w))
    aux_buses = net._pd2ppc_lookups["aux"]["trafo3w"]
    trafo2["hv_bus"] = np.concatenate([t3.hv_bus.values, aux_buses, aux_buses])
    trafo2["lv_bus"] = np.concatenate([aux_buses, t3.mv_bus.values, t3.lv_bus.values])
    trafo2["in_service"] = np.concatenate([t3.in_service.values for _ in range(3)])
    if loss_location == "hv":
        trafo2["i0_percent"] = np.concatenate([t3.i0_percent.values, zeros, zeros])
        trafo2["pfe_mw"] = np.concatenate([t3.pfe_mw.values, zeros, zeros])
    elif loss_location == "mv":
        trafo2["i0_percent"] = np.concatenate([zeros, t3.i0_percent.values, zeros])
        trafo2["pfe_mw"] = np.concatenate([zeros, t3.pfe_mw.values, zeros])
    elif loss_location == "lv":
        trafo2["i0_percent"] = np.concatenate([zeros, zeros, t3.i0_percent.values])
        trafo2["pfe_mw"] = np.concatenate([zeros, zeros, t3.pfe_mw.values])
    elif loss_location == "star":
        trafo2["i0_percent"] = np.concatenate([zeros for _ in range(3)])
        trafo2["pfe_mw"] = np.concatenate([zeros for _ in range(3)])
    else:
        raise UserWarning("Invalid trafo3w loss location '%s'"%loss_location)
    trafo2["vn_hv_kv"] = np.concatenate([t3.vn_hv_kv, t3.vn_hv_kv, t3.vn_hv_kv])
    trafo2["vn_lv_kv"] = np.concatenate([t3.vn_hv_kv, t3.vn_mv_kv, t3.vn_lv_kv])
    trafo2["shift_degree"] = np.concatenate([np.zeros(nr_trafos), t3.shift_mv_degree, t3.shift_lv_degree])
    trafo2["tp_phase_shifter"] = np.zeros(nr_trafos*3).astype(bool)
    trafo2["parallel"] = np.ones(nr_trafos*3)
    trafo2["df"] = np.ones(nr_trafos*3)
    if net._options["mode"] == "opf" and "max_loading_percent" in net.trafo3w:
        trafo2["max_loading_percent"] = np.concatenate([net.trafo3w.max_loading_percent.values for _ in range(3)])
    return trafo2

if __name__ == '__main__':
    from pandapower.test.conftest import result_test_network
    net = result_test_network()