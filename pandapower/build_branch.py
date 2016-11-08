# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

import copy
import numpy as np
import pandas as pd
import math
from functools import partial

from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT, BR_STATUS, RATE_A, QT
from pypower.idx_bus import BASE_KV

from pandapower.auxiliary import get_indices, get_values


def _build_branch_ppc(net, ppc, is_elems, bus_lookup, calculate_voltage_angles, trafo_model, set_opf_constraints=False):
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
        **net** -The Pandapower format network

        **ppc** - The PYPOWER format network to fill in values

    """
#    if len(net["trafo3w"]) > 0:
#        _one_3w_to_three_2w(net)
    line_end = len(net["line"])
    trafo_end = line_end + len(net["trafo"])
    trafo3w_end = trafo_end + len(net["trafo3w"]) * 3
    impedance_end = trafo3w_end + len(net["impedance"])
    xward_end = impedance_end + len(net["xward"])

    ppc["branch"] = np.zeros(shape=(xward_end, QT + 1), dtype=np.complex128)
    ppc["branch"][:, :13] = np.array([0, 0, 0, 0, 0, 250, 250, 250, 1, 0, 1, -360, 360])

    if line_end > 0:
        ppc["branch"][:line_end, [F_BUS, T_BUS, BR_R, BR_X, BR_B, BR_STATUS, RATE_A]] = \
            _calc_line_parameter(net, ppc, bus_lookup, set_opf_constraints)
    if trafo_end > line_end:
        ppc["branch"][line_end:trafo_end,
                      [F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT, BR_STATUS,  RATE_A]] = \
            _calc_trafo_parameter(net, ppc, bus_lookup, calculate_voltage_angles,
                                  trafo_model, set_opf_constraints)
    if trafo3w_end > trafo_end:
        ppc["branch"][trafo_end:trafo3w_end, [F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT, BR_STATUS]] = \
            _calc_trafo3w_parameter(net, ppc, bus_lookup, calculate_voltage_angles,  trafo_model)
    if impedance_end > trafo3w_end:
        ppc["branch"][trafo3w_end:impedance_end, [F_BUS, T_BUS, BR_R, BR_X, BR_STATUS]] = \
            _calc_impedance_parameter(net, bus_lookup)
    if xward_end > impedance_end:
        ppc["branch"][impedance_end:xward_end, [F_BUS, T_BUS, BR_R, BR_X, BR_STATUS]] = \
            _calc_xward_parameter(net, ppc, is_elems, bus_lookup)


def _calc_trafo3w_parameter(net, ppc, bus_lookup, calculate_voltage_angles, trafo_model):
    trafo_df = _trafo_df_from_trafo3w(net)

    temp_para = np.zeros(shape=(len(trafo_df), 8), dtype=np.complex128)
    temp_para[:, 0] = get_indices(trafo_df["hv_bus"].values, bus_lookup)
    temp_para[:, 1] = get_indices(trafo_df["lv_bus"].values, bus_lookup)
    temp_para[:, 2:6] = _calc_branch_values_from_trafo_df(
        net, ppc, bus_lookup, trafo_model, trafo_df)
    if calculate_voltage_angles:
        temp_para[:, 6] = trafo_df["shift_degree"].values
    else:
        temp_para[:, 6] = np.zeros(shape=(len(trafo_df.index),), dtype=np.complex128)
    temp_para[:, 7] = trafo_df["in_service"].values
    return temp_para


def _calc_line_parameter(net, ppc, bus_lookup, set_opf_constraints=False):
    """
    calculates the line parameter in per unit.

    **INPUT**:
        **net** -The Pandapower format network

    **RETURN**:
        **t** - Temporary line parameter. Which is a complex128
                Nunmpy array. with the following order:
                0:bus_a; 1:bus_b; 2:r_pu; 3:x_pu; 4:b_pu
    """

    # baseR converts Ohm to p.u. Formula is U^2/Sref. Sref is 1 MVA and vn_kv is
    # in kV U^2* ((10^3 V)^2/10^6 VA) = U^2
    # Therefore division by 1 MVA is not necessary.
    line = net["line"]
    fb = get_indices(line["from_bus"], bus_lookup)
    tb = get_indices(line["to_bus"], bus_lookup)
    length = line["length_km"].values
    parallel = line["parallel"]
    baseR = np.square(ppc["bus"][fb, BASE_KV])
    t = np.zeros(shape=(len(line.index), 7), dtype=np.complex128)

    t[:, 0] = fb
    t[:, 1] = tb

    t[:, 2] = line["r_ohm_per_km"] * length / baseR / parallel
    t[:, 3] = line["x_ohm_per_km"] * length / baseR / parallel
    t[:, 4] = 2 * net.f_hz * math.pi * line["c_nf_per_km"] * 1e-9 * baseR * length * parallel
    t[:, 5] = line["in_service"]
    if set_opf_constraints:
        max_load = line.max_loading_percent if "max_loading_percent" in line else 1000
        vr = net.bus.vn_kv[fb].values * np.sqrt(3)
        t[:, 6] = max_load / 100 * line.imax_ka * line.df * parallel * vr
    return t


def _calc_trafo_parameter(net, ppc, bus_lookup, calculate_voltage_angles, trafo_model,
                          set_opf_constraints=False):
    '''
    Calculates the transformer parameter in per unit.

    **INPUT**:
        **net** - The Pandapower format network

    **RETURN**:
        **temp_para** -
        Temporary transformer parameter. Which is a np.complex128
        Numpy array. with the following order:
        0:hv_bus; 1:lv_bus; 2:r_pu; 3:x_pu; 4:b_pu; 5:tab, 6:shift
    '''
    temp_para = np.zeros(shape=(len(net["trafo"].index), 9), dtype=np.complex128)
    trafo = net["trafo"]
    temp_para[:, 0] = get_indices(trafo["hv_bus"].values, bus_lookup)
    temp_para[:, 1] = get_indices(trafo["lv_bus"].values, bus_lookup)
    temp_para[:, 2:6] = _calc_branch_values_from_trafo_df(net, ppc, bus_lookup, trafo_model)
    if calculate_voltage_angles:
        temp_para[:, 6] = trafo["shift_degree"].values
    else:
        temp_para[:, 6] = np.zeros(shape=(len(trafo.index),), dtype=np.complex128)
    temp_para[:, 7] = trafo["in_service"].values
    if set_opf_constraints:
        max_load = trafo.max_loading_percent if "max_loading_percent" in trafo else 1000
        temp_para[:, 8] = max_load / 100 * trafo.sn_kva / 1000
    return temp_para


def _calc_branch_values_from_trafo_df(net, ppc, bus_lookup, trafo_model, trafo_df=None):
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
        **pd_trafo** - The Pandapower format Transformer Dataframe.
                        The Transformer modell will only readfrom pd_net

    **RETURN**:
        **temp_para** - Temporary transformer parameter. Which is a complex128
                        Nunmpy array. with the following order:
                        0:r_pu; 1:x_pu; 2:b_pu; 3:tab;

    """
    if trafo_df is None:
        trafo_df = net["trafo"]
    baseR = np.square(get_values(ppc["bus"][:, BASE_KV], trafo_df["lv_bus"].values,
                                 bus_lookup))

    ### Construct np.array to parse results in ###
    # 0:r_pu; 1:x_pu; 2:b_pu; 3:tab;
    temp_para = np.zeros(shape=(len(trafo_df), 4), dtype=np.complex128)
    unh, unl = _calc_vn_from_dataframe(trafo_df)
    r, x, y = _calc_r_x_y_from_dataframe(trafo_df, unl, baseR, trafo_model)
    temp_para[:, 0] = r
    temp_para[:, 1] = x
    temp_para[:, 2] = y
    temp_para[:, 3] = _calc_tap_from_dataframe(ppc, trafo_df, unh, unl, bus_lookup)
    return temp_para


def _calc_r_x_y_from_dataframe(trafo_df, unl, baseR, trafo_model):
    y = _calc_y_from_dataframe(trafo_df, baseR)
    r, x = _calc_r_x_from_dataframe(trafo_df)
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


def _calc_y_from_dataframe(trafo_df, baseR):
    """
    Calculate the subsceptance y from the transformer dataframe.

    INPUT:

        **trafo** (Dataframe) - The dataframe in net.trafo
        which contains transformer calculation values.

    RETURN:

        **subsceptance** (1d array, np.complex128) - The subsceptance in pu in
        the form (-b_img, -b_real)
    """
    ### Calculate subsceptance ###
    unl_squared = trafo_df["vn_lv_kv"].values**2
    b_real = trafo_df["pfe_kw"].values / (1000. * unl_squared) * baseR

    b_img = (trafo_df["i0_percent"].values / 100. * trafo_df["sn_kva"].values / 1000.)**2 \
        - (trafo_df["pfe_kw"].values / 1000.)**2

    b_img[b_img < 0] = 0
    b_img = np.sqrt(b_img) * baseR / unl_squared

    return -b_real * 1j - b_img


def _calc_vn_from_dataframe(trafo_df):
    """
    Adjust the nominal voltage vnh and vnl to the active tab position "tp_pos".
    If "side" is 1 (high-voltage side) the high voltage vnh is adjusted.
    If "side" is 2 (low-voltage side) the low voltage vnl is adjusted

    INPUT:

        **trafo** (Dataframe) - The dataframe in pd_net["structure"]["trafo"]
        which contains transformer calculation values.

    RETURN:

        **vn_hv_kv** (1d array, float) - The adusted high voltages

        **vn_lv_kv** (1d array, float) - The adjusted low voltages

    """
    # Changing Voltage on high-voltage side
    unh = copy.copy(trafo_df["vn_hv_kv"].values)
    m = (trafo_df["tp_side"] == "hv").values
    tap_os = np.isfinite(trafo_df["tp_pos"].values) & m
    if any(tap_os):
        unh[tap_os] *= np.ones((tap_os.sum()), dtype=np.float) + \
            (trafo_df["tp_pos"].values[tap_os] - trafo_df["tp_mid"].values[tap_os]) * \
            trafo_df["tp_st_percent"].values[tap_os] / 100.

    # Changing Voltage on high-voltage side
    unl = copy.copy(trafo_df["vn_lv_kv"].values)
    tap_us = np.logical_and(np.isfinite(trafo_df["tp_pos"].values),
                            (trafo_df["tp_side"] == "lv").values)
    if any(tap_us):
        unl[tap_us] *= np.ones((tap_us.sum()), dtype=np.float) \
            + (trafo_df["tp_pos"].values[tap_us] - trafo_df["tp_mid"].values[tap_us]) \
            * trafo_df["tp_st_percent"].values[tap_us] / 100.

    return unh, unl


def _calc_r_x_from_dataframe(trafo_df):
    """
    Calculates (Vectorized) the resitance and reactance according to the
    transformer values

    """
    z_sc = trafo_df["vsc_percent"].values / 100. / trafo_df.sn_kva.values * 1000.
    r_sc = trafo_df["vscr_percent"].values / 100. / trafo_df.sn_kva.values * 1000.
    x_sc = np.sqrt(z_sc**2 - r_sc**2)
    return r_sc, x_sc


def _calc_tap_from_dataframe(ppc, trafo_df, vn_hv_kv, vn_lv_kv, bus_lookup):
    """
    Calculates (Vectorized) the off nominal tap ratio::

                  (vn_hv_kv / vn_lv_kv) / (ub1_in_kv / ub2_in_kv)

    INPUT:

        **net** (Dataframe) - The net for which to calc the tap ratio.

        **vn_hv_kv** (1d array, float) - The adjusted nominal high voltages

        **vn_lv_kv** (1d array, float) - The adjusted nominal low voltages

    RETURN:

        **tab** (1d array, float) - The off-nominal tap ratio
    """
    # Calculating tab (trasformer off nominal turns ratio)
    tap_rat = vn_hv_kv / vn_lv_kv
    nom_rat = get_values(ppc["bus"][:, BASE_KV], trafo_df["hv_bus"].values, bus_lookup) / \
        get_values(ppc["bus"][:, BASE_KV], trafo_df["lv_bus"].values, bus_lookup)
    return tap_rat / nom_rat


def z_br_to_bus(z, s):
    zbr_n = s[0] * np.array([z[0] / min(s[0], s[1]), z[1] /
                             min(s[1], s[2]), z[2] / min(s[0], s[2])])

    return .5 * s / s[0] * np.array([(zbr_n[0] + zbr_n[2] - zbr_n[1]),
                                     (zbr_n[1] + zbr_n[0] - zbr_n[2]),
                                     (zbr_n[2] + zbr_n[1] - zbr_n[0])])


def _trafo_df_from_trafo3w(net):
    trafos2w = {}
    nr_trafos = len(net["trafo3w"])
    tap_variables = ("tp_pos", "tp_mid", "tp_max", "tp_min", "tp_st_percent")
    i = 0
    for _, ttab in net["trafo3w"].iterrows():
        uk = np.array([ttab.vsc_hv_percent, ttab.vsc_mv_percent, ttab.vsc_lv_percent])
        ur = np.array([ttab.vscr_hv_percent, ttab.vscr_mv_percent, ttab.vscr_lv_percent])
        sn = np.array([ttab.sn_hv_kva, ttab.sn_mv_kva, ttab.sn_lv_kva])
        uk_2w = z_br_to_bus(uk, sn)
        ur_2w = z_br_to_bus(ur, sn)
        taps = [{tv: np.nan for tv in tap_variables} for _ in range(3)]
        for k in range(3):
            taps[k]["tp_side"] = None

        if pd.notnull(ttab.tp_side):
            if ttab.tp_side == "hv":
                tp_trafo = 0
            elif ttab.tp_side == "mv":
                tp_trafo = 1
            elif ttab.tp_side == "lv":
                tp_trafo = 3
            for tv in tap_variables:
                taps[tp_trafo][tv] = ttab[tv]
            taps[tp_trafo]["tp_side"] = "hv" if tp_trafo == 0 else "lv"
        trafos2w[i] = {"hv_bus": ttab.hv_bus, "lv_bus": ttab.ad_bus, "sn_kva": ttab.sn_hv_kva,
                       "vn_hv_kv": ttab.vn_hv_kv, "vn_lv_kv": ttab.vn_hv_kv, "vscr_percent": ur_2w[0],
                       "vsc_percent": uk_2w[0], "pfe_kw": ttab.pfe_kw,
                       "i0_percent": ttab.i0_percent, "tp_side": taps[0]["tp_side"],
                       "tp_mid": taps[0]["tp_mid"], "tp_max": taps[0]["tp_max"],
                       "tp_min": taps[0]["tp_min"], "tp_pos": taps[0]["tp_pos"],
                       "tp_st_percent": taps[0]["tp_st_percent"],
                       "in_service": ttab.in_service, "shift_degree": 0}
        trafos2w[i + nr_trafos] = {"hv_bus": ttab.ad_bus, "lv_bus": ttab.mv_bus,
                                   "sn_kva": ttab.sn_mv_kva, "vn_hv_kv": ttab.vn_hv_kv, "vn_lv_kv": ttab.vn_mv_kv,
                                   "vscr_percent": ur_2w[1], "vsc_percent": uk_2w[1], "pfe_kw": 0,
                                   "i0_percent": 0, "tp_side": taps[1]["tp_side"],
                                   "tp_mid": taps[1]["tp_mid"], "tp_max": taps[1]["tp_max"],
                                   "tp_min": taps[1]["tp_min"], "tp_pos": taps[1]["tp_pos"],
                                   "tp_st_percent": taps[1]["tp_st_percent"],
                                   "in_service": ttab.in_service, "shift_degree": ttab.shift_mv_degree}
        trafos2w[i + 2 * nr_trafos] = {"hv_bus": ttab.ad_bus, "lv_bus": ttab.lv_bus,
                                       "sn_kva": ttab.sn_lv_kva,
                                       "vn_hv_kv": ttab.vn_hv_kv, "vn_lv_kv": ttab.vn_lv_kv, "vscr_percent": ur_2w[2],
                                       "vsc_percent": uk_2w[2], "pfe_kw": 0, "i0_percent": 0,
                                       "tp_side": taps[2]["tp_side"], "tp_mid": taps[2]["tp_mid"],
                                       "tp_max": taps[2]["tp_max"], "tp_min": taps[2]["tp_min"],
                                       "tp_pos": taps[2]["tp_pos"], "tp_st_percent": taps[2]["tp_st_percent"],
                                       "in_service": ttab.in_service, "shift_degree":  ttab.shift_lv_degree}
        i += 1
    trafo_df = pd.DataFrame(trafos2w).T
    for var in list(tap_variables) + ["i0_percent", "sn_kva", "vsc_percent", "vscr_percent",
                                      "vn_hv_kv", "vn_lv_kv", "pfe_kw"]:
        trafo_df[var] = pd.to_numeric(trafo_df[var])
    return trafo_df


def _calc_impedance_parameter(net, bus_lookup):
    t = np.zeros(shape=(len(net["impedance"].index), 5), dtype=np.complex128)
    t[:, 0] = get_indices(net["impedance"]["from_bus"].values, bus_lookup)
    t[:, 1] = get_indices(net["impedance"]["to_bus"].values, bus_lookup)
    t[:, 2] = net["impedance"]["r_pu"] / net["impedance"]["sn_kva"] * 1000.
    t[:, 3] = net["impedance"]["x_pu"] / net["impedance"]["sn_kva"] * 1000.
    t[:, 4] = net["impedance"]["in_service"].values
    return t


def _calc_xward_parameter(net, ppc, is_elems, bus_lookup):
    bus_is = is_elems['bus']
    baseR = np.square(get_values(ppc["bus"][:, BASE_KV], net["xward"]["bus"].values, bus_lookup))
    t = np.zeros(shape=(len(net["xward"].index), 5), dtype=np.complex128)
    xw_is = np.in1d(net["xward"].bus.values, bus_is.index) \
        & net["xward"].in_service.values.astype(bool)
    t[:, 0] = get_indices(net["xward"]["bus"].values, bus_lookup)
    t[:, 1] = get_indices(net["xward"]["ad_bus"].values, bus_lookup)
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


def _switch_branches(n, ppc, is_elems, bus_lookup):
    """
    Updates the ppc["branch"] matrix with the changed from or to values
    according of the status of switches

    **INPUT**:
        **pd_net** - The Pandapower format network

        **ppc** - The PYPOWER format network to fill in values
    """

    # get in service elements
    lines_is = is_elems['line']
    bus_is = is_elems['bus']

    # opened bus line switches
    slidx = (n["switch"]["closed"].values == 0) \
        & (n["switch"]["et"].values == "l")

    # check if there are multiple opened switches at a line (-> set line out of service)
    sw_elem = n['switch'].ix[slidx].element
    m = np.zeros_like(sw_elem, dtype=bool)
    m[np.unique(sw_elem, return_index=True)[1]] = True

    # if non unique elements are in sw_elem (= multiple opened bus line switches)
    if np.count_nonzero(m) < len(sw_elem):
        # set branch in ppc out of service
        # get from and to buses of these branches
        ppc_from = get_indices(lines_is.ix[sw_elem[~m]].from_bus, bus_lookup)
        ppc_to = get_indices(lines_is.ix[sw_elem[~m]].to_bus, bus_lookup)
        ppc_idx = np.in1d(ppc['branch'][:, 0], ppc_from)\
            & np.in1d(ppc['branch'][:, 1], ppc_to)
        ppc["branch"][ppc_idx, BR_STATUS] = 0

        # drop from in service lines as well
        lines_is = lines_is.drop(sw_elem[~m])

    # opened switches at in service lines
    slidx = slidx\
        & (np.in1d(n["switch"]["element"].values, lines_is.index)) \
        & (np.in1d(n["switch"]["bus"].values, bus_is.index))
    nlo = np.count_nonzero(slidx)

    stidx = (n.switch["closed"].values == 0) & (n.switch["et"].values == "t")
    nto = np.count_nonzero(stidx)

    if (nlo + nto) > 0:
        n_bus = len(ppc["bus"])

        if nlo:
            future_buses = [ppc["bus"]]
            line_switches = n["switch"].loc[slidx]

            # determine on which side the switch is located
            mapfunc = partial(_gather_branch_switch_info, branch_type="l", net=n)
            ls_info = list(map(mapfunc,
                               line_switches["bus"].values,
                               line_switches["element"].values))
            # we now have the following matrix
            # 0: 1 if switch is at to_bus, 0 else
            # 1: bus of the switch
            # 2: position of the line a switch is connected to
            ls_info = np.array(ls_info, dtype=int)

            # build new buses
            new_ls_buses = np.zeros(shape=(nlo, 13), dtype=float)
            new_indices = np.arange(n_bus, n_bus + nlo)
            # the newly created buses
            new_ls_buses[:] = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1.1, 0.9])
            new_ls_buses[:, 0] = new_indices
            new_ls_buses[:, 9] = get_values(ppc["bus"][:, BASE_KV], ls_info[:, 1], bus_lookup)
#             set voltage of new buses to voltage on other branch end
            to_buses = ppc["branch"][ls_info[ls_info[:, 0].astype(bool), 2], 1].real.astype(int)
            from_buses = ppc["branch"][ls_info[np.logical_not(ls_info[:, 0]), 2], 0].real\
                .astype(int)
            if len(to_buses):
                ix = ls_info[:, 0] == 1
                new_ls_buses[ix, 7] = ppc["bus"][to_buses, 7]
                new_ls_buses[ix, 8] = ppc["bus"][to_buses, 8]
            if len(from_buses):
                ix = ls_info[:, 0] == 0
                new_ls_buses[ix, 7] = ppc["bus"][from_buses, 7]
                new_ls_buses[ix, 8] = ppc["bus"][from_buses, 8]

            future_buses.append(new_ls_buses)

            # re-route the end of lines to a new bus
            ppc["branch"][ls_info[ls_info[:, 0].astype(bool), 2], 1] = \
                new_indices[ls_info[:, 0].astype(bool)]
            ppc["branch"][ls_info[np.logical_not(ls_info[:, 0]), 2], 0] = \
                new_indices[np.logical_not(ls_info[:, 0])]

            ppc["bus"] = np.vstack(future_buses)

        if nto:
            future_buses = [ppc["bus"]]
            trafo_switches = n["switch"].loc[stidx]

            # determine on which side the switch is located
            mapfunc = partial(_gather_branch_switch_info, branch_type="t", net=n)
            ts_info = list(map(mapfunc,
                               trafo_switches["bus"].values,
                               trafo_switches["element"].values))
            # we now have the following matrix
            # 0: 1 if switch is at lv_bus, 0 else
            # 1: bus of the switch
            # 2: position of the trafo a switch is connected to
            ts_info = np.array(ts_info, dtype=int)

            # build new buses
            new_ts_buses = np.zeros(shape=(nto, 13), dtype=float)
            new_indices = np.arange(n_bus + nlo, n_bus + nlo + nto)
            new_ts_buses[:] = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1.1, 0.9])
            new_ts_buses[:, 0] = new_indices
            new_ts_buses[:, 9] = get_values(ppc["bus"][:, BASE_KV],
                                            ts_info[:, 1], bus_lookup)
            # set voltage of new buses to voltage on other branch end
            to_buses = ppc["branch"][ts_info[ts_info[:, 0].astype(bool), 2], 1].real.astype(int)
            from_buses = ppc["branch"][ts_info[np.logical_not(ts_info[:, 0]), 2], 0].real\
                .astype(int)

            # set newly created buses to voltage on other side of
            if len(to_buses):
                ix = ts_info[:, 0] == 1
                taps = ppc["branch"][ts_info[ts_info[:, 0].astype(bool), 2], 8].real
                shift = ppc["branch"][ts_info[ts_info[:, 0].astype(bool), 2], 9].real
                new_ts_buses[ix, 7] = ppc["bus"][to_buses, 7] * taps
                new_ts_buses[ix, 8] = ppc["bus"][to_buses, 8] + shift
            if len(from_buses):
                ix = ts_info[:, 0] == 0
                taps = ppc["branch"][ts_info[np.logical_not(ts_info[:, 0]), 2], 8].real
                shift = ppc["branch"][ts_info[np.logical_not(ts_info[:, 0]), 2], 9].real
                new_ts_buses[ix, 7] = ppc["bus"][from_buses, 7] * taps
                new_ts_buses[ix, 8] = ppc["bus"][from_buses, 8] + shift

            future_buses.append(new_ts_buses)

            # re-route the hv/lv-side of the trafo to a new bus
            # (trafo entries follow line entries)
            at_lv_bus = ts_info[:, 0].astype(bool)
            at_hv_bus = ~at_lv_bus
            ppc["branch"][len(n.line) + ts_info[at_lv_bus, 2], 1] = \
                new_indices[at_lv_bus]
            ppc["branch"][len(n.line) + ts_info[at_hv_bus, 2], 0] = \
                new_indices[at_hv_bus]

            ppc["bus"] = np.vstack(future_buses)


def _branches_with_oos_buses(n, ppc, is_elems, bus_lookup):
    """
    Updates the ppc["branch"] matrix with the changed from or to values
    if the branch is connected to an out of service bus

    Adds auxiliary buses if branch is connected to an out of service bus
    Sets branch out of service if connected to two out of service buses

    **INPUT**:
        **n** - The Pandapower format network

        **ppc** - The PYPOWER format network to fill in values
        **bus_is** - The in service buses
        **bus_lookup** - Lookup table for pandapower indices -> ppc indices
    """

    # get in service elements
    bus_is = is_elems['bus']
    line_is = is_elems['line']

    n_oos_buses = len(n['bus']) - len(bus_is)

    # only filter lines at oos buses if oos buses exists
    if n_oos_buses > 0:
        n_bus = len(ppc["bus"])
        future_buses = [ppc["bus"]]
        # out of service buses
        bus_oos = n['bus'].drop(bus_is.index).index
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
            ls_info[:, 2] = np.nonzero(np.in1d(n['line'].index, line_is.index[mask_or]))[0]

            # ls_info = list(map(mapfunc,
            #               line_switches["bus"].values,
            #               line_switches["element"].values))
            # we now have the following matrix
            # 0: 1 if switch is at to_bus, 0 else
            # 1: bus of the switch
            # 2: position of the line a switch is connected to
            # ls_info = np.array(ls_info, dtype=int)

            # build new buses
            new_ls_buses = np.zeros(shape=(n_oos_buses_at_lines, 13), dtype=float)
            new_indices = np.arange(n_bus, n_bus + n_oos_buses_at_lines)
            # the newly created buses
            new_ls_buses[:] = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1.1, 0.9])
            new_ls_buses[:, 0] = new_indices
            new_ls_buses[:, 9] = get_values(ppc["bus"][:, BASE_KV], ls_info[:, 1], bus_lookup)

            future_buses.append(new_ls_buses)

            # re-route the end of lines to a new bus
            ppc["branch"][ls_info[ls_info[:, 0].astype(bool), 2], 1] = \
                new_indices[ls_info[:, 0].astype(bool)]
            ppc["branch"][ls_info[np.logical_not(ls_info[:, 0]), 2], 0] = \
                new_indices[np.logical_not(ls_info[:, 0])]

            ppc["bus"] = np.vstack(future_buses)
