# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import copy
import math
from functools import partial
import warnings

import numpy as np
import pandas as pd

from pandapower.auxiliary import get_values
from pandapower.pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, BR_G, TAP, SHIFT, BR_STATUS, RATE_A, \
    BR_R_ASYM, BR_X_ASYM, BR_G_ASYM, BR_B_ASYM, branch_cols
from pandapower.pypower.idx_brch_dc import branch_dc_cols, DC_RATE_A, DC_RATE_B, DC_RATE_C, DC_BR_STATUS, DC_F_BUS, \
    DC_T_BUS, DC_BR_R, DC_BR_G
from pandapower.pypower.idx_brch_tdpf import BR_R_REF_OHM_PER_KM, BR_LENGTH_KM, RATE_I_KA, T_START_C, R_THETA, \
    WIND_SPEED_MPS, ALPHA, TDPF, OUTER_DIAMETER_M, MC_JOULE_PER_M_K, WIND_ANGLE_DEGREE, SOLAR_RADIATION_W_PER_SQ_M, \
    GAMMA, EPSILON, T_AMBIENT_C, T_REF_C, branch_cols_tdpf
from pandapower.pypower.idx_brch_sc import branch_cols_sc
from pandapower.pypower.idx_bus import BASE_KV, VM, VA, BUS_TYPE, BUS_AREA, ZONE, VMAX, VMIN, PQ
from pandapower.pypower.idx_bus_dc import DC_BUS_AREA, DC_VM, DC_ZONE, DC_VMAX, DC_VMIN, DC_P, DC_BASE_KV, DC_BUS_TYPE
from pandapower.pypower.idx_bus_sc import C_MIN, C_MAX
from pandapower.pypower.idx_tcsc import TCSC_F_BUS, TCSC_T_BUS, TCSC_X_L, TCSC_X_CVAR, TCSC_SET_P, \
    TCSC_THYRISTOR_FIRING_ANGLE, TCSC_STATUS, TCSC_CONTROLLABLE, tcsc_cols, TCSC_MIN_FIRING_ANGLE, TCSC_MAX_FIRING_ANGLE


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
    tdpf = net._options["tdpf"]
    if mode == "sc" and tdpf:
        raise NotImplementedError("indexing for ppc branch columns not implemented for tdpf and sc together")
    # initialize "normal" ppc branch
    all_branch_columns = branch_cols_tdpf + branch_cols if tdpf else branch_cols
    ppc["branch"] = np.zeros(shape=(length, all_branch_columns), dtype=np.float64)
    # add optional columns for short-circuit calculation
    # Check if this should be moved to somewhere else
    if mode == "sc":
        branch_sc = np.empty(shape=(length, branch_cols_sc), dtype=np.float64)
        branch_sc.fill(np.nan)
        ppc["branch"] = np.hstack((ppc["branch"], branch_sc))
    ppc["branch"][:, :13] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, -360, 360])
    if "line" in lookup:
        _calc_line_parameter(net, ppc)
    if "trafo" in lookup:
        _calc_trafo_parameter(net, ppc)
    if "trafo3w" in lookup:
        _calc_trafo3w_parameter(net, ppc)
    if "impedance" in lookup:
        _calc_impedance_parameter(net, ppc)
    if "xward" in lookup:
        _calc_xward_parameter(net, ppc)
    if "switch" in lookup:
        _calc_switch_parameter(net, ppc)


def _build_branch_dc_ppc(net, ppc):
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
    mode = net._options["mode"]
    length = _initialize_branch_lookup(net, dc=True)
    lookup = net._pd2ppc_lookups["branch_dc"]
    tdpf = net._options["tdpf"]
    # todo: check how it works when calculating SC
    # if mode == "sc":
    #     raise NotImplementedError("indexing for ppc branch columns not implemented for tdpf and sc together")
    # initialize "normal" ppc branch
    all_branch_columns = branch_cols_tdpf + branch_dc_cols if tdpf else branch_dc_cols
    ppc["branch_dc"] = np.zeros(shape=(length, all_branch_columns), dtype=np.float64)
    ppc["branch_dc"][:, [DC_RATE_A, DC_RATE_B, DC_RATE_C, DC_BR_STATUS]] = np.array([250, 250, 250, 1])
    if mode != "pf":
        return
    if "line_dc" in lookup:
        _calc_line_dc_parameter(net, ppc)


def _build_tcsc_ppc(net, ppc, mode):
    length = len(net.tcsc)
    ppc["tcsc"] = np.zeros(shape=(length, tcsc_cols), dtype=np.float64)
    if mode != "pf":
        return

    if length > 0:
        _calc_tcsc_parameter(net, ppc)


def _initialize_branch_lookup(net, dc=False):
    start = 0
    end = 0
    table = "branch" if not dc else "branch_dc"
    net._pd2ppc_lookups[table] = {}
    elements = ["line", "trafo", "trafo3w", "impedance", "xward"] if not dc else ["line_dc"]
    for element in elements:
        if len(net[element]) > 0:
            if element == "trafo3w":
                end = start + len(net[element]) * 3
            else:
                end = start + len(net[element])
            net._pd2ppc_lookups[table][element] = (start, end)
            start = end
    if "_impedance_bb_switches" in net and net._impedance_bb_switches.any() and not dc:
        end = start + net._impedance_bb_switches.sum()
        net._pd2ppc_lookups[table]["switch"] = (start, end)
    return end


def _calc_trafo3w_parameter(net, ppc):
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    branch = ppc["branch"]
    f, t = net["_pd2ppc_lookups"]["branch"]["trafo3w"]
    trafo_df = _trafo_df_from_trafo3w(net)
    hv_bus = get_trafo_values(trafo_df, "hv_bus").astype(np.int64)
    lv_bus = get_trafo_values(trafo_df, "lv_bus").astype(np.int64)
    in_service = get_trafo_values(trafo_df, "in_service").astype(np.int64)
    branch[f:t, F_BUS] = bus_lookup[hv_bus]
    branch[f:t, T_BUS] = bus_lookup[lv_bus]
    r, x, g, b, g_asym, b_asym, ratio, shift = _calc_branch_values_from_trafo_df(net, ppc, trafo_df)
    branch[f:t, BR_R] = r
    branch[f:t, BR_X] = x
    branch[f:t, BR_G] = g
    branch[f:t, BR_B] = b
    branch[f:t, BR_G_ASYM] = g_asym
    branch[f:t, BR_B_ASYM] = b_asym
    branch[f:t, TAP] = ratio
    branch[f:t, SHIFT] = shift
    branch[f:t, BR_STATUS] = in_service
    # always set RATE_A for completeness
    # RATE_A is considered by the (PowerModels) OPF. If zero -> unlimited
    if "max_loading_percent" in trafo_df:
        max_load = get_trafo_values(trafo_df, "max_loading_percent")
        sn_mva = get_trafo_values(trafo_df, "sn_mva")
        branch[f:t, RATE_A] = max_load / 100. * sn_mva
    else:
        # PowerModels considers "0" as "no limit"
        # todo: inf and convert only when using PowerModels to 0., pypower opf converts the zero to inf
        branch[f:t, RATE_A] = 0. if net["_options"]["mode"] == "opf" else 100.


def _calc_line_parameter(net, ppc, elm="line", ppc_elm="branch"):
    """
    calculates the line parameter in per unit.

    **INPUT**:
        **net** - The pandapower format network

        **ppc** - the ppc array

    **OPTIONAL**:
        **elm** - The pandapower element (normally "line")

        **ppc_elm** - The ppc element (normally "branch")

    **RETURN**:
        **t** - Temporary line parameter. Which is a complex128
                Numpy array. with the following order:
                0:bus_a; 1:bus_b; 2:r_pu; 3:x_pu; 4:b_pu
    """
    f, t = net._pd2ppc_lookups[ppc_elm][elm]
    branch = ppc[ppc_elm]
    mode = net["_options"]["mode"]
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    line = net[elm]
    from_bus = bus_lookup[line["from_bus"].values.astype(np.int64)]
    to_bus = bus_lookup[line["to_bus"].values.astype(np.int64)]
    length_km = line["length_km"].values
    parallel = line["parallel"].values
    base_kv = ppc["bus"][from_bus, BASE_KV]
    baseR = np.square(base_kv) / (3 * net.sn_mva) if mode == "pf_3ph" else np.square(
        base_kv) / net.sn_mva

    branch[f:t, F_BUS] = from_bus
    branch[f:t, T_BUS] = to_bus
    branch[f:t, BR_R] = line["r_ohm_per_km"].values * length_km / baseR / parallel
    branch[f:t, BR_X] = line["x_ohm_per_km"].values * length_km / baseR / parallel

    if net._options["tdpf"]:
        branch[f:t, TDPF] = line["in_service"].values & line["tdpf"].fillna(False).values.astype(bool)
        branch[f:t, BR_R_REF_OHM_PER_KM] = line["r_ohm_per_km"].values / parallel
        branch[f:t, BR_LENGTH_KM] = length_km
        branch[f:t, RATE_I_KA] = line["max_i_ka"].values * line["df"].values * parallel
        branch[f:t, T_START_C] = line["temperature_degree_celsius"].values
        branch[f:t, T_REF_C] = line["reference_temperature_degree_celsius"].values
        branch[f:t, T_AMBIENT_C] = line["air_temperature_degree_celsius"].values
        branch[f:t, ALPHA] = line["alpha"].values
        branch[f:t, WIND_SPEED_MPS] = line.get("wind_speed_m_per_s", default=np.nan)
        branch[f:t, WIND_ANGLE_DEGREE] = line.get("wind_angle_degree", default=np.nan)
        branch[f:t, SOLAR_RADIATION_W_PER_SQ_M] = line.get("solar_radiation_w_per_sq_m", default=np.nan)
        branch[f:t, GAMMA] = line.get("solar_absorptivity", default=np.nan)
        branch[f:t, EPSILON] = line.get("emissivity", default=np.nan)
        branch[f:t, R_THETA] = line.get("r_theta_kelvin_per_mw", default=np.nan)
        branch[f:t, OUTER_DIAMETER_M] = line.get("conductor_outer_diameter_m", default=np.nan)
        branch[f:t, MC_JOULE_PER_M_K] = line.get("mc_joule_per_m_k", default=np.nan)

    if mode == "sc" and not net._options.get("use_pre_fault_voltage", False):
        # temperature correction
        if net["_options"]["case"] == "min":
            branch[f:t, BR_R] *= _end_temperature_correction_factor(net, short_circuit=True)
    else:
        # temperature correction
        if net["_options"]["consider_line_temperature"]:
            branch[f:t, BR_R] *= _end_temperature_correction_factor(net)

        b = 2 * net.f_hz * math.pi * line["c_nf_per_km"].values * 1e-9 * baseR * length_km * parallel
        g = line["g_us_per_km"].values * 1e-6 * baseR * length_km * parallel
        branch[f:t, BR_B] = b
        branch[f:t, BR_G] = g

    # in service of lines
    branch[f:t, BR_STATUS] = line["in_service"].values
    # always set RATE_A for completeness:
    # RATE_A is considered by the (PowerModels) OPF. If zero -> unlimited
    if "max_loading_percent" in line:
        max_load = line.max_loading_percent.values
        vr = net.bus.loc[line["from_bus"].values, "vn_kv"].values * np.sqrt(3.)
        max_i_ka = line.max_i_ka.values
        df = line.df.values
        branch[f:t, RATE_A] = max_load / 100. * max_i_ka * df * parallel * vr
    else:
        # PowerModels considers "0" as "no limit"
        # todo: inf and convert only when using PowerModels to 0., pypower opf converts the zero to inf
        branch[f:t, RATE_A] = 0. if mode == "opf" else 100.


def _calc_line_dc_parameter(net, ppc, elm="line_dc", ppc_elm="branch_dc"):
    """
    calculates the line_dc parameter in per unit.

    **INPUT**:
        **net** - The pandapower format network

        **ppc** - the ppc array

    **OPTIONAL**:
        **elm** - The pandapower element (normally "line_dc")

        **ppc_elm** - The ppc element (normally "branch_dc")

    **RETURN**:
        **t** - Temporary line_dc parameter. Which is a complex128
                Numpy array. with the following order:
                0:bus_a; 1:bus_b; 2:r_pu; 3:x_pu; 4:b_pu
    """
    f, t = net._pd2ppc_lookups[ppc_elm][elm]
    branch_dc = ppc[ppc_elm]
    mode = net["_options"]["mode"]
    bus_lookup = net["_pd2ppc_lookups"]["bus_dc"]
    line_dc = net[elm]
    from_bus_dc = bus_lookup[line_dc["from_bus_dc"].values.astype(np.int64)]
    to_bus_dc = bus_lookup[line_dc["to_bus_dc"].values.astype(np.int64)]
    length_km = line_dc["length_km"].values
    parallel = line_dc["parallel"].values
    base_kv = ppc["bus_dc"][from_bus_dc, DC_BASE_KV]
    baseR = np.square(base_kv) / net.sn_mva

    branch_dc[f:t, DC_F_BUS] = from_bus_dc
    branch_dc[f:t, DC_T_BUS] = to_bus_dc
    branch_dc[f:t, DC_BR_R] = line_dc["r_ohm_per_km"].values * length_km / baseR / parallel

    if net._options["tdpf"]:
        # todo implement idx_brch_dc_tdpf
        raise NotImplementedError("temperature-related calculation for line_dc not implemented")
        # branch_dc[f:t, TDPF] = line_dc["in_service"].values & line_dc["tdpf"].fillna(False).values.astype(bool)
        # branch_dc[f:t, BR_R_REF_OHM_PER_KM] = line_dc["r_ohm_per_km"].values / parallel
        # branch_dc[f:t, BR_LENGTH_KM] = length_km
        # branch_dc[f:t, RATE_I_KA] = line_dc["max_i_ka"].values * line_dc["df"].values * parallel
        # branch_dc[f:t, T_START_C] = line_dc["temperature_degree_celsius"].values
        # branch_dc[f:t, T_REF_C] = line_dc["reference_temperature_degree_celsius"].values
        # branch_dc[f:t, T_AMBIENT_C] = line_dc["air_temperature_degree_celsius"].values
        # branch_dc[f:t, ALPHA] = line_dc["alpha"].values
        # branch_dc[f:t, WIND_SPEED_MPS] = line_dc.get("wind_speed_m_per_s", default=np.nan)
        # branch_dc[f:t, WIND_ANGLE_DEGREE] = line_dc.get("wind_angle_degree", default=np.nan)
        # branch_dc[f:t, SOLAR_RADIATION_W_PER_SQ_M] = line_dc.get("solar_radiation_w_per_sq_m", default=np.nan)
        # branch_dc[f:t, GAMMA] = line_dc.get("solar_absorptivity", default=np.nan)
        # branch_dc[f:t, EPSILON] = line_dc.get("emissivity", default=np.nan)
        # branch_dc[f:t, R_THETA] = line_dc.get("r_theta_kelvin_per_mw", default=np.nan)
        # branch_dc[f:t, OUTER_DIAMETER_M] = line_dc.get("conductor_outer_diameter_m", default=np.nan)
        # branch_dc[f:t, MC_JOULE_PER_M_K] = line_dc.get("mc_joule_per_m_k", default=np.nan)

    # temperature correction
    if net["_options"]["consider_line_temperature"]:
        branch_dc[f:t, DC_BR_R] *= _end_temperature_correction_factor(net)

    # todo check if DC line_dc model has shunt components
    # b = 2 * net.f_hz * math.pi * line_dc["c_nf_per_km"].values * 1e-9 * baseR * length_km * parallel
    g = line_dc["g_us_per_km"].values * 1e-6 * baseR * length_km * parallel
    branch_dc[f:t, DC_BR_G] = g

    # in service of lines
    branch_dc[f:t, DC_BR_STATUS] = line_dc["in_service"].values
    # always set RATE_A for completeness:
    # RATE_A is considered by the (PowerModels) OPF. If zero -> unlimited
    max_load = line_dc.max_loading_percent.values if "max_loading_percent" in line_dc else 0.
    vr = net.bus_dc.loc[line_dc["from_bus_dc"].values, "vn_kv"].values * np.sqrt(3.)
    max_i_ka = line_dc.max_i_ka.values
    df = line_dc.df.values
    # This calculates the maximum apparent power at 1.0 p.u.
    branch_dc[f:t, DC_RATE_A] = max_load / 100. * max_i_ka * df * parallel * vr
    # RATE_A is considered by the (PowerModels) OPF. If zero -> unlimited
    if "max_loading_percent" in line_dc:
        max_load = line_dc.max_loading_percent.values
        vr = net.bus_dc.loc[line_dc["from_bus_dc"].values, "vn_kv"].values * np.sqrt(3.)
        max_i_ka = line_dc.max_i_ka.values
        df = line_dc.df.values
        # This calculates the maximum apparent power at 1.0 p.u.
        branch_dc[f:t, DC_RATE_A] = max_load / 100. * max_i_ka * df * parallel * vr
    else:
        # PowerModels considers "0" as "no limit"
        # todo: inf and convert only when using PowerModels to 0., pypower opf converts the zero to inf
        branch_dc[f:t, DC_RATE_A] = 0. if mode == "opf" else 100.


def _calc_trafo_parameter(net, ppc):
    """
    Calculates the transformer parameter in per unit.

    **INPUT**:
        **net** - The pandapower format network

    **RETURN**:
        **temp_para** -
        Temporary transformer parameter. Which is a np.complex128
        Numpy array. with the following order:
        0:hv_bus; 1:lv_bus; 2:r_pu; 3:x_pu; 4:b_pu; 5:tab, 6:shift
    """

    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    f, t = net["_pd2ppc_lookups"]["branch"]["trafo"]
    branch = ppc["branch"]
    trafo = net["trafo"]
    parallel = trafo["parallel"].values
    branch[f:t, F_BUS] = bus_lookup[trafo["hv_bus"].values]
    branch[f:t, T_BUS] = bus_lookup[trafo["lv_bus"].values]
    r, x, g, b, g_asym, b_asym, ratio, shift = _calc_branch_values_from_trafo_df(
        net, ppc)
    branch[f:t, BR_R] = r
    branch[f:t, BR_X] = x
    branch[f:t, BR_G] = g
    branch[f:t, BR_B] = b
    branch[f:t, BR_G_ASYM] = g_asym
    branch[f:t, BR_B_ASYM] = b_asym
    branch[f:t, TAP] = ratio
    branch[f:t, SHIFT] = shift
    branch[f:t, BR_STATUS] = trafo["in_service"].values
    if any(trafo.df.values <= 0):
        raise UserWarning("Rating factor df must be positive. Transformers with false "
                          "rating factors: %s" % trafo.query('df<=0').index.tolist())
    # always set RATE_A for completeness
    # RATE_A is considered by the (PowerModels) OPF. If zero -> unlimited
    if "max_loading_percent" in trafo:
        max_load = trafo.max_loading_percent.values
        sn_mva = trafo.sn_mva.values
        df = trafo.df.values
        branch[f:t, RATE_A] = max_load / 100. * sn_mva * df * parallel
    else:
        # PowerModels considers "0" as "no limit"
        # todo: inf and convert only when using PowerModels to 0., pypower opf converts the zero to inf
        branch[f:t, RATE_A] = 0. if net["_options"]["mode"] == "opf" else 100.


def get_trafo_values(trafo_df, par):
    if isinstance(trafo_df, dict):
        return trafo_df[par]
    else:
        return trafo_df[par].values


def _calc_branch_values_from_trafo_df(net, ppc, trafo_df=None, sequence=1):
    """
    Calculates the MAT/PYPOWER-branch-attributes from the pandapower trafo dataframe.

    PYPOWER and MATPOWER uses the PI-model to model transformers.
    This function calculates the resistance r, reactance x, complex susceptance c and the tap ratio
    according to the given parameters.

    .. warning:: This function returns the susceptance b as a complex number
        **(-img + -re*i)**. MAT/PYPOWER is only intended to calculate the
        imaginary part of the susceptance. However, internally c is
        multiplied by i. By using susceptance in this way, it is possible
        to consider the ferromagnetic loss of the coil. Which would
        otherwise be neglected.


    .. warning:: Tab switches effect calculation as following:
        On **high-voltage** side(=1) -> only **tab** gets adapted.
        On **low-voltage** side(=2) -> **tab, x, r** get adapted.
        This is consistent with Sincal.
        The Sincal method in this case is questionable.


    **INPUT**:
        **pd_trafo** - The pandapower format Transformer Dataframe.
                        The Transformer model will only read from pd_net

    **RETURN**:
        **temp_para** - Temporary transformer parameter. Which is a complex128
                        Numpy array. with the following order:
                        0:r_pu; 1:x_pu; 2:b_pu; 3:tab;

    """
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    if trafo_df is None:
        trafo_df = net["trafo"]
    lv_bus = get_trafo_values(trafo_df, "lv_bus")
    vn_lv = ppc["bus"][bus_lookup[lv_bus], BASE_KV]
    ### Construct np.array to parse results in ###
    # 0:r_pu; 1:x_pu; 2:b_pu; 3:tab;
    vn_trafo_hv, vn_trafo_lv, shift = _calc_tap_from_dataframe(net, trafo_df)
    ratio = _calc_nominal_ratio_from_dataframe(ppc, trafo_df, vn_trafo_hv, vn_trafo_lv,
                                               bus_lookup)
    r, x, g, b, g_asym, b_asym = _calc_r_x_y_from_dataframe(net, trafo_df, vn_trafo_lv, vn_lv, ppc, sequence=sequence)
    return r, x, g, b, g_asym, b_asym, ratio, shift


def _calc_r_x_y_from_dataframe(net, trafo_df, vn_trafo_lv, vn_lv, ppc, sequence=1):
    mode = net["_options"]["mode"]
    trafo_model = net["_options"]["trafo_model"]
    if 'tap_dependency_table' in trafo_df:
        if 'trafo_characteristic_table' in net:
            r, x = _calc_r_x_from_dataframe(
                mode, trafo_df, vn_lv, vn_trafo_lv, net.sn_mva, sequence=sequence,
                trafo_characteristic_table=net.trafo_characteristic_table)
        else:
            r, x = _calc_r_x_from_dataframe(mode, trafo_df, vn_lv, vn_trafo_lv, net.sn_mva,
                                            sequence=sequence)
    else:
        warnings.warn(DeprecationWarning("tap_dependency_table is missing in net, which is most probably due to "
                                         "unsupported net data. tap_dependency_table was introduced with "
                                         "pandapower 3.0 and replaced spline characteristics. Spline "
                                         "characteristics will still work, but they are deprecated and will be "
                                         "removed in future releases."))
        r, x = _calc_r_x_from_dataframe(mode, trafo_df, vn_lv, vn_trafo_lv, net.sn_mva,
                                        sequence=sequence, characteristic=net.get("characteristic"))

    if mode == "sc":
        if net._options.get("use_pre_fault_voltage", False):
            g, b = _calc_y_from_dataframe(mode, trafo_df, vn_lv, vn_trafo_lv, net.sn_mva)
        else:
            g, b = 0, 0  # why for sc are we assigning y directly as 0?
        if isinstance(trafo_df, pd.DataFrame):  # 2w trafo is dataframe, 3w trafo is dict
            bus_lookup = net._pd2ppc_lookups["bus"]
            cmax = ppc["bus"][bus_lookup[net.trafo.lv_bus.values], C_MAX]
            # todo: kt is only used for case = max and only for network transformers! (IEC 60909-0:2016 section 6.3.3)
            # kt is only calculated for network transformers (IEC 60909-0:2016 section 6.3.3)
            if not net._options.get("use_pre_fault_voltage", False):
                kt = _transformer_correction_factor(
                    trafo_df, trafo_df.vk_percent, trafo_df.vkr_percent, trafo_df.sn_mva, cmax)
                r *= kt
                x *= kt
    else:
        g, b = _calc_y_from_dataframe(mode, trafo_df, vn_lv, vn_trafo_lv, net.sn_mva)

    if trafo_model == "pi":
        return r, x, g, b, 0, 0  # g_asym and b_asym are 0 here
    elif trafo_model == "t":
        r_ratio = get_trafo_values(trafo_df, "leakage_resistance_ratio_hv") \
            if "leakage_resistance_ratio_hv" in trafo_df else np.full_like(r, fill_value=0.5, dtype=np.float64)
        x_ratio = get_trafo_values(trafo_df, "leakage_reactance_ratio_hv") \
            if "leakage_reactance_ratio_hv" in trafo_df else np.full_like(r, fill_value=0.5, dtype=np.float64)
        return _wye_delta(r, x, g, b, r_ratio, x_ratio)
    else:
        raise ValueError("Unknown Transformer Model %s - valid values ar 'pi' or 't'" % trafo_model)


@np.errstate(all="raise")
def _wye_delta(r, x, g, b, r_ratio, x_ratio):
    """
    20.05.2016 added by Lothar Löwer

    Calculate transformer Pi-Data based on T-Data

    """
    tidx = (g != 0) | (b != 0)
    za_star = r[tidx] * r_ratio[tidx] + x[tidx] * x_ratio[tidx] * 1j
    zb_star = r[tidx] * (1 - r_ratio[tidx]) + x[tidx] * (1 - x_ratio[tidx]) * 1j
    zc_star = 1 / (g + 1j*b)[tidx]
    zSum_triangle = za_star * zb_star + za_star * zc_star + zb_star * zc_star
    zab_triangle = zSum_triangle / zc_star
    zac_triangle = zSum_triangle / zb_star
    zbc_triangle = zSum_triangle / za_star
    r[tidx] = zab_triangle.real
    x[tidx] = zab_triangle.imag
    yf = 1 / zac_triangle
    yt = 1 / zbc_triangle
    # 2 because in makeYbus Bcf, Bct are divided by 2:
    g[tidx] = yf.real * 2
    b[tidx] = yf.imag * 2
    g_asym = np.zeros_like(g)
    b_asym = np.zeros_like(b)
    g_asym[tidx] = 2 * yt.real - g[tidx]
    b_asym[tidx] = 2 * yt.imag - b[tidx]
    return r, x, g, b, g_asym, b_asym


def _calc_y_from_dataframe(mode, trafo_df, vn_lv, vn_trafo_lv, net_sn_mva):
    """
    Calculate the susceptance y from the transformer dataframe.

    INPUT:

        **trafo** (Dataframe) - The dataframe in net.trafo
        which contains transformer calculation values.

    OUTPUT:
        **susceptance** (1d array, np.complex128) - The susceptance in pu in
        the form (-b_img, -b_real)
    """

    baseZ = np.square(vn_lv) / (3*net_sn_mva) if mode == 'pf_3ph' else np.square(vn_lv) / net_sn_mva
    vn_lv_kv = get_trafo_values(trafo_df, "vn_lv_kv")
    pfe_mw = (get_trafo_values(trafo_df, "pfe_kw") * 1e-3) / 3 if mode == 'pf_3ph'\
        else get_trafo_values(trafo_df, "pfe_kw") * 1e-3
    parallel = get_trafo_values(trafo_df, "parallel")
    trafo_sn_mva = get_trafo_values(trafo_df, "sn_mva")

    ### Calculate susceptance ###
    vnl_squared = (vn_lv_kv ** 2)/3 if mode == 'pf_3ph' else vn_lv_kv ** 2
    g_mva = pfe_mw
    i0 = get_trafo_values(trafo_df, "i0_percent") / 3 if mode == 'pf_3ph'\
        else get_trafo_values(trafo_df, "i0_percent")

    ym_mva = i0 / 100 * trafo_sn_mva
    b_mva_squared = np.square(ym_mva) - np.square(pfe_mw)
    b_mva_squared[b_mva_squared < 0] = 0
    b_mva = -np.sqrt(b_mva_squared)

    g_pu = g_mva / vnl_squared * baseZ * parallel / np.square(vn_trafo_lv / vn_lv_kv)
    b_pu = b_mva / vnl_squared * baseZ * parallel / np.square(vn_trafo_lv / vn_lv_kv)
    return g_pu, b_pu


def _calc_tap_from_dataframe(net, trafo_df):
    """
    Adjust the nominal voltage vnh, vnl and phase shift to the active tab position "tap_pos".
    If "side" is 1 (high-voltage side) the high voltage vnh is adjusted.
    If "side" is 2 (low-voltage side) the low voltage vnl is adjusted.

    INPUT:
        **net** - The pandapower format network

        **trafo** (Dataframe) - The dataframe in pd_net["structure"]["trafo"]
        which contains transformer calculation values.

    OUTPUT:
        **vn_hv_kv** (1d array, float) - The adjusted high voltages

        **vn_lv_kv** (1d array, float) - The adjusted low voltages

        **trafo_shift** (1d array, float) - phase shift angle

    """
    calculate_voltage_angles = net["_options"]["calculate_voltage_angles"]
    mode = net["_options"]["mode"]
    vnh = copy.copy(get_trafo_values(trafo_df, "vn_hv_kv").astype(float))
    vnl = copy.copy(get_trafo_values(trafo_df, "vn_lv_kv").astype(float))
    trafo_shift = get_trafo_values(trafo_df, "shift_degree").astype(float) if calculate_voltage_angles else \
        np.zeros(len(vnh))
    if mode == "sc" and not net._options.get("use_pre_fault_voltage", False):  # todo type c?
        return vnh, vnl, trafo_shift

    for t in ("", "2"):
        if f"tap{t}_pos" not in trafo_df:
            continue
        tap_pos = get_trafo_values(trafo_df, f"tap{t}_pos")
        tap_neutral = get_trafo_values(trafo_df, f"tap{t}_neutral")
        tap_diff = tap_pos - tap_neutral
        tap_side = get_trafo_values(trafo_df, f"tap{t}_side")
        tap_step_percent = get_trafo_values(trafo_df, f"tap{t}_step_percent")
        tap_step_degree = get_trafo_values(trafo_df, f"tap{t}_step_degree")

        cos = lambda x: np.cos(np.deg2rad(x))
        sin = lambda x: np.sin(np.deg2rad(x))
        arctan = lambda x: np.rad2deg(np.arctan(x))

        if f'tap{t}_changer_type' in trafo_df:
            # tap_changer_type is only in dataframe starting from pp Version 3.0, older version use different logic
            tap_changer_type = get_trafo_values(trafo_df, f"tap{t}_changer_type")
            if f'tap{t}_dependency_table' in trafo_df:
                tap_dependency_table = get_trafo_values(trafo_df, "tap_dependency_table")
                tap_dependency_table = np.array(
                    [False if isinstance(x, float) and np.isnan(x) else x for x in tap_dependency_table])
            else:
                tap_table = np.array([False])
                tap_dependency_table = np.array([False])
            # tap_changer_type = pd.Series(tap_changer_type)
            tap_table = np.logical_and(tap_dependency_table, np.logical_not(tap_changer_type == "None"))
            tap_no_table = np.logical_and(~tap_dependency_table, np.logical_not(tap_changer_type == "None"))
            if any(tap_table):
                id_characteristic_table = get_trafo_values(trafo_df, "id_characteristic_table")
                if np.any(tap_dependency_table & pd.isna(id_characteristic_table)):
                    raise UserWarning(
                        "Trafo with tap_dependency_table True and id_characteristic_table NA detected.\n"
                        "Please set an id_characteristic_table or set tap_dependency_table to False.")
                for side, vn, direction in [("hv", vnh, 1), ("lv", vnl, -1)]:
                    mask = tap_table & (side == tap_side)
                    filter_df = pd.DataFrame({
                        'id_characteristic': id_characteristic_table,
                        'step': tap_pos,
                        'mask': mask
                    })

                    filtered_df = net.trafo_characteristic_table.merge(filter_df[filter_df['mask']],
                                                                       on=['id_characteristic', 'step'])

                    cleaned_id_characteristic = id_characteristic_table[(~pd.isna(id_characteristic_table)) & mask]

                    voltage_mapping = dict(zip(filtered_df['id_characteristic'], filtered_df['voltage_ratio']))
                    shift_mapping = dict(zip(filtered_df['id_characteristic'], filtered_df['angle_deg']))

                    if direction == 1:
                        ratio = [voltage_mapping.get(id_val, 1) for id_val in cleaned_id_characteristic]
                        shift = [shift_mapping.get(id_val, 1) for id_val in cleaned_id_characteristic]
                    else:
                        ratio = [voltage_mapping.get(id_val, 1) for id_val in cleaned_id_characteristic]
                        shift = [-shift_mapping.get(id_val, 1) for id_val in cleaned_id_characteristic]

                    vn[mask] = vn[mask] * ratio
                    trafo_shift[mask] += shift
            if any(tap_no_table):
                tap_ideal = np.logical_and(tap_changer_type == "Ideal", tap_no_table)
                tap_complex = np.logical_and(np.logical_or(tap_changer_type == "Ratio",
                                                           tap_changer_type == "Symmetrical"), tap_no_table)
                for side, vn, direction in [("hv", vnh, 1), ("lv", vnl, -1)]:
                    mask_ideal = (tap_ideal & (tap_side == side))
                    mask_complex = (tap_complex & (tap_side == side))
                    if any(mask_ideal):
                        degree_is_set = _replace_nan(tap_step_degree[mask_ideal]) != 0
                        percent_is_set = _replace_nan(tap_step_percent[mask_ideal]) != 0
                        if (degree_is_set & percent_is_set).any():
                            raise UserWarning(
                                "Both tap_step_degree and tap_step_percent set for ideal phase shifter")
                        trafo_shift[mask_ideal] += np.where(
                            degree_is_set,
                            (direction * tap_diff[mask_ideal] * tap_step_degree[mask_ideal]),
                            (direction * 2 * np.rad2deg(np.arcsin(tap_diff[mask_ideal] *
                                                                  tap_step_percent[mask_ideal] / 100 / 2)))
                        )
                    if any(mask_complex):
                        tap_steps = tap_step_percent[mask_complex] * tap_diff[mask_complex] / 100
                        tap_angles = _replace_nan(tap_step_degree[mask_complex])
                        u1 = vn[mask_complex]
                        du = u1 * _replace_nan(tap_steps)
                        vn[mask_complex] = np.sqrt((u1 + du * cos(tap_angles)) ** 2 + (du * sin(tap_angles)) ** 2)
                        trafo_shift[mask_complex] += (arctan(direction * du * sin(tap_angles) /
                                                      (u1 + du * cos(tap_angles))))
        elif f'tap{t}_phase_shifter' in trafo_df:
            warnings.warn(DeprecationWarning("tap{t}_phase_shifter was removed with pandapower 3.0 and replaced by "
                                             "tap{t}_changer_type. Using old net data will still work, but usage of "
                                             "tap{t}_phase_shifter is deprecated and will be removed in future "
                                             "releases."))
            tap_phase_shifter = get_trafo_values(trafo_df, f"tap{t}_phase_shifter")
            for side, vn, direction in [("hv", vnh, 1), ("lv", vnl, -1)]:
                tap_ideal = tap_phase_shifter & (tap_side == side)
                tap_complex = np.isfinite(tap_step_percent) & np.isfinite(tap_pos) & (tap_side == side) & \
                    ~tap_ideal
                if tap_complex.any():
                    tap_steps = tap_step_percent[tap_complex] * tap_diff[tap_complex] / 100
                    tap_angles = _replace_nan(tap_step_degree[tap_complex])
                    u1 = vn[tap_complex]
                    du = u1 * _replace_nan(tap_steps)
                    vn[tap_complex] = np.sqrt((u1 + du * cos(tap_angles)) ** 2 + (du * sin(tap_angles)) ** 2)
                    trafo_shift[tap_complex] += (arctan(direction * du * sin(tap_angles) /
                                                        (u1 + du * cos(tap_angles))))
                if tap_ideal.any():
                    degree_is_set = _replace_nan(tap_step_degree[tap_ideal]) != 0
                    percent_is_set = _replace_nan(tap_step_percent[tap_ideal]) != 0
                    if (degree_is_set & percent_is_set).any():
                        raise UserWarning(
                            "Both tap_step_degree and tap_step_percent set for ideal phase shifter")
                    trafo_shift[tap_ideal] += np.where(
                        degree_is_set,
                        (direction * tap_diff[tap_ideal] * tap_step_degree[tap_ideal]),
                        (direction * 2 * np.rad2deg(np.arcsin(tap_diff[tap_ideal] *
                                                              tap_step_percent[tap_ideal] / 100 / 2)))
                    )
    return vnh, vnl, trafo_shift


def _replace_nan(array, value=0):
    mask = np.isnan(array)
    array[mask] = value
    return array


def _get_vk_values_from_table(trafo_df, trafo_characteristic_table, trafotype="2W"):
    if trafotype == "2W":
        vk_variables = ("vk_percent", "vkr_percent")
    elif trafotype == "3W":
        vk_variables = ("vk_hv_percent", "vkr_hv_percent", "vk_mv_percent", "vkr_mv_percent",
                        "vk_lv_percent", "vkr_lv_percent")
    else:
        raise UserWarning("Unknown trafotype")

    tap_dependency_table = get_trafo_values(trafo_df, "tap_dependency_table")
    tap_dependency_table = np.array(
        [False if isinstance(x, float) and np.isnan(x) else x for x in tap_dependency_table])
    if np.any(np.isnan(tap_dependency_table)):
        raise UserWarning("tap_dependent_impedance has NaN values, but must be of type "
                          "bool and set to True or False")
    tap_pos = get_trafo_values(trafo_df, "tap_pos")

    vals = ()

    for _, vk_var in enumerate(vk_variables):
        vk_value = get_trafo_values(trafo_df, vk_var)
        if any(tap_dependency_table):
            id_characteristic_table = get_trafo_values(trafo_df, "id_characteristic_table")
            if np.any(tap_dependency_table & pd.isna(id_characteristic_table)):
                raise UserWarning(
                    "Trafo with tap_dependency_table True and id_characteristic_table NA detected.\n"
                    "Please set an id_characteristic_table or set tap_dependency_table to False.")
            mask = tap_dependency_table
            filter_df = pd.DataFrame({
                'id_characteristic': id_characteristic_table,
                'step': tap_pos,
                'mask': mask
            })

            filtered_df = trafo_characteristic_table.merge(filter_df[filter_df['mask']],
                                                           on=['id_characteristic', 'step'])
            cleaned_id_characteristic = id_characteristic_table[(~pd.isna(id_characteristic_table)) & mask]

            vk_mapping = dict(zip(filtered_df['id_characteristic'], filtered_df[vk_var]))
            vk_new = [vk_mapping.get(id_val, 1) for id_val in cleaned_id_characteristic]

            vk_value[mask] = vk_new

            vals += (vk_value,)
        else:
            vals += (vk_value,)

    return vals


def _get_vk_values(trafo_df, characteristic, trafotype="2W"):
    if trafotype == "2W":
        vk_variables = ("vk_percent", "vkr_percent")
    elif trafotype == "3W":
        vk_variables = ("vk_hv_percent", "vkr_hv_percent", "vk_mv_percent", "vkr_mv_percent",
                        "vk_lv_percent", "vkr_lv_percent")
    else:
        raise UserWarning("Unknown trafotype")

    if "tap_dependent_impedance" in trafo_df:
        tap_dependent_impedance = get_trafo_values(trafo_df, "tap_dependent_impedance")
        if np.any(np.isnan(tap_dependent_impedance)):
            raise UserWarning("tap_dependent_impedance has NaN values, but must be of type "
                              "bool and set to True or False")
        tap_pos = get_trafo_values(trafo_df, "tap_pos")
    else:
        tap_dependent_impedance = False
        tap_pos = None

    use_tap_dependent_impedance = np.any(tap_dependent_impedance)

    if use_tap_dependent_impedance:
        # is net.characteristic table in net?
        if characteristic is None:
            raise UserWarning("tap_dependent_impedance of transformers requires net.characteristic")

        # if any but 1 characteristic is missing per trafo, we assume it's by design;
        # but if all are missing, we raise an error
        # first, we read all characteristic indices
        # we also allow that some columns are not included in the net.trafo table
        all_columns = trafo_df.keys() if isinstance(trafo_df, dict) else trafo_df.columns.values
        char_columns = [v for v in vk_variables if f"{v}_characteristic" in all_columns]
        if len(char_columns) == 0:
            raise UserWarning(f"At least one of the columns for characteristics "
                              f"({[v+'_characteristic' for v in vk_variables]}) "
                              f"must be defined for {trafotype} trafo")
        # must cast to float64 unfortunately, because numpy.vstack casts arrays to object
        # because it doesn't know pandas.NA, np.isnan fails
        all_characteristic_idx = np.vstack([get_trafo_values(
            trafo_df, f"{c}_characteristic").astype(np.float64) for c in char_columns]).T
        index_column = {c: i for i, c in enumerate(char_columns)}
        # now we check if any trafos that have tap_dependent_impedance have all characteristics missing
        all_missing = np.isnan(all_characteristic_idx).all(axis=1) & tap_dependent_impedance
        if np.any(all_missing):
            trafo_index = trafo_df['index'] if isinstance(trafo_df, dict) else trafo_df.index.values
            raise UserWarning(f"At least one characteristic must be defined for {trafotype} "
                              f"trafo: {trafo_index[all_missing]}")

    vals = ()

    for _, vk_var in enumerate(vk_variables):
        vk_value = get_trafo_values(trafo_df, vk_var)
        if use_tap_dependent_impedance and vk_var in char_columns:
            vals += (_calc_tap_dependent_value(
                tap_pos, vk_value, tap_dependent_impedance,
                characteristic, all_characteristic_idx[:, index_column[vk_var]]),)
        else:
            vals += (vk_value,)

    return vals


def _calc_tap_dependent_value(tap_pos, value, tap_dependent_impedance, characteristic, characteristic_idx):
    # we skip the trafos with NaN characteristics even if tap_dependent_impedance is True
    # (we already checked for missing characteristics)
    relevant_idx = tap_dependent_impedance & ~np.isnan(characteristic_idx)
    vk_characteristic = np.zeros_like(tap_dependent_impedance, dtype="object")
    vk_characteristic[relevant_idx] = characteristic.loc[characteristic_idx[relevant_idx], 'object'].values
    # here dtype must be float otherwise the load flow calculation will fail

    def custom_func(f, t, c):
        return c(t).item() if f else np.nan

    custom_func_vec = np.vectorize(custom_func)
    return np.where(relevant_idx, custom_func_vec(relevant_idx, tap_pos, vk_characteristic), value)


def _calc_r_x_from_dataframe(mode, trafo_df, vn_lv, vn_trafo_lv, sn_mva, sequence=1, characteristic=None,
                             trafo_characteristic_table=None):
    """
    Calculates (Vectorized) the resistance and reactance according to the
    transformer values
    """
    parallel = get_trafo_values(trafo_df, "parallel")
    if sequence == 1:
        if "tap_dependency_table" in trafo_df:
            tap_dependency = get_trafo_values(trafo_df, "tap_dependency_table")
            tap_dependency = np.array(
                [False if isinstance(x, float) and np.isnan(x) else x for x in tap_dependency])
            if any(tap_dependency) and not isinstance(trafo_df, dict):
                if np.any(tap_dependency) and trafo_characteristic_table is None:
                    raise UserWarning("Trafo with tap_dependency_table True, but no trafo_characteristic_table found.")
                vk_percent, vkr_percent = _get_vk_values_from_table(trafo_df, trafo_characteristic_table)
                # update for 3W already in _calc_sc_voltages_of_equivalent_transformers
            else:
                vk_percent = get_trafo_values(trafo_df, "vk_percent")
                vkr_percent = get_trafo_values(trafo_df, "vkr_percent")
        else:
            warnings.warn(DeprecationWarning("tap_dependency_table is missing in net, which is most probably due to "
                                             "unsupported net data. tap_dependency_table was introduced with "
                                             "pandapower 3.0 and replaced spline characteristics. Spline "
                                             "characteristics will still work, but they are deprecated and will be "
                                             "removed in future releases."))

            vk_percent, vkr_percent = _get_vk_values(trafo_df, characteristic)

    elif sequence == 0:
        vk_percent = get_trafo_values(trafo_df, "vk0_percent")
        vkr_percent = get_trafo_values(trafo_df, "vkr0_percent")
    else:
        raise UserWarning("Unsupported sequence")

    # adjust for low voltage side voltage converter:
    if mode == 'pf_3ph':
        tap_lv = np.square(vn_trafo_lv / vn_lv) * (3 * sn_mva)
    else:
        tap_lv = np.square(vn_trafo_lv / vn_lv) * sn_mva

    sn_trafo_mva = get_trafo_values(trafo_df, "sn_mva")
    z_sc = vk_percent / 100. / sn_trafo_mva * tap_lv
    r_sc = vkr_percent / 100. / sn_trafo_mva * tap_lv
    x_sc = np.sign(z_sc) * np.sqrt((z_sc ** 2 - r_sc ** 2).astype(float))
    return r_sc / parallel, x_sc / parallel


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
    # Calculating tab (transformer off nominal turns ratio)
    tap_rat = vn_hv_kv / vn_lv_kv
    hv_bus = get_trafo_values(trafo_df, "hv_bus")
    lv_bus = get_trafo_values(trafo_df, "lv_bus")
    nom_rat = get_values(ppc["bus"][:, BASE_KV], hv_bus, bus_lookup) / \
        get_values(ppc["bus"][:, BASE_KV], lv_bus, bus_lookup)
    return tap_rat / nom_rat


def _calc_impedance_parameter(net, ppc):
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    f, t = net["_pd2ppc_lookups"]["branch"]["impedance"]
    branch = ppc["branch"]
    rij, xij, r_asym, x_asym, gi, bi, g_asym, b_asym = _calc_impedance_parameters_from_dataframe(net)
    branch[f:t, BR_R] = rij
    branch[f:t, BR_X] = xij
    branch[f:t, BR_R_ASYM] = r_asym
    branch[f:t, BR_X_ASYM] = x_asym
    branch[f:t, BR_G] = gi
    branch[f:t, BR_B] = bi
    branch[f:t, BR_G_ASYM] = g_asym
    branch[f:t, BR_B_ASYM] = b_asym
    branch[f:t, RATE_A] = net.impedance["sn_mva"].values
    branch[f:t, F_BUS] = bus_lookup[net.impedance["from_bus"].values]
    branch[f:t, T_BUS] = bus_lookup[net.impedance["to_bus"].values]
    branch[f:t, BR_STATUS] = net["impedance"]["in_service"].values


def _calc_tcsc_parameter(net, ppc):
    f = 0
    t = len(net.tcsc)

    if t == 0:
        return

    baseMVA = ppc["baseMVA"]
    bus_lookup = net["_pd2ppc_lookups"]["bus"]

    f_bus = bus_lookup[net.tcsc["from_bus"].values]
    t_bus = bus_lookup[net.tcsc["to_bus"].values]

    tcsc = ppc["tcsc"]
    baseV = ppc["bus"][f_bus, BASE_KV]
    baseZ = baseV ** 2 / baseMVA

    tcsc[f:t, TCSC_F_BUS] = f_bus
    tcsc[f:t, TCSC_T_BUS] = t_bus

    tcsc[f:t, TCSC_X_L] = net["tcsc"]["x_l_ohm"].values / baseZ
    tcsc[f:t, TCSC_X_CVAR] = net["tcsc"]["x_cvar_ohm"].values / baseZ
    tcsc[f:t, TCSC_SET_P] = net["tcsc"]["set_p_to_mw"].values / baseMVA
    tcsc[f:t, TCSC_THYRISTOR_FIRING_ANGLE] = np.deg2rad(net["tcsc"]["thyristor_firing_angle_degree"].values)
    tcsc[f:t, TCSC_MIN_FIRING_ANGLE] = np.deg2rad(net["tcsc"]["min_angle_degree"].values)
    tcsc[f:t, TCSC_MAX_FIRING_ANGLE] = np.deg2rad(net["tcsc"]["max_angle_degree"].values)

    tcsc[f:t, TCSC_STATUS] = net["tcsc"]["in_service"].values
    tcsc[f:t, TCSC_CONTROLLABLE] = (net["tcsc"]["controllable"].values.astype(bool) &
                                    net["tcsc"]["in_service"].values.astype(bool))


def _calc_impedance_parameters_from_dataframe(net, zero_sequence=False):
    impedance = net.impedance
    suffix = "0" if zero_sequence else ""

    rij = impedance[f"rft{suffix}_pu"].values
    xij = impedance[f"xft{suffix}_pu"].values
    rji = impedance[f"rtf{suffix}_pu"].values
    xji = impedance[f"xtf{suffix}_pu"].values
    gi = impedance[f"gf{suffix}_pu"].values
    bi = impedance[f"bf{suffix}_pu"].values
    gj = impedance[f"gt{suffix}_pu"].values
    bj = impedance[f"bt{suffix}_pu"].values

    mode = net["_options"]["mode"]
    sn_factor = 3. if mode == 'pf_3ph' else 1.
    sn_impedance = impedance["sn_mva"].values
    sn_net = net.sn_mva

    # background for the sn_calculations in the next lines:
    # r_ij_ohm = r_ij * v**2 / sn_impedance
    # r_ij_pu_branch = r_ij_ohm / (v**2 / sn_net)
    # r_ij_pu_branch = r_ij / sn_impedance / (1 / sn_net)

    r_f = (rij * sn_factor) / sn_impedance * sn_net
    x_f = (xij * sn_factor) / sn_impedance * sn_net
    r_t = (rji * sn_factor) / sn_impedance * sn_net
    x_t = (xji * sn_factor) / sn_impedance * sn_net
    # todo sn_factor + formulas in general for g_f, b_f, g_t, b_t
    # 2 because Bcf, Bct is divided by 2 in makeYbus (maybe change?)
    g_f = 2 * (gi * sn_factor) * sn_impedance / sn_net
    b_f = 2 * (bi * sn_factor) * sn_impedance / sn_net
    g_t = 2 * (gj * sn_factor) * sn_impedance / sn_net
    b_t = 2 * (bj * sn_factor) * sn_impedance / sn_net
    r_asym = r_t - r_f
    x_asym = x_t - x_f
    g_asym = g_t - g_f
    b_asym = b_t - b_f
    return r_f, x_f, r_asym, x_asym, g_f, b_f, g_asym, b_asym


def _calc_xward_parameter(net, ppc):
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    f, t = net["_pd2ppc_lookups"]["branch"]["xward"]
    branch = ppc["branch"]
    baseR = np.square(get_values(ppc["bus"][:, BASE_KV], net["xward"]["bus"].values, bus_lookup)) / \
        net.sn_mva
    xw_is = net["_is_elements"]["xward"]
    branch[f:t, F_BUS] = bus_lookup[net["xward"]["bus"].values]
    branch[f:t, T_BUS] = bus_lookup[net._pd2ppc_lookups["aux"]["xward"]]
    branch[f:t, BR_R] = net["xward"]["r_ohm"] / baseR
    branch[f:t, BR_X] = net["xward"]["x_ohm"] / baseR
    branch[f:t, BR_STATUS] = xw_is


def _gather_branch_switch_info(bus, branch_id, branch_type, net):
    # determine at which end the switch is located
    # 1 = to-bus/lv-bus; 0 = from-bus/hv-bus
    branch_id = int(branch_id)
    lookup = net._pd2ppc_lookups["branch"]
    if branch_type == "l":
        side = "to" if net["line"]["to_bus"].at[branch_id] == bus else "from"
        branch_idx = net["line"].index.get_loc(branch_id)
        return side, int(bus), int(branch_idx)
    elif branch_type == "t":
        side = "hv" if net["trafo"]["hv_bus"].at[branch_id] == bus else "lv"
        branch_idx = lookup["trafo"][0] + net["trafo"].index.get_loc(branch_id)
        return side, int(bus), int(branch_idx)
    elif branch_type == "t3":
        f, t = lookup["trafo3w"]
        if net["trafo3w"]["hv_bus"].at[branch_id] == bus:
            side = "hv"
            offset = 0
        elif net["trafo3w"]["mv_bus"].at[branch_id] == bus:
            side = "mv"
            offset = (t - f) / 3
        elif net["trafo3w"]["lv_bus"].at[branch_id] == bus:
            side = "lv"
            offset = (t - f) / 3 * 2
        branch_idx = lookup["trafo3w"][0] + net["trafo3w"].index.get_loc(branch_id) + offset
        return side, int(bus), int(branch_idx)


def _switch_branches(net, ppc):
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    calculate_voltage_angles = net._options["calculate_voltage_angles"]
    neglect_open_switch_branches = net._options["neglect_open_switch_branches"]
    mode = net._options["mode"]
    n_bus = ppc["bus"].shape[0]
    for et, element in [("l", "line"), ("t", "trafo"), ("t3", "trafo3w")]:
        switch_mask = ~net.switch.closed.values & (net.switch.et.values == et)
        if not switch_mask.any():
            continue
        nr_open_switches = np.count_nonzero(switch_mask)
        mapfunc = partial(_gather_branch_switch_info, branch_type=et, net=net)
        switch_element = net["switch"]["element"].values[switch_mask]
        switch_buses = net["switch"]["bus"].values[switch_mask]
        switch_info = np.array(list(map(mapfunc, switch_buses, switch_element)))
        sw_sides = switch_info[:, 0]
        sw_bus_index = bus_lookup[switch_info[:, 1].astype(np.int64)]
        sw_branch_index = switch_info[:, 2].astype(np.int64)
        if neglect_open_switch_branches:
            # deactivate switches which have an open switch instead of creating aux buses
            ppc["branch"][sw_branch_index, BR_STATUS] = 0
            continue

        new_buses = np.zeros(shape=(nr_open_switches, ppc["bus"].shape[1]), dtype=float)
        new_buses[:, :15] = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1.1, 0.9, 0, 0])
        new_indices = np.arange(n_bus, n_bus + nr_open_switches)
        new_buses[:, 0] = new_indices
        new_buses[:, BASE_KV] = ppc["bus"][sw_bus_index, BASE_KV]
        ppc["bus"] = np.vstack([ppc["bus"], new_buses])
        n_bus += new_buses.shape[0]
        init_vm = net._options["init_vm_pu"]
        init_va = net._options["init_va_degree"]
        for location in np.unique(sw_sides):
            mask = sw_sides == location
            buses = new_indices[mask]
            side = F_BUS if location == "hv" or location == "from" else T_BUS
            for init, col in [(init_vm, VM), (init_va, VA)]:
                if isinstance(init, str) and init == "results":
                    if col == VM:
                        res_column = net["res_%s" % element]["vm_%s_pu" % location]
                    else:
                        res_column = net["res_%s" % element]["va_%s_degree" % location]
                    init_values = res_column.loc[switch_element].values[mask]
                else:
                    if element == "line":
                        opposite_buses = ppc["branch"][sw_branch_index[mask], side].real.astype(np.int64)
                        init_values = ppc["bus"][opposite_buses, col]
                    else:
                        opposite_side = T_BUS if side == F_BUS else F_BUS
                        opposite_buses = ppc["branch"][sw_branch_index[mask],
                                                       opposite_side].real.astype(np.int64)
                        if col == VM:
                            taps = ppc["branch"][sw_branch_index[mask], TAP].real
                            init_values = ppc["bus"][opposite_buses, col] * taps
                        else:
                            if calculate_voltage_angles:
                                shift = ppc["branch"][sw_branch_index[mask], SHIFT].real.astype(np.int64)
                                init_values = ppc["bus"][opposite_buses, col] + shift
                            else:
                                init_values = ppc["bus"][opposite_buses, col]
                        if mode == "sc":
                            ppc["bus"][buses, C_MAX] = ppc["bus"][opposite_buses, C_MAX]
                            ppc["bus"][buses, C_MIN] = ppc["bus"][opposite_buses, C_MIN]
                ppc["bus"][buses, col] = init_values
            ppc["branch"][sw_branch_index[mask], side] = new_indices[mask]


def _branches_with_oos_buses(net, ppc, dc=False):
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
    line_table = "line_dc" if dc else "line"
    bus_table = "bus_dc" if dc else "bus"
    branch_table = "branch_dc" if dc else "branch"

    bus_lookup = net["_pd2ppc_lookups"][bus_table]
    # get in service elements
    _is_elements = net["_is_elements"]
    bus_is_idx = _is_elements[f"{bus_table}_is_idx"]
    line_is_idx = _is_elements[f"{line_table}_is_idx"]

    n_oos_buses = len(net[bus_table]) - len(bus_is_idx)

    # only filter lines at oos buses if oos buses exists
    if n_oos_buses > 0:
        n_bus = len(ppc[bus_table])
        future_buses = [ppc[bus_table]]
        # out of service buses
        bus_oos = np.setdiff1d(net[bus_table].index.values, bus_is_idx)
        # from buses of line
        ft_cols = ["from_bus_dc", "to_bus_dc"] if dc else ["from_bus", "to_bus"]
        line_buses = net[line_table][ft_cols].loc[line_is_idx].values
        f_bus = line_buses[:, 0]
        t_bus = line_buses[:, 1]

        # determine on which side of the line the oos bus is located
        mask_from = np.isin(f_bus, bus_oos)
        mask_to = np.isin(t_bus, bus_oos)

        mask_and = mask_to & mask_from
        if np.any(mask_and):
            mask_from[mask_and] = False
            mask_to[mask_and] = False

        # get lines that are connected to oos bus at exactly one side
        # buses that are connected to two oos buses will be removed by ext2int
        mask_or = mask_to | mask_from
        # check whether buses are connected to line
        oos_buses_at_lines = np.hstack([f_bus[mask_from], t_bus[mask_to]])
        n_oos_buses_at_lines = len(oos_buses_at_lines)

        # only if oos_buses are at lines (they could be isolated as well)
        if n_oos_buses_at_lines > 0:
            ls_info = np.zeros((n_oos_buses_at_lines, 3), dtype=np.int64)
            ls_info[:, 0] = mask_to[mask_or] & ~mask_from[mask_or]
            ls_info[:, 1] = oos_buses_at_lines
            ls_info[:, 2] = np.nonzero(np.isin(net[line_table].index, line_is_idx[mask_or]))[0]

            # ls_info = list(map(mapfunc,
            #               line_switches["bus"].values,
            #               line_switches["element"].values))
            # we now have the following matrix
            # 0: 1 if switch is at to_bus, 0 else
            # 1: bus of the switch
            # 2: position of the line a switch is connected to
            # ls_info = np.array(ls_info, dtype=np.int64)

            # build new buses
            new_ls_buses = np.zeros(shape=(n_oos_buses_at_lines, ppc[bus_table].shape[1]), dtype=np.float64)
            new_indices = np.arange(n_bus, n_bus + n_oos_buses_at_lines)
            # the newly created buses
            if dc:
                ppc_col_var = [DC_BUS_TYPE, DC_BUS_AREA, DC_VM, DC_ZONE, DC_VMAX, DC_VMIN]
                ppc_col_val = np.array([DC_P, 1, 1, 1, 2, 0], dtype=np.int64)
            else:
                ppc_col_var = [BUS_TYPE, BUS_AREA, VM, ZONE, VMAX, VMIN]
                ppc_col_val = np.array([PQ, 1, 1, 1, 2, 0], dtype=np.int64)
            new_ls_buses[:, ppc_col_var] = ppc_col_val
            new_ls_buses[:, 0] = new_indices
            new_ls_buses[:, DC_BASE_KV if dc else BASE_KV] = \
                get_values(ppc[bus_table][:, DC_BASE_KV if dc else BASE_KV], ls_info[:, 1], bus_lookup)

            future_buses.append(new_ls_buses)

            # re-route the end of lines to a new bus
            ppc[branch_table][ls_info[ls_info[:, 0].astype(bool), 2], 1] = \
                new_indices[ls_info[:, 0].astype(bool)]
            ppc[branch_table][ls_info[np.logical_not(ls_info[:, 0]), 2], 0] = \
                new_indices[np.logical_not(ls_info[:, 0])]

            ppc[bus_table] = np.vstack(future_buses)


def _calc_switch_parameter(net, ppc):
    """
    calculates the line parameter in per unit.

    **INPUT**:
        **net** -The pandapower format network

    **RETURN**:
        **t** - Temporary line parameter. Which is a complex128
                Numpy array. with the following order:
                0:bus_a; 1:bus_b; 2:r_pu; 3:x_pu; 4:b_pu
    """
    rx_ratio = net["_options"]["switch_rx_ratio"]
    rz_ratio = rx_ratio / np.sqrt(1 + rx_ratio ** 2)
    xz_ratio = 1 / np.sqrt(1 + rx_ratio ** 2)

    f, t = net["_pd2ppc_lookups"]["branch"]["switch"]
    branch = ppc["branch"]
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    switch = net.switch[net._impedance_bb_switches]
    fb = bus_lookup[switch["bus"].values]
    tb = bus_lookup[switch["element"].values]
    baseR = np.square(ppc["bus"][fb, BASE_KV]) / net.sn_mva
    branch[f:t, F_BUS] = fb
    branch[f:t, T_BUS] = tb

    z_switch = switch['z_ohm'].values
    # x_switch will have the same value of r_switch to avoid zero division
    branch[f:t, BR_R] = z_switch / baseR * rz_ratio
    branch[f:t, BR_X] = z_switch / baseR * xz_ratio


def _end_temperature_correction_factor(net, short_circuit=False, dc=False):
    """
    Function to calculate resistance correction factor for the given temperature ("endtemp_degree").
    When multiplied by the factor, the value of r_ohm_per_km will correspond to the resistance at
    the given temperature.

    In case of short circuit calculation, the relevant value for the temperature is
    "endtemp_degree", which stands for the final temperature of a line after the short circuit.
    The temperature coefficient "alpha" is a constant value of 0.004 in the short circuit
    calculation standard IEC 60909-0:2016.

    In case of a load flow calculation, the relevant parameter is "temperature_degree_celsius",
    which is specified by the user and allows calculating load flow for a given operating
    temperature.

    The alpha value can be provided according to the used material for the purpose of load flow
    calculation, e.g. 0.0039 for copper or 0.00403 for aluminum. If alpha is not provided in the
    net.line table, the default value of 0.004 is used.

    The calculation of the electrical resistance is based on the formula R = R20(1+alpha*(T-20°C)),
    where R is the calculated resistance, R20 is the resistance at 20 °C, alpha is the temperature
    coefficient of resistance of the conducting material and T is the line temperature in °C.
    Accordingly, the resulting correction factor is (1+alpha*(T-20°C)).

    Args:
        net: pandapowerNet
        short_circuit: whether the factor is calculated in the scope of a short circuit calculation

    Returns:
        correction factor for line R, by which the line parameter should be multiplied to
                obtain the value of resistance at line temperature "endtemp_degree"

    """

    element = "line_dc" if dc else "line"
    if short_circuit:
        # endtemp_degree is line temperature that is reached as the result of a short circuit
        # this value is the property of the lines
        if "endtemp_degree" not in net[element].columns:
            raise UserWarning(f"Specify end temperature for {element}s in net.{element}.endtemp_degree")

        delta_t_degree_celsius = net[element].endtemp_degree.values.astype(np.float64) - 20
        # alpha is the temperature correction factor for the electric resistance of the material
        # formula from standard, used this way in short-circuit calculation
        alpha = 4e-3
    else:
        # temperature_degree_celsius is line temperature for load flow calculation
        if "temperature_degree_celsius" not in net[element].columns:
            raise UserWarning(f"Specify {element} temperature in net.{element}.temperature_degree_celsius")

        delta_t_degree_celsius = net[element].temperature_degree_celsius.values.astype(np.float64) - 20

        if 'alpha' in net[element].columns:
            alpha = net[element].alpha.values.astype(np.float64)
        else:
            alpha = 4e-3

    r_correction_for_temperature = 1 + alpha * delta_t_degree_celsius

    return r_correction_for_temperature


def _transformer_correction_factor(trafo_df, vk, vkr, sn, cmax):
    """
        2W-Transformer impedance correction factor in short circuit calculations,
        based on the IEC 60909-0:2016 standard.
        Args:
            vk: transformer short-circuit voltage, percent
            vkr: real-part of transformer short-circuit voltage, percent
            sn: transformer rating, kVA
            cmax: voltage factor to account for maximum worst-case currents, based on the lv side

        Returns:
            kt: transformer impedance correction factor for short-circuit calculations

    Parameters
    ----------
    trafo_df

        """

    if "power_station_unit" in trafo_df.columns:
        power_station_unit = trafo_df.power_station_unit.fillna(False).values.astype(bool)
    else:
        power_station_unit = np.zeros(len(trafo_df)).astype(bool)

    zt = vk / 100 / sn
    rt = vkr / 100 / sn
    xt = np.sqrt(zt ** 2 - rt ** 2)
    kt = 0.95 * cmax / (1 + .6 * xt * sn)
    return np.where(~power_station_unit, kt, 1)


def get_is_lines(net):
    """
    get indices of lines that are in service and save that information in net
    """
    _is_elements = net["_is_elements"]
    _is_elements["line"] = net["line"][net["line"]["in_service"].values.astype(bool)]


def _trafo_df_from_trafo3w(net, sequence=1):
    trafo2 = dict()
    sides = ["hv", "mv", "lv"]
    mode = net._options["mode"]
    t3 = net["trafo3w"]
    # todo check magnetizing impedance implementation:
    # loss_side = net._options["trafo3w_losses"].lower()
    loss_side = t3.loss_side.values if "loss_side" in t3.columns else np.full(len(t3),
                                                                              net._options["trafo3w_losses"].lower())
    nr_trafos = len(net["trafo3w"])
    if sequence == 1:
        if 'tap_dependency_table' in t3:
            mode_tmp = "type_c" if mode == "sc" and net._options.get("use_pre_fault_voltage", False) else mode
            _calculate_sc_voltages_of_equivalent_transformers(t3, trafo2, mode_tmp, net=net)
        else:
            mode_tmp = "type_c" if mode == "sc" and net._options.get("use_pre_fault_voltage", False) else mode
            _calculate_sc_voltages_of_equivalent_transformers(t3, trafo2, mode_tmp, characteristic=net.get(
                'characteristic'))
    elif sequence == 0:
        if mode != "sc":
            raise NotImplementedError(
                "0 seq impedance calculation only implemented for short-circuit calculation!")
        _calculate_sc_voltages_of_equivalent_transformers_zero_sequence(t3, trafo2,)
    else:
        raise UserWarning("Unsupported sequence for trafo3w convertion")
    _calculate_3w_tap_changers(t3, trafo2, sides)
    zeros = np.zeros(len(net.trafo3w))
    aux_buses = net._pd2ppc_lookups["aux"]["trafo3w"]
    trafo2["hv_bus"] = {"hv": t3.hv_bus.values, "mv": aux_buses, "lv": aux_buses}
    trafo2["lv_bus"] = {"hv": aux_buses, "mv": t3.mv_bus.values, "lv": t3.lv_bus.values}
    trafo2["in_service"] = {side: t3.in_service.values for side in sides}
    # todo check magnetizing impedance implementation:
    # trafo2["i0_percent"] = {side: t3.i0_percent.values if loss_side == side else zeros for side in sides}
    # trafo2["pfe_kw"] = {side: t3.pfe_kw.values if loss_side == side else zeros for side in sides}
    trafo2["i0_percent"] = {side: np.where(loss_side == side, t3.i0_percent.values, zeros) for side in sides}
    trafo2["pfe_kw"] = {side: np.where(loss_side == side, t3.pfe_kw.values, zeros) for side in sides}
    trafo2["vn_hv_kv"] = {side: t3.vn_hv_kv.values for side in sides}
    trafo2["vn_lv_kv"] = {side: t3["vn_%s_kv" % side].values for side in sides}
    trafo2["shift_degree"] = {"hv": np.zeros(nr_trafos), "mv": t3.shift_mv_degree.values,
                              "lv": t3.shift_lv_degree.values}
    for param in ["tap_changer_type", "tap_dependency_table", "id_characteristic_table", "tap_phase_shifter"]:
        if param in t3:
            trafo2[param] = {side: t3[param] for side in sides}
    trafo2["parallel"] = {side: np.ones(nr_trafos) for side in sides}
    trafo2["df"] = {side: np.ones(nr_trafos) for side in sides}
    # even though this is not relevant (at least now), the values cannot be empty:
    trafo2["leakage_resistance_ratio_hv"] = {
        side: np.full(nr_trafos, fill_value=0.5, dtype=np.float64) for side in sides}
    trafo2["leakage_reactance_ratio_hv"] = {
        side: np.full(nr_trafos, fill_value=0.5, dtype=np.float64) for side in sides}
    if "max_loading_percent" in net.trafo3w:
        trafo2["max_loading_percent"] = {side: net.trafo3w.max_loading_percent.values for side in sides}
    return {var: np.concatenate([trafo2[var][side] for side in sides]) for var in trafo2.keys()}


def _calculate_sc_voltages_of_equivalent_transformers(
        t3, t2, mode, characteristic=None, net=None):
    if "tap_dependency_table" in t3:
        tap_dependency_table = get_trafo_values(t3, "tap_dependency_table")
        tap_dependency_table = np.array(
            [False if isinstance(x, float) and np.isnan(x) else x for x in tap_dependency_table])
        if any(tap_dependency_table):
            vk_hv, vkr_hv, vk_mv, vkr_mv, vk_lv, vkr_lv = _get_vk_values_from_table(
                t3, net.trafo_characteristic_table, "3W")
        else:
            vk_hv, vkr_hv, vk_mv, vkr_mv, vk_lv, vkr_lv = (
                t3['vk_hv_percent'], t3['vkr_hv_percent'], t3['vk_mv_percent'],
                t3['vkr_mv_percent'], t3['vk_lv_percent'], t3['vkr_lv_percent'])
    else:
        warnings.warn(DeprecationWarning("tap_dependency_table is missing in net, which is most probably due to "
                                         "old net data. tap_dependency_table was introduced with "
                                         "pandapower 3.0 and replaced spline characteristics. Spline "
                                         "characteristics will still work, but they are deprecated and will be "
                                         "removed in future releases."))
        vk_hv, vkr_hv, vk_mv, vkr_mv, vk_lv, vkr_lv = _get_vk_values(t3, characteristic, "3W")

    vk_3w = np.stack([vk_hv, vk_mv, vk_lv])
    vkr_3w = np.stack([vkr_hv, vkr_mv, vkr_lv])
    sn = np.stack([t3.sn_hv_mva.values, t3.sn_mv_mva.values, t3.sn_lv_mva.values])

    vk_2w_delta = z_br_to_bus_vector(vk_3w, sn)
    vkr_2w_delta = z_br_to_bus_vector(vkr_3w, sn)
    if mode == "sc":
        kt = _transformer_correction_factor(t3, vk_3w, vkr_3w, sn, 1.1)
        vk_2w_delta *= kt
        vkr_2w_delta *= kt
    vki_2w_delta = np.sqrt(vk_2w_delta ** 2 - vkr_2w_delta ** 2)
    vkr_2w = wye_delta_vector(vkr_2w_delta, sn)
    vki_2w = wye_delta_vector(vki_2w_delta, sn)
    vk_2w = np.sign(vki_2w) * np.sqrt(vki_2w ** 2 + vkr_2w ** 2)
    if np.any(vk_2w == 0):
        raise UserWarning("Equivalent transformer with zero impedance!")
    t2["vk_percent"] = {"hv": vk_2w[0, :], "mv": vk_2w[1, :], "lv": vk_2w[2, :]}
    t2["vkr_percent"] = {"hv": vkr_2w[0, :], "mv": vkr_2w[1, :], "lv": vkr_2w[2, :]}
    t2["sn_mva"] = {"hv": sn[0, :], "mv": sn[1, :], "lv": sn[2, :]}


def _calculate_sc_voltages_of_equivalent_transformers_zero_sequence(t3, t2):
    vk_3w = np.stack([t3.vk_hv_percent.values, t3.vk_mv_percent.values, t3.vk_lv_percent.values])
    vkr_3w = np.stack([t3.vkr_hv_percent.values, t3.vkr_mv_percent.values, t3.vkr_lv_percent.values])
    vk0_3w = np.stack([t3.vk0_hv_percent.values, t3.vk0_mv_percent.values, t3.vk0_lv_percent.values])
    vkr0_3w = np.stack([t3.vkr0_hv_percent.values, t3.vkr0_mv_percent.values, t3.vkr0_lv_percent.values])
    sn = np.stack([t3.sn_hv_mva.values, t3.sn_mv_mva.values, t3.sn_lv_mva.values])

    vk0_2w_delta = z_br_to_bus_vector(vk0_3w, sn)
    vkr0_2w_delta = z_br_to_bus_vector(vkr0_3w, sn)

    # Only for "sc", calculated with positive sequence value
    kt = _transformer_correction_factor(t3, vk_3w, vkr_3w, sn, 1.1)
    vk0_2w_delta *= kt
    vkr0_2w_delta *= kt

    vki0_2w_delta = np.sqrt(vk0_2w_delta ** 2 - vkr0_2w_delta ** 2)
    vkr0_2w = wye_delta_vector(vkr0_2w_delta, sn)
    vki0_2w = wye_delta_vector(vki0_2w_delta, sn)
    vk0_2w = np.sign(vki0_2w) * np.sqrt(vki0_2w ** 2 + vkr0_2w ** 2)
    if np.any(vk0_2w == 0):
        raise UserWarning("Equivalent transformer with zero impedance!")
    t2["vk0_percent"] = {"hv": vk0_2w[0, :], "mv": vk0_2w[1, :], "lv": vk0_2w[2, :]}
    t2["vkr0_percent"] = {"hv": vkr0_2w[0, :], "mv": vkr0_2w[1, :], "lv": vkr0_2w[2, :]}
    t2["sn_mva"] = {"hv": sn[0, :], "mv": sn[1, :], "lv": sn[2, :]}


def z_br_to_bus_vector(z, sn):
    return sn[0, :] * np.array([z[0, :] / sn[[0, 1], :].min(axis=0), z[1, :] /
                                sn[[1, 2], :].min(axis=0), z[2, :] / sn[[0, 2], :].min(axis=0)])


def wye_delta(zbr_n, s):
    return .5 * s / s[0] * np.array([(zbr_n[0] + zbr_n[2] - zbr_n[1]),
                                     (zbr_n[1] + zbr_n[0] - zbr_n[2]),
                                     (zbr_n[2] + zbr_n[1] - zbr_n[0])])


def wye_delta_vector(zbr_n, s):
    return .5 * s / s[0, :] * np.array([(zbr_n[0, :] + zbr_n[2, :] - zbr_n[1, :]),
                                        (zbr_n[1, :] + zbr_n[0, :] - zbr_n[2, :]),
                                        (zbr_n[2, :] + zbr_n[1, :] - zbr_n[0, :])])


def _calculate_3w_tap_changers(t3, t2, sides):
    tap_variables = ["tap_side", "tap_pos", "tap_neutral", "tap_max", "tap_min", "tap_step_percent",
                     "tap_step_degree"]
    sides = ["hv", "mv", "lv"]
    nr_trafos = len(t3)
    empty = np.zeros(nr_trafos)
    empty.fill(np.nan)
    tap_arrays = {var: {side: empty.copy() for side in sides} for var in tap_variables}
    tap_arrays["tap_side"] = {side: np.array([None] * nr_trafos) for side in sides}
    at_star_point = t3.tap_at_star_point.values
    any_at_star_point = at_star_point.any()
    for side in sides:
        tap_mask = t3.tap_side.values == side
        for var in tap_variables:
            tap_arrays[var][side][tap_mask] = t3[var].values[tap_mask]

        # t3 trafos with tap changer at terminals
        tap_arrays["tap_side"][side][tap_mask] = "hv" if side == "hv" else "lv"

        # t3 trafos with tap changer at star points
        if any_at_star_point & np.any(mask_star_point := (tap_mask & at_star_point)):
            t = (tap_arrays["tap_step_percent"][side][mask_star_point] *
                 np.exp(1j * np.deg2rad(tap_arrays["tap_step_degree"][side][mask_star_point])))
            tap_pos = tap_arrays["tap_pos"][side][mask_star_point]
            tap_neutral = tap_arrays["tap_neutral"][side][mask_star_point]
            t_corrected = 100 * t / (100 + (t * (tap_pos-tap_neutral)))
            tap_arrays["tap_step_percent"][side][mask_star_point] = np.abs(t_corrected)
            tap_arrays["tap_side"][side][mask_star_point] = "lv" if side == "hv" else "hv"
            tap_arrays["tap_step_degree"][side][mask_star_point] = np.rad2deg(np.angle(t_corrected))
            tap_arrays["tap_step_degree"][side][mask_star_point] -= 180
    t2.update(tap_arrays)
