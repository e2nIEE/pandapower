# -*- coding: utf-8 -*-

# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


"""Solves the power flow using a full Newton's method.
"""
import numpy as np
from numpy import float64, array, angle, sqrt, square, exp, linalg, conj, r_, inf, arange, zeros, \
    max, zeros_like, column_stack, flatnonzero, nan_to_num
from pandapower.pypower.bustypes import bustypes_dc
from pandapower.pypower.idx_brch_dc import DC_BR_R, DC_PF, DC_IF, DC_PT, DC_IT, DC_BR_STATUS, DC_F_BUS, DC_T_BUS
from scipy.sparse import csr_matrix, eye, vstack
from scipy.sparse.linalg import spsolve

from pandapower.auxiliary import _sum_by_group
from pandapower.pf.iwamoto_multiplier import _iwamoto_step
from pandapower.pf.makeYbus_facts import makeYbus_svc, makeYft_tcsc, calc_y_svc_pu, \
    makeYbus_ssc_vsc, make_Ybus_facts, make_Yft_facts
from pandapower.pypower.idx_bus_dc import DC_PD, DC_VM, DC_BUS_TYPE, DC_NONE, DC_BUS_I, DC_REF, DC_P
from pandapower.pypower.idx_vsc import VSC_CONTROLLABLE, VSC_MODE_AC, VSC_VALUE_AC, VSC_MODE_DC, VSC_VALUE_DC, VSC_R, \
    VSC_X, VSC_Q, VSC_P, VSC_BUS_DC, VSC_P_DC, VSC_MODE_AC_SL, VSC_MODE_AC_V, VSC_MODE_AC_Q, VSC_MODE_DC_P, \
    VSC_MODE_DC_V, VSC_INTERNAL_BUS_DC, VSC_R_DC, VSC_PL_DC, VSC_STATUS, VSC_BUS, VSC_INTERNAL_BUS
from pandapower.pypower.makeSbus import makeSbus
from pandapower.pf.create_jacobian import create_jacobian_matrix, get_fastest_jacobian_function
from pandapower.pypower.idx_gen import PG
from pandapower.pypower.idx_bus import PD, SL_FAC, BASE_KV
from pandapower.pypower.idx_brch import BR_R, BR_X, F_BUS, T_BUS
from pandapower.pypower.idx_brch_tdpf import BR_R_REF_OHM_PER_KM, BR_LENGTH_KM, RATE_I_KA, \
    T_START_C, R_THETA, WIND_SPEED_MPS, ALPHA, TDPF, OUTER_DIAMETER_M, MC_JOULE_PER_M_K, \
    WIND_ANGLE_DEGREE, SOLAR_RADIATION_W_PER_SQ_M, GAMMA, EPSILON, T_AMBIENT_C, T_REF_C
from pandapower.pypower.idx_tcsc import TCSC_F_BUS, TCSC_T_BUS, TCSC_X_L, TCSC_X_CVAR, TCSC_SET_P, \
    TCSC_THYRISTOR_FIRING_ANGLE, TCSC_STATUS, TCSC_CONTROLLABLE, TCSC_MIN_FIRING_ANGLE, \
    TCSC_MAX_FIRING_ANGLE, TCSC_PF, TCSC_QF, TCSC_PT, TCSC_QT, TCSC_IF, TCSC_IT, TCSC_X_PU
from pandapower.pypower.idx_svc import SVC_BUS, SVC_STATUS, SVC_CONTROLLABLE, SVC_X_L, SVC_X_CVAR, \
    SVC_X_PU, SVC_SET_VM_PU, SVC_THYRISTOR_FIRING_ANGLE, SVC_MAX_FIRING_ANGLE, \
    SVC_MIN_FIRING_ANGLE, SVC_Q
from pandapower.pypower.idx_ssc import SSC_BUS, SSC_R, SSC_X, SSC_SET_VM_PU, SSC_STATUS, \
    SSC_CONTROLLABLE, SSC_Q, SSC_X_CONTROL_VM, SSC_X_CONTROL_VA, SSC_INTERNAL_BUS

from pandapower.pf.create_jacobian_tdpf import calc_g_b, calc_a0_a1_a2_tau, calc_r_theta, \
    calc_T_frank, calc_i_square_p_loss, create_J_tdpf

from pandapower.pf.create_jacobian_facts import create_J_modification_svc, \
    create_J_modification_tcsc, create_J_modification_ssc_vsc, create_J_modification_hvdc


def newtonpf(Ybus, Sbus, V0, ref, pv, pq, ppci, options, makeYbus=None):
    """Solves the power flow using a full Newton's method.
    Solves for bus voltages given the full system admittance matrix (for
    all buses), the complex bus power injection vector (for all buses),
    the initial vector of complex bus voltages, and column vectors with
    the lists of bus indices for the swing bus, PV buses, and PQ buses,
    respectively. The bus voltage vector contains the set point for
    generator (including ref bus) buses, and the reference angle of the
    swing bus, as well as an initial guess for remaining magnitudes and
    angles.
    @see: L{runpf}
    @author: Ray Zimmerman (PSERC Cornell)
    @author: Richard Lincoln
    Modified by University of Kassel (Florian Schaefer) to use numba
    """

    # options
    tol = options['tolerance_mva']
    max_it = options["max_iteration"]
    numba = options["numba"]
    iwamoto = options["algorithm"] == "iwamoto_nr"
    voltage_depend_loads = options["voltage_depend_loads"]
    dist_slack = options["distributed_slack"]
    v_debug = options["v_debug"]
    use_umfpack = options["use_umfpack"]
    permc_spec = options["permc_spec"]

    baseMVA = ppci['baseMVA']
    bus = ppci['bus']
    bus_dc = ppci['bus_dc']
    gen = ppci['gen']
    branch = ppci['branch']
    branch_dc = ppci['branch_dc']
    tcsc = ppci['tcsc']
    svc = ppci['svc']
    ssc = ppci['ssc']
    vsc = ppci['vsc']
    slack_weights = bus[:, SL_FAC].astype(float64)  ## contribution factors for distributed slack
    tdpf = options.get('tdpf', False)

    # FACTS
    ## svc
    svc_idx = flatnonzero(nan_to_num(svc[:, SVC_STATUS]))
    svc_buses = svc[svc_idx, SVC_BUS].astype(np.int64)
    svc_set_vm_pu = svc[svc_idx, SVC_SET_VM_PU]
    x_control_svc = svc[svc_idx, SVC_THYRISTOR_FIRING_ANGLE]
    svc_controllable = svc[svc_idx, SVC_CONTROLLABLE].astype(bool)
    svc_x_l_pu = svc[svc_idx, SVC_X_L]
    svc_x_cvar_pu = svc[svc_idx, SVC_X_CVAR]
    svc_min_x = svc[svc_idx[svc_controllable], SVC_MIN_FIRING_ANGLE].real
    svc_max_x = svc[svc_idx[svc_controllable], SVC_MAX_FIRING_ANGLE].real
    num_svc_controllable = len(x_control_svc[svc_controllable])
    num_svc = len(x_control_svc)
    any_svc = num_svc > 0
    any_svc_controllable = num_svc_controllable > 0

    ## tcsc
    tcsc_branches = flatnonzero(nan_to_num(tcsc[:, TCSC_STATUS]))
    tcsc_fb = tcsc[tcsc_branches, [TCSC_F_BUS]].real.astype(np.int64)
    tcsc_tb = tcsc[tcsc_branches, [TCSC_T_BUS]].real.astype(np.int64)
    tcsc_controllable = tcsc[tcsc_branches, TCSC_CONTROLLABLE].real.astype(bool)
    tcsc_set_p_pu = tcsc[tcsc_branches[tcsc_controllable], TCSC_SET_P].real
    tcsc_min_x = tcsc[tcsc_branches[tcsc_controllable], TCSC_MIN_FIRING_ANGLE].real
    tcsc_max_x = tcsc[tcsc_branches[tcsc_controllable], TCSC_MAX_FIRING_ANGLE].real
    x_control_tcsc = tcsc[tcsc_branches, TCSC_THYRISTOR_FIRING_ANGLE].real
    tcsc_x_l_pu = tcsc[tcsc_branches, TCSC_X_L].real #### should not this be imag ?
    tcsc_x_cvar_pu = tcsc[tcsc_branches, TCSC_X_CVAR].real
    num_tcsc = len(x_control_tcsc)
    num_tcsc_controllable = len(x_control_tcsc[tcsc_controllable])
    any_tcsc = num_tcsc > 0
    any_tcsc_controllable = num_tcsc_controllable > 0

    ## ssc
    ssc_branches = flatnonzero(nan_to_num(ssc[:, SSC_STATUS]))
    ssc_fb = ssc[ssc_branches, [SSC_BUS]].real.astype(np.int64)
    ssc_tb = ssc[ssc_branches, [SSC_INTERNAL_BUS]].real.astype(np.int64)
    size_y = Ybus.shape[0]
    num_ssc = len(ssc_fb)
    ssc_controllable = ssc[ssc_branches, SSC_CONTROLLABLE].real.astype(bool)
    x_control_ssc = ssc[ssc_branches, SSC_X_CONTROL_VM].real * np.exp(1j * ssc[ssc_branches, SSC_X_CONTROL_VA].real)
    num_ssc_controllable = len(x_control_ssc[ssc_controllable])
    ssc_set_vm_pu = ssc[ssc_branches[ssc_controllable], SSC_SET_VM_PU]
    ssc_mode_ac = np.zeros_like(ssc_set_vm_pu)
    ssc_y_pu = 1/(ssc[ssc_branches, SSC_R].real + 1j * ssc[ssc_branches, SSC_X].real)
    any_ssc = num_ssc > 0
    any_ssc_controllable = num_ssc_controllable > 0

    ## vsc
    bus_dc[:, DC_BUS_TYPE] = DC_P # todo vsc
    vsc_branches = flatnonzero(nan_to_num(vsc[:, VSC_STATUS]))
    vsc_fb = vsc[vsc_branches, [VSC_BUS]].real.astype(np.int64)
    vsc_tb = vsc[vsc_branches, [VSC_INTERNAL_BUS]].real.astype(np.int64)
    vsc_dc_fb = vsc[vsc_branches, [VSC_BUS_DC]].real.astype(np.int64)
    vsc_dc_tb = vsc[vsc_branches, [VSC_INTERNAL_BUS_DC]].real.astype(np.int64)
    num_vsc = len(vsc_fb)
    vsc_controllable = vsc[vsc_branches, VSC_CONTROLLABLE].real.astype(bool)
    num_vsc_controllable = sum(vsc_controllable)
    vsc_mode_ac = vsc[vsc_branches[vsc_controllable], VSC_MODE_AC]
    vsc_value_ac = vsc[vsc_branches[vsc_controllable], VSC_VALUE_AC]
    vsc_mode_dc = vsc[vsc_branches[vsc_controllable], VSC_MODE_DC]
    vsc_value_dc = vsc[vsc_branches[vsc_controllable], VSC_VALUE_DC]
    vsc_y_pu = 1 / (vsc[vsc_branches, VSC_R] + 1j * vsc[vsc_branches, VSC_X].real)
    # make sure this is not 0 in the inputs!
    with np.errstate(all="raise"):
        vsc_g_pu = 1 / vsc[vsc_branches, VSC_R_DC]
    vsc_gl_pu = vsc[vsc_branches, VSC_PL_DC] # / np.square(bus_dc[vsc[vsc_branches, VSC_BUS_DC].astype(np.int64), DC_BASE_KV])
    any_vsc = num_vsc > 0
    any_vsc_controllable = num_vsc_controllable > 0
    vsc_dc_mode_p = (vsc_mode_dc == VSC_MODE_DC_P) & (vsc_mode_ac != VSC_MODE_AC_SL)
    vsc_dc_mode_v = (vsc_mode_dc == VSC_MODE_DC_V) & (vsc_mode_ac != VSC_MODE_AC_SL)

    hvdc_fb = branch_dc[:, DC_F_BUS].astype(np.int64)
    hvdc_tb = branch_dc[:, DC_T_BUS].astype(np.int64)

    baseR = 110 ** 2 / baseMVA
    hvdc_branches = np.flatnonzero(branch_dc[:, DC_BR_STATUS] != 0)
    num_branch_dc = len(hvdc_branches) + num_vsc  # internal vsc resistance is also a dc branch
    with np.errstate(all="raise"):
        hvdc_y_pu = 1 / branch_dc[hvdc_branches, DC_BR_R]
    relevant_bus_dc = flatnonzero(bus_dc[:, DC_BUS_TYPE] != DC_NONE)
    num_bus_dc = len(relevant_bus_dc)
    any_branch_dc = num_branch_dc > 0
    P_dc = -bus_dc[relevant_bus_dc, DC_PD]  # load is negative here
    p_set_point_index = vsc_controllable & (vsc_mode_dc == VSC_MODE_DC_P) & (vsc_mode_ac != VSC_MODE_AC_SL)
    # P_dc[vsc[p_set_point_index, VSC_BUS_DC].astype(np.int64)] = -vsc_value_dc[vsc_mode_dc == 1]  # todo sum by group
    # todo vsc
    vsc_group_buses_p, P_dc_sum, vsc_group_buses_p_number = _sum_by_group(vsc_dc_tb[p_set_point_index], -vsc_value_dc[p_set_point_index], np.ones(sum(p_set_point_index)))
    #P_dc[vsc_group_buses_p] = P_dc_sum
    vsc_slack_p_dc_bus, _, _ = _sum_by_group(vsc_dc_tb[vsc_mode_ac == VSC_MODE_AC_SL], vsc_dc_tb[vsc_mode_ac == VSC_MODE_AC_SL], vsc_dc_tb[vsc_mode_ac == VSC_MODE_AC_SL])
    P_dc_sum_sl = P_dc[vsc_slack_p_dc_bus].copy()  # later used in mismatch for vsc
    #vsc_group_buses_ref, _, vsc_group_buses_ref_number = _sum_by_group(vsc_dc_tb[p_set_point_index], -vsc_value_dc[vsc_mode_dc == 1], np.ones(sum(p_set_point_index)))

    # J for HVDC is expanded by the number of DC "P" buses (added below)
    num_facts_controllable = num_svc_controllable + num_tcsc_controllable # + 2 * num_ssc_controllable
    num_facts = num_svc + num_tcsc + num_ssc  # todo schould it be num_bus?


    #
    # tcsc_in_pq_f = np.isin(branch[tcsc_branches, F_BUS].real.astype(np.int64), pq)
    # tcsc_in_pq_t = np.isin(branch[tcsc_branches, T_BUS].real.astype(np.int64), pq)
    # tcsc_in_pvpq_f = np.isin(branch[tcsc_branches, F_BUS].real.astype(np.int64), pvpq)
    # tcsc_in_pvpq_t = np.isin(branch[tcsc_branches, T_BUS].real.astype(np.int64), pvpq)
    # # else:
    # #   tcsc_fb = tcsc_tb = tcsc_i = tcsc_j = None

    ############

    # initialize
    i = 0
    V = V0
    V_dc = np.ones(num_bus_dc, dtype=np.float64)  # initial voltage vector for the DC line
    v_set_point_index = vsc_controllable & (vsc_mode_dc == 0)
    if len(vsc_value_dc[v_set_point_index]) > 0:
        V_dc[:] = np.mean(vsc_value_dc[v_set_point_index])
        # V_dc[vsc[v_set_point_index, VSC_BUS_DC].astype(np.int64)] = vsc_value_dc[v_set_point_index]

    Ybus_svc = makeYbus_svc(Ybus, x_control_svc, svc_x_l_pu, svc_x_cvar_pu, svc_buses)
    y_tcsc_pu = -1j * calc_y_svc_pu(x_control_tcsc, tcsc_x_l_pu, tcsc_x_cvar_pu)
    Ybus_tcsc = make_Ybus_facts(tcsc_fb, tcsc_tb, y_tcsc_pu, Ybus.shape[0])
    # Ybus_tcsc = makeYbus_tcsc(Ybus, x_control_tcsc, tcsc_x_l_pu, tcsc_x_cvar_pu, tcsc_fb, tcsc_tb)
    # SSC
    Ybus_ssc_not_controllable, Ybus_ssc_controllable, Ybus_ssc = \
        makeYbus_ssc_vsc(Ybus, ssc_y_pu, ssc_fb, ssc_tb, ssc_controllable)
    # VSC
    Ybus_vsc_not_controllable, Ybus_vsc_controllable, Ybus_vsc = \
        makeYbus_ssc_vsc(Ybus, vsc_y_pu, vsc_fb, vsc_tb, vsc_controllable)
    # HVDC
    Ybus_vsc_dc = make_Ybus_facts(vsc_dc_fb, vsc_dc_tb, vsc_g_pu, num_bus_dc, ysf_pu=vsc_gl_pu, dtype=np.float64)
    Ybus_hvdc = make_Ybus_facts(hvdc_fb, hvdc_tb, hvdc_y_pu, num_bus_dc, dtype=np.float64) + Ybus_vsc_dc
    Yf_vsc_dc, Yt_vsc_dc = make_Yft_facts(vsc_dc_fb, vsc_dc_tb, vsc_g_pu, num_bus_dc, ysf_pu=vsc_gl_pu)
    Yf_vsc, Yt_vsc = make_Yft_facts(vsc_fb, vsc_tb, vsc_y_pu, Ybus_vsc.shape[0])

    # to avoid non-convergence due to zero-terms in the Jacobian:
    if any_tcsc_controllable and np.all(V[tcsc_fb] == V[tcsc_tb]):
        V[tcsc_tb] -= 0.01 + 0.001j
    # if any_ssc_controllable and np.all(V[ssc_fb[ssc_controllable]] == V[ssc_tb[ssc_controllable]]):
    #     V[ssc_tb[ssc_controllable]] -= 0.01 + 0.001j
    # if any_vsc_controllable and np.all(V[vsc_fb[vsc_controllable]] == V[vsc_tb[vsc_controllable]]):
    #     V[vsc_tb[vsc_controllable]] -= 0.01 + 0.001j

    Va = angle(V)
    Vm = abs(V)
    dVa, dVm = None, None
    if iwamoto:
        dVm, dVa = zeros_like(Vm), zeros_like(Va)

    if v_debug:
        Vm_it = Vm.copy()
        Va_it = Va.copy()
    else:
        Vm_it = None
        Va_it = None

    # set up indexing for updating V
    if dist_slack and len(ref) > 1:
        pv = r_[ref[1:], pv]
        ref = ref[[0]]

    pvpq = r_[pv, pq]
    # reference buses are always at the top, no matter where they are in the grid (very confusing...)
    # so in the refpvpq, the indices must be adjusted so that ref bus(es) starts with 0
    # todo: is it possible to simplify the indices/lookups and make the code clearer?
    # for columns: columns are in the normal order in Ybus; column numbers for J are reduced by 1 internally
    refpvpq = r_[ref, pvpq]
    # generate lookup pvpq -> index pvpq (used in createJ):
    #   shows for a given row from Ybus, which row in J it becomes
    #   e.g. the first row in J is a PV bus. If the first PV bus in Ybus is in the row 2, the index of the row in Jbus must be 0.
    #   pvpq_lookup will then have a 0 at the index 2
    pvpq_lookup = zeros(max((Ybus + Ybus_svc + Ybus_tcsc + Ybus_ssc + Ybus_vsc).indices) + 1, dtype=np.int64)
    if dist_slack:
        # slack bus is relevant for the function createJ_ds
        pvpq_lookup[refpvpq] = arange(len(refpvpq))
    else:
        pvpq_lookup[pvpq] = arange(len(pvpq))

    pq_lookup = zeros(max(refpvpq) + 1, dtype=np.int64)
    pq_lookup[pq] = arange(len(pq))

    # lookups for the DC system. For now, we assume the DC "from" bus is always slack
    dc_ref, dc_b2b, dc_p = bustypes_dc(bus_dc)
    num_dc_p = len(dc_p)
    dc_refb2bp = r_[dc_ref, dc_b2b, dc_p]
    if len(dc_refb2bp) > 0:
        dc_p_lookup = zeros(max(dc_refb2bp) + 1, dtype=np.int64)
        dc_p_lookup[dc_p] = arange(len(dc_p))
        dc_ref_lookup = zeros(max(dc_refb2bp) + 1, dtype=np.int64)
        dc_ref_lookup[dc_ref] = arange(len(dc_ref))
        dc_b2b_lookup = zeros(max(dc_refb2bp) + 1, dtype=np.int64)
        dc_b2b_lookup[dc_b2b] = arange(len(dc_b2b))
        num_facts_controllable += num_dc_p
    else:
        dc_p_lookup = np.array([], dtype=np.int64)
        dc_ref_lookup = np.array([], dtype=np.int64)
        dc_b2b_lookup = np.array([], dtype=np.int64)

    # get jacobian function
    createJ = get_fastest_jacobian_function(pvpq, pq, numba, dist_slack)

    nref = len(ref)
    npv = len(pv)
    npq = len(pq)
    # todo: tidy up the indices below
    j0 = 0
    j1 = nref if dist_slack else 0
    j2 = j1 + npv  # j1:j2 - V angle of pv buses
    j3 = j2
    j4 = j2 + npq  # j3:j4 - V angle of pq buses
    j5 = j4
    j6 = j4 + npq  # j5:j6 - V mag of pq buses
    j6a = j6 + num_svc_controllable  # svc
    j6b = j6a + num_tcsc_controllable  # tcsc
    j6c = j6b + 0 # num_ssc_controllable # ssc Vq angle
    j7 = j6c
    j6d = j6c + 0 # num_ssc_controllable # ssc Vq mag
    j6e = j6d + (num_dc_p if any_branch_dc else 0)  # V of the p-buses of the DC grid
    j8 = j6e

    # make initial guess for the slack
    slack = (gen[:, PG].sum() - bus[:, PD].sum()) / baseMVA
    # evaluate F(x0)
    any_facts_controllable = num_facts_controllable > 0
    F = _evaluate_Fx(Ybus + Ybus_svc + Ybus_tcsc + Ybus_ssc + Ybus_vsc, V, Sbus, ref, pv, pq, slack_weights, dist_slack, slack)
    if any_facts_controllable or any_ssc_controllable or any_vsc_controllable:
        mis_facts = _evaluate_Fx_facts(V, pq, svc_buses[svc_controllable], svc_set_vm_pu[svc_controllable],
                                       tcsc_controllable, tcsc_set_p_pu, tcsc_tb, Ybus_tcsc, ssc_fb[ssc_controllable],
                                       ssc_tb[ssc_controllable], Ybus_ssc, ssc_controllable, ssc_set_vm_pu, F,
                                       pq_lookup, vsc_controllable, vsc_fb, vsc_tb, Ybus_vsc, Ybus_vsc_dc, Yf_vsc_dc,
                                       Yt_vsc_dc, Yf_vsc, vsc_mode_ac, vsc_mode_dc, vsc_value_ac, vsc_value_dc,
                                       vsc_dc_fb, vsc_dc_tb, vsc_dc_mode_v, vsc_dc_mode_p, vsc_gl_pu, V_dc, Ybus_hvdc,
                                       num_branch_dc, P_dc, dc_p, dc_ref, dc_b2b, dc_p_lookup, dc_ref_lookup,
                                       dc_b2b_lookup, P_dc_sum_sl)
        F = r_[F, mis_facts]

    T_base = 100  # T in p.u. for better convergence
    T = 20 / T_base
    r_theta_pu = 0
    if tdpf:
        if len(pq) > 0:
            pq_lookup = zeros(max(refpvpq) + 1, dtype=np.int64)  # for TDPF
            pq_lookup[pq] = arange(len(pq))
        else:
            pq_lookup = array([])
        tdpf_update_r_theta = options.get('tdpf_update_r_theta', True)
        tdpf_delay_s = options.get('tdpf_delay_s')
        tdpf_lines = flatnonzero(nan_to_num(branch[:, TDPF]))
        # set up the necessary parameters for TDPF:
        T0 = branch[tdpf_lines, T_START_C].real / T_base
        t_ref_pu = branch[tdpf_lines, T_REF_C].real / T_base
        t_air_pu = branch[tdpf_lines, T_AMBIENT_C].real / T_base
        alpha_pu = branch[tdpf_lines, ALPHA].real * T_base

        i_max_a = branch[tdpf_lines, RATE_I_KA].real * 1e3
        v_base_kv = bus[branch[tdpf_lines, F_BUS].real.astype(np.int64), BASE_KV]
        z_base_ohm = square(v_base_kv) / baseMVA
        r_ref_pu = branch[tdpf_lines, BR_R_REF_OHM_PER_KM].real * branch[tdpf_lines, BR_LENGTH_KM].real / z_base_ohm
        i_base_a = baseMVA / (v_base_kv * sqrt(3)) * 1e3
        i_max_pu = i_max_a / i_base_a
        # p_rated_loss_pu = square(i_max_pu) * r_ref_pu * (1 + alpha_pu * (25/T_base+t_air_pu - t_ref_pu))
        # p_rated_loss_mw = square(branch[tdpf_lines, RATE_I_KA].real * sqrt(3)) * branch[tdpf_lines, BR_R_REF_OHM_PER_KM].real * branch[tdpf_lines, BR_LENGTH_KM].real * (1 + alpha_pu * (25/T_base+t_air_pu - t_ref_pu))
        # assert np.allclose(p_rated_loss_mw / baseMVA, p_rated_loss_pu)
        # defined in Frank et.al. as T_Rated_Rise / p_rated_loss. Given in net.line based on °C, kA, kV:
        r_theta_pu = branch[tdpf_lines, R_THETA].real * baseMVA / T_base
        x = branch[tdpf_lines, BR_X].real

        # calculate parameters for J:
        Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
        # todo: add parameters to the create function
        a0, a1, a2, tau = calc_a0_a1_a2_tau(t_air_pu=t_air_pu, t_max_pu=80 / T_base, t_ref_pu=t_ref_pu,
                                            r_ref_ohm_per_m=1e-3 * branch[tdpf_lines, BR_R_REF_OHM_PER_KM].real,
                                            conductor_outer_diameter_m=branch[tdpf_lines, OUTER_DIAMETER_M].real,
                                            mc_joule_per_m_k=branch[tdpf_lines, MC_JOULE_PER_M_K].real,
                                            wind_speed_m_per_s=branch[tdpf_lines, WIND_SPEED_MPS].real,
                                            wind_angle_degree=branch[tdpf_lines, WIND_ANGLE_DEGREE].real,
                                            s_w_per_square_meter=branch[tdpf_lines, SOLAR_RADIATION_W_PER_SQ_M].real,
                                            alpha_pu=alpha_pu, solar_absorptivity=branch[tdpf_lines, GAMMA].real,
                                            emissivity=branch[tdpf_lines, EPSILON].real, T_base=T_base,
                                            i_base_a=i_base_a)
        g, b = calc_g_b(r_ref_pu, x)
        i_square_pu, p_loss_pu = calc_i_square_p_loss(branch, tdpf_lines, g, b, Vm, Va)
        if tdpf_update_r_theta:
            r_theta_pu = calc_r_theta(t_air_pu, a0, a1, a2, i_square_pu, p_loss_pu)
        # initial guess for T:
        # T = calc_T_frank(p_loss_pu, t_air_pu, r_theta_pu, tdpf_delay_s, T0, tau)
        T = T0.copy()  # better for e.g. timeseries calculation
        F_t = zeros(len(branch))
        # F_t[tdpf_lines] = T - T0
        F = r_[F, F_t]

    converged = _check_for_convergence(F, tol)

    Ybus = Ybus.tocsr()
    J = None


    # do Newton iterations
    while (not converged and i < max_it):
        # update iteration counter
        i = i + 1

        if tdpf:
            # update the R, g, b for the tdpf_lines, and the Y-matrices
            # here: limit the change of the R to reflect a realistic range of values for T to avoid numerical issues
            branch[tdpf_lines, BR_R] = r = r_ref_pu * (1 + alpha_pu * np.clip(np.nan_to_num(T - t_ref_pu), -50, 250 / T_base))
            # todo expansion with SSC and VSC (that are not controllable)
            Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
            g, b = calc_g_b(r, x)

        # todo: adjust the SSC J function to take care about the Ybus_ssc_not_controllable instead
        J = create_jacobian_matrix(Ybus+Ybus_ssc_not_controllable+Ybus_vsc_not_controllable, V, ref, refpvpq, pvpq, pq, createJ, pvpq_lookup, nref, npv, npq, numba, slack_weights, dist_slack)

        if tdpf:
            # p.u. values for T, a1, a2, I, S
            # todo: distributed_slack works fine if sn_mva is rather high (e.g. 1000), otherwise no convergence. Why?
            J = create_J_tdpf(branch, tdpf_lines, alpha_pu, r_ref_pu, refpvpq if dist_slack else pvpq, pq, pvpq_lookup,
                              pq_lookup, tau, tdpf_delay_s, Vm, Va, r_theta_pu, J, r, x, g)

        if any_facts_controllable:
            K_J = vstack([eye(J.shape[0], format="csr"),
                          csr_matrix((num_facts_controllable, J.shape[0]))], format="csr")
            J = K_J * J * K_J.T  # this extends the J matrix with 0-rows and 0-columns
        if any_svc:
            J_m_svc = create_J_modification_svc(J, svc_buses, refpvpq if dist_slack else pvpq, pq, pq_lookup, Vm,
                                                x_control_svc, svc_x_l_pu, svc_x_cvar_pu, num_svc_controllable,
                                                svc_controllable)
            J = J + J_m_svc
        if any_tcsc:
            J_m_tcsc = create_J_modification_tcsc(J, V, y_tcsc_pu, x_control_tcsc, tcsc_controllable, tcsc_x_l_pu,
                                                  tcsc_fb, tcsc_tb, refpvpq if dist_slack else pvpq, pq, pvpq_lookup,
                                                  pq_lookup, num_svc_controllable)
            J = J + J_m_tcsc
        if any_ssc_controllable:
            J_m_ssc = create_J_modification_ssc_vsc(J, V, Vm, ssc_y_pu[ssc_controllable], ssc_fb[ssc_controllable],
                                                    ssc_tb[ssc_controllable], refpvpq if dist_slack else pvpq, pq,
                                                    pvpq_lookup, pq_lookup, ssc_mode_ac == 0,
                                                    np.full_like(ssc_mode_ac, False, dtype=bool))
            J = J + J_m_ssc
        if any_vsc_controllable:
            J_m_vsc = create_J_modification_ssc_vsc(J, V, Vm, vsc_y_pu[vsc_controllable], vsc_fb[vsc_controllable],
                                                    vsc_tb[vsc_controllable], refpvpq if dist_slack else pvpq, pq,
                                                    pvpq_lookup, pq_lookup,
                                                    (vsc_mode_ac == VSC_MODE_AC_V) | (vsc_mode_ac == VSC_MODE_AC_SL),
                                                    vsc_mode_ac == VSC_MODE_AC_SL)
            J = J + J_m_vsc
        if any_branch_dc:
            J_m_hvdc = create_J_modification_hvdc(J, V_dc, Ybus_hvdc, Ybus_vsc_dc, vsc_g_pu, vsc_gl_pu, dc_p,
                                                  dc_p_lookup, vsc_dc_fb, vsc_dc_tb, vsc_dc_mode_v, vsc_dc_mode_p)
            J = J + J_m_hvdc

        dx = -1 * spsolve(J, F, permc_spec=permc_spec, use_umfpack=use_umfpack)
        # update voltage
        if dist_slack:
            slack = slack + dx[j0:j1]
        if npv and not iwamoto:
            Va[pv] = Va[pv] + dx[j1:j2]
        if npq and not iwamoto:
            Va[pq] = Va[pq] + dx[j3:j4]
            Vm[pq] = Vm[pq] + dx[j5:j6]
        if any_svc_controllable:
            x_control_svc[svc_controllable] += dx[j6:j6a]
        if any_tcsc_controllable:
            x_control_tcsc[tcsc_controllable] += dx[j6a:j6b]
        if any_branch_dc:
            V_dc[dc_p] += dx[j6d:j6e].real

        if tdpf:
            T = T + dx[j8:][tdpf_lines]

        # iwamoto multiplier to increase convergence
        if iwamoto and not tdpf:
            Vm, Va = _iwamoto_step(Ybus, J, F, dx, pq, npv, npq, dVa, dVm, Vm, Va, pv, j1, j2, j3, j4, j5, j6)

        V = Vm * exp(1j * Va)
        Vm = abs(V)  # update Vm and Va again in case
        Va = angle(V)  # we wrapped around with a negative Vm

        if v_debug:
            Vm_it = column_stack((Vm_it, Vm))
            Va_it = column_stack((Va_it, Va))

        if voltage_depend_loads:
            Sbus = makeSbus(baseMVA, bus, gen, vm=Vm)

        if any_svc_controllable:
            Ybus_svc = makeYbus_svc(Ybus, x_control_svc, svc_x_l_pu, svc_x_cvar_pu, svc_buses)

        if any_tcsc_controllable:
            y_tcsc_pu = -1j * calc_y_svc_pu(x_control_tcsc, tcsc_x_l_pu, tcsc_x_cvar_pu)
            Ybus_tcsc = make_Ybus_facts(tcsc_fb, tcsc_tb, y_tcsc_pu, Ybus.shape[0])
            # Ybus_tcsc = makeYbus_tcsc(Ybus, x_control_tcsc, tcsc_x_l_pu, tcsc_x_cvar_pu, tcsc_fb, tcsc_tb)
        # Ybus_ssc does not change
        # if any_ssc_controllable:
        #     Ybus_ssc = makeYbus_ssc(Ybus, ssc_y_pu, ssc_fb, ssc_tb, any_ssc)

        F = _evaluate_Fx(Ybus + Ybus_svc + Ybus_tcsc + Ybus_ssc + Ybus_vsc, V, Sbus, ref, pv, pq, slack_weights, dist_slack, slack)
        if any_facts_controllable or any_ssc_controllable or any_vsc_controllable:
            mis_facts = _evaluate_Fx_facts(V, pq, svc_buses[svc_controllable], svc_set_vm_pu[svc_controllable],
                                           tcsc_controllable, tcsc_set_p_pu, tcsc_tb, Ybus_tcsc,
                                           ssc_fb[ssc_controllable], ssc_tb[ssc_controllable], Ybus_ssc,
                                           ssc_controllable, ssc_set_vm_pu, F, pq_lookup, vsc_controllable,
                                           vsc_fb[vsc_controllable], vsc_tb[vsc_controllable], Ybus_vsc, Ybus_vsc_dc,
                                           Yf_vsc_dc, Yt_vsc_dc, Yf_vsc, vsc_mode_ac, vsc_mode_dc, vsc_value_ac,
                                           vsc_value_dc, vsc_dc_fb, vsc_dc_tb, vsc_dc_mode_v, vsc_dc_mode_p, vsc_gl_pu,
                                           V_dc, Ybus_hvdc, num_branch_dc, P_dc, dc_p, dc_ref, dc_b2b, dc_p_lookup,
                                           dc_ref_lookup, dc_b2b_lookup, P_dc_sum_sl)
            F = r_[F, mis_facts]

        if tdpf:
            i_square_pu, p_loss_pu = calc_i_square_p_loss(branch, tdpf_lines, g, b, Vm, Va)
            if tdpf_update_r_theta:
                r_theta_pu = calc_r_theta(t_air_pu, a0, a1, a2, i_square_pu, p_loss_pu)
            T_calc = calc_T_frank(p_loss_pu, t_air_pu, r_theta_pu, tdpf_delay_s, T0, tau)
            F_t[tdpf_lines] = T - T_calc
            F = r_[F, F_t]

        converged = _check_for_convergence(F, tol)

    # write q_svc, x_control in ppc["bus"] and then later calculate q_mvar for net.res_shunt
    # todo: move to pf.run_newton_raphson_pf.ppci_to_pfsoln
    if any_svc:
        y_svc_pu = calc_y_svc_pu(x_control_svc, svc_x_l_pu, svc_x_cvar_pu)
        q_svc_pu = np.square(np.abs(V[svc_buses])) * y_svc_pu
        svc[svc_idx, SVC_Q] = q_svc_pu * baseMVA
        svc[svc_idx, SVC_THYRISTOR_FIRING_ANGLE] = x_control_svc
        svc[svc_idx, SVC_X_PU] = 1 / y_svc_pu

    # todo: move to pf.run_newton_raphson_pf.ppci_to_pfsoln
    if any_tcsc:
        Yf_tcsc, Yt_tcsc = makeYft_tcsc(Ybus_tcsc, tcsc_fb, tcsc_tb, x_control_tcsc, tcsc_x_l_pu, tcsc_x_cvar_pu)
        # todo use make_Ybus_facts, make_Yft_facts
        baseI = baseMVA / (bus[tcsc_tb, BASE_KV] * sqrt(3))
        i_tcsc_f = Yf_tcsc.dot(V)
        i_tcsc_t = Yt_tcsc.dot(V)
        s_tcsc_f = np.conj(i_tcsc_f) * V[tcsc_fb] * baseMVA
        s_tcsc_t = np.conj(i_tcsc_t) * V[tcsc_tb] * baseMVA
        tcsc[tcsc_branches, TCSC_THYRISTOR_FIRING_ANGLE] = x_control_tcsc
        tcsc[tcsc_branches, TCSC_PF] = s_tcsc_f.real
        tcsc[tcsc_branches, TCSC_QF] = s_tcsc_f.imag
        tcsc[tcsc_branches, TCSC_PT] = s_tcsc_t.real
        tcsc[tcsc_branches, TCSC_QT] = s_tcsc_t.imag
        tcsc[tcsc_branches, TCSC_IF] = np.abs(i_tcsc_f) * baseI
        tcsc[tcsc_branches, TCSC_IT] = np.abs(i_tcsc_t) * baseI
        tcsc[tcsc_branches, TCSC_X_PU] = 1 / calc_y_svc_pu(x_control_tcsc, tcsc_x_l_pu, tcsc_x_cvar_pu)

    # todo: move to pf.run_newton_raphson_pf.ppci_to_pfsoln
    if any_ssc:
        # todo use make_Ybus_facts, make_Yft_facts
        Yf_ssc, Yt_ssc = make_Yft_facts(ssc_fb, ssc_tb, ssc_y_pu, Ybus_ssc.shape[0])
        s_ssc_f = conj(Yf_ssc.dot(V)) * V[ssc_fb] * baseMVA
        s_ssc_t = conj(Yt_ssc.dot(V)) * V[ssc_tb] * baseMVA
        ssc[ssc_branches, SSC_Q] = s_ssc_f.imag

    # todo move this to pfsoln
    if any_vsc:
        # Yf_vsc, Yt_vsc = make_Yft_facts(vsc_fb, vsc_tb, vsc_y_pu, Ybus_vsc.shape[0])
        s_vsc_f = conj(Yf_vsc.dot(V)) * V[vsc_fb] * baseMVA
        s_vsc_t = conj(Yt_vsc.dot(V)) * V[vsc_tb] * baseMVA
        vsc[vsc_branches, VSC_P] = s_vsc_f.real
        vsc[vsc_branches, VSC_Q] = s_vsc_f.imag

        # Yf_vsc_dc, Yt_vsc_dc = make_Yft_facts(vsc_dc_fb, vsc_dc_tb, vsc_g_pu, num_bus_dc)
        i_vsc_dc_f = Yf_vsc_dc.dot(V_dc)
        p_vsc_dc_f = i_vsc_dc_f * V_dc[vsc_dc_fb] * baseMVA
        i_vsc_dc_t = Yt_vsc_dc.dot(V_dc)
        p_vsc_dc_t = i_vsc_dc_t * V_dc[vsc_dc_tb] * baseMVA

        vsc[vsc_branches, VSC_P_DC] = p_vsc_dc_f
        # print(p_vsc_dc_f, p_vsc_dc_t)
        # print(s_vsc_f.real, s_vsc_t.real)
        # no_load_losses = vsc_gl_pu * np.square(V_dc[vsc_dc_fb])
        # print(f"{no_load_losses=}")
        # print(f"{p_vsc_dc_t + p_vsc_dc_f}")
        # print(f"{p_vsc_dc_t + p_vsc_dc_f + no_load_losses}")

    if len(relevant_bus_dc) > 0:
        # duplication with "if any branch dc" because it is possible to have only vsc and no dc lines:
        bus_dc[relevant_bus_dc, DC_VM] = V_dc
        # bus_dc[relevant_bus_dc, DC_PD] = -P_dc

    # todo move this to pfsoln
    if any_branch_dc:
        Yf_hvdc, Yt_hvdc = make_Yft_facts(hvdc_fb, hvdc_tb, hvdc_y_pu, num_bus_dc)
        # Pbus_dc = V_dc * (Ybus_hvdc + Ybus_vsc_dc).dot(V_dc)
        Pbus_dc = V_dc * Ybus_hvdc.dot(V_dc)
        bus_dc[relevant_bus_dc, DC_PD] = Pbus_dc
        i_hvdc_f = Yf_hvdc.dot(V_dc)
        p_hvdc_f = i_hvdc_f * V_dc[hvdc_fb] * baseMVA
        i_hvdc_t = Yt_hvdc.dot(V_dc)
        p_hvdc_t = i_hvdc_t * V_dc[hvdc_tb] * baseMVA
        branch_dc[hvdc_branches, DC_PF] = p_hvdc_f.real
        branch_dc[hvdc_branches, DC_IF] = i_hvdc_f.real
        branch_dc[hvdc_branches, DC_PT] = p_hvdc_t.real
        branch_dc[hvdc_branches, DC_IT] = i_hvdc_t.real

    # todo: remove this
    # Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
    # Sf =  conj(Yf.dot(V)) * V[[0, 0]] * baseMVA
    # St =  conj(Yt.dot(V)) * V[[1, 2]] * baseMVA
    # print("P from:", Sf.real)
    # print("P to:", St.real)
    # print("Sbus:", Sbus.real)

    # because we now have updates of the Ybus matrices due to TDPF, SVC, TCSC, SSC,
    # we are interested in storing them for later use:
    ppci["internal"]["Ybus"] = Ybus
    ppci["internal"]["Ybus_svc"] = Ybus_svc
    ppci["internal"]["Ybus_tcsc"] = Ybus_tcsc
    ppci["internal"]["Ybus_ssc"] = Ybus_ssc
    ppci["internal"]["Ybus_vsc"] = Ybus_vsc
    ppci["internal"]["Ybus_hvdc"] = Ybus_hvdc
    # ppci["internal"]["Yf_tcsc"] = Yf_tcsc
    # ppci["internal"]["Yt_tcsc"] = Yt_tcsc
    ppci["internal"]["tcsc_fb"] = tcsc_fb
    ppci["internal"]["tcsc_tb"] = tcsc_tb

    return V, converged, i, J, Vm_it, Va_it, r_theta_pu / baseMVA * T_base, T * T_base


def _evaluate_Fx(Ybus, V, Sbus, ref, pv, pq, slack_weights=None, dist_slack=False, slack=None):
    # evalute F(x)
    if dist_slack:
        # we include the slack power (slack * contribution factors) in the mismatch calculation
        mis = V * conj(Ybus * V) - Sbus + slack_weights * slack
        F = r_[mis[ref].real, mis[pv].real, mis[pq].real, mis[pq].imag]
    else:
        mis = V * conj(Ybus * V) - Sbus
        F = r_[mis[pv].real, mis[pq].real, mis[pq].imag]
    return F


def _evaluate_Fx_facts(V,pq ,svc_buses=None, svc_set_vm_pu=None, tcsc_controllable=None, tcsc_set_p_pu=None, tcsc_tb=None,
                       Ybus_tcsc=None, ssc_fb=None, ssc_tb=None, Ybus_ssc=None,
                       ssc_controllable=None, ssc_set_vm_pu=None, old_F=None, pq_lookup=None,
                       vsc_controllable=None, vsc_fb=None, vsc_tb=None,
                       Ybus_vsc=None, Ybus_vsc_dc=None, Yf_vsc_dc=None, Yt_vsc_dc=None, Yf_vsc=None,
                       vsc_mode_ac=None, vsc_mode_dc=None, vsc_value_ac=None, vsc_value_dc=None, vsc_dc_fb=None, vsc_dc_tb=None,
                       vsc_dc_mode_v=None, vsc_dc_mode_p=None, vsc_gl_pu=None,
                       V_dc=None, Ybus_hvdc=None, num_branch_dc=None, P_dc=None, dc_p=None, dc_ref=None, dc_b2b=None,
                       dc_p_lookup=None, dc_ref_lookup=None, dc_b2b_lookup=None, P_dc_sum_sl=None):
    mis_facts = np.array([], dtype=np.float64)

    if svc_buses is not None and len(svc_buses) > 0:
        mis_facts = np.r_[mis_facts, np.abs(V[svc_buses]) - svc_set_vm_pu]

    if tcsc_tb is not None and len(tcsc_tb) > 0:
        if np.any(tcsc_controllable):
            Sbus_tcsc = V * conj(Ybus_tcsc * V)
            mis_tcsc = Sbus_tcsc[tcsc_tb[tcsc_controllable]].real - tcsc_set_p_pu
            mis_facts = np.r_[mis_facts, mis_tcsc]

    # if ssc_fb is not None and len(ssc_fb) > 0:
    if np.any(ssc_controllable):
        mis_ssc_v = np.abs(V[ssc_fb]) - ssc_set_vm_pu
        # mis_facts = np.r_[mis_facts, mis_ssc_p, mis_ssc_v]
        # old_F[-(len(pq)+len(ssc_fb))] = mis_ssc_p
        # old_F[-len(ssc_fb)] = mis_ssc_v
        old_F[-len(pq)+pq_lookup[ssc_tb]] = mis_ssc_v

    # todo: fix this condition (because there can be a mix of dc_p and dc_ref buses) - use np.where?
    if num_branch_dc > 0:  # todo if hvdc lines
        # This part only relevant for DC lines:
        Pbus_hvdc = V_dc * Ybus_hvdc.dot(V_dc)
        Pbus_vsc = V_dc * Ybus_vsc_dc.dot(V_dc)
        p_vsc_dc_f = V_dc[vsc_dc_fb] * Yf_vsc_dc.dot(V_dc)
        p_vsc_dc_t = V_dc[vsc_dc_tb] * Yt_vsc_dc.dot(V_dc)
        # vsc_dc_ref = vsc_mode_dc == VSC_MODE_DC_V
        # if np.any(vsc_dc_ref):
        #     P_dc[vsc_dc_tb[vsc_dc_ref]] = Pbus_hvdc[vsc_dc_fb[vsc_dc_ref]]
        # Pbus_hvdc = V_dc * (Ybus_hvdc+Ybus_vsc_dc).dot(V_dc)

        mis_hvdc = Pbus_hvdc - P_dc  # todo vsc
        # mis_hvdc = Pbus_hvdc - np.zeros_like(Pbus_hvdc)
        # mis_hvdc = np.zeros_like(Pbus_hvdc)
        if np.any(vsc_dc_mode_p):
            mis_hvdc[vsc_dc_tb[vsc_dc_mode_p]] = p_vsc_dc_f[vsc_dc_mode_p] - vsc_value_dc[vsc_dc_mode_p]

        if np.any(vsc_dc_mode_v):
            mis_hvdc[vsc_dc_tb[vsc_dc_mode_v]] = V_dc[vsc_dc_fb[vsc_dc_mode_v]] - vsc_value_dc[vsc_dc_mode_v]

        mis_facts = np.r_[mis_facts, mis_hvdc[dc_p]]
    else:
        Pbus_hvdc = -P_dc

    if np.any(vsc_controllable):
        Sbus_vsc = V * conj(Ybus_vsc * V)
        sf_vsc = V[vsc_fb] * conj(Yf_vsc.dot(V))
        ###### Mismatch for the first VSC variable:
        # the mismatch value is written for the aux bus, but calculated based on the AC bus
        ac_mode_v = (vsc_mode_ac == VSC_MODE_AC_V) | (vsc_mode_ac == VSC_MODE_AC_SL)
        if np.any(ac_mode_v):
            mis_vsc_v = np.abs(V[vsc_fb[ac_mode_v]]) - vsc_value_ac[ac_mode_v]
            old_F[-len(pq)+pq_lookup[vsc_tb[ac_mode_v]]] = mis_vsc_v

        # Mismatch for Q:
        ac_mode_q = vsc_mode_ac == VSC_MODE_AC_Q
        if np.any(ac_mode_q):
            # unique_vsc_q_bus, c_q, _ = _sum_by_group(vsc_fb[ac_mode_q], np.ones_like(vsc_fb[ac_mode_q]), np.ones_like(vsc_fb[ac_mode_q]))
            # count_q = np.zeros(vsc_fb.max() + 1, dtype=np.int64)
            # count_q[unique_vsc_q_bus] = c_q
            # mis_vsc_q = Sbus_vsc[vsc_fb[ac_mode_q]].imag / count_q[vsc_fb[ac_mode_q]] - vsc_value_ac[ac_mode_q]  # todo test for when they share same bus
            mis_vsc_q = sf_vsc.imag[ac_mode_q] - vsc_value_ac[ac_mode_q]  # todo test for when they share same bus
            old_F[-len(pq) + pq_lookup[vsc_tb[ac_mode_q]]] = mis_vsc_q


        ##### Mismatch for the second VSC variable:
        # Mismatch for AC slack - delta
        ac_mode_sl = vsc_mode_ac == VSC_MODE_AC_SL
        if np.any(ac_mode_sl):
            mis_vsc_delta = np.angle(V[vsc_fb[ac_mode_sl]]) - 0  # <- here we set delta set point to zero, but can be a parameter in the future
            old_F[-len(pq) * 2 + pq_lookup[vsc_tb[ac_mode_sl]]] = mis_vsc_delta
            # this connects the AC slack result and the DC bus P set-point:
            vsc_slack_p = -Sbus_vsc[vsc_tb[ac_mode_sl]].real
            vsc_slack_p_dc_bus, vsc_slack_p_dc, _ = _sum_by_group(vsc_dc_tb[ac_mode_sl], vsc_slack_p, vsc_slack_p)
            P_dc[vsc_slack_p_dc_bus] = P_dc_sum_sl + vsc_slack_p_dc

        # find the connection between the DC buses and VSC buses
        # find the slack DC buses
        # find the P DC buses
        vsc_set_p_pu = np.zeros(len(vsc_fb[vsc_controllable]), dtype=np.float64)  # power sign convention issue
        # vsc_dc_p_bus = vsc_dc_fb[vsc_controllable][np.isin(vsc_dc_fb[vsc_controllable], dc_p)]
        # vsc_dc_ref_bus = vsc_dc_fb[vsc_controllable][np.isin(vsc_dc_fb[vsc_controllable], dc_ref)]
        # vsc_dc_b2b_bus = vsc_dc_fb[vsc_controllable][np.isin(vsc_dc_fb[vsc_controllable], dc_b2b)]

        vsc_ac_sl = vsc_mode_ac == VSC_MODE_AC_SL
        vsc_dc_p = (vsc_mode_dc == VSC_MODE_DC_P) & (vsc_mode_ac != VSC_MODE_AC_SL)
        vsc_dc_ref = vsc_mode_dc == VSC_MODE_DC_V
        # vsc_b2b_p = np.isin(vsc_dc_fb, vsc_dc_b2b_bus) & (vsc_mode_dc == VSC_MODE_DC_P)
        # vsc_b2b_ref = np.isin(vsc_dc_fb, vsc_dc_b2b_bus) & (vsc_mode_dc == VSC_MODE_DC_V)

        # unique_vsc_dc_bus, c_ref, c_b2b_ref = _sum_by_group(vsc_dc_fb, vsc_dc_ref.astype(np.float64), vsc_b2b_ref.astype(np.float64))
        # count_ref = np.zeros(vsc_dc_fb.max() + 1, dtype=np.int64)
        # count_b2b_ref = np.zeros(vsc_dc_fb.max() + 1, dtype=np.int64)
        # count_ref[unique_vsc_dc_bus] = c_ref
        # count_b2b_ref[unique_vsc_dc_bus] = c_b2b_ref

        # unique_vsc_p_bus, c_p, _ = _sum_by_group(vsc_fb[vsc_dc_p], np.ones_like(vsc_fb[vsc_dc_p]), np.ones_like(vsc_fb[vsc_dc_p]))
        # count_p = np.zeros(vsc_fb.max() + 1, dtype=np.int64)
        # count_p[unique_vsc_p_bus] = c_p


        # todo fix this part
        # a configuration of B2B bus connecting also with DC lines could be implemented here in the future if needed:
        # if np.any(vsc_ac_sl):
        #     P_dc[vsc_dc_tb[vsc_ac_sl]] = -Sbus_vsc[vsc_tb[vsc_controllable]].real[vsc_ac_sl]
        if np.any(vsc_dc_p):
            # vsc_set_p_pu[vsc_dc_p] = -P_dc[dc_p][dc_p_lookup[vsc_dc_p_bus]]
            # vsc_set_p_pu[vsc_dc_p] = vsc_value_dc[vsc_dc_p] * count_p[vsc_fb[vsc_dc_p]] # todo test for when they share same bus
            # vsc_set_p_pu[vsc_dc_p] = vsc_value_dc[vsc_dc_p]  # todo consider count
            no_load_losses = vsc_gl_pu * np.square(V_dc[vsc_dc_fb])
            vsc_set_p_pu[vsc_dc_p] = -p_vsc_dc_t[vsc_dc_p] #+ no_load_losses[vsc_dc_p]
            # P_dc[dc_p] = -Sbus_vsc[vsc_tb[vsc_controllable]].real[vsc_dc_p]
        if np.any(vsc_dc_ref):
            # vsc_set_p_pu[vsc_dc_ref] = -Pbus_hvdc[dc_ref][dc_ref_lookup[vsc_dc_ref_bus]] / count_ref[vsc_dc_bus[vsc_dc_ref]]
            vsc_set_p_pu[vsc_dc_ref] = -Pbus_hvdc[vsc_dc_tb[vsc_dc_ref]] #- Pbus_hvdc[vsc_dc_fb[vsc_dc_ref]]
            # vsc_set_p_pu[vsc_dc_ref] = -P_dc[vsc_dc_tb[vsc_dc_ref]]
            # # vsc_set_p_pu[vsc_mode_dc == 1] = -P_dc[dc_ref][dc_ref_lookup[vsc_dc_ref_bus]]  # dc_p
            #
            # # Create a list of tuples containing the original value and its index
            # vsc_dc_bus_idx = np.array(list(enumerate(vsc_dc_bus)))[vsc_dc_ref]  ##TODO: consider define the vscs indices list instead
            #
            # # Sort the list of tuples based on the values
            # vsc_dc_bus_sorted = sorted(vsc_dc_bus_idx, key=lambda x: x[1])
            #
            # # Create the output array by extracting the indices from the sorted list of tuples
            # vsc_dc_bus_sorted_idx = [i for i, value in vsc_dc_bus_sorted]
            #
            # vsc_set_p_pu[vsc_dc_bus_sorted_idx] = -Pbus_hvdc[dc_ref][dc_ref_lookup[vsc_dc_ref_bus]] / count_ref[vsc_dc_bus[vsc_dc_ref]]

        # if len(dc_b2b):
        #     # vsc_set_p_pu[vsc_mode_dc == 1] = -P_dc[dc_ref][dc_ref_lookup[vsc_dc_ref_bus]]  # dc_p
        #     vsc_set_p_pu[vsc_b2b_p] = vsc_value_dc[vsc_b2b_p]
        #     # vsc_set_p_pu[vsc_b2b_ref] = P_dc[dc_b2b][dc_b2b_lookup[vsc_dc_b2b_bus[vsc_b2b_ref]]] / count_b2b_ref[vsc_dc_bus[vsc_b2b_ref]]
        #     vsc_set_p_pu[vsc_b2b_ref] = P_dc[dc_b2b][dc_b2b_lookup[vsc_dc_fb[vsc_b2b_ref]]] / count_b2b_ref[vsc_dc_fb[vsc_b2b_ref]]
        ####  here used vsc_tb refereing to the q bus
        # S_temp = Sbus_vsc.real
        # S_temp[vsc_tb[vsc_dc_p]] /= count_p[vsc_fb[vsc_dc_p]]
        # mis_vsc_p = S_temp[vsc_tb[vsc_controllable]] - vsc_set_p_pu  # this is coupling the AC and the DC sides
        mis_vsc_p = Sbus_vsc[vsc_tb[vsc_controllable]].real - vsc_set_p_pu  # this is coupling the AC and the DC sides # todo old
        # todo: adjust the lookup to work with 1) VSC at ext_grid bus 2) only 1 VSC connected to HVDC line
        old_F[-len(pq) * 2 + pq_lookup[vsc_tb[~ac_mode_sl]]] = mis_vsc_p[~ac_mode_sl]

    return mis_facts


def _check_for_convergence(F, tol):
    # calc infinity norm
    return linalg.norm(F, inf) < tol

