# -*- coding: utf-8 -*-

# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


"""Solves the power flow using a full Newton's method.
"""
import numpy as np
from numpy import float64, array, angle, sqrt, square, exp, linalg, conj, r_, Inf, arange, zeros, \
    max, zeros_like, column_stack, flatnonzero, nan_to_num
from scipy.sparse import csr_matrix, eye, vstack
from scipy.sparse.linalg import spsolve

from pandapower.pf.iwamoto_multiplier import _iwamoto_step
from pandapower.pf.makeYbus_facts import makeYbus_svc, makeYbus_tcsc, makeYft_tcsc, calc_y_svc_pu, \
    makeYbus_ssc
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
    create_J_modification_tcsc, create_J_modification_ssc


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
    gen = ppci['gen']
    branch = ppci['branch']
    tcsc = ppci['tcsc']
    svc = ppci['svc']
    ssc = ppci['ssc']
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
    ssc_y_pu = 1/(ssc[ssc_branches, SSC_R].real + 1j * ssc[ssc_branches, SSC_X].real)
    any_ssc = num_ssc > 0
    any_ssc_controllable = num_ssc_controllable > 0





    num_facts_controllable = num_svc_controllable + num_tcsc_controllable # + 2 * num_ssc_controllable
    any_facts_controllable = num_facts_controllable > 0
    num_facts = num_svc + num_tcsc + num_ssc


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

    Ybus_svc = makeYbus_svc(Ybus, x_control_svc, svc_x_l_pu, svc_x_cvar_pu, svc_buses)
    Ybus_tcsc = makeYbus_tcsc(Ybus, x_control_tcsc, tcsc_x_l_pu, tcsc_x_cvar_pu, tcsc_fb, tcsc_tb)
    Ybus_ssc_not_controllable = makeYbus_ssc(Ybus, ssc_y_pu[~ssc_controllable], ssc_fb[~ssc_controllable], ssc_tb[~ssc_controllable], np.any(~ssc_controllable))
    Ybus_ssc_controllable = makeYbus_ssc(Ybus, ssc_y_pu[ssc_controllable], ssc_fb[ssc_controllable], ssc_tb[ssc_controllable], any_ssc_controllable)
    Ybus_ssc = Ybus_ssc_not_controllable + Ybus_ssc_controllable

    # to avoid non-convergence due to zero-terms in the Jacobian:
    if any_tcsc_controllable and np.all(V[tcsc_fb] == V[tcsc_tb]):
        V[tcsc_tb] -= 0.01 + 0.001j
    if any_ssc_controllable and np.all(V[ssc_fb[ssc_controllable]] == V[ssc_tb[ssc_controllable]]):
        V[ssc_tb[ssc_controllable]] -= 0.01 + 0.001j

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
    pvpq_lookup = zeros(max((Ybus + Ybus_svc + Ybus_tcsc + Ybus_ssc).indices) + 1, dtype=np.int64)
    if dist_slack:
        # slack bus is relevant for the function createJ_ds
        pvpq_lookup[refpvpq] = arange(len(refpvpq))
    else:
        pvpq_lookup[pvpq] = arange(len(pvpq))

    pq_lookup = zeros(max(refpvpq) + 1, dtype=np.int64)
    pq_lookup[pq] = arange(len(pq))

    # get jacobian function
    createJ = get_fastest_jacobian_function(pvpq, pq, numba, dist_slack)

    nref = len(ref)
    npv = len(pv)
    npq = len(pq)
    j0 = 0
    j1 = nref if dist_slack else 0
    j2 = j1 + npv  # j1:j2 - V angle of pv buses
    j3 = j2
    j4 = j2 + npq  # j3:j4 - V angle of pq buses
    j5 = j4
    j6 = j4 + npq  # j5:j6 - V mag of pq buses
    j6a = j6 + num_svc_controllable  # svc
    j6b = j6a + num_tcsc_controllable  # tcsc
    j6c = j6b + num_ssc_controllable # ssc Vq angle
    j7 = j6c
    j6d = j6c + num_ssc_controllable # ssc Vq mag
    j8 = j6d

    # make initial guess for the slack
    slack = (gen[:, PG].sum() - bus[:, PD].sum()) / baseMVA
    # evaluate F(x0)

    F = _evaluate_Fx(Ybus + Ybus_svc + Ybus_tcsc + Ybus_ssc, V, Sbus, ref, pv, pq, slack_weights, dist_slack, slack)
    if any_facts_controllable or any_ssc_controllable:
        mis_facts = _evaluate_Fx_facts(V, pq, svc_buses[svc_controllable], svc_set_vm_pu[svc_controllable],
                                       tcsc_controllable, tcsc_set_p_pu, tcsc_tb, Ybus_tcsc, ssc_fb[ssc_controllable],
                                       ssc_tb[ssc_controllable], ssc_controllable, ssc_set_vm_pu, F, pq_lookup)
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
        J = create_jacobian_matrix(Ybus+Ybus_ssc_not_controllable, V, ref, refpvpq, pvpq, pq, createJ, pvpq_lookup, nref, npv, npq, numba, slack_weights, dist_slack)

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
            J_m_svc = create_J_modification_svc(J, svc_buses, refpvpq if dist_slack else pvpq, pq, pq_lookup, V,
                                                x_control_svc, svc_x_l_pu, svc_x_cvar_pu,
                                                num_svc, num_svc_controllable, svc_controllable)
            J = J + J_m_svc
        if any_tcsc:
            J_m_tcsc = create_J_modification_tcsc(V, Ybus_tcsc, x_control_tcsc, svc_controllable, tcsc_controllable,
                                                  tcsc_x_l_pu, tcsc_x_cvar_pu, tcsc_fb, tcsc_tb,
                                                  refpvpq if dist_slack else pvpq, pq, pvpq_lookup,
                                                  pq_lookup, num_svc_controllable, num_tcsc)
            J = J + J_m_tcsc
        if any_ssc_controllable:
            J_m_ssc = create_J_modification_ssc(J, V, Ybus_ssc_controllable, ssc_fb[ssc_controllable],
                                                ssc_tb[ssc_controllable], refpvpq if dist_slack else pvpq, pq,
                                                pvpq_lookup, pq_lookup)
            J = J + J_m_ssc

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
            Ybus_tcsc = makeYbus_tcsc(Ybus, x_control_tcsc, tcsc_x_l_pu, tcsc_x_cvar_pu, tcsc_fb, tcsc_tb)
        # Ybus_ssc does not change
        # if any_ssc_controllable:
        #     Ybus_ssc = makeYbus_ssc(Ybus, ssc_y_pu, ssc_fb, ssc_tb, any_ssc)

        F = _evaluate_Fx(Ybus + Ybus_svc + Ybus_tcsc + Ybus_ssc, V, Sbus, ref, pv, pq, slack_weights, dist_slack, slack)
        if any_facts_controllable or any_ssc_controllable:
            mis_facts = _evaluate_Fx_facts(V, pq, svc_buses[svc_controllable], svc_set_vm_pu[svc_controllable],
                                           tcsc_controllable, tcsc_set_p_pu, tcsc_tb, Ybus_tcsc,
                                           ssc_fb[ssc_controllable], ssc_tb[ssc_controllable], ssc_controllable,
                                           ssc_set_vm_pu, F, pq_lookup)
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

    Yf_tcsc, Yt_tcsc = makeYft_tcsc(Ybus_tcsc, tcsc_fb, tcsc_tb)
    # todo: move to pf.run_newton_raphson_pf.ppci_to_pfsoln
    if any_tcsc:
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
        Yf_ssc, Yt_ssc = makeYft_tcsc(Ybus_ssc, ssc_fb, ssc_tb)
        s_ssc_f = conj(Yf_ssc.dot(V)) * V[ssc_fb] * baseMVA
        ssc[ssc_branches, SSC_Q] = s_ssc_f.imag

    # because we now have updates of the Ybus matrices due to TDPF, SVC, TCSC, SSC,
    # we are interested in storing them for later use:
    ppci["internal"]["Ybus"] = Ybus
    ppci["internal"]["Ybus_svc"] = Ybus_svc
    ppci["internal"]["Ybus_tcsc"] = Ybus_tcsc
    ppci["internal"]["Ybus_ssc"] = Ybus_ssc
    ppci["internal"]["Yf_tcsc"] = Yf_tcsc
    ppci["internal"]["Yt_tcsc"] = Yt_tcsc
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
                       Ybus_tcsc=None, ssc_fb=None, ssc_tb=None,
                       ssc_controllable=None, ssc_set_vm_pu=None, old_F=None, pq_lookup=None):
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
        # Sbus_ssc = V * conj(Ybus_ssc * V)
        # ssc_set_p_pu = 0
        # mis_ssc_p = Sbus_ssc[ssc_tb[ssc_controllable]].real - ssc_set_p_pu  ####  here used ssc_tb refereing to the q bus
        mis_ssc_v = np.abs(V[ssc_fb]) - ssc_set_vm_pu
        # mis_facts = np.r_[mis_facts, mis_ssc_p, mis_ssc_v]
        # old_F[-(len(pq)+len(ssc_fb))] = mis_ssc_p
        # old_F[-len(ssc_fb)] = mis_ssc_v
        old_F[-len(pq)+pq_lookup[ssc_tb]] = mis_ssc_v

    return mis_facts


def _check_for_convergence(F, tol):
    # calc infinity norm
    return linalg.norm(F, Inf) < tol

