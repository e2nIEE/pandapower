# -*- coding: utf-8 -*-

# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


"""Solves the power flow using a full Newton's method.
"""

from numpy import float64, array, angle, sqrt, square, exp, linalg, conj, r_, inf, arange, zeros, max, \
    zeros_like, column_stack, flatnonzero, nan_to_num
from scipy.sparse.linalg import spsolve

from pandapower.pf.iwamoto_multiplier import _iwamoto_step
from pandapower.pypower.makeSbus import makeSbus
from pandapower.pf.create_jacobian import create_jacobian_matrix, get_fastest_jacobian_function
from pandapower.pypower.idx_gen import PG
from pandapower.pypower.idx_bus import PD, SL_FAC, BASE_KV
from pandapower.pypower.idx_brch import BR_R, BR_X, F_BUS
from pandapower.pypower.idx_brch_tdpf import BR_R_REF_OHM_PER_KM, BR_LENGTH_KM, RATE_I_KA, T_START_C, R_THETA, \
    WIND_SPEED_MPS, ALPHA, TDPF, OUTER_DIAMETER_M, MC_JOULE_PER_M_K, WIND_ANGLE_DEGREE, SOLAR_RADIATION_W_PER_SQ_M, \
    GAMMA, EPSILON, T_AMBIENT_C, T_REF_C

from pandapower.pf.create_jacobian_tdpf import calc_g_b, calc_a0_a1_a2_tau, calc_r_theta, \
    calc_T_frank, calc_i_square_p_loss, create_J_tdpf


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
    slack_weights = bus[:, SL_FAC].astype(float64)  ## contribution factors for distributed slack
    tdpf = options.get('tdpf', False)

    # initialize
    i = 0
    V = V0
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
    pvpq_lookup = zeros(max(Ybus.indices) + 1, dtype=int)
    if dist_slack:
        # slack bus is relevant for the function createJ_ds
        pvpq_lookup[refpvpq] = arange(len(refpvpq))
    else:
        pvpq_lookup[pvpq] = arange(len(pvpq))

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
    j7 = j6

    # make initial guess for the slack
    slack = (gen[:, PG].sum() - bus[:, PD].sum()) / baseMVA
    # evaluate F(x0)
    F = _evaluate_Fx(Ybus, V, Sbus, ref, pv, pq, slack_weights, dist_slack, slack)

    T_base = 100  # T in p.u. for better convergence
    T = 20 / T_base
    r_theta_pu = 0
    if tdpf:
        if len(pq) > 0:
            pq_lookup = zeros(max(refpvpq) + 1, dtype=int)  # for TDPF
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
        v_base_kv = bus[branch[tdpf_lines, F_BUS].real.astype(int), BASE_KV]
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
            branch[tdpf_lines, BR_R] = r = r_ref_pu * (1 + alpha_pu * (T - t_ref_pu))
            Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
            g, b = calc_g_b(r, x)

        J = create_jacobian_matrix(Ybus, V, ref, refpvpq, pvpq, pq, createJ, pvpq_lookup, nref, npv, npq, numba, slack_weights, dist_slack)

        if tdpf:
            # p.u. values for T, a1, a2, I, S
            # todo: distributed_slack works fine if sn_mva is rather high (e.g. 1000), otherwise no convergence. Why?
            J = create_J_tdpf(branch, tdpf_lines, alpha_pu, r_ref_pu, refpvpq if dist_slack else pvpq, pq, pvpq_lookup,
                              pq_lookup, tau, tdpf_delay_s, Vm, Va, r_theta_pu, J, r, x, g)

        dx = -1 * spsolve(J, F, permc_spec=permc_spec, use_umfpack=use_umfpack)
        # update voltage
        if dist_slack:
            slack = slack + dx[j0:j1]
        if npv and not iwamoto:
            Va[pv] = Va[pv] + dx[j1:j2]
        if npq and not iwamoto:
            Va[pq] = Va[pq] + dx[j3:j4]
            Vm[pq] = Vm[pq] + dx[j5:j6]
        if tdpf:
            T = T + dx[j7:][tdpf_lines]

        # iwamoto multiplier to increase convergence
        if iwamoto:
            Vm, Va = _iwamoto_step(Ybus, J, F, dx, pq, npv, npq, dVa, dVm, Vm, Va, pv, j1, j2, j3, j4, j5, j6)

        V = Vm * exp(1j * Va)
        Vm = abs(V)  # update Vm and Va again in case
        Va = angle(V)  # we wrapped around with a negative Vm

        if v_debug:
            Vm_it = column_stack((Vm_it, Vm))
            Va_it = column_stack((Va_it, Va))

        if voltage_depend_loads:
            Sbus = makeSbus(baseMVA, bus, gen, vm=Vm)

        F = _evaluate_Fx(Ybus, V, Sbus, ref, pv, pq, slack_weights, dist_slack, slack)

        if tdpf:
            i_square_pu, p_loss_pu = calc_i_square_p_loss(branch, tdpf_lines, g, b, Vm, Va)
            if tdpf_update_r_theta:
                r_theta_pu = calc_r_theta(t_air_pu, a0, a1, a2, i_square_pu, p_loss_pu)
            T_calc = calc_T_frank(p_loss_pu, t_air_pu, r_theta_pu, tdpf_delay_s, T0, tau)
            F_t[tdpf_lines] = T - T_calc
            F = r_[F, F_t]

        converged = _check_for_convergence(F, tol)

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


def _check_for_convergence(F, tol):
    # calc infinity norm
    return linalg.norm(F, inf) < tol
