# -*- coding: utf-8 -*-
"""
# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

@author: sghosh (Intern : Feb 2018-July 2018)
@author: Alexander Prostejovsky (alepros), Technical University of Denmark
"""
from time import time

import numpy as np
import scipy as sp

try:
    import pplog as logging
except ImportError:
    import logging
logger = logging.getLogger(__name__)

from pandapower.pd2ppc import _pd2ppc
from pandapower.pd2ppc_zero import _pd2ppc_zero
from pandapower.pf.makeYbus import makeYbus
from pandapower.idx_bus import PD, QD, VM, VA
from pandapower.auxiliary import _sum_by_group, _check_if_numba_is_installed, \
    _check_bus_index_and_print_warning_if_high, _check_gen_index_and_print_warning_if_high, \
    _add_pf_options, _add_ppc_options
from pandapower.pf.run_newton_raphson_pf import _run_newton_raphson_pf
from pandapower.build_bus import _add_ext_grid_sc_impedance
from pandapower.pf.bustypes import bustypes
from pandapower.run import _passed_runpp_parameters
from pandapower.idx_bus_3ph import VM_A, VA_A, VM_B, VA_B, VM_C, VA_C
from pandapower.results import _extract_results, _copy_results_ppci_to_ppc
from pandapower.results_bus import _get_bus_idx


# =============================================================================
# Functions for 3 Phase Unbalanced Load Flow
# =============================================================================
# =============================================================================
# Convert to three decoupled sequence networks
# =============================================================================

def X012_to_X0(X012):
    return np.transpose(X012[0, :])


def X012_to_X1(X012):
    return np.transpose(X012[1, :])


def X012_to_X2(X012):
    return np.transpose(X012[2, :])


# =============================================================================
# Three decoupled sequence network to 012 matrix conversion
# =============================================================================

def combine_X012(X0, X1, X2):
    return np.transpose(np.concatenate((X0, X1, X2), axis=1))


# =============================================================================
# Symmetrical transformation matrix
# Tabc : 012 > abc
# T012 : abc >012
# =============================================================================

def phase_shift_unit_operator(angle_deg):
    return 1 * np.exp(1j * np.deg2rad(angle_deg))


a = phase_shift_unit_operator(120)
asq = phase_shift_unit_operator(-120)
Tabc = np.matrix(
    [
        [1, 1, 1],
        [1, asq, a],
        [1, a, asq]
    ], dtype=np.complex)

T012 = np.divide(np.matrix(
    [
        [1, 1, 1],
        [1, a, asq],
        [1, asq, a]
    ], dtype=np.complex), 3)


def sequence_to_phase(X012):
    return np.matmul(Tabc, X012)


def phase_to_sequence(Xabc):
    return np.matmul(T012, Xabc)


# =============================================================================
# Calculating Sequence Current from sequence Voltages
# =============================================================================

def I0_from_V012(V012, Y):
    V0 = X012_to_X0(V012)
    if type(Y) == sp.sparse.csr.csr_matrix:
        return np.matmul(Y.todense(), V0)
    else:
        return np.matmul(Y, V0)


def I1_from_V012(V012, Y):
    V1 = X012_to_X1(V012)
    if type(Y) == sp.sparse.csr.csr_matrix:
        return np.matmul(Y.todense(), V1)
    else:
        return np.matmul(Y, V1)


def I2_from_V012(V012, Y):
    V2 = X012_to_X2(V012)
    if type(Y) == sp.sparse.csr.csr_matrix:
        return np.matmul(Y.todense(), V2)
    else:
        return np.matmul(Y, V2)


def V1_from_ppc(ppc):
    return np.transpose(
        np.matrix(
            ppc["bus"][:, 7] * np.exp(1j * np.deg2rad(ppc["bus"][:, 8]))
            , dtype=np.complex
        )
    )


def V_from_I(Y, I):
    return np.transpose(np.matrix(sp.sparse.linalg.spsolve(Y, I)))


def I_from_V(Y, V):
    if type(Y) == sp.sparse.csr.csr_matrix:
        return np.matmul(Y.todense(), V)
    else:
        return np.matmul(Y, V)


# =============================================================================
# Calculating Power
# =============================================================================
def S_from_VI(V, I):
    return np.multiply(V, I.conjugate())


# =============================================================================
# Mapping load for positive sequence loads
# =============================================================================
def load_mapping(net):
    _is_elements = net["_is_elements"]
    b = np.array([0], dtype=int)
    SA, SB, SC = np.array([0]), np.array([]), np.array([])
    q_a, QA = np.array([0]), np.array([])
    p_a, PA = np.array([0]), np.array([])
    q_b, QB = np.array([0]), np.array([])
    p_b, PB = np.array([0]), np.array([])
    q_c, QC = np.array([0]), np.array([])
    p_c, PC = np.array([0]), np.array([])

    l3 = net["load_3ph"]
    if len(l3) > 0:
        vl = _is_elements["load_3ph"] * l3["scaling"].values.T
        q_a = np.hstack([q_a, l3["q_kvar_A"].values * vl])
        p_a = np.hstack([p_a, l3["p_kw_A"].values * vl])
        q_b = np.hstack([q_b, l3["q_kvar_B"].values * vl])
        p_b = np.hstack([p_b, l3["p_kw_B"].values * vl])
        q_c = np.hstack([q_c, l3["q_kvar_C"].values * vl])
        p_c = np.hstack([p_c, l3["p_kw_C"].values * vl])
        b = np.hstack([b, l3["bus"].values])

    sgen_3ph = net["sgen_3ph"]
    if len(sgen_3ph) > 0:
        vl = _is_elements["sgen_3ph"] * sgen_3ph["scaling"].values.T
        q_a = np.hstack([q_a, sgen_3ph["q_kvar_A"].values * vl])
        p_a = np.hstack([p_a, sgen_3ph["p_kw_A"].values * vl])
        q_b = np.hstack([q_b, sgen_3ph["q_kvar_B"].values * vl])
        p_b = np.hstack([p_b, sgen_3ph["p_kw_B"].values * vl])
        q_c = np.hstack([q_c, sgen_3ph["q_kvar_C"].values * vl])
        p_c = np.hstack([p_c, sgen_3ph["p_kw_C"].values * vl])
        b = np.hstack([b, sgen_3ph["bus"].values])
    # For Network Symmetric loads with unsymmetric loads
    #    Since the bus values of ppc values are not known, it is added again, fresh
    l = net["load"]
    if len(l) > 0:
        vl = _is_elements["load"] * l["scaling"].values.T
        q_a = np.hstack([q_a, l["q_kvar"].values / 3 * vl])
        p_a = np.hstack([p_a, l["p_kw"].values / 3 * vl])
        q_b = np.hstack([q_b, l["q_kvar"].values / 3 * vl])
        p_b = np.hstack([p_b, l["p_kw"].values / 3 * vl])
        q_c = np.hstack([q_c, l["q_kvar"].values / 3 * vl])
        p_c = np.hstack([p_c, l["p_kw"].values / 3 * vl])
        b = np.hstack([b, l["bus"].values])

    sgen = net["sgen"]
    if len(sgen) > 0:
        vl = _is_elements["load"] * l["scaling"].values.T
        q_a = np.hstack([q_a, sgen["q_kvar"].values / 3 * vl])
        p_a = np.hstack([p_a, sgen["p_kw"].values / 3 * vl])
        q_b = np.hstack([q_b, sgen["q_kvar"].values / 3 * vl])
        p_b = np.hstack([p_b, sgen["p_kw"].values / 3 * vl])
        q_c = np.hstack([q_c, sgen["q_kvar"].values / 3 * vl])
        p_c = np.hstack([p_c, sgen["p_kw"].values / 3 * vl])
        b = np.hstack([b, sgen["bus"].values / 3])
    if b.size:
        bus_lookup = net["_pd2ppc_lookups"]["bus"]
        ba = bus_lookup[b]
        bb = bus_lookup[b]
        bc = bus_lookup[b]
        ba, PA, QA = _sum_by_group(ba, p_a, q_a * 1j)
        bb, PB, QB = _sum_by_group(bb, p_b, q_b * 1j)
        bc, PC, QC = _sum_by_group(bc, p_c, q_c * 1j)
        SA, SB, SC = PA + QA, PB + QB, PC + QC
    return np.vstack([SA, SB, SC])


# =============================================================================
# 3 phase algorithm function
# =============================================================================
def runpp_3ph(net, algorithm='nr', calculate_voltage_angles="auto", init="auto", max_iteration="auto",
              tolerance_kva=1e-6, trafo_model="t", trafo_loading="current", enforce_q_lims=False,
              numba=True, recycle=None, check_connectivity=True, r_switch=0.0, voltage_depend_loads=False,
              delta_q=0, **kwargs):
    overrule_options = {}
    if "user_pf_options" in net.keys() and len(net.user_pf_options) > 0:
        passed_parameters = _passed_runpp_parameters(locals())
        overrule_options = {key: val for key, val in net.user_pf_options.items()
                            if key not in passed_parameters.keys()}
    if numba:
        numba = _check_if_numba_is_installed(numba)

    if voltage_depend_loads:
        if not (np.any(net["load"]["const_z_percent"].values) or
                np.any(net["load"]["const_i_percent"].values)):
            voltage_depend_loads = False

    if algorithm not in ['nr', 'bfsw'] and voltage_depend_loads == True:
        logger.warning("voltage-dependent loads not supported for {0} power flow algorithm -> "
                       "loads will be considered as constant power".format(algorithm))

    ac = True
    mode = "pf_3ph"  # TODO: Make valid modes (pf, pf_3ph, se, etc.) available in seperate file (similar to idx_bus.py)
    copy_constraints_to_ppc = False
    if calculate_voltage_angles == "auto":
        calculate_voltage_angles = False
        hv_buses = np.where(net.bus.vn_kv.values > 70)[0]
        if len(hv_buses) > 0:
            line_buses = net.line[["from_bus", "to_bus"]].values.flatten()
            if len(set(net.bus.index[hv_buses]) & set(line_buses)) > 0:
                calculate_voltage_angles = True
    if init == "auto":
        init = "dc" if calculate_voltage_angles else "flat"
    if init == "results" and len(net.res_bus) == 0:
        init = "auto"
    default_max_iteration = {"nr": 10, "bfsw": 100, "gs": 10000, "fdxb": 30, "fdbx": 30}
    if max_iteration == "auto":
        max_iteration = default_max_iteration[algorithm]

    # init options
    # net.__internal_options = {}
    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode=mode, copy_constraints_to_ppc=copy_constraints_to_ppc,
                     r_switch=r_switch, init=init, enforce_q_lims=enforce_q_lims,
                     recycle=recycle, voltage_depend_loads=voltage_depend_loads, delta=delta_q)
    _add_pf_options(net, tolerance_kva=tolerance_kva, trafo_loading=trafo_loading,
                    numba=numba, ac=ac, algorithm=algorithm, max_iteration=max_iteration)
    # net.__internal_options.update(overrule_options)
    net._options.update(overrule_options)
    _check_bus_index_and_print_warning_if_high(net)
    _check_gen_index_and_print_warning_if_high(net)
    # =============================================================================
    # Y Bus formation for Sequence Networks
    # =============================================================================
    #    net._options = {'calculate_voltage_angles': 'auto', 'check_connectivity': True, 'init': 'auto',
    #        'r_switch': 0.0,'voltage_depend_loads': False, 'mode': "pf_3ph",'copy_constraints_to_ppc': False,
    #        'enforce_q_lims': False, 'numba': True, 'recycle': {'Ybus': False, '_is_elements': False, 'bfsw': False, 'ppc': False},
    #        "tolerance_kva": 1e-5, "max_iteration": 10}
    _, ppci1 = _pd2ppc(net)

    _, ppci2 = _pd2ppc(net)
    _add_ext_grid_sc_impedance(net, ppci2)

    _, ppci0 = _pd2ppc_zero(net)

    # Y0_pu,_,_ = makeYbus(ppci0["baseMVA"], ppci0["bus"], ppci0["branch"])
    #
    # Y1_pu,_,_ = makeYbus(ppci1["baseMVA"], ppci1["bus"], ppci1["branch"])
    #
    # Y2_pu,_,_ = makeYbus(ppci2["baseMVA"], ppci2["bus"], ppci2["branch"])

    ppci0, ppci1, ppci2, Y0_pu, Y1_pu, Y2_pu, _, _, _, _, _, _ = _get_Y_bus(ppci0, ppci1, ppci2, recycle, makeYbus)

    sl_bus, pv_bus, pq_bus = bustypes(ppci1['bus'], ppci1['gen'])

    # =============================================================================
    # Initial voltage values
    # =============================================================================
    nb = ppci1["bus"].shape[0]
    V012_it = np.concatenate(
        (
            np.matrix(np.zeros((1, nb), dtype=np.complex))
            , np.matrix(np.ones((1, nb), dtype=np.complex))
            , np.matrix(np.zeros((1, nb), dtype=np.complex))
        )
        , axis=0
    )

    Vabc_it = sequence_to_phase(V012_it)

    # =============================================================================
    # Initialise iteration variables
    # =============================================================================
    count = 0
    S_mismatch = np.matrix([[True], [True]], dtype=bool)
    Sabc = load_mapping(net)
    # =============================================================================
    #             Iteration using Power mismatch criterion
    # =============================================================================
    t0 = time()
    while (S_mismatch > tolerance_kva).any() and count < 3 * max_iteration:
        # =============================================================================
        #     Voltages and Current transformation for PQ and Slack bus
        # =============================================================================
        Sabc_pu = -np.divide(Sabc, net.sn_kva)
        Iabc_it = np.divide(Sabc_pu, Vabc_it).conjugate()
        I012_it = phase_to_sequence(Iabc_it)

        I0_pu_it = X012_to_X0(I012_it)
        I2_pu_it = X012_to_X2(I012_it)

        V1_for_S1 = V012_it[1, :]
        I1_for_S1 = -I012_it[1, :]
        S1 = np.multiply(V1_for_S1, I1_for_S1.conjugate())

        # =============================================================================
        # Current used to find S1 Positive sequence power
        # =============================================================================

        ppci1["bus"][pq_bus, PD] = np.real(S1[:, pq_bus]) * net.sn_kva * 1e-3
        ppci1["bus"][pq_bus, QD] = np.imag(S1[:, pq_bus]) * net.sn_kva * 1e-3

        _run_newton_raphson_pf(ppci1, net._options)

        I1_from_V_it = -np.transpose(I1_from_V012(V012_it, Y1_pu))
        s_from_voltage = S_from_VI(V1_for_S1, I1_from_V_it)

        V1_pu_it = V1_from_ppc(ppci1)
        V0_pu_it = V_from_I(Y0_pu, I0_pu_it)
        V2_pu_it = V_from_I(Y2_pu, I2_pu_it)
        # =============================================================================
        #     This current is YV for the present iteration
        # =============================================================================
        V012_new = combine_X012(V0_pu_it, V1_pu_it, V2_pu_it)

        #        V_abc_new = sequence_to_phase(V012_new)

        # =============================================================================
        #     Mismatch from Sabc to Vabc Needs to be done tomorrow
        # =============================================================================
        S_mismatch = np.abs(S1[:, pq_bus] - s_from_voltage[:, pq_bus])
        V012_it = V012_new
        Vabc_it = sequence_to_phase(V012_it)
        count += 1

    ppci0["et"] = time() - t0
    ppci0["success"] = (count < 3 * max_iteration)

    _copy_bus_results_to_results_table(net, ppci0, V012_it, I012_it, Y0_pu, Y1_pu)

    return count, V012_it, I012_it, ppci0, Y1_pu


def show_results(V_base, kVA_base, count, ppci0, Y1_pu, V012_new, I012_new):
    V_base_res = V_base / np.sqrt(3)
    I_base_res = kVA_base / V_base_res * 1e-3
    print("\n No of Iterations: %u" % count)
    print('\n\n Final  Values Pandapower ')

    ppci0["bus"][0, 4] = 0
    ppci0["bus"][0, 5] = 0
    Y0_pu, _, _ = makeYbus(ppci0["baseMVA"], ppci0["bus"], ppci0["branch"])

    # Y0_pu = Y0_pu.todense()
    I012_new = combine_X012(I0_from_V012(V012_new, Y0_pu),
                            I1_from_V012(V012_new, Y1_pu),
                            I2_from_V012(V012_new, Y1_pu))
    I_abc_new = sequence_to_phase(I012_new)
    V_abc_new = sequence_to_phase(V012_new)
    Sabc_new = S_from_VI(V_abc_new, I_abc_new) * kVA_base
    print('\n SABC New using I=YV\n')
    print(Sabc_new)
    print(' \n Voltage  ABC\n')
    print(abs(V_abc_new) * V_base_res)

    print('\n Current  ABC\n')
    print(abs(I_abc_new) * I_base_res)

    # update data matrices with solution
    # baseMVA, bus, gen, branch, ref, pv, pq, _, _, V0 = _get_pf_variables_from_ppci(ppci0)
    # V_abc_res = abs(V_abc_new)*V_base_res
    # bus[:,VM_A] = np.abs(V_abc_res[:,1])
    # bus[:,VM_B] = np.abs(V_abc_res[:,2])
    # bus[:,VM_C] = np.abs(V_abc_res[:,3])
    # bus[:,(VA_A,VA_B,VA_C)] = np.angle(V_abc_res)

    # bus, gen, branch = pfsoln_3ph(kVA_base*1e-3, bus, gen, branch, Ybus, Yf, Yt, V, ref)

    return V_abc_new, I_abc_new, Sabc_new


def _get_Y_bus(ppci0, ppci1, ppci2, recycle, makeYbus):
    if recycle is not None and recycle["Ybus"] and ppci0["internal"]["Ybus"].size and ppci1["internal"]["Ybus"].size and \
            ppci2["internal"]["Ybus"].size:
        Y0_bus, Y0_f, Y0_t = ppci0["internal"]['Ybus'], ppci0["internal"]['Yf'], ppci0["internal"]['Yt']
        Y1_bus, Y1_f, Y1_t = ppci1["internal"]['Ybus'], ppci1["internal"]['Yf'], ppci1["internal"]['Yt']
        Y2_bus, Y2_f, Y2_t = ppci2["internal"]['Ybus'], ppci2["internal"]['Yf'], ppci2["internal"]['Yt']
    else:
        ## build admittance matrices
        Y0_bus, Y0_f, Y0_t = makeYbus(ppci0["baseMVA"], ppci0["bus"], ppci0["branch"])
        Y1_bus, Y1_f, Y1_t = makeYbus(ppci1["baseMVA"], ppci1["bus"], ppci1["branch"])
        Y2_bus, Y2_f, Y2_t = makeYbus(ppci2["baseMVA"], ppci2["bus"], ppci2["branch"])
        if recycle is not None and recycle["Ybus"]:
            ppci0["internal"]['Ybus'], ppci0["internal"]['Yf'], ppci0["internal"]['Yt'] = Y0_bus, Y0_f, Y0_t
            ppci1["internal"]['Ybus'], ppci1["internal"]['Yf'], ppci1["internal"]['Yt'] = Y1_bus, Y1_f, Y1_t
            ppci2["internal"]['Ybus'], ppci2["internal"]['Yf'], ppci2["internal"]['Yt'] = Y2_bus, Y2_f, Y2_t

    return ppci0, ppci1, ppci2, Y0_bus, Y1_bus, Y2_bus, Y0_f, Y1_f, Y2_f, Y0_t, Y1_t, Y2_t


# =============================================================================
# First draft for copying three-phase pf results obtained via runpp_3ph() into results tables.
# Three-phase bus results are taken directly from runpp_3ph() and written to res_bus_3ph.
# res_bus is filled with positive sequence voltages and the sum of three-phase powers.
#
# Todo: Consider V_base for each bus
# Todo: Extend _store_results_from_pf_in_ppci / _copy_results_ppci_to_ppc / _get_bus_v_results / _get_p_q_results / get_bus_results / _clean_up()
# Todo: Include branch and generation results
# Open questions: Is ppci0 the right structure to extract results? Is ppci1 needed somewhere?
#                 Does it make sense to write positive sequence results into res_bus?
#
# @author: alepros
# =============================================================================
def _copy_bus_results_to_results_table(net, ppci0, V012_it, I012_it, Y0_pu, Y1_pu):
    kVA_base = net.sn_kva
    V_base = net["bus"].get("vn_kv").get(net["bus"].get("vn_kv").first_valid_index())

    # Results related to pfsoln
    V_base_res = V_base / np.sqrt(3)
    I_base_res = kVA_base / V_base_res
    ppci0["bus"][0, 4] = 0
    ppci0["bus"][0, 5] = 0
    Y0_pu, Y0_f, Y0_t = makeYbus(ppci0["baseMVA"], ppci0["bus"], ppci0["branch"])

    # Y0_pu = Y0_pu.todense()
    I012_new = combine_X012(I0_from_V012(V012_it, Y0_pu),
                            I1_from_V012(V012_it, Y1_pu),
                            I2_from_V012(V012_it, Y1_pu))
    I_abc_new = sequence_to_phase(I012_it)
    V_abc_new = sequence_to_phase(V012_it)
    Sabc_new = S_from_VI(V_abc_new, I_abc_new) * kVA_base
    S012_it = phase_to_sequence(Sabc_new)

    bus = ppci0["bus"]
    gen = ppci0["gen"]
    branch = ppci0["branch"]

    bus[:, VM] = np.abs(V012_it[1, :]) * V_base_res
    bus[:, VA] = np.angle(V012_it[1, :]) * 180 / np.pi
    bus[:, VM_A] = np.abs(V_abc_new[0, :]) * V_base_res
    bus[:, VM_B] = np.abs(V_abc_new[1, :]) * V_base_res
    bus[:, VM_C] = np.abs(V_abc_new[2, :]) * V_base_res
    bus[:, VA_A] = np.angle(V_abc_new[0, :]) * 180 / np.pi
    bus[:, VA_B] = np.angle(V_abc_new[1, :]) * 180 / np.pi
    bus[:, VA_C] = np.angle(V_abc_new[2, :]) * 180 / np.pi

    # Results related to _store_results_from_pf_in_ppci
    ppci = ppci0
    ppci["bus"], ppci["gen"], ppci["branch"] = bus, gen, branch  # ppci=result

    ppc = net["_ppc"]
    ppc["mode"] = "pf_3ph"
    result = _copy_results_ppci_to_ppc(ppci, ppc, mode=ppc["mode"])
    net["_ppc"] = result
    _extract_results(net, ppc)

    pA_kw = np.transpose(np.real(Sabc_new[0, :]))
    qA_kvar = np.transpose(np.imag(Sabc_new[0, :]))
    pB_kw = np.transpose(np.real(Sabc_new[1, :]))
    qB_kvar = np.transpose(np.imag(Sabc_new[1, :]))
    pC_kw = np.transpose(np.real(Sabc_new[2, :]))
    qC_kvar = np.transpose(np.imag(Sabc_new[2, :]))

    net["res_bus_3ph"]["pA_kw"] = pA_kw
    net["res_bus_3ph"]["qA_kvar"] = qA_kvar
    net["res_bus_3ph"]["pB_kw"] = pB_kw
    net["res_bus_3ph"]["qB_kvar"] = qB_kvar
    net["res_bus_3ph"]["pC_kw"] = pC_kw
    net["res_bus_3ph"]["qC_kvar"] = qC_kvar
    net["res_bus"]["p_kw"] = pA_kw + pB_kw + pC_kw
    net["res_bus"]["q_kvar"] = qA_kvar + qB_kvar + qC_kvar

    # bus_idx = _get_bus_idx(net)
    print(_get_bus_idx(net))

    return net, ppci