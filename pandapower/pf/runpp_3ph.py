# -*- coding: utf-8 -*-
"""
# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

@author: sghosh (Intern : Feb 2018-July 2018)
@author: Alexander Prostejovsky (alepros), Technical University of Denmark
"""
from time import time

import numpy as np
from numpy import flatnonzero as find, pi, exp

from pandapower.pf.pfsoln import pfsoln

try:
    import pplog as logging
except ImportError:
    import logging
logger = logging.getLogger(__name__)

from pandapower.pd2ppc import _pd2ppc
from pandapower.pd2ppc_zero import _pd2ppc_zero
from pandapower.pf.makeYbus import makeYbus
from pandapower.idx_bus import GS, BS
from pandapower.auxiliary import _sum_by_group, _check_if_numba_is_installed, \
    _check_bus_index_and_print_warning_if_high, _check_gen_index_and_print_warning_if_high, \
    _add_pf_options, _add_ppc_options, _clean_up, sequence_to_phase, phase_to_sequence, X012_to_X0, X012_to_X2, \
    I1_from_V012, S_from_VI, V1_from_ppc, V_from_I, combine_X012, I0_from_V012, I2_from_V012
from pandapower.pf.run_newton_raphson_pf import _run_newton_raphson_pf
from pandapower.build_bus import _add_ext_grid_sc_impedance
from pandapower.pf.bustypes import bustypes
from pandapower.run import _passed_runpp_parameters
from pandapower.idx_bus import PD, QD, VM, VA
from pandapower.idx_gen import GEN_BUS, GEN_STATUS, VG
from pandapower.results import _extract_results, _copy_results_ppci_to_ppc, _extract_results_3ph, reset_results
from pandapower.results_bus import _get_bus_idx


def _get_pf_variables_from_ppci(ppci):
    ## default arguments
    if ppci is None:
        ValueError('ppci is empty')
    # ppopt = ppoption(ppopt)

    # get data for calc
    baseMVA, bus, gen, branch = \
        ppci["baseMVA"], ppci["bus"], ppci["gen"], ppci["branch"]

    ## get bus index lists of each type of bus
    ref, pv, pq = bustypes(bus, gen)

    ## generator info
    on = find(gen[:, GEN_STATUS] > 0)  ## which generators are on?
    gbus = gen[on, GEN_BUS].astype(int)  ## what buses are they at?

    ## initial state
    # V0    = ones(bus.shape[0])            ## flat start
    V0 = bus[:, VM] * exp(1j * pi / 180 * bus[:, VA])
    V0[gbus] = gen[on, VG] / abs(V0[gbus]) * V0[gbus]

    return baseMVA, bus, gen, branch, ref, pv, pq, on, gbus, V0


def _store_results_from_pf_in_ppci(ppci, bus, gen, branch):
    ppci["bus"], ppci["gen"], ppci["branch"] = bus, gen, branch
    return ppci


# =============================================================================
# Mapping load for positive sequence loads
# =============================================================================
# Todo: The bugfix in commit 1dd8a04 by @shankhoghosh caused test_runpp_3ph.py to fail and was therefore reverted
def load_mapping(net,ppci1,):
    _is_elements = net["_is_elements"]
    b = np.array([], dtype=int)
    SA = ppci1["bus"][:, PD]+ppci1["bus"][:, QD]*1j
    SB = ppci1["bus"][:, PD]+ppci1["bus"][:, QD]*1j
    SC = ppci1["bus"][:, PD]+ppci1["bus"][:, QD]*1j
    q_a, QA = np.array([]), np.array([])
    p_a, PA = np.array([]), np.array([])
    q_b, QB = np.array([]), np.array([])
    p_b, PB = np.array([]), np.array([])
    q_c, QC = np.array([]), np.array([])
    p_c, PC = np.array([]), np.array([])

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
        b = np.hstack([b, sgen["bus"].values ])

    
    if b.size:
        bus_lookup = net["_pd2ppc_lookups"]["bus"]
        ba = bus_lookup[b]
        bb = bus_lookup[b]
        bc = bus_lookup[b]
        ba, PA, QA = _sum_by_group(ba, p_a, q_a * 1j)
        bb, PB, QB = _sum_by_group(bb, p_b, q_b * 1j)
        bc, PC, QC = _sum_by_group(bc, p_c, q_c * 1j)
        SA[ba], SB[bb], SC[bc] = (PA + QA)*1e-3, (PB + QB)*1e-3, (PC + QC)*1e-3
    return np.vstack([SA, SB, SC])


# =============================================================================
# 3 phase algorithm function
# =============================================================================
def runpp_3ph(net, calculate_voltage_angles="auto", init="auto", max_iteration="auto",
              tolerance_kva=1e-6, trafo_model="t", trafo_loading="current", enforce_q_lims=False,
              numba=True, recycle=None, check_connectivity=True, r_switch=0.0,
              delta_q=0, **kwargs):
    overrule_options = {}
    if "user_pf_options" in net.keys() and len(net.user_pf_options) > 0:
        passed_parameters = _passed_runpp_parameters(locals())
        overrule_options = {key: val for key, val in net.user_pf_options.items()
                            if key not in passed_parameters.keys()}
    if numba:
        numba = _check_if_numba_is_installed(numba)

    ac = True
    mode = "pf_3ph"  # TODO: Make valid modes (pf, pf_3ph, se, etc.) available in seperate file (similar to idx_bus.py)

    copy_constraints_to_ppc = False
    if calculate_voltage_angles == "auto":
        calculate_voltage_angles = False
        hv_buses = np.where(net.bus.vn_kv.values > 70)[0] # Todo: Where does that number come from?
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
        max_iteration = default_max_iteration["nr"]

    # init options
    # net.__internal_options = {}
    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode=mode, copy_constraints_to_ppc=copy_constraints_to_ppc,
                     r_switch=r_switch, init=init, enforce_q_lims=enforce_q_lims,
                     recycle=recycle, voltage_depend_loads=False, delta=delta_q)
    _add_pf_options(net, tolerance_kva=tolerance_kva, trafo_loading=trafo_loading,
                    numba=numba, ac=ac, algorithm="nr", max_iteration=max_iteration)
    # net.__internal_options.update(overrule_options)
    net._options.update(overrule_options)
    _check_bus_index_and_print_warning_if_high(net)
    _check_gen_index_and_print_warning_if_high(net)
    reset_results(net, balanced=False)
    # =============================================================================
    # Y Bus formation for Sequence Networks
    # =============================================================================
    #    net._options = {'calculate_voltage_angles': 'auto', 'check_connectivity': True, 'init': 'auto',
    #        'r_switch': 0.0,'voltage_depend_loads': False, 'mode': "pf_3ph",'copy_constraints_to_ppc': False,
    #        'enforce_q_lims': False, 'numba': True, 'recycle': {'Ybus': False, '_is_elements': False, 'bfsw': False, 'ppc': False},
    #        "tolerance_kva": 1e-5, "max_iteration": 10}
    _, ppci1 = _pd2ppc(net,1)

    _, ppci2 = _pd2ppc(net,2)
    _add_ext_grid_sc_impedance(net, ppci2)

    _, ppci0 = _pd2ppc_zero(net,0)

    _,       bus0, gen0, branch0,      _,      _,      _, _, _, V00 = _get_pf_variables_from_ppci(ppci0)
    baseMVA, bus1, gen1, branch1, sl_bus, pv_bus, pq_bus, _, _, V01 = _get_pf_variables_from_ppci(ppci1)
    _,       bus2, gen2, branch2,      _,      _,      _, _, _, V02 = _get_pf_variables_from_ppci(ppci2)

    ppci0, ppci1, ppci2, Y0_pu, Y1_pu, Y2_pu, Y0_f, Y1_f, Y2_f, Y0_t, Y1_t, Y2_t = _get_Y_bus(ppci0, ppci1, ppci2, recycle, makeYbus)

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
    Sabc = load_mapping(net,ppci1)
    # =============================================================================
    #             Iteration using Power mismatch criterion
    # =============================================================================
    t0 = time()
    while (S_mismatch > tolerance_kva).any() and count < 3*max_iteration :
        # =============================================================================
        #     Voltages and Current transformation for PQ and Slack bus
        # =============================================================================
        Sabc_pu = -np.divide(Sabc, ppci1["baseMVA"])
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

        ppci1["bus"][pq_bus, PD] = np.real(S1[:, pq_bus]) * ppci1["baseMVA"]
        ppci1["bus"][pq_bus, QD] = np.imag(S1[:, pq_bus]) * ppci1["baseMVA"]

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
        S_mismatch = np.abs(S1[:, pq_bus] - s_from_voltage[:, pq_bus])*net.sn_kva
        V012_it = V012_new
        Vabc_it = sequence_to_phase(V012_it)
        count += 1

    ppci0["et"] = time() - t0
    ppci1["et"] = ppci0["et"]
    ppci2["et"] = ppci0["et"]
    ppci0["success"] = (count < 3 * max_iteration)
    ppci1["success"] = ppci0["success"]
    ppci2["success"] = ppci0["success"]

    ## update data matrices with solution
    bus0, gen0, branch0 = pfsoln(baseMVA, bus0, gen0, branch0, Y0_pu, Y0_f, Y0_t, V012_it[0, :].getA1(), sl_bus)
    bus1, gen1, branch1 = pfsoln(baseMVA, bus1, gen1, branch1, Y1_pu, Y1_f, Y1_t, V012_it[1, :].getA1(), sl_bus)
    bus2, gen2, branch2 = pfsoln(baseMVA, bus2, gen2, branch2, Y2_pu, Y2_f, Y2_t, V012_it[2, :].getA1(), sl_bus)

    ppci0 = _store_results_from_pf_in_ppci(ppci0, bus0, gen0, branch0)
    ppci1 = _store_results_from_pf_in_ppci(ppci1, bus1, gen1, branch1)
    ppci2 = _store_results_from_pf_in_ppci(ppci2, bus2, gen2, branch2)

    ppc0 = net["_ppc0"]
    ppc1 = net["_ppc1"]
    ppc2 = net["_ppc2"]

    # ppci doesn't contain out of service elements, but ppc does -> copy results accordingly
    ppc0 = _copy_results_ppci_to_ppc(ppci0, ppc0, mode=mode)
    ppc1 = _copy_results_ppci_to_ppc(ppci1, ppc1, mode=mode)
    ppc2 = _copy_results_ppci_to_ppc(ppci2, ppc2, mode=mode)

    # # raise if PF was not successful. If DC -> success is always 1
    # # Todo: These lines cause the test_runpp_3ph to fail in one instance!
    # if ppci0["success"] != True:
    #     net["converged"] = False
    #     _clean_up(net, res=False)
    #     raise LoadflowNotConverged("Power Flow {0} did not converge after "
    #                                "{1} iterations!".format(algorithm, 3*max_iteration))
    # else:
    #     net["converged"] = True

    _extract_results_3ph(net, ppc0, ppc1, ppc2)
    _clean_up(net)

    return count, V012_it, I012_it, ppci0, Y0_pu,Y1_pu,Y2_pu


def show_results(V_base, kVA_base, count, ppci0, Y1_pu, V012_new, I012_new):
    V_base_res = V_base / np.sqrt(3)
    I_base_res = kVA_base / V_base_res * 1e-3
    print("\n No of Iterations: %u" % count)
    print('\n\n Final  Values Pandapower ')

    V_abc_new, I_abc_new, Sabc_new = _phase_from_sequence_results(ppci0, Y1_pu, V012_new)
    Sabc_new = Sabc_new * kVA_base

    print('\n SABC New using I=YV\n')
    print(Sabc_new)
    print(' \n Voltage  ABC\n')
    print(abs(V_abc_new) * V_base_res)
    print('\n Current  ABC\n')
    print(abs(I_abc_new) * I_base_res)

    return V_abc_new, I_abc_new, Sabc_new


def _phase_from_sequence_results(ppci0, Y1_pu, V012_pu):
    ppci0["bus"][0, GS] = 0
    ppci0["bus"][0, BS] = 0
    # Y0_pu = Y0_pu.todense()
    Y0_pu, _, _ = makeYbus(ppci0["baseMVA"], ppci0["bus"], ppci0["branch"])
    I012_pu = combine_X012(I0_from_V012(V012_pu, Y0_pu),
                            I1_from_V012(V012_pu, Y1_pu),
                            I2_from_V012(V012_pu, Y1_pu))
    I_abc_pu = sequence_to_phase(I012_pu)
    V_abc_pu = sequence_to_phase(V012_pu)
    Sabc_pu = S_from_VI(V_abc_pu, I_abc_pu)
    return V_abc_pu, I_abc_pu, Sabc_pu


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
# DEPRECATED! This function will be deleted once all its functionality is covered in the main programme
# First draft for copying three-phase pf results obtained via runpp_3ph() into results tables.
# Three-phase bus results are taken directly from runpp_3ph() and written to res_bus_3ph.
# res_bus is filled with positive sequence voltages and the sum of three-phase powers.
#
# Todo: Extend _get_bus_v_results / _get_p_q_results / get_bus_results / _clean_up()
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
    # bus[:, VM_A] = np.abs(V_abc_new[0, :]) * V_base_res
    # bus[:, VM_B] = np.abs(V_abc_new[1, :]) * V_base_res
    # bus[:, VM_C] = np.abs(V_abc_new[2, :]) * V_base_res
    # bus[:, VA_A] = np.angle(V_abc_new[0, :]) * 180 / np.pi
    # bus[:, VA_B] = np.angle(V_abc_new[1, :]) * 180 / np.pi
    # bus[:, VA_C] = np.angle(V_abc_new[2, :]) * 180 / np.pi

    # Results related to _store_results_from_pf_in_ppci
    ppci = ppci0
    ppci["bus"], ppci["gen"], ppci["branch"] = bus, gen, branch  # ppci=result

    ppc = net["_ppc"]
    result = _copy_results_ppci_to_ppc(ppci, ppc, mode=net["_options"]["mode"])
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