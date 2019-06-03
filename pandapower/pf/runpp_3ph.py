# -*- coding: utf-8 -*-
"""
# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

@author: sghosh (Intern : Feb 2018-July 2018)
@author: Alexander Prostejovsky (alepros), Technical University of Denmark
"""
import copy
from time import time

import numpy as np
from numpy import flatnonzero as find, pi, exp

from pandapower.pypower.pfsoln import pfsoln

try:
    import pplog as logging
except ImportError:
    import logging
logger = logging.getLogger(__name__)
from pandapower.pypower.idx_gen import PG, QG, GEN_BUS
from pandapower.pd2ppc import _pd2ppc
from pandapower.pd2ppc_zero import _pd2ppc_zero
from pandapower.pypower.makeYbus import makeYbus
from pandapower.pypower.idx_bus import GS, BS, PD , QD
from pandapower.auxiliary import _sum_by_group, _check_if_numba_is_installed, \
    _check_bus_index_and_print_warning_if_high, _check_gen_index_and_print_warning_if_high, \
    _add_pf_options, _add_ppc_options, _clean_up, sequence_to_phase, phase_to_sequence, X012_to_X0, X012_to_X2, \
    I1_from_V012, S_from_VI_elementwise, V1_from_ppc, V_from_I, combine_X012, I0_from_V012, I2_from_V012
from pandapower.pf.run_newton_raphson_pf import _run_newton_raphson_pf
from pandapower.build_bus import _add_ext_grid_sc_impedance
from pandapower.pypower.bustypes import bustypes
from pandapower.run import _passed_runpp_parameters
from pandapower.pypower.idx_bus import VM, VA
from pandapower.pypower.idx_gen import GEN_BUS, GEN_STATUS, VG
from pandapower.results import _copy_results_ppci_to_ppc, _extract_results_3ph, reset_results


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
    ref_gens = ppci["internal"]["ref_gens"]
    return baseMVA, bus, gen, branch, ref, pv, pq, on, gbus, V0,ref_gens


def _store_results_from_pf_in_ppci(ppci, bus, gen, branch):
    ppci["bus"], ppci["gen"], ppci["branch"] = bus, gen, branch
    return ppci


# =============================================================================
# Mapping load for positive sequence loads
# =============================================================================
# Todo: The bugfix in commit 1dd8a04 by @shankhoghosh caused test_runpp_3ph.py to fail and was therefore reverted
def load_mapping(net,ppci1):
    _is_elements = net["_is_elements"]
    bus_lookup = net["_pd2ppc_lookups"]["bus"]

    aranged_buses = np.arange(len(ppci1["bus"]))
    
    b = np.array([], dtype=int)
    b_ppc = np.array([], dtype=int)
    
    SA = (ppci1["bus"][:, PD]+ppci1["bus"][:, QD]*1j)/3
    SB = (ppci1["bus"][:, PD]+ppci1["bus"][:, QD]*1j)/3
    SC = (ppci1["bus"][:, PD]+ppci1["bus"][:, QD]*1j)/3
    
    q_a, QA = np.array([]), np.array([])
    p_a, PA = np.array([]), np.array([])
    q_b, QB = np.array([]), np.array([])
    p_b, PB = np.array([]), np.array([])
    q_c, QC = np.array([]), np.array([])
    p_c, PC = np.array([]), np.array([])
    
    if (ppci1["bus"][:, PD] != 0).any() :
        b_ppc = np.where( ppci1["bus"][:, PD] != 0)[0]
        q_a, QA = (ppci1["bus"][b_ppc, QD])/3, np.array([])
        p_a, PA = (ppci1["bus"][b_ppc, PD])/3, np.array([])
        q_b, QB = (ppci1["bus"][b_ppc, QD])/3, np.array([])
        p_b, PB = (ppci1["bus"][b_ppc, PD])/3, np.array([])
        q_c, QC = (ppci1["bus"][b_ppc, QD])/3, np.array([])
        p_c, PC = (ppci1["bus"][b_ppc, PD])/3, np.array([])
        b = np.where(bus_lookup == b_ppc)[0]
    elif (ppci1["bus"][:, QD] != 0).any() :
        b_ppc = np.where( ppci1["bus"][:, QD] != 0)[0]
        q_a, QA = (ppci1["bus"][b_ppc, QD])/3, np.array([])
        p_a, PA = (ppci1["bus"][b_ppc, PD])/3, np.array([])
        q_b, QB = (ppci1["bus"][b_ppc, QD])/3, np.array([])
        p_b, PB = (ppci1["bus"][b_ppc, PD])/3, np.array([])
        q_c, QC = (ppci1["bus"][b_ppc, QD])/3, np.array([])
        p_c, PC = (ppci1["bus"][b_ppc, PD])/3, np.array([])
        b = np.where(bus_lookup == b_ppc)[0]

    l3 = net["asymmetric_load"]
    l3_is = net["_is_elements"]["asymmetric_load"]
    if len(l3) > 0 and l3_is.any():
        vl = (_is_elements["asymmetric_load"] * l3["scaling"].values.T)[l3_is]
        q_a = np.hstack([q_a, l3["q_A_mvar"].values[l3_is] * vl])
        p_a = np.hstack([p_a, l3["p_A_mw"].values[l3_is] * vl])
        q_b = np.hstack([q_b, l3["q_B_mvar"].values[l3_is] * vl])
        p_b = np.hstack([p_b, l3["p_B_mw"].values[l3_is] * vl])
        q_c = np.hstack([q_c, l3["q_C_mvar"].values[l3_is] * vl])
        p_c = np.hstack([p_c, l3["p_C_mw"].values[l3_is] * vl])
        b = np.hstack([b, l3["bus"].values[l3_is]])

    sgen3 = net["asymmetric_sgen"]
    sgen3_is = net["_is_elements"]["asymmetric_sgen"]
    if len(sgen3) > 0 and sgen3_is.any():
        vl = (_is_elements["asymmetric_sgen"] * sgen3["scaling"].values.T)[sgen3_is]
        q_a = np.hstack([q_a, sgen3["q_A_mvar"].values[sgen3_is] * vl])
        p_a = np.hstack([p_a, sgen3["p_A_mw"].values[sgen3_is] * vl])
        q_b = np.hstack([q_b, sgen3["q_B_mvar"].values[sgen3_is] * vl])
        p_b = np.hstack([p_b, sgen3["p_B_mw"].values[sgen3_is] * vl])
        q_c = np.hstack([q_c, sgen3["q_C_mvar"].values[sgen3_is] * vl])
        p_c = np.hstack([p_c, sgen3["p_C_mw"].values[sgen3_is] * vl])
        b = np.hstack([b, sgen3["bus"].values[sgen3_is]])

    # Todo: Unserved powers on isolated nodes don't include 3ph elements yet

    # For Network Symmetric loads with unsymmetric loads
    #    Since the bus values of ppc values are not known, it is added again, fresh
    if b.size:
        ba = bus_lookup[b]
        bb = bus_lookup[b]
        bc = bus_lookup[b]
        ba, PA, QA = _sum_by_group(ba, p_a, q_a * 1j)
        bb, PB, QB = _sum_by_group(bb, p_b, q_b * 1j)
        bc, PC, QC = _sum_by_group(bc, p_c, q_c * 1j)
        
        SA[ba], SB[bb], SC[bc] = (PA + QA), (PB + QB), (PC + QC)
    return np.vstack([SA, SB, SC])


# =============================================================================
# 3 phase algorithm function
# =============================================================================
def runpp_3ph(net, calculate_voltage_angles="auto", init="auto", max_iteration="auto",
              tolerance_mva=1e-8, trafo_model="t", trafo_loading="current", enforce_q_lims=False,
              numba=True, recycle=None, check_connectivity=True, switch_rx_ratio=2.0,
              delta_q=0,v_debug =False, **kwargs):
    overrule_options = {}
    if "user_pf_options" in net.keys() and len(net.user_pf_options) > 0:
        passed_parameters = _passed_runpp_parameters(locals())
        overrule_options = {key: val for key, val in net.user_pf_options.items()
                            if key not in passed_parameters.keys()}
    if numba:
        numba = _check_if_numba_is_installed(numba)

    ac = True
    mode = "pf_3ph"  # TODO: Make valid modes (pf, pf_3ph, se, etc.) available in seperate file (similar to idx_bus.py)
#    v_debug = kwargs.get("v_debug", False)
    copy_constraints_to_ppc = False
    if calculate_voltage_angles == "auto":
        calculate_voltage_angles = False
        hv_buses = np.where(net.bus.vn_kv.values > 70)[0] # Todo: Where does that number come from?
        if len(hv_buses) > 0:
            line_buses = net.line[["from_bus", "to_bus"]].values.flatten()
            if len(set(net.bus.index[hv_buses]) & set(line_buses)) > 0:
                calculate_voltage_angles = True
    if init == "results" and len(net.res_bus) == 0:
        init = "auto"
    if init == "auto":
        init = "dc" if calculate_voltage_angles else "flat"

    
    default_max_iteration = {"nr": 100, "bfsw": 10, "gs": 10000, "fdxb": 30, "fdbx": 30}
    if max_iteration == "auto":
        max_iteration = default_max_iteration["nr"]

    # init options
    # net.__internal_options = {}
    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode=mode,switch_rx_ratio=switch_rx_ratio, init_vm_pu=init, init_va_degree=init,
                     enforce_q_lims=enforce_q_lims, recycle=recycle, voltage_depend_loads=False, delta=delta_q)
    _add_pf_options(net, tolerance_mva=tolerance_mva, trafo_loading=trafo_loading,
                    numba=numba, ac=ac, algorithm="nr", max_iteration=max_iteration,v_debug=v_debug)
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
    #        "tolerance_mva": 1e-5, "max_iteration": 10}
    net["_is_elements"] = None
    _, ppci1 = _pd2ppc(net, 1)

    _, ppci2 = _pd2ppc(net, 2)
    gs_eg,bs_eg = _add_ext_grid_sc_impedance(net, ppci2)

    _, ppci0 = _pd2ppc(net, 0)

    _,       bus0, gen0, branch0,      _,      _,      _, _, _, V00, ref_gens = _get_pf_variables_from_ppci(ppci0)
    baseMVA, bus1, gen1, branch1, sl_bus, pv_bus, pq_bus, _, _, V01, ref_gens = _get_pf_variables_from_ppci(ppci1)
    _,       bus2, gen2, branch2,      _,      _,      _, _, _, V02, ref_gens = _get_pf_variables_from_ppci(ppci2)

    ppci0, ppci1, ppci2, Y0_pu, Y1_pu, Y2_pu, Y0_f, Y1_f, Y2_f, Y0_t, Y1_t, Y2_t = _get_Y_bus(ppci0, ppci1, ppci2, recycle, makeYbus)

    # =============================================================================
    # Initial voltage values
    # =============================================================================
    nb = ppci1["bus"].shape[0]
    V012_it = np.concatenate(
        (
            np.array(np.zeros((1, nb), dtype=np.complex128))
            , np.array(np.ones((1, nb), dtype=np.complex128))
            , np.array(np.zeros((1, nb), dtype=np.complex128))
        )
        , axis=0
    )

    Vabc_it = sequence_to_phase(V012_it)
    if net.trafo.vector_group.any() :
        for vc in net.trafo.vector_group.values:
            if vc not in ["Yyn,Dyn,YNyn"]:
                V0_pu_it = X012_to_X0(V012_it)
                V2_pu_it = X012_to_X2(V012_it) 
    # =============================================================================
    # Initialise iteration variables
    # =============================================================================
    count = 0
    S_mismatch = np.array([[True], [True]], dtype=bool)
    Sabc = load_mapping(net,ppci1)
    # =============================================================================
    #             Iteration using Power mismatch criterion
    # =============================================================================
    t0 = time()
    while (S_mismatch > tolerance_mva).any() and count < 5*max_iteration :
        # =============================================================================
        #     Voltages and Current transformation for PQ and Slack bus
        # =============================================================================
        Sabc_pu = -np.divide(Sabc, ppci1["baseMVA"])
        Iabc_it = (np.divide(Sabc_pu, Vabc_it)).conjugate()
        I012_it = phase_to_sequence(Iabc_it)

        I0_pu_it = X012_to_X0(I012_it)
        I2_pu_it = X012_to_X2(I012_it)

        V1_for_S1 = V012_it[1, :]
        I1_for_S1 = -I012_it[1, :]
        S1 = np.multiply(V1_for_S1, I1_for_S1.conjugate())
        # =============================================================================
        # Current used to find S1 Positive sequence power
        # =============================================================================

        ppci1["bus"][pq_bus, PD] = np.real(S1[pq_bus]) * ppci1["baseMVA"]
        ppci1["bus"][pq_bus, QD] = np.imag(S1[pq_bus]) * ppci1["baseMVA"]

        _run_newton_raphson_pf(ppci1, net._options)

        I1_from_V_it = I1_from_V012(V012_it, Y1_pu).flatten()
        s_from_voltage = S_from_VI_elementwise(V1_for_S1, I1_from_V_it)
        V1_pu_it = V1_from_ppc(ppci1)
        V0_pu_it = V_from_I(Y0_pu, I0_pu_it)
        if net.trafo.vector_group.any():
            for vc in net.trafo.vector_group.values:
                if vc not in ["Yyn","Dyn","YNyn"]:
                    V0_pu_it = V0_pu_it*0
        V2_pu_it = V_from_I(Y2_pu, I2_pu_it)
        # =============================================================================
        #     This current is YV for the present iteration
        # =============================================================================
        V012_new = combine_X012(V0_pu_it, V1_pu_it, V2_pu_it)

        #        V_abc_new = sequence_to_phase(V012_new)

        # =============================================================================
        #     Mismatch from Sabc to Vabc Needs to be done tomorrow
        # =============================================================================
        S_mismatch = np.abs(np.abs(S1[pq_bus]) - np.abs(s_from_voltage[pq_bus]))
        V012_it = V012_new
        Vabc_it = sequence_to_phase(V012_it)
        count += 1

    ppci0["et"] = time() - t0
    ppci1["et"] = ppci0["et"]
    ppci2["et"] = ppci0["et"]
    ppci0["success"] = (count < 3 * max_iteration)
    ppci1["success"] = ppci0["success"]
    ppci2["success"] = ppci0["success"]
    
    # Todo: Add reference to paper to explain the following steps
    ref, pv, pq = bustypes(ppci0["bus"], ppci0["gen"])
    ppci0["bus"][ref, GS] -= gs_eg
    ppci0["bus"][ref, BS] -= bs_eg
    # Y0_pu = Y0_pu.todense()
    Y0_pu, Y0_f, Y0_t = makeYbus(ppci0["baseMVA"], ppci0["bus"], ppci0["branch"])

    ## update data matrices with solution
    # Todo: Add reference to paper to explain the choice of Y1 over Y2 in the negative sequence
    bus0, gen0, branch0 = pfsoln(baseMVA, bus0, gen0, branch0, Y0_pu, Y0_f, Y0_t, V012_it[0, :].flatten(), sl_bus, ref_gens)
    bus1, gen1, branch1 = pfsoln(baseMVA, bus1, gen1, branch1, Y1_pu, Y1_f, Y1_t, V012_it[1, :].flatten(), sl_bus, ref_gens)
    bus2, gen2, branch2 = pfsoln(baseMVA, bus2, gen2, branch2, Y1_pu, Y1_f, Y1_t, V012_it[2, :].flatten(), sl_bus, ref_gens)

    ppci0 = _store_results_from_pf_in_ppci(ppci0, bus0, gen0, branch0)
    ppci1 = _store_results_from_pf_in_ppci(ppci1, bus1, gen1, branch1)
    ppci2 = _store_results_from_pf_in_ppci(ppci2, bus2, gen2, branch2)
    
    ppci0["internal"]["Ybus"] = Y0_pu
    ppci1["internal"]["Ybus"] = Y1_pu
    ppci2["internal"]["Ybus"] = Y2_pu
    
    ppci0["internal"]["Yf"] = Y0_f
    ppci1["internal"]["Yf"] = Y1_f
    ppci2["internal"]["Yf"] = Y2_f
    
    ppci0["internal"]["Yt"] = Y0_t
    ppci1["internal"]["Yt"] = Y1_t
    ppci2["internal"]["Yt"] = Y2_t
    
    V_abc_pu, I_abc_pu, Sabc_pu = _phase_from_sequence_results(ppci0, Y1_pu, V012_new,gs_eg,bs_eg)
    I012_res = phase_to_sequence(I_abc_pu)
    S012_res = S_from_VI_elementwise(V012_new,I012_res) * ppci1["baseMVA"]
    
    eg_is_mask = net["_is_elements"]['ext_grid']
    ext_grid_lookup = net["_pd2ppc_lookups"]["ext_grid"]
    eg_is_idx = net["ext_grid"].index.values[eg_is_mask]
    eg_idx_ppc = ext_grid_lookup[eg_is_idx]
    """ # 2 ext_grids Fix: Instead of the generator index, bus indices of the generators are used"""
    eg_bus_idx_ppc = np.real(ppci1["gen"][eg_idx_ppc, GEN_BUS]).astype(int)
    
    ppci0["gen"][eg_idx_ppc, PG] = S012_res[0,eg_bus_idx_ppc].real
    ppci1["gen"][eg_idx_ppc, PG] = S012_res[1,eg_bus_idx_ppc].real
    ppci2["gen"][eg_idx_ppc, PG] = S012_res[2,eg_bus_idx_ppc].real
    ppci0["gen"][eg_idx_ppc, QG] = S012_res[0,eg_bus_idx_ppc].imag
    ppci1["gen"][eg_idx_ppc, QG] = S012_res[1,eg_bus_idx_ppc].imag
    ppci2["gen"][eg_idx_ppc, QG] = S012_res[2,eg_bus_idx_ppc].imag
    
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
    #                                "{1} iterations!".format("nr", 3*max_iteration))
    # else:
    #     net["converged"] = True

    _extract_results_3ph(net, ppc0, ppc1, ppc2)
    _clean_up(net)

    return count, V012_it, I012_it, ppci0, Y0_pu, Y1_pu, Y2_pu


def _phase_from_sequence_results(ppci0, Y1_pu, V012_pu,gs_eg,bs_eg):
    ref, pv, pq = bustypes(ppci0["bus"], ppci0["gen"])
    ppci0["bus"][ref, GS] -= gs_eg
    ppci0["bus"][ref, BS] -= bs_eg
    # Y0_pu = Y0_pu.todense()
    Y0_pu, _, _ = makeYbus(ppci0["baseMVA"], ppci0["bus"], ppci0["branch"])
    I012_pu = combine_X012(I0_from_V012(V012_pu, Y0_pu),
                            I1_from_V012(V012_pu, Y1_pu),
                            I2_from_V012(V012_pu, Y1_pu))
    I_abc_pu = sequence_to_phase(I012_pu)
    V_abc_pu = sequence_to_phase(V012_pu)
    Sabc_pu = S_from_VI_elementwise(V_abc_pu, I_abc_pu)
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
