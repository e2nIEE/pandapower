# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
import pandas as pd
from copy import deepcopy

# pandapower utilities
from pandapower.auxiliary import sequence_to_phase
from pandapower.results import BRANCH_RESULTS_KEYS
from pandapower.results_branch import _copy_switch_results_from_branches

# PPCI bus indeces
from pandapower.pypower.idx_bus import BUS_TYPE, BASE_KV, VM, GS, BS
#PPCI bus sc indeces
from pandapower.pypower.idx_bus_sc import (
    IKSSV, IP, ITH, IKSSC, R_EQUIV, R_EQUIV_OHM, X_EQUIV, X_EQUIV_OHM,
    SKSS, PHI_IKSSV_DEGREE, PHI_IKSSC_DEGREE,
    GS_GEN, BS_GEN, V_G, K_G, C_MAX, C_MIN, IKCV, PHI_IKCV_DEGREE
)

# PPCI branch indices
from pandapower.pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, TAP

# PPCI branch sc indices
from pandapower.pypower.idx_brch_sc import (
    IKSS_F, IKSS_T, IP_F, IP_T, ITH_F, ITH_T,
    PKSS_F, QKSS_F, PKSS_T, QKSS_T,
    VKSS_MAGN_F, VKSS_MAGN_T,
    VKSS_ANGLE_F, VKSS_ANGLE_T,
    IKSS_ANGLE_F, IKSS_ANGLE_T,
    K_T
)

'''
Following functions are intended to be called within the `extract_results` function in results.py.  
In that script, the finalized versions of all should be written as well.  
Since these are draft implementations, they are not yet confirmed and should not be pushed to Git.  
This serves only as an example of the intended structure.

However, even if you call these functions now from this script(new_results_format_testing.py_ into results.py 
and include only this within the `def_extract_results` function(example given under this comment), you will still 
get results. These results are explained in the following presentation that I 
published on Teams channel - Short circuit and protection; in folder - results format.


Functions should be called exactly  like this,nothing else should interfere resutls tbales as ther are many undifiend thigns in extract-reusts fucntuion , but when i comemtne dout them ,this was the 

def _extract_results(net, ppc_0, ppc_1, ppc_2, bus):
    _calculate_bus_results_all_cases(net, ppc_0, ppc_1, ppc_2, bus)
    _get_bus_results(net, ppc_0, ppc_1, ppc_2, bus)
    _get_gen_results(net, ppc_0, ppc_1, ppc_2)
    _get_sgen_results(net, ppc_1, bus)
    _get_ext_grid_results(net, ppc_0, ppc_1, ppc_2, bus)
    _calculate_line_results_all_cases(ppc_0, ppc_1, ppc_2, net)

'''

#BUSES
def _calculate_bus_results_all_cases(net, ppc_0, ppc_1, ppc_2, bus):
    """Calculate bus short-circuit results for all fault cases.
    This function is called together with the `_get_bus_results` function that is already implemented in results.py,
    which returns currents and power in three phase values, as well as the xk
    and rk values in all sequences.
    """

    bus_lookup = net._pd2ppc_lookups["bus"]
    ppc_index = bus_lookup[net.bus.index]
    n_ppc = len(ppc_index)

    # Initialize result arrays for all ppc buses
    skss_abc_mva = np.full((n_ppc, 3), np.nan, dtype=np.float64)
    ikss_abc_ka = np.full((n_ppc, 3), np.nan, dtype=np.float64)
    ikss_abc_degree = np.full((n_ppc, 3), np.nan, dtype=np.float64)
    p_abc_mw = np.full((n_ppc, 3), np.nan, dtype=np.float64)
    q_abc_mvar = np.full((n_ppc, 3), np.nan, dtype=np.float64)
    v_abc_abs = np.full((n_ppc, 3), np.nan, dtype=np.float64)
    v_abc_deg = np.full((n_ppc, 3), np.nan, dtype=np.float64)

    # -----------------------------
    # 1) Extract sequence currents from all three networks
    # -----------------------------
    def _get_sequence_current(ppc, current_col):
        """Helper to extract sequence current with phase."""
        return (ppc['bus'][:, current_col] * np.exp(1j * np.deg2rad(ppc['bus'][:, PHI_IKSSV_DEGREE].real)))[:, None]

    #Voltage source contributions(synchronuos generators, external grids)
    i_1_ka_0 = _get_sequence_current(ppc_0, IKSSV)
    i_1_ka_1 = _get_sequence_current(ppc_1, IKSSV)
    i_1_ka_2 = _get_sequence_current(ppc_2, IKSSV)

    # Current source contributions(static generators)
    i_2_ka_0 = _get_sequence_current(ppc_0, IKSSC)
    i_2_ka_1 = _get_sequence_current(ppc_1, IKSSC)
    i_2_ka_2 = _get_sequence_current(ppc_2, IKSSC)

    # -----------------------------
    # 2) Adjust sequence components based on fault type
    #Explanation for scaling factor that is used below is given for every case separtely
    # -----------------------------
    fault_type = net["_options"]["fault"]
    ref_shape = ppc_1['bus'][:, IKSSV].shape[0]

    if fault_type == "LLL":
        # Three-phase fault: only positive sequence is needed
        i_1_ka_0 = np.zeros((ref_shape, 1), dtype=complex)
        i_1_ka_2 = np.zeros((ref_shape, 1), dtype=complex)
        i_2_ka_0 = np.zeros((ref_shape, 1), dtype=complex)
        i_2_ka_2 = np.zeros((ref_shape, 1), dtype=complex)
        scaling_factor = 3
        # In a symmetrical LLL fault, all three phases contribute equally.
        # Pandapower stores the total 3-phase value for LLL in ppc structure, so dividing by 3 gives the correct per-phase quantities.


    elif fault_type == "LL":
        # Line-to-line fault: I2 = -I1, I0 = 0
        i_1_ka_0 = np.zeros((ref_shape, 1), dtype=complex)
        i_1_ka_2 = -i_1_ka_1
        i_2_ka_0 = np.zeros((ref_shape, 1), dtype=complex)
        i_2_ka_2 = -i_2_ka_1
        scaling_factor = np.sqrt(3)
        # In an LL fault, pandapower gives the line-to-line current.
        # Since I_LL = √3 · I_phase, dividing by √3 gives the per-phase value.

    elif fault_type == "LG":
        # Single line-to-ground fault: I0 = I1 = I2
        i_1_ka_0 = i_1_ka_1
        i_1_ka_2 = i_1_ka_1
        i_2_ka_0 = i_2_ka_1
        i_2_ka_2 = i_2_ka_1
        scaling_factor = 3
        # In an LG fault only one phase conducts the actual fault current,
        # but pandapower stores the sum of all three sequence currents (0, 1, and 2).
        # Since each sequence contributes equally to the single phase involved,
        # dividing by 3 gives the correct per-phase fault current.


    else:  # LLG
        scaling_factor = 1
        # In an LLG fault, the two faulted phases each carry their actual physical currents,
        # and pandapower already reports these directly as per-phase values.
        # No sequence-based summation needs to be reduced, so the scaling factor is 1.

    # -----------------------------
    # 3) Transform to phase components
    # -----------------------------
    i_1_012_ka = np.stack([i_1_ka_0, i_1_ka_1, i_1_ka_2], axis=2)
    i_2_012_ka = np.stack([i_2_ka_0, i_2_ka_1, i_2_ka_2], axis=2)

    i_1_abc_ka = np.apply_along_axis(sequence_to_phase, 2, i_1_012_ka / scaling_factor)
    i_2_abc_ka = np.apply_along_axis(sequence_to_phase, 2, i_2_012_ka / scaling_factor)

    # Zero small values
    i_1_abc_ka[np.abs(i_1_abc_ka) < 1e-5] = 0
    i_2_abc_ka[np.abs(i_2_abc_ka) < 1e-5] = 0

    # Total phase currents
    i_total_abc_ka = i_1_abc_ka + i_2_abc_ka

    # -----------------------------
    # 4) Select requested buses
    # -----------------------------
    ppc_indices = [ppc_index[b] for b in bus]

    # Extract currents for requested buses (diagonal elements for LLG bus fault)

    i_total_abc_ka_complex = i_total_abc_ka[ppc_indices, :]
    i_total_abc_ka_abs = np.abs(i_total_abc_ka_complex)
    i_total_abc_ka_angle = np.angle(i_total_abc_ka_complex, deg=True)

    # -----------------------------
    # 5) Voltages: transform from sequence to phase
    # -----------------------------
    v_pu_0 = ppc_0["internal"]["V_ikss"][ppc_indices][:, np.newaxis]
    v_pu_1 = ppc_1["internal"]["V_ikss"][ppc_indices][:, np.newaxis]
    v_pu_2 = ppc_2["internal"]["V_ikss"][ppc_indices][:, np.newaxis]

    v_012_pu = np.stack([v_pu_0, v_pu_1, v_pu_2], 2)
    v_abc_pu = np.apply_along_axis(sequence_to_phase, 2, v_012_pu)

    # Force proper shape - squeeze and reshape if needed
    v_abc_pu = np.squeeze(v_abc_pu)
    if v_abc_pu.ndim == 1:  # Single bus case: (3,)
        v_abc_pu = v_abc_pu.reshape(1, 3)
    elif v_abc_pu.ndim > 2:  # Still has extra dimensions
        v_abc_pu = v_abc_pu.reshape(len(bus), 3)

    v_abc_pu[np.abs(v_abc_pu) < 1e-5] = 0

    # -----------------------------
    # 6) Apparent power per phase
    # -----------------------------
    baseV = ppc_1['bus'][ppc_indices, BASE_KV]

    # Reshape baseV for proper broadcasting
    if baseV.ndim == 1:
        baseV = baseV[:, np.newaxis]  # shape (len(bus), 1)

    # Apparent power is calculated differently for LLL and asymmetrical faults.
    if fault_type == 'LLL':
        skss_abc_mva_phase = np.sqrt(3) * i_total_abc_ka_abs * baseV
    # In an LLL (three-phase) fault the system is fully symmetrical and baseV represents
    # the line-to-line voltage, so per-phase apparent power follows S = √3 · V_LL · I_phase.
    else:
        skss_abc_mva_phase = i_total_abc_ka_abs * baseV / np.sqrt(3)
    # For LL, LG and LLG faults the system is asymmetrical and currents are phase quantities.
    # In these cases per-phase power is computed using phase voltage V_phase = V_LL / √3,
    # resulting in S = V_LL / √3 · I_phase.

    # -----------------------------
    # 7) Active and reactive power per phase
    # -----------------------------
    v_abc_kv = v_abc_pu * (baseV / np.sqrt(3))
    s_abc_mva = v_abc_kv * np.conj(i_total_abc_ka_complex)

    p_abc_mw_phase = np.real(s_abc_mva)
    q_abc_mvar_phase = np.imag(s_abc_mva)

    # Zero out small values
    p_abc_mw_phase[np.abs(p_abc_mw_phase) < 1e-6] = 0
    q_abc_mvar_phase[np.abs(q_abc_mvar_phase) < 1e-6] = 0

    # -----------------------------
    # 8) Fill result matrices
    # -----------------------------
    ikss_abc_ka[np.ix_(ppc_indices, [0, 1, 2])] = i_total_abc_ka_abs
    ikss_abc_degree[np.ix_(ppc_indices, [0, 1, 2])] = i_total_abc_ka_angle
    skss_abc_mva[np.ix_(ppc_indices, [0, 1, 2])] = skss_abc_mva_phase
    p_abc_mw[np.ix_(ppc_indices, [0, 1, 2])] = p_abc_mw_phase
    q_abc_mvar[np.ix_(ppc_indices, [0, 1, 2])] = q_abc_mvar_phase

    v_abc_abs[np.ix_(ppc_indices, [0, 1, 2])] = np.abs(v_abc_pu)
    v_abc_deg[np.ix_(ppc_indices, [0, 1, 2])] = np.angle(v_abc_pu, deg=True)

    # -----------------------------
    # 9) Write results to net.res_bus_sc
    # -----------------------------
    for i, phase in enumerate(['a', 'b', 'c']):
        # Current magnitude and angle
        net.res_bus_sc[f'ikss_{phase}_ka'] = ikss_abc_ka[:, i]
        net.res_bus_sc[f'ikss_{phase}_degree'] = ikss_abc_degree[:, i]

        # Apparent power
        net.res_bus_sc[f'skss_{phase}_mva'] = skss_abc_mva[:, i]

        # Active and reactive power
        net.res_bus_sc[f'p_{phase}_mw'] = p_abc_mw[:, i]
        net.res_bus_sc[f'q_{phase}_mvar'] = q_abc_mvar[:, i]

        # Voltage magnitude and angle
        net.res_bus_sc[f'vm_{phase}_pu'] = v_abc_abs[:, i]
        net.res_bus_sc[f'va_{phase}_degree'] = v_abc_deg[:, i]

#==============================================================================================================
#LINES
def _calculate_line_results_all_cases(ppc_0, ppc_1, ppc_2, net):

    branch_lookup = net._pd2ppc_lookups["branch"]

    if "line" not in branch_lookup:
        return

    f, t = branch_lookup["line"]
    fault_type = net["_options"]["fault"]
    case = net._options["case"]

    # -----------------------------
    # 1) Extract sequence currents and voltages from all three networks
    # -----------------------------
    def _get_sequence_branch(ppc, mag_cols, angle_cols):
        """Helper to extract sequence branch quantities with phase."""
        magnitude = ppc['branch'][:, mag_cols]
        angle = np.deg2rad(ppc['branch'][:, angle_cols].real)
        return magnitude * np.exp(1j * angle)

    # Currents: [from, to] for each sequence
    i_ka_0 = _get_sequence_branch(ppc_0, [IKSS_F, IKSS_T], [IKSS_ANGLE_F, IKSS_ANGLE_T])
    i_ka_1 = _get_sequence_branch(ppc_1, [IKSS_F, IKSS_T], [IKSS_ANGLE_F, IKSS_ANGLE_T])
    i_ka_2 = _get_sequence_branch(ppc_2, [IKSS_F, IKSS_T], [IKSS_ANGLE_F, IKSS_ANGLE_T])

    # Voltages: [from, to] for each sequence
    v_pu_0 = _get_sequence_branch(ppc_0, [VKSS_MAGN_F, VKSS_MAGN_T], [VKSS_ANGLE_F, VKSS_ANGLE_T])
    v_pu_1 = _get_sequence_branch(ppc_1, [VKSS_MAGN_F, VKSS_MAGN_T], [VKSS_ANGLE_F, VKSS_ANGLE_T])
    v_pu_2 = _get_sequence_branch(ppc_2, [VKSS_MAGN_F, VKSS_MAGN_T], [VKSS_ANGLE_F, VKSS_ANGLE_T])

    # -----------------------------
    # 2) Adjust sequence components based on fault type
    # -----------------------------
    ref_shape = i_ka_1.shape[0]

    if fault_type == "LLL":
        scaling_factor = 3
    else:  # LL ,LG, LLG
        scaling_factor = 1

    # -----------------------------
    # 3) Transform to phase components
    # -----------------------------
    i_012_ka = np.stack([i_ka_0, i_ka_1, i_ka_2], axis=2)  # Shape: (n_branches, 2, 3)
    i_abc_ka = np.apply_along_axis(sequence_to_phase, 2, i_012_ka/scaling_factor)
    i_abc_ka[np.abs(i_abc_ka) < 1e-5] = 0
    # Pandapower stores branch short-circuit currents already as phase quantities
    # for asymmetrical faults (LL, LG, LLG).
    # In these cases, the sequence-to-phase transformation directly yields the
    # physical phase currents, so no additional scaling is required.
    #
    # For an LLL fault, the system is fully symmetrical and pandapower stores the
    # total three-phase current contribution in the positive-sequence network.
    # Since all three phases carry identical currents, dividing by 3 converts the
    # stored value into the correct per-phase current before the transformation.

    v_012_pu = np.stack([v_pu_0, v_pu_1, v_pu_2], axis=2)  # Shape: (n_branches, 2, 3)
    v_abc_pu = np.apply_along_axis(sequence_to_phase, 2, v_012_pu)
    v_abc_pu[np.abs(v_abc_pu) < 1e-5] = 0

    baseV = ppc_1["internal"]["baseV"][ppc_1["branch"][:, [F_BUS, T_BUS]].real.astype(np.int64)]
    v_base_kv = np.stack([baseV, baseV, baseV], axis=2)  # Shape: (n_branches, 2, 3)


    # -----------------------------
    # 4) Apparent, active and reactive power per phase
    # -----------------------------
    # Power calculation is intentionally handled differently for LLL and asymmetrical faults.
    # Even though different formulas are used internally, the results stored in
    # net.res_line_sc are always per-phase values (a, b, c) for consistency across all fault types.
    if fault_type == 'LLL':
        #   For LLL (three-phase) faults, the system is fully symmetrical.
        #   Pandapower branch currents in this case represent balanced phase currents,
        #   while baseV corresponds to the line-to-line voltage.
        #   Therefore, three-phase power relations are used:
        #       S_3ph = √3 · V_LL · I_phase
        #   and the resulting power is equally shared by all three phases.

        skss_abc_mva = np.sqrt(3) * np.abs(i_abc_ka) * v_base_kv #apparent power in absolute values per phase
        skss_abc_complex = np.sqrt(3) * v_abc_pu * np.conj(i_abc_ka) * v_base_kv #apparent power in complex values per phase
    else:
        #   For LL, LG and LLG faults, the system is asymmetrical.
        #   Currents and voltages are true phase quantities and power must be computed
        #   using per-phase relations:
        #       S_phase = V_phase · I_phase
        #       with V_phase = V_LL / √3

        skss_abc_mva = np.abs(i_abc_ka) * v_base_kv / np.sqrt(3) #apparent power in absolute values per phase
        skss_abc_complex =  v_abc_pu * np.conj(i_abc_ka) * v_base_kv / np.sqrt(3) #apparent power in complex values per phase

    # -----------------------------
    # 5) Extract line results
    # -----------------------------
    i_abc_ka_lines = i_abc_ka[f:t, :, :]
    v_abc_pu_lines = v_abc_pu[f:t, :, :]
    skss_abc_complex_lines = skss_abc_complex[f:t, :, :]
    skss_abc_mva_lines = skss_abc_mva[f:t, :, :]

    # -----------------------------
    # 6) Fill result dataframe in specified order
    # -----------------------------
    # Bus connections
    net.res_line_sc["from_bus"] = ppc_1["branch"][f:t, F_BUS].real.astype(int)
    net.res_line_sc["to_bus"] = ppc_1["branch"][f:t, T_BUS].real.astype(int)

    # Per-phase currents - FROM side first, then TO side
    for phase_idx, phase in enumerate(['a', 'b', 'c']):
        net.res_line_sc[f"ikss_{phase}_from_ka"] = np.abs(i_abc_ka_lines[:, 0, phase_idx])
        net.res_line_sc[f"ikss_{phase}_from_degree"] = np.angle(i_abc_ka_lines[:, 0, phase_idx], deg=True)

    for phase_idx, phase in enumerate(['a', 'b', 'c']):
        net.res_line_sc[f"ikss_{phase}_to_ka"] = np.abs(i_abc_ka_lines[:, 1, phase_idx])
        net.res_line_sc[f"ikss_{phase}_to_degree"] = np.angle(i_abc_ka_lines[:, 1, phase_idx], deg=True)

    # Per-phase apparent power
    for phase_idx, phase in enumerate(['a', 'b', 'c']):
        net.res_line_sc[f"skss_{phase}_mva"] = np.mean(skss_abc_mva_lines[:, :, phase_idx], axis=1)

    # Per-phase active and reactive power - FROM side
    for phase_idx, phase in enumerate(['a', 'b', 'c']):
        net.res_line_sc[f"p_{phase}_from_mw"] = skss_abc_complex_lines[:, 0, phase_idx].real
        net.res_line_sc[f"q_{phase}_from_mvar"] = skss_abc_complex_lines[:, 0, phase_idx].imag

    # Per-phase active and reactive power - TO side
    for phase_idx, phase in enumerate(['a', 'b', 'c']):
        net.res_line_sc[f"p_{phase}_to_mw"] =  skss_abc_complex_lines[:, 1, phase_idx].real
        net.res_line_sc[f"q_{phase}_to_mvar"] = skss_abc_complex_lines[:, 1, phase_idx].imag

    # Optional: ip and ith
    if net._options.get("ip", False):
        net.res_line_sc["ip_ka"] = minmax(ppc_1["branch"][f:t, [IP_F, IP_T]].real, axis=1)
    if net._options.get("ith", False):
        net.res_line_sc["ith_ka"] = minmax(ppc_1["branch"][f:t, [ITH_F, ITH_T]].real, axis=1)


#==============================================================================================================
#SYNCHRONOUS GENERATORS
def _get_gen_results(net, ppc_0, ppc_1, ppc_2):
    """
    Calculates the short-circuit contribution from generators.  
    `Uses a BRANCH lookup to identify the branch where the FROM bus equals the generator bus.  

    Currently, this approach works only when the generator is located at an "end bus" that 
    does not connect two lines, but instead connects the generator directly to the rest of 
    the grid.  

    This method could potentially be expanded to handle more general cases, but further 
    investigation is needed. For example, scenarios may include static generators and 
    synchronous generators connected to the same bus, or even parallel synchronous     
    generators operating together. This should be investigated further, as FROM-bus branch values represent the
    sum of contributions from all elements connected to the same bus.

    Per-phase FROM-side values are stored in `net.res_gen_sc`.

    """
    #NOTE there is a problem when static generator or more synchronus generators are turned on synchronous on the same bus

    # ----------------------------------------
    # 0) Prepare output table
    # ----------------------------------------
    net.res_gen_sc = pd.DataFrame()

    fault_type = net._options["fault"]

    # ----------------------------------------
    # 1) Loop through each generator
    # ----------------------------------------
    results = []

    for _, gen in net.gen.iterrows():

        gen_bus = int(gen["bus"])

        # ----------------------------------------
        # 2) Locate branch where FROM bus == generator bus
        # ----------------------------------------
        branch = ppc_1["branch"]
        f_bus = branch[:, F_BUS].astype(int)

        gen_idx = np.where(f_bus == gen_bus)[0]

        if len(gen_idx) == 0:
            continue

        gen_idx = gen_idx[0]  # assume one outgoing branch
        #todo - Case when two generators are connected to the same bus!-

        # ----------------------------------------
        # 3) Helper to extract sequence components
        # ----------------------------------------
        def _seq(ppc, mag_col, ang_col):
            mag = ppc["branch"][gen_idx, mag_col]
            ang = np.deg2rad(ppc["branch"][gen_idx, ang_col].real)
            return mag * np.exp(1j * ang)

        # currents
        i0 = _seq(ppc_0, IKSS_F, IKSS_ANGLE_F)
        i1 = _seq(ppc_1, IKSS_F, IKSS_ANGLE_F)
        i2 = _seq(ppc_2, IKSS_F, IKSS_ANGLE_F)
        i012 = np.array([i0, i1, i2])

        # voltages
        v0 = _seq(ppc_0, VKSS_MAGN_F, VKSS_ANGLE_F)
        v1 = _seq(ppc_1, VKSS_MAGN_F, VKSS_ANGLE_F)
        v2 = _seq(ppc_2, VKSS_MAGN_F, VKSS_ANGLE_F)
        v012 = np.array([v0, v1, v2])

        # scaling (same as ext_grid and lines - explained in lines function)
        scaling_factor = 3 if fault_type == "LLL" else 1

        # ----------------------------------------
        # 4) Convert sequence → phase
        # ----------------------------------------
        i_abc = sequence_to_phase(i012 / scaling_factor)
        v_abc = sequence_to_phase(v012)

        # zero cleaning
        if abs(i_abc).max() < 1e-5:
            i_abc[:] = 0
        if abs(v_abc).max() < 1e-5:
            v_abc[:] = 0

        # ----------------------------------------
        # 5) Compute S per phase
        # ----------------------------------------
        baseV = ppc_1["internal"]["baseV"][gen_bus]


        if fault_type == "LLL":
            s_gen = np.sqrt(3) * abs(i_abc) * baseV
            s_gen_complex = np.sqrt(3) * v_abc * np.conj(i_abc) * baseV
        else:
            s_gen = abs(i_abc) * baseV / np.sqrt(3)
            s_gen_complex =  v_abc * np.conj(i_abc) * baseV / np.sqrt(3)

        # ----------------------------------------
        # 6) Store results for each generator
        # ----------------------------------------
        phases = ["a", "b", "c"]

        row = {"bus": gen_bus, "gen_idx": gen.name}

        for idx, ph in enumerate(phases):
            row[f"ikss_{ph}_ka"] = abs(i_abc[idx])
            row[f"ikss_{ph}_degree"] = np.angle(i_abc[idx], deg=True)
            row[f"skss_{ph}_mva"] = s_gen[idx]
            row[f"p_{ph}_mw"] = s_gen_complex[idx].real
            row[f"q_{ph}_mvar"] = s_gen_complex[idx].imag

        results.append(row)

    # ----------------------------------------
    # 7) Populate output dataframe
    # ----------------------------------------
    net.res_gen_sc = pd.DataFrame(results)

#===============================================================================================================================================

def _get_sgen_results(net, ppci, bus):
    """
    Compute INDIVIDUAL short-circuit contribution of each static generator (current-source).
    Results are stored into net.res_sgen_sc.Active and reactive power seem to be only problematic parameters.
    """

    # Prepare output table
    net.res_sgen_sc = pd.DataFrame()
    fault_type = net._options["fault"]

    # Lookups
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    baseI = ppci["internal"]["baseI"]
    Zbus = ppci["internal"]["Zbus"]
    baseV = ppci["internal"]["baseV"][bus]

    # Fault impedance modification (LL faults)
    fault_impedance = net._options["fault_impedance"]
    if net._options["fault"] == "LL":
        fault_impedance /= 2

    # Select active static generators working as current sources
    sg = net.sgen[net._is_elements_final["sgen"] & net.sgen.current_source]
    if len(sg) == 0:
        return

    rows = []

    # ---------------------------------------------------------
    # Loop through all static generators one-by-one
    # ---------------------------------------------------------
    for idx, sgen in sg.iterrows():

        bus = int(sgen.bus)
        bus_ppc = bus_lookup[bus]

        # ---------------------------------------------------------
        # 1) Reset ppci bus injections (remove contribution of all other elements)
        # ---------------------------------------------------------
        ppci["bus"][:, IKCV] = 0
        ppci["bus"][:, PHI_IKCV_DEGREE] = -90
        ppci["bus"][:, IKSSC] = 0

        # ---------------------------------------------------------
        # 2) Compute the static generator current injection (p.u.)
        # ---------------------------------------------------------
        if sgen.active_current:
            # Active-current mode: uses P_mw
            i_pu = (sgen.p_mw * sgen.scaling / net.sn_mva) * sgen.k
        else:
            # Nominal apparent power mode
            i_pu = (sgen.sn_mva * sgen.scaling / net.sn_mva) * sgen.k

        # Angle (if present)
        if "current_angle_degree" in sgen:
            angle = np.deg2rad(sgen.current_angle_degree)
            i_pu *= np.exp(1j * angle)

        # Inject this single SG into ppci
        ppci["bus"][bus_ppc, IKCV] = np.abs(i_pu)
        ppci["bus"][bus_ppc, PHI_IKCV_DEGREE] = np.angle(i_pu, deg=True)

        # ---------------------------------------------------------
        # 3) Superposition – compute resulting fault current contribution
        # ---------------------------------------------------------
        # Zbus diagonal modified by fault impedance
        diagZ = np.diag(Zbus).copy()
        diagZ[bus] += fault_impedance

        # Complex bus injections
        inj = ppci["bus"][:, IKCV] * np.exp(1j * np.deg2rad(ppci["bus"][:, PHI_IKCV_DEGREE]))

        # Fault current at all buses
        i_complex = (Zbus @ inj) / diagZ

        # Convert to actual fault current in kA
        ikss = i_complex[bus] / baseI[bus]

        # ---------------------------------------------------------
        # 4) Positive-sequence → phase currents (full converter = only seq1)
        # ---------------------------------------------------------

        i012 = np.array([0, ikss, 0], dtype=complex)
        scaling = 3 if fault_type == "LLL" else 1
        i_abc = sequence_to_phase(i012 / scaling)

        # ---------------------------------------------------------
        # 5) Compute per-phase complex powers
        # ---------------------------------------------------------
        if fault_type == "LLL":
            skss = np.sqrt(3) * abs(i_abc) * baseV
        else:
            skss = abs(i_abc) * baseV / np.sqrt(3)


        # ---------------------------------------------------------
        # 6) Store result for this generator
        # ---------------------------------------------------------
        rows.append({
            "sgen_idx": idx,
            "bus": bus,

            "ikss_a_ka": abs(i_abc[0]),
            "ikss_b_ka": abs(i_abc[1]),
            "ikss_c_ka": abs(i_abc[2]),

            "ikss_a_deg": np.angle(i_abc[0], deg=True),
            "ikss_b_deg": np.angle(i_abc[1], deg=True),
            "ikss_c_deg": np.angle(i_abc[2], deg=True),

            "s_a_mva": abs(skss[0]),
            "s_b_mva": abs(skss[1]),
            "s_c_mva": abs(skss[2]),


            "p_a_mw": skss[0].real, "q_a_mvar": skss[0].imag,
            "p_b_mw": skss[1].real, "q_b_mvar": skss[1].imag,
            "p_c_mw": skss[2].real, "q_c_mvar": skss[2].imag,
        })

    net.res_sgen_sc = pd.DataFrame(rows)


#===================================================================================================================================================

#EXTERNAL GRID
def _get_ext_grid_results(net, ppc_0, ppc_1, ppc_2, bus):
    """
    Calculate short-circuit contribution from external grid using BRANCH lookup.
    Identifies the branch whose FROM bus equals the ext_grid bus.
    Stores per-phase FROM-side results in net.res_ext_grid_sc.
    """

    # -----------------------------
    # 0) Prepare output table
    # -----------------------------
    net.res_ext_grid_sc = pd.DataFrame()

    # -----------------------------
    # 1) Locate ext_grid bus in pandapower model
    # -----------------------------
    if len(net.ext_grid) == 0:

        return

    eg_bus = int(net.ext_grid["bus"].iloc[0])

    # -----------------------------
    # 2) Locate corresponding branch in ppc
    # -----------------------------
    branch = ppc_1["branch"]
    f_bus = branch[:, F_BUS].astype(int)

    # branch index where FROM bus equals ext_grid bus
    ext_idx = np.where(f_bus == eg_bus)[0]

    if len(ext_idx) == 0:

        return

    ext_idx = ext_idx[0]  # assume one branch

    # -----------------------------
    # 3) Extract sequence quantities
    # -----------------------------
    def _seq(ppc, mag_col, ang_col):
        mag = ppc["branch"][ext_idx, mag_col]
        ang = np.deg2rad(ppc["branch"][ext_idx, ang_col].real)
        return mag * np.exp(1j * ang)

    # currents
    i0 = _seq(ppc_0, IKSS_F, IKSS_ANGLE_F)
    i1 = _seq(ppc_1, IKSS_F, IKSS_ANGLE_F)
    i2 = _seq(ppc_2, IKSS_F, IKSS_ANGLE_F)
    i012 = np.array([i0, i1, i2])

    # voltages
    v0 = _seq(ppc_0, VKSS_MAGN_F, VKSS_ANGLE_F)
    v1 = _seq(ppc_1, VKSS_MAGN_F, VKSS_ANGLE_F)
    v2 = _seq(ppc_2, VKSS_MAGN_F, VKSS_ANGLE_F)
    v012 = np.array([v0, v1, v2])

    # scaling - well explained lines functions- similar logic
    fault_type = net._options["fault"]
    scaling_factor = 3 if fault_type == "LLL" else 1

    # -----------------------------
    # 4) Transform sequence → phase
    # -----------------------------
    i_abc = sequence_to_phase(i012 / scaling_factor)
    v_abc = sequence_to_phase(v012)

    # zero cleaning
    if abs(i_abc).max() < 1e-5:
        i_abc[:] = 0
    if abs(v_abc).max() < 1e-5:
        v_abc[:] = 0

    # -----------------------------
    # 5) Compute per-phase power
    # -----------------------------
    baseV = ppc_1["internal"]["baseV"][eg_bus]

    if fault_type == "LLL":
        s_eg =  np.sqrt(3)* abs(i_abc) * baseV
        s_eg_complex = np.sqrt(3) * v_abc * np.conj(i_abc) * baseV
    else:
        s_eg = abs(i_abc) * baseV / np.sqrt(3)
        s_eg_complex =  v_abc * np.conj(i_abc) * baseV / np.sqrt(3)

    # -----------------------------
    # 6) Fill result table
    # -----------------------------
    phases = ["a", "b", "c"]

    net.res_ext_grid_sc["bus"] = [eg_bus]

    for idx, ph in enumerate(phases):
        net.res_ext_grid_sc[f"ikss_{ph}_ka"] = [abs(i_abc[idx])]
        net.res_ext_grid_sc[f"ikss_{ph}_degree"] = [np.angle(i_abc[idx], deg=True)]

    for idx, ph in enumerate(phases):
        net.res_ext_grid_sc[f"skss_{ph}_mva"] = [s_eg[idx]]

    for idx, ph in enumerate(phases):
        net.res_ext_grid_sc[f"p_{ph}_mw"] = [s_eg_complex[idx].real]
        net.res_ext_grid_sc[f"q_{ph}_mvar"] = [s_eg_complex[idx].imag]



