# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
import pandas as pd
from copy import deepcopy

from pandapower.auxiliary import sequence_to_phase
from pandapower.pypower.idx_brch import F_BUS, T_BUS
from pandapower.pypower.idx_brch_sc import IKSS_F, IKSS_T, IP_F, IP_T, ITH_F, ITH_T, PKSS_F, QKSS_F, PKSS_T, QKSS_T, \
    VKSS_MAGN_F, VKSS_MAGN_T, VKSS_ANGLE_F, VKSS_ANGLE_T, IKSS_ANGLE_F, IKSS_ANGLE_T
from pandapower.pypower.idx_bus_sc import IKSSV, IP, ITH, IKSSC, R_EQUIV_OHM, X_EQUIV_OHM, SKSS, PHI_IKSSV_DEGREE, \
    PHI_IKSSC_DEGREE
from pandapower.pypower.idx_bus import BUS_TYPE, BASE_KV
from pandapower.results_branch import _copy_switch_results_from_branches
from pandapower.results import BRANCH_RESULTS_KEYS


def _copy_result_to_ppci_orig(ppci_orig, ppci, ppci_bus, calc_options):
    if ppci_orig is ppci:
        return

    ppci_orig["bus"][ppci_bus, :] = ppci["bus"][ppci_bus, :]
    if calc_options["branch_results"]:
        if calc_options["return_all_currents"]:
            ppci_orig["internal"]["br_res_ks_ppci_bus"] = \
                ppci_bus if "br_res_ks_ppci_bus" not in ppci_orig["internal"] \
                    else np.r_[ppci_orig["internal"]["br_res_ks_ppci_bus"], ppci_bus]

            for res_key in BRANCH_RESULTS_KEYS:
                # Skip not required data points
                if res_key not in ppci["internal"]:
                    continue

                if res_key not in ppci_orig["internal"]:
                    ppci_orig["internal"][res_key] = ppci["internal"][res_key]
                else:
                    ppci_orig["internal"][res_key] = np.c_[ppci_orig["internal"][res_key],
                    ppci["internal"][res_key]]
        else:
            case = calc_options["case"]
            branch_results_cols = [IKSS_F, IKSS_T, IP_F, IP_T, ITH_F, ITH_T]
            # added new calculation values:
            branch_results_cols_add = [IKSS_ANGLE_F, IKSS_ANGLE_T, PKSS_F, QKSS_F, PKSS_T, QKSS_T,
                                       VKSS_MAGN_F, VKSS_MAGN_T, VKSS_ANGLE_F, VKSS_ANGLE_T]
            if case == "max":
                ppci_orig["branch"][:, branch_results_cols] = \
                    np.maximum(np.nan_to_num(ppci["branch"][:, branch_results_cols]),
                               np.nan_to_num(ppci_orig["branch"][:, branch_results_cols]))

                # excluding new values from nan to num
                ppci_orig["branch"][:, branch_results_cols_add] = ppci["branch"][:, branch_results_cols_add]
                ppci_orig["branch"][:, branch_results_cols_add] = ppci["branch"][:, branch_results_cols_add]

                if "branch_LL" in ppci.keys():
                    ppci_orig["branch_LL"] = deepcopy(ppci_orig["branch"])
                    ppci_orig["branch_LL"][:, branch_results_cols] = \
                        np.maximum(np.nan_to_num(ppci["branch_LL"][:, branch_results_cols]),
                                   np.nan_to_num(ppci_orig["branch_LL"][:, branch_results_cols]))
                    ppci_orig["branch_LL"][:, branch_results_cols_add] = ppci["branch_LL"][:, branch_results_cols_add]

            else:
                ppci_orig["branch"][:, branch_results_cols] = \
                    np.minimum(np.nan_to_num(ppci["branch"][:, branch_results_cols], nan=1e10),
                               np.nan_to_num(ppci_orig["branch"][:, branch_results_cols], nan=1e10))
                # excluding new values from nan to num
                ppci_orig["branch"][:, branch_results_cols_add] = ppci["branch"][:, branch_results_cols_add]
                if "branch_LL" in ppci.keys():
                    ppci_orig["branch_LL"] = deepcopy(ppci_orig["branch"])
                    ppci_orig["branch_LL"][:, branch_results_cols] = \
                        np.minimum(np.nan_to_num(ppci["branch_LL"][:, branch_results_cols], nan=1e10),
                                   np.nan_to_num(ppci_orig["branch_LL"][:, branch_results_cols], nan=1e10))
                    # excluding new values from nan to num
                    ppci_orig["branch_LL"][:, branch_results_cols_add] = ppci["branch_LL"][:, branch_results_cols_add]


def _get_bus_ppc_idx_for_br_all_results(net, ppc, bus):
    bus_lookup = net._pd2ppc_lookups["bus"]
    if bus is None:
        bus = net.bus.index

    ppc_index = bus_lookup[bus]
    ppc_index[ppc["bus"][ppc_index, BUS_TYPE] == 4] = -1
    return bus, ppc_index


def _calculate_branch_phase_results(ppc_0, ppc_1, ppc_2):
    # we use 3D arrays here to easily identify via axis:
    # 0: line index, 1: from/to, 2: phase
    i_ka_0 = ppc_0['branch'][:, [IKSS_F, IKSS_T]] * np.exp(
        1j * np.deg2rad(ppc_0['branch'][:, [IKSS_ANGLE_F, IKSS_ANGLE_T]].real))
    i_ka_1 = ppc_1['branch'][:, [IKSS_F, IKSS_T]] * np.exp(
        1j * np.deg2rad(ppc_1['branch'][:, [IKSS_ANGLE_F, IKSS_ANGLE_T]].real))
    i_ka_2 = ppc_2['branch'][:, [IKSS_F, IKSS_T]] * np.exp(
        1j * np.deg2rad(ppc_2['branch'][:, [IKSS_ANGLE_F, IKSS_ANGLE_T]].real))

    """i_ka_0_c = ppc_0["bus"][:, IKSSC]
    i_ka_1_c = ppc_1["bus"][:, IKSSC]
    i_ka_2_c = ppc_2["bus"][:, IKSSC]
    i_012_c_ka = np.stack([i_ka_0_c, i_ka_1_c, i_ka_2_c], 0)
    i_abc_c_ka = np.apply_along_axis(sequence_to_phase, 0, i_012_c_ka)
    i_abc_c_ka = abs(i_abc_c_ka)
    i_abc_c_ka[np.abs(i_abc_c_ka) < 1e-5] = 0"""

    # TODO branch phase reuslts for all currents
    """branch_lookup = net._pd2ppc_lookups["branch"]
    ppc_0["internal"]["branch_ikss_f"] = np.nan_to_num(np.abs(ikss_all_f)) / baseI[fb, None]
    ppc_0["internal"]["branch_ikss_t"] = np.nan_to_num(np.abs(ikss_all_t)) / baseI[tb, None]

    ppc_0["internal"]["branch_ikss_angle_f"] = np.nan_to_num(np.angle(ikss_all_f, deg=True))
    ppc_0["internal"]["branch_ikss_angle_t"] = np.nan_to_num(np.angle(ikss_all_t, deg=True))"""

    i_012_ka = np.stack([i_ka_0, i_ka_1, i_ka_2], 2)
    i_abc_ka = np.apply_along_axis(sequence_to_phase, 2, i_012_ka)
    # i_abc_ka = sequence_to_phase(np.vstack([i_ka_0, i_ka_1, i_ka_2]))
    i_abc_ka[np.abs(i_abc_ka) < 1e-5] = 0
    # baseI = ppc_1["internal"]["baseI"][ppc_1["branch"][:, [F_BUS, T_BUS]].real.astype(np.int64)]
    # i_base_ka = np.stack([baseI, baseI, baseI], 2)
    # i_abc_ka /= i_base_ka

    v_pu_0 = ppc_0['branch'][:, [VKSS_MAGN_F, VKSS_MAGN_T]] * np.exp(
        1j * np.deg2rad(ppc_0['branch'][:, [VKSS_ANGLE_F, VKSS_ANGLE_T]].real))
    v_pu_1 = ppc_1['branch'][:, [VKSS_MAGN_F, VKSS_MAGN_T]] * np.exp(
        1j * np.deg2rad(ppc_1['branch'][:, [VKSS_ANGLE_F, VKSS_ANGLE_T]].real))
    v_pu_2 = ppc_2['branch'][:, [VKSS_MAGN_F, VKSS_MAGN_T]] * np.exp(
        1j * np.deg2rad(ppc_2['branch'][:, [VKSS_ANGLE_F, VKSS_ANGLE_T]].real))

    v_012_pu = np.stack([v_pu_0, v_pu_1, v_pu_2], 2)
    v_abc_pu = np.apply_along_axis(sequence_to_phase, 2, v_012_pu)
    # v_abc_pu = sequence_to_phase(np.vstack([v_pu_0, v_pu_1, v_pu_2]))
    v_abc_pu[np.abs(v_abc_pu) < 1e-5] = 0

    # this is inefficient because it copies data to fit into a shape, better to use a slice,
    # and even better to find how to use sequence-based powers:
    baseV = ppc_1["internal"]["baseV"][ppc_1["branch"][:, [F_BUS, T_BUS]].real.astype(np.int64)]
    v_base_kv = np.stack([baseV, baseV, baseV], 2)

    s_abc_mva = np.conj(i_abc_ka) * v_abc_pu * v_base_kv / np.sqrt(3)

    return v_abc_pu, i_abc_ka, s_abc_mva



#===============================================================================================================================================

import numpy as np
import pandas as pd
from numpy import sqrt
from pandapower.pypower.idx_bus_sc import GS_GEN, BS_GEN, R_EQUIV, X_EQUIV, V_G, K_G, C_MAX, C_MIN
from pandapower.pypower.idx_bus import BASE_KV, VM, GS, BS
from pandapower.pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, TAP
from pandapower.pypower.idx_brch_sc import K_T


def _get_gen_results(net, ppci_0, ppci_1, ppci_2, bus):

    # STEP 1: Map pandapower bus indices to PPC indices
    bus_lookup = net._pd2ppc_lookups["bus"]
    bus_fault_ppc = bus_lookup[bus]
    gen_ppc_idx = bus_lookup[net.gen.bus.values]

    # STEP 2: Voltages of generators and correction factors

    # Extract nominal voltage at each generator bus and calculate base impedance (ohm)
    vn_net = net.bus.loc[gen_ppc_idx, "vn_kv"].values
    gen_base_z_ohm = vn_net ** 2  # baseZ = V^2 / S_base (S_base normalized to 1 MVA in pandapower??)
    # NOTE: This is the way that base impedance is calculated in the _add_gen_sc_z_kg_ks
    # function from pandapower/shortcircuit/ppc_conversion.py

    # Internal voltage and generator scaling factor from positive sequence bus data
    # NOTE: These values are the same for all three sequences, so we can extract them
    # from any of them – in this case, from the positive sequence.
    vn_gen = ppci_1["bus"][gen_ppc_idx, V_G]
    kg = ppci_1["bus"][gen_ppc_idx, K_G]

    # STEP 3: Forming generator impedance according to IEC 60909 (Chapter 6.6.1)
    # Extract per-sequence generator admittance data (GS_GEN, BS_GEN, C_MAX)
    # GS_GEN, BS_GEN → conductance and susceptance per generator bus (in pu)
    ppcis = [ppci_0, ppci_1, ppci_2]
    seq_params = np.stack([
        np.column_stack([
            p["bus"][gen_ppc_idx, GS_GEN],
            p["bus"][gen_ppc_idx, BS_GEN],
            p["bus"][gen_ppc_idx, BASE_KV] ** 2
        ])
        for p in ppcis
    ], axis=1)

    # Convert generator admittance (Y = G + jB) to impedance (Z = 1/Y)
    y_gen_pu = seq_params[:, :, 0] + 1j * seq_params[:, :, 1]
    with np.errstate(divide='ignore', invalid='ignore'):
        z_gen_pu = 1 / y_gen_pu
    z_gen = z_gen_pu * seq_params[:, :, 2]  # convert to ohms

    # Sequence impedances per generator (0, 1, 2) with positive/negative scaling factor kg
    # (zero sequence is already scaled)
    z_genk0, z_genk1, z_genk2 = (
        z_gen[:, 0],
    # NOTE: Zero sequence already includes the kg factor in _add_trafo_sc_impedance_zero from pandapower/pd2ppc_zero.py
        z_gen[:, 1] * kg,
        z_gen[:, 2] * kg
    )

    # STEP 4: Equivalent internal voltage source of the generator (Eg) according to IEC 60909 (Chapter 5.3.1)
    # C_MAX adjusts Eg for voltage level under short-circuit conditions in max case
    cmax = ppci_1["bus"][gen_ppc_idx, C_MAX]
    cmin = ppci_1["bus"][gen_ppc_idx, C_MIN]
    case = net._options["case"]
    if case == "max":
        c = cmax
    else:
        c = cmin
    Eg = c * vn_gen / np.sqrt(3)

    # STEP 5: Calculate base impedance at the fault bus for each sequence network
    # (although they are the same for every sequence)
    baseZ1 = (ppci_1["bus"][bus_fault_ppc, BASE_KV] ** 2) / ppci_1["baseMVA"]
    baseZ2 = (ppci_2["bus"][bus_fault_ppc, BASE_KV] ** 2) / ppci_2["baseMVA"]
    baseZ0 = (ppci_0["bus"][bus_fault_ppc, BASE_KV] ** 2) / ppci_0["baseMVA"]

    # STEP 6: Helper function that computes the equivalent impedance between the faulted bus and generator bus
    # This ensures that cases where the fault does not occur directly at the generator bus
    # are also considered.
    def Zbr(ppci, baseZ, bus_fault_ppc, gen_ppc_idx):
        """
        Compute equivalent impedance Zeq between the fault bus and generator buses
        using the Zbus matrix (Zeq = Zii + Zjj - 2*Zij). This gives the exact impedance
        between two buses.
        """
        Zbus = ppci["internal"]["Zbus"]
        gen_ppc_idx = np.atleast_1d(gen_ppc_idx)
        Z_from_from = Zbus[bus_fault_ppc, bus_fault_ppc]
        Z_to_to = np.diag(Zbus[gen_ppc_idx][:, gen_ppc_idx])
        Z_from_to = Zbus[bus_fault_ppc, gen_ppc_idx]
        Zeq = Z_from_from + Z_to_to - 2 * Z_from_to
        Zeq[np.isclose(Zeq, 0)] = 1e-12  # avoid singularities
        return Zeq * baseZ

    # STEP 6 (continued): Equivalent impedances for sequences
    # between faulted bus and generator buses are calculated.
    Zb1 = Zbr(ppci_1, baseZ1, bus_fault_ppc, gen_ppc_idx)
    Zb2 = Zbr(ppci_2, baseZ2, bus_fault_ppc, gen_ppc_idx)
    Zb0 = Zbr(ppci_0, baseZ0, bus_fault_ppc, gen_ppc_idx)

    # sum of the generator impedance and the branch impedance between the generator and the faulted bus

    Zb_gen1 =  z_genk1 + Zb1

    Zb_gen2 = Zb2 + z_genk2

    Zb_gen0 = Zb0 + z_genk0

    # STEP 7: Combine generator and network impedances per sequence
    same_bus = (gen_ppc_idx == bus_fault_ppc)  # generator directly at faulted bus
    # In this way, we check if any generator is located at the faulted bus.
    # Based on this, we decide whether to use only generator impedance or
    # the combined network and generator impedance.

    Z1 = np.where(same_bus, z_genk1, Zb_gen1)
    Z2 = np.where(same_bus, z_genk2, z_genk2)
    Z0 = np.where(same_bus, z_genk0, z_genk0)

    # STEP 8: Determine fault type and compute corresponding symmetrical component currents
    fault_type = net["_options"]["fault"]

    if fault_type == "LLL":      # 3-phase fault
        I1 = Eg / Z1
        I0 = I2 = np.zeros_like(I1)

    elif fault_type == "LG":     # single line-to-ground fault
        I0 = I1 = I2 = Eg / (Z1 + Z2 + Z0)

    elif fault_type == "LL":     # line-to-line fault
        I1 = Eg / (Z1 + Z2)
        I2 = -I1
        I0 = np.zeros_like(I1)

    elif fault_type == "LLG":    # double line-to-ground fault
        I1 = Eg / (Z1 + (Z2 * Z0) / (Z2 + Z0))
        I2 = -I1 * Z0 / (Z2 + Z0)
        I0 = -I1 * Z2 / (Z2 + Z0)

    # STEP 9: Convert symmetrical component currents to phase currents and compute results
    results = []
    for i in range(len(gen_ppc_idx)):
        I_phase = sequence_to_phase(np.array([I0[i], I1[i], I2[i]], dtype=complex)).squeeze()
        I_a, I_b, I_c = I_phase

        # Per-phase voltage base and apparent powers
        V_phase_kv = vn_gen[i] / np.sqrt(3)
        skss_a_mva = V_phase_kv * abs(I_a)
        skss_b_mva = V_phase_kv * abs(I_b)
        skss_c_mva = V_phase_kv * abs(I_c)
        skss = skss_a_mva + skss_b_mva + skss_c_mva

        results.append({
            "bus": net.gen.bus.values[i],
            "ikss_a_ka": abs(I_a), "ikss_b_ka": abs(I_b), "ikss_c_ka": abs(I_c),
            "ikss_a_degree": np.angle(I_a, deg=True),
            "ikss_b_degree": np.angle(I_b, deg=True),
            "ikss_c_degree": np.angle(I_c, deg=True),
            "skss": skss,
            "skss_a_mva": skss_a_mva, "skss_b_mva": skss_b_mva, "skss_c_mva": skss_c_mva,
            "p_a_mw": skss_a_mva * np.cos(np.angle(I_a)),
            "q_a_mvar": skss_a_mva * np.sin(np.angle(I_a)),
            "p_b_mw": skss_b_mva * np.cos(np.angle(I_b)),
            "q_b_mvar": skss_b_mva * np.sin(np.angle(I_b)),
            "p_c_mw": skss_c_mva * np.cos(np.angle(I_c)),
            "q_c_mvar": skss_c_mva * np.sin(np.angle(I_c)),
        })

    # --- Store generator short-circuit results in the network results table
    net.res_gen_sc = pd.DataFrame(results)




#===============================================================================================================================================

def _get_sgen_results(net, ppci_0, ppci_1, ppci_2, bus):

    # NOTE: This implementation is based on _current_source_current() from
    # pandapower/shortcircuit/currents.py

    # STEP 0: Early Exit if No Static Generators Exist

    if "sgen" not in net or len(net.sgen) == 0:
        return

    sgen_df = net.sgen
    n_sgen = len(sgen_df)
    if n_sgen == 0:
        return

    # STEP 1: Map pandapower Bus Indices to Internal ppc Indices
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    bus_fault_ppc = np.array([bus_lookup[bus]])
    n_faults = len(bus_fault_ppc)
    sgen_buses_ppc = np.array([bus_lookup[b] for b in sgen_df.bus.values])

    # STEP 2: Extract Equivalent Impedances at Fault Location(s)
    z_equiv = np.zeros((n_faults, 3), dtype=complex)
    for i, bf_idx in enumerate(bus_fault_ppc):
        z_equiv[i, 0] = ppci_0["bus"][bf_idx, R_EQUIV] + ppci_0["bus"][bf_idx, X_EQUIV] * 1j  # Zero sequence
        z_equiv[i, 1] = ppci_1["bus"][bf_idx, R_EQUIV] + ppci_1["bus"][bf_idx, X_EQUIV] * 1j  # Positive sequence
        z_equiv[i, 2] = ppci_2["bus"][bf_idx, R_EQUIV] + ppci_2["bus"][bf_idx, X_EQUIV] * 1j  # Negative sequence

    # STEP 3: Setup Fault Impedance
    fault_impedance = net._options["fault_impedance"]
    if net["_options"]["fault"] == "LL":
        fault_impedance /= 2  # Effective impedance halved for line-to-line fault

    # STEP 4: Identify Active Static Generators and Calculate Injection Currents
    mask_in_service = sgen_df.in_service.values
    mask_current_source = (
        sgen_df.current_source.values if "current_source" in sgen_df.columns
        else np.ones(n_sgen, dtype=bool)
    )
    active_mask = mask_in_service & mask_current_source

    i_sgen_pu = np.zeros(n_sgen, dtype=complex)
    mask_active = sgen_df.active_current.values if "active_current" in sgen_df.columns else np.zeros(n_sgen, dtype=bool)

    # Currents based on active power (P)
    if mask_active.any():
        i_sgen_pu[mask_active] = (
            sgen_df.loc[mask_active, "p_mw"].values *
            sgen_df.loc[mask_active, "scaling"].values /
            net.sn_mva *
            sgen_df.loc[mask_active, "k"].values
        )

    # Currents based on apparent power (S)
    if (~mask_active).any():
        i_sgen_pu[~mask_active] = (
            sgen_df.loc[~mask_active, "sn_mva"].values *
            sgen_df.loc[~mask_active, "scaling"].values /
            net.sn_mva *
            sgen_df.loc[~mask_active, "k"].values
        )

    # Apply current angle if defined
    if "current_angle_degree" in sgen_df.columns:
        angles = sgen_df.current_angle_degree.values
        i_sgen_pu *= np.exp(1j * np.deg2rad(angles))

    # Set inactive sgen currents to zero
    i_sgen_pu[~active_mask] = 0

    # STEP 5: Propagate Currents Through Network (Superposition Principle)
    n_buses = ppci_1["bus"].shape[0]
    ikssc_all = np.zeros((n_faults, n_sgen), dtype=complex)
    baseI_val = ppci_1["internal"]["baseI"]

    for fault_idx, bf_idx in enumerate(bus_fault_ppc):

        # Build current injection matrix (each column = one sgen)
        i_inj_matrix = np.zeros((n_buses, n_sgen), dtype=complex)
        for sgen_idx, (bus_ppc, current) in enumerate(zip(sgen_buses_ppc, i_sgen_pu)):
            i_inj_matrix[bus_ppc, sgen_idx] = current

        if net["_options"]["inverse_y"]:
            # Method 1: Zbus approach ---
            Zbus = ppci_1["internal"]["Zbus"].copy()
            diagZ = np.diag(Zbus).copy()
            diagZ[bf_idx] += fault_impedance
            V = Zbus @ i_inj_matrix
            i_kss_complex = V / diagZ[:, np.newaxis]
        else:
            # Method 2: Ybus factorization approach ---
            ybus_fact = ppci_1["internal"]["ybus_fact"]
            diagZ = np.diag(ppci_1["internal"]["Zbus"]).copy()
            diagZ[bf_idx] += fault_impedance
            i_kss_complex = np.zeros((n_buses, n_sgen), dtype=complex)
            for sgen_idx in range(n_sgen):
                i_kss_complex[:, sgen_idx] = ybus_fact(i_inj_matrix[:, sgen_idx]) / diagZ

        # Extract fault bus current contribution
        fault_currents = i_kss_complex[bf_idx, :]

        # Convert per-unit to physical units (kA)
        if isinstance(baseI_val, np.ndarray) and baseI_val.ndim > 0 and len(baseI_val) > 1:
            ikssc_all[fault_idx, :] = fault_currents / baseI_val[bf_idx]
        else:
            base_i_scalar = float(baseI_val) if np.isscalar(baseI_val) else float(
                baseI_val.item() if baseI_val.size == 1 else baseI_val
            )
            ikssc_all[fault_idx, :] = fault_currents / base_i_scalar

    # STEP 6: Extract Single Fault Case
    if n_faults == 1:
        ikssc_sgen = ikssc_all[0, :]
        z_equiv_0, z_equiv_1, z_equiv_2 = z_equiv[0, :]
    else:
        raise NotImplementedError("Multiple fault buses not yet implemented")

    # STEP 7: Calculate Sequence Currents (IEC 60909 Fault Types)
    I_a = np.full(n_sgen, np.nan, dtype=complex)
    I_b = np.full(n_sgen, np.nan, dtype=complex)
    I_c = np.full(n_sgen, np.nan, dtype=complex)

    if net["_options"]["fault"] == "LLL":
        I1 = ikssc_sgen
        I2 = np.zeros_like(I1)
        I0 = np.zeros_like(I1)
    elif net["_options"]["fault"] == "LG":
        factor = abs(z_equiv_1 / (z_equiv_1 + z_equiv_2 + z_equiv_0))
        I1 = I2 = I0 = factor * ikssc_sgen

    elif net["_options"]["fault"] == "LL":
        I1 = ikssc_sgen / 2
        I2 = -I1
        I0 = np.zeros_like(I1)

    elif net["_options"]["fault"] == "LLG":
        I1 = abs(ikssc_sgen * ((z_equiv_1 * (z_equiv_0 + z_equiv_2)) /
                               (z_equiv_2 * z_equiv_0 + z_equiv_1 * z_equiv_0 + z_equiv_1 * z_equiv_2)))
        I2 = -I1 * z_equiv_0 / (z_equiv_2 + z_equiv_0)
        I0 = -I1 * z_equiv_2 / (z_equiv_2 + z_equiv_0)

    # STEP 8: Transform Sequence to Phase Currents
    I_seq_all = np.vstack([I0, I1, I2])
    for idx in range(n_sgen):
        if active_mask[idx]:
            I_seq = I_seq_all[:, idx]
            I_phase = sequence_to_phase(I_seq).squeeze()
            I_a[idx], I_b[idx], I_c[idx] = I_phase

    # STEP 9: Calculate Apparent Power Contributions
    vn_sgen = np.array([net.bus.at[b, "vn_kv"] for b in sgen_df.bus.values])
    V_phase_kv = vn_sgen / np.sqrt(3)
    skss_a_mva = V_phase_kv * np.abs(I_a)
    skss_b_mva = V_phase_kv * np.abs(I_b)
    skss_c_mva = V_phase_kv * np.abs(I_c)
    skss = skss_a_mva + skss_b_mva + skss_c_mva

    # STEP 10: Build Results DataFrame
    results_dict = {
        "bus": sgen_df.bus.values,
        "ikss_a_ka": np.abs(I_a),
        "ikss_b_ka": np.abs(I_b),
        "ikss_c_ka": np.abs(I_c),
        "ikss_a_degree": np.angle(I_a, deg=True),
        "ikss_b_degree": np.angle(I_b, deg=True),
        "ikss_c_degree": np.angle(I_c, deg=True),
        "skss_mva": skss,
        "skss_a_mva": skss_a_mva,
        "skss_b_mva": skss_b_mva,
        "skss_c_mva": skss_c_mva,
        "p_a_mw": skss_a_mva * np.cos(np.angle(I_a)),
        "q_a_mvar": skss_a_mva * np.sin(np.angle(I_a)),
        "p_b_mw": skss_b_mva * np.cos(np.angle(I_b)),
        "q_b_mvar": skss_b_mva * np.sin(np.angle(I_b)),
        "p_c_mw": skss_c_mva * np.cos(np.angle(I_c)),
        "q_c_mvar": skss_c_mva * np.sin(np.angle(I_c)),
    }

    net.res_sgen_sc = pd.DataFrame(results_dict, index=sgen_df.index)
    return net



#===================================================================================================================================================


def _get_ext_grid_results(net, ppci_0, ppci_1, ppci_2, bus):

    # STEP 1: Map pandapower buses to internal PPC indices
    bus_lookup = net._pd2ppc_lookups["bus"]
    bus_fault_ppc = bus_lookup[bus]

    # STEP 2: Select active external grids
    # External grids represent Thevenin equivalents.
    ext_grids = net.ext_grid[net.ext_grid.get("in_service", True)].copy()
    if ext_grids.empty:
        # If no external grid exists, store NaN placeholders for results.
        net.res_ext_grid_sc = pd.DataFrame([{
            "bus": np.nan,
            "ikss_a_ka": np.nan, "ikss_b_ka": np.nan, "ikss_c_ka": np.nan,
            "ikss_a_degree": np.nan, "ikss_b_degree": np.nan, "ikss_c_degree": np.nan,
            "skss": np.nan, "skss_a_mva": np.nan, "skss_b_mva": np.nan, "skss_c_mva": np.nan,
            "p_a_mw": np.nan, "q_a_mvar": np.nan,
            "p_b_mw": np.nan, "q_b_mvar": np.nan,
            "p_c_mw": np.nan, "q_c_mvar": np.nan
        }])
        return

    # STEP 3: Extract external grid parameters
    bus_ext_ppc = bus_lookup[ext_grids.bus.values]
    vn_net = net.bus.loc[ext_grids.bus.values, "vn_kv"].values     # Nominal voltage [kV]
    s_sc = ext_grids.s_sc_max_mva.values                           # Short-circuit power [MVA]
    rx = ext_grids.rx_max.values                                   # R/X ratio
    fault_type = net["_options"]["fault"]                          # Fault type

    # STEP 4: Calculate Thevenin equivalent voltage and impedance
    # According to IEC 60909- Chapter 6.2: Network feeders
    cmax = ppci_1["bus"][bus_ext_ppc, C_MAX]
    cmin = ppci_1["bus"][bus_ext_ppc, C_MIN]
    case = net._options["case"]
    if case == "max":
        c = cmax
    else:
        c = cmin

    Eg = c * vn_net / np.sqrt(3)                                # Equivalent source voltage

    ppcis = [ppci_0, ppci_1, ppci_2]
    seq_params = np.stack([
        np.column_stack([
            p["bus"][bus_ext_ppc, GS],
            p["bus"][bus_ext_ppc, BS],
            p["bus"][bus_ext_ppc, BASE_KV]**2
        ])
        for p in ppcis
    ], axis=1)

    # Convert external grid admittance (Y = G + jB) to impedance (Z = 1/Y)
    y_ext_grid_pu = seq_params[:, :, 0] + 1j * seq_params[:, :, 1]
    with np.errstate(divide='ignore', invalid='ignore'):
        z_ext_grid_pu = 1 / y_ext_grid_pu
    z_ext_grid = z_ext_grid_pu * seq_params[:, :, 2]

    z_exgk0, z_exgk1, z_exgk2 = z_ext_grid[:, 0], z_ext_grid[:, 1], z_ext_grid[:, 2]

    # STEP 5: Calculate base impedance for each sequence network
    baseZ0 = (ppci_0["bus"][bus_fault_ppc, BASE_KV] ** 2) / ppci_0["baseMVA"]
    baseZ1 = (ppci_1["bus"][bus_fault_ppc, BASE_KV] ** 2) / ppci_1["baseMVA"]
    baseZ2 = (ppci_2["bus"][bus_fault_ppc, BASE_KV] ** 2) / ppci_2["baseMVA"]

    # STEP 6: Define helper to get transfer impedance via Zbus
    def Zbr(ppci, baseZ, bus_fault_ppc, bus_ext_ppc):
        """
        Compute equivalent impedance between faulted bus and external grid bus
        using the Zbus matrix:
            Zeq = Zii + Zjj - 2*Zij
        """
        Zbus = ppci["internal"]["Zbus"]
        Z_from_from = Zbus[bus_fault_ppc, bus_fault_ppc]

        if np.isscalar(bus_ext_ppc):
            Z_to_to = Zbus[bus_ext_ppc, bus_ext_ppc]
            Z_from_to = Zbus[bus_fault_ppc, bus_ext_ppc]
        else:
            Z_to_to = np.diag(Zbus[bus_ext_ppc][:, bus_ext_ppc])
            Z_from_to = Zbus[bus_fault_ppc, bus_ext_ppc]

        Zeq = Z_from_from + Z_to_to - 2 * Z_from_to
        Zeq[np.isclose(Zeq, 0)] = 1e-12
        return Zeq * baseZ

    # STEP 7: Check if a transformer connects the external grid(NOTE: only direct connection of grid to transformer ,not anything in between)
    trafo = net.trafo[                                             #for example - bus/bus switch
        net.trafo["hv_bus"].isin(ext_grids.bus.values) |
        net.trafo["lv_bus"].isin(ext_grids.bus.values)
    ]

    if not trafo.empty:
        # Transformer parameters
        tr = trafo.iloc[0]
        vn_hv, vn_lv, sn = float(tr.vn_hv_kv), float(tr.vn_lv_kv), float(tr.sn_mva)
        t_nominal = vn_hv / vn_lv                                     # Turns ratio
        hv_bus, lv_bus = bus_lookup[tr.hv_bus], bus_lookup[tr.lv_bus]

        branch = ppci_1["branch"]
        mask = ((branch[:, F_BUS] == hv_bus) & (branch[:, T_BUS] == lv_bus)) | \
               ((branch[:, F_BUS] == lv_bus) & (branch[:, T_BUS] == hv_bus))

        # Transformer impedance (positive & negative sequences)
        ZT1 = (branch[mask, BR_R] + 1j * branch[mask, BR_X]) * baseZ1
        ZT2 = (branch[mask, BR_R] + 1j * branch[mask, BR_X]) * baseZ2

        ZTLVK_1, ZTLVK_2 = ZT1, ZT2

        # Note: Zero-sequence impedance should be obtained directly from ppci_0['bus'] so more complex cases can work
        # Grid impedance referred to LV side
        z_exgk0, z_exgk1, z_exgk2= z_exgk0/(t_nominal ** 2), z_exgk1 / (t_nominal ** 2), z_exgk2 / (t_nominal ** 2)

        # Combine transformer + grid impedances
        z_exgk1 = z_exgk1  + ZTLVK_1
        z_exgk2 = z_exgk2 + ZTLVK_2
        z_exgk0 = (ppci_0["bus"][bus_fault_ppc, R_EQUIV] + 1j *
                   ppci_0["bus"][bus_fault_ppc, X_EQUIV]) * baseZ0

        # LV or HV fault condition
        if lv_bus <= bus_fault_ppc:
            # Fault on LV side → refer Eg and impedances to LV
            Eg = Eg / t_nominal
            vn_net = vn_net / t_nominal

            Zb1 = Zbr(ppci_1, baseZ1, bus_fault_ppc, lv_bus)
            Zb2 = Zbr(ppci_2, baseZ2, bus_fault_ppc, lv_bus)

            Zb_exg1 = Zb1 + z_exgk1 #Combine transformer + grid impedances+rest of the net if fault is happening on distanced bus
            Zb_exg2 = Zb2 + z_exgk2
            Zb_exg0 = z_exgk0
            same_bus = (lv_bus == bus_fault_ppc)

            Z1 = np.where(same_bus, z_exgk1, Zb_exg1)
            Z2 = np.where(same_bus, z_exgk2, Zb_exg2)
            Z0 = np.where(same_bus, z_exgk0, Zb_exg0)
        else:
            # Fault on HV side → transformer not in path
            Z1 = Z2 = Z0 = ZQ_complex
    else:
        #  Direct connection to external grid

        Zb0 = Zbr(ppci_0, baseZ0, bus_fault_ppc, bus_ext_ppc)
        Zb1 = Zbr(ppci_1, baseZ1, bus_fault_ppc, bus_ext_ppc)
        Zb2 = Zbr(ppci_2, baseZ2, bus_fault_ppc, bus_ext_ppc)

        same_bus = (bus_ext_ppc == bus_fault_ppc)

        Zb_exg1 = Zb1 + z_exgk1
        Zb_exg2 = Zb2 + z_exgk2
        Zb_exg0 = Zb0 + z_exgk0

        Z1 = np.where(same_bus, z_exgk1, Zb_exg1)
        Z2 = np.where(same_bus, z_exgk2, Zb_exg2)
        Z0 = np.where(same_bus, z_exgk0, Zb_exg0)

    # STEP 8: Compute fault currents by fault type
    if fault_type == "LLL":      # 3-phase balanced fault
        I1 = Eg / Z1
        I2 = I0 = np.zeros_like(I1)

    elif fault_type == "LG":     # Single line-to-ground fault
        I0 = I1 = I2 = Eg / (Z1 + Z2 + Z0)

    elif fault_type == "LL":     # Line-to-line fault
        I1 = Eg / (Z1 + Z2)
        I2 = -I1
        I0 = np.zeros_like(I1)

    elif fault_type == "LLG":    # Double line-to-ground fault
        I1 = Eg / (Z1 + (Z2 * Z0) / (Z2 + Z0))
        I2 = -I1 * Z0 / (Z2 + Z0)
        I0 = -I1 * Z2 / (Z2 + Z0)

    # STEP 9: Convert to phase domain and compute results
    results = []
    for i, ext_bus in enumerate(ext_grids.bus.values):
        I_phase = sequence_to_phase(np.array([I0[i], I1[i], I2[i]], dtype=complex)).squeeze()
        I_a, I_b, I_c = I_phase
        V_phase_kv = vn_net[i] / np.sqrt(3)

        # Apparent power per phase and total
        skss_a_mva = V_phase_kv * abs(I_a)
        skss_b_mva = V_phase_kv * abs(I_b)
        skss_c_mva = V_phase_kv * abs(I_c)
        skss = skss_a_mva + skss_b_mva + skss_c_mva

        results.append({
            "bus": ext_bus,
            "ikss_a_ka": abs(I_a), "ikss_b_ka": abs(I_b), "ikss_c_ka": abs(I_c),
            "ikss_a_degree": np.angle(I_a, deg=True),
            "ikss_b_degree": np.angle(I_b, deg=True),
            "ikss_c_degree": np.angle(I_c, deg=True),
            "skss": skss,
            "skss_a_mva": skss_a_mva, "skss_b_mva": skss_b_mva, "skss_c_mva": skss_c_mva,
            "p_a_mw": skss_a_mva * np.cos(np.angle(I_a)),
            "q_a_mvar": skss_a_mva * np.sin(np.angle(I_a)),
            "p_b_mw": skss_b_mva * np.cos(np.angle(I_b)),
            "q_b_mvar": skss_b_mva * np.sin(np.angle(I_b)),
            "p_c_mw": skss_c_mva * np.cos(np.angle(I_c)),
            "q_c_mvar": skss_c_mva * np.sin(np.angle(I_c)),
        })

    # STEP 10: Store results in network results table
    net.res_ext_grid_sc = pd.DataFrame(results)

#===================================================================================================================================================
# def _get_line_to_g_results(net, ppc_1, v_abc_pu, i_abc_ka, s_abc_mva):
def _get_line_branch_results(net, ppc_1, v_abc_pu, i_abc_ka, s_abc_mva):  # renamaed the function properly
    branch_lookup = net._pd2ppc_lookups["branch"]
    case = net._options["case"]
    fault = net._options["fault"]

    if "line" in branch_lookup:
        f, t = branch_lookup["line"]
        minmax = np.max if case == "max" else np.min

        # todo: check axis of max with more lines in the grid
        i_max_per_line_ka = np.max(np.abs(i_abc_ka), axis=1)
        net.res_line_sc["ikss_ka"] = minmax(i_max_per_line_ka[f:t, :], axis=1)

        if fault == 'LLL':  # corrected bramch p amd q values
            mult = 3
        else:
            mult = 1

        for phase_idx, phase in enumerate(("a", "b", "c")):
            for side_idx, side in enumerate(("from", "to")):
                net.res_line_sc[f"ikss_{phase}_{side}_ka"] = np.abs(i_abc_ka[f:t, side_idx, phase_idx])
                net.res_line_sc[f"ikss_{phase}_{side}_degree"] = np.angle(i_abc_ka[f:t, side_idx, phase_idx], deg=True)

            for side_idx, side in enumerate(("from", "to")):
                net.res_line_sc[f"p_{phase}_{side}_mw"] = s_abc_mva[f:t, side_idx, phase_idx].real * mult
                net.res_line_sc[f"q_{phase}_{side}_mvar"] = s_abc_mva[f:t, side_idx, phase_idx].imag * mult

            for side_idx, side in enumerate(("from", "to")):
                net.res_line_sc[f"vm_{phase}_{side}_pu"] = np.abs(v_abc_pu[f:t, side_idx, phase_idx])
                net.res_line_sc[f"va_{phase}_{side}_degree"] = np.angle(v_abc_pu[f:t, side_idx, phase_idx], deg=True)

        # todo: ip, ith
        if net._options["ip"]:
            net.res_line_sc["ip_ka"] = minmax(ppc_1["branch"][f:t, [IP_F, IP_T]].real, axis=1)
        if net._options["ith"]:
            net.res_line_sc["ith_ka"] = minmax(ppc_1["branch"][f:t, [ITH_F, ITH_T]].real, axis=1)


def _get_trafo_lg_results(net, v_abc_pu, i_abc_ka, s_abc_mva):
    branch_lookup = net._pd2ppc_lookups["branch"]
    if "trafo" in branch_lookup:
        f, t = branch_lookup["trafo"]

        for phase_idx, phase in enumerate(("a", "b", "c")):
            for side_idx, side in enumerate(("hv", "lv")):
                net.res_trafo_sc[f"ikss_{phase}_{side}_ka"] = np.abs(i_abc_ka[f:t, side_idx, phase_idx])
                net.res_trafo_sc[f"ikss_{phase}_{side}_degree"] = np.angle(i_abc_ka[f:t, side_idx, phase_idx], deg=True)

            for side_idx, side in enumerate(("hv", "lv")):
                net.res_trafo_sc[f"p_{phase}_{side}_mw"] = s_abc_mva[f:t, side_idx, phase_idx].real
                net.res_trafo_sc[f"q_{phase}_{side}_mvar"] = s_abc_mva[f:t, side_idx, phase_idx].imag

            for side_idx, side in enumerate(("hv", "lv")):
                net.res_trafo_sc[f"vm_{phase}_{side}_pu"] = np.abs(v_abc_pu[f:t, side_idx, phase_idx])
                net.res_trafo_sc[f"va_{phase}_{side}_degree"] = np.angle(v_abc_pu[f:t, side_idx, phase_idx], deg=True)

    # todo: ip, ith


def _calculate_bus_results_llg(ppc_0, ppc_1, ppc_2, bus, net):
    # we use 3D arrays here to easily identify via axis:
    # 0: line index, 1: from/to, 2: phase
    # short-ciruit for rotating machine (ext-grid and gen)
    bus_lookup = net._pd2ppc_lookups["bus"]
    ppc_index = bus_lookup[net.bus.index]
    skss_abc_mva = np.full((len(ppc_index), 3), np.nan, dtype=np.float64)
    ikss_abc_ka = np.full((len(ppc_index), 3), np.nan, dtype=np.float64)

    i_1_ka_0 = ppc_0['bus'][:, IKSSV] * np.exp(1j * np.deg2rad(ppc_0['bus'][:, PHI_IKSSV_DEGREE].real))[:, np.newaxis]
    i_1_ka_1 = ppc_1['bus'][:, IKSSV] * np.exp(1j * np.deg2rad(ppc_1['bus'][:, PHI_IKSSV_DEGREE].real))[:, np.newaxis]
    i_1_ka_2 = ppc_2['bus'][:, IKSSV] * np.exp(1j * np.deg2rad(ppc_2['bus'][:, PHI_IKSSV_DEGREE].real))[:, np.newaxis]

    # TODO check results with sgen
    # short-ciruit for inverter-based generation (current source)
    i_2_ka_0 = ppc_0['bus'][:, IKSSC] * np.exp(1j * np.deg2rad(ppc_0['bus'][:, PHI_IKSSV_DEGREE].real))[:, np.newaxis]
    i_2_ka_1 = ppc_1['bus'][:, IKSSC] * np.exp(1j * np.deg2rad(ppc_1['bus'][:, PHI_IKSSV_DEGREE].real))[:, np.newaxis]
    i_2_ka_2 = ppc_2['bus'][:, IKSSC] * np.exp(1j * np.deg2rad(ppc_2['bus'][:, PHI_IKSSV_DEGREE].real))[:, np.newaxis]

    i_1_012_ka = np.stack([i_1_ka_0, i_1_ka_1, i_1_ka_2], 2)
    i_2_012_ka = np.stack([i_2_ka_0, i_2_ka_1, i_2_ka_2], 2)

    i_1_abc_ka = np.apply_along_axis(sequence_to_phase, 2, i_1_012_ka)
    i_2_abc_ka = np.apply_along_axis(sequence_to_phase, 2, i_2_012_ka)

    # i_abc_ka = sequence_to_phase(np.vstack([i_ka_0, i_ka_1, i_ka_2]))
    i_1_abc_ka[np.abs(i_1_abc_ka) < 1e-5] = 0
    i_2_abc_ka[np.abs(i_2_abc_ka) < 1e-5] = 0

    # ToDo: check if this works without sgen
    i_total_abc_ka = i_1_abc_ka + i_2_abc_ka

    # Todo adapt to new reult format
    # Initialize a new matrix to store the selected rows
    # The shape is determined by the length of 'bus' and the number of columns in 'i_1_abc_ka'
    i_total_abc_ka_abs = np.zeros((len(bus), i_total_abc_ka.shape[2]))

    # Extract the specified rows from 'i_1_abc_ka' based on the indices in 'bus'
    # for index in range(len(bus)):
    #     i_total_abc_ka_abs[index] = abs(i_total_abc_ka[bus[index], bus[index]])
    ppc_indices = [ppc_index[b] for b in bus]
    i_total_abc_ka_abs = abs(i_total_abc_ka[ppc_indices, ppc_indices])

    # ToDo: check voltages
    v_pu_0 = ppc_0["internal"]["V_ikss"][bus][:, np.newaxis]
    v_pu_1 = ppc_1["internal"]["V_ikss"][bus][:, np.newaxis]
    v_pu_2 = ppc_2["internal"]["V_ikss"][bus][:, np.newaxis]

    v_012_pu = np.stack([v_pu_0, v_pu_1, v_pu_2], 2)
    v_abc_pu = np.apply_along_axis(sequence_to_phase, 2, v_012_pu)
    # v_abc_pu = sequence_to_phase(np.vstack([v_pu_0, v_pu_1, v_pu_2]))
    v_abc_pu[np.abs(v_abc_pu) < 1e-5] = 0

    # this is inefficient because it copies data to fit into a shape, better to use a slice,
    # and even better to find how to use sequence-based powers:
    baseV = ppc_1['bus'][bus, BASE_KV][:, np.newaxis]
    # baseV = ppc_1["internal"]["baseV"][bus][:, np.newaxis]
    # v_base_kv = np.stack([baseV, baseV, baseV], 2)

    skss_abc_mva_phase = i_total_abc_ka_abs * baseV / np.sqrt(3)
    skss_abc_mva[np.ix_(bus, [0, 1, 2, ])] = skss_abc_mva_phase
    ikss_abc_ka[np.ix_(bus, [0, 1, 2, ])] = i_total_abc_ka_abs

    # Adding the ikss and skss values
    for i, phase in enumerate(['a', 'b', 'c']):
        net.res_bus_sc[f'ikss_{phase}_ka'] = ikss_abc_ka[:, i]  # ikss values
        net.res_bus_sc[f'skss_{phase}_mw'] = skss_abc_mva[:, i]  # skss values


def _extract_results(net, ppc_0, ppc_1, ppc_2, bus):
    if net["_options"]["fault"] == "LLG":
        _calculate_bus_results_llg(ppc_0, ppc_1, ppc_2, bus, net)
    _get_bus_results(net, ppc_0, ppc_1, ppc_2, bus)
    _get_gen_results(net, ppc_0, ppc_1, ppc_2)
    if net._options["branch_results"]:
        # TODO check option return all current here
        if (~net["_options"]['return_all_currents']):
            v_abc_pu, i_abc_ka, s_abc_mva = _calculate_branch_phase_results(ppc_0, ppc_1, ppc_2)
            _get_line_branch_results(net, ppc_1, v_abc_pu, i_abc_ka, s_abc_mva)
            # TODO might need to be adapted
            _get_trafo_lg_results(net, v_abc_pu, i_abc_ka, s_abc_mva)
        # elif (net["_options"]["fault"] in ("LL")) & (~net["_options"]['return_all_currents']):
        #     _get_line_ll_results(net, ppc_1)
        #     #TODO
        #     # _get_trafo_ll_results(net, ppc_1)
        else:
            # if net._options['return_all_currents']:
            _get_line_all_results(net, ppc_1, bus)
            _get_trafo_all_results(net, ppc_1, bus)
            _get_trafo3w_all_results(net, ppc_1, bus)
            _get_switch_all_results(net, ppc_1, bus)
            # else:
            #     _get_line_results(net, ppc_1)
            #     _get_trafo_results(net, ppc_1)
            #     _get_trafo3w_results(net, ppc_1)
            #     _get_switch_results(net, ppc_1)


# def _get_line_ll_results(net, ppc):
#     branch_lookup = net._pd2ppc_lookups["branch"]
#     case = net._options["case"]
#     if "line" in branch_lookup:
#         f, t = branch_lookup["line"]
#         minmax = np.max if case == "max" else np.min
#         net.res_line_sc["ikss_ka"] = minmax(ppc["branch"][f:t, [IKSS_F, IKSS_T]].real, axis=1)
#         for phase, name in ("b","branch"),("c","branch_LL"):
#             net.res_line_sc[f"ikss_{phase}_from_ka"] = ppc[name][f:t, IKSS_F].real
#             net.res_line_sc[f"ikss_{phase}_from_degree"] = ppc[name][f:t, IKSS_ANGLE_F].real
#             net.res_line_sc[f"ikss_{phase}_to_ka"] = ppc[name][f:t, IKSS_T].real
#             net.res_line_sc[f"ikss_{phase}_to_degree"] = ppc[name][f:t, IKSS_ANGLE_T].real

#             # adding columns for new calculated VPQ
#             net.res_line_sc[f"p_{phase}_from_mw"] = ppc[name][f:t, PKSS_F].real
#             net.res_line_sc[f"q_{phase}_from_mvar"] = ppc[name][f:t, QKSS_F].real

#             net.res_line_sc[f"p_{phase}_to_mw"] = ppc[name][f:t, PKSS_T].real
#             net.res_line_sc[f"q_{phase}_to_mvar"] = ppc[name][f:t, QKSS_T].real

#             net.res_line_sc[f"vm_{phase}_from_pu"] = ppc[name][f:t, VKSS_MAGN_F].real
#             net.res_line_sc[f"va_{phase}_from_degree"] = ppc[name][f:t, VKSS_ANGLE_F].real

#             net.res_line_sc[f"vm_{phase}_to_pu"] = ppc[name][f:t, VKSS_MAGN_T].real
#             net.res_line_sc[f"va_{phase}_to_degree"] = ppc[name][f:t, VKSS_ANGLE_T].real

#         if net._options["ip"]:
#             net.res_line_sc["ip_ka"] = minmax(ppc["branch"][f:t, [IP_F, IP_T]].real, axis=1)
#         if net._options["ith"]:
#             net.res_line_sc["ith_ka"] = minmax(ppc["branch"][f:t, [ITH_F, ITH_T]].real, axis=1)

def _get_switch_results(net, ppc):
    if len(net.switch) == 0:
        return
    branch_lookup = net._pd2ppc_lookups["branch"]
    case = net._options["case"]
    if "switch" in branch_lookup:
        f, t = branch_lookup["switch"]
        minmax = np.max if case == "max" else np.min
        bb_switches = net._impedance_bb_switches
        net.res_switch_sc.loc[bb_switches, "ikss_ka"] = minmax(ppc["branch"][f:t, [IKSS_F, IKSS_T]].real, axis=1)
        if net._options["ip"]:
            net.res_switch_sc.loc[bb_switches, "ip_ka"] = minmax(ppc["branch"][f:t, [IP_F, IP_T]].real, axis=1)
        if net._options["ith"]:
            net.res_switch_sc.loc[bb_switches, "ith_ka"] = minmax(ppc["branch"][f:t, [ITH_F, ITH_T]].real, axis=1)
    _copy_switch_results_from_branches(net, suffix="_sc", current_parameter="ikss_ka")
    if "in_ka" in net.switch.columns:
        net.res_switch_sc["loading_percent"] = net.res_switch_sc["ikss_ka"].values / net.switch["in_ka"].values * 100


def _get_branch_result_from_internal(variable, ppc, ppc_index, f, t):
    if variable in ppc["internal"]:
        return ppc["internal"][variable].iloc[f:t, :].loc[:, ppc_index].values.real.reshape(-1, 1)
    else:
        return np.full((t - f) * len(ppc_index), np.nan, dtype=np.float64)


def _get_line_all_results(net, ppc, bus):
    case = net._options["case"]

    bus, ppc_index = _get_bus_ppc_idx_for_br_all_results(net, ppc, bus)
    branch_lookup = net._pd2ppc_lookups["branch"]

    multindex = pd.MultiIndex.from_product([net.res_line_sc.index, bus], names=['line', 'bus'])
    net.res_line_sc = net.res_line_sc.reindex(multindex)

    if "line" in branch_lookup:
        f, t = branch_lookup["line"]
        minmax = np.maximum if case == "max" else np.minimum

        net.res_line_sc["ikss_ka"] = minmax(
            ppc["internal"]["branch_ikss_f"].iloc[f:t, :].loc[:, ppc_index].values.real.reshape(-1, 1),
            ppc["internal"]["branch_ikss_t"].iloc[f:t, :].loc[:, ppc_index].values.real.reshape(-1, 1))

        net.res_line_sc["ikss_from_ka"] = _get_branch_result_from_internal("branch_ikss_f", ppc, ppc_index, f, t)
        net.res_line_sc["ikss_from_degree"] = _get_branch_result_from_internal("branch_ikss_angle_f", ppc, ppc_index, f,
                                                                               t)

        net.res_line_sc["ikss_to_ka"] = _get_branch_result_from_internal("branch_ikss_t", ppc, ppc_index, f, t)
        net.res_line_sc["ikss_to_degree"] = _get_branch_result_from_internal("branch_ikss_angle_t", ppc, ppc_index, f,
                                                                             t)

        net.res_line_sc["p_from_mw"] = _get_branch_result_from_internal("branch_pkss_f", ppc, ppc_index, f, t)
        net.res_line_sc["q_from_mvar"] = _get_branch_result_from_internal("branch_qkss_f", ppc, ppc_index, f, t)

        net.res_line_sc["p_to_mw"] = _get_branch_result_from_internal("branch_pkss_t", ppc, ppc_index, f, t)
        net.res_line_sc["q_to_mvar"] = _get_branch_result_from_internal("branch_qkss_t", ppc, ppc_index, f, t)

        net.res_line_sc["vm_from_pu"] = _get_branch_result_from_internal("branch_vkss_f", ppc, ppc_index, f, t)
        net.res_line_sc["va_from_degree"] = _get_branch_result_from_internal("branch_vkss_angle_f", ppc, ppc_index, f,
                                                                             t)

        net.res_line_sc["vm_to_pu"] = _get_branch_result_from_internal("branch_vkss_t", ppc, ppc_index, f, t)
        net.res_line_sc["va_to_degree"] = _get_branch_result_from_internal("branch_vkss_angle_t", ppc, ppc_index, f, t)

        if net._options["ip"]:
            net.res_line_sc["ip_ka"] = minmax(
                ppc["internal"]["branch_ip_f"].iloc[f:t, :].loc[:, ppc_index].values.real.reshape(-1, 1),
                ppc["internal"]["branch_ip_t"].iloc[f:t, :].loc[:, ppc_index].values.real.reshape(-1, 1))
        if net._options["ith"]:
            net.res_line_sc["ith_ka"] = minmax(
                ppc["internal"]["branch_ith_f"].iloc[f:t, :].loc[:, ppc_index].values.real.reshape(-1, 1),
                ppc["internal"]["branch_ith_t"].iloc[f:t, :].loc[:, ppc_index].values.real.reshape(-1, 1))


def _get_switch_all_results(net, ppc, bus):
    case = net._options["case"]

    bus, ppc_index = _get_bus_ppc_idx_for_br_all_results(net, ppc, bus)
    branch_lookup = net._pd2ppc_lookups["branch"]

    multindex = pd.MultiIndex.from_product([net.res_switch_sc.index, bus], names=['switch', 'bus'])
    net.res_switch_sc = net.res_switch_sc.reindex(multindex)

    if "switch" in branch_lookup:
        f, t = branch_lookup["switch"]
        minmax = np.maximum if case == "max" else np.minimum

        net.res_switch_sc["ikss_ka"] = minmax(
            ppc["internal"]["branch_ikss_f"].iloc[f:t, :].loc[:, ppc_index].values.real.reshape(-1, 1),
            ppc["internal"]["branch_ikss_t"].iloc[f:t, :].loc[:, ppc_index].values.real.reshape(-1, 1))
        if net._options["ip"]:
            net.res_switch_sc["ip_ka"] = minmax(
                ppc["internal"]["branch_ip_f"].iloc[f:t, :].loc[:, ppc_index].values.real.reshape(-1, 1),
                ppc["internal"]["branch_ip_t"].iloc[f:t, :].loc[:, ppc_index].values.real.reshape(-1, 1))
        if net._options["ith"]:
            net.res_switch_sc["ith_ka"] = minmax(
                ppc["internal"]["branch_ith_f"].iloc[f:t, :].loc[:, ppc_index].values.real.reshape(-1, 1),
                ppc["internal"]["branch_ith_t"].iloc[f:t, :].loc[:, ppc_index].values.real.reshape(-1, 1))


def _get_trafo_results(net, ppc):
    branch_lookup = net._pd2ppc_lookups["branch"]
    if "trafo" in branch_lookup:
        f, t = branch_lookup["trafo"]
        net.res_trafo_sc["ikss_hv_ka"] = ppc["branch"][f:t, IKSS_F].real
        net.res_trafo_sc["ikss_hv_degree"] = ppc["branch"][f:t, IKSS_ANGLE_F].real
        net.res_trafo_sc["ikss_lv_ka"] = ppc["branch"][f:t, IKSS_T].real
        net.res_trafo_sc["ikss_lv_degree"] = ppc["branch"][f:t, IKSS_ANGLE_T].real

        # adding columns for new calculated VPQ
        net.res_trafo_sc["p_hv_mw"] = ppc["branch"][f:t, PKSS_F].real
        net.res_trafo_sc["q_hv_mvar"] = ppc["branch"][f:t, QKSS_F].real

        net.res_trafo_sc["p_lv_mw"] = ppc["branch"][f:t, PKSS_T].real
        net.res_trafo_sc["q_lv_mvar"] = ppc["branch"][f:t, QKSS_T].real

        net.res_trafo_sc["vm_hv_pu"] = ppc["branch"][f:t, VKSS_MAGN_F].real
        net.res_trafo_sc["va_hv_degree"] = ppc["branch"][f:t, VKSS_ANGLE_F].real

        net.res_trafo_sc["vm_lv_pu"] = ppc["branch"][f:t, VKSS_MAGN_T].real
        net.res_trafo_sc["va_lv_degree"] = ppc["branch"][f:t, VKSS_ANGLE_T].real


def _get_trafo_all_results(net, ppc, bus):
    bus, ppc_index = _get_bus_ppc_idx_for_br_all_results(net, ppc, bus)
    branch_lookup = net._pd2ppc_lookups["branch"]

    multindex = pd.MultiIndex.from_product([net.res_trafo_sc.index, bus], names=['trafo', 'bus'])
    net.res_trafo_sc = net.res_trafo_sc.reindex(multindex)

    if "trafo" in branch_lookup:
        f, t = branch_lookup["trafo"]
        net.res_trafo_sc["ikss_hv_ka"] = ppc["internal"]["branch_ikss_f"].iloc[f:t, :].loc[
            :, ppc_index].values.real.reshape(-1, 1)
        net.res_trafo_sc["ikss_lv_ka"] = ppc["internal"]["branch_ikss_t"].iloc[f:t, :].loc[
            :, ppc_index].values.real.reshape(-1, 1)


def _get_trafo3w_results(net, ppc):
    branch_lookup = net._pd2ppc_lookups["branch"]
    if "trafo3w" in branch_lookup:
        f, t = net._pd2ppc_lookups["branch"]["trafo3w"]
        hv = int(f + (t - f) / 3)
        mv = int(f + 2 * (t - f) / 3)
        lv = t
        net.res_trafo3w_sc["ikss_hv_ka"] = ppc["branch"][f:hv, IKSS_F].real
        net.res_trafo3w_sc["ikss_mv_ka"] = ppc["branch"][hv:mv, IKSS_T].real
        net.res_trafo3w_sc["ikss_lv_ka"] = ppc["branch"][mv:lv, IKSS_T].real


def _get_trafo3w_all_results(net, ppc, bus):
    bus, ppc_index = _get_bus_ppc_idx_for_br_all_results(net, ppc, bus)
    branch_lookup = net._pd2ppc_lookups["branch"]

    multindex = pd.MultiIndex.from_product([net.res_trafo3w_sc.index, bus], names=['trafo3w', 'bus'])
    net.res_trafo3w_sc = net.res_trafo3w_sc.reindex(multindex)

    if "trafo3w" in branch_lookup:
        f, t = branch_lookup["trafo3w"]
        hv = int(f + (t - f) / 3)
        mv = int(f + 2 * (t - f) / 3)
        lv = t
        net.res_trafo3w_sc["ikss_hv_ka"] = ppc["internal"]["branch_ikss_f"].iloc[f:hv, :].loc[
            :, ppc_index].values.real.reshape(-1, 1)
        net.res_trafo3w_sc["ikss_mv_ka"] = ppc["internal"]["branch_ikss_t"].iloc[hv:mv, :].loc[
            :, ppc_index].values.real.reshape(-1, 1)
        net.res_trafo3w_sc["ikss_lv_ka"] = ppc["internal"]["branch_ikss_t"].iloc[mv:lv, :].loc[
            :, ppc_index].values.real.reshape(-1, 1)
