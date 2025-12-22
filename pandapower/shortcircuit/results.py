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
import logging
logger = logging.getLogger(__name__)


def _copy_result_to_ppci_orig(ppci_orig, ppci, ppci_bus, calc_options):
    if ppci_orig is ppci:
        return

    ppci_orig["bus"][ppci_bus, :] = ppci["bus"][ppci_bus, :]
    if calc_options["branch_results"]:
        if calc_options["return_all_currents"]:
            ppci_orig["internal"]["br_res_ks_ppci_bus"] =\
                ppci_bus if "br_res_ks_ppci_bus" not in ppci_orig["internal"]\
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
                ppci_orig["branch"][:, branch_results_cols] =\
                    np.maximum(np.nan_to_num(ppci["branch"][:, branch_results_cols]),
                               np.nan_to_num(ppci_orig["branch"][:, branch_results_cols]))

                # excluding new values from nan to num
                ppci_orig["branch"][:, branch_results_cols_add] = ppci["branch"][:, branch_results_cols_add]
                ppci_orig["branch"][:, branch_results_cols_add] = ppci["branch"][:, branch_results_cols_add]

                if "branch_LL" in ppci.keys():
                    ppci_orig["branch_LL"] = deepcopy(ppci_orig["branch"])
                    ppci_orig["branch_LL"][:, branch_results_cols] =\
                        np.maximum(np.nan_to_num(ppci["branch_LL"][:, branch_results_cols]),
                                np.nan_to_num(ppci_orig["branch_LL"][:, branch_results_cols]))
                    ppci_orig["branch_LL"][:, branch_results_cols_add] = ppci["branch_LL"][:, branch_results_cols_add]

            else:
                ppci_orig["branch"][:, branch_results_cols] =\
                    np.minimum(np.nan_to_num(ppci["branch"][:, branch_results_cols], nan=1e10),
                               np.nan_to_num(ppci_orig["branch"][:, branch_results_cols], nan=1e10))
                # excluding new values from nan to num
                ppci_orig["branch"][:, branch_results_cols_add] = ppci["branch"][:, branch_results_cols_add]
                if "branch_LL" in ppci.keys(): 
                    ppci_orig["branch_LL"] = deepcopy(ppci_orig["branch"])
                    ppci_orig["branch_LL"][:, branch_results_cols] =\
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
    i_ka_0 = ppc_0['branch'][:, [IKSS_F, IKSS_T]] * np.exp(1j * np.deg2rad(ppc_0['branch'][:, [IKSS_ANGLE_F, IKSS_ANGLE_T]].real))
    i_ka_1 = ppc_1['branch'][:, [IKSS_F, IKSS_T]] * np.exp(1j * np.deg2rad(ppc_1['branch'][:, [IKSS_ANGLE_F, IKSS_ANGLE_T]].real))
    i_ka_2 = ppc_2['branch'][:, [IKSS_F, IKSS_T]] * np.exp(1j * np.deg2rad(ppc_2['branch'][:, [IKSS_ANGLE_F, IKSS_ANGLE_T]].real))

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

    v_pu_0 = ppc_0['branch'][:, [VKSS_MAGN_F, VKSS_MAGN_T]] * np.exp(1j * np.deg2rad(ppc_0['branch'][:, [VKSS_ANGLE_F, VKSS_ANGLE_T]].real))
    v_pu_1 = ppc_1['branch'][:, [VKSS_MAGN_F, VKSS_MAGN_T]] * np.exp(1j * np.deg2rad(ppc_1['branch'][:, [VKSS_ANGLE_F, VKSS_ANGLE_T]].real))
    v_pu_2 = ppc_2['branch'][:, [VKSS_MAGN_F, VKSS_MAGN_T]] * np.exp(1j * np.deg2rad(ppc_2['branch'][:, [VKSS_ANGLE_F, VKSS_ANGLE_T]].real))

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


# def _get_line_to_g_results(net, ppc_1, v_abc_pu, i_abc_ka, s_abc_mva):
def _get_line_branch_results(net, ppc_1, v_abc_pu, i_abc_ka, s_abc_mva): #renamaed the function properly
    branch_lookup = net._pd2ppc_lookups["branch"]
    case = net._options["case"]
    fault = net._options["fault"]

    if "line" in branch_lookup:
        f, t = branch_lookup["line"]
        minmax = np.max if case == "max" else np.min

        # todo: check axis of max with more lines in the grid
        i_max_per_line_ka = np.max(np.abs(i_abc_ka), axis=1)
        net.res_line_sc["ikss_ka"] = minmax(i_max_per_line_ka[f:t, :], axis=1)
        
        if fault == 'LLL': #corrected bramch p amd q values
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
    skss_abc_mva[np.ix_(bus, [0,1,2,])] = skss_abc_mva_phase
    ikss_abc_ka[np.ix_(bus, [0,1,2,])] = i_total_abc_ka_abs

    # Adding the ikss and skss values
    for i, phase in enumerate(['a', 'b', 'c']):
        net.res_bus_sc[f'ikss_{phase}_ka'] = ikss_abc_ka[:, i]  # ikss values
        net.res_bus_sc[f'skss_{phase}_mw'] = skss_abc_mva[:, i]  # skss values


def _extract_results(net, ppc_0, ppc_1, ppc_2, bus):
    if net["_options"]["fault"] == "LLG":
       _calculate_bus_results_llg(ppc_0, ppc_1, ppc_2, bus, net)
    _get_bus_results(net, ppc_0, ppc_1, ppc_2, bus)
    if net._options["branch_results"]:
        # TODO check option return all current here
        if (~net["_options"]['return_all_currents']):
            v_abc_pu, i_abc_ka, s_abc_mva = _calculate_branch_phase_results(ppc_0, ppc_1, ppc_2)
            _get_line_branch_results(net, ppc_1, v_abc_pu, i_abc_ka, s_abc_mva)
            #TODO might need to be adapted
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
    if ("grounding_type" in net.trafo.columns) and (net.trafo["grounding_type"] == "resonant").any():
        _validate_resonate_grounding(net, ppc_0, bus)



def _get_bus_results(net, ppc_0, ppc_1, ppc_2, bus):
    bus_lookup = net._pd2ppc_lookups["bus"]
    ppc_index = bus_lookup[net.bus.index]

    ppc_sequence = {0: ppc_0, 1: ppc_1, 2: ppc_2, "": ppc_1}
    if net["_options"]["fault"] == "LG":
        net.res_bus_sc["ikss_ka"] = ppc_0["bus"][ppc_index, IKSSV] + ppc_1["bus"][ppc_index, IKSSC]
        net.res_bus_sc["skss_mw"] = ppc_0["bus"][ppc_index, SKSS]
        sequence_relevant = range(3)
    elif net["_options"]["fault"] == "LLG":
        sequence_relevant = range(3)
    elif net["_options"]["fault"] == "LL":
        net.res_bus_sc["ikss_ka"] = ppc_1["bus"][ppc_index, IKSSV] + ppc_1["bus"][ppc_index, IKSSC]
        net.res_bus_sc["skss_mw"] = ppc_1["bus"][ppc_index, SKSS]
        sequence_relevant = range(3)
    else:
        net.res_bus_sc["ikss_ka"] = ppc_1["bus"][ppc_index, IKSSV] + ppc_1["bus"][ppc_index, IKSSC]
        net.res_bus_sc["skss_mw"] = ppc_1["bus"][ppc_index, SKSS]
        sequence_relevant = ("",)
    for sequence in sequence_relevant:
        ppc_s = ppc_sequence[sequence]
        if net["_options"]["fault"] == "LL":
            # TODO: ask Marco where this can be done before results are written into tables
            if sequence in [1, 2]:
                fault_ohm_factor = 2
            elif sequence == 0:
                fault_ohm_factor = 1
            net.res_bus_sc[f"rk{sequence}_ohm"] = (ppc_s["bus"][ppc_index, R_EQUIV_OHM] +
                                                   net["_options"]["r_fault_ohm"]/fault_ohm_factor)
            net.res_bus_sc[f"xk{sequence}_ohm"] = (ppc_s["bus"][ppc_index, X_EQUIV_OHM] +
                                                   net["_options"]["x_fault_ohm"]/fault_ohm_factor)
        else:
            net.res_bus_sc[f"rk{sequence}_ohm"] = ppc_s["bus"][ppc_index, R_EQUIV_OHM]
            net.res_bus_sc[f"xk{sequence}_ohm"] = ppc_s["bus"][ppc_index, X_EQUIV_OHM]
        # in trafo3w, we add very high numbers (1e10) as impedances to block current
        # here, we need to replace such high values by np.inf
        baseZ = ppc_s["bus"][ppc_index, BASE_KV] ** 2 / ppc_s["baseMVA"]
        net.res_bus_sc.loc[net.res_bus_sc[f"xk{sequence}_ohm"] / baseZ > 1e9, f"xk{sequence}_ohm"] = np.inf
        net.res_bus_sc.loc[net.res_bus_sc[f"rk{sequence}_ohm"] / baseZ > 1e9, f"rk{sequence}_ohm"] = np.inf
    if net._options["ip"]:
        net.res_bus_sc["ip_ka"] = ppc_1["bus"][ppc_index, IP]
    if net._options["ith"]:
        net.res_bus_sc["ith_ka"] = ppc_1["bus"][ppc_index, ITH]

    net.res_bus_sc = net.res_bus_sc.loc[bus, :]


def _get_line_results(net, ppc):
    branch_lookup = net._pd2ppc_lookups["branch"]
    case = net._options["case"]
    if "line" in branch_lookup:
        f, t = branch_lookup["line"]
        minmax = np.max if case == "max" else np.min
        net.res_line_sc["ikss_ka"] = minmax(ppc["branch"][f:t, [IKSS_F, IKSS_T]].real, axis=1)
        net.res_line_sc["ikss_from_ka"] = ppc["branch"][f:t, IKSS_F].real
        net.res_line_sc["ikss_from_degree"] = ppc["branch"][f:t, IKSS_ANGLE_F].real
        net.res_line_sc["ikss_to_ka"] = ppc["branch"][f:t, IKSS_T].real
        net.res_line_sc["ikss_to_degree"] = ppc["branch"][f:t, IKSS_ANGLE_T].real

        # adding columns for new calculated VPQ
        net.res_line_sc["p_from_mw"] = ppc["branch"][f:t, PKSS_F].real
        net.res_line_sc["q_from_mvar"] = ppc["branch"][f:t, QKSS_F].real

        net.res_line_sc["p_to_mw"] = ppc["branch"][f:t, PKSS_T].real
        net.res_line_sc["q_to_mvar"] = ppc["branch"][f:t, QKSS_T].real

        net.res_line_sc["vm_from_pu"] = ppc["branch"][f:t, VKSS_MAGN_F].real
        net.res_line_sc["va_from_degree"] = ppc["branch"][f:t, VKSS_ANGLE_F].real

        net.res_line_sc["vm_to_pu"] = ppc["branch"][f:t, VKSS_MAGN_T].real
        net.res_line_sc["va_to_degree"] = ppc["branch"][f:t, VKSS_ANGLE_T].real

        if net._options["ip"]:
            net.res_line_sc["ip_ka"] = minmax(ppc["branch"][f:t, [IP_F, IP_T]].real, axis=1)
        if net._options["ith"]:
            net.res_line_sc["ith_ka"] = minmax(ppc["branch"][f:t, [ITH_F, ITH_T]].real, axis=1)

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
        return np.full((t-f) * len(ppc_index), np.nan, dtype=np.float64)


def _get_line_all_results(net, ppc, bus):
    case = net._options["case"]

    bus, ppc_index = _get_bus_ppc_idx_for_br_all_results(net, ppc, bus)
    branch_lookup = net._pd2ppc_lookups["branch"]

    multindex = pd.MultiIndex.from_product([net.res_line_sc.index, bus], names=['line','bus'])
    net.res_line_sc = net.res_line_sc.reindex(multindex)

    if "line" in branch_lookup:
        f, t = branch_lookup["line"]
        minmax = np.maximum if case == "max" else np.minimum

        net.res_line_sc["ikss_ka"] = minmax(ppc["internal"]["branch_ikss_f"].iloc[f:t,:].loc[:, ppc_index].values.real.reshape(-1, 1),
                                            ppc["internal"]["branch_ikss_t"].iloc[f:t,:].loc[:, ppc_index].values.real.reshape(-1, 1))

        net.res_line_sc["ikss_from_ka"] = _get_branch_result_from_internal("branch_ikss_f", ppc, ppc_index, f, t)
        net.res_line_sc["ikss_from_degree"] = _get_branch_result_from_internal("branch_ikss_angle_f", ppc, ppc_index, f, t)

        net.res_line_sc["ikss_to_ka"] = _get_branch_result_from_internal("branch_ikss_t", ppc, ppc_index, f, t)
        net.res_line_sc["ikss_to_degree"] = _get_branch_result_from_internal("branch_ikss_angle_t", ppc, ppc_index, f, t)

        net.res_line_sc["p_from_mw"] = _get_branch_result_from_internal("branch_pkss_f", ppc, ppc_index, f, t)
        net.res_line_sc["q_from_mvar"] = _get_branch_result_from_internal("branch_qkss_f", ppc, ppc_index, f, t)

        net.res_line_sc["p_to_mw"] = _get_branch_result_from_internal("branch_pkss_t", ppc, ppc_index, f, t)
        net.res_line_sc["q_to_mvar"] = _get_branch_result_from_internal("branch_qkss_t", ppc, ppc_index, f, t)

        net.res_line_sc["vm_from_pu"] = _get_branch_result_from_internal("branch_vkss_f", ppc, ppc_index, f, t)
        net.res_line_sc["va_from_degree"] = _get_branch_result_from_internal("branch_vkss_angle_f", ppc, ppc_index, f, t)

        net.res_line_sc["vm_to_pu"] = _get_branch_result_from_internal("branch_vkss_t", ppc, ppc_index, f, t)
        net.res_line_sc["va_to_degree"] = _get_branch_result_from_internal("branch_vkss_angle_t", ppc, ppc_index, f, t)

        if net._options["ip"]:
            net.res_line_sc["ip_ka"] = minmax(ppc["internal"]["branch_ip_f"].iloc[f:t,:].loc[:, ppc_index].values.real.reshape(-1, 1),
                                              ppc["internal"]["branch_ip_t"].iloc[f:t,:].loc[:, ppc_index].values.real.reshape(-1, 1))
        if net._options["ith"]:
            net.res_line_sc["ith_ka"] = minmax(ppc["internal"]["branch_ith_f"].iloc[f:t,:].loc[:, ppc_index].values.real.reshape(-1, 1),
                                               ppc["internal"]["branch_ith_t"].iloc[f:t,:].loc[:, ppc_index].values.real.reshape(-1, 1))

def _get_switch_all_results(net, ppc, bus):
    case = net._options["case"]

    bus, ppc_index = _get_bus_ppc_idx_for_br_all_results(net, ppc, bus)
    branch_lookup = net._pd2ppc_lookups["branch"]

    multindex = pd.MultiIndex.from_product([net.res_switch_sc.index, bus], names=['switch','bus'])
    net.res_switch_sc = net.res_switch_sc.reindex(multindex)

    if "switch" in branch_lookup:
        f, t = branch_lookup["switch"]
        minmax = np.maximum if case == "max" else np.minimum

        net.res_switch_sc["ikss_ka"] = minmax(ppc["internal"]["branch_ikss_f"].iloc[f:t,:].loc[:, ppc_index].values.real.reshape(-1, 1),
                                            ppc["internal"]["branch_ikss_t"].iloc[f:t,:].loc[:, ppc_index].values.real.reshape(-1, 1))
        if net._options["ip"]:
            net.res_switch_sc["ip_ka"] = minmax(ppc["internal"]["branch_ip_f"].iloc[f:t,:].loc[:, ppc_index].values.real.reshape(-1, 1),
                                              ppc["internal"]["branch_ip_t"].iloc[f:t,:].loc[:, ppc_index].values.real.reshape(-1, 1))
        if net._options["ith"]:
            net.res_switch_sc["ith_ka"] = minmax(ppc["internal"]["branch_ith_f"].iloc[f:t,:].loc[:, ppc_index].values.real.reshape(-1, 1),
                                               ppc["internal"]["branch_ith_t"].iloc[f:t,:].loc[:, ppc_index].values.real.reshape(-1, 1))
    
    
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
        net.res_trafo_sc["ikss_hv_ka"] = ppc["internal"]["branch_ikss_f"].iloc[f:t,:].loc[:, ppc_index].values.real.reshape(-1, 1)
        net.res_trafo_sc["ikss_lv_ka"] = ppc["internal"]["branch_ikss_t"].iloc[f:t,:].loc[:, ppc_index].values.real.reshape(-1, 1)


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
        net.res_trafo3w_sc["ikss_hv_ka"] = ppc["internal"]["branch_ikss_f"].iloc[f:hv,:].loc[:, ppc_index].values.real.reshape(-1, 1)
        net.res_trafo3w_sc["ikss_mv_ka"] = ppc["internal"]["branch_ikss_t"].iloc[hv:mv, :].loc[:, ppc_index].values.real.reshape(-1, 1)
        net.res_trafo3w_sc["ikss_lv_ka"] = ppc["internal"]["branch_ikss_t"].iloc[mv:lv, :].loc[:, ppc_index].values.real.reshape(-1, 1)

def _validate_resonate_grounding(net, ppc_0, bus):
    bus_lookup = net._pd2ppc_lookups["bus"]
    ppc_index = bus_lookup[net.bus.index]
    net.res_bus_sc["3xI0"] = (ppc_0["bus"][ppc_index, IKSSV] + ppc_0["bus"][ppc_index, IKSSC])[bus]
    voltage_level = net.bus.loc[bus, 'vn_kv']
    if voltage_level.nunique() == 1:
        voltage_level = voltage_level.iloc[0]
    else:
        raise ValueError("Different voltage levels encountered for resonant grounding,"
                         " please check the location of fault buses")
    # voltage in kV, maximum current in neutral/earth conenction
    thresholds = {
        "high": (110.0, 0.130),
        "low": (20.0, 0.060)
    }
    for key, (voltage, current) in thresholds.items():
        if (key == "high" and voltage_level >= voltage) or (key == "low" and voltage_level <= voltage):
            if (net.res_bus_sc.loc[bus, '3xI0'] > current).any():
                logger.warning(f"\nWarning: Current 3xI0 exceeds threshold of {current} A at {voltage} kV level."
                               f" Please check the parameters of your grounding impedance {net.name}")