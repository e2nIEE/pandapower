# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
import pandas as pd
from copy import deepcopy

from pandapower.auxiliary import sequence_to_phase
from pandapower.pypower.idx_bus import VM, VA, BUS_TYPE, BASE_KV, GS, BS
from pandapower.pypower.idx_brch import F_BUS, T_BUS
from pandapower.pypower.idx_brch_sc import IKSS_F, IKSS_T, IP_F, IP_T, ITH_F, ITH_T, PKSS_F, QKSS_F, PKSS_T, QKSS_T, \
    VKSS_MAGN_F, VKSS_MAGN_T, VKSS_ANGLE_F, VKSS_ANGLE_T, IKSS_ANGLE_F, IKSS_ANGLE_T
from pandapower.pypower.idx_bus_sc import IKSSV, IP, ITH, IKSSC, R_EQUIV_OHM, X_EQUIV_OHM, SKSS, PHI_IKSSV_DEGREE, \
    PHI_IKSSC_DEGREE, IKCV, PHI_IKCV_DEGREE, C_MAX, C_MIN, GS_GEN, BS_GEN
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


def _set_buses_and_branches_out_of_service(ppc):
    disco = np.where(ppc["bus"][:, BUS_TYPE] == 4)[0]
    ppc["bus"][disco, VM] = np.nan
    ppc["bus"][disco, VA] = np.nan
    ppc["bus"][disco, IKSSV] = np.nan
    ppc["bus"][disco, IKSSC] = np.nan
    ppc["bus"][disco, IKCV] = np.nan

    fb = ppc["branch"][:,0]
    tb = ppc["branch"][:,1]
    foos = [fb[i] in disco for i in range(len(fb))]
    toos = [tb[i] in disco for i in range(len(tb))]
    oos = np.logical_or(foos, toos)
    ppc["branch"][oos, IKSS_F] = np.nan
    ppc["branch"][oos, IKSS_T] = np.nan
    ppc["branch"][oos, IKSS_ANGLE_F] = np.nan
    ppc["branch"][oos, IKSS_ANGLE_T] = np.nan
    ppc["branch"][oos, VKSS_MAGN_F] = np.nan
    ppc["branch"][oos, VKSS_MAGN_T] = np.nan
    ppc["branch"][oos, VKSS_ANGLE_F] = np.nan
    ppc["branch"][oos, VKSS_ANGLE_T] = np.nan



def _get_bus_ppc_idx_for_br_all_results(net, ppc, bus):
    bus_lookup = net._pd2ppc_lookups["bus"]
    if bus is None:
        bus = net.bus.index

    ppc_index = bus_lookup[bus]
    ppc_index[ppc["bus"][ppc_index, BUS_TYPE] == 4] = -1
    return bus, ppc_index


def _compute_x_abc(ppc_0, ppc_1, ppc_2, el_idx, el, idx1, idx2):
    x_pu_0 = ppc_0[el][el_idx, idx1] * np.exp(1j * np.deg2rad(ppc_0[el][el_idx, idx2].real))
    x_pu_1 = ppc_1[el][el_idx, idx1] * np.exp(1j * np.deg2rad(ppc_1[el][el_idx, idx2].real))
    x_pu_2 = ppc_2[el][el_idx, idx1] * np.exp(1j * np.deg2rad(ppc_2[el][el_idx, idx2].real))

    x_012_pu = np.stack([x_pu_0, x_pu_1, x_pu_2], 1)
    x_abc_pu = sequence_to_phase(np.transpose(x_012_pu))

    return x_abc_pu


def _get_bus_results(ppc_0, ppc_1, ppc_2, net):

    bus_lookup = net._pd2ppc_lookups["bus"]
    ppc_index = bus_lookup[net.bus.index]
    nb = len(ppc_index)
    baseV = ppc_1['bus'][0:nb, BASE_KV]
    baseI = ppc_1["baseMVA"] / (baseV * np.sqrt(3))
    el_idx = list(range(0, nb))

    i_kssv_abc_pu = _compute_x_abc(ppc_0, ppc_1, ppc_2, el_idx, 'bus', IKSSV, PHI_IKSSV_DEGREE)
    i_kssc_abc_pu = _compute_x_abc(ppc_0, ppc_1, ppc_2, el_idx, 'bus', IKSSC, PHI_IKSSC_DEGREE)
    i_kss_abc_ka = (i_kssv_abc_pu + i_kssc_abc_pu) * baseI
    i_kss_abc_ka[abs(i_kss_abc_ka)<10**-6] = 0 + 1j*0

    v_abc_pu = _compute_x_abc(ppc_0, ppc_1, ppc_2, el_idx, 'bus', VM, VA)

    s_kss_abc_mva = baseV * np.abs(i_kss_abc_ka) / np.sqrt(3)
    s_abc_mva = np.conj(i_kss_abc_ka) * v_abc_pu * baseV / np.sqrt(3)

    # Adding the ikss and skss values
    net.res_bus_sc["index"] = net.bus.index
    for i, phase in enumerate(['a', 'b', 'c']):
        net.res_bus_sc[f'vm_{phase}_pu'] = np.abs(v_abc_pu[i, ppc_index])
        net.res_bus_sc[f'va_{phase}_degree'] = np.angle(v_abc_pu[i, ppc_index], deg=True)
        net.res_bus_sc[f'ikss_{phase}_ka'] = np.abs(i_kss_abc_ka[i, ppc_index])
        net.res_bus_sc[f'ikss_{phase}_degree'] = np.angle(i_kss_abc_ka[i, ppc_index], deg=True)
        net.res_bus_sc[f'skss_{phase}_mva'] = s_kss_abc_mva[i, ppc_index]
        net.res_bus_sc[f'p_{phase}_mw'] = s_abc_mva[i, ppc_index].real
        net.res_bus_sc[f'q_{phase}_mvar'] = s_abc_mva[i, ppc_index].imag


def _get_branch_results(ppc_0, ppc_1, ppc_2, el_idx, side):

    if (side == "from") or (side == "hv"): 
        bus_side = F_BUS
        imag_idx = IKSS_F
        iangle_idx = IKSS_ANGLE_F
        vmag_idx = VKSS_MAGN_F
        vangle_idx = VKSS_ANGLE_F
    elif (side == "to") or (side == "lv"):
        bus_side = T_BUS
        imag_idx = IKSS_T
        iangle_idx = IKSS_ANGLE_T
        vmag_idx = VKSS_MAGN_T
        vangle_idx = VKSS_ANGLE_T
    
    bus = np.real(ppc_1["branch"][el_idx, bus_side]).astype(np.int64)
    baseV = baseV = ppc_1['bus'][bus, BASE_KV]
    baseI = ppc_1["baseMVA"] / (baseV * np.sqrt(3))

    i_abc_pu = _compute_x_abc(ppc_0, ppc_1, ppc_2, el_idx, 'branch', imag_idx, iangle_idx)

    i_abc_ka = i_abc_pu * baseI
    i_abc_ka[abs(i_abc_ka)<10**-6] = 0 + 1j*0

    v_abc_pu = _compute_x_abc(ppc_0, ppc_1, ppc_2, el_idx, 'branch', vmag_idx, vangle_idx)

    s_kss_abc_mva = baseV * np.abs(i_abc_ka) / np.sqrt(3)
    s_abc_mva = np.conj(i_abc_ka) * v_abc_pu * baseV / np.sqrt(3)

    return i_abc_ka, s_abc_mva, s_kss_abc_mva


def _get_trafo_zero_sequence_injection(ppc_0, ppc_1, ppc_2, el_idx, side):

    if (side == "from") or (side == "hv"): 
        bus_side = F_BUS
        vmag_idx = VKSS_MAGN_F
        vangle_idx = VKSS_ANGLE_F
    elif (side == "to") or (side == "lv"):
        bus_side = T_BUS
        vmag_idx = VKSS_MAGN_T
        vangle_idx = VKSS_ANGLE_T

    baseI = ppc_1["internal"]["baseI"]
    bus = ppc_0['branch'][el_idx, bus_side].astype(int)

    v_abc_pu = _compute_x_abc(ppc_0, ppc_1, ppc_2, el_idx, 'branch', vmag_idx, vangle_idx)
    v_pu_0 = ppc_0['branch'][el_idx, vmag_idx] * np.exp(1j * np.deg2rad(ppc_0['branch'][el_idx, vangle_idx].real))

    ys = (ppc_0["bus"][bus, GS] + 1j * ppc_0["bus"][bus, BS]) / ppc_0["baseMVA"]
    i0 = ys * v_pu_0 * baseI[bus]

    i_012_ka = np.stack([i0, 0*i0, 0*i0], 1)
    i_abc_ka = sequence_to_phase(np.transpose(i_012_ka))

    baseV = ppc_1["internal"]["baseV"][ppc_1["branch"][el_idx, bus_side].real.astype(np.int64)].ravel()
    s_abc_mva = np.conj(i_abc_ka) * v_abc_pu * baseV / np.sqrt(3)

    return i_abc_ka, s_abc_mva


def _get_line_results(ppc_0, ppc_1, ppc_2, net):

    branch_lookup = net._pd2ppc_lookups["branch"]
    if "line" not in branch_lookup:
        return
    f, t = branch_lookup["line"]
    el_idx = list(range(f, t))

    net.res_line_sc["index"] = net.line.index
    for side in ["from", "to"]:
        i_abc_ka, s_abc_mva, s_kss_abc_mva = _get_branch_results(ppc_0, ppc_1, ppc_2, el_idx, side)
        for phase_idx, phase in enumerate(("a", "b", "c")):
            net.res_line_sc[f"ikss_{phase}_{side}_ka"] = np.abs(i_abc_ka[phase_idx,:])
            net.res_line_sc[f"ikss_{phase}_{side}_degree"] = np.angle(i_abc_ka[phase_idx,:], deg=True)
            net.res_line_sc[f"skss_{phase}_{side}_mva"] = s_kss_abc_mva[phase_idx,:]
            net.res_line_sc[f"p_{phase}_{side}_mw"] = s_abc_mva[phase_idx,:].real
            net.res_line_sc[f"q_{phase}_{side}_mvar"] = s_abc_mva[phase_idx,:].imag


def _get_trafo_results(ppc_0, ppc_1, ppc_2, net):

    branch_lookup = net._pd2ppc_lookups["branch"]
    if "trafo" not in branch_lookup:
        return
    f, t = branch_lookup["trafo"]
    el_idx = list(range(f, t))

    net.res_trafo_sc["index"] = net.trafo.index
    for side in ["hv", "lv"]: 
        i_abc_ka_br, s_abc_mva_br, _ = _get_branch_results(ppc_0, ppc_1, ppc_2, el_idx, side)
        i_abc_ka_zinj, s_abc_mva_zinj  = _get_trafo_zero_sequence_injection(ppc_0, ppc_1, ppc_2, el_idx, side)

        i_abc_ka = i_abc_ka_br + i_abc_ka_zinj
        i_abc_ka[abs(i_abc_ka)<10**-6] = 0 + 1j*0
        s_abc_mva = s_abc_mva_br + s_abc_mva_zinj

        if side == "hv":
            baseV = ppc_1["internal"]["baseV"][ppc_1["branch"][el_idx, 0].real.astype(np.int64)].ravel()
        else:
            baseV = ppc_1["internal"]["baseV"][ppc_1["branch"][el_idx, 1].real.astype(np.int64)].ravel()
        s_kss_abc_mva = baseV * np.abs(i_abc_ka) / np.sqrt(3)

        for phase_idx, phase in enumerate(("a", "b", "c")):
            net.res_trafo_sc[f"ikss_{phase}_{side}_ka"] = np.abs(i_abc_ka[phase_idx,:])
            net.res_trafo_sc[f"ikss_{phase}_{side}_degree"] = np.angle(i_abc_ka[phase_idx,:], deg=True)
            net.res_trafo_sc[f"skss_{phase}_{side}_mva"] = s_kss_abc_mva[phase_idx, :]
            net.res_trafo_sc[f"p_{phase}_{side}_mw"] = s_abc_mva[phase_idx,:].real
            net.res_trafo_sc[f"q_{phase}_{side}_mvar"] = s_abc_mva[phase_idx,:].imag


def _get_trafo3w_results(ppc_0, ppc_1, ppc_2, net):

    branch_lookup = net._pd2ppc_lookups["branch"]
    if "trafo3w" not in branch_lookup:
        return
    f, t = branch_lookup["trafo3w"]
    hv = int(f + (t - f) / 3)
    mv = int(f + 2 * (t - f) / 3)
    lv = t

    net.res_trafo3w_sc["index"] = net.trafo3w.index
    for f, t, side_br, side_tr in [(f, hv, "from", "hv"), (hv, mv, "to", "mv"), (mv, lv, "to", "lv")]:

        el_idx = list(range(f, t))

        i_abc_ka_br, s_abc_mva_br, _ = _get_branch_results(ppc_0, ppc_1, ppc_2, el_idx, side_br)
        i_abc_ka_zinj, s_abc_mva_zinj  = _get_trafo_zero_sequence_injection(ppc_0, ppc_1, ppc_2, el_idx, side_br)

        i_abc_ka = i_abc_ka_br + i_abc_ka_zinj
        s_abc_mva = s_abc_mva_br + s_abc_mva_zinj

        if side_br == "from":
            baseV = ppc_1["internal"]["baseV"][ppc_1["branch"][el_idx, 0].real.astype(np.int64)].ravel()
        else:
            baseV = ppc_1["internal"]["baseV"][ppc_1["branch"][el_idx, 1].real.astype(np.int64)].ravel()
        s_kss_abc_mva = baseV * np.abs(i_abc_ka) / np.sqrt(3)

        for phase_idx, phase in enumerate(("a", "b", "c")):
            net.res_trafo_sc[f"ikss_{phase}_{side_tr}_ka"] = np.abs(i_abc_ka[:, 0, phase_idx])
            net.res_trafo_sc[f"ikss_{phase}_{side_tr}_degree"] = np.angle(i_abc_ka[:, 0, phase_idx], deg=True)
            net.res_trafo_sc[f"skss_{phase}_{side_tr}_mva"] = s_kss_abc_mva[:, 0, phase_idx]
            net.res_trafo_sc[f"p_{phase}_{side_tr}_mw"] = s_abc_mva[:, 0, phase_idx].real
            net.res_trafo_sc[f"q_{phase}_{side_tr}_mvar"] = s_abc_mva[:, 0, phase_idx].imag


def _get_impedance_results(ppc_0, ppc_1, ppc_2, net):
    pass


def _get_ext_grid_results(ppc_0, ppc_1, ppc_2, net, bus):

    eg_bus = net.ext_grid["bus"]
    eg_index = net.ext_grid.index
    if len(eg_bus) == 0:
        return
    bus_lookup = net._pd2ppc_lookups["bus"]
    eg_ppc_index = bus_lookup[eg_bus]
    ppc_fault_bus = bus_lookup[bus]

    baseV = ppc_1['bus'][eg_ppc_index, BASE_KV]
    baseI = ppc_1["baseMVA"] / (baseV * np.sqrt(3))
    case = net._options["case"]

    ynet12 = (ppc_1["bus"][eg_ppc_index, GS] + 1j*ppc_1["bus"][eg_ppc_index, BS]) / ppc_1['baseMVA']
    ynet0 = (ppc_0["bus"][eg_ppc_index, GS] + 1j*ppc_0["bus"][eg_ppc_index, BS]) / ppc_0['baseMVA']

    if case == "min":
        Vg = ppc_1["bus"][ppc_fault_bus, C_MIN]
    else:
        Vg = ppc_1["bus"][ppc_fault_bus, C_MAX]

    V_0 = ppc_0["bus"][eg_ppc_index, VM] * np.exp(1j * np.deg2rad(ppc_0["bus"][eg_ppc_index, VA].real))
    V_1 = ppc_1["bus"][eg_ppc_index, VM] * np.exp(1j * np.deg2rad(ppc_1["bus"][eg_ppc_index, VA].real))
    V_2 = ppc_2["bus"][eg_ppc_index, VM] * np.exp(1j * np.deg2rad(ppc_2["bus"][eg_ppc_index, VA].real))

    i_pu_0 = - ynet0*V_0
    i_pu_1 = ynet12*(Vg-V_1)
    i_pu_2 = - ynet12*V_2

    i_012_pu = np.stack([i_pu_0, i_pu_1, i_pu_2], 1)
    i_abc_pu = sequence_to_phase(np.transpose(i_012_pu))
    i_abc_ka = i_abc_pu * baseI

    v_012_pu = np.stack([[0], Vg, [0]], 1)
    v_abc_pu = sequence_to_phase(np.transpose(v_012_pu))

    s_kss_abc_mva = baseV * np.abs(i_abc_ka) / np.sqrt(3)
    s_abc_mva = np.conj(i_abc_ka) * v_abc_pu * baseV / np.sqrt(3)

    net.res_ext_grid_sc["index"] = net.ext_grid.index
    for i, phase in enumerate(['a', 'b', 'c']):
        net.res_ext_grid_sc.loc[eg_index, f'ikss_{phase}_ka'] = np.abs(i_abc_ka[i, :])
        net.res_ext_grid_sc.loc[eg_index, f'ikss_{phase}_degree'] = np.angle(i_abc_ka[i, :], deg=True)
        net.res_ext_grid_sc.loc[eg_index, f'skss_{phase}_mva'] = s_kss_abc_mva[i, :]
        net.res_ext_grid_sc.loc[eg_index, f'p_{phase}_mw'] = s_abc_mva[i, :].real
        net.res_ext_grid_sc.loc[eg_index, f'q_{phase}_mvar'] = s_abc_mva[i, :].imag
    

def _get_sgen_results(ppc_0, ppc_1, ppc_2, net):
    active_sgens = net.sgen[net._is_elements_final["sgen"] & net.sgen.current_source]
    if len(active_sgens) == 0:
        return
    
    sgen_idx = active_sgens.index
    bus_lookup = net._pd2ppc_lookups["bus"]
    ppc_index = bus_lookup[active_sgens.bus]
    baseV = ppc_1['bus'][ppc_index, BASE_KV]
    baseI = ppc_1["baseMVA"] / (baseV * np.sqrt(3))

    i_sgen_abc_pu = _compute_x_abc(ppc_0, ppc_1, ppc_2, ppc_index, 'bus', IKCV, PHI_IKCV_DEGREE)
    i_sgen_abc_ka = i_sgen_abc_pu * baseI
    i_sgen_abc_ka[abs(i_sgen_abc_ka)<10**-6] = 0 + 1j*0

    v_abc_pu = _compute_x_abc(ppc_0, ppc_1, ppc_2, ppc_index, 'bus', VM, VA)

    s_kss_abc_mva = baseV * np.abs(i_sgen_abc_ka) / np.sqrt(3)
    s_abc_mva = np.conj(i_sgen_abc_ka) * v_abc_pu * baseV / np.sqrt(3)

    # Adding the ikss and skss values
    net.res_sgen_sc["index"] = net.sgen.index
    for i, phase in enumerate(['a', 'b', 'c']):
        net.res_sgen_sc.loc[sgen_idx, f'ikss_{phase}_ka'] = np.abs(i_sgen_abc_ka[i, :])
        net.res_sgen_sc.loc[sgen_idx, f'ikss_{phase}_degree'] = np.angle(i_sgen_abc_ka[i, :], deg=True)
        net.res_sgen_sc.loc[sgen_idx, f'skss_{phase}_mva'] = s_kss_abc_mva[i, :]
        net.res_sgen_sc.loc[sgen_idx, f'p_{phase}_mw'] = s_abc_mva[i, :].real
        net.res_sgen_sc.loc[sgen_idx, f'q_{phase}_mvar'] = s_abc_mva[i, :].imag



def _get_gen_results(ppc_0, ppc_1, ppc_2, net, bus):
    active_gens = net.gen[net._is_elements_final["gen"]]
    if len(active_gens) == 0:
        return
    gen_bus = active_gens["bus"]
    gen_idx = active_gens.index
    bus_lookup = net._pd2ppc_lookups["bus"]
    gen_ppc_index = bus_lookup[gen_bus]
    ppc_fault_bus = bus_lookup[bus]

    baseV = ppc_1['bus'][gen_ppc_index, BASE_KV]
    baseI = ppc_1["baseMVA"] / (baseV * np.sqrt(3))
    case = net._options["case"]

    ygen0 = (ppc_0["bus"][gen_ppc_index, GS_GEN] + 1j*ppc_0["bus"][gen_ppc_index, BS_GEN]) / ppc_0['baseMVA']
    ygen1 = (ppc_1["bus"][gen_ppc_index, GS_GEN] + 1j*ppc_1["bus"][gen_ppc_index, BS_GEN]) / ppc_1['baseMVA']
    ygen2 = (ppc_2["bus"][gen_ppc_index, GS_GEN] + 1j*ppc_2["bus"][gen_ppc_index, BS_GEN]) / ppc_2['baseMVA']

    if case == "min":
        Vg = ppc_1["bus"][ppc_fault_bus, C_MIN]
    else:
        Vg = ppc_1["bus"][ppc_fault_bus, C_MAX]

    V_0 = ppc_0["bus"][gen_ppc_index, VM] * np.exp(1j * np.deg2rad(ppc_0["bus"][gen_ppc_index, VA].real))
    V_1 = ppc_1["bus"][gen_ppc_index, VM] * np.exp(1j * np.deg2rad(ppc_1["bus"][gen_ppc_index, VA].real))
    V_2 = ppc_2["bus"][gen_ppc_index, VM] * np.exp(1j * np.deg2rad(ppc_2["bus"][gen_ppc_index, VA].real))

    i_pu_0 = - ygen0 * V_0
    i_pu_1 = ygen1 * (Vg - V_1)
    i_pu_2 = - ygen2 * V_2

    i_012_pu = np.stack([i_pu_0, i_pu_1, i_pu_2], 1)
    i_abc_pu = sequence_to_phase(np.transpose(i_012_pu))
    i_abc_ka = i_abc_pu * baseI

    v_012_pu = np.stack([V_0, V_1, V_2], 1)
    v_abc_pu = sequence_to_phase(np.transpose(v_012_pu))

    s_kss_abc_mva = baseV * np.abs(i_abc_ka) / np.sqrt(3)
    s_abc_mva = np.conj(i_abc_ka) * v_abc_pu * baseV / np.sqrt(3)

    net.res_gen_sc["index"] = net.gen.index
    for i, phase in enumerate(['a', 'b', 'c']):
        net.res_gen_sc.loc[gen_idx, f'ikss_{phase}_ka'] = np.abs(i_abc_ka[i, :])
        net.res_gen_sc.loc[gen_idx, f'ikss_{phase}_degree'] = np.angle(i_abc_ka[i, :], deg=True)
        net.res_gen_sc.loc[gen_idx, f'skss_{phase}_mva'] = s_kss_abc_mva[i, :]
        net.res_gen_sc.loc[gen_idx, f'p_{phase}_mw'] = s_abc_mva[i, :].real
        net.res_gen_sc.loc[gen_idx, f'q_{phase}_mvar'] = s_abc_mva[i, :].imag
        net.res_bus_sc.loc[gen_bus, f'ikss_{phase}_ka'] += np.abs(i_abc_ka[i, :])
        net.res_bus_sc.loc[gen_bus, f'ikss_{phase}_degree'] += np.angle(i_abc_ka[i, :], deg=True)
        net.res_bus_sc.loc[gen_bus, f'skss_{phase}_mva'] += s_kss_abc_mva[i, :]
        net.res_bus_sc.loc[gen_bus, f'p_{phase}_mw'] += s_abc_mva[i, :].real
        net.res_bus_sc.loc[gen_bus, f'q_{phase}_mvar'] += s_abc_mva[i, :].imag


def _get_motor_results(ppc_0, ppc_1, ppc_2, net):
    pass


def _get_load_results(ppc_0, ppc_1, ppc_2, net):

    idx = net.load.index.values
    if len(idx) < 1:
        return
    net.res_load_sc["index"] = idx
    for i, phase in enumerate(['a', 'b', 'c']):
        net.res_load_sc[f'ikss_{phase}_ka'] = 0
        net.res_load_sc[f'ikss_{phase}_degree'] = 0
        net.res_load_sc[f'skss_{phase}_mva'] = 0
        net.res_load_sc[f'p_{phase}_mw'] = 0
        net.res_load_sc[f'q_{phase}_mvar'] = 0


def _get_shunt_results(ppc_0, ppc_1, ppc_2, net):

    idx = net.shunt.index
    if len(idx) < 1:
        return
    net.res_shunt_sc["index"] = idx
    for i, phase in enumerate(['a', 'b', 'c']):
        net.res_shunt_sc[f'ikss_{phase}_ka'] = 0
        net.res_shunt_sc[f'ikss_{phase}_degree'] = 0
        net.res_shunt_sc[f'skss_{phase}_mva'] = 0
        net.res_shunt_sc[f'p_{phase}_mw'] = 0
        net.res_shunt_sc[f'q_{phase}_mvar'] = 0


# def _get_line_branch_results(net, ppc_1, v_abc_pu, i_abc_ka, s_abc_mva): #renamaed the function properly
#     branch_lookup = net._pd2ppc_lookups["branch"]
#     case = net._options["case"]
#     fault = net._options["fault"]

#     if "line" in branch_lookup:
#         f, t = branch_lookup["line"]
#         minmax = np.max if case == "max" else np.min

#         # todo: check axis of max with more lines in the grid
#         i_max_per_line_ka = np.max(np.abs(i_abc_ka), axis=1)
#         net.res_line_sc["ikss_ka"] = minmax(i_max_per_line_ka[f:t, :], axis=1)
        
#         if fault == 'LLL': #corrected bramch p amd q values
#             mult = 3
#         else:
#             mult = 1

#         for phase_idx, phase in enumerate(("a", "b", "c")):
#             for side_idx, side in enumerate(("from", "to")):
#                 net.res_line_sc[f"ikss_{phase}_{side}_ka"] = np.abs(i_abc_ka[f:t, side_idx, phase_idx])
#                 net.res_line_sc[f"ikss_{phase}_{side}_degree"] = np.angle(i_abc_ka[f:t, side_idx, phase_idx], deg=True)

#             for side_idx, side in enumerate(("from", "to")):
#                 net.res_line_sc[f"p_{phase}_{side}_mw"] = s_abc_mva[f:t, side_idx, phase_idx].real * mult
#                 net.res_line_sc[f"q_{phase}_{side}_mvar"] = s_abc_mva[f:t, side_idx, phase_idx].imag * mult

#             for side_idx, side in enumerate(("from", "to")):
#                 net.res_line_sc[f"vm_{phase}_{side}_pu"] = np.abs(v_abc_pu[f:t, side_idx, phase_idx])
#                 net.res_line_sc[f"va_{phase}_{side}_degree"] = np.angle(v_abc_pu[f:t, side_idx, phase_idx], deg=True)

#         # todo: ip, ith
#         if net._options["ip"]:
#             net.res_line_sc["ip_ka"] = minmax(ppc_1["branch"][f:t, [IP_F, IP_T]].real, axis=1)
#         if net._options["ith"]:
#             net.res_line_sc["ith_ka"] = minmax(ppc_1["branch"][f:t, [ITH_F, ITH_T]].real, axis=1)


# def _get_trafo_lg_results(net, v_abc_pu, i_abc_ka, s_abc_mva):
#     branch_lookup = net._pd2ppc_lookups["branch"]
#     if "trafo" in branch_lookup:
#         f, t = branch_lookup["trafo"]

#         for phase_idx, phase in enumerate(("a", "b", "c")):
#             for side_idx, side in enumerate(("hv", "lv")):
#                 net.res_trafo_sc[f"ikss_{phase}_{side}_ka"] = np.abs(i_abc_ka[f:t, side_idx, phase_idx])
#                 net.res_trafo_sc[f"ikss_{phase}_{side}_degree"] = np.angle(i_abc_ka[f:t, side_idx, phase_idx], deg=True)

#             for side_idx, side in enumerate(("hv", "lv")):
#                 net.res_trafo_sc[f"p_{phase}_{side}_mw"] = s_abc_mva[f:t, side_idx, phase_idx].real
#                 net.res_trafo_sc[f"q_{phase}_{side}_mvar"] = s_abc_mva[f:t, side_idx, phase_idx].imag

#             for side_idx, side in enumerate(("hv", "lv")):
#                 net.res_trafo_sc[f"vm_{phase}_{side}_pu"] = np.abs(v_abc_pu[f:t, side_idx, phase_idx])
#                 net.res_trafo_sc[f"va_{phase}_{side}_degree"] = np.angle(v_abc_pu[f:t, side_idx, phase_idx], deg=True)


def _extract_net_results(net, ppc_0, ppc_1, ppc_2, bus):
    # Extraction of results for the case branch_results = True
    _set_buses_and_branches_out_of_service(ppc_1)
    _get_bus_results(ppc_0, ppc_1, ppc_2, net)
    _get_line_results(ppc_0, ppc_1, ppc_2, net)
    _get_trafo_results(ppc_0, ppc_1, ppc_2, net)
    _get_trafo3w_results(ppc_0, ppc_1, ppc_2, net)
    _get_impedance_results(ppc_0, ppc_1, ppc_2, net)    # TODO
    _get_ext_grid_results(ppc_0, ppc_1, ppc_2, net, bus)
    _get_sgen_results(ppc_0, ppc_1, ppc_2, net)
    _get_gen_results(ppc_0, ppc_1, ppc_2, net, bus)
    _get_motor_results(ppc_0, ppc_1, ppc_2, net)        # TODO
    _get_load_results(ppc_0, ppc_1, ppc_2, net)
    _get_shunt_results(ppc_0, ppc_1, ppc_2, net)


        # TODO: _get_switch_all_results(net, ppc_1, bus)
    # if ("grounding_type" in net.trafo.columns) and (net.trafo["grounding_type"] == "resonant").any():
    #     _validate_resonate_grounding(net, ppc_0, bus)

def _extract_bus_results(net, ppc_0, ppc_1, ppc_2, bus):
    # Extraction of the results for the case branch_results = False
    net.res_bus_sc["index"] = net.bus.index
    bus_lookup = net._pd2ppc_lookups["bus"]
    ppc_index = bus_lookup[bus]
    baseI = ppc_1["internal"]["baseI"][ppc_index]
    baseV = ppc_1['bus'][:, BASE_KV][ppc_index]

    i_kssv_abc_pu = _compute_x_abc(ppc_0, ppc_1, ppc_2, ppc_index, 'bus', IKSSV, PHI_IKSSV_DEGREE)
    i_kssc_abc_pu = _compute_x_abc(ppc_0, ppc_1, ppc_2, ppc_index, 'bus', IKSSC, PHI_IKSSC_DEGREE)
    i_kss_abc_ka = (i_kssv_abc_pu + i_kssc_abc_pu) * baseI
    i_kss_abc_ka[abs(i_kss_abc_ka)<10**-6] = 0 + 1j*0

    s_kss_abc_mva = baseV * np.abs(i_kss_abc_ka) / np.sqrt(3)

    # Adding the ikss and skss values
    for i, phase in enumerate(['a', 'b', 'c']):
        net.res_bus_sc.loc[bus, f'ikss_{phase}_ka'] = np.abs(i_kss_abc_ka[i, :])
        net.res_bus_sc.loc[bus,f'ikss_{phase}_degree'] = np.angle(i_kss_abc_ka[i, :], deg=True)
        net.res_bus_sc.loc[bus,f'skss_{phase}_mva'] = s_kss_abc_mva[i, :]
        del net.res_bus_sc[f'vm_{phase}_pu']
        del net.res_bus_sc[f'va_{phase}_degree']
        del net.res_bus_sc[f'p_{phase}_mw']
        del net.res_bus_sc[f'q_{phase}_mvar']

    net.res_bus_sc.loc[bus, "rk0_ohm"] = ppc_0["bus"][ppc_index, R_EQUIV_OHM]
    net.res_bus_sc.loc[bus, "xk0_ohm"] = ppc_0["bus"][ppc_index, X_EQUIV_OHM]
    net.res_bus_sc.loc[bus, "rk1_ohm"] = ppc_1["bus"][ppc_index, R_EQUIV_OHM]
    net.res_bus_sc.loc[bus, "xk1_ohm"] = ppc_1["bus"][ppc_index, X_EQUIV_OHM]
    net.res_bus_sc.loc[bus, "rk2_ohm"] = ppc_2["bus"][ppc_index, R_EQUIV_OHM]
    net.res_bus_sc.loc[bus, "xk2_ohm"] = ppc_2["bus"][ppc_index, X_EQUIV_OHM]

    net.res_bus_sc.drop(net.res_bus_sc.index[np.isnan(net.res_bus_sc["ikss_a_ka"]).values], inplace=True)



# def _get_branch_result_from_internal(variable, ppc, ppc_index, f, t):
#     if variable in ppc["internal"]:
#         return ppc["internal"][variable].iloc[f:t, :].loc[:, ppc_index].values.real.reshape(-1, 1)
#     else:
#         return np.full((t-f) * len(ppc_index), np.nan, dtype=np.float64)
    

# def _get_bus_results(net, ppc_0, ppc_1, ppc_2, bus):
#     bus_lookup = net._pd2ppc_lookups["bus"]
#     ppc_index = bus_lookup[net.bus.index]

#     ppc_sequence = {0: ppc_0, 1: ppc_1, 2: ppc_2, "": ppc_1}
#     if net["_options"]["fault"] == "LG":
#         sequence_relevant = range(3)
#     elif net["_options"]["fault"] == "LLG":
#         sequence_relevant = range(3)
#     elif net["_options"]["fault"] == "LL":
#         sequence_relevant = range(3)
#     elif net["_options"]["fault"] == "LLL":
#         sequence_relevant = ("",)
#     for sequence in sequence_relevant:
#         ppc_s = ppc_sequence[sequence]
#         if net["_options"]["fault"] == "LL":
#             # TODO: ask Marco where this can be done before results are written into tables
#             if sequence in [1, 2]:
#                 fault_ohm_factor = 2
#             elif sequence == 0:
#                 fault_ohm_factor = 1
#             net.res_bus_sc[f"rk{sequence}_ohm"] = (ppc_s["bus"][ppc_index, R_EQUIV_OHM] +
#                                                    net["_options"]["r_fault_ohm"]/fault_ohm_factor)
#             net.res_bus_sc[f"xk{sequence}_ohm"] = (ppc_s["bus"][ppc_index, X_EQUIV_OHM] +
#                                                    net["_options"]["x_fault_ohm"]/fault_ohm_factor)
#         else:
#             net.res_bus_sc[f"rk{sequence}_ohm"] = ppc_s["bus"][ppc_index, R_EQUIV_OHM]
#             net.res_bus_sc[f"xk{sequence}_ohm"] = ppc_s["bus"][ppc_index, X_EQUIV_OHM]
#         # in trafo3w, we add very high numbers (1e10) as impedances to block current
#         # here, we need to replace such high values by np.inf
#         baseZ = ppc_s["bus"][ppc_index, BASE_KV] ** 2 / ppc_s["baseMVA"]
#         net.res_bus_sc.loc[net.res_bus_sc[f"xk{sequence}_ohm"] / baseZ > 1e9, f"xk{sequence}_ohm"] = np.inf
#         net.res_bus_sc.loc[net.res_bus_sc[f"rk{sequence}_ohm"] / baseZ > 1e9, f"rk{sequence}_ohm"] = np.inf
#     if net._options["ip"]:
#         net.res_bus_sc["ip_ka"] = ppc_1["bus"][ppc_index, IP]
#     if net._options["ith"]:
#         net.res_bus_sc["ith_ka"] = ppc_1["bus"][ppc_index, ITH]

#     net.res_bus_sc = net.res_bus_sc.loc[bus, :]


# def _get_line_results(net, ppc):
#     branch_lookup = net._pd2ppc_lookups["branch"]
#     case = net._options["case"]
#     if "line" in branch_lookup:
#         f, t = branch_lookup["line"]
#         minmax = np.max if case == "max" else np.min
#         net.res_line_sc["ikss_ka"] = minmax(ppc["branch"][f:t, [IKSS_F, IKSS_T]].real, axis=1)
#         net.res_line_sc["ikss_from_ka"] = ppc["branch"][f:t, IKSS_F].real
#         net.res_line_sc["ikss_from_degree"] = ppc["branch"][f:t, IKSS_ANGLE_F].real
#         net.res_line_sc["ikss_to_ka"] = ppc["branch"][f:t, IKSS_T].real
#         net.res_line_sc["ikss_to_degree"] = ppc["branch"][f:t, IKSS_ANGLE_T].real

#         # adding columns for new calculated VPQ
#         net.res_line_sc["p_from_mw"] = ppc["branch"][f:t, PKSS_F].real
#         net.res_line_sc["q_from_mvar"] = ppc["branch"][f:t, QKSS_F].real

#         net.res_line_sc["p_to_mw"] = ppc["branch"][f:t, PKSS_T].real
#         net.res_line_sc["q_to_mvar"] = ppc["branch"][f:t, QKSS_T].real

#         net.res_line_sc["vm_from_pu"] = ppc["branch"][f:t, VKSS_MAGN_F].real
#         net.res_line_sc["va_from_degree"] = ppc["branch"][f:t, VKSS_ANGLE_F].real

#         net.res_line_sc["vm_to_pu"] = ppc["branch"][f:t, VKSS_MAGN_T].real
#         net.res_line_sc["va_to_degree"] = ppc["branch"][f:t, VKSS_ANGLE_T].real

#         if net._options["ip"]:
#             net.res_line_sc["ip_ka"] = minmax(ppc["branch"][f:t, [IP_F, IP_T]].real, axis=1)
#         if net._options["ith"]:
#             net.res_line_sc["ith_ka"] = minmax(ppc["branch"][f:t, [ITH_F, ITH_T]].real, axis=1)


# def _get_line_all_result_old(net, ppc, bus):
#     case = net._options["case"]

#     bus, ppc_index = _get_bus_ppc_idx_for_br_all_results(net, ppc, bus)
#     branch_lookup = net._pd2ppc_lookups["branch"]

#     multindex = pd.MultiIndex.from_product([net.res_line_sc.index, bus], names=['line','bus'])
#     net.res_line_sc = net.res_line_sc.reindex(multindex)

#     if "line" in branch_lookup:
#         f, t = branch_lookup["line"]
#         minmax = np.maximum if case == "max" else np.minimum

#         net.res_line_sc["ikss_ka"] = minmax(ppc["internal"]["branch_ikss_f"].iloc[f:t,:].loc[:, ppc_index].values.real.reshape(-1, 1),
#                                             ppc["internal"]["branch_ikss_t"].iloc[f:t,:].loc[:, ppc_index].values.real.reshape(-1, 1))

#         net.res_line_sc["ikss_from_ka"] = _get_branch_result_from_internal("branch_ikss_f", ppc, ppc_index, f, t)
#         net.res_line_sc["ikss_from_degree"] = _get_branch_result_from_internal("branch_ikss_angle_f", ppc, ppc_index, f, t)

#         net.res_line_sc["ikss_to_ka"] = _get_branch_result_from_internal("branch_ikss_t", ppc, ppc_index, f, t)
#         net.res_line_sc["ikss_to_degree"] = _get_branch_result_from_internal("branch_ikss_angle_t", ppc, ppc_index, f, t)

#         net.res_line_sc["p_from_mw"] = _get_branch_result_from_internal("branch_pkss_f", ppc, ppc_index, f, t)
#         net.res_line_sc["q_from_mvar"] = _get_branch_result_from_internal("branch_qkss_f", ppc, ppc_index, f, t)

#         net.res_line_sc["p_to_mw"] = _get_branch_result_from_internal("branch_pkss_t", ppc, ppc_index, f, t)
#         net.res_line_sc["q_to_mvar"] = _get_branch_result_from_internal("branch_qkss_t", ppc, ppc_index, f, t)

#         net.res_line_sc["vm_from_pu"] = _get_branch_result_from_internal("branch_vkss_f", ppc, ppc_index, f, t)
#         net.res_line_sc["va_from_degree"] = _get_branch_result_from_internal("branch_vkss_angle_f", ppc, ppc_index, f, t)

#         net.res_line_sc["vm_to_pu"] = _get_branch_result_from_internal("branch_vkss_t", ppc, ppc_index, f, t)
#         net.res_line_sc["va_to_degree"] = _get_branch_result_from_internal("branch_vkss_angle_t", ppc, ppc_index, f, t)

#         if net._options["ip"]:
#             net.res_line_sc["ip_ka"] = minmax(ppc["internal"]["branch_ip_f"].iloc[f:t,:].loc[:, ppc_index].values.real.reshape(-1, 1),
#                                               ppc["internal"]["branch_ip_t"].iloc[f:t,:].loc[:, ppc_index].values.real.reshape(-1, 1))
#         if net._options["ith"]:
#             net.res_line_sc["ith_ka"] = minmax(ppc["internal"]["branch_ith_f"].iloc[f:t,:].loc[:, ppc_index].values.real.reshape(-1, 1),
#                                                ppc["internal"]["branch_ith_t"].iloc[f:t,:].loc[:, ppc_index].values.real.reshape(-1, 1))



# def _get_trafo_results(net, ppc):
#     branch_lookup = net._pd2ppc_lookups["branch"]
#     if "trafo" in branch_lookup:
#         f, t = branch_lookup["trafo"]
#         net.res_trafo_sc["ikss_hv_ka"] = ppc["branch"][f:t, IKSS_F].real
#         net.res_trafo_sc["ikss_hv_degree"] = ppc["branch"][f:t, IKSS_ANGLE_F].real
#         net.res_trafo_sc["ikss_lv_ka"] = ppc["branch"][f:t, IKSS_T].real
#         net.res_trafo_sc["ikss_lv_degree"] = ppc["branch"][f:t, IKSS_ANGLE_T].real

#         # adding columns for new calculated VPQ
#         net.res_trafo_sc["p_hv_mw"] = ppc["branch"][f:t, PKSS_F].real
#         net.res_trafo_sc["q_hv_mvar"] = ppc["branch"][f:t, QKSS_F].real

#         net.res_trafo_sc["p_lv_mw"] = ppc["branch"][f:t, PKSS_T].real
#         net.res_trafo_sc["q_lv_mvar"] = ppc["branch"][f:t, QKSS_T].real

#         net.res_trafo_sc["vm_hv_pu"] = ppc["branch"][f:t, VKSS_MAGN_F].real
#         net.res_trafo_sc["va_hv_degree"] = ppc["branch"][f:t, VKSS_ANGLE_F].real

#         net.res_trafo_sc["vm_lv_pu"] = ppc["branch"][f:t, VKSS_MAGN_T].real
#         net.res_trafo_sc["va_lv_degree"] = ppc["branch"][f:t, VKSS_ANGLE_T].real


# def _get_trafo_all_results(net, ppc, bus):
#     bus, ppc_index = _get_bus_ppc_idx_for_br_all_results(net, ppc, bus)
#     branch_lookup = net._pd2ppc_lookups["branch"]

#     multindex = pd.MultiIndex.from_product([net.res_trafo_sc.index, bus], names=['trafo', 'bus'])
#     net.res_trafo_sc = net.res_trafo_sc.reindex(multindex)

#     if "trafo" in branch_lookup:
#         f, t = branch_lookup["trafo"]
#         net.res_trafo_sc["ikss_hv_ka"] = ppc["internal"]["branch_ikss_f"].iloc[f:t,:].loc[:, ppc_index].values.real.reshape(-1, 1)
#         net.res_trafo_sc["ikss_lv_ka"] = ppc["internal"]["branch_ikss_t"].iloc[f:t,:].loc[:, ppc_index].values.real.reshape(-1, 1)


# def _get_trafo3w_results(net, ppc):
#     branch_lookup = net._pd2ppc_lookups["branch"]
#     if "trafo3w" in branch_lookup:
#         f, t = net._pd2ppc_lookups["branch"]["trafo3w"]
#         hv = int(f + (t - f) / 3)
#         mv = int(f + 2 * (t - f) / 3)
#         lv = t
#         net.res_trafo3w_sc["ikss_hv_ka"] = ppc["branch"][f:hv, IKSS_F].real
#         net.res_trafo3w_sc["ikss_mv_ka"] = ppc["branch"][hv:mv, IKSS_T].real
#         net.res_trafo3w_sc["ikss_lv_ka"] = ppc["branch"][mv:lv, IKSS_T].real


# def _get_trafo3w_all_results(net, ppc, bus):
#     bus, ppc_index = _get_bus_ppc_idx_for_br_all_results(net, ppc, bus)
#     branch_lookup = net._pd2ppc_lookups["branch"]

#     multindex = pd.MultiIndex.from_product([net.res_trafo3w_sc.index, bus], names=['trafo3w', 'bus'])
#     net.res_trafo3w_sc = net.res_trafo3w_sc.reindex(multindex)

#     if "trafo3w" in branch_lookup:
#         f, t = branch_lookup["trafo3w"]
#         hv = int(f + (t - f) / 3)
#         mv = int(f + 2 * (t - f) / 3)
#         lv = t
#         net.res_trafo3w_sc["ikss_hv_ka"] = ppc["internal"]["branch_ikss_f"].iloc[f:hv,:].loc[:, ppc_index].values.real.reshape(-1, 1)
#         net.res_trafo3w_sc["ikss_mv_ka"] = ppc["internal"]["branch_ikss_t"].iloc[hv:mv, :].loc[:, ppc_index].values.real.reshape(-1, 1)
#         net.res_trafo3w_sc["ikss_lv_ka"] = ppc["internal"]["branch_ikss_t"].iloc[mv:lv, :].loc[:, ppc_index].values.real.reshape(-1, 1)


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