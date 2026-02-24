# -*- coding: utf-8 -*-
# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
import warnings
import copy
import numpy as np
import pandas as pd
from copy import deepcopy
from pandapower.auxiliary import _sum_by_group
from pandapower.pypower.idx_bus import BASE_KV, VM, VA, PD, QD
from pandapower.pypower.idx_brch import TAP
from pandapower.pypower.idx_brch_sc import IKSS_F, IKSS_T, IP_F, IP_T, ITH_F, ITH_T, \
    PKSS_F, QKSS_F, PKSS_T, QKSS_T, VKSS_MAGN_F, VKSS_MAGN_T, VKSS_ANGLE_F, VKSS_ANGLE_T, IKSS_ANGLE_F, IKSS_ANGLE_T
from pandapower.pypower.idx_bus_sc import C_MIN, C_MAX, KAPPA, R_EQUIV, IKSSV, IP, ITH, \
    X_EQUIV, IKSSC, IKCV, M, V_G, K_SG, SKSS, \
    PHI_IKSSV_DEGREE, PHI_IKSSC_DEGREE, PHI_IKCV_DEGREE
from pandapower.shortcircuit.impedance import _calc_zbus_diag

import logging

logger = logging.getLogger(__name__)


def _calc_ikss(net, ppci_0, ppci_1, ppci_2, bus_idx):
    # TODO parameter to tell which Phase is affected
    fault = net._options["fault"]
    case = net._options["case"]

    n_sc_bus = np.shape(bus_idx)[0]
    n_bus = ppci_1["bus"].shape[0]

    # Determine value of the voltage source
    if net._options.get("use_pre_fault_voltage", False):
        V0 = np.full((n_bus, n_sc_bus), ppci_1["bus"][:, [VM]] * np.exp(np.deg2rad(ppci_1["bus"][:, [VA]]) * 1j))
    else:
        V0 = np.full((n_bus, n_sc_bus), ppci_1["bus"][bus_idx, C_MAX if case == "max" else C_MIN], dtype=np.complex128)

        if np.any(ppci_1["branch"][:, TAP] != 1):
            msg = "Calculation does not support calculation of voltages and branch powers for grids that have" \
                  " transformers with rated voltages unequal to bus voltages. Change the transformer data or" \
                  "try using the superposition method by passing 'use_pre_fault_voltage=True'"
            # raise NotImplementedError(msg)
            warnings.warn(msg)

    _calc_ikssv(net, ppci_0, ppci_1, ppci_2, fault, bus_idx, V0)

    ppci_0["bus"][:, IKSSC] = 0
    ppci_1["bus"][:, IKSSC] = 0
    ppci_2["bus"][:, IKSSC] = 0

    ppci_0["bus"][:, PHI_IKSSC_DEGREE] = 0
    ppci_1["bus"][:, PHI_IKSSC_DEGREE] = 0
    ppci_2["bus"][:, PHI_IKSSC_DEGREE] = 0

    for ix, b in enumerate(bus_idx):

        _calc_ikssc(net, ppci_0, ppci_1, ppci_2, b)
        zfault = net._options["z_fault_pu"][ix]
        
        _calc_branch_currents_complex(net, b, ppci_0, ppci_1, ppci_2, 0, zfault)
        _calc_branch_currents_complex(net, b, ppci_0, ppci_1, ppci_2, 1, zfault)
        _calc_branch_currents_complex(net, b, ppci_0, ppci_1, ppci_2, 2, zfault)



def _calc_ikssv(net, ppci_0, ppci_1, ppci_2, fault, bus_idx, V0):

    n_sc_bus = np.shape(bus_idx)[0]

    z_equiv_0 = ppci_0["bus"][bus_idx, R_EQUIV] + ppci_0["bus"][bus_idx, X_EQUIV] * 1j
    z_equiv_1 = ppci_1["bus"][bus_idx, R_EQUIV] + ppci_1["bus"][bus_idx, X_EQUIV] * 1j
    z_equiv_2 = ppci_2["bus"][bus_idx, R_EQUIV] + ppci_2["bus"][bus_idx, X_EQUIV] * 1j

    ppci_0["bus"][:, IKSSV] = 0
    ppci_1["bus"][:, IKSSV] = 0
    ppci_2["bus"][:, IKSSV] = 0

    ppci_0["bus"][:, PHI_IKSSV_DEGREE] = 0
    ppci_1["bus"][:, PHI_IKSSV_DEGREE] = 0
    ppci_2["bus"][:, PHI_IKSSV_DEGREE] = 0

    # Compute fault current contribution from voltage sources
    if fault == "LLL":
        ikssv_1 = V0[bus_idx, np.arange(n_sc_bus)] / z_equiv_1
        ikssv_2 = 0
        ikssv_0 = 0
    elif fault == "LG":
        ikssv_1 = V0[bus_idx, np.arange(n_sc_bus)] / (z_equiv_0 + z_equiv_1 + z_equiv_2)
        ikssv_2 = copy.deepcopy(ikssv_1)
        ikssv_0 = copy.deepcopy(ikssv_1)
    elif fault == "LLG":
        ikssv_1 = V0[bus_idx, np.arange(n_sc_bus)] / (z_equiv_1 + ((z_equiv_2 * z_equiv_0) / (z_equiv_2 + z_equiv_0)))
        ikssv_2 = -ikssv_1 * z_equiv_0 / (z_equiv_0 + z_equiv_2)
        ikssv_0 = -ikssv_1 * z_equiv_2 / (z_equiv_0 + z_equiv_2)
    elif fault == "LL":
        ikssv_1 = V0[bus_idx, np.arange(n_sc_bus)] / (z_equiv_1 + z_equiv_2)
        ikssv_2 = -ikssv_1
        ikssv_0 = 0

    # Save current contribution from voltage sources
    ppci_0["bus"][bus_idx, IKSSV] = abs(ikssv_0)
    ppci_1["bus"][bus_idx, IKSSV] = abs(ikssv_1)
    ppci_2["bus"][bus_idx, IKSSV] = abs(ikssv_2)

    ppci_0["bus"][bus_idx, PHI_IKSSV_DEGREE] = np.angle(ikssv_0, deg=True)
    ppci_1["bus"][bus_idx, PHI_IKSSV_DEGREE] = np.angle(ikssv_1, deg=True)
    ppci_2["bus"][bus_idx, PHI_IKSSV_DEGREE] = np.angle(ikssv_2, deg=True)

    if net._options["branch_results"]:
        # calculate grid voltages based on ikssv component
        Zbus_0 = ppci_0["internal"]["Zbus"]
        Zbus_1 = ppci_1["internal"]["Zbus"]
        Zbus_2 = ppci_2["internal"]["Zbus"]

        V_ikss_0 = 0 - ikssv_0 * Zbus_0[:, bus_idx]  # initial value for zero-sequence voltage is 0
        V_ikss_1 = V0 - ikssv_1 * Zbus_1[:, bus_idx] # initial value for positive-sequence voltage is V0
        V_ikss_2 = 0 - ikssv_2 * Zbus_2[:, bus_idx]  # initial value for negative-sequence voltage is 0

        # Add voltage information to ppci:
        ppci_0["internal"]["V_ikss"] = V_ikss_0
        ppci_1["internal"]["V_ikss"] = V_ikss_1
        ppci_2["internal"]["V_ikss"] = V_ikss_2



def _calc_ikssc(net, ppci_0, ppci_1, ppci_2, bus_idx):
    # Fault current contribution ikssc from current sources (e.g., sgens)
    _current_source_current(net, ppci_0, bus_idx, 0)
    _current_source_current(net, ppci_1, bus_idx, 1)
    _current_source_current(net, ppci_2, bus_idx, 2)



def _current_source_current(net, ppci, bus_idx, sequence=1):
    case = net._options["case"]
    ppci["bus"][:, IKCV] = 0
    ppci["bus"][:, PHI_IKCV_DEGREE] = 0
    type_c = net._options["use_pre_fault_voltage"]

    # sgen current source contribution only for Type A and case "max" or type C:
    if case != "max" and not type_c or sequence != 1:
        return

    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    # Collect active sgen and gen entries with current_source=True
    active_sgens = net.sgen[net._is_elements_final["sgen"] & net.sgen.current_source]
    # consider generators with current_source type
    active_gens = net.gen[net._is_elements_final["gen"] & net.gen.current_source]
    # Combine active sgens and gens into one DataFrame
    combined_sources = pd.concat([active_sgens, active_gens], ignore_index=True)

    if len(combined_sources) == 0:
        return
    if any(pd.isnull(combined_sources.sn_mva)):
        raise ValueError("sn_mva needs to be specified for all sgens in net.sgen.sn_mva,"
                         " also check net.gen.sn_mva if you are using current_source generators")
    if any(pd.isnull(combined_sources.current_source)):
        raise ValueError("current_source needs to be specified for all sgens in net.sgen.current_source,"
                         " also check net.gen.current_source if you are using current_source generators")
    if "current_angle_degree" in combined_sources.columns:
        sgen_angle = np.deg2rad(combined_sources.current_angle_degree.values)
    else:
        sgen_angle = None

    sgen_buses = combined_sources.bus.values
    sgen_buses_ppc = bus_lookup[sgen_buses]

    if "k" not in combined_sources:
        raise ValueError("Nominal to short-circuit current has to specified in net.sgen.k also check net.gen.k"
                         " if you are using current_source generators")
    if type_c and "kappa" not in combined_sources:
        raise ValueError("Max. short-circuit current in p.u. must be specified in net.sgen.kappa for the "
                         "short-circuit calculation with superposition method")

    if type_c:
        # voltage difference between pre-fault condition and the voltage from the calculation with
        # rotating machines only is used to determine the current injection. The parameter kappa here
        # is used to denote the maximum current contribution of the sgen in the short-circuit case "Type C"
        V_ikss = ppci["internal"]["V_ikss"]
        i_sgen_n_pu = (combined_sources.sn_mva.values.reshape(-1, 1) / net.sn_mva)
        delta_V = ppci["bus"][sgen_buses_ppc][:, [VM]] - np.abs(V_ikss[sgen_buses_ppc])
        i_sgen_pu = np.where(combined_sources.k.values.reshape(-1, 1) * delta_V < combined_sources.kappa.values.reshape(-1, 1),
                             combined_sources.k.values.reshape(-1, 1) * i_sgen_n_pu * delta_V,
                             i_sgen_n_pu * combined_sources.kappa.values.reshape(-1, 1))
        i_sgen_pu = np.abs(i_sgen_pu)
    else:
        # short-circuit contribution from sgen nominal and active current method
        i_sgen_pu = np.where(combined_sources.active_current.values,
                             (combined_sources.p_mw.values * combined_sources.scaling.values / net.sn_mva * combined_sources.k.values),
                             (combined_sources.sn_mva.values * combined_sources.scaling.values / net.sn_mva * combined_sources.k.values))

    if sgen_angle is not None:  # check logic here of type_c
        i_sgen_pu = i_sgen_pu * np.exp(sgen_angle * 1j)

    buses, ikcv_pu, _ = _sum_by_group(sgen_buses_ppc, i_sgen_pu, i_sgen_pu)

    if type_c:
        extra_angle = ppci["bus"][buses, VA]
    else:
        extra_angle = 0

    ikcv_pu = ikcv_pu.flatten()
    ppci["bus"][buses, [IKCV]] = ikcv_pu if sgen_angle is None else np.abs(ikcv_pu)
    if sgen_angle is not None:  # check logic here of type_c
        ppci["bus"][buses, PHI_IKCV_DEGREE] = np.angle(ikcv_pu, deg=True)

    Zbus = ppci["internal"]["Zbus"]
    if sgen_angle is None:  # check logic here of type_c
        ppci["bus"][buses, PHI_IKCV_DEGREE] = -np.angle(Zbus[buses, bus_idx], deg=True) + extra_angle



def _calc_branch_currents_complex(net, bus_idx, ppci0, ppci1, ppci2, sequence, z_fault_pu):
    if sequence == 0:
        ppci = ppci0
    elif sequence == 1:
        ppci = ppci1
    elif sequence == 2:
        ppci = ppci2
    fault = net._options["fault"]
    n_sc_bus = bus_idx.size
    b = bus_idx

    baseI = ppci["internal"]["baseI"]
    fb = np.real(ppci["branch"][:, 0]).astype(np.int64)
    tb = np.real(ppci["branch"][:, 1]).astype(np.int64)
    ppci["internal"]["baseV"] = ppci["bus"][:, BASE_KV]

    # add current source branch current if there is one
    current_sources = any(~np.isnan(ppci1["bus"][:, IKCV])) and np.any(ppci1["bus"][:, IKCV] != 0)
    if current_sources:
        ikcv = ppci1["bus"][:, IKCV] * np.exp(np.deg2rad(ppci1["bus"][:, PHI_IKCV_DEGREE]) * 1j)
        sgen_bus = ikcv != 0

        current = np.tile(ikcv, (n_sc_bus, 1))
        # for b in bus_idx:
        current_sources_on_seq = any(~np.isnan(ppci["bus"][:, IKCV])) and np.any(ppci["bus"][:, IKCV] != 0)
        if ~current_sources_on_seq:
            current[:, sgen_bus] = 0

        Zbus0 = ppci0["internal"]["Zbus"]
        Zbus1 = ppci1["internal"]["Zbus"]
        Zbus2 = ppci2["internal"]["Zbus"]
        isgen = np.array([0])

        if fault == "LLL":
            if sequence == 1:
                isgen = ikcv[sgen_bus] * Zbus1[sgen_bus, b] / (Zbus1[b, b] + z_fault_pu)

        elif fault == "LG":
            isgen = ikcv[sgen_bus] * Zbus1[sgen_bus, b] / (
                        Zbus0[b, b] + Zbus1[b, b] + Zbus2[b, b] + 3 * z_fault_pu)

        elif fault == "LL":
            if sequence == 1:
                isgen = ikcv[sgen_bus] * Zbus1[sgen_bus, b] / (Zbus1[b, b] + Zbus2[b, b] + z_fault_pu)
            elif sequence == 2:
                isgen = - ikcv[sgen_bus] * Zbus1[sgen_bus, b] / (Zbus1[b, b] + Zbus2[b, b] + z_fault_pu)

        elif fault == "LLG":
            i02 = ikcv[sgen_bus] * Zbus1[sgen_bus, b] / (Zbus1[b, b] + z_fault_pu + \
                                                            ((Zbus2[b, b] + z_fault_pu) * (
                                                                        Zbus0[b, b] + z_fault_pu)) / (
                                                                        Zbus2[b, b] + Zbus0[
                                                                    b, b] + 2 * z_fault_pu))
            if sequence == 0:
                isgen = - i02 * (Zbus2[b, b] + z_fault_pu) / (Zbus2[b, b] + Zbus0[b, b] + 2 * z_fault_pu)
            if sequence == 1:
                isgen = i02
            elif sequence == 2:
                isgen = - i02 * (Zbus0[b, b] + z_fault_pu) / (Zbus2[b, b] + Zbus0[b, b] + 2 * z_fault_pu)

        if net._options["branch_results"]:
            current[:, b] -= sum(isgen)
        else:
            itot = sum(isgen)
            ppci["bus"][b, IKSSC] = abs(itot)
            ppci["bus"][b, PHI_IKSSC_DEGREE] = np.angle(itot, deg=True)

        # calculate voltage source branch current
        if net._options["branch_results"]:
            Ybus = ppci["internal"]["Ybus"]
            Yf = ppci["internal"]["Yf"]
            Yt = ppci["internal"]["Yt"]
            V_ikss = ppci["internal"]["V_ikss"]

            ikssv_all_f = Yf.dot(V_ikss)
            ikssv_all_t = Yt.dot(V_ikss)

            Zbus = ppci["internal"]["Zbus"]
            V = np.dot(Zbus, (current).T)
            V_ikss += V  # superposition

            ikssc_all = Ybus.dot(V)
            ppci["bus"][:, IKSSC] = np.abs(ikssc_all).ravel()
            ppci["bus"][:, PHI_IKSSC_DEGREE] = np.angle(-ikssc_all, deg=True).ravel()

            ikssc_all_f = Yf.dot(V)
            ikssc_all_t = Yt.dot(V)
            ikss_all_f = ikssv_all_f.ravel() + ikssc_all_f.ravel()
            ikss_all_t = ikssv_all_t.ravel() + ikssc_all_t.ravel()
    else:
        if net._options["branch_results"]:
            Yf = ppci["internal"]["Yf"]
            Yt = ppci["internal"]["Yt"]
            V_ikss = ppci["internal"]["V_ikss"]

            ikss_all_f = Yf.dot(V_ikss)
            ikss_all_t = Yt.dot(V_ikss)    

    if net._options["branch_results"]:
        # Bus results
        ppci["bus"][:, VM] = np.abs(V_ikss).ravel()
        ppci["bus"][:, VA] = np.angle(V_ikss, deg=True).ravel()
        
        # Branch results
        ppci["branch"][:, IKSS_F] = np.abs(ikss_all_f).ravel()
        ppci["branch"][:, IKSS_T] = np.abs(ikss_all_t).ravel()
        ppci["branch"][:, IKSS_ANGLE_F] = np.angle(ikss_all_f, deg=True).ravel()
        ppci["branch"][:, IKSS_ANGLE_T] = np.angle(ikss_all_t, deg=True).ravel()
        ppci["branch"][:, VKSS_MAGN_F] = np.abs(V_ikss[fb]).ravel()
        ppci["branch"][:, VKSS_MAGN_T] = np.abs(V_ikss[tb]).ravel()
        ppci["branch"][:, VKSS_ANGLE_F] = np.angle(V_ikss[fb], deg=True).ravel()
        ppci["branch"][:, VKSS_ANGLE_T] = np.angle(V_ikss[tb], deg=True).ravel()

    # else:
    #     # ikss_all_f[np.abs(ikss_all_f) < 1e-10] = np.nan
    #     # ikss_all_t[np.abs(ikss_all_t) < 1e-10] = np.nan
    #     ppci["branch"][:, IKSS_F] = np.abs(minmax_ikss_all_f) * baseI[fb]
    #     ppci["branch"][:, IKSS_ANGLE_F] = np.angle(minmax_ikss_all_f, deg=True)
    #     ppci["branch"][:, IKSS_T] = np.abs(minmax_ikss_all_t) * baseI[tb]
    #     ppci["branch"][:, IKSS_ANGLE_T] = np.angle(minmax_ikss_all_t, deg=True)

    #     if valid_V:

    #         if (fault == "LLL") or (fault == "LG") or (fault == "LLG"):
    #             sn_mva = ppci["baseMVA"]
    #         elif fault == "LL":
    #             sn_mva = ppci["baseMVA"] / 3

    #         ppci["branch"][:, VKSS_MAGN_F] = np.abs(minmax_vkss_all_f)
    #         ppci["branch"][:, VKSS_MAGN_T] = np.abs(minmax_vkss_all_t)

    #         ppci["branch"][:, VKSS_ANGLE_F] = np.angle(minmax_vkss_all_f, deg=True)
    #         ppci["branch"][:, VKSS_ANGLE_T] = np.angle(minmax_vkss_all_t, deg=True)

    #         ppci["branch"][:, PKSS_F] = np.nan_to_num(minmax(pkss_all_f, axis=1)) * sn_mva
    #         ppci["branch"][:, QKSS_F] = np.nan_to_num(minmax(qkss_all_f, axis=1)) * sn_mva

    #         ppci["branch"][:, PKSS_T] = np.nan_to_num(minmax(pkss_all_t, axis=1)) * sn_mva
    #         ppci["branch"][:, QKSS_T] = np.nan_to_num(minmax(qkss_all_t, axis=1)) * sn_mva

    if net._options["ip"]:  # LL detailed calculations not included!!!
        kappa = ppci["bus"][:, KAPPA]
        if current_sources:
            ip_all_f = np.sqrt(2) * (ikssv_all_f * kappa[bus_idx] + ikssc_all_f)
            ip_all_t = np.sqrt(2) * (ikssv_all_t * kappa[bus_idx] + ikssc_all_t)
        else:
            ip_all_f = np.sqrt(2) * ikssv_all_f * kappa[bus_idx]
            ip_all_t = np.sqrt(2) * ikssv_all_t * kappa[bus_idx]

        if net._options["return_all_currents"]:
            ppci["internal"]["branch_ip_f"] = np.nan_to_num(np.abs(ip_all_f)) * baseI[fb, None]
            ppci["internal"]["branch_ip_t"] = np.nan_to_num(np.abs(ip_all_t)) * baseI[tb, None]
        else:
            ip_all_f[np.abs(ip_all_f) < 1e-10] = np.nan
            ip_all_t[np.abs(ip_all_t) < 1e-10] = np.nan
            # ppci["branch"][:, IP_F] = minmax(np.abs(ip_all_f), axis=1) * baseI[fb]
            # ppci["branch"][:, IP_T] = minmax(np.abs(ip_all_t), axis=1) * baseI[tb]
            ppci["branch"][:, IP_F] = np.nanmax(np.abs(ip_all_f), axis=1) * baseI[fb]
            ppci["branch"][:, IP_T] = np.nanmax(np.abs(ip_all_t), axis=1) * baseI[tb]

    if net._options["ith"]:  # LL detailed calculations not included!!!
        n = 1
        m = ppci["bus"][bus_idx, M]
        ith_all_f = np.abs(ikss_all_f * np.sqrt(m + n))
        ith_all_t = np.abs(ikss_all_t * np.sqrt(m + n))

        if net._options["return_all_currents"]:
            ppci["internal"]["branch_ith_f"] = np.nan_to_num(np.abs(ith_all_f)) * baseI[fb, None]
            ppci["internal"]["branch_ith_t"] = np.nan_to_num(np.abs(ith_all_t)) * baseI[tb, None]
        else:
            ith_all_f[np.abs(ith_all_f) < 1e-10] = np.nan
            ith_all_t[np.abs(ith_all_t) < 1e-10] = np.nan
            # ppci["branch"][:, ITH_F] = minmax(np.abs(ith_all_f), axis=1) * baseI[fb]
            # ppci["branch"][:, ITH_T] = minmax(np.abs(ith_all_t), axis=1) * baseI[fb]
            ppci["branch"][:, ITH_F] = np.nanmax(np.abs(ith_all_f), axis=1) * baseI[fb]
            ppci["branch"][:, ITH_T] = np.nanmax(np.abs(ith_all_t), axis=1) * baseI[fb]

    # Update bus index for branch results
    if net._options["branch_results"]:
        ppci["internal"]["br_res_ks_ppci_bus"] = bus_idx

    return ppci


def _calc_ip(net, ppci):
    ip = np.sqrt(2) * (ppci["bus"][:, KAPPA] * ppci["bus"][:, IKSSV] + ppci["bus"][:, IKSSC])
    ppci["bus"][:, IP] = ip


def _calc_ith(net, ppci):
    tk_s = net["_options"]["tk_s"]
    kappa = ppci["bus"][:, KAPPA]
    f = 50
    n = 1
    m = (np.exp(4 * f * tk_s * np.log(kappa - 1)) - 1) / (2 * f * tk_s * np.log(kappa - 1))
    m[np.where(kappa > 1.99)] = 0
    ppci["bus"][:, M] = m
    ith = (ppci["bus"][:, IKSSV] + ppci["bus"][:, IKSSC]) * np.sqrt(m + n)
    ppci["bus"][:, ITH] = ith


# TODO: Ib for generation close bus
# def _calc_ib_generator(net, ppci):
#     # Zbus = ppci["internal"]["Zbus"]
#     # baseI = ppci["internal"]["baseI"]
#     tk_s = net._options['tk_s']
#     c = 1.1

#     z_equiv = ppci["bus"][:, R_EQUIV] + ppci["bus"][:, X_EQUIV] * 1j
#     I_ikss = c / z_equiv / ppci["bus"][:, BASE_KV] / np.sqrt(3) * ppci["baseMVA"]

#     # calculate voltage source branch current
#     # I_ikss = ppci["bus"][:, IKSSV]
#     # V_ikss = (I_ikss / baseI) * Zbus

#     gen = net["gen"][net._is_elements["gen"]]
#     gen_vn_kv = gen.vn_kv.values

#     # Check difference ext_grid and gen
#     gen_buses = ppci['gen'][:, GEN_BUS].astype(np.int64)
#     gen_mbase = ppci['gen'][:, MBASE]
#     gen_i_rg = gen_mbase / (np.sqrt(3) * gen_vn_kv)

#     gen_buses_ppc, gen_sn_mva, I_rG = _sum_by_group(gen_buses, gen_mbase, gen_i_rg)

#     # shunt admittance of generator buses and generator short circuit current
#     # YS = ppci["bus"][gen_buses_ppc, GS] + ppci["bus"][gen_buses_ppc, BS] * 1j
#     # I_kG = V_ikss.T[:, gen_buses_ppc] * YS * baseI[gen_buses_ppc]

#     xdss_pu = gen.xdss_pu.values
#     rdss_pu = gen.rdss_pu.values
#     cosphi = gen.cos_phi.values
#     X_dsss = xdss_pu * np.square(gen_vn_kv) / gen_mbase
#     R_dsss = rdss_pu * np.square(gen_vn_kv) / gen_mbase

#     K_G = ppci['bus'][gen_buses, BASE_KV] / gen_vn_kv * c / (1 + xdss_pu * np.sin(np.arccos(cosphi)))
#     Z_G = (R_dsss + 1j * X_dsss)

#     I_kG = c * ppci['bus'][gen_buses, BASE_KV] / np.sqrt(3) / (Z_G * K_G) * ppci["baseMVA"]

#     dV_G = 1j * X_dsss * K_G * I_kG
#     V_Is = c * ppci['bus'][gen_buses, BASE_KV] / np.sqrt(3)

#     # I_kG_contribution = I_kG.sum(axis=1)
#     # ratio_SG_ikss = I_kG_contribution / I_ikss
#     # close_to_SG = ratio_SG_ikss > 5e-2

#     close_to_SG = I_kG / I_rG > 2

#     if tk_s == 2e-2:
#         mu = 0.84 + 0.26 * np.exp(-0.26 * abs(I_kG) / I_rG)
#     elif tk_s == 5e-2:
#         mu = 0.71 + 0.51 * np.exp(-0.3 * abs(I_kG) / I_rG)
#     elif tk_s == 10e-2:
#         mu = 0.62 + 0.72 * np.exp(-0.32 * abs(I_kG) / I_rG)
#     elif tk_s >= 25e-2:
#         mu = 0.56 + 0.94 * np.exp(-0.38 * abs(I_kG) / I_rG)
#     else:
#         raise UserWarning('not implemented for other tk_s than 20ms, 50ms, 100ms and >=250ms')

#     mu = np.clip(mu, 0, 1)

#     I_ikss_G = abs(I_ikss - np.sum((1 - mu) * I_kG, axis=1))

#     # I_ikss_G = I_ikss - np.sum(abs(V_ikss.T[:, gen_buses_ppc]) * (1-mu) * I_kG, axis=1)

#     I_ikss_G = abs(I_ikss - np.sum(dV_G / V_Is * (1 - mu) * I_kG, axis=1))

#     return I_ikss_G

def nan_minmax(a, rows, argminmax):
    # because numpy won't sort complex values by magnitude :(
    rows_allnan = np.isnan(a[rows, :]).all(axis=1)
    aa = a[rows, :]
    minmax_a = np.zeros(a.shape[0], dtype=np.complex128)
    minmax_a[~rows_allnan] = aa[~rows_allnan, argminmax(np.abs(aa[~rows_allnan, :]), axis=1)]
    return minmax_a
