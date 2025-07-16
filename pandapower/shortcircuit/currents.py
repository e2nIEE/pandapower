# -*- coding: utf-8 -*-
# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
import warnings
import copy
import numpy as np
import pandas as pd
from copy import deepcopy
from pandapower.auxiliary import _sum_by_group
from pandapower.pypower.idx_bus import BASE_KV, VM, VA
from pandapower.pypower.idx_brch import TAP
from pandapower.pypower.idx_brch_sc import IKSS_F, IKSS_T, IP_F, IP_T, ITH_F, ITH_T, \
    PKSS_F, QKSS_F, PKSS_T, QKSS_T, VKSS_MAGN_F, VKSS_MAGN_T, VKSS_ANGLE_F, VKSS_ANGLE_T, IKSS_ANGLE_F, IKSS_ANGLE_T
from pandapower.pypower.idx_bus_sc import C_MIN, C_MAX, KAPPA, R_EQUIV, IKSSV, IP, ITH, \
    X_EQUIV, IKSSC, IKCV, M, R_EQUIV_OHM, X_EQUIV_OHM, V_G, K_SG, SKSS, \
    PHI_IKSSV_DEGREE, PHI_IKSSC_DEGREE, PHI_IKCV_DEGREE
from pandapower.shortcircuit.impedance import _calc_zbus_diag


import logging

logger = logging.getLogger(__name__)


# def _calc_ikss(net, ppci, bus_idx):
#     fault = net._options["fault"]
#     case = net._options["case"]
#     baseI = ppci["internal"]["baseI"] = ppci["bus"][:, BASE_KV] * np.sqrt(3) / ppci["baseMVA"]

#     # Only for test, should correspondant to PF result
#     baseZ = ppci["bus"][bus_idx, BASE_KV] ** 2 / ppci["baseMVA"]
#     ppci["bus"][bus_idx, R_EQUIV_OHM] = baseZ * ppci["bus"][bus_idx, R_EQUIV]
#     ppci["bus"][bus_idx, X_EQUIV_OHM] = baseZ * ppci["bus"][bus_idx, X_EQUIV]

#     # init V0 for the superposition method
#     # todo: explain the background better
#     n_sc_bus = np.shape(bus_idx)[0]
#     n_bus = ppci["bus"].shape[0]

#     valid_V = True
#     if net._options.get("use_pre_fault_voltage", False):
#         V0 = np.full((n_bus, n_sc_bus), ppci["bus"][:, [VM]] * np.exp(np.deg2rad(ppci["bus"][:, [VA]]) * 1j))
#     else:
#         V0 = np.full((n_bus, n_sc_bus), ppci["bus"][bus_idx, C_MAX if case == "max" else C_MIN], dtype=np.complex128)

#         if np.any(ppci["branch"][:, TAP] != 1):
#             msg = "Calculation does not support calculation of voltages and branch powers for grids that have" \
#                   " transformers with rated voltages unequal to bus voltages. Change the transformer data or" \
#                   "try using the superposition method by passing 'use_pre_fault_voltage=True'"
#             # raise NotImplementedError(msg)
#             warnings.warn(msg)
#             valid_V = False
#     ppci["internal"]["valid_V"] = valid_V

#     z_equiv = ppci["bus"][bus_idx, R_EQUIV] + ppci["bus"][bus_idx, X_EQUIV] * 1j  # removed the abs()
#     if fault == "LLL":
#         ikssv = V0[bus_idx, np.arange(n_sc_bus)] / z_equiv
#     elif fault == "LL":
#         # z_equiv = z_equiv - fault_impedance / 2
#         # Now only BC fault, to be changed to emulate fault between different pairs of phases
#         ikssv = V0[bus_idx, np.arange(n_sc_bus)] / (z_equiv + z_equiv)
#         # V0b = Vbus * np.exp(np.deg2rad(-120) * 1j)
#         # V0c = Vbus * np.exp(np.deg2rad(120) * 1j)
#         # ikssv = (V0b - V0c) / z_equiv / 2

#     if fault == "LLL":
#         if net["_options"]["inverse_y"]:
#             Zbus = ppci["internal"]["Zbus"]
#             # I don't know how to do this without reshape
#             V_ikss = V0 - ikssv * Zbus[:, bus_idx] if valid_V else -ikssv * Zbus[:, bus_idx]
#             V_ikss[np.abs(V_ikss) < 1e-10] = 0
#         else:
#             ybus_fact = ppci["internal"]["ybus_fact"]
#             V_ikss = np.zeros((n_bus, n_sc_bus), dtype=np.complex128)
#             for ix, b in enumerate(bus_idx):
#                 ikss = np.zeros((n_bus, 1), dtype=np.complex128)
#                 ikss[b] = ikssv[ix]
#                 V_ikss[:, [ix]] = V0[:, [ix]] - ybus_fact(ikss) if valid_V else -ybus_fact(ikss)
#                 V_ikss[np.abs(V_ikss) < 1e-10] = 0
#     elif fault == "LL":
#         if net["_options"]["inverse_y"]:
#             Zbus = ppci["internal"]["Zbus"]
#             # I don't know how to do this without reshape
#             V_ikss = V0 - ikssv * Zbus[:, bus_idx] if valid_V else -ikssv * Zbus[:, bus_idx]
#             # V_ikss_bis = V0c + ikssv * Zbus[:, bus_idx] if valid_V else +ikssv * Zbus[:, bus_idx]
#             # V_ikss[np.abs(V_ikss) < 1e-10] = 0
#             # V_ikss_bis[np.abs(V_ikss_bis) < 1e-10] = 0
#         else:
#             pass # NOTE: not implemented yet for this other option

#     # ikssv = -Ybus.dot(V_ikss) / baseI.reshape(-1, 1)
#     # ikssv = ikssv[bus_idx]
#     # ikssv = c / z_equiv / ppci["bus"][bus_idx, BASE_KV] / np.sqrt(3) * ppci["baseMVA"]
#     # ikssv = c / z_equiv / baseI[bus_idx]  # should be same as above
#     ikssv /= baseI[bus_idx]
#     # added abs here:
#     ppci["bus"][bus_idx, IKSSV] = abs(ikssv)
#     # added angle calculation in degree:
#     ppci["bus"][bus_idx, PHI_IKSSV_DEGREE] = np.angle(ikssv, deg=True)
#     ppci["internal"]["V_ikss"] = V_ikss
#     # if fault == "LL":
#     #     ppci["internal"]["V_ikss_bis"] = V_ikss_bis
#     #     z_equiv = z_equiv - fault_impedance / 2
#     #     ppci["bus"][bus_idx, IKSSV] = np.abs(c / z_equiv / ppci["bus"][bus_idx, BASE_KV] / 2 * ppci["baseMVA"])

#     _current_source_current(net, ppci, bus_idx)

#     ikss = ppci["bus"][bus_idx, IKSSV] + ppci["bus"][bus_idx, IKCV]
#     if fault == "LLL":
#         ppci["bus"][bus_idx, SKSS] = np.sqrt(3) * ikss * ppci["bus"][bus_idx, BASE_KV]
#     elif fault == "LL":
#         ppci["bus"][bus_idx, SKSS] = ikss * ppci["bus"][bus_idx, BASE_KV] / np.sqrt(3)

#     # Correct voltage of generator bus inside power station  
#     if np.any(~np.isnan(ppci["bus"][:, K_SG])):
#         gen_bus_idx = bus_idx[~np.isnan(ppci["bus"][bus_idx, K_SG])]
#         ppci["bus"][gen_bus_idx, IKSSV] *=\
#             (ppci["bus"][gen_bus_idx, V_G] / ppci["bus"][gen_bus_idx, BASE_KV])
#         ppci["bus"][gen_bus_idx, SKSS] *=\
#             (ppci["bus"][gen_bus_idx, V_G] / ppci["bus"][gen_bus_idx, BASE_KV])


def _calc_ikss_to_g(net, ppci_0, ppci_1, ppci_2, bus_idx):
    # TODO parameter to tell which Phase is affected
    fault = net._options["fault"]
    case = net._options["case"]
    fault_impedance = net._options["fault_impedance"]

    ppci_0["internal"]["baseI"] = ppci_0["bus"][:, BASE_KV] * np.sqrt(3) / ppci_0["baseMVA"]
    ppci_1["internal"]["baseI"] = ppci_1["bus"][:, BASE_KV] * np.sqrt(3) / ppci_1["baseMVA"]
    ppci_2["internal"]["baseI"] = ppci_2["bus"][:, BASE_KV] * np.sqrt(3) / ppci_2["baseMVA"]

    z_equiv_0 = ppci_0["bus"][bus_idx, R_EQUIV] + ppci_0["bus"][bus_idx, X_EQUIV] * 1j
    z_equiv_1 = ppci_1["bus"][bus_idx, R_EQUIV] + ppci_1["bus"][bus_idx, X_EQUIV] * 1j
    z_equiv_2 = ppci_2["bus"][bus_idx, R_EQUIV] + ppci_2["bus"][bus_idx, X_EQUIV] * 1j

    # Only for test, should correspondant to PF result
    baseZ = ppci_1["bus"][bus_idx, BASE_KV] ** 2 / ppci_1["baseMVA"]
    ppci_0["bus"][bus_idx, R_EQUIV_OHM] = baseZ * ppci_0['bus'][bus_idx, R_EQUIV]
    ppci_0["bus"][bus_idx, X_EQUIV_OHM] = baseZ * ppci_0['bus'][bus_idx, X_EQUIV]
    ppci_1["bus"][bus_idx, R_EQUIV_OHM] = baseZ * ppci_1['bus'][bus_idx, R_EQUIV]
    ppci_1["bus"][bus_idx, X_EQUIV_OHM] = baseZ * ppci_1['bus'][bus_idx, X_EQUIV]
    ppci_2["bus"][bus_idx, R_EQUIV_OHM] = baseZ * ppci_2['bus'][bus_idx, R_EQUIV]
    ppci_2["bus"][bus_idx, X_EQUIV_OHM] = baseZ * ppci_2['bus'][bus_idx, X_EQUIV]

    n_sc_bus = np.shape(bus_idx)[0]
    n_bus = ppci_1["bus"].shape[0]

    valid_V = True
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
            valid_V = False

    ppci_0["internal"]["valid_V"] = valid_V
    ppci_1["internal"]["valid_V"] = valid_V
    ppci_2["internal"]["valid_V"] = valid_V

    if fault == "LLL": 
        ikssv_1 = V0[bus_idx, np.arange(n_sc_bus)] / z_equiv_1
        ikssv_2 = 0
        ikssv_0 = 0
    elif fault == "LG":
        ikssv_1 = V0[bus_idx, np.arange(n_sc_bus)] / (z_equiv_0 + z_equiv_1 + z_equiv_2)
        ikssv_2 = copy.deepcopy(ikssv_1)
        ikssv_0 = copy.deepcopy(ikssv_1)
    elif fault == "LLG":
        ikssv_1 = V0[bus_idx, np.arange(n_sc_bus)] / (z_equiv_1+((z_equiv_2*z_equiv_0)/(z_equiv_2+z_equiv_0)))
        ikssv_2 = -ikssv_1*z_equiv_0/(z_equiv_0+z_equiv_2)
        ikssv_0 = -ikssv_1*z_equiv_2/(z_equiv_0+z_equiv_2)
    elif fault == "LL":
        ikssv_1 = V0[bus_idx, np.arange(n_sc_bus)] / (z_equiv_1+z_equiv_2)
        ikssv_2 = -ikssv_1
        ikssv_0 = 0

    if net["_options"]["inverse_y"]:
        Zbus_0 = ppci_0["internal"]["Zbus"]
        Zbus_1 = ppci_1["internal"]["Zbus"]
        Zbus_2 = ppci_2["internal"]["Zbus"]

        V_ikss_0 = 0 - ikssv_0 * Zbus_0[:, bus_idx]  # initial value for zero-sequence voltage is 0
        V_ikss_1 = V0 - ikssv_1 * Zbus_1[:, bus_idx] if valid_V else 0 - ikssv_1 * Zbus_1[:, bus_idx]
        V_ikss_2 = 0 - ikssv_2 * Zbus_2[:, bus_idx]  # initial value for negative-sequence voltage is 0

    else:
        ybus_fact_0 = ppci_0["internal"]["ybus_fact"]
        ybus_fact_1 = ppci_1["internal"]["ybus_fact"]
        ybus_fact_2 = ppci_2["internal"]["ybus_fact"]
        V_ikss_0 = np.zeros((n_bus, n_sc_bus), dtype=np.complex128)
        V_ikss_1 = np.zeros((n_bus, n_sc_bus), dtype=np.complex128)
        V_ikss_2 = np.zeros((n_bus, n_sc_bus), dtype=np.complex128)
        for ix, b in enumerate(bus_idx):
            ikss = np.zeros((n_bus, 1), dtype=np.complex128)
            ikss[b] = ikssv_1[ix]
            # TODO for LLG fault ikks0 and ikss2 need to be respected?:
            V_ikss_0[:, [ix]] = 0 - ybus_fact_0(ikss) 
            V_ikss_1[:, [ix]] = V0[:, [ix]] - ybus_fact_1(ikss) if valid_V else 0 - ybus_fact_1(ikss) 
            V_ikss_2[:, [ix]] = 0 - ybus_fact_2(ikss)

    V_ikss_0[np.abs(V_ikss_0) < 1e-10] = 0
    V_ikss_1[np.abs(V_ikss_1) < 1e-10] = 0
    V_ikss_2[np.abs(V_ikss_2) < 1e-10] = 0

    # added abs here:
    if fault == "LL":
        ppci_0["bus"][bus_idx, IKSSV] = np.sqrt(3) * abs(ikssv_0 / ppci_0["internal"]["baseI"][bus_idx])
        ppci_1["bus"][bus_idx, IKSSV] = np.sqrt(3) * abs(ikssv_1 / ppci_1["internal"]["baseI"][bus_idx])
        ppci_2["bus"][bus_idx, IKSSV] = np.sqrt(3) * abs(ikssv_2 / ppci_2["internal"]["baseI"][bus_idx])

    elif fault == "LG":
        ppci_0["bus"][bus_idx, IKSSV] = 3 * abs(ikssv_0 / ppci_0["internal"]["baseI"][bus_idx])
        ppci_1["bus"][bus_idx, IKSSV] = 3 * abs(ikssv_1 / ppci_1["internal"]["baseI"][bus_idx])
        ppci_2["bus"][bus_idx, IKSSV] = 3 * abs(ikssv_2 / ppci_2["internal"]["baseI"][bus_idx])

    else:
        ppci_0["bus"][bus_idx, IKSSV] = abs(ikssv_0 / ppci_0["internal"]["baseI"][bus_idx])
        ppci_1["bus"][bus_idx, IKSSV] = abs(ikssv_1 / ppci_1["internal"]["baseI"][bus_idx])
        ppci_2["bus"][bus_idx, IKSSV] = abs(ikssv_2 / ppci_2["internal"]["baseI"][bus_idx])

    # added angle calculation in degree:
    ppci_0["bus"][bus_idx, PHI_IKSSV_DEGREE] = np.angle(ikssv_0, deg=True)
    ppci_1["bus"][bus_idx, PHI_IKSSV_DEGREE] = np.angle(ikssv_1, deg=True)
    ppci_2["bus"][bus_idx, PHI_IKSSV_DEGREE] = np.angle(ikssv_2, deg=True)

    # V_ikss_abc_pu = sequence_to_phase(np.vstack([V_ikss_0.T, V_ikss_1.T, V_ikss_2.T]))
    # V_ikss_abc_pu[np.abs(V_ikss_abc_pu) < 1e-10] = 0
    #
    # # add voltage information to ppci:
    # ppci_0["internal"]["V_ikss"] = V_ikss_abc_pu.T[:, [0]]
    # ppci_1["internal"]["V_ikss"] = V_ikss_abc_pu.T[:, [1]]
    # ppci_2["internal"]["V_ikss"] = V_ikss_abc_pu.T[:, [2]]

    _current_source_current(net, ppci_0, bus_idx, 0)
    _current_source_current(net, ppci_1, bus_idx, 1)
    _current_source_current(net, ppci_2, bus_idx, 2)

    ikssc_1 = ppci_1["bus"][bus_idx, IKSSC]

    # calculate correct ikssc for bus results
    if fault == "LL":
        ppci_0["bus"][bus_idx, IKSSC] = 0
        ppci_1["bus"][bus_idx, IKSSC] = np.sqrt(3)/2 * ikssc_1
        ppci_2["bus"][bus_idx, IKSSC] = 0
    elif fault == "LG":
        factor = abs(3 * z_equiv_1 / (z_equiv_1 + z_equiv_2 + z_equiv_0))
        ppci_0["bus"][bus_idx, IKSSC] = factor * ikssc_1
        ppci_1["bus"][bus_idx, IKSSC] = factor * ikssc_1
        ppci_2["bus"][bus_idx, IKSSC] = factor * ikssc_1
    elif fault == "LLG":
        ppci_0["bus"][bus_idx, IKSSC] = 0
        ppci_1["bus"][bus_idx, IKSSC] = abs((3 * z_equiv_1 / (z_equiv_1 + 2 * z_equiv_0))) * ikssc_1
        ppci_2["bus"][bus_idx, IKSSC] = 0

    # calculate skss
    if fault == "LLL":
        ppci_0["bus"][bus_idx, SKSS] = np.sqrt(3) * (ppci_0["bus"][bus_idx, IKSSV] + ppci_0["bus"][bus_idx, IKSSC]) * ppci_0["bus"][bus_idx, BASE_KV]
        ppci_1["bus"][bus_idx, SKSS] = np.sqrt(3) * (ppci_1["bus"][bus_idx, IKSSV] + ppci_1["bus"][bus_idx, IKSSC]) * ppci_1["bus"][bus_idx, BASE_KV]
        ppci_2["bus"][bus_idx, SKSS] = np.sqrt(3) * (ppci_2["bus"][bus_idx, IKSSV] + ppci_2["bus"][bus_idx, IKSSC]) * ppci_2["bus"][bus_idx, BASE_KV]
    else:
        ppci_0["bus"][bus_idx, SKSS] = (ppci_0["bus"][bus_idx, IKSSV] + ppci_0["bus"][bus_idx, IKSSC]) * ppci_0["bus"][bus_idx, BASE_KV] / np.sqrt(3)
        ppci_1["bus"][bus_idx, SKSS] = (ppci_1["bus"][bus_idx, IKSSV] + ppci_1["bus"][bus_idx, IKSSC]) * ppci_1["bus"][bus_idx, BASE_KV] / np.sqrt(3)
        ppci_2["bus"][bus_idx, SKSS] = (ppci_2["bus"][bus_idx, IKSSV] + ppci_2["bus"][bus_idx, IKSSC]) * ppci_2["bus"][bus_idx, BASE_KV] / np.sqrt(3)

    # add voltage information to ppci:
    ppci_0["internal"]["V_ikss"] = V_ikss_0
    ppci_1["internal"]["V_ikss"] = V_ikss_1
    ppci_2["internal"]["V_ikss"] = V_ikss_2


def _current_source_current(net, ppci, bus_idx, sequence=1):
    case = net._options["case"]
    ppci["bus"][:, IKCV] = 0
    ppci["bus"][:, PHI_IKCV_DEGREE] = -90
    ppci["bus"][:, IKSSC] = 0
    type_c = net._options["use_pre_fault_voltage"]
    fault_impedance = net._options["fault_impedance"]
    if net._options["fault"] == "LL":
        fault_impedance = fault_impedance/2
    # sgen current source contribution only for Type A and case "max" or type C:
    if case != "max" and not type_c or sequence != 1:
        return

    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    fault = net._options["fault"]

    # _is_elements_final exists for some reason, and weirdly it can be different than _is_elements. 
    # it is not documented anywhere why it exists and I don't have any time to find out, but this here fixes the problem.
    if np.all(net.sgen.current_source.values):
        sgen = net.sgen[net._is_elements_final["sgen"]]
    else:
        sgen = net.sgen[net._is_elements_final["sgen"] & net.sgen.current_source]
    if len(sgen) == 0:
        return
    if any(pd.isnull(sgen.sn_mva)):
        raise ValueError("sn_mva needs to be specified for all sgens in net.sgen.sn_mva")
    if "current_angle_degree" in sgen.columns:
        sgen_angle = np.deg2rad(sgen.current_angle_degree.values)
    else:
        sgen_angle = None

    baseI = ppci["internal"]["baseI"]
    sgen_buses = sgen.bus.values
    sgen_buses_ppc = bus_lookup[sgen_buses]

    if "k" not in sgen:
        raise ValueError("Nominal to short-circuit current has to specified in net.sgen.k")
    if type_c and "kappa" not in sgen:
        raise ValueError("Max. short-circuit current in p.u. must be specified in net.sgen.kappa for the "
                         "short-circuit calculation with superposition method")

    if type_c:
        # voltage difference between pre-fault condition and the voltage from the calculation with
        # rotating machines only is used to determine the current injection. The parameter kappa here
        # is used to denote the maximal current contribution of the sgen in the short-circuit case "Type C"
        V_ikss = ppci["internal"]["V_ikss"]
        i_sgen_n_pu = (sgen.sn_mva.values.reshape(-1, 1) / net.sn_mva)
        delta_V = ppci["bus"][sgen_buses_ppc][:, [VM]] - np.abs(V_ikss[sgen_buses_ppc])
        i_sgen_pu = np.where(sgen.k.values.reshape(-1, 1) * delta_V < sgen.kappa.values.reshape(-1, 1),
                             sgen.k.values.reshape(-1, 1) * i_sgen_n_pu * delta_V,
                             i_sgen_n_pu * sgen.kappa.values.reshape(-1, 1))
        i_sgen_pu = np.abs(i_sgen_pu)
    else:
        # i_sgen_pu = (sgen.sn_mva.values / net.sn_mva * sgen.k.values)
        # short-circuit contribution from sgen nominal and active current method
        i_sgen_pu = np.where(sgen.active_current.values,
                            (sgen.p_mw.values * sgen.scaling.values  / net.sn_mva * sgen.k.values),
                            (sgen.sn_mva.values / net.sn_mva * sgen.k.values))

    if sgen_angle is not None: # check logic here of type_c
        i_sgen_pu = i_sgen_pu * np.exp(sgen_angle * 1j)

    buses, ikcv_pu, _ = _sum_by_group(sgen_buses_ppc, i_sgen_pu, i_sgen_pu)

    if type_c :
        extra_angle = ppci["bus"][buses, VA]
    else :
        extra_angle = 0

    ikcv_pu = ikcv_pu.flatten()
    ppci["bus"][buses, [IKCV]] = ikcv_pu if sgen_angle is None else np.abs(ikcv_pu)
    if sgen_angle is not None: # check logic here of type_c
        ppci["bus"][buses, PHI_IKCV_DEGREE] = np.angle(ikcv_pu, deg=True)

    if net["_options"]["inverse_y"]:
        Zbus = ppci["internal"]["Zbus"]
        diagZ = np.diag(Zbus).copy()  # here diagZ is not writeable
        if sgen_angle is None: # check logic here of type_c
            ppci["bus"][buses, PHI_IKCV_DEGREE] = -np.angle(Zbus[buses, bus_idx], deg=True) + extra_angle
        diagZ[bus_idx] += fault_impedance
        i_kss_2 = 1 / diagZ * np.dot(Zbus, ppci["bus"][:, IKCV] * np.exp(np.deg2rad(ppci["bus"][:, PHI_IKCV_DEGREE]) * 1j))
    else:
        ybus_fact = ppci["internal"]["ybus_fact"]   # TO BE UPDATED, does not include modifications done in SCE project
        diagZ = _calc_zbus_diag(net, ppci)
        if sgen_angle is None:
            ppci["bus"][buses, PHI_IKCV_DEGREE] = -np.angle(diagZ[buses], deg=True) + extra_angle
        diagZ[bus_idx] += fault_impedance
        i_kss_2 = ybus_fact(ppci["bus"][:, IKCV] * np.exp(np.deg2rad(ppci["bus"][:, PHI_IKCV_DEGREE]) * 1j)) / diagZ

    ppci["bus"][:, IKSSC] = np.abs(i_kss_2 / baseI)
    # ppci["bus"][:, PHI_IKSSC_DEGREE] = np.angle(i_kss_2, deg=True) if (fault == "LLL" or fault == "LG") else 0
    ppci["bus"][buses, IKCV] /= baseI[buses]


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
#     # V_ikss = (I_ikss * baseI) * Zbus

#     gen = net["gen"][net._is_elements["gen"]]
#     gen_vn_kv = gen.vn_kv.values

#     # Check difference ext_grid and gen
#     gen_buses = ppci['gen'][:, GEN_BUS].astype(np.int64)
#     gen_mbase = ppci['gen'][:, MBASE]
#     gen_i_rg = gen_mbase / (np.sqrt(3) * gen_vn_kv)

#     gen_buses_ppc, gen_sn_mva, I_rG = _sum_by_group(gen_buses, gen_mbase, gen_i_rg)

#     # shunt admittance of generator buses and generator short circuit current
#     # YS = ppci["bus"][gen_buses_ppc, GS] + ppci["bus"][gen_buses_ppc, BS] * 1j
#     # I_kG = V_ikss.T[:, gen_buses_ppc] * YS / baseI[gen_buses_ppc]

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


# def _calc_branch_currents(net, ppci, bus_idx):
#     n_sc_bus = np.shape(bus_idx)[0]

#     case = net._options["case"]
#     minmax = np.nanmin if case == "min" else np.nanmax

#     Yf = ppci["internal"]["Yf"]
#     Yt = ppci["internal"]["Yt"]
#     baseI = ppci["internal"]["baseI"]
#     n_bus = ppci["bus"].shape[0]
#     fb = np.real(ppci["branch"][:, 0]).astype(np.int64)
#     tb = np.real(ppci["branch"][:, 1]).astype(np.int64)
#     ppci["internal"]["baseV"] = ppci["bus"][:, BASE_KV]

#     # calculate voltage source branch current
#     if net["_options"]["inverse_y"]:
#         Zbus = ppci["internal"]["Zbus"]
#         V_ikss = (ppci["bus"][:, IKSSV] * baseI) * Zbus
#         V_ikss = V_ikss[:, bus_idx]
#     else:
#         ybus_fact = ppci["internal"]["ybus_fact"]
#         V_ikss = np.zeros((n_bus, n_sc_bus), dtype=np.complex128)
#         for ix, b in enumerate(bus_idx):
#             ikss = np.zeros(n_bus, dtype=np.complex128)
#             ikss[b] = ppci["bus"][b, IKSSV] * baseI[b]
#             V_ikss[:, ix] = ybus_fact(ikss)

#     ikssv_all_f = np.conj(Yf.dot(V_ikss))
#     ikssv_all_t = np.conj(Yt.dot(V_ikss))
#     ikssv_all_f[abs(ikssv_all_f) < 1e-10] = 0.
#     ikssv_all_t[abs(ikssv_all_t) < 1e-10] = 0.

#     # add current source branch current if there is one
#     current_sources = any(~np.isnan(ppci["bus"][:, IKCV])) and np.any(ppci["bus"][:, IKCV] != 0)
#     if current_sources:
#         current = np.tile(-ppci["bus"][:, IKCV], (n_sc_bus, 1))
#         for ix, b in enumerate(bus_idx):
#             current[ix, b] += ppci["bus"][b, IKSSC]

#         # calculate voltage source branch current
#         if net["_options"]["inverse_y"]:
#             Zbus = ppci["internal"]["Zbus"]
#             V = np.dot((current * baseI), Zbus).T
#         else:
#             ybus_fact = ppci["internal"]["ybus_fact"]
#             V = np.zeros((n_bus, n_sc_bus), dtype=np.complex128)
#             for ix, b in enumerate(bus_idx):
#                 V[:, ix] = ybus_fact(current[ix, :] * baseI[b])

#         fb = np.real(ppci["branch"][:, 0]).astype(np.int64)
#         tb = np.real(ppci["branch"][:, 1]).astype(np.int64)
#         ikssc_all_f = np.conj(Yf.dot(V))
#         ikssc_all_t = np.conj(Yt.dot(V))

#         ikss_all_f = abs(ikssv_all_f + ikssc_all_f)
#         ikss_all_t = abs(ikssv_all_t + ikssc_all_t)
#     else:
#         ikss_all_f = abs(ikssv_all_f)
#         ikss_all_t = abs(ikssv_all_t)

#     if net._options["return_all_currents"]:
#         ppci["internal"]["branch_ikss_f"] = ikss_all_f / baseI[fb, None]
#         ppci["internal"]["branch_ikss_t"] = ikss_all_t / baseI[tb, None]
#     else:
#         ikss_all_f[abs(ikss_all_f) < 1e-10] = np.nan
#         ikss_all_t[abs(ikss_all_t) < 1e-10] = np.nan
#         ppci["branch"][:, IKSS_F] = np.nan_to_num(minmax(ikss_all_f, axis=1) / baseI[fb])
#         ppci["branch"][:, IKSS_T] = np.nan_to_num(minmax(ikss_all_t, axis=1) / baseI[tb])

#     if net._options["ip"]:
#         kappa = ppci["bus"][:, KAPPA]
#         if current_sources:
#             ip_all_f = np.sqrt(2) * (ikssv_all_f * kappa[bus_idx] + ikssc_all_f)
#             ip_all_t = np.sqrt(2) * (ikssv_all_t * kappa[bus_idx] + ikssc_all_t)
#         else:
#             ip_all_f = np.sqrt(2) * ikssv_all_f * kappa[bus_idx]
#             ip_all_t = np.sqrt(2) * ikssv_all_t * kappa[bus_idx]

#         if net._options["return_all_currents"]:
#             ppci["internal"]["branch_ip_f"] = abs(ip_all_f) / baseI[fb, None]
#             ppci["internal"]["branch_ip_t"] = abs(ip_all_t) / baseI[tb, None]
#         else:
#             ip_all_f[abs(ip_all_f) < 1e-10] = np.nan
#             ip_all_t[abs(ip_all_t) < 1e-10] = np.nan
#             ppci["branch"][:, IP_F] = np.nan_to_num(minmax(abs(ip_all_f), axis=1) / baseI[fb])
#             ppci["branch"][:, IP_T] = np.nan_to_num(minmax(abs(ip_all_t), axis=1) / baseI[tb])

#     if net._options["ith"]:
#         n = 1
#         m = ppci["bus"][bus_idx, M]
#         ith_all_f = ikss_all_f * np.sqrt(m + n)
#         ith_all_t = ikss_all_t * np.sqrt(m + n)

#         if net._options["return_all_currents"]:
#             ppci["internal"]["branch_ith_f"] = ith_all_f / baseI[fb, None]
#             ppci["internal"]["branch_ith_t"] = ith_all_t / baseI[tb, None]
#         else:
#             ppci["branch"][:, ITH_F] = np.nan_to_num(minmax(ith_all_f, axis=1) / baseI[fb])
#             ppci["branch"][:, ITH_T] = np.nan_to_num(minmax(ith_all_t, axis=1) / baseI[fb])

#     # Update bus index for branch results
#     if net._options["return_all_currents"]:
#         ppci["internal"]["br_res_ks_ppci_bus"] = bus_idx


def nan_minmax(a, rows, argminmax):
    # because numpy won't sort complex values by magnitude :(
    rows_allnan = np.isnan(a[rows, :]).all(axis=1)
    aa = a[rows, :]
    minmax_a = np.zeros(a.shape[0], dtype=np.complex128)
    minmax_a[~rows_allnan] = aa[~rows_allnan, argminmax(np.abs(aa[~rows_allnan, :]), axis=1)]
    return minmax_a


def _calc_branch_currents_complex(net, bus_idx, ppci0, ppci1, ppci2, sequence):
    if sequence == 0:
        ppci = ppci0
    elif sequence == 1:
        ppci = ppci1
    elif sequence == 2:
        ppci = ppci2
    fault = net._options["fault"]
    fault_impedance = net._options["fault_impedance"]
    n_sc_bus = np.shape(bus_idx)[0]

    case = net._options["case"]
    minmax = np.nanmin if case == "min" else np.nanmax
    argminmax = np.nanargmin if case == "min" else np.nanargmax

    # ikssv = ppci["bus"][:, IKSSV] * np.exp(1j * np.deg2rad(ppci["bus"][:, PHI_IKSSV_DEGREE]))
    V_ikss = ppci["internal"]["V_ikss"]
    valid_V = ppci["internal"]["valid_V"]

    Yf = ppci["internal"]["Yf"]
    Yt = ppci["internal"]["Yt"]
    baseI = ppci["internal"]["baseI"]
    n_bus = ppci["bus"].shape[0]
    fb = np.real(ppci["branch"][:, 0]).astype(np.int64)
    tb = np.real(ppci["branch"][:, 1]).astype(np.int64)
    ppci["internal"]["baseV"] = ppci["bus"][:, BASE_KV]  # shape issue

    # branch = ppci["branch"]
    # offtap = np.flatnonzero(branch[:, TAP] != 1)
    # if len(offtap) != 0:
    #     branch[offtap, BR_R] *= branch[offtap, TAP]
    #     branch[offtap, BR_X] *= branch[offtap, TAP]
    #     Ybus_corr, _, _ = makeYbus(ppci["baseMVA"], ppci["bus"], branch)
    #     Zbus = inv(Ybus_corr.toarray())
    # else:
    #     Zbus = ppci["internal"]["Zbus"]

    # # calculate voltage source branch current
    # if net["_options"]["inverse_y"]:
    #     Zbus = ppci["internal"]["Zbus"]
    #     ikssv = ppci["bus"][:, IKSSV] * np.exp(1j * np.deg2rad(ppci["bus"][:, PHI_IKSSV_DEGREE]))
    #     # V_ikssv = (ikssv * baseI * Zbus)  # making it a complex calculation
    #     # V_ikssv = V_ikssv[:, bus_idx]
    #     # V_ikssv1 = V_ikssv[np.argmax(np.abs(V_ikssv), axis=0), np.arange(V_ikssv.shape[1])] - V_ikssv  # numpy indexing issue
    #     # todo explain formula
    #     V_ikss = V0 - Zbus[:, bus_idx] / Zbus[bus_idx, bus_idx] * V0[bus_idx]
    #     V_ikss[np.abs(V_ikss) < 1e-10] = 0
    #     # ikssv = ppci["internal"]["Ybus"] * V_ikss
    #     V_ikss[0] / Zbus[bus_idx,bus_idx] / ppci["bus"][bus_idx, BASE_KV] / np.sqrt(3) * ppci["baseMVA"]
    #     print(abs(V_ikss), np.angle(V_ikss, deg=True))
    #     I0 = ppci["internal"]["Ybus"] * V0
    #     ikssv[bus_idx] = I0[bus_idx] - ikssv[bus_idx]
    #     abs(V0[0]) / Zbus[bus_idx, bus_idx] / ppci["bus"][bus_idx, BASE_KV] / np.sqrt(3) * ppci["baseMVA"]
    #
    # else:
    #     # todo
    #     ybus_fact = ppci["internal"]["ybus_fact"]
    #     V_ikss = np.zeros((n_bus, n_sc_bus), dtype=np.complex128)
    #     for ix, b in enumerate(bus_idx):
    #         ikss = np.zeros(n_bus, dtype=np.complex128)
    #         ikss[b] = ppci["bus"][b, IKSSV] * np.exp(1j * np.deg2rad(ppci["bus"][b, PHI_IKSSV_DEGREE])) * baseI[b]
    #         V_ikss[:, ix] = ybus_fact(ikss)

    # net_copy = net.deepcopy()
    # pp.runpp(net_copy)
    # Ybus_p = net_copy._ppc["internal"]["Ybus"]
    # Yf_p = net_copy._ppc["internal"]["Yf"]
    # Yt_ = net_copy._ppc["internal"]["Yt"]
    # from scipy.linalg import inv
    # Zbus_p = inv(Ybus_p.toarray())

    # V_ikss_p = (ikssv * baseI * Zbus_p)[:, bus_idx]
    #

    # z = 1.05 / (ikssv * baseI)[bus_idx]
    # # ppci["bus"][0, [4, 5]] = 0
    # ppci["bus"][bus_idx, 4] = z.real
    # ppci["bus"][bus_idx, 5] = z.imag
    # Ybus, Yf, Yt = makeYbus(ppci["baseMVA"], ppci["bus"], ppci["branch"])
    # Zbus = inv(Ybus.toarray())
    # V_ikss = (ikssv * baseI * Zbus)[:, bus_idx]

    # if len(tap_branch) != 0:
    #     factor[tap_branch[:, F_BUS].real.astype(np.int64)] = tap_branch[:, TAP].real

    # branch = ppci["branch"]
    # offtap = np.flatnonzero(branch[:, TAP] != 1)
    # if len(offtap) != 0:
    #     branch[offtap, BR_R] *= branch[offtap, TAP]
    #     branch[offtap, BR_X] *= branch[offtap, TAP]
    #     Ybus_corr, Yf, Yt = makeYbus(ppci["baseMVA"], ppci["bus"], branch)
    #     Zbus = inv(Ybus_corr.toarray())
    # else:
    #     Zbus = ppci["internal"]["Zbus"]

    # V_ikss = V_ikss[np.argmax(np.abs(V_ikss), axis=0), np.arange(V_ikss.shape[1])] - V_ikss  # numpy indexing issue

    # V_ikss = V0 - V_ikss  # numpy indexing issue
    # V_ikss[bus_idx] = - 1.05  # numpy indexing issue
    # Ybus = ppci["internal"]["Ybus"]
    # diff_i = Ybus @ V_ikss
    # V2=((diff_i + ikssv * baseI) * Zbus)[:, bus_idx]

    ikssv_all_f = Yf.dot(V_ikss)
    ikssv_all_t = Yt.dot(V_ikss)
    ikssv_all_f[np.abs(ikssv_all_f) < 1e-6] = 0.
    ikssv_all_t[np.abs(ikssv_all_t) < 1e-6] = 0.

    # add current source branch current if there is one
    current_sources = any(~np.isnan(ppci1["bus"][:, IKCV])) and np.any(ppci1["bus"][:, IKCV] != 0)
    if current_sources:
        ikcv = ppci1["bus"][:, IKCV] * np.exp(np.deg2rad(ppci1["bus"][:, PHI_IKCV_DEGREE]) * 1j)
        # ikssc = ppci1["bus"][:, IKSSC] * np.exp(1j * np.deg2rad(ppci1["bus"][:, PHI_IKSSC_DEGREE]))
        sgen_bus = sgen_bus = ikcv!=0

        current = np.tile(ikcv, (n_sc_bus, 1))
        # np.fill_diagonal(current, current.diagonal() - ikssc)
        for ix, b in enumerate(bus_idx):
            # current[ix, b] -= ppci["bus"][b, IKSSC] * np.exp(np.deg2rad(ppci["bus"][b, PHI_IKSSC_DEGREE])*1j)
            current_sources_on_seq = any(~np.isnan(ppci["bus"][:, IKCV])) and np.any(ppci["bus"][:, IKCV] != 0)
            if ~current_sources_on_seq:
                current[ix,sgen_bus] = 0

            Zbus0 = ppci0["internal"]["Zbus"]
            Zbus1 = ppci1["internal"]["Zbus"]
            Zbus2 = ppci2["internal"]["Zbus"]
            isgen = np.array([0])

            if fault == "LLL":
                if sequence == 1:
                    isgen = ikcv[sgen_bus] * Zbus1[sgen_bus,b] / (Zbus1[b,b] + fault_impedance)

            elif fault == "LG":
                isgen = ikcv[sgen_bus] * Zbus1[sgen_bus,b] / (Zbus0[b,b] + Zbus1[b,b] + Zbus2[b,b] + 3*fault_impedance)

            elif fault == "LL":
                if sequence == 1: 
                    isgen = ikcv[sgen_bus] * Zbus1[sgen_bus,b] / (Zbus1[b,b] + Zbus2[b,b] + fault_impedance)
                elif sequence == 2: 
                    isgen = - ikcv[sgen_bus] * Zbus1[sgen_bus,b] / (Zbus1[b,b] + Zbus2[b,b] + fault_impedance)

            elif fault == "LLG":
                i02 = ikcv[sgen_bus] * Zbus1[sgen_bus,b] / (Zbus1[b,b] + fault_impedance + \
                                    ((Zbus2[b,b]+fault_impedance)*(Zbus0[b,b]+fault_impedance))/(Zbus2[b,b]+Zbus0[b,b]+2*fault_impedance))
                if sequence == 0:
                    isgen = - i02 * (Zbus2[b,b] +fault_impedance) / (Zbus2[b,b]+Zbus0[b,b]+2*fault_impedance)
                if sequence == 1:
                    isgen = i02
                elif sequence == 2:
                    isgen = - i02 * (Zbus0[b,b] + fault_impedance) / (Zbus2[b,b]+Zbus0[b,b]+2*fault_impedance)
                
            current[ix, b] -= sum(isgen)

        # calculate voltage source branch current
        if net["_options"]["inverse_y"]:
            Zbus = ppci["internal"]["Zbus"]
            V = np.dot(Zbus, (current * baseI).T)
        else:
            ybus_fact = ppci["internal"]["ybus_fact"]
            V = np.zeros((n_bus, n_sc_bus), dtype=np.complex128)
            for ix, b in enumerate(bus_idx):
                V[:, ix] = ybus_fact(current[ix, :] * baseI)

        V[np.abs(V) < 1e-10] = 0

        ikssc_all_f = Yf.dot(V)
        ikssc_all_t = Yt.dot(V)
        ikssc_all_f[np.abs(ikssc_all_f) < 1e-10] = 0.
        ikssc_all_t[np.abs(ikssc_all_t) < 1e-10] = 0.

        V_ikss += V  # superposition

        ikss_all_f = ikssv_all_f + ikssc_all_f
        ikss_all_t = ikssv_all_t + ikssc_all_t

    else:

        ikss_all_f = ikssv_all_f
        ikss_all_t = ikssv_all_t

    # calculate active and reactive power, voltages
    skss_all_f = np.conj(ikss_all_f) * V_ikss[fb]
    pkss_all_f = skss_all_f.real
    qkss_all_f = skss_all_f.imag

    skss_all_t = np.conj(ikss_all_t) * V_ikss[tb]
    pkss_all_t = skss_all_t.real
    qkss_all_t = skss_all_t.imag

    rows_fb = np.arange(len(fb))
    rows_tb = np.arange(len(tb))
    minmax_vkss_all_f = V_ikss[fb][rows_fb, argminmax(np.abs(V_ikss[fb]), axis=1)].flatten()
    minmax_vkss_all_t = V_ikss[tb][rows_tb, argminmax(np.abs(V_ikss[tb]), axis=1)].flatten()

    ikss_all_f[abs(ikss_all_f) < 1e-10] = np.nan
    ikss_all_t[abs(ikss_all_t) < 1e-10] = np.nan
    # minmax_ikss_all_f = ikss_all_f[rows_fb, argminmax(np.abs(ikss_all_f), axis=1)].flatten()
    # minmax_ikss_all_t = ikss_all_t[rows_tb, argminmax(np.abs(ikss_all_t), axis=1)].flatten()
    minmax_ikss_all_f = nan_minmax(ikss_all_f, rows_fb, argminmax)
    minmax_ikss_all_t = nan_minmax(ikss_all_t, rows_tb, argminmax)

    if net._options["return_all_currents"]:
        ppci["internal"]["branch_ikss_f"] = np.nan_to_num(np.abs(ikss_all_f)) / baseI[fb, None]
        ppci["internal"]["branch_ikss_t"] = np.nan_to_num(np.abs(ikss_all_t)) / baseI[tb, None]

        ppci["internal"]["branch_ikss_angle_f"] = np.nan_to_num(np.angle(ikss_all_f, deg=True))
        ppci["internal"]["branch_ikss_angle_t"] = np.nan_to_num(np.angle(ikss_all_t, deg=True))


        if valid_V:
            ppci["internal"]["branch_pkss_f"] = np.nan_to_num(pkss_all_f) * ppci["baseMVA"]
            ppci["internal"]["branch_pkss_t"] = np.nan_to_num(pkss_all_t) * ppci["baseMVA"]

            ppci["internal"]["branch_qkss_f"] = np.nan_to_num(qkss_all_f) * ppci["baseMVA"]
            ppci["internal"]["branch_qkss_t"] = np.nan_to_num(qkss_all_t) * ppci["baseMVA"]

            ppci["internal"]["branch_vkss_f"] = np.nan_to_num(np.abs(V_ikss[fb]))
            ppci["internal"]["branch_vkss_t"] = np.nan_to_num(np.abs(V_ikss[tb]))

            ppci["internal"]["branch_vkss_angle_f"] = np.nan_to_num(np.angle(V_ikss[fb], deg=True))
            ppci["internal"]["branch_vkss_angle_t"] = np.nan_to_num(np.angle(V_ikss[tb], deg=True))

    else:
        # ikss_all_f[np.abs(ikss_all_f) < 1e-10] = np.nan
        # ikss_all_t[np.abs(ikss_all_t) < 1e-10] = np.nan
        ppci["branch"][:, IKSS_F] = np.abs(minmax_ikss_all_f) / baseI[fb]
        ppci["branch"][:, IKSS_ANGLE_F] = np.angle(minmax_ikss_all_f, deg=True)
        ppci["branch"][:, IKSS_T] = np.abs(minmax_ikss_all_t) / baseI[tb]
        ppci["branch"][:, IKSS_ANGLE_T] = np.angle(minmax_ikss_all_t, deg=True)

        if valid_V:

            if (fault == "LLL") or (fault == "LG") or (fault == "LLG"):
                sn_mva = ppci["baseMVA"]
            elif fault == "LL":
                sn_mva = ppci["baseMVA"] / 3

            ppci["branch"][:, VKSS_MAGN_F] = np.abs(minmax_vkss_all_f)
            ppci["branch"][:, VKSS_MAGN_T] = np.abs(minmax_vkss_all_t)

            ppci["branch"][:, VKSS_ANGLE_F] = np.angle(minmax_vkss_all_f, deg=True)
            ppci["branch"][:, VKSS_ANGLE_T] = np.angle(minmax_vkss_all_t, deg=True)
            
            ppci["branch"][:, PKSS_F] = np.nan_to_num(minmax(pkss_all_f, axis=1)) * sn_mva
            ppci["branch"][:, QKSS_F] = np.nan_to_num(minmax(qkss_all_f, axis=1)) * sn_mva

            ppci["branch"][:, PKSS_T] = np.nan_to_num(minmax(pkss_all_t, axis=1)) * sn_mva
            ppci["branch"][:, QKSS_T] = np.nan_to_num(minmax(qkss_all_t, axis=1)) * sn_mva

    if net._options["ip"]:  # LL detailed calculations not included!!!
        kappa = ppci["bus"][:, KAPPA]
        if current_sources:
            ip_all_f = np.sqrt(2) * (ikssv_all_f * kappa[bus_idx] + ikssc_all_f)
            ip_all_t = np.sqrt(2) * (ikssv_all_t * kappa[bus_idx] + ikssc_all_t)
        else:
            ip_all_f = np.sqrt(2) * ikssv_all_f * kappa[bus_idx]
            ip_all_t = np.sqrt(2) * ikssv_all_t * kappa[bus_idx]

        if net._options["return_all_currents"]:
            ppci["internal"]["branch_ip_f"] = np.nan_to_num(np.abs(ip_all_f)) / baseI[fb, None]
            ppci["internal"]["branch_ip_t"] = np.nan_to_num(np.abs(ip_all_t)) / baseI[tb, None]
        else:
            ip_all_f[np.abs(ip_all_f) < 1e-10] = np.nan
            ip_all_t[np.abs(ip_all_t) < 1e-10] = np.nan
            # ppci["branch"][:, IP_F] = minmax(np.abs(ip_all_f), axis=1) / baseI[fb]
            # ppci["branch"][:, IP_T] = minmax(np.abs(ip_all_t), axis=1) / baseI[tb]
            ppci["branch"][:, IP_F] = np.nanmax(np.abs(ip_all_f), axis=1) / baseI[fb]
            ppci["branch"][:, IP_T] = np.nanmax(np.abs(ip_all_t), axis=1) / baseI[tb]

    if net._options["ith"]: # LL detailed calculations not included!!!
        n = 1
        m = ppci["bus"][bus_idx, M]
        ith_all_f = np.abs(ikss_all_f * np.sqrt(m + n))
        ith_all_t = np.abs(ikss_all_t * np.sqrt(m + n))

        if net._options["return_all_currents"]:
            ppci["internal"]["branch_ith_f"] = np.nan_to_num(np.abs(ith_all_f)) / baseI[fb, None]
            ppci["internal"]["branch_ith_t"] = np.nan_to_num(np.abs(ith_all_t)) / baseI[tb, None]
        else:
            ith_all_f[np.abs(ith_all_f) < 1e-10] = np.nan
            ith_all_t[np.abs(ith_all_t) < 1e-10] = np.nan
            # ppci["branch"][:, ITH_F] = minmax(np.abs(ith_all_f), axis=1) / baseI[fb]
            # ppci["branch"][:, ITH_T] = minmax(np.abs(ith_all_t), axis=1) / baseI[fb]
            ppci["branch"][:, ITH_F] = np.nanmax(np.abs(ith_all_f), axis=1) / baseI[fb]
            ppci["branch"][:, ITH_T] = np.nanmax(np.abs(ith_all_t), axis=1) / baseI[fb]

    # Update bus index for branch results
    if net._options["return_all_currents"]:
        ppci["internal"]["br_res_ks_ppci_bus"] = bus_idx

    return ppci
