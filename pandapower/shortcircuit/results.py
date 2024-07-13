# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
import pandas as pd

from pandapower.pypower.idx_brch_sc import IKSS_F, IKSS_T, IP_F, IP_T, ITH_F, ITH_T, PKSS_F, QKSS_F, PKSS_T, QKSS_T, \
    VKSS_MAGN_F, VKSS_MAGN_T, VKSS_ANGLE_F, VKSS_ANGLE_T, IKSS_ANGLE_F, IKSS_ANGLE_T
from pandapower.pypower.idx_bus_sc import IKSS1, IP, ITH, IKSS2, R_EQUIV_OHM, X_EQUIV_OHM, SKSS
from pandapower.pypower.idx_bus import BUS_TYPE, BASE_KV
from pandapower.results_branch import _copy_switch_results_from_branches
from pandapower.results import BRANCH_RESULTS_KEYS


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
            else:
                ppci_orig["branch"][:, branch_results_cols] =\
                    np.minimum(np.nan_to_num(ppci["branch"][:, branch_results_cols], nan=1e10),
                               np.nan_to_num(ppci_orig["branch"][:, branch_results_cols], nan=1e10))
                # excluding new values from nan to num
                ppci_orig["branch"][:, branch_results_cols_add] = ppci["branch"][:, branch_results_cols_add]


def _get_bus_ppc_idx_for_br_all_results(net, ppc, bus):
    bus_lookup = net._pd2ppc_lookups["bus"]
    if bus is None:
        bus = net.bus.index

    ppc_index = bus_lookup[bus]
    ppc_index[ppc["bus"][ppc_index, BUS_TYPE] == 4] = -1
    return bus, ppc_index


def _extract_results(net, ppc, ppc_0, bus):
    _get_bus_results(net, ppc, ppc_0, bus)
    if net._options["branch_results"]:
        if net._options['return_all_currents']:
            _get_line_all_results(net, ppc, bus)
            _get_trafo_all_results(net, ppc, bus)
            _get_trafo3w_all_results(net, ppc, bus)
            _get_switch_all_results(net, ppc, bus)
        else:
            _get_line_results(net, ppc)
            _get_trafo_results(net, ppc)
            _get_trafo3w_results(net, ppc)
            _get_switch_results(net, ppc)


def _get_bus_results(net, ppc, ppc_0, bus):
    bus_lookup = net._pd2ppc_lookups["bus"]
    ppc_index = bus_lookup[net.bus.index]

    if net["_options"]["fault"] == "1ph":
        net.res_bus_sc["ikss_ka"] = ppc_0["bus"][ppc_index, IKSS1] + ppc["bus"][ppc_index, IKSS2]
        net.res_bus_sc["rk0_ohm"] = ppc_0["bus"][ppc_index, R_EQUIV_OHM]
        net.res_bus_sc["xk0_ohm"] = ppc_0["bus"][ppc_index, X_EQUIV_OHM]
        # in trafo3w, we add very high numbers (1e10) as impedances to block current
        # here, we need to replace such high values by np.inf
        baseZ = ppc_0["bus"][ppc_index, BASE_KV] ** 2 / ppc_0["baseMVA"]
        net.res_bus_sc.loc[net.res_bus_sc["xk0_ohm"]/baseZ > 1e9, "xk0_ohm"] = np.inf
        net.res_bus_sc.loc[net.res_bus_sc["rk0_ohm"]/baseZ > 1e9, "rk0_ohm"] = np.inf
    else:
        net.res_bus_sc["ikss_ka"] = ppc["bus"][ppc_index, IKSS1] + ppc["bus"][ppc_index, IKSS2]
        net.res_bus_sc["skss_mw"] = ppc["bus"][ppc_index, SKSS]
    if net._options["ip"]:
        net.res_bus_sc["ip_ka"] = ppc["bus"][ppc_index, IP]
    if net._options["ith"]:
        net.res_bus_sc["ith_ka"] = ppc["bus"][ppc_index, ITH]

    # Export also equivalent rk, xk on the calculated bus
    net.res_bus_sc["rk_ohm"] = ppc["bus"][ppc_index, R_EQUIV_OHM]
    net.res_bus_sc["xk_ohm"] = ppc["bus"][ppc_index, X_EQUIV_OHM]
    # if for some reason (e.g. contribution of ext_grid set close to 0) we used very high values for rk, xk, we replace them by np.inf
    baseZ = ppc["bus"][ppc_index, BASE_KV] ** 2 / ppc["baseMVA"]
    net.res_bus_sc.loc[net.res_bus_sc["rk_ohm"] / baseZ > 1e9, "rk_ohm"] = np.inf
    net.res_bus_sc.loc[net.res_bus_sc["xk_ohm"] / baseZ > 1e9, "xk_ohm"] = np.inf

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
