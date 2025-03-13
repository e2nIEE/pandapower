# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


from collections import UserDict

import numpy as np
import pandas as pd

import pandapower.pypower.idx_bus as idx_bus
from pandapower.auxiliary import _init_runse_options
from pandapower.estimation.util import estimate_voltage_vector
from pandapower.pd2ppc import _pd2ppc
from pandapower.pf.run_newton_raphson_pf import _run_dc_pf
from pandapower.pypower.idx_brch import branch_cols
from pandapower.pypower.idx_bus import bus_cols
from pandapower.pypower.makeYbus import makeYbus

from pandapower.estimation.idx_bus import (VM, VM_IDX, VM_STD,
                                           VA, VA_IDX, VA_STD,
                                           P, P_IDX, P_STD,
                                           Q, Q_IDX, Q_STD,
                                           ZERO_INJ_FLAG, bus_cols_se)
from pandapower.estimation.idx_brch import (P_FROM, P_FROM_IDX, P_FROM_STD,
                                            Q_FROM, Q_FROM_IDX, Q_FROM_STD,
                                            IM_FROM, IM_FROM_IDX, IM_FROM_STD,
                                            IA_FROM, IA_FROM_IDX, IA_FROM_STD,
                                            P_TO, P_TO_IDX, P_TO_STD,
                                            Q_TO, Q_TO_IDX, Q_TO_STD,
                                            IM_TO, IM_TO_IDX, IM_TO_STD,
                                            IA_TO, IA_TO_IDX, IA_TO_STD,
                                            branch_cols_se)

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging
std_logger = logging.getLogger(__name__)
ZERO_INJECTION_STD_DEV = 0.001

# Constant Lookup
BR_SIDE = {"line": {"f": "from", "t": "to"},
           "trafo": {"f": "hv", "t": "lv"}}
BR_MEAS_PPCI_IX = {("p", "f"): {"VALUE": P_FROM, "IDX": P_FROM_IDX, "STD": P_FROM_STD},
                   ("q", "f"): {"VALUE": Q_FROM, "IDX": Q_FROM_IDX, "STD": Q_FROM_STD},
                   ("i", "f"): {"VALUE": IM_FROM, "IDX": IM_FROM_IDX, "STD": IM_FROM_STD},
                   ("ia", "f"): {"VALUE": IA_FROM, "IDX": IA_FROM_IDX, "STD": IA_FROM_STD},
                   ("p", "t"): {"VALUE": P_TO, "IDX": P_TO_IDX, "STD": P_TO_STD},
                   ("q", "t"): {"VALUE": Q_TO, "IDX": Q_TO_IDX, "STD": Q_TO_STD},
                   ("i", "t"): {"VALUE": IM_TO, "IDX": IM_TO_IDX, "STD": IM_TO_STD},
                   ("ia", "t"): {"VALUE": IA_TO, "IDX": IA_TO_IDX, "STD": IA_TO_STD}}
BUS_MEAS_PPCI_IX = {"v": {"VALUE": VM, "IDX": VM_IDX, "STD": VM_STD},
                    "va": {"VALUE": VA, "IDX": VA_IDX, "STD": VA_STD},
                    "p": {"VALUE": P, "IDX": P_IDX, "STD": P_STD},
                    "q": {"VALUE": Q, "IDX": Q_IDX, "STD": Q_STD}}


def _initialize_voltage(net, init, calculate_voltage_angles):
    v_start, delta_start = None, None
    if init == 'results':
        v_start, delta_start = 'results', 'results'
    elif init == 'slack':
        res_bus = estimate_voltage_vector(net)
        v_start = res_bus.vm_pu.values
        if calculate_voltage_angles:
            delta_start = res_bus.va_degree.values
    elif init != 'flat':
        raise UserWarning("Unsupported init value. Using flat initialization.")
    return v_start, delta_start


def _init_ppc(net, v_start, delta_start, calculate_voltage_angles):
    # select elements in service and convert pandapower ppc to ppc
    _init_runse_options(net, v_start=v_start, delta_start=delta_start,
                        calculate_voltage_angles=calculate_voltage_angles)
    ppc, ppci = _pd2ppc(net)

    # do dc power flow for phase shifting transformers
    if np.any(net.trafo.shift_degree):
        vm_backup = ppci["bus"][:, 7].copy()
        pq_backup = ppci["bus"][:, [2, 3]].copy()
        # ppci["bus"][:, [2, 3]] = 0.
        ppci = _run_dc_pf(ppci)
        ppci["bus"][:, 7] = vm_backup
        ppci["bus"][:, [2, 3]] = pq_backup

    return ppc, ppci


def _add_measurements_to_ppci(net, ppci, zero_injection, algorithm):
    """

    Add pandapower measurements to the ppci structure by adding new columns
    :param net: pandapower net
    :param ppci: generated ppci
    :return: ppc with added columns
    """
    meas = net.measurement.copy(deep=True)
    if meas.empty:
        raise Exception("No measurements are available in pandapower Network! Abort estimation!")

    # Convert side from string to bus id
    meas["side"] = meas.apply(lambda row:
                              net['line'].at[row["element"], row["side"] + "_bus"] if
                              row["side"] in ("from", "to") else
                              net[row["element_type"]].at[row["element"], row["side"] + '_bus'] if
                              row["side"] in ("hv", "mv", "lv") else row["side"], axis=1)

    # convert p, q, i measurement to p.u., u already in p.u.
    meas.loc[meas.measurement_type == "p", ["value", "std_dev"]] /= ppci["baseMVA"]
    meas.loc[meas.measurement_type == "q", ["value", "std_dev"]] /= ppci["baseMVA"]

    if not meas.query("measurement_type=='i'").empty:
        meas_i_mask = (meas.measurement_type == 'i')
        base_i_ka = ppci["baseMVA"] / net.bus.loc[(meas.side.fillna(meas.element))[meas_i_mask].values,
                                                  "vn_kv"].values
        meas.loc[meas_i_mask, "value"] /= base_i_ka / np.sqrt(3)
        meas.loc[meas_i_mask, "std_dev"] /= base_i_ka / np.sqrt(3)

    if not meas.query("(measurement_type=='ia' )| (measurement_type=='va')").empty:
        meas_dg_mask = (meas.measurement_type == 'ia') | (meas.measurement_type == 'va')
        meas.loc[meas_dg_mask, "value"] = np.deg2rad(meas.loc[meas_dg_mask, "value"])
        meas.loc[meas_dg_mask, "std_dev"] = np.deg2rad(meas.loc[meas_dg_mask, "std_dev"])

    # Get elements mapping from pandapower to ppc
    map_bus = net["_pd2ppc_lookups"]["bus"]
    meas_bus = meas[(meas['element_type'] == 'bus')]
    if (map_bus[meas_bus['element'].values.astype(np.int64)] >= ppci["bus"].shape[0]).any():
        std_logger.warning("Measurement defined in pp-grid does not exist in ppci, will be deleted!")
        meas_bus = meas_bus[map_bus[meas_bus['element'].values.astype(np.int64)] < ppci["bus"].shape[0]]

    # mapping to dict instead of np array ensures good performance for large indices
    # (e.g., 999999999 requires a large np array even if there are only 2 buses)
    # downside is loop comprehension to access the map
    map_line, map_trafo, map_trafo3w = None, None, None
    br_is_mask = ppci['internal']['branch_is']
    if not net.line.empty:
        line_is_mask = br_is_mask[np.arange(*net["_pd2ppc_lookups"]["branch"]["line"])]
        num_line_is = np.sum(line_is_mask)
        map_line = pd.Series(index=net.line.index.values[line_is_mask],
                             data=np.arange(num_line_is))

    if not net.trafo.empty:
        trafo_ix_start, trafo_ix_end = net["_pd2ppc_lookups"]["branch"]["trafo"]
        trafo_is_mask = br_is_mask[np.arange(trafo_ix_start, trafo_ix_end)]
        num_trafo_is = np.sum(trafo_is_mask)
        trafo_ix_offset = np.sum(br_is_mask[:trafo_ix_start])
        map_trafo = pd.Series(index=net.trafo.index.values[trafo_is_mask],
                              data=np.arange(trafo_ix_offset, trafo_ix_offset+num_trafo_is))

    if not net.trafo3w.empty:
        trafo3w_ix_start = net["_pd2ppc_lookups"]["branch"]["trafo3w"][0]
        num_trafo3w = net.trafo3w.shape[0]
        # Only the HV side branch is needed to evaluate is/os status
        trafo3w_is_mask = br_is_mask[np.arange(trafo3w_ix_start,
                                               trafo3w_ix_start+num_trafo3w)]
        num_trafo3w_is = np.sum(trafo3w_is_mask)

        trafo3w_ix_offset = np.sum(br_is_mask[:trafo3w_ix_start])
        map_trafo3w = {trafo3w_ix: {'hv': br_ix,
                                    'mv': br_ix + num_trafo3w_is,
                                    'lv': br_ix + 2 * num_trafo3w_is}
                       for trafo3w_ix, br_ix in
                        zip(net.trafo3w.index.values[trafo3w_is_mask],
                            np.arange(trafo3w_ix_offset,
                                      trafo3w_ix_offset+num_trafo3w_is))}

    # set measurements for ppc format
    # add 9 columns to ppc[bus] for Vm, Vm std dev, P, P std dev, Q, Q std dev,
    # pandapower measurement indices V, P, Q
    bus_append = np.full((ppci["bus"].shape[0], bus_cols_se), np.nan, dtype=ppci["bus"].dtype)

    # Add measurements for bus
    for meas_type in ("v", "va", "p", "q"):
        this_meas = meas_bus[(meas_bus.measurement_type == meas_type)]
        if len(this_meas):
            bus_positions = map_bus[this_meas.element.values.astype(np.int64)]
            if meas_type in ("p", "q"):
                # Convert injection reference to consumption reference (P, Q)
                this_meas.value *= -1
            unique_bus_positions = np.unique(bus_positions)
            if len(unique_bus_positions) < len(bus_positions):
                std_logger.debug("P,Q Measurement duplication will be automatically merged!")
                for bus in unique_bus_positions:
                    this_meas_on_bus = this_meas.iloc[np.argwhere(bus_positions == bus).ravel(), :]
                    element_positions = this_meas_on_bus["element"].values.astype(np.int64)
                    unique_element_positions = np.unique(element_positions)
                    if meas_type in ("v", "va"):
                        merged_value, merged_std_dev = merge_measurements(this_meas_on_bus.value, this_meas_on_bus.std_dev)
                        bus_append[bus, BUS_MEAS_PPCI_IX[meas_type]["VALUE"]] = merged_value
                        bus_append[bus, BUS_MEAS_PPCI_IX[meas_type]["STD"]] = merged_std_dev
                        bus_append[bus, BUS_MEAS_PPCI_IX[meas_type]["IDX"]] = this_meas_on_bus.index[0]
                        continue
                    for element in unique_element_positions:
                        this_meas_on_element = this_meas_on_bus.iloc[np.argwhere(element_positions == element).ravel(), :]
                        merged_value, merged_std_dev = merge_measurements(this_meas_on_element.value, this_meas_on_element.std_dev)
                        this_meas_on_bus.loc[this_meas_on_element.index[0], ["value", "std_dev"]] = [merged_value, merged_std_dev]
                        this_meas_on_bus.loc[this_meas_on_element.index[1:], ["value", "std_dev"]] = [0, 0]
                    sum_value, sum_std_dev = sum_measurements(this_meas_on_bus.value, this_meas_on_bus.std_dev)
                    bus_append[bus, BUS_MEAS_PPCI_IX[meas_type]["VALUE"]] = sum_value
                    bus_append[bus, BUS_MEAS_PPCI_IX[meas_type]["STD"]] = sum_std_dev
                    bus_append[bus, BUS_MEAS_PPCI_IX[meas_type]["IDX"]] = this_meas_on_bus.index[0]
                continue

            bus_append[bus_positions, BUS_MEAS_PPCI_IX[meas_type]["VALUE"]] = this_meas.value.values
            bus_append[bus_positions, BUS_MEAS_PPCI_IX[meas_type]["STD"]] = this_meas.std_dev.values
            bus_append[bus_positions, BUS_MEAS_PPCI_IX[meas_type]["IDX"]] = this_meas.index.values

    # add zero injection measurement and labels defined in parameter zero_injection
    bus_append = _add_zero_injection(net, ppci, bus_append, zero_injection)
    # add virtual measurements for artificial buses, which were created because
    # of an open line switch. p/q are 0. and std dev is 1e-6. (small value)
    new_in_line_buses = np.setdiff1d(np.arange(ppci["bus"].shape[0]), map_bus[map_bus >= 0])
    bus_append[new_in_line_buses, 2] = 0.
    bus_append[new_in_line_buses, 3] = 1e-6
    bus_append[new_in_line_buses, 4] = 0.
    bus_append[new_in_line_buses, 5] = 1e-6

    # add 15 columns to mpc[branch] for Im_from, Im_from std dev, Im_to, Im_to std dev,
    # P_from, P_from std dev, P_to, P_to std dev, Q_from, Q_from std dev,  Q_to, Q_to std dev,
    # pandapower measurement index I, P, Q
    branch_append = np.full((ppci["branch"].shape[0], branch_cols_se),
                            np.nan, dtype=ppci["branch"].dtype)

    # Add measurements for line and trafo
    for br_type,  br_map in (("line", map_line), ("trafo", map_trafo)):
        if br_map is None:
            continue
        for meas_type in ("p", "q", "i", "ia"):
            this_meas = meas[(meas.measurement_type == meas_type) &
                             (meas.element_type == br_type) &
                              meas.element.isin(br_map.index)]
            if len(this_meas):
                for br_side in ("f", "t"):
                    meas_this_side = this_meas[(this_meas.side.values.astype(np.int64) ==
                                                net[br_type][BR_SIDE[br_type][br_side]+"_bus"]
                                                [this_meas.element]).values]
                    ix_side = br_map[meas_this_side.element.values].values
                    unique_ix_side = np.unique(ix_side)
                    if len(unique_ix_side) < len(ix_side):
                        for branch in unique_ix_side:
                            this_meas_on_branch = meas_this_side.iloc[np.argwhere(ix_side == branch).ravel(), :]
                            merged_value, merged_std_dev = merge_measurements(this_meas_on_branch.value, this_meas_on_branch.std_dev)
                            branch_append[branch, BR_MEAS_PPCI_IX[(meas_type, br_side)]["VALUE"]] = merged_value
                            branch_append[branch, BR_MEAS_PPCI_IX[(meas_type, br_side)]["STD"]] = merged_std_dev
                            branch_append[branch, BR_MEAS_PPCI_IX[(meas_type, br_side)]["IDX"]] = this_meas_on_branch.index[0]
                        continue

                    branch_append[ix_side, BR_MEAS_PPCI_IX[(meas_type, br_side)]["VALUE"]] = meas_this_side.value.values
                    branch_append[ix_side, BR_MEAS_PPCI_IX[(meas_type, br_side)]["STD"]] = meas_this_side.std_dev.values
                    branch_append[ix_side, BR_MEAS_PPCI_IX[(meas_type, br_side)]["IDX"]] = meas_this_side.index.values

    # Add measurements for trafo3w
    if map_trafo3w is not None:
        for meas_type in ("p", "q", "i", "ia"):
            this_trafo3w_meas = meas[(meas.measurement_type == meas_type) &
                                     (meas.element_type == "trafo3w") &
                                      meas.element.isin(map_trafo3w)]
            if len(this_trafo3w_meas):
                meas_hv = this_trafo3w_meas[(this_trafo3w_meas.side.values ==
                                             net.trafo3w.hv_bus[this_trafo3w_meas.element]).values]
                meas_mv = this_trafo3w_meas[(this_trafo3w_meas.side.values ==
                                             net.trafo3w.mv_bus[this_trafo3w_meas.element]).values]
                meas_lv = this_trafo3w_meas[(this_trafo3w_meas.side.values ==
                                             net.trafo3w.lv_bus[this_trafo3w_meas.element]).values]
                ix_hv = [map_trafo3w[t]['hv'] for t in meas_hv.element.values]
                ix_mv = [map_trafo3w[t]['mv'] for t in meas_mv.element.values]
                ix_lv = [map_trafo3w[t]['lv'] for t in meas_lv.element.values]
                unique_ix_hv = np.unique(ix_hv)
                unique_ix_mv = np.unique(ix_mv)
                unique_ix_lv = np.unique(ix_lv)
                if len(unique_ix_hv) < len(ix_hv):
                    for branch in unique_ix_hv:
                        this_meas_on_branch = meas_hv.iloc[np.argwhere(ix_hv == branch).ravel(), :]
                        merged_value, merged_std_dev = merge_measurements(this_meas_on_branch.value, this_meas_on_branch.std_dev)
                        branch_append[branch, BR_MEAS_PPCI_IX[(meas_type, "f")]["VALUE"]] = merged_value
                        branch_append[branch, BR_MEAS_PPCI_IX[(meas_type, "f")]["STD"]] = merged_std_dev
                        branch_append[branch, BR_MEAS_PPCI_IX[(meas_type, "f")]["IDX"]] = this_meas_on_branch.index[0]
                if len(unique_ix_mv) < len(ix_mv):
                    for branch in unique_ix_mv:
                        this_meas_on_branch = meas_mv.iloc[np.argwhere(ix_mv == branch).ravel(), :]
                        merged_value, merged_std_dev = merge_measurements(this_meas_on_branch.value, this_meas_on_branch.std_dev)
                        branch_append[branch, BR_MEAS_PPCI_IX[(meas_type, "t")]["VALUE"]] = merged_value
                        branch_append[branch, BR_MEAS_PPCI_IX[(meas_type, "t")]["STD"]] = merged_std_dev
                        branch_append[branch, BR_MEAS_PPCI_IX[(meas_type, "t")]["IDX"]] = this_meas_on_branch.index[0]
                if len(unique_ix_lv) < len(ix_lv):
                    for branch in unique_ix_lv:
                        this_meas_on_branch = meas_lv.iloc[np.argwhere(ix_lv == branch).ravel(), :]
                        merged_value, merged_std_dev = merge_measurements(this_meas_on_branch.value, this_meas_on_branch.std_dev)
                        branch_append[branch, BR_MEAS_PPCI_IX[(meas_type, "t")]["VALUE"]] = merged_value
                        branch_append[branch, BR_MEAS_PPCI_IX[(meas_type, "t")]["STD"]] = merged_std_dev
                        branch_append[branch, BR_MEAS_PPCI_IX[(meas_type, "t")]["IDX"]] = this_meas_on_branch.index[0]
                    continue
                branch_append[ix_hv, BR_MEAS_PPCI_IX[(meas_type, "f")]["VALUE"]] = meas_hv.value.values
                branch_append[ix_hv, BR_MEAS_PPCI_IX[(meas_type, "f")]["STD"]] = meas_hv.std_dev.values
                branch_append[ix_hv, BR_MEAS_PPCI_IX[(meas_type, "f")]["IDX"]] = meas_hv.index.values
                branch_append[ix_mv, BR_MEAS_PPCI_IX[(meas_type, "t")]["VALUE"]] = meas_mv.value.values
                branch_append[ix_mv, BR_MEAS_PPCI_IX[(meas_type, "t")]["STD"]] = meas_mv.std_dev.values
                branch_append[ix_mv, BR_MEAS_PPCI_IX[(meas_type, "t")]["IDX"]] = meas_mv.index.values
                branch_append[ix_lv, BR_MEAS_PPCI_IX[(meas_type, "t")]["VALUE"]] = meas_lv.value.values
                branch_append[ix_lv, BR_MEAS_PPCI_IX[(meas_type, "t")]["STD"]] = meas_lv.std_dev.values
                branch_append[ix_lv, BR_MEAS_PPCI_IX[(meas_type, "t")]["IDX"]] = meas_lv.index.values

    # Check append or update
    if ppci["bus"].shape[1] == bus_cols:
        ppci["bus"] = np.hstack((ppci["bus"], bus_append))
    else:
        ppci["bus"][:, bus_cols: bus_cols + bus_cols_se] = bus_append

    if ppci["branch"].shape[1] == branch_cols:
        ppci["branch"] = np.hstack((ppci["branch"], branch_append))
    else:
        ppci["branch"][:, branch_cols: branch_cols + branch_cols_se] = branch_append

    # Add rated power information for AF-WLS estimator
    if algorithm == 'af-wls':
        cluster_list_loads = net.load["type"].unique()
        cluster_list_gen = net.sgen["type"].unique()
        cluster_list_tot = np.concatenate((cluster_list_loads, cluster_list_gen), axis=0)
        ppci["clusters"] = cluster_list_tot
        num_clusters = len(cluster_list_tot)
        num_buses = ppci["bus"].shape[0]
        ppci["rated_power_clusters"] = np.zeros([num_buses, 4*num_clusters])
        for var in ["load", "sgen"]:
            in_service = net[var]["in_service"]
            active_elements = net[var][in_service]
            bus = net._pd2ppc_lookups["bus"][active_elements.bus].astype(int)
            P = active_elements.p_mw.values/ppci["baseMVA"]
            Q = active_elements.q_mvar.values/ppci["baseMVA"]
            if var == 'load':
                P *= -1
                Q *= -1
            cluster = active_elements.type.values
            if (bus >= ppci["bus"].shape[0]).any():
                std_logger.warning("Loads or sgen defined in pp-grid do not exist in ppci, will be deleted!")
                P = P[bus < ppci["bus"].shape[0]]
                Q = Q[bus < ppci["bus"].shape[0]]
                cluster = cluster[bus < ppci["bus"].shape[0]]
                bus = bus[bus < ppci["bus"].shape[0]]
            for k in range(num_clusters):
                cluster[cluster == cluster_list_tot[k]] = k
            cluster = cluster.astype(int)
            for i in range(len(P)):
                bus_i, cluster_i, P_i, Q_i = bus[i], cluster[i], P[i], Q[i]
                ppci["rated_power_clusters"][bus_i, cluster_i] += P_i
                ppci["rated_power_clusters"][bus_i, cluster_i + num_clusters] += Q_i
                ppci["rated_power_clusters"][bus_i, cluster_i + 2*num_clusters] += abs(0.03*P_i)    # std dev cluster variability hardcoded, think how to change it
                ppci["rated_power_clusters"][bus_i, cluster_i + 3*num_clusters] += abs(0.03*Q_i)    # std dev cluster variability hardcoded, think how to change it

    return ppci

def merge_measurements(value, std_dev):
    weight = np.divide(1, np.square(std_dev))
    merged_variance = np.divide(1, weight.sum())
    merged_std_dev = np.sqrt(merged_variance)
    weighted_value = np.multiply(value, weight)
    merged_value = np.multiply(weighted_value.sum(), merged_variance)
    return merged_value, merged_std_dev
    
def sum_measurements(value, std_dev):
    sum_values = value.values.sum()
    variance = np.square(std_dev.values)
    sum_variance = variance.sum()
    sum_std_dev = np.sqrt(sum_variance)
    return sum_values, sum_std_dev

def _add_zero_injection(net, ppci, bus_append, zero_injection):
    """
    Add zero injection labels to the ppci structure and add virtual measurements to those buses
    :param net: pandapower net
    :param ppci: generated ppci
    :param bus_append: added columns to the ppci bus with zero injection label
    :param zero_injection: parameter to control which bus to be identified as zero injection
    :return bus_append: added columns
    """
    bus_append[:, ZERO_INJ_FLAG] = False
    if zero_injection is not None:
        # identify aux bus to zero injection
        if net._pd2ppc_lookups['aux']:
            aux_bus_lookup = np.concatenate([v for k, v in net._pd2ppc_lookups['aux'].items() if k != 'xward'])
            aux_bus = net._pd2ppc_lookups['bus'][aux_bus_lookup]
            aux_bus = aux_bus[aux_bus < ppci["bus"].shape[0]]
            bus_append[aux_bus, ZERO_INJ_FLAG] = True

        if isinstance(zero_injection, str):
            if zero_injection == 'auto':
                # identify bus without elements and pq measurements as zero injection
                zero_inj_bus_mask = (ppci["bus"][:, 1] == 1) & (ppci["bus"][:, 2:4] == 0).all(axis=1) & \
                                    np.isnan(bus_append[:, P:(Q_STD + 1)]).all(axis=1)
                bus_append[zero_inj_bus_mask, ZERO_INJ_FLAG] = True
            elif zero_injection != "aux_bus":
                raise UserWarning("zero injection parameter is not correctly initialized")
        elif hasattr(zero_injection, '__iter__'):
            zero_inj_bus = net._pd2ppc_lookups['bus'][zero_injection]
            bus_append[zero_inj_bus, ZERO_INJ_FLAG] = True

        zero_inj_bus = np.argwhere(bus_append[:, ZERO_INJ_FLAG]).ravel()
        bus_append[zero_inj_bus, P] = 0
        bus_append[zero_inj_bus, P_STD] = ZERO_INJECTION_STD_DEV
        bus_append[zero_inj_bus, Q] = 0
        bus_append[zero_inj_bus, Q_STD] = ZERO_INJECTION_STD_DEV
    return bus_append


def _build_measurement_vectors(ppci, update_meas_only=False):
    """
    Building measurement vector z, pandapower to ppci measurement mapping and covariance matrix R
    :param ppci: generated ppci which contains the measurement columns
    :param branch_cols: number of columns in original ppci["branch"] without measurements
    :param bus_cols: number of columns in original ppci["bus"] without measurements
    :return: both created vectors
    """
    p_bus_not_nan = ~np.isnan(ppci["bus"][:, bus_cols + P])
    p_line_f_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + P_FROM])
    p_line_t_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + P_TO])
    q_bus_not_nan = ~np.isnan(ppci["bus"][:, bus_cols + Q])
    q_line_f_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + Q_FROM])
    q_line_t_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + Q_TO])
    v_bus_not_nan = ~np.isnan(ppci["bus"][:, bus_cols + VM])
    v_degree_bus_not_nan = ~np.isnan(ppci["bus"][:, bus_cols + VA])
    i_line_f_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + IM_FROM])
    i_line_t_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + IM_TO])
    i_degree_line_f_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + IA_FROM])
    i_degree_line_t_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + IA_TO])
    # piece together our measurement vector z
    z = np.concatenate((ppci["bus"][p_bus_not_nan, bus_cols + P],
                        ppci["branch"][p_line_f_not_nan, branch_cols + P_FROM],
                        ppci["branch"][p_line_t_not_nan, branch_cols + P_TO],
                        ppci["bus"][q_bus_not_nan, bus_cols + Q],
                        ppci["branch"][q_line_f_not_nan, branch_cols + Q_FROM],
                        ppci["branch"][q_line_t_not_nan, branch_cols + Q_TO],
                        ppci["bus"][v_bus_not_nan, bus_cols + VM],
                        ppci["bus"][v_degree_bus_not_nan, bus_cols + VA],
                        ppci["branch"][i_line_f_not_nan, branch_cols + IM_FROM],
                        ppci["branch"][i_line_t_not_nan, branch_cols + IM_TO],
                        ppci["branch"][i_degree_line_f_not_nan, branch_cols + IA_FROM],
                        ppci["branch"][i_degree_line_t_not_nan, branch_cols + IA_TO]
                        )).real.astype(np.float64)
    imag_meas = np.concatenate((np.zeros(sum(p_bus_not_nan)),
                               np.zeros(sum(p_line_f_not_nan)),
                               np.zeros(sum(p_line_t_not_nan)),
                               np.zeros(sum(q_bus_not_nan)),
                               np.zeros(sum(q_line_f_not_nan)),
                               np.zeros(sum(q_line_t_not_nan)),
                               np.zeros(sum(v_bus_not_nan)),
                               np.zeros(sum(v_degree_bus_not_nan)),
                               np.ones(sum(i_line_f_not_nan)),
                               np.ones(sum(i_line_t_not_nan)),
                               np.zeros(sum(i_degree_line_f_not_nan)),
                               np.zeros(sum(i_degree_line_t_not_nan))
                               )).astype(bool)
    idx_non_imeas = np.flatnonzero(~imag_meas)
    if ppci.algorithm == "af-wls":
        balance_eq_meas = np.zeros(ppci["rated_power_clusters"].shape[0]).astype(np.float64)
        af_vmeas = 0.4*np.ones(len(ppci["clusters"]))
        z = np.concatenate((z, balance_eq_meas[ppci.non_slack_bus_mask], balance_eq_meas[ppci.non_slack_bus_mask], af_vmeas))
    
    if not update_meas_only:
        # conserve the pandapower indices of measurements in the ppci order
        pp_meas_indices = np.concatenate((ppci["bus"][p_bus_not_nan, bus_cols + P_IDX],
                                          ppci["branch"][p_line_f_not_nan, branch_cols + P_FROM_IDX],
                                          ppci["branch"][p_line_t_not_nan, branch_cols + P_TO_IDX],
                                          ppci["bus"][q_bus_not_nan, bus_cols + Q_IDX],
                                          ppci["branch"][q_line_f_not_nan, branch_cols + Q_FROM_IDX],
                                          ppci["branch"][q_line_t_not_nan, branch_cols + Q_TO_IDX],
                                          ppci["bus"][v_bus_not_nan, bus_cols + VM_IDX],
                                          ppci["bus"][v_degree_bus_not_nan, bus_cols + VA_IDX],
                                          ppci["branch"][i_line_f_not_nan, branch_cols + IM_FROM_IDX],
                                          ppci["branch"][i_line_t_not_nan, branch_cols + IM_TO_IDX],
                                          ppci["branch"][i_degree_line_f_not_nan, branch_cols + IA_FROM_IDX],
                                          ppci["branch"][i_degree_line_t_not_nan, branch_cols + IA_TO_IDX]
                                          )).real.astype(np.int64)
        # Covariance matrix R
        r_cov = np.concatenate((ppci["bus"][p_bus_not_nan, bus_cols + P_STD],
                                ppci["branch"][p_line_f_not_nan, branch_cols + P_FROM_STD],
                                ppci["branch"][p_line_t_not_nan, branch_cols + P_TO_STD],
                                ppci["bus"][q_bus_not_nan, bus_cols + Q_STD],
                                ppci["branch"][q_line_f_not_nan, branch_cols + Q_FROM_STD],
                                ppci["branch"][q_line_t_not_nan, branch_cols + Q_TO_STD],
                                ppci["bus"][v_bus_not_nan, bus_cols + VM_STD],
                                ppci["bus"][v_degree_bus_not_nan, bus_cols + VA_STD],
                                ppci["branch"][i_line_f_not_nan, branch_cols + IM_FROM_STD],
                                ppci["branch"][i_line_t_not_nan, branch_cols + IM_TO_STD],
                                ppci["branch"][i_degree_line_f_not_nan, branch_cols + IA_FROM_STD],
                                ppci["branch"][i_degree_line_t_not_nan, branch_cols + IA_TO_STD]
                                )).real.astype(np.float64)
        meas_mask = np.concatenate([p_bus_not_nan,
                                    p_line_f_not_nan,
                                    p_line_t_not_nan,
                                    q_bus_not_nan,
                                    q_line_f_not_nan,
                                    q_line_t_not_nan,
                                    v_bus_not_nan,
                                    v_degree_bus_not_nan,
                                    i_line_f_not_nan,
                                    i_line_t_not_nan,
                                    i_degree_line_f_not_nan,
                                    i_degree_line_t_not_nan])
        any_i_meas = np.any(np.r_[i_line_f_not_nan, i_line_t_not_nan])
        any_degree_meas = np.any(np.r_[v_degree_bus_not_nan,
                                       i_degree_line_f_not_nan,
                                       i_degree_line_t_not_nan])
        if ppci.algorithm == "af-wls":
            num_clusters = len(ppci["clusters"])
            P_balance_dev_std = np.sqrt(np.sum(np.square(ppci["rated_power_clusters"][:,2*num_clusters:3*num_clusters]),axis=1))
            Q_balance_dev_std = np.sqrt(np.sum(np.square(ppci["rated_power_clusters"][:,3*num_clusters:4*num_clusters]),axis=1))
            af_vmeas_dev_std = 0.15*np.ones(len(ppci["clusters"]))
            r_cov = np.concatenate((r_cov, P_balance_dev_std[ppci.non_slack_bus_mask], Q_balance_dev_std[ppci.non_slack_bus_mask], af_vmeas_dev_std))
            meas_mask = np.concatenate((meas_mask, ppci.non_slack_bus_mask, ppci.non_slack_bus_mask, np.ones(len(ppci["clusters"]))))

        return z, pp_meas_indices, r_cov, meas_mask, any_i_meas, any_degree_meas, idx_non_imeas
    else:
        return z


def pp2eppci(net, v_start=None, delta_start=None,
             calculate_voltage_angles=True, zero_injection="aux_bus",
             algorithm='wls', ppc=None, eppci=None):
    if isinstance(eppci, ExtendedPPCI):
        eppci.algorithm = algorithm
        eppci.data = _add_measurements_to_ppci(net, eppci.data, zero_injection, algorithm)
        eppci.update_meas()
        return net, ppc, eppci
    else:
        # initialize ppc
        ppc, ppci = _init_ppc(net, v_start, delta_start, calculate_voltage_angles)

        # add measurements to ppci structure
        # Finished converting pandapower network to ppci
        ppci = _add_measurements_to_ppci(net, ppci, zero_injection, algorithm)
        return net, ppc, ExtendedPPCI(ppci, algorithm)


class ExtendedPPCI(UserDict):
    def __init__(self, ppci, algorithm):
        """Initialize ppci object with measurements."""
        self.data = ppci
        self.algorithm = algorithm

        # Measurement relevant parameters
        self.z = None
        self.r_cov = None
        self.pp_meas_indices = None
        self.non_nan_meas_mask = None
        self.non_nan_meas_selector = None
        self.any_i_meas = False
        self.any_degree_meas = False

        # check slack bus
        self.non_slack_buses = np.argwhere(ppci["bus"][:, idx_bus.BUS_TYPE] != 3).ravel()
        self.non_slack_bus_mask = (ppci['bus'][:, idx_bus.BUS_TYPE] != 3).ravel()
        self.num_non_slack_bus = np.sum(self.non_slack_bus_mask)
        self.delta_v_bus_mask = np.r_[self.non_slack_bus_mask,
                                      np.ones(self.non_slack_bus_mask.shape[0], dtype=bool)].ravel()
        self.delta_v_bus_selector = np.flatnonzero(self.delta_v_bus_mask)

        # Iniialize measurements 
        self._initialize_meas()

        # Initialize state variable
        self.v_init = ppci["bus"][:, idx_bus.VM]
        self.delta_init = np.radians(ppci["bus"][:, idx_bus.VA])
        self.E_init = np.r_[self.delta_init[self.non_slack_bus_mask], self.v_init]
        self.v = self.v_init.copy()
        self.delta = self.delta_init.copy()
        self.E = self.E_init.copy()
        if algorithm == "af-wls":
            self.E = np.concatenate((self.E, np.full(ppci["clusters"].shape,0.5)))

    def _initialize_meas(self):
        # calculate relevant vectors from ppci measurements
        self.z, self.pp_meas_indices, self.r_cov, self.non_nan_meas_mask,\
            self.any_i_meas, self.any_degree_meas, self.idx_non_imeas =\
            _build_measurement_vectors(self, update_meas_only=False)
        self.non_nan_meas_selector = np.flatnonzero(self.non_nan_meas_mask)

    def update_meas(self):
        self.z = _build_measurement_vectors(self, update_meas_only=True)

    @property
    def V(self):
        return self.v * np.exp(1j * self.delta)

    def reset(self):
        self.v, self.delta, self.E =\
            self.v_init.copy(), self.delta_init.copy(), self.E_init.copy()

    def update_E(self, E):
        self.E = E
        self.v = E[self.num_non_slack_bus:]
        self.delta[self.non_slack_buses] = E[:self.num_non_slack_bus]

    def E2V(self, E):
        self.update_E(E)
        return self.V

    def get_Y(self):
        # Using recycled version if available
        if "Ybus" in self["internal"] and self["internal"]["Ybus"].size:
            Ybus, Yf, Yt = self["internal"]['Ybus'], self["internal"]['Yf'], self["internal"]['Yt']
        else:
            # build admittance matrices
            Ybus, Yf, Yt = makeYbus(self['baseMVA'], self['bus'], self['branch'])
            self["internal"]['Ybus'], self["internal"]['Yf'], self["internal"]['Yt'] = Ybus, Yf, Yt
        return Ybus, Yf, Yt
