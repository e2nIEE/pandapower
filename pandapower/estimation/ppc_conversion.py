# -*- coding: utf-8 -*-
# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


from collections import UserDict
from typing import Dict

import numpy as np
import pandas as pd

from pandapower.pypower.idx_bus import BUS_TYPE as pypower_BUS_TYPE, VM as pypower_VM, VA as pypower_VA
from pandapower.auxiliary import _init_runse_options, pandapowerNet
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


def _initialize_voltage(net, init):
    v_start, delta_start = None, None
    if init == 'results':
        v_start, delta_start = 'results', 'results'
    elif init == 'slack':
        res_bus = estimate_voltage_vector(net)
        v_start = res_bus.vm_pu.values
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



def _calculate_weighted_measurements(measurements, grouped_column: str):
    """Calculate weighted measurements."""
    measurements["weight"] = 1 / (measurements["std_dev"] ** 2)
    measurements["weighted_value"] = measurements["weight"] * measurements["value"]

    merged_weight = 1 / measurements.groupby(grouped_column)["weight"].sum()
    merged_value = measurements.groupby(grouped_column)["weighted_value"].sum()

    merged_value = merged_value.to_frame(name="weighted_measurement")
    merged_value["merged_weight"] = merged_weight
    merged_value["weighted_measurement"] *= merged_value["merged_weight"]
    merged_value["merged_weight"] = np.sqrt(merged_weight)

    return merged_value


def _add_measurements_to_branch(
        branch_append: np.ndarray,
        meas: pd.DataFrame,
        element_name: str,
        side_map: Dict[str, str],
        map_branch: pd.Series,
) -> None:
    """
        Appends weighted branch measurements (power, current, etc.) to the branch_append array.

        Parameters:
        - branch_append: NumPy array (ppci branch matrix) to append measurement values to.
        - meas: DataFrame of measurements.
        - element_name: Name of the element type, e.g., 'line', 'trafo', 'trafo3w'.
        - side_map: Mapping of measurement side values to branch side strings (e.g., {'hv': 'from', 'lv': 'to'}).
        - map_branch: pd.Series mapping element indices to PPCI branch indices.
    """

    for meas_type in ('p', 'q', 'i'):
        filtered = meas[
            (meas.measurement_type == meas_type)
            & (meas.element_type == element_name)
            & meas.element.isin(map_branch.index)
            ]
        if filtered.empty:
            continue

        for side_val, br_side in side_map.items():
            side_df = filtered[filtered.side == side_val].copy()
            if side_df.empty:
                continue

            # Remap element index → PPCI branch index
            side_df['element'] = side_df['element'].map(map_branch)

            # Create index map from PPCI index to original measurement index (for later reference)
            idx_map = (
                side_df[['element']]
                .drop_duplicates(subset='element', keep='first')
                .reset_index()[['element', 'index']]
                .set_index('element')['index']
            )

            # Calculate weighted measurement and std dev by branch index
            merged = _calculate_weighted_measurements(side_df, 'element')

            # Get column indices for VALUE, STD, IDX in branch_append array
            specs = BR_MEAS_PPCI_IX[(meas_type, br_side)]

            branch_append[merged.index, specs['VALUE']] = merged.weighted_measurement
            branch_append[merged.index, specs['STD']] = merged.merged_weight
            branch_append[merged.index, specs['IDX']] = merged.index.map(idx_map)


def _get_branch_map(
        net: pandapowerNet,
        br_is_mask: np.ndarray,
        element_type: str,
) -> pd.Series:
    """
    Builds a Series mapping element indices (e.g., line, trafo) to their corresponding PPCI branch indices.

    Parameters:
    - net: pandapower network object.
    - br_is_mask: Boolean array (ppci['internal']['branch_is']) indicating which PPCI rows are active.
    - element_type: Type of branch element (e.g., 'line', 'trafo', 'trafo3w').

    Returns:
    - map_branch: pd.Series with element indices (from net.<element_type>.index) as index
                  and PPCI branch indices as values.
    """
    # Get the start and end positions in the PPCI branch array for this element type
    start, end = net._pd2ppc_lookups['branch'][element_type]

    # Extract mask for just the entries related to the current element type
    mask = br_is_mask[start:end]

    # Compute PPCI offset: number of active branches before the current element type
    offset = np.sum(br_is_mask[:start])

    # Get the original element indices (e.g., net.line.index) filtered by active mask
    element_indices = getattr(net, element_type).index.values[mask]

    # Compute PPCI indices corresponding to these elements
    ppc_indices = np.arange(offset, offset + mask.sum())

    # Return mapping from element index to PPCI branch index
    return pd.Series(data=ppc_indices, index=element_indices)


def _add_measurements_to_line(
        net: pandapowerNet,
        branch_append: np.ndarray,
        meas: pd.DataFrame,
        br_is_mask: np.ndarray
) -> None:
    """
        Adds line-related measurements (power, current, etc.) to the PPCI branch array.

        Parameters:
        - net: pandapower network object.
        - branch_append: NumPy array representing the PPCI branch matrix where measurements are stored.
        - meas: DataFrame of measurements.
        - br_is_mask:  Boolean array (ppci['internal']['branch_is']) indicating the active branches in ppci['branch'].
    """
    if net.line.empty:
        return

    map_branch = _get_branch_map(net, br_is_mask, "line")

    _add_measurements_to_branch(
        branch_append=branch_append,
        meas=meas,
        element_name="line",
        side_map={"from": "f", "to": "t"},
        map_branch=map_branch
    )


def _add_measurements_to_trafo(
        net: pandapowerNet,
        branch_append: np.ndarray,
        meas: pd.DataFrame,
        br_is_mask: np.ndarray
) -> None:
    """
        Adds transformer (2-winding) related measurements to the PPCI branch matrix.

        Parameters:
        - net: pandapower network object.
        - branch_append: NumPy array representing the PPCI branch matrix where measurements will be written.
        - meas: DataFrame containing measurements.
        - br_is_mask: Boolean array (ppci['internal']['branch_is']) marking active rows in the PPCI branch matrix.
    """

    if net.trafo.empty:
        return

    map_branch = _get_branch_map(net, br_is_mask, "trafo")

    _add_measurements_to_branch(
        branch_append=branch_append,
        meas=meas,
        element_name="trafo",
        side_map={"hv": "f", "lv": "t"},
        map_branch=map_branch,
    )


def _add_measurements_to_trafo3w(
        net: pandapowerNet,
        branch_append: np.ndarray,
        meas: pd.DataFrame,
        br_is_mask: np.ndarray
) -> None:
    """
    Adds measurements for 3-winding transformers (HV, MV, LV sides) to the PPCI branch matrix.

    Parameters:
    - net: pandapower network object.
    - branch_append: NumPy array representing the PPCI branch matrix where measurements will be stored.
    - meas: DataFrame of measurements.
    - br_is_mask: Boolean array (ppci['internal']['branch_is']) marking active rows in the PPCI branch matrix.
    """
    if net.trafo3w.empty:
        return

    # Retrieve starting index for trafo3w entries in the PPCI branch matrix
    trafo3w_ix_start = net["_pd2ppc_lookups"]["branch"]["trafo3w"][0]
    num_trafo3w = len(net.trafo3w)

    # Get mask for HV-side branches (used as base reference for all three sides)
    trafo3w_is_hv = br_is_mask[trafo3w_ix_start: trafo3w_ix_start + num_trafo3w]
    num_active = trafo3w_is_hv.sum()

    # Compute PPCI offset: number of active branches before trafo3w section
    offset = np.count_nonzero(br_is_mask[:trafo3w_ix_start])

    # Indices of active 3-winding trafos
    indices = net.trafo3w.index[trafo3w_is_hv]
    ppc_indices = np.arange(offset, offset + num_active)

    # Build mapping per side: HV, MV, LV
    map_branch_by_side = {
        "hv": pd.Series(data=ppc_indices, index=indices),
        "mv": pd.Series(data=ppc_indices + num_active, index=indices),
        "lv": pd.Series(data=ppc_indices + 2 * num_active, index=indices)
    }

    # Define PPCI side labels for each trafo3w side
    side_map_labels = {"hv": "f", "mv": "t", "lv": "t"}  # HV is 'from', MV and LV are 'to'

    # Process and store measurements per side
    for side, map_branch in map_branch_by_side.items():
        _add_measurements_to_branch(
            branch_append=branch_append,
            meas=meas,
            element_name="trafo3w",
            side_map={side: side_map_labels[side]},
            map_branch=map_branch
        )


def _add_measurements_to_bus(meas_bus, bus_append, map_bus):
    """
        Aggregate measurements by bus index and append results to bus_append array.

        Parameters:
        - meas_bus: subset of measurement DataFrame containing only measurements at buses
        - bus_append: NumPy array to store measurements, std devs, and indices to add to ppci
        - map_bus: dict mapping bus IDs to PPCI bus indices
        """

    # Process voltage (v) and voltage angle (va) measurements
    for meas_type in ("v", "va"):
        this_meas = meas_bus[(meas_bus.measurement_type == meas_type)]

        if this_meas.empty:
            continue

        this_meas["ppci_index"] = this_meas.element.map(lambda x: map_bus[int(x)])

        ind_map = this_meas.drop_duplicates(subset=["ppci_index"], keep="first")
        ind_map = ind_map.reset_index().set_index("ppci_index")

        # voltage measurements at the same ppci bus will be merged via a weighted average
        meas_merged = _calculate_weighted_measurements(this_meas, "ppci_index")
        meas_merged["index"] = ind_map["index"]

        bus_append[meas_merged.index, BUS_MEAS_PPCI_IX[meas_type]["VALUE"]] = meas_merged.weighted_measurement
        bus_append[meas_merged.index, BUS_MEAS_PPCI_IX[meas_type]["STD"]] = meas_merged.merged_weight
        bus_append[meas_merged.index, BUS_MEAS_PPCI_IX[meas_type]["IDX"]] = meas_merged["index"]

    # Process active (p) and reactive (q) power injections
    for meas_type in ("p", "q"):
        this_meas = meas_bus[(meas_bus.measurement_type == meas_type)]
        this_meas.value *= -1

        if this_meas.empty:
            continue

        this_meas["ppci_index"] = this_meas.element.map(lambda x: map_bus[int(x)])
        ind_map = this_meas.drop_duplicates(subset=["ppci_index"], keep="first")
        ind_map = ind_map.reset_index().set_index("ppci_index")

        # power measurements at the same pp bus will be merged via a weighted average
        this_meas = _calculate_weighted_measurements(this_meas, "element")
        this_meas["ppci_index"] = this_meas.index.map(lambda x: map_bus[int(x)])

        # power measurements at different pp buses but same ppci bus will be aggregated
        sum_values = this_meas.groupby("ppci_index")["weighted_measurement"].sum()
        this_meas["merged_weight"] = np.square(this_meas["merged_weight"])
        sum_variance = this_meas.groupby("ppci_index")["merged_weight"].sum()
        sum_std_dev = np.sqrt(sum_variance)

        merged_value = sum_values.to_frame(name="sum_values")
        merged_value["sum_std_dev"] = sum_std_dev
        merged_value["index"] = ind_map["index"]

        bus_append[merged_value.index, BUS_MEAS_PPCI_IX[meas_type]["VALUE"]] = merged_value.sum_values
        bus_append[merged_value.index, BUS_MEAS_PPCI_IX[meas_type]["STD"]] = merged_value.sum_std_dev
        bus_append[merged_value.index, BUS_MEAS_PPCI_IX[meas_type]["IDX"]] = merged_value["index"]


def _add_rated_power_information_af_wls(net, ppci):
    cluster_list_loads = net.load["type"].unique()
    cluster_list_gen = net.sgen["type"].unique()
    cluster_list_tot = np.concatenate((cluster_list_loads, cluster_list_gen), axis=0)
    ppci["clusters"] = cluster_list_tot
    num_clusters = len(cluster_list_tot)
    num_buses = ppci["bus"].shape[0]
    ppci["rated_power_clusters"] = np.zeros([num_buses, 4 * num_clusters])
    for var in ["load", "sgen"]:
        in_service = net[var]["in_service"]
        active_elements = net[var][in_service]
        bus = net._pd2ppc_lookups["bus"][active_elements.bus].astype(int)
        P = active_elements.p_mw.values / ppci["baseMVA"]
        Q = active_elements.q_mvar.values / ppci["baseMVA"]
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
            ppci["rated_power_clusters"][bus_i, cluster_i + 2 * num_clusters] += abs(
                0.03 * P_i)  # std dev cluster variability hardcoded, think how to change it
            ppci["rated_power_clusters"][bus_i, cluster_i + 3 * num_clusters] += abs(
                0.03 * Q_i)  # std dev cluster variability hardcoded, think how to change it


def _add_measurements_to_ppci(net, ppci, zero_injection, algorithm):
    """
       Add measurement data from pandapower to the internal ppci structure.
       Extends ppci with additional columns for measurement values, standard deviations, and indices.

       Parameters:
       - net: pandapower net object
       - ppci: internal ppci dictionary (result of _pd2ppc)
       - zero_injection: string with desired option for zero injection measurement creation
       - algorithm: estimator algorithm name

       Returns:
       - ppci: updated ppci dictionary with additional measurement data columns
    """
    meas = net.measurement.copy(deep=True)
    if meas.empty:
        raise Exception("No measurements are available in pandapower Network! Abort estimation!")

    # Convert power measurements (p, q) to per unit (p.u.)
    meas.loc[meas.measurement_type == "p", ["value", "std_dev"]] /= ppci["baseMVA"]
    meas.loc[meas.measurement_type == "q", ["value", "std_dev"]] /= ppci["baseMVA"]

    # Convert current (i) measurements to p.u.
    i_meas = meas.query("measurement_type=='i'")
    if not i_meas.empty:
        # Convert side from string to bus id
        i_meas["side"] = i_meas.apply(lambda row:
                                      net['line'].at[row["element"], row["side"] + "_bus"] if
                                      row["side"] in ("from", "to") else
                                      net[row["element_type"]].at[row["element"], row["side"] + '_bus'] if
                                      row["side"] in ("hv", "mv", "lv") else row["side"], axis=1)
        base_i_ka = ppci["baseMVA"] / i_meas.side.map(net.bus.vn_kv)
        meas.loc[i_meas.index, "value"] /= base_i_ka / np.sqrt(3)
        meas.loc[i_meas.index, "std_dev"] /= base_i_ka / np.sqrt(3)

    # Convert angle measurements (va) from degrees to radians
    meas_dg_mask = (meas.measurement_type == 'va')
    if not meas[meas_dg_mask].empty:
        meas.loc[meas_dg_mask, "value"] = np.deg2rad(meas.loc[meas_dg_mask, "value"])
        meas.loc[meas_dg_mask, "std_dev"] = np.deg2rad(meas.loc[meas_dg_mask, "std_dev"])

    # Get bus mapping from pandapower to ppc index
    map_bus = net["_pd2ppc_lookups"]["bus"]
    meas_bus = meas[(meas['element_type'] == 'bus')]

    # Drop invalid bus measurements (those that map outside the ppci bus array)
    if (map_bus[meas_bus['element'].values.astype(np.int64)] >= ppci["bus"].shape[0]).any():
        std_logger.warning("Measurement defined in pp-grid does not exist in ppci, will be deleted!")
        meas_bus = meas_bus[map_bus[meas_bus['element'].values.astype(np.int64)] < ppci["bus"].shape[0]]

   # Create empty append array for bus measurements
    bus_append = np.full((ppci["bus"].shape[0], bus_cols_se), np.nan, dtype=ppci["bus"].dtype)

    # Add bus measurements (v, va, p, q)
    _add_measurements_to_bus(meas_bus, bus_append, map_bus)

   # Add zero injection measurements if specified
    bus_append = _add_zero_injection(net, ppci, bus_append, zero_injection)

    # Add virtual measurements for artificial buses (from open line switches)
    new_in_line_buses = np.setdiff1d(np.arange(ppci["bus"].shape[0]), map_bus[map_bus >= 0])
    bus_append[new_in_line_buses, 2] = 0.
    bus_append[new_in_line_buses, 3] = 1e-6
    bus_append[new_in_line_buses, 4] = 0.
    bus_append[new_in_line_buses, 5] = 1e-6

    # Create empty append array for branch measurements
    branch_append = np.full((ppci["branch"].shape[0], branch_cols_se), np.nan, dtype=ppci["branch"].dtype)
    br_is_mask = ppci['internal']['branch_is']

    # Add line, trafo, and trafo3w measurements
    _add_measurements_to_line(net, branch_append, meas, br_is_mask)
    _add_measurements_to_trafo(net, branch_append, meas, br_is_mask)
    _add_measurements_to_trafo3w(net, branch_append, meas, br_is_mask)

    # Integrate new measurement columns into ppci bus matrix
    if ppci["bus"].shape[1] == bus_cols:
        ppci["bus"] = np.hstack((ppci["bus"], bus_append))
    else:
        ppci["bus"][:, bus_cols: bus_cols + bus_cols_se] = bus_append

   # Integrate new measurement columns into ppci branch matrix
    if ppci["branch"].shape[1] == branch_cols:
        ppci["branch"] = np.hstack((ppci["branch"], branch_append))
    else:
        ppci["branch"][:, branch_cols: branch_cols + branch_cols_se] = branch_append

    # Add rated power information needed for AF-WLS estimator
    if algorithm == 'af-wls':
        _add_rated_power_information_af_wls(net, ppci)

    return ppci


def _add_zero_injection(net, ppci, bus_append, zero_injection):
    """
    Add zero injection labels to the ppci structure and add virtual measurements to those buses
    :param net: pandapower net
    :param ppci: generated ppci
    :param bus_append: added columns to the ppci bus with zero injection label
    :param zero_injection: parameter to control which bus to be identified as zero injection
        - None: no zero injection buses added
        - "aux_bus": only auxiliary buses created in ppc 
        - "no_inj_bus": aux buses + buses without load, gen, sgen, etc.
        - "zero_pwr_bus": aux buses + all buses with a zero power (also if there is load, sgen, etc.)
    :return bus_append: added columns
    """
    bus_append[:, ZERO_INJ_FLAG] = False
    if zero_injection is not None:
        # identify aux bus as zero injection
        if net._pd2ppc_lookups['aux']:
            aux_bus_lookup = np.concatenate([v for k, v in net._pd2ppc_lookups['aux'].items() if k != 'xward'])
            aux_bus = net._pd2ppc_lookups['bus'][aux_bus_lookup]
            aux_bus = aux_bus[aux_bus < ppci["bus"].shape[0]]
            bus_append[aux_bus, ZERO_INJ_FLAG] = True

        if isinstance(zero_injection, str):
            if zero_injection in ['zero_pwr_bus', 'no_inj_bus']:
                # identify all buses with zero power and no pq measurements as zero injection
                zero_inj_bus_mask = (ppci["bus"][:, 1] == 1) & (ppci["bus"][:, 2:4] == 0).all(axis=1) & \
                                    np.isnan(bus_append[:, P:(Q_STD + 1)]).all(axis=1)
                bus_append[zero_inj_bus_mask, ZERO_INJ_FLAG] = True
                if zero_injection == 'no_inj_bus':
                    b = np.array([], dtype=np.int64)
                    pq_elements = ["load", "motor", "sgen", "storage", "ward", "xward", 
                                   "asymmetric_load", "asymmetric_sgen"]
                    bus_lookup = net["_pd2ppc_lookups"]["bus"]
                    for element in pq_elements:
                        tab = net[element]
                        if len(tab) == 0:
                            continue
                        in_service = (tab["in_service"]) & (net.bus["in_service"][tab["bus"].values])
                        b = np.hstack([b, tab["bus"][in_service]])
                    active_buses = np.unique(b)
                    active_buses = bus_lookup[active_buses]
                    bus_append[active_buses, ZERO_INJ_FLAG] = False
            elif zero_injection != "aux_bus":
                raise UserWarning("zero injection parameter is not correctly initialized")
        elif hasattr(zero_injection, '__iter__'):
            zero_inj_bus = net._pd2ppc_lookups['bus'][zero_injection]
            bus_append[zero_inj_bus, ZERO_INJ_FLAG] = True

        zero_inj_bus = np.argwhere(bus_append[:, ZERO_INJ_FLAG]).ravel()
        bus_append[zero_inj_bus, P] = 0
        bus_append[zero_inj_bus, P_STD] = ZERO_INJECTION_STD_DEV
        bus_append[zero_inj_bus, P_IDX] = -1
        bus_append[zero_inj_bus, Q] = 0
        bus_append[zero_inj_bus, Q_STD] = ZERO_INJECTION_STD_DEV
        bus_append[zero_inj_bus, Q_IDX] = -1
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
    # i_degree_line_f_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + IA_FROM])
    # i_degree_line_t_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + IA_TO])

    # piece together our measurement vector z
    z = np.concatenate((ppci["bus"][p_bus_not_nan, bus_cols + P],
                        ppci["bus"][q_bus_not_nan, bus_cols + Q],
                        ppci["branch"][p_line_f_not_nan, branch_cols + P_FROM],
                        ppci["branch"][q_line_f_not_nan, branch_cols + Q_FROM],
                        ppci["branch"][p_line_t_not_nan, branch_cols + P_TO],
                        ppci["branch"][q_line_t_not_nan, branch_cols + Q_TO],
                        ppci["bus"][v_bus_not_nan, bus_cols + VM],
                        ppci["bus"][v_degree_bus_not_nan, bus_cols + VA],
                        ppci["branch"][i_line_f_not_nan, branch_cols + IM_FROM],
                        ppci["branch"][i_line_t_not_nan, branch_cols + IM_TO]
                        )).real.astype(np.float64)
    imag_meas = np.concatenate((np.zeros(sum(p_bus_not_nan)),
                                np.zeros(sum(q_bus_not_nan)),
                                np.zeros(sum(p_line_f_not_nan)),
                                np.zeros(sum(q_line_f_not_nan)),
                                np.zeros(sum(p_line_t_not_nan)),
                                np.zeros(sum(q_line_t_not_nan)),
                                np.zeros(sum(v_bus_not_nan)),
                                np.zeros(sum(v_degree_bus_not_nan)),
                                np.ones(sum(i_line_f_not_nan)),
                                np.ones(sum(i_line_t_not_nan))
                                )).astype(bool)
    if ppci.algorithm == "af-wls":
        balance_eq_meas = np.zeros(ppci["rated_power_clusters"].shape[0]).astype(np.float64)
        af_vmeas = 0.4 * np.ones(len(ppci["clusters"]))
        z = np.concatenate(
            (z, balance_eq_meas[ppci.non_slack_bus_mask], balance_eq_meas[ppci.non_slack_bus_mask], af_vmeas))
        imag_meas = np.concatenate((imag_meas,
                                    np.zeros(2 * balance_eq_meas[ppci.non_slack_bus_mask].shape[0]),
                                    np.zeros(af_vmeas.shape[0]))).astype(bool)
    idx_non_imeas = np.flatnonzero(~imag_meas)

    if not update_meas_only:
        # conserve the pandapower indices of measurements in the ppci order
        pp_meas_indices = np.concatenate((ppci["bus"][p_bus_not_nan, bus_cols + P_IDX],
                                          ppci["bus"][q_bus_not_nan, bus_cols + Q_IDX],
                                          ppci["branch"][p_line_f_not_nan, branch_cols + P_FROM_IDX],
                                          ppci["branch"][q_line_f_not_nan, branch_cols + Q_FROM_IDX],
                                          ppci["branch"][p_line_t_not_nan, branch_cols + P_TO_IDX],
                                          ppci["branch"][q_line_t_not_nan, branch_cols + Q_TO_IDX],
                                          ppci["bus"][v_bus_not_nan, bus_cols + VM_IDX],
                                          ppci["bus"][v_degree_bus_not_nan, bus_cols + VA_IDX],
                                          ppci["branch"][i_line_f_not_nan, branch_cols + IM_FROM_IDX],
                                          ppci["branch"][i_line_t_not_nan, branch_cols + IM_TO_IDX],
                                          )).real.astype(np.int64)
        # Covariance matrix R
        r_cov = np.concatenate((ppci["bus"][p_bus_not_nan, bus_cols + P_STD],
                                ppci["bus"][q_bus_not_nan, bus_cols + Q_STD],
                                ppci["branch"][p_line_f_not_nan, branch_cols + P_FROM_STD],
                                ppci["branch"][q_line_f_not_nan, branch_cols + Q_FROM_STD],
                                ppci["branch"][p_line_t_not_nan, branch_cols + P_TO_STD],
                                ppci["branch"][q_line_t_not_nan, branch_cols + Q_TO_STD],
                                ppci["bus"][v_bus_not_nan, bus_cols + VM_STD],
                                ppci["bus"][v_degree_bus_not_nan, bus_cols + VA_STD],
                                ppci["branch"][i_line_f_not_nan, branch_cols + IM_FROM_STD],
                                ppci["branch"][i_line_t_not_nan, branch_cols + IM_TO_STD],
                                )).real.astype(np.float64)
        meas_mask = {"pbus" : np.flatnonzero(p_bus_not_nan),
                     "qbus" : np.flatnonzero(q_bus_not_nan),
                     "pfrom" : np.flatnonzero(p_line_f_not_nan),
                     "qfrom" : np.flatnonzero(q_line_f_not_nan),
                     "pto" : np.flatnonzero(p_line_t_not_nan),
                     "qto" : np.flatnonzero(q_line_t_not_nan),
                     "vm" : np.flatnonzero(v_bus_not_nan),
                     "va" : np.flatnonzero(v_degree_bus_not_nan),
                     "ifrom" : np.flatnonzero(i_line_f_not_nan),
                     "ito" : np.flatnonzero(i_line_t_not_nan)}
        
        if ppci.algorithm == "af-wls":
            num_clusters = len(ppci["clusters"])
            P_balance_dev_std = np.sqrt(
                np.sum(np.square(ppci["rated_power_clusters"][:, 2 * num_clusters:3 * num_clusters]), axis=1))
            Q_balance_dev_std = np.sqrt(
                np.sum(np.square(ppci["rated_power_clusters"][:, 3 * num_clusters:4 * num_clusters]), axis=1))
            af_vmeas_dev_std = 0.15 * np.ones(len(ppci["clusters"]))
            r_cov = np.concatenate(
                (r_cov, P_balance_dev_std[ppci.non_slack_bus_mask], Q_balance_dev_std[ppci.non_slack_bus_mask],
                 af_vmeas_dev_std))
            meas_mask["pbalance"] = np.flatnonzero(ppci.non_slack_bus_mask)
            meas_mask["qbalance"] = np.flatnonzero(ppci.non_slack_bus_mask)
            meas_mask["afactor"] = np.arange(num_clusters)

        return z, pp_meas_indices, r_cov, meas_mask, idx_non_imeas
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
        self.non_slack_buses = np.argwhere(ppci["bus"][:, pypower_BUS_TYPE] != 3).ravel()
        self.non_slack_bus_mask = (ppci['bus'][:, pypower_BUS_TYPE] != 3).ravel()
        self.num_non_slack_bus = np.sum(self.non_slack_bus_mask)
        self.delta_v_bus_mask = np.r_[self.non_slack_bus_mask,
                                      np.ones(self.non_slack_bus_mask.shape[0], dtype=bool)].ravel()
        self.delta_v_bus_selector = np.flatnonzero(self.delta_v_bus_mask)

        # Iniialize measurements 
        self._initialize_meas()

        # Initialize state variable
        self.v_init = ppci["bus"][:, pypower_VM]
        self.delta_init = np.radians(ppci["bus"][:, pypower_VA])
        self.E_init = np.r_[self.delta_init[self.non_slack_bus_mask], self.v_init]
        self.v = self.v_init.copy()
        self.delta = self.delta_init.copy()
        self.E = self.E_init.copy()
        if algorithm == "af-wls":
            self.E = np.concatenate((self.E, np.full(ppci["clusters"].shape, 0.5)))

    def _initialize_meas(self):
        # calculate relevant vectors from ppci measurements
        self.z, self.pp_meas_indices, self.r_cov, self.non_nan_meas_mask, \
            self.idx_non_imeas = \
            _build_measurement_vectors(self, update_meas_only=False)
        # self.non_nan_meas_selector = np.flatnonzero(self.non_nan_meas_mask)

    def update_meas(self):
        self.z = _build_measurement_vectors(self, update_meas_only=True)

    @property
    def V(self):
        return self.v * np.exp(1j * self.delta)

    def reset(self):
        self.v, self.delta, self.E = \
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