# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from typing import Union, Tuple
import numpy as np
import pandas as pd
import pandapower as pp
from pandapower.auxiliary import pandapowerNet

import logging

logger = logging.getLogger(__name__)

DISCONNECTED_PADDING_VALUE = np.nan
BR_SIDE_MAPPING = {"line": "from", "dcline": "from", "trafo": "hv", "impedance": "from", "trafo3w": "hv"}
BR_SIDE_MAPPING_1 = {"line": "to", "dcline": "to", "trafo": "lv", "impedance": "to", "trafo3w": "mv"}
BR_PTDF_MAPPING = {"line": "", "dcline": "", "trafo": "", "impedance": "", "trafo3w": "_hv"}
BR_PTDF_MAPPING_1 = {"line": "", "dcline": "", "trafo": "", "impedance": "", "trafo3w": "_mv"}
BR_NAN_CHECK = {
    "line": "va_from_degree",
    "dcline": "va_from_degree",
    "trafo": "va_hv_degree",
    "impedance": "i_from_ka",
    "trafo3w": "va_hv_degree",
}
LOAD_REFRENCE = ("load", "storage")
ELE_IX_TYPE = Union[int, list, np.ndarray]
PP_SLACK_PRIO_COL = "slack_weight"


def _get_source_bus_ix(
        net: pandapowerNet,
        source_bus: Union[int, np.ndarray] | None = None
):
    if source_bus is None:
        return net.bus.index.to_numpy()

    if np.isscalar(source_bus):
        source_bus = np.array([source_bus]).astype(int)
    if isinstance(source_bus, np.ndarray):
        # Convert to 1d np array
        source_bus = source_bus.ravel()
    else:
        source_bus = np.array([source_bus]).ravel()

    unique_source_bus = np.unique(source_bus)
    return unique_source_bus if unique_source_bus.size < source_bus.size else source_bus


def _get_outage_branch_ix(
        net: pandapowerNet,
        outage_branch_type: str,
        outage_branch_ix: np.ndarray | None = None
) -> np.ndarray:
    assert outage_branch_type in ("line", "dcline", "trafo", "impedance", "trafo3w"), (
        outage_branch_type + " as outage branch type not supported!"
    )
    assert not net[outage_branch_type].empty, outage_branch_type + " is empty, outage test not possible!"

    if outage_branch_ix is None:
        outage_branch_ix = net[outage_branch_type].index.to_numpy()
    elif np.isscalar(outage_branch_ix):
        outage_branch_ix = np.array([outage_branch_ix]).astype(int)
    elif isinstance(outage_branch_ix, np.ndarray):
        outage_branch_ix = outage_branch_ix.ravel()
    else:
        # if index in list/tuple or similar data structures
        outage_branch_ix = np.array(outage_branch_ix).ravel()

    unique_outage_branch_ix = np.unique(outage_branch_ix)
    return unique_outage_branch_ix if unique_outage_branch_ix.size < outage_branch_ix.size else outage_branch_ix


def _get_bus_lookup(net: pandapowerNet) -> np.ndarray:
    pp_ppci_bus_lookup = net._pd2ppc_lookups["bus"]
    # Set out-of-service bus index to -1 (for padded array)
    bus_in_service_mask = np.in1d(np.arange(pp_ppci_bus_lookup.shape[0]), net._is_elements["bus_is_idx"])
    pp_ppci_bus_lookup[~bus_in_service_mask] = -1
    return pp_ppci_bus_lookup


def _get_branch_lookup(net: pandapowerNet, branch_type) -> np.ndarray | None:
    # Find the branch lookup table from pandapower net of ppci layer
    assert branch_type in ("line", "trafo", "trafo3w", "impedance"), "Branch Type not supported for lookup creation"

    if branch_type in net["_pd2ppc_lookups"]["branch"]:
        br_ix_start, br_ix_end = net["_pd2ppc_lookups"]["branch"][branch_type]

        branch_in_service_mask = net["_ppc"]["internal"]["branch_is"][br_ix_start:br_ix_end]
        ppci_ix_start_offset = np.sum(net["_ppc"]["internal"]["branch_is"][:br_ix_start]) if br_ix_start > 0 else 0
        num_active_branch = np.sum(branch_in_service_mask)

        # Initialize branch lookups as empty integer array
        pp_ppci_br_lookup = np.zeros(br_ix_end - br_ix_start, dtype=int)
        # Find lookup index of in_service branch
        pp_ppci_br_lookup[branch_in_service_mask] = np.arange(
            ppci_ix_start_offset, ppci_ix_start_offset + num_active_branch
        )
        # Set out_of_service branch index to -1 (for padded array)
        pp_ppci_br_lookup[~branch_in_service_mask] = -1
        return pp_ppci_br_lookup.astype(int)
    else:
        return None


def _get_trafo3w_lookup(net: pandapowerNet) -> dict | None:
    pp_ppci_trafo3w_lookup = _get_branch_lookup(net, "trafo3w")
    if pp_ppci_trafo3w_lookup is not None:
        trafo3w_keys = ["trafo3w_hv", "trafo3w_mv", "trafo3w_lv"]
        num_trafo3w = net.trafo3w.shape[0]
        pp_ppci_trafo3w_lookups = {
            key: pp_ppci_trafo3w_lookup[range(num_trafo3w * ix, num_trafo3w * (ix + 1))]
            for ix, key in enumerate(trafo3w_keys)
        }
        return pp_ppci_trafo3w_lookups
    else:
        return None


def branch_dict_to_ppci_branch_list(
        net: pandapowerNet,
        branch_dict: dict[str, Union[list[int], None]]
) -> Tuple[list, dict]:
    """
    This function transforms a dictionary with branches of a net into a list of the corresponding internal ppci indices
    and produces a lookup for tha branch type intervals.
    :param net: pp-net, on which a powerflow has been executed
    :param branch_dict: dictionary should include branch types as keys 'line', 'trafo', 'trafo3, 'impedance' and
                        for each key a list of indices.
    :return: list of ppci branch indices, dict for branch type ppci lookup
    """

    branch_id_ppci = []
    ppci_branch_lookup = {}
    s = 0
    t = 0
    for br_type in ("line", "trafo", "impedance", "trafo3w"):
        if branch_dict.get(br_type, None) is not None:
            branches = list(net[br_type].index)
            branch_id = [branches.index(x) for x in branch_dict[br_type]]
            t += len(branch_id)
            if br_type == "trafo3w":
                trafo3w_lookup = _get_trafo3w_lookup(net)
                for type in ["trafo3w_hv", "trafo3w_mv", "trafo3w_lv"]:
                    branch_id_ppci += list(trafo3w_lookup[type][branch_id])
                    ppci_branch_lookup[type] = [s, t]
                    s = t
                    t += len(branch_id)
            else:
                branch_id_ppci += list(_get_branch_lookup(net, br_type)[branch_id])
                ppci_branch_lookup[br_type] = [s, t]

            s = t

    return branch_id_ppci, ppci_branch_lookup


def get_dist_slack(net: pandapowerNet, pf_required: bool=True) -> Tuple[pd.DataFrame, dict]:
    """
    Find active slacks of a pp net and check multi area
    of the grid
    return: A dataframe contains info to distributed slack and
            A dict to area_bus_mapping
    """
    if pf_required:
        pp.rundcpp(net)

    slack_df = pd.DataFrame(columns=["ele_type", "ele_id", "bus_id", "priority", "new_ele_type", "new_ele_id"])

    # select all possible slacks of pp net
    all_pp_slack = {"gen": net.gen.loc[net.gen.slack == True],
                    "ext_grid": net.ext_grid}
    for ele_type, ele_slack_df in all_pp_slack.items():
        if ele_slack_df.empty:
            continue

        for ix, slack in ele_slack_df.iterrows():
            if not np.isnan(net.res_bus.at[net[ele_type].at[ix, "bus"], "va_degree"]) \
                    and net[ele_type].at[ix, "in_service"]:
                # Skip out-of-service slack
                this_priority = net[ele_type].at[ix, PP_SLACK_PRIO_COL] \
                    if PP_SLACK_PRIO_COL in net[ele_type].columns else 1.0
                # slack_df = slack_df.append({"ele_type": ele_type, "ele_id": ix,
                #                             "bus_id": slack.bus, "priority": this_priority,
                #                             "new_ele_type": "", "new_ele_id":-1}, ignore_index=True)
                slack_df = pd.concat([slack_df,
                                      pd.DataFrame({"ele_type": ele_type, "ele_id": ix,
                                                    "bus_id": slack.bus, "priority": this_priority,
                                                    "new_ele_type": "", "new_ele_id": -1}, index=[0])],
                                     ignore_index=True, axis=0)

    # Check slack df plausibility
    assert not slack_df.empty, "No slack in network available! Calculation not possible!"
    if slack_df['priority'].isna().any():
        logger.warning("Some slack has NaN as priority! Force priority to equally distributed!")
        slack_df['priority'] = 1.0

    # Sort and normalization
    slack_df.sort_values(by="priority", ascending=False, inplace=True)
    # Initialize area and priority in area variable
    slack_df["area"], slack_df["priority_in_area"] = 0, 0.0

    # detect multi area
    pp_area_bus_mapping = _check_multi_area(net, slack_df)
    return slack_df, pp_area_bus_mapping


def get_ppci_dist_slack(net: pandapowerNet, ppci: dict, slack_df: pd.DataFrame) -> np.ndarray:
    """ Convert the priority defined in slack_df to a numpy array required for
        pypower ptdf calculation
    """
    # Check number of slacks
    pp_slack = slack_df["bus_id"].to_numpy(dtype=int)
    assert np.all(np.isin(pp_slack, net["_is_elements"]["bus_is_idx"])), \
        "Some selected slacks are out of service"
    ppci_slack = net["_pd2ppc_lookups"]["bus"][pp_slack]
    ppci_slack_priority = slack_df["priority"].to_numpy()

    ppci_slack_mask = np.zeros(ppci["bus"].shape[0], dtype=float)
    ppci_slack_mask[ppci_slack] = ppci_slack_priority
    return ppci_slack_mask


def _check_multi_area(net: pandapowerNet, slack_df: pd.DataFrame) -> dict:
    """ Check the multi grid areas of a pandapower networks with distributed slack
        and update the area and priority area in slack_df
        return dict: {area: bus_in_area}
    """
    # Set all active slacks to out-of-service
    for ix, slack in slack_df.iterrows():
        net[slack.ele_type].at[slack.ele_id, "in_service"] = False

    area_ix = 0
    pp_area_bus_mapping = {}
    updated_slack_mask = np.zeros(slack_df.shape[0], dtype=bool)
    # Set selected slack to in-service and identify grid area
    for ix, slack in slack_df.iterrows():
        if not updated_slack_mask[ix]:
            net[slack.ele_type].at[slack.ele_id, "in_service"] = True
            pp.rundcpp(net)
            net[slack.ele_type].at[slack.ele_id, "in_service"] = False

            bus_this_area = net.bus.index.to_numpy()[~np.isnan(net.res_bus.va_degree)]
            slack_in_area = np.isin(slack_df.bus_id.to_numpy(), bus_this_area)
            updated_slack_mask[slack_in_area] = True
            slack_df.loc[slack_in_area, "area"] = area_ix
            pp_area_bus_mapping.update({area_ix: bus_this_area})
            area_ix += 1

    # Restore all active slacks to in-service
    for ix, slack in slack_df.iterrows():
        net[slack.ele_type].at[slack.ele_id, "in_service"] = True
    pp.rundcpp(net)

    # Update slack priority in area
    sum_priority_in_area = slack_df.groupby("area")["priority"].sum()
    slack_df["priority_in_area"] = 0.0
    # for i, val in sum_priority_in_area.iteritems():
    for i, val in sum_priority_in_area.items():
        slack_df.loc[slack_df.area == i, "priority_in_area"] = \
            slack_df.loc[slack_df.area == i, "priority_in_area"] / val if np.isclose(sum_priority_in_area.at[i], 0.0) else 0.0

    # slack_df["priority_in_area"] = slack_df.apply(lambda slack: slack.priority/sum_priority_in_area.at[slack.area],
    #                                               axis=1)
    return pp_area_bus_mapping
