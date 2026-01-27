from typing import Union
import numpy as np

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


def _get_source_bus_ix(net, source_bus=None):
    if source_bus is None:
        return net.bus.index.to_numpy()

    if np.isscalar(source_bus):
        source_bus = np.array([source_bus]).astype(np.int)
    if isinstance(source_bus, np.ndarray):
        # Convert to 1d np array
        source_bus = source_bus.ravel()
    else:
        source_bus = np.array([source_bus]).ravel()

    unique_source_bus = np.unique(source_bus)
    return unique_source_bus if unique_source_bus.size < source_bus.size else source_bus


def _get_outage_branch_ix(net, outage_branch_type, outage_branch_ix=None):
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


def _get_bus_lookup(net):
    pp_ppci_bus_lookup = net._pd2ppc_lookups["bus"]
    # Set out-of-service bus index to -1 (for padded array)
    bus_in_service_mask = np.in1d(np.arange(pp_ppci_bus_lookup.shape[0]), net._is_elements["bus_is_idx"])
    pp_ppci_bus_lookup[~bus_in_service_mask] = -1
    return pp_ppci_bus_lookup


def _get_branch_lookup(net, branch_type):
    # Find the branch lookup table from pandapower net of ppci layer
    assert branch_type in ("line", "trafo", "trafo3w", "impedance"), "Branch Type not supported for lookup creation"

    if branch_type in net["_pd2ppc_lookups"]["branch"]:
        br_ix_start, br_ix_end = net["_pd2ppc_lookups"]["branch"][branch_type]

        branch_in_service_mask = net["_ppc"]["internal"]["branch_is"][br_ix_start:br_ix_end]
        ppci_ix_start_offset = np.sum(net["_ppc"]["internal"]["branch_is"][:br_ix_start]) if br_ix_start > 0 else 0
        num_active_branch = np.sum(branch_in_service_mask)

        # Initialize branch lookups as empty integer array
        pp_ppci_br_lookup = np.zeros(br_ix_end - br_ix_start, dtype=np.int)
        # Find lookup index of in_service branch
        pp_ppci_br_lookup[branch_in_service_mask] = np.arange(
            ppci_ix_start_offset, ppci_ix_start_offset + num_active_branch
        )
        # Set out_of_service branch index to -1 (for padded array)
        pp_ppci_br_lookup[~branch_in_service_mask] = -1
        return pp_ppci_br_lookup.astype(int)
    else:
        return None


def _get_trafo3w_lookup(net):
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


def branch_dict_to_ppci_branch_list(net, branch_dict):
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


# All functions should be called from external
def run_dc_profile(
    net,
    profiles: dict,
    result_side=0,
    distributed_slack: bool = True,
    perturb: bool = False,
    extra_data_points: list = None,
    ptdf: dict = None,
):
    """
    this function runs a dc profile simulation with ptdf
    :param net: A pandapower network
    :param profiles: a dict of p profiles of pp elements as dataframe:
        {(element ("load", "sgen", "gen", "storage"), "p_mw"):
         pd.DataFrame(index=calculation_steps, columns=element_index, data=profile_data)}
            all the profiles must have the same index, the columns could be a subset of the element,
            the default value of not selected elements in pandapower networks is used in profile simulation
    :param result_side: 0 means ("from", "hv") side, 1 means ("to", "lv") side
    :param distributed_slack: Set True if p distribution amount distributed wished, or else slacks are
         only all voltage references! For non-perturb only True possible!!
    :param perturb: Set True to use the perturb version (brute-force) which is faster for calculating
        only a few elements on large networks, if a lot of elements required please set to False
    :param extra_data_points: Extra data points from pandapower as a list of tuples (perturb Only!)
        e.g. [("bus", "va_degree"), ("load", "p_mw")]
    :param ptdf: precalculated ptdf matrix to accelerate the calculation (Only required in the non-perturb version)
    :return: {(res_{branch_type}, p_{side}_mw):
        DataFrame(data=p_side_mw, index=calc_ix, columns=outage_branch_pp_index)}
    if extra_data_points defined, further pp data points also returned
    """
    if perturb or extra_data_points is not None or not distributed_slack:
        if extra_data_points is not None:
            logger.info(f"Extra data points: {extra_data_points} required, using perturb method!")
        if not distributed_slack:
            logger.info("distributed_slack deactivated! Distirbuted slacks are used as Vref! Only Perturb Possible")
        res = _get_dc_profile_perturb(
            net,
            profiles,
            result_side=result_side,
            distributed_slack=distributed_slack,
            extra_data_points=extra_data_points,
        )
    else:
        res = _get_dc_profile_with_PTDF(net, profiles, result_side=result_side, ptdf=ptdf)

    res_renamed = {}
    THIS_RES_BR_SIDE_MAPPING = BR_SIDE_MAPPING if result_side == 0 else BR_SIDE_MAPPING_1
    for br_type, value in res.items():
        if isinstance(br_type, str):
            if not br_type.startswith("trafo3w"):
                side = THIS_RES_BR_SIDE_MAPPING[br_type]
            else:
                side = br_type.split("_")[-1]
            res_renamed[(f"res_{br_type}", f"p_{side}_mw")] = value
        else:
            # rename extra data points
            res_renamed[(f"res_{br_type[0]}", br_type[1])] = value
    return res_renamed

