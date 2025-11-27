from typing import Union, List, Dict
from copy import deepcopy

import pandas as pd
import numpy as np

from pandapower import pandapowerNet
from pandapower.analysis.sensitivity_dc import run_dc_profile
from pandapower.analysis.utils import _get_bus_lookup, _get_branch_lookup, _get_trafo3w_lookup, \
    branch_dict_to_ppci_branch_list, _get_source_bus_ix, DISCONNECTED_PADDING_VALUE, BR_SIDE_MAPPING, BR_SIDE_MAPPING_1, \
    ELE_IX_TYPE
from pandapower.run import rundcpp
from pandapower.create import create_load
from pandapower.pd2ppc import _pd2ppc

# replace pandapower makePTDF with custom function
from pandapower.pypower.makePTDF import makePTDF

import logging
logger = logging.getLogger(__name__)

def _get_PTDF_direct(
    net, source_bus=None, result_side=0, using_sparse_solver=True, random_verify=True, branch_dict=None, reduced=True
):
    """
    this function calculates PTDF (ratio without unit) of bus to a pp branch
    with matrix based internal calculation.
    """
    if net.bus.shape[0] > 3000 and not using_sparse_solver:
        logger.warning("Calculating ptdf for large network, please use sparse_solver for better numerical stability!")

    # If branch_dict not None compute list of ppci branch indices and its branch type intervals as lookup
    branch_ppci_lookup = None
    branch_id = None
    if branch_dict is not None:
        branch_id, branch_ppci_lookup = branch_dict_to_ppci_branch_list(net=net, branch_dict=branch_dict)
    else:
        reduced = False

    ptdf_ppci, _ = _makePTDF_ppci(
        net, using_sparse_solver=using_sparse_solver, result_side=result_side, branch_id=branch_id, reduced=reduced
    )

    # Use lookup to convert ppci ptdf to pp
    if reduced:
        ptdf_pp_np = _PTDF_ppci_to_pp(net, ptdf_ppci, result_side=result_side, branch_ppci_lookup=branch_ppci_lookup)
    else:
        ptdf_pp_np = _PTDF_ppci_to_pp(net, ptdf_ppci, result_side=result_side)

    # Convert numpy array to pandas dataframe with the pp element index
    # All bus data points are available no definition of perturb bus needed
    ptdf = _PTDF_pp_np_to_df(net, ptdf_pp_np, source_bus=None, branch_dict=branch_dict, reduced=reduced)

    # Select only required source buses
    source_bus = _get_source_bus_ix(net, source_bus)
    for key in ptdf.keys():
        ptdf[key] = ptdf[key].loc[:, source_bus]

    # Verify with random selection of bus
    if random_verify and source_bus.size >= 3:
        # Skip test if too few elements are calculated
        # Select three random buses and verify against perturb method
        verify_bus = np.random.choice(ptdf["line"].columns.values, 3, replace=False)
        verify_PTDF(
            net,
            source_bus=verify_bus,
            result_side=result_side,
            ptdf={key: data.loc[:, verify_bus] for key, data in ptdf.items()},
        )
    return ptdf


def _get_PTDF_perturb(net, source_bus=None, result_side=0, distributed_slack=True):
    """
    this function calculates PTDF (ratio without unit) of bus to
    a pp branch with perturb method (brute-force)
    """
    THIS_BR_SIDE_MAPPING = BR_SIDE_MAPPING if result_side == 0 else BR_SIDE_MAPPING_1
    # ToDo: remove distributed slack option

    source_bus = _get_source_bus_ix(net, source_bus)

    # Init ptdf numpy array
    ptdf_pp_np = _init_PTDF_pp_np(net, source_bus.shape[0])

    rundcpp(net, distributed_slack=distributed_slack)
    # Using new net_mod object to do perturb
    net_mod = deepcopy(net)
    for ix, bus_ix in enumerate(source_bus):
        create_load(net_mod, bus_ix, p_mw=-1)
        rundcpp(net_mod, distributed_slack=distributed_slack)
        # Delete the new load
        net_mod.load = net_mod.load.iloc[:-1, :]

        for br_type in ("line", "dcline", "trafo", "impedance"):
            if not net[br_type].empty:
                value_type = "p_" + THIS_BR_SIDE_MAPPING[br_type] + "_mw"
                ptdf_pp_np[br_type][:, ix] = (
                    net_mod["res_" + br_type][value_type].to_numpy() - net["res_" + br_type][value_type].to_numpy()
                )
        if not net.trafo3w.empty:
            for side in ("hv", "mv", "lv"):
                value_type = "p_" + side + "_mw"
                ptdf_pp_np["trafo3w_" + side][:, ix] = (
                    net_mod["res_trafo3w"][value_type].to_numpy() - net["res_trafo3w"][value_type].to_numpy()
                )

    # Convert numpy array to pandas dataframe with the pp element index
    ptdf = _PTDF_pp_np_to_df(net, ptdf_pp_np, source_bus=source_bus)
    return ptdf


def _get_dc_profile_with_PTDF(net, profiles, result_side=0, ptdf=None):
    """
    Run dc profile with ptdf method, if ptdf not given will be recalculated
    :return: {branch_type ("line", "trafo", "impedance", "trafo3w_{hv,mv,lv}"):
              DataFrame(data=p_side_mw, index=calc_ix, columns=branch_index)}
    """

    THIS_RES_BR_SIDE_MAPPING = BR_SIDE_MAPPING if result_side == 0 else BR_SIDE_MAPPING_1
    slack_df, _ = get_dist_slack(net, pf_required=True)

    if ptdf is None:
        ptdf = _get_PTDF_direct(net, result_side=result_side)

    net_mod = deepcopy(net)
    num_calc = None

    # Check profile integrity and calculate to delta profile
    delta_profiles = {}
    for key in profiles.keys():
        assert key in (("load", "p_mw"), ("sgen", "p_mw"), ("gen", "p_mw"), ("storage", "p_mw")), (
            str(key) + " not supported for superposition ptdf"
        )
        assert isinstance(profiles[key], pd.DataFrame), "Only profile as pandas dataframe supported!"
        ele_type, value_type = key

        if num_calc is None:
            num_calc = profiles[key].shape[0]
        else:
            assert num_calc == profiles[key].shape[0], str(key) + " profile has wrong dimension"

        # Update network with profiles calc_ix 0
        net_mod[ele_type].loc[profiles[key].columns.to_numpy(), value_type] = profiles[key].iloc[0, :].to_numpy()

        # Create differential profile
        delta_profiles[key] = profiles[key].copy()
        if key[0] == "gen":
            # Exclude gen slack from the profile
            gen_slack = slack_df.query("ele_type=='gen'")
            if not gen_slack.empty:
                logger.info("Gen slacks will be excluded from profile simulation!")
                delta_profiles[key].loc[:, gen_slack["ele_id"].to_numpy()] = 0.0
        delta_profiles[key].to_numpy()[:] -= delta_profiles[key].to_numpy()[0, :]

    # Calculate delta_bus_p profile for profile simulation
    rundcpp(net_mod, distributed_slack=True)
    delta_bus_p = pd.DataFrame(
        np.zeros((num_calc, net_mod.bus.shape[0]), dtype=np.float64),
        index=np.arange(num_calc),
        columns=net_mod.bus.index.to_numpy(),
    )
    required_bus_mask = np.zeros(net.bus.shape[0], dtype=bool)
    for (ele_type, value_type), this_delta_profile in delta_profiles.items():
        this_ele_ix = this_delta_profile.columns.to_numpy()
        this_bus_ix = net_mod[ele_type].loc[this_ele_ix, "bus"].to_numpy()
        sign_corr = -1 if ele_type in LOAD_REFRENCE else 1
        delta_bus_p.loc[:, this_bus_ix] += delta_profiles[(ele_type, value_type)].to_numpy() * sign_corr

        # Update required bus mask
        required_bus_mask[np.isin(net.bus.index.to_numpy(), this_bus_ix)] = True

    # Subsets only the required value in delta p
    required_bus_ix = net.bus.index.to_numpy()[required_bus_mask]
    delta_bus_p = delta_bus_p.loc[:, required_bus_ix]

    # Calculate branch flow with ptdf and delta_bus_p profile
    # No extra initialization needed
    res_pp_np = {}
    for br_type in ptdf.keys():
        if br_type.startswith("trafo3w"):
            ele_type = "trafo3w"
            side = br_type.split("_")[1]
        else:
            ele_type = br_type
            side = THIS_RES_BR_SIDE_MAPPING[ele_type]
        ptdf_this_br = ptdf[br_type]
        assert np.all(np.isin(required_bus_ix, ptdf_this_br.columns.to_numpy())), (
            "Some bus required for profile simulation not available in ptdf!"
        )
        ptdf_this_br = ptdf_this_br.loc[:, required_bus_ix]

        br_p0 = net_mod["res_" + ele_type]["p_" + side + "_mw"].to_numpy()
        res_pp_np[br_type] = np.tile(br_p0, (num_calc, 1)) + np.matmul(
            delta_bus_p.to_numpy(), ptdf_this_br.to_numpy().T
        )

    res = _profile_pp_np_to_df(net, res_pp_np, num_calc)
    return res


# Convert data in numpy array to pandas dataframe with pp index
def _PTDF_pp_np_to_df(net, res_pp, source_bus=None, nan_to_num=True, branch_dict=None, reduced=False):
    res = {}
    for br_type, data in res_pp.items():
        if nan_to_num:
            data = np.nan_to_num(data)

        pp_br_type = "trafo3w" if br_type.startswith("trafo3w") else br_type
        if branch_dict is not None and not reduced:
            branch_complement = [x for x in range(data.shape[0]) if x not in branch_dict[pp_br_type]]
            data[branch_complement, :] = np.NaN
        if source_bus is None:
            if reduced:
                res[br_type] = pd.DataFrame(data=data, index=branch_dict[pp_br_type], columns=net.bus.index.to_numpy())
            else:
                res[br_type] = pd.DataFrame(
                    data=data, index=net[pp_br_type].index.to_numpy(), columns=net.bus.index.to_numpy()
                )
        else:
            if reduced:
                res[br_type] = pd.DataFrame(data=data, index=branch_dict[pp_br_type], columns=source_bus)
            else:
                res[br_type] = pd.DataFrame(data=data, index=net[pp_br_type].index.to_numpy(), columns=source_bus)
    return res


# Init result numpy array filled with zeros
def _init_PTDF_pp_np(net, num_source_bus):
    ptdf_pp = {}
    for br_type in ("line", "dcline", "trafo", "impedance"):
        if not net[br_type].empty:
            ptdf_pp[br_type] = np.zeros((net[br_type].shape[0], num_source_bus), dtype=np.float)
    if not net.trafo3w.empty:
        for side in ("hv", "mv", "lv"):
            ptdf_pp["trafo3w_" + side] = np.zeros((net.trafo3w.shape[0], num_source_bus), dtype=np.float)
    return ptdf_pp


def _makePTDF_ppci(net, using_sparse_solver, result_side, branch_id=None, reduced=False):
    # Select subnet areas
    slack_df, pp_area_bus_mapping = get_dist_slack(net)
    _, ppci = _pd2ppc(net)
    # Make PTDF of the ppci data stucture
    ppci_slack_mask_with_prio = get_ppci_dist_slack(net, ppci, slack_df)
    if len(pp_area_bus_mapping) > 1:
        ptdf_ppci = makePTDF_multi_area(
            net,
            ppci,
            pp_area_bus_mapping,
            ppci_slack_mask_with_prio,
            using_sparse_solver=using_sparse_solver,
            result_side=result_side,
        )
    else:
        ptdf_ppci = makePTDF(
            ppci["baseMVA"],
            ppci["bus"],
            ppci["branch"],
            slack=ppci_slack_mask_with_prio,
            using_sparse_solver=using_sparse_solver,
            result_side=result_side,
            branch_id=branch_id,
            reduced=reduced,
        )
    return ptdf_ppci, ppci


def _PTDF_ppci_to_pp(net, ptdf_ppci, result_side, branch_ppci_lookup=None):
    # Padding the sensitivity matrix for out-of-service elements
    ptdf_ppci_padding = np.pad(ptdf_ppci, ((0, 1), (0, 1)), mode="constant", constant_values=DISCONNECTED_PADDING_VALUE)

    # Get bus pp ppci lookup
    pp_ppci_bus_lookup = _get_bus_lookup(net)

    results = dict()
    # Get branch pp ppci lookup and update the matrix
    for br_type in ("line", "trafo", "impedance"):
        if branch_ppci_lookup is not None:
            if br_type in branch_ppci_lookup.keys():
                pp_ppci_br_lookup = range(branch_ppci_lookup[br_type][0], branch_ppci_lookup[br_type][1])
            else:
                pp_ppci_br_lookup = None
        else:
            pp_ppci_br_lookup = _get_branch_lookup(net, br_type)
        if pp_ppci_br_lookup is not None:
            results[br_type] = ptdf_ppci_padding[pp_ppci_br_lookup, :][:, pp_ppci_bus_lookup[net.bus.index.to_numpy()]]

    # Trafo3w needs to be handled differently
    if branch_ppci_lookup is not None:
        pp_ppci_trafo3w_lookups = {
            type: range(branch_ppci_lookup[type][0], branch_ppci_lookup[type][1])
            for type in ("trafo3w_hv", "trafo3w_mv", "trafo3w_lv")
        }
    else:
        pp_ppci_trafo3w_lookups = _get_trafo3w_lookup(net)
    if pp_ppci_trafo3w_lookups is not None:
        for trafo3w_side in pp_ppci_trafo3w_lookups.keys():
            results[trafo3w_side] = ptdf_ppci_padding[pp_ppci_trafo3w_lookups[trafo3w_side], :][
                :, pp_ppci_bus_lookup[net.bus.index.to_numpy()]
            ]
            if result_side == 0:
                # Sign correction only for "mv", "lv" side
                if not trafo3w_side.endswith("hv"):
                    results[trafo3w_side] *= -1
            else:
                # Sign correction only for "hv" side
                if trafo3w_side.endswith("hv"):
                    results[trafo3w_side] *= -1
    return results


def run_PTDF(
    net: pandapowerNet,
    source_bus: ELE_IX_TYPE = None,
    distributed_slack: bool = True,
    result_side=0,
    perturb: bool = False,
    using_sparse_solver: bool = True,
    random_verify: bool = False,
    branch_dict: Dict[str, Union[List[int], None]] = None,
    reduced: bool = True,
):
    """
    this function is a wrapper of calculating PTDF (ratio without unit) of bus to a pp branch
    with matrix based internal calculation or perturb.
    The PTDF is defined as: (p_{side}_mw_new - p_{side}_mw_old) / delta_P_injection
    delta_P_injection means the injection at a bus increases or the load decreases
    Side corresponds to the pandapower results definition (see result_side definition)

    :param net: A pandapower network
    :param source_bus: Select a subset of pp buses for the PTDF calculation, if None given then all buses are used
    :param result_side: 0 means ("from", "hv") side, 1 means ("to", "lv") side of p_{side}_mw
    :param distributed_slack: Set True if p distribution amount distributed wished, or else slacks are
         only all voltage references! For non-perturb only True possible!!
    :param perturb: Set True to use the perturb version (brute-force) which is faster for calculating
        only a few elements on large networks, if a lot of elements required please set to False
    :param using_sparse_solver: Select whether sparse linear system should be used (more efficient for large network)
    :param random_verify: Set to True to check the direct version against perturb version
        with 3 randomly selected elements
    :param: branch_dict: dictionary with keys "line", "trafo", "impedance", "trafo3w"; if not None the computation is
        restricted to the branch indices given in the dict
    :param reduced: if True, the output is reduced to the branches given in branch_dict
    :return: {goal_branch_type ("line", "trafo", "impedance", "trafo3w_{hv,mv,lv}"):
        DataFrame(data=ptdf, index=goal_branch_pp_index, columns=bus_pp_index)}
    """
    if perturb and source_bus is None:
        logger.info("If a lot of buses required in ptdf, please set perturb to False!")

    # ToDo: Check distributed slack option here
    if perturb:  # or not distributed_slack:
        # if not distributed_slack:
        #     logger.info("distributed_slack deactivated! Distirbuted slacks are used as Vref! Only Perturb Possible")
        ptdf = _get_PTDF_perturb(
            net, source_bus=source_bus, result_side=result_side, distributed_slack=distributed_slack
        )
    else:
        ptdf = _get_PTDF_direct(
            net,
            source_bus=source_bus,
            result_side=result_side,
            using_sparse_solver=using_sparse_solver,
            random_verify=random_verify,
            branch_dict=branch_dict,
            reduced=reduced,
        )

    # Update ptdf on net
    net._ptdf = {"bus": None}
    for br_type, data in ptdf.items():
        if net._ptdf["bus"] is None:
            net._ptdf["bus"] = data.columns.to_numpy(copy=True)
        net["ptdf_" + br_type] = data
    return ptdf


def verify_PTDF(net, source_bus: ELE_IX_TYPE = None, result_side=0, using_sparse_solver=True, ptdf=None):
    """
    this function verifies the result of PTDF and perturb method,
    raise AssertionError on mismatches!
    """
    net = deepcopy(net)
    # ToDo: Verify what the distributed_slack options does for both functions (perturb and classic)
    if ptdf is None:
        ptdf = run_PTDF(
            net,
            source_bus=source_bus,
            result_side=result_side,
            using_sparse_solver=using_sparse_solver,
            perturb=False,
            random_verify=False,
            distributed_slack=False,
        )
    ptdf_perturb = run_PTDF(net, source_bus=source_bus, result_side=result_side, distributed_slack=False, perturb=True)

    assert len(ptdf) > 0, "Empty ptdf, verification not possible!"
    for key in ptdf.keys():
        assert np.allclose(ptdf[key], ptdf_perturb[key], atol=1e-8, equal_nan=True), (
            f"{key} PTDF results verification failed!"
        )
        logger.info(str(key) + " PTDF results verified!")
    logger.info("All PTDF results verified with perturb method!")


def verify_dc_profile_with_PTDF(net, profiles: dict, result_side=0, ptdf=None):
    """
    this function verifies the result of run profile with PTDF and perturb method,
    raise AssertionError on mismatches!
    """
    # ToDo: Verify what the distributed_slack options does for both functions (perturb and classic)
    res_profile_ptdf = run_dc_profile(
        net, profiles, result_side=result_side, perturb=False, ptdf=ptdf, distributed_slack=False
    )
    res_profile_perturb = run_dc_profile(net, profiles, result_side=result_side, distributed_slack=False, perturb=True)

    assert len(res_profile_ptdf) > 0, "Empty result profile, verification not possible!"
    for key in res_profile_ptdf.keys():
        assert np.allclose(res_profile_ptdf[key], res_profile_perturb[key], atol=1e-8), f"{key} verification failed!"
        logger.info(str(key) + " profile verified!")
    logger.info("Run dc profile with PTDF verified!")

