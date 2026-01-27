from typing import Union, List, Dict, Tuple
from copy import deepcopy
from itertools import product

import pandas as pd
import numpy as np

from pandapower import pandapowerNet
from pandapower.analysis.PTDF import _makePTDF_ppci, _get_PTDF_perturb
from pandapower.analysis.sensitivity_dc import run_dc_n1
from pandapower.analysis.utils import _get_branch_lookup, _get_trafo3w_lookup, \
    branch_dict_to_ppci_branch_list, _get_outage_branch_ix, DISCONNECTED_PADDING_VALUE, BR_SIDE_MAPPING, \
    BR_SIDE_MAPPING_1, BR_PTDF_MAPPING, BR_PTDF_MAPPING_1, BR_NAN_CHECK, ELE_IX_TYPE
from pandapower.run import rundcpp
from pandapower.create import create_load, create_ext_grid
from pandapower.pypower.makeLODF import makeLODF

import logging
logger = logging.getLogger(__name__)


def _get_LODF_direct(
    net,
    outage_branch_type,
    outage_branch_ix=None,
    using_sparse_solver=True,
    random_verify=True,
    branch_dict=None,
    reduced=True,
):
    """
    this function calculate LODF (ratio without unit) of a pp branch from the outage of a pp branch
    with pypower matrix function.
    """
    if net.bus.shape[0] > 3000 and not using_sparse_solver:
        logger.warning("Calculating lodf for large network, switched to sparse solver!")

    # If branch_dict not None compute list of ppci branch indices and its branch type intervals as lookup
    branch_ppci_lookup = None
    branch_id = None
    if branch_dict is not None:
        branch_id, branch_ppci_lookup = branch_dict_to_ppci_branch_list(net=net, branch_dict=branch_dict)
    else:
        reduced = False

    ptdf_ppci, ppci = _makePTDF_ppci(
        net, using_sparse_solver=using_sparse_solver, result_side=0, branch_id=branch_id, reduced=reduced
    )

    # Set super small value to 0 for better numerical stability
    ptdf_ppci[np.isclose(ptdf_ppci, 0, atol=1e-10)] = 0
    # Identify bridge branches not allowed to outage
    bridge_branch_mask = np.any(np.isclose(np.abs(ptdf_ppci), 1, atol=1e-8), axis=1) | np.all(
        np.isclose(np.nan_to_num(ptdf_ppci), 0, atol=1e-8), axis=1
    )

    # Create lodf ppci with ptdf ppci
    if reduced:
        lodf_ppci = makeLODF(ppci["branch"][branch_id], ptdf_ppci)
    else:
        lodf_ppci = makeLODF(ppci["branch"], ptdf_ppci)

    # Set results to default value of bridge branch
    lodf_ppci[:, bridge_branch_mask] = np.NaN
    if branch_id is not None and not reduced:
        branch_id_complement = [x for x in range(list(branch_ppci_lookup.values())[-1][1]) if x not in branch_id]
        lodf_ppci[:, branch_id_complement] = np.NaN
        lodf_ppci[branch_id_complement, :] = np.NaN

    # Checkout ppci lodf to pp level
    if reduced:
        lodf_pp_np = _LODF_ppci_to_pp(net, lodf_ppci, branch_ppci_lookup=branch_ppci_lookup)
    else:
        lodf_pp_np = _LODF_ppci_to_pp(net, lodf_ppci)

    # lodf pp contains all data
    # Convert numpy array to pandas dataframe with the pandapower element index
    if reduced:
        lodf = _LODF_pp_np_to_df(net, lodf_pp_np, branch_dict=branch_dict)
    else:
        lodf = _LODF_pp_np_to_df(net, lodf_pp_np)

    # Select only required data points according to the outage_branch_type
    if outage_branch_type is not None:
        outage_branch_ix = _get_outage_branch_ix(net, outage_branch_type, outage_branch_ix)
        if reduced:
            lodf = {key: value for key, value in lodf.items() if key[1] == outage_branch_type}
        else:
            lodf = {key: value.loc[:, outage_branch_ix] for key, value in lodf.items() if key[1] == outage_branch_type}

    # Verify with random selection of branches
    if random_verify and outage_branch_ix.size >= 3:
        # Skip test if too few elements are calculated
        # Select three random branches and verify against perturb method
        verify_branch_ix = np.random.choice(outage_branch_ix, 3)
        verify_LODF(
            net,
            outage_branch_type=outage_branch_type,
            outage_branch_ix=verify_branch_ix,
            lodf={key: data.loc[:, verify_branch_ix] for key, data in lodf.items()},
        )
    return lodf


def _init_LODF_pp_np(net, outage_branch_type, num_outage_branch):
    lodf_pp = {}
    for br_type in ("line", "dcline", "trafo", "impedance"):
        if not net[br_type].empty:
            lodf_pp[(br_type, outage_branch_type)] = np.zeros((net[br_type].shape[0], num_outage_branch), dtype=float)
    if not net.trafo3w.empty:
        for side in ("hv", "mv", "lv"):
            lodf_pp[("trafo3w_" + side, outage_branch_type)] = np.zeros(
                (net.trafo3w.shape[0], num_outage_branch), dtype=float
            )

    for data in lodf_pp.values():
        data[:] = np.NaN
    return lodf_pp


def _LODF_ppci_to_pp(net, lodf_ppci, branch_ppci_lookup=None):
    # convert the branch sensitivity of the ppci layer to pandapower net layer
    if branch_ppci_lookup is not None:
        pp_ppci_branch_lookups = {
            br_type: range(branch_ppci_lookup[br_type][0], branch_ppci_lookup[br_type][1])
            for br_type in ("line", "trafo", "impedance")
            if br_type in branch_ppci_lookup.keys()
        }
        pp_ppci_trafo3w_lookups = {
            type: range(branch_ppci_lookup[type][0], branch_ppci_lookup[type][1])
            for type in ("trafo3w_hv", "trafo3w_mv", "trafo3w_lv")
            if type in branch_ppci_lookup.keys()
        }
    else:
        pp_ppci_branch_lookups = {
            br_type: _get_branch_lookup(net, br_type) for br_type in ("line", "trafo", "impedance")
        }
        pp_ppci_trafo3w_lookups = _get_trafo3w_lookup(net)

    lodf_ppci_padding = np.pad(lodf_ppci, ((0, 1), (0, 1)), mode="constant", constant_values=DISCONNECTED_PADDING_VALUE)

    results = dict()
    available_branch_types = [br_type for br_type, lookup in pp_ppci_branch_lookups.items() if lookup is not None]

    for goal_br_type, source_br_type in product(available_branch_types, repeat=2):
        results[(goal_br_type, source_br_type)] = lodf_ppci_padding[pp_ppci_branch_lookups[goal_br_type], :][
            :, pp_ppci_branch_lookups[source_br_type]
        ]

    if pp_ppci_trafo3w_lookups is not None:
        # goal_trafo3w_side: ("trafo3w_hv", "trafo3w_mv", "trafo3w_lv")
        for source_br_type in available_branch_types:
            for goal_trafo3w_side in pp_ppci_trafo3w_lookups.keys():
                results[(goal_trafo3w_side, source_br_type)] = lodf_ppci_padding[
                    pp_ppci_trafo3w_lookups[goal_trafo3w_side], :
                ][:, pp_ppci_branch_lookups[source_br_type]]
    return results


def _LODF_pp_np_to_df(net, res_pp_np, outage_branch_type=None, outage_branch_ix=None, branch_dict=None):
    res = {}
    for key, data in res_pp_np.items():
        data = res_pp_np[key]

        # Avoid inf
        data[np.isinf(data)] = np.NaN
        # Find "columns" contains only NaN
        # ATTENTION: following two lines need to be commented out to neglect LODF of isolated lines
        # only_nan_mask = np.all(np.isnan(data), axis=0)
        # data[:, ~only_nan_mask] = np.nan_to_num(data[:, ~only_nan_mask])

        goal_element, source_element = key
        if outage_branch_type is not None:
            if source_element != outage_branch_type:
                # Skip unrequired data point
                continue

            if outage_branch_ix is None:
                outage_branch_ix = net[source_element].index.to_numpy()
        else:
            outage_branch_ix = net[source_element].index.to_numpy()

        goal_br_type = "trafo3w" if goal_element.startswith("trafo3w") else goal_element
        if branch_dict is not None:
            res[key] = pd.DataFrame(data=data, index=branch_dict[goal_br_type], columns=branch_dict[source_element])
        else:
            res[key] = pd.DataFrame(data=data, index=net[goal_br_type].index.to_numpy(), columns=outage_branch_ix)
    return res


def _get_LODF_perturb(
    net: pandapowerNet,
    outage_branch_type: str,
    outage_branch_ix: ELE_IX_TYPE = None,
    distributed_slack=True,
    recycle="lodf",
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """
    this function calculate LODF (ratio without unit) of a pp branch from the outage of a pp branch
    with perturb method (brute-force)
    """
    # No side selection needed
    THIS_RES_BR_SIDE_MAPPING = BR_SIDE_MAPPING
    net_mod = deepcopy(net)

    # Using net_mod for outage branch, net for original value
    rundcpp(net_mod, distributed_slack=distributed_slack, recycle=recycle)
    net_mod_n1 = deepcopy(net_mod)

    outage_branch_ix = _get_outage_branch_ix(net_mod, outage_branch_type, outage_branch_ix)

    # Init lodf array, Using Numpy array for better performance
    lodf_pp_np = _init_LODF_pp_np(net_mod, outage_branch_type, outage_branch_ix.shape[0])

    outage_res_table, outage_res_type = (
        "res_" + outage_branch_type,
        "p_" + THIS_RES_BR_SIDE_MAPPING[outage_branch_type] + "_mw",
    )

    # number of out of service buses
    num_out_of_service_bus = np.sum(np.isnan(net_mod.res_bus.va_degree.to_numpy()))
    # list with the indices of out of service buses
    # list_out_of_service_bus = list(net_mod.res_bus.va_degree[np.isnan(net_mod.res_bus.va_degree.to_numpy())].index)

    outage_p0_series = net_mod[outage_res_table][outage_res_type].copy()

    for ix, br_ix in enumerate(outage_branch_ix):
        if net[outage_branch_type].at[br_ix, "in_service"] == False:
            # Skip out of service branch
            continue
        else:
            if np.isclose(outage_p0_series.at[br_ix], 0, atol=1e-6) & (outage_branch_type != "dcline"):
                bus_0 = net_mod[outage_branch_type].at[br_ix, BR_SIDE_MAPPING[outage_branch_type] + "_bus"]
                bus_1 = net_mod[outage_branch_type].at[br_ix, BR_SIDE_MAPPING_1[outage_branch_type] + "_bus"]

                # If the branch flow close to zero
                # Fix low loading branch with ptdf
                this_ptdf = _get_PTDF_perturb(
                    net_mod, source_bus=[bus_0, bus_1], distributed_slack=distributed_slack
                )  # distributed slack is default True
                if np.abs(this_ptdf[outage_branch_type + BR_PTDF_MAPPING[outage_branch_type]].at[br_ix, bus_0]) > 0.1:
                    bus_to_add_load = bus_0
                elif (
                    np.abs(this_ptdf[outage_branch_type + BR_PTDF_MAPPING_1[outage_branch_type]].at[br_ix, bus_1]) > 0.1
                ):
                    bus_to_add_load = bus_1
                else:
                    bus_to_add_load = None

                if bus_to_add_load is not None:
                    create_load(net_mod, bus=bus_to_add_load, p_mw=-1)
                    create_load(net_mod_n1, bus=bus_to_add_load, p_mw=-1)
                    logger.info("Added load on %s to fix low loading branch" % str(br_ix))
                else:
                    logger.warning("Add load not possible on %s to fix low loading branch" % str(br_ix))
                    continue

                # Update branch p0 in n-0 net
                rundcpp(net_mod, distributed_slack=distributed_slack, recycle=recycle)
                outage_p0_series = net_mod[outage_res_table][outage_res_type].copy()

            elif np.isclose(outage_p0_series.at[br_ix], 0, atol=1e-6) & (outage_branch_type == "dcline"):
                bus_0 = net_mod[outage_branch_type].at[br_ix, BR_SIDE_MAPPING[outage_branch_type] + "_bus"]
                bus_1 = net_mod[outage_branch_type].at[br_ix, BR_SIDE_MAPPING_1[outage_branch_type] + "_bus"]

                ## ext_grid auf offshore seite
                # res_dcline -> spannung am offshore knoten -> wenn ja dann slack schon da, wenn nan dann keine rechnung/konvergenz offshore ----> slack
                if net_mod[outage_res_table].loc[br_ix].vm_from_pu == 0:
                    bus_to_add_ext_grid = bus_0
                elif net_mod[outage_res_table].loc[br_ix].vm_to_pu == 0:
                    bus_to_add_ext_grid = bus_1
                else:
                    bus_to_add_ext_grid = None

                # add ext_grid to bus
                create_ext_grid(net_mod, bus_to_add_ext_grid)

                # Update branch p0 in n-0 net
                rundcpp(net_mod, distributed_slack=distributed_slack, recycle=recycle)
                outage_p0_series = net_mod[outage_res_table][outage_res_type].copy()

            net_mod_n1[outage_branch_type].at[br_ix, "in_service"] = False
            rundcpp(net_mod_n1, outage_branch_type=outage_branch_type, outage_branch_ix=br_ix, distributed_slack=distributed_slack, recycle=recycle)
            net_mod_n1[outage_branch_type].at[br_ix, "in_service"] = True

            if np.sum(np.isnan(net_mod_n1.res_bus.va_degree.to_numpy())) > num_out_of_service_bus:
                # outage of the considered line is resulting in islanding of the network
                logger.warning(f"Outage of line {br_ix} is causing isolated nodes!")

            # calculate the LODF factor
            for br_type in ("line", "dcline", "trafo", "impedance"):
                br_res_table, br_res_type, br_res_nan = (
                    "res_" + br_type,
                    "p_" + THIS_RES_BR_SIDE_MAPPING[br_type] + "_mw",
                    BR_NAN_CHECK[br_type],
                )
                if not net[br_type].empty:
                    lodf_pp_np[(br_type, outage_branch_type)][:, ix] = (
                        net_mod_n1[br_res_table][br_res_type].to_numpy() - net_mod[br_res_table][br_res_type].to_numpy()
                    ) / outage_p0_series.at[br_ix]
                    # replace LODF factors with NaN values,
                    # if the considered net element has NaN values in predefined columns
                    lodf_pp_np[(br_type, outage_branch_type)][net_mod_n1[br_res_table][br_res_nan].isna(), ix] = np.nan

            if not net.trafo3w.empty:
                for side in ("hv", "mv", "lv"):
                    br_res_type = "p_" + side + "_mw"
                    lodf_pp_np[("trafo3w_" + side, outage_branch_type)][:, ix] = (
                        net_mod_n1["res_trafo3w"][br_res_type].to_numpy()
                        - net_mod["res_trafo3w"][br_res_type].to_numpy()
                    ) / outage_p0_series.at[br_ix]
                    # replace LODF factors with NaN values,
                    # if the considered net element has NaN values in predefined columns
                    lodf_pp_np[("trafo3w_" + side, outage_branch_type)][
                        net_mod_n1["res_trafo3w"][br_res_type].isna(), ix
                    ] = np.nan

    # lodf pp contains only a subset
    lodf = _LODF_pp_np_to_df(net, lodf_pp_np, outage_branch_type=outage_branch_type, outage_branch_ix=outage_branch_ix)
    return lodf


# Example application function with LODF
def _get_dc_n1_with_LODF(net, outage_branch_type, outage_branch_ix=None, result_side=0, lodf=None):
    """
    this function calculate p_mw of a side of branch under the outage
    of another branch with LODF method
    """

    THIS_RES_BR_SIDE_MAPPING = BR_SIDE_MAPPING if result_side == 0 else BR_SIDE_MAPPING_1

    if lodf is None:
        lodf = _get_LODF_direct(net, outage_branch_type=outage_branch_type, outage_branch_ix=outage_branch_ix)

    outage_branch_ix = _get_outage_branch_ix(net, outage_branch_type, outage_branch_ix)

    res_n1_pp_np = _init_LODF_pp_np(net, outage_branch_type, outage_branch_ix.shape[0])

    rundcpp(net, distributed_slack=True)
    outage_br_p0_series = net["res_" + outage_branch_type][
        "p_" + THIS_RES_BR_SIDE_MAPPING[outage_branch_type] + "_mw"
    ].copy()
    for ix, br_ix in enumerate(outage_branch_ix):
        for br_type in ("line", "trafo", "impedance"):
            if not net[br_type].empty:
                this_lodf = lodf[(br_type, outage_branch_type)].loc[:, br_ix].to_numpy()
                # Skip invalid branch and branch with zero flow
                if np.all(np.isnan(this_lodf)) or np.isclose(outage_br_p0_series.at[br_ix], 0, atol=1e-6):
                    logger.info(f"""{outage_branch_type}: {ix} skipped!
                                     p_mw: {np.abs(outage_br_p0_series.at[br_ix]):.2f}""")
                    continue
                res_n1_pp_np[(br_type, outage_branch_type)][:, ix] = (
                    net["res_" + br_type]["p_" + THIS_RES_BR_SIDE_MAPPING[br_type] + "_mw"]
                    + this_lodf * outage_br_p0_series.at[br_ix]
                )

        if not net.trafo3w.empty:
            for side in ("hv", "mv", "lv"):
                this_lodf = lodf[("trafo3w_" + side, outage_branch_type)].loc[:, br_ix].to_numpy()
                # Skip invalid outage branch
                if np.all(np.isnan(this_lodf)):
                    continue
                # Sign correction considered already in LODF
                res_n1_pp_np[("trafo3w_" + side, outage_branch_type)][:, ix] = (
                    net["res_trafo3w"]["p_" + side + "_mw"] + this_lodf * outage_br_p0_series.at[br_ix]
                )

    res_n1 = _LODF_pp_np_to_df(
        net, res_n1_pp_np, outage_branch_type=outage_branch_type, outage_branch_ix=outage_branch_ix
    )
    return res_n1

def run_LODF(
    net: pandapowerNet,
    outage_branch_type: str,
    outage_branch_ix: ELE_IX_TYPE = None,
    distributed_slack: bool = True,
    perturb: bool = False,
    recycle: Union[str, None] = None,
    using_sparse_solver: bool = True,
    random_verify: bool = False,
    branch_dict: Dict[str, Union[List[int], None]] = None,
    reduced: bool = True,
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """
    this function is a wrapper of calculating LODF (ratio without unit) of a pp branch from the outage of a pp branch
    with pypower matrix function or perturb function.

    The LODF is defined as: (p_{side}_mw_new - p_{side}_mw_old) / (p_{side}_mw_outage)
    Side corresponds to the pandapower results definition, for LODF calculation both
    sides give the same result, thus no side definition required

    :param net: A pandapower network
    :param outage_branch_type: The name of the type of the outage branch ("line", "trafo", "impedance")
    :param outage_branch_ix: The pandapower index of the outage branch (int/list/np.ndarray), if None then all branches
        will be used (except bridge branch and very low loading branch)
    :param distributed_slack: Set True if p distribution amount distributed wished, or else slacks are
         only all voltage references! For non-perturb only False possible!!
    :param perturb: Set True to use the perturb version (brute-force) which is faster for calculating
        only a few elements on large networks, if a lot of elements required please set to False
    :param using_sparse_solver: Select whether sparse linear system should be used (more efficient for large network)
    :param random_verify: Set to True to check the direct version against perturb version
        with 3 randomly selected elements
    :param branch_dict: dictionary with keys "line", "trafo", "impedance", "trafo3w"; if not None the computation is
        restricted to the branch indices given in the dict
    :param reduced: if True, the output is reduced to the branches given in branch_dict
    :return: {(goal_branch_type ("line", "trafo", "impedance", "trafo3w_{hv,mv,lv}"),
               outage_branch_type (("line", "trafo", "impedance")):
        DataFrame(data=lodf, index=goal_branch_pp_index, columns=outage_branch_ix)}
    """

    if perturb and outage_branch_ix is None:
        logger.info("If a lot of branch required in lodf, please set perturb to False!")

    if perturb:
        if recycle == "lodf" and distributed_slack == True:
            logger.warning("distributed_slack deactivated! recycling does not allow distributed slack")
        lodf = _get_LODF_perturb(
            net,
            outage_branch_type=outage_branch_type,
            outage_branch_ix=outage_branch_ix,
            distributed_slack=distributed_slack,
            recycle=recycle,
        )
    else:
        if distributed_slack:
            logger.warning("distributed_slack deactivated! Distirbuted slacks are used as Vref! Only Perturb Possible")
        lodf = _get_LODF_direct(
            net,
            outage_branch_type=outage_branch_type,
            outage_branch_ix=outage_branch_ix,
            using_sparse_solver=using_sparse_solver,
            random_verify=random_verify,
            branch_dict=branch_dict,
            reduced=reduced,
        )

    if outage_branch_ix is not None and (np.isscalar(outage_branch_ix) or len(outage_branch_ix) == 1):
        if np.all(np.isnan(lodf[("line", outage_branch_type)].to_numpy())):
            # if only one branch selected for outage,
            # raise an error when not allowed to outage
            # logger.error(f"{outage_branch_ix} is not allowed to outage!")
            raise UserWarning(f"{outage_branch_ix} is not allowed to outage!")

    # Update lodf on net
    net._lodf = {"branch": {"table": outage_branch_type, "element": None}}
    for (br_type, _), data in lodf.items():
        if net._lodf["branch"]["element"] is None:
            net._lodf["branch"]["element"] = data.columns.to_numpy(copy=True)
        net["lodf_" + br_type] = data
    return lodf


def verify_dc_n1_with_LODF(
    net, outage_branch_type: str, outage_branch_ix: ELE_IX_TYPE = None, result_side=0, lodf=None
):
    """
    this function verifies the result of dc_n1 with LODF and perturb method,
    raise AssertionError on mismatches!
    """
    net = deepcopy(net)
    res_n1_lodf = run_dc_n1(net, outage_branch_type, outage_branch_ix, result_side, perturb=False, lodf=lodf)
    res_n1_perturb = run_dc_n1(
        net, outage_branch_type, outage_branch_ix, result_side, distributed_slack=True, perturb=True
    )

    assert len(res_n1_lodf) > 0, "Empty res n1 lodf, verification not possible!"
    for key in res_n1_lodf.keys():
        filter = ~res_n1_lodf[key].isna().any(axis=0)
        assert np.allclose(
            res_n1_lodf[key].loc[:, filter], res_n1_perturb[key].loc[:, filter], equal_nan=True, atol=1e-8
        ), f"{key} verification failed!"
        logger.info(str(key) + " dc n-1 results verified!")
    logger.info("Run dc n-1 with LODF verified!")


def verify_LODF(
    net, outage_branch_type: str, outage_branch_ix: ELE_IX_TYPE = None, using_sparse_solver=True, lodf=None
):
    """
    this function verifies the result of LODF and perturb method,
    raise AssertionError on mismatches!
    """
    net = deepcopy(net)
    if lodf is None:
        lodf = run_LODF(
            net,
            outage_branch_type,
            outage_branch_ix,
            perturb=False,
            using_sparse_solver=using_sparse_solver,
            random_verify=False,
        )
    lodf_perturb = run_LODF(net, outage_branch_type, outage_branch_ix, distributed_slack=True, perturb=True)

    assert len(lodf) > 0, "Empty lodf, verification not possible!"
    for key in lodf.keys():
        filter = ~lodf[key].isna().any(axis=0)
        assert np.allclose(lodf[key].loc[:, filter], lodf_perturb[key].loc[:, filter], atol=1e-8, equal_nan=True), (
            f"{key} LODF results verification failed!"
        )
        logger.info(str(key) + " LODF results verified!")
    logger.info("All LODF results verified with perturb method!")


