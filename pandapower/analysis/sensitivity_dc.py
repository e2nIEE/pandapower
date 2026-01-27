# -*- coding: utf-8 -*-
from copy import deepcopy

import pandas as pd
import numpy as np

from pandapower.analysis.LODF import _get_dc_n1_with_LODF, _LODF_pp_np_to_df, _init_LODF_pp_np
from pandapower.analysis.PTDF import _get_dc_profile_with_PTDF
from pandapower.analysis.utils import _get_outage_branch_ix, BR_SIDE_MAPPING, BR_SIDE_MAPPING_1, ELE_IX_TYPE
from pandapower.run import rundcpp

#from lib_powerflow.dc_distributed_slack import (
#    get_dist_slack,
#    makePTDF_multi_area,
#    get_ppci_dist_slack,
#)

# basic logging setups
import logging
logger = logging.getLogger(__name__)

# Global variable


""" ppci to pp conversion """


def _profile_pp_np_to_df(net, res_pp_np, num_calc, res_extra_dp=None):
    res = {}
    for br_type, data in res_pp_np.items():
        pp_br_type = "trafo3w" if br_type.startswith("trafo3w") else br_type
        res[br_type] = pd.DataFrame(data=data, index=np.arange(num_calc), columns=net[pp_br_type].index.to_numpy())
    if res_extra_dp is not None:
        for (ele_type, data_type), data in res_extra_dp.items():
            res[(ele_type, data_type)] = pd.DataFrame(
                data=data, index=np.arange(num_calc), columns=net[ele_type].index.to_numpy()
            )
    return res


def _get_dc_profile_perturb(net, profiles, result_side=0, distributed_slack=True, extra_data_points=None):
    """
    Run dc profile with perturb method
    :return: {branch_type ("line", "trafo", "impedance", "trafo3w_{hv,mv,lv}"):
              DataFrame(data=p_side_mw, index=calc_ix, columns=branch_index)}
    if extra_data_points defined, further pp data points also returned
    """
    THIS_RES_BR_SIDE_MAPPING = BR_SIDE_MAPPING if result_side == 0 else BR_SIDE_MAPPING_1

    net_mod = deepcopy(net)
    num_calc = None
    # Check profile integrity
    for key in profiles.keys():
        assert isinstance(profiles[key], pd.DataFrame), "Only profile as pandas dataframe supported!"

        # Check only dimension
        if num_calc is None:
            num_calc = profiles[key].shape[0]
        else:
            assert num_calc == profiles[key].shape[0], f"{key} profile has wrong dimension"

    # Init pp result table as np array
    res_pp_np = {}
    for br_type in ("line", "trafo", "impedance"):
        if not net[br_type].empty:
            res_pp_np[br_type] = np.zeros((num_calc, net[br_type].shape[0]), dtype=np.float)
    if not net.trafo3w.empty:
        for side in ("hv", "mv", "lv"):
            res_pp_np["trafo3w_" + side] = np.zeros((num_calc, net["trafo3w"].shape[0]), dtype=np.float)

    res_pp_extra_dp = {}
    if extra_data_points is not None:
        for ele_type, data_point in extra_data_points:
            res_pp_extra_dp[(ele_type, data_point)] = np.zeros((num_calc, net[ele_type].shape[0]), dtype=np.float)

    # run all timesteps of the profiles
    for calc_ix in range(num_calc):
        # Update network with profile
        for ele_type, value_type in profiles.keys():
            this_ele_profile = profiles[(ele_type, value_type)]
            ele_ix = this_ele_profile.columns.to_numpy()
            net_mod[ele_type].loc[ele_ix, value_type] = this_ele_profile.to_numpy()[calc_ix, :]

        # Update result table
        rundcpp(net_mod, distributed_slack=True)
        for res_br_type in res_pp_np.keys():
            if not res_br_type.startswith("trafo3w"):
                res_pp_np[res_br_type][calc_ix, :] = net_mod["res_" + res_br_type][
                    "p_" + THIS_RES_BR_SIDE_MAPPING[res_br_type] + "_mw"
                ].to_numpy()
            else:
                trafo3w_side = res_br_type.split("_")[-1]
                res_pp_np[res_br_type][calc_ix, :] = net_mod["res_trafo3w"]["p_" + trafo3w_side + "_mw"].to_numpy()

        if extra_data_points is not None:
            for ele_type, data_point in extra_data_points:
                res_pp_extra_dp[(ele_type, data_point)][calc_ix, :] = net_mod["res_" + ele_type][data_point].to_numpy()

    # Convert numpy array to pandas dataframe with pp indexing
    res = _profile_pp_np_to_df(net, res_pp_np, num_calc, res_pp_extra_dp)
    return res


def _get_dc_n1_perturb(net, outage_branch_type, outage_branch_ix=None, result_side=0, distributed_slack=True):
    """
    this function calculate p_mw of a side of branch under the outage
    of another branch with perturb (brute-force) method
    """
    THIS_RES_BR_SIDE_MAPPING = BR_SIDE_MAPPING if result_side == 0 else BR_SIDE_MAPPING_1
    # this_rundcpp = get_dcpp_runner(net, distributed_slack=distributed_slack)

    # Only net_mod is required in the function
    net_mod = deepcopy(net)
    outage_branch_ix = _get_outage_branch_ix(net_mod, outage_branch_type, outage_branch_ix)

    rundcpp(net_mod, distributed_slack=distributed_slack)
    num_out_of_service_bus = np.sum(np.isnan(net_mod.res_bus.va_degree.to_numpy()))
    outage_br_p0_series = net_mod["res_" + outage_branch_type][
        "p_" + THIS_RES_BR_SIDE_MAPPING[outage_branch_type] + "_mw"
    ].copy()

    res_n1_pp_np = _init_LODF_pp_np(net, outage_branch_type, outage_branch_ix.shape[0])
    for ix, br_ix in enumerate(outage_branch_ix):
        # Skip out-of-service line
        if net_mod[outage_branch_type].at[br_ix, "in_service"]:
            net_mod[outage_branch_type].at[br_ix, "in_service"] = False
            rundcpp(net_mod, distributed_slack=distributed_slack)
            net_mod[outage_branch_type].at[br_ix, "in_service"] = True
            if (
                np.isclose(outage_br_p0_series.at[br_ix], 0, atol=1e-6)
                or np.sum(np.isnan(net_mod.res_bus.va_degree.to_numpy())) > num_out_of_service_bus
            ):
                logger.info(f"""{outage_branch_type}: {ix} skipped!
                                   p_mw: {np.abs(outage_br_p0_series.at[br_ix]):.2f},
                                   num oos bus: {np.sum(np.isnan(net_mod.res_bus.va_degree.to_numpy()))}""")
                continue

            for br_type in ("line", "trafo", "impedance"):
                if not net[br_type].empty:
                    value_type = "p_" + THIS_RES_BR_SIDE_MAPPING[br_type] + "_mw"
                    res_n1_pp_np[(br_type, outage_branch_type)][:, ix] = net_mod["res_" + br_type][
                        value_type
                    ].to_numpy()
            if not net.trafo3w.empty:
                for side in ("hv", "mv", "lv"):
                    res_n1_pp_np[("trafo3w_" + side, outage_branch_type)][:, ix] = net_mod["res_trafo3w"][
                        "p_" + side + "_mw"
                    ].to_numpy()

    # Convert np array to pd dataframe with pp indexing
    res_n1 = _LODF_pp_np_to_df(
        net, res_n1_pp_np, outage_branch_type=outage_branch_type, outage_branch_ix=outage_branch_ix
    )
    return res_n1


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


def run_dc_n1(
    net,
    outage_branch_type: str,
    outage_branch_ix: ELE_IX_TYPE = None,
    result_side=0,
    distributed_slack: bool = True,
    perturb: bool = False,
    lodf: dict = None,
):
    """
    this function calculate p_mw of a side of branch under the outage of another branch with LODF
    :param net: A pandapower network
    :param outage_branch_type: The name of the type of the outage branch ("line", "trafo", "impedance")
    :param outage_branch_ix: The pandapower index of the outage branch (int/list/np.ndarray), if None then all branches
        will be used (except bridge branch and extra low loading branch)
    :param result_side: 0 means ("from", "hv") side, 1 means ("to", "lv") side
    :param distributed_slack: Set True if p distribution amount distributed wished, or else slacks are
         only all voltage references! For non-perturb only True possible!!
    :param perturb: Set True to use the perturb version (brute-force) which is faster for calculating
        only a few elements on large networks, if a lot of elements required please set to False
    :param lodf: precomputed load matrices in dictionary, if None it will be calculated internally
    :return: {(res_{branch_type}, p_{side}_mw):
        DataFrame(data=p_side_mw, index=goal_branch_pp_index, columns=outage_branch_pp_index)}
    """
    # ToDo: Check distributed slack option here
    if perturb or not distributed_slack:
        if not distributed_slack:
            logger.info("distributed_slack deactivated! Distirbuted slacks are used as Vref! Only Perturb Possible")
        res = _get_dc_n1_perturb(
            net,
            outage_branch_type=outage_branch_type,
            outage_branch_ix=outage_branch_ix,
            result_side=result_side,
            distributed_slack=distributed_slack,
        )
    else:
        res = _get_dc_n1_with_LODF(
            net,
            outage_branch_type=outage_branch_type,
            outage_branch_ix=outage_branch_ix,
            result_side=result_side,
            lodf=lodf,
        )

    res_renamed = {}
    THIS_RES_BR_SIDE_MAPPING = BR_SIDE_MAPPING if result_side == 0 else BR_SIDE_MAPPING_1
    for (br_type, _), value in res.items():
        if not br_type.startswith("trafo3w"):
            side = THIS_RES_BR_SIDE_MAPPING[br_type]
        else:
            side = br_type.split("_")[-1]
        res_renamed[(f"res_{br_type}", f"p_{side}_mw")] = value
    return res_renamed

