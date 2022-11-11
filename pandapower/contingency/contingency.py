# -*- coding: utf-8 -*-

# Copyright (c) 2016-2022 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pandas as pd

import pandapower as pp
import pandapower.networks
from pandapower.pd2ppc import _pd2ppc

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def run_contingency(net, nminus1_cases, pf_options=None, pf_options_nminus1=None, write_to_net=True, run=pp.runpp):
    """
    Obtain either loading (N-0) or max. loading (N-0 and all N-1 cases), and min/max bus voltage magnitude.
    The variable "temperature_degree_celsius" can be used insteas of "loading_percent" to obtain max. temperature.

    Parameters
    ----------
    net : pandapowerNet
    nminus1_cases : dict
        describes all N-1 cases, e.g. {"line": {"index": [1, 2, 3]}, "trafo": {"index": [0]}, "trafo3w": {"index": [1]}}
    pf_options : dict
        options for power flow calculation in N-0 case
    pf_options_nminus1 : dict
        options for power flow calculation in N-1 cases
    write_to_net: bool
        whether to write the results of contingency analysis to net (in "res_" tables)
    run : func
        function to use for power flow calculation, default pp.runpp

    Returns
    -------
    contingency_results : dict
        dict of arrays per element for index, min/max result
    """
    # set up the dict for results and relevant variables
    # ".get" in case the options have been set in pp.set_user_pf_options:
    if pf_options is None: pf_options = net.user_pf_options.get("pf_options", net.user_pf_options)
    if pf_options_nminus1 is None: pf_options_nminus1 = net.user_pf_options.get("pf_options_nminus1", net.user_pf_options)

    contingency_results = {element: {"index": net[element].index.values}
                           for element in ("bus", "line", "trafo", "trafo3w") if len(net[element]) > 0}
    result_variables = {**{"bus": ["vm_pu"]},
                        **{key: ["loading_percent"] for key in ("line", "trafo", "trafo3w") if len(net[key]) > 0}}
    if len(net.line) > 0 and (net.get("_options", {}).get("tdpf", False) or
                              pf_options.get("tdpf", False) or pf_options_nminus1.get("tdpf", False)):
        result_variables["line"].append("temperature_degree_celsius")

    # for n-1
    for element, val in nminus1_cases.items():
        for i in val["index"]:
            if ~net[element].at[i, "in_service"]:
                continue
            net[element].at[i, 'in_service'] = False
            try:
                run(net, **pf_options_nminus1)
                _update_contingency_results(net, contingency_results, result_variables, nminus1=True)
            except Exception as err:
                logger.error(f"{element} {i} causes {err}")
            finally:
                net[element].at[i, 'in_service'] = True

    # for n-0
    run(net, **pf_options)
    _update_contingency_results(net, contingency_results, result_variables, nminus1=False)

    return contingency_results


def _update_contingency_results(net, contingency_results, result_variables, nminus1):
    for element, vars in result_variables.items():
        for var in vars:
            val = net[f"res_{element}"][var].values
            if nminus1:
                for func, min_max in ((np.fmax, "max"), (np.fmin, "min")):
                    key = f"{min_max}_{var}"
                    func(val,
                         contingency_results[element].setdefault(key, np.full_like(val, np.nan, dtype=np.float64)),
                         out=contingency_results[element][key],
                         where=net[element]["in_service"].values & ~np.isnan(val))
                if element == "line": print(val)
                # np.fmax(val, contingency_results[element].setdefault(f"max_{var}", np.full_like(val, np.nan, dtype=np.float64)), out=contingency_results[element][f"max_{var}"])
                # np.fmin(val, contingency_results[element].setdefault(f"min_{var}", np.full_like(val, np.nan, dtype=np.float64)), out=contingency_results[element][f"min_{var}"])
                # max_value = contingency_results[element].setdefault(f"max_{var}",
                #                                                     np.full_like(val, np.nan, dtype=np.float64))
                # min_value = contingency_results[element].setdefault(f"min_{var}",
                #                                                     np.full_like(val, np.nan, dtype=np.float64))
                # # max_value = np.nanmax([val, max_value], axis=0)
                # # min_value = np.nanmin([val, min_value], axis=0)
                # np.fmax(val, max_value, out=max_value, axis=0)
                # np.fmin(val, min_value, out=min_value, axis=0)
            else:
                contingency_results[element][var] = val


def get_element_limits(net):
    """
    Construct the dictionary of element limits

    Parameters
    ----------
    net : pandapowerNet

    Returns
    -------
    element_limits : dict
    """
    element_limits = {}
    if "max_vm_pu" in net.bus and "min_vm_pu" in net.bus:
        bus_index = net.bus.loc[~pd.isnull(net.bus.max_vm_pu) & ~pd.isnull(net.bus.min_vm_pu)].index.values
        if len(bus_index) != 0:
            element_limits.update({"bus": {
                "index": bus_index,
                "max_limit": net.bus.loc[bus_index, "max_vm_pu"].values,
                "min_limit": net.bus.loc[bus_index, "min_vm_pu"].values,
                "max_limit_nminus1": net.line.loc[bus_index, "max_vm_nminus1_pu"].values \
                    if "max_vm_nminus1_pu" in net.bus.columns \
                    else net.bus.loc[bus_index, "max_vm_pu"].values,
                "min_limit_nminus1": net.line.loc[bus_index, "min_vm_nminus1_pu"].values \
                    if "min_vm_nminus1_pu" in net.bus.columns \
                    else net.bus.loc[bus_index, "min_vm_pu"].values}})

    for element in ("line", "trafo", "trafo3w"):
        if net[element].empty or ("max_loading_percent" not in net[element] and
                                  "max_temperature_degree_celsius" not in net[element]):
            continue

        element_index = net[element].loc[~pd.isnull(net[element].get("max_loading_percent")) |
                                         ~pd.isnull(net[element].get("max_temperature_degree_celsius"))].index.values
        if len(element_index) == 0:
            continue

        d = {element: {"index": element_index}}

        if "max_loading_percent" in net[element].columns:
            d[element].update({
                "max_limit": net[element].loc[element_index, "max_loading_percent"].values,
                "min_limit": -net[element].loc[element_index, "max_loading_percent"].values,
                "max_limit_nminus1":
                    net[element].loc[element_index, "max_loading_nminus1_percent"].values
                    if "max_loading_nminus1_percent" in net[element].columns
                    else net[element].loc[element_index, "max_loading_percent"].values,
                "min_limit_nminus1":
                    -net[element].loc[element_index, "max_loading_nminus1_percent"].values
                    if "max_loading_nminus1_percent" in net[element].columns
                    else - net[element].loc[element_index, "max_loading_percent"].values})

        if element == "line" and "max_temperature_degree_celsius" in net[element].columns:
            col = "max_temperature_degree_celsius"
            d[element].update({"max_temperature_degree_celsius": net.line.loc[element_index, col].values,
                               "min_temperature_degree_celsius": -net.line.loc[element_index, col].values})

        element_limits.update(d)
    return element_limits


def get_nminus1_cases(net, element_limits=None):
    """
    Generate dictionary for N-1 cases.

    Parameters
    ----------
    net : pandapowerNet
    element_limits : dict

    Returns
    -------
    nminus1_cases : dict
        describes all N-1 cases, e.g. {"line": {"index": [1, 2, 3]}, "trafo": {"index": [0]}, "trafo3w": {"index": [1]}}
    """
    if element_limits is None:
        element_limits = get_element_limits(net)

    in_service = elements_in_service(net, element_limits)

    nminus1_cases = {element: {"index": element_limits[element]["index"][in_service[element]]}
                     for element in element_limits.keys() if element != "bus"}
    return nminus1_cases


def elements_in_service(net, element_limits):
    """
    Obtain a dict of boolean masks of in_service status for each limit element (lines etc.)

    Parameters
    ----------
    net : pandapowerNet
    element_limits : dict

    Returns
    -------
    in_service : dict
    """
    ppc, _ = _pd2ppc(net)
    # cannot use set operations because we want to preserve the order
    in_service = {}
    for element in element_limits.keys():
        if element == "bus":
            # in_service_mask = net[element].loc[element_limits[element]["index"], "in_service"].values
            in_service_mask = net[element].index.isin(net._is_elements["bus_is_idx"]) & \
                              net[element].index.isin(element_limits[element]["index"])
        elif element == "trafo3w":
            f, t = net._pd2ppc_lookups["branch"][element]
            ft = np.arange(f, t, 3)
            in_service_mask = net._ppc["internal"]['branch_is'][ft] & \
                              net[element].index.isin(element_limits[element]["index"])
        else:
            f, t = net._pd2ppc_lookups["branch"][element]
            in_service_mask = net._ppc["internal"]['branch_is'][f:t] & \
                              net[element].index.isin(element_limits[element]["index"])
        # use branch_is from ppc internal
        in_service[element] = in_service_mask
    return in_service


def check_elements_within_limits(element_limits, max_loading, nminus1=False, branch_tol=1e-3, bus_tol=1e-6):
    """
    Check if elements are within limits

    Parameters
    ----------
    element_limits : dict
    max_loading : dict
    nminus1 : bool

    Returns
    -------
    bool
    """
    for element, values in max_loading.items():
        limit = element_limits[element]
        if 'max_temperature_degree_celsius' in limit and 'temperature_degree_celsius' in values:
            if np.any(values['temperature_degree_celsius'] > limit['max_temperature_degree_celsius'] + branch_tol):
                return False
            if nminus1 and np.any(values['max_temperature_degree_celsius'] >
                                  limit['max_temperature_degree_celsius'] + branch_tol):
                return False
        if "max_limit" not in limit:
            continue
        if element == "bus":
            if np.any(values["vm_pu"] > limit["max_limit"] + bus_tol) or \
                    np.any(values["vm_pu"] < limit["min_limit"] - bus_tol):
                return False
            if nminus1 and (np.any(values["max_vm_pu"] > limit["max_limit_nminus1"] + bus_tol) or
                            np.any(values["min_vm_pu"] < limit["min_limit_nminus1"] - bus_tol)):
                return False
            continue
        if np.any(values['loading_percent'] > limit["max_limit"] + branch_tol):
            return False
        if nminus1 and np.any(values['max_loading_percent'] > limit["max_limit_nminus1"] + branch_tol):
            return False
    return True


if __name__ == "__main__":
    net = pp.networks.case5()

    element_limits = get_element_limits(net)
    nminus1_cases = {"line": {"index": net.line.index.values}}
    res = run_contingency(net, nminus1_cases)

    check_elements_within_limits(element_limits, res, True)

    # todo: convenience "wieviel wurde überschritten?"
    #  - report function
