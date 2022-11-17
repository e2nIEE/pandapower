# -*- coding: utf-8 -*-

# Copyright (c) 2016-2022 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pandas as pd

import pandapower as pp

try:
    from lightsim2grid.gridmodel import init as init_ls2g
    from lightsim2grid_cpp import SecurityAnalysisCPP, SolverType
    from lightsim2grid.newtonpf.newtonpf import _get_valid_solver

    lightsim2grid_installed = True
except ImportError:
    lightsim2grid_installed = False

try:
    from lightsim2grid_cpp import KLUSolver, KLUSolverSingleSlack
    KLU_solver_available = True
except ImportError:
    KLU_solver_available = False

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def run_contingency(net, nminus1_cases, pf_options=None, pf_options_nminus1=None, write_to_net=True,
                    contingency_evaluation_function=pp.runpp, **kwargs):
    """
    Obtain either loading (N-0) or max. loading (N-0 and all N-1 cases), and min/max bus voltage magnitude.
    The variable "temperature_degree_celsius" can be used instead of "loading_percent" to obtain max. temperature.

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
    contingency_evaluation_function : func
        function to use for power flow calculation, default pp.runpp

    Returns
    -------
    contingency_results : dict
        dict of arrays per element for index, min/max result
    """
    # set up the dict for results and relevant variables
    # ".get" in case the options have been set in pp.set_user_pf_options:
    if "recycle" in kwargs: kwargs["recycle"] = False  # so that we can be sure it doesn't happen
    if pf_options is None: pf_options = net.user_pf_options.get("pf_options", net.user_pf_options)
    if pf_options_nminus1 is None: pf_options_nminus1 = net.user_pf_options.get("pf_options_nminus1",
                                                                                net.user_pf_options)

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
                contingency_evaluation_function(net, **pf_options_nminus1, **kwargs)
                _update_contingency_results(net, contingency_results, result_variables, nminus1=True)
            except Exception as err:
                logger.error(f"{element} {i} causes {err}")
            finally:
                net[element].at[i, 'in_service'] = True

    # for n-0
    contingency_evaluation_function(net, **pf_options, **kwargs)
    _update_contingency_results(net, contingency_results, result_variables, nminus1=False)

    if write_to_net:
        for element, element_results in contingency_results.items():
            index = element_results["index"]
            for var, val in element_results.items():
                if var == "index" or var in net[f"res_{element}"].columns.values:
                    continue
                net[f"res_{element}"].loc[index, var] = val

    return contingency_results


def _get_solver_type(options):
    if options.get("distributed_slack", False):
        solver_type = 1 if KLU_solver_available else 0
    else:
        solver_type = 7 if KLU_solver_available else 6

    return SolverType(solver_type)


def run_contingency_ls2g(net, **kwargs):
    """
    Execute contingency analysis using the lightsim2grid library. This works much faster than using pandapower.
    Limitation: the results for branch flows are valid only for the "from_bus" of lines and "hv_bus" of transformers.
    This can lead to a small difference to the results using pandapower.
    The results are written in pandapower results tables.
    Make sure that the N-1 cases do not lead to isolated grid, otherwise results with pandapower and this function will
    be different. Reason: pandapower selects a different gen as slack if the grid becomes isolated, but
    lightsim2grid would simply return nan as results for such a contingency situation.
    WARNING: continuous bus indices, 0-start, are required!

    Parameters
    ----------
    net : pandapowerNet

    Returns
    -------
    None
    """
    if not lightsim2grid_installed:
        raise UserWarning("lightsim2grid package not installed. "
                          "Install lightsim2grid e.g. by running 'pip install lightsim2grid' in command prompt.")
    pp.runpp(net, **kwargs)
    lightsim_grid_model = init_ls2g(net)
    s = SecurityAnalysisCPP(lightsim_grid_model)
    solver_type = _get_solver_type(net._options)
    s.change_solver(solver_type)
    # s.add_all_n1()
    # todo: 1) take nminus1_cases argument 2) map line, trafo indices to lookup indices 3) define cases 4) map results
    s.add_multiple_n1(np.arange(len(net.line), dtype=np.int64))
    # s.add_multiple_n1(net.line.index.values.astype(int))
    v_init = net._ppc["internal"]["V"].astype(np.complex128)
    s.compute(v_init, net._options["max_iteration"], net._options["tolerance_mva"])
    v_res = s.get_voltages()
    s.compute_flows()
    kamps_all = s.get_flows()
    kamps = kamps_all[0:len(net.line), 0:len(net.line)]

    vm_pu = np.abs(v_res)
    net.res_bus["max_vm_pu"] = np.nanmax(vm_pu, axis=0)
    net.res_bus["min_vm_pu"] = np.nanmin(vm_pu, axis=0)

    # max_i_ka = np.nanmax(kamps, where=~np.isnan(kamps), axis=0, initial=0)
    max_i_ka = np.nanmax(kamps, axis=0)
    net.res_line["max_loading_percent"] = max_i_ka / net.line.max_i_ka.values * 100
    # min_i_ka = np.nanmin(kamps, axis=0, where=kamps != 0, initial=100)  # this is quite daring tbh
    min_i_ka = np.nanmin(kamps, axis=0)
    net.res_line["min_loading_percent"] = min_i_ka / net.line.max_i_ka.values * 100


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
        fill_val = np.full_like(net.bus.index, np.nan, np.float64)
        bus_index = net.bus.loc[~pd.isnull(net.bus.get("max_vm_pu", fill_val)) &
                                ~pd.isnull(net.bus.get("min_vm_pu", fill_val))].index.values
        if len(bus_index) != 0:
            element_limits.update({"bus": {
                "index": bus_index,
                "max_limit": net.bus.loc[bus_index, "max_vm_pu"].values,
                "min_limit": net.bus.loc[bus_index, "min_vm_pu"].values,
                "max_limit_nminus1":
                    net.line.loc[bus_index, "max_vm_nminus1_pu"].values
                    if "max_vm_nminus1_pu" in net.bus.columns
                    else net.bus.loc[bus_index, "max_vm_pu"].values,
                "min_limit_nminus1":
                    net.line.loc[bus_index, "min_vm_nminus1_pu"].values
                    if "min_vm_nminus1_pu" in net.bus.columns
                    else net.bus.loc[bus_index, "min_vm_pu"].values}})

    for element in ("line", "trafo", "trafo3w"):
        if net[element].empty or ("max_loading_percent" not in net[element] and
                                  "max_temperature_degree_celsius" not in net[element]):
            continue

        fill_val = np.full_like(net[element].index, np.nan, np.float64)
        element_index = net[element].loc[
            ~pd.isnull(net[element].get("max_loading_percent", fill_val)) |
            ~pd.isnull(net[element].get("max_temperature_degree_celsius", fill_val))].index.values
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


def check_elements_within_limits(element_limits, contingency_results, nminus1=False, branch_tol=1e-3, bus_tol=1e-6):
    """
    Check if elements are within limits

    Parameters
    ----------
    element_limits : dict
    contingency_results : dict
    nminus1 : bool
    branch_tol : float
        tolerance of the limit violation check for branch limits
    bus_tol : float
        tolerance of the limit violation check for bus limits

    Returns
    -------
    bool
        True if all within limits (no violations), False if any limits violated
    """
    for element, values in contingency_results.items():
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


def _get_iloc_index(index, sub_index):
    lookup_index = np.arange(index.max() + 1)
    lookup_index[index] = np.arange(len(index))
    return lookup_index[sub_index]


def _log_violation(element, var, val, limit_index, mask):
    if np.any(mask):
        s = ' (N-1)' if 'max' in var else ''
        with np.printoptions(precision=3, suppress=True):
            logger.info(f"{element}: {var}{s} violation at index {limit_index[mask]} ({val[mask]})")


def report_contingency_results(element_limits, contingency_results, branch_tol=1e-3, bus_tol=1e-6):
    """
    Print log messages for elements with violations of limits

    Parameters
    ----------
    element_limits : dict
    contingency_results : dict
    branch_tol : float
        tolerance for branch results
    bus_tol : float
        tolerance for bus results

    Returns
    -------
    None
    """
    for element, results in contingency_results.items():
        limit = element_limits[element]
        index = _get_iloc_index(results["index"], limit["index"])
        for var, val in results.items():
            tol = bus_tol if element == "bus" else branch_tol
            if var == "index":
                continue
            if "min" in var:
                mask = val[index] < limit['min_limit_nminus1'] - tol
                _log_violation(element, var, val[index], limit["index"], mask)
            elif "max" in var:
                mask = val[index] > limit['max_limit_nminus1'] + tol
                _log_violation(element, var, val[index], limit["index"], mask)
            else:
                mask_max = val[index] > limit['max_limit'] + tol
                _log_violation(element, var, val[index], limit["index"], mask_max)
                mask_min = val[index] < limit['min_limit'] - tol
                _log_violation(element, var, val[index], limit["index"], mask_min)
