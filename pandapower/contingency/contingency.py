# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pandas as pd
import warnings

import pandapower as pp

try:
    from lightsim2grid.gridmodel import init as init_ls2g
    from lightsim2grid.securityAnalysis import ContingencyAnalysisCPP
    from lightsim2grid_cpp import SolverType

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
    The variable "temperature_degree_celsius" can be used in addition to "loading_percent" to obtain max. temperature.
    In the returned dictionary, the variable "loading_percent" represents the loading in N-0 case,
    "max_loading_percent" and "min_loading_percent" represent highest and lowest observed loading_percent among all
    calculated N-1 cases. The same convention applies to "temperature_degree_celsius" when applicable.
    This function can be passed through to pandapower.timeseries.run_timeseries as the run_control_fct argument.

    INPUT
    ----------
    **net** - pandapowerNet
    **nminus1_cases** - dict
        describes all N-1 cases, e.g. {"line": {"index": [1, 2, 3]}, "trafo": {"index": [0]}, "trafo3w": {"index": [1]}}
    **pf_options** - dict
        options for power flow calculation in N-0 case
    **pf_options_nminus1** - dict
        options for power flow calculation in N-1 cases
    **write_to_net** - bool
        whether to write the results of contingency analysis to net (in "res_" tables). The results will be written for
        the following additional variables: table res_bus with columns "max_vm_pu", "min_vm_pu",
        tables res_line, res_trafo, res_trafo3w with columns "max_loading_percent", "min_loading_percent",
        "causes_overloading", "cause_element", "cause_index", table res_line with columns
        "max_temperature_degree_celsius", "min_temperature_degree_celsius" (if "tdpf" set to True)
        "causes_overloading": does this element, when defining the N-1 case, cause overloading of other elements? the
        overloading is defined by net.line["max_loading_percent_nminus1"] (if set) or net.line["max_loading_percent"]
        "cause_element": element ("line", "trafo", "trafo3w") that causes max. loading of this element
        "cause_index": index of the element ("line", "trafo", "trafo3w") that causes max. loading of this element
    **contingency_evaluation_function** - func
        function to use for power flow calculation, default pp.runpp

    OUTPUT
    -------
    **contingency_results** - dict
        dict of arrays per element for index, min/max result
    """
    # set up the dict for results and relevant variables
    # ".get" in case the options have been set in pp.set_user_pf_options:
    raise_errors = kwargs.get("raise_errors", False)
    if "recycle" in kwargs: kwargs["recycle"] = False  # so that we can be sure it doesn't happen
    if pf_options is None: pf_options = net.user_pf_options.get("pf_options", net.user_pf_options)
    if pf_options_nminus1 is None: pf_options_nminus1 = net.user_pf_options.get("pf_options_nminus1",
                                                                                net.user_pf_options)

    contingency_results = {element: {"index": net[element].index.values}
                           for element in ("bus", "line", "trafo", "trafo3w") if len(net[element]) > 0}
    for element in contingency_results.keys():
        if element == "bus":
            continue
        contingency_results[element].update(
            {"causes_overloading": np.zeros_like(net[element].index.values, dtype=bool),
             "cause_element": np.empty_like(net[element].index.values, dtype=object),
             "cause_index": np.empty_like(net[element].index.values, dtype=np.int64)})
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
                _update_contingency_results(net, contingency_results, result_variables, nminus1=True,
                                            cause_element=element, cause_index=i)
            except Exception as err:
                logger.error(f"{element} {i} causes {err}")
                if raise_errors:
                    raise err
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


def run_contingency_ls2g(net, nminus1_cases, contingency_evaluation_function=pp.runpp, **kwargs):
    """
    Execute contingency analysis using the lightsim2grid library. This works much faster than using pandapower.
    Limitation: the results for branch flows are valid only for the "from_bus" of lines and "hv_bus" of transformers.
    This can lead to a small difference to the results using pandapower.
    The results are written in pandapower results tables.
    Make sure that the N-1 cases do not lead to isolated grid, otherwise results with pandapower and this function will
    be different. Reason: pandapower selects a different gen as slack if the grid becomes isolated, but
    lightsim2grid would simply return nan as results for such a contingency situation.
    WARNING: continuous bus indices, 0-start, are required!
    This function can be passed through to pandapower.timeseries.run_timeseries as the run_control_fct argument.

    The results will written for the
    following additional variables: table res_bus with columns "max_vm_pu", "min_vm_pu",
    tables res_line, res_trafo, res_trafo3w with columns "max_loading_percent", "min_loading_percent",
    "causes_overloading", "cause_element", "cause_index", table res_line with columns "max_temperature_degree_celsius",
    "min_temperature_degree_celsius" (if "tdpf" set to True)
    "causes_overloading": does this element, when defining the N-1 case, cause overloading of other elements? the
    overloading is defined by net.line["max_loading_percent_nminus1"] (if set) or net.line["max_loading_percent"]
    "cause_element": element ("line", "trafo", "trafo3w") that causes max. loading of this element
    "cause_index": index of the element ("line", "trafo", "trafo3w") that causes max. loading of this element
    "congestion_caused_mva": overall congestion in the grid in MVA during the N-1 case due to the failure of the element

    INPUT
    ----------
    **net** - pandapowerNet
    **nminus1_cases** - dict
        describes all N-1 cases, e.g. {"line": {"index": [1, 2, 3]}, "trafo": {"index": [0]}}
        Note: trafo3w is not supported
    **contingency_evaluation_function** - func
        function to use for power flow calculation, default pp.runpp (but only relevant for N-0 case)
    """
    if not lightsim2grid_installed:
        raise UserWarning("lightsim2grid package not installed. "
                          "Install lightsim2grid e.g. by running 'pip install lightsim2grid' in command prompt.")
    # check for continuous bus index starting with 0:
    n_bus = len(net.bus)
    last_bus = net.bus.index[-1]
    if net.bus.index[0] != 0 or last_bus != n_bus - 1 or sum(net.bus.index) != last_bus * n_bus / 2:
        raise UserWarning("bus index must be continuous and start with 0 (use pandapower.create_continuous_bus_index)")
    contingency_evaluation_function(net, **kwargs)

    trafo_flag = False
    if np.any(net.trafo.tap_phase_shifter):
        trafo_flag = True
        tap_phase_shifter, tap_pos, shift_degree = _convert_trafo_phase_shifter(net)

    # setting "slack" back-and-forth is due to the difference in interpretation of generators as "distributed slack"
    if net._options.get("distributed_slack", False):
        slack_backup = net.gen.slack.copy()
        net.gen.loc[net.gen.slack_weight != 0, 'slack'] = True
        msg = "LightSim cannot handle multiple slack bus at the moment. Only the first " \
              "slack bus of pandapower will be used."
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", msg)
            lightsim_grid_model = init_ls2g(net)
        net.gen['slack'] = slack_backup
        solver_type = SolverType.KLU if KLU_solver_available else SolverType.SparseLU
    else:
        lightsim_grid_model = init_ls2g(net)
        solver_type = SolverType.KLUSingleSlack if KLU_solver_available else SolverType.SparseLUSingleSlack

    if trafo_flag:
        net.trafo.tap_phase_shifter = tap_phase_shifter
        net.trafo.tap_pos = tap_pos
        net.trafo.shift_degree = shift_degree

    n_lines = len(net.line)
    n_lines_cases = len(nminus1_cases.get("line", {}).get("index", []))
    n_trafos = len(net.trafo)
    n_trafos_cases = len(nminus1_cases.get("trafo", {}).get("index", []))

    # todo: add option for DC power flow
    s = ContingencyAnalysisCPP(lightsim_grid_model)
    s.change_solver(solver_type)

    map_index = {}
    for element, values in nminus1_cases.items():
        index = np.array(_get_iloc_index(net[element].index.values, values["index"]), dtype=np.int64)
        map_index[element] = index.copy()  # copy() because += n_lines happens later
        if element == "trafo3w":
            raise NotImplementedError("trafo3w not implemented for lightsim2grid contingency analysis")
        elif element == "trafo":
            index += n_lines
        s.add_multiple_n1(index)

    # s.add_multiple_n1(net.line.index.values.astype(int))
    v_init = net._ppc["internal"]["V"]
    s.compute(v_init, net._options["max_iteration"], net._options["tolerance_mva"])
    v_res = s.get_voltages()
    s.compute_flows()
    kamps_all = s.get_flows()

    vm_pu = np.abs(v_res)
    net.res_bus["max_vm_pu"] = np.nanmax(vm_pu, axis=0)
    net.res_bus["min_vm_pu"] = np.nanmin(vm_pu, axis=0)

    max_i_ka_limit_all = np.r_[
        net.line.max_i_ka.values, net.trafo.sn_mva.values / (net.trafo.vn_hv_kv.values * np.sqrt(3))]
    max_loading_limit_all = np.r_[net.line["max_loading_percent_nminus1"]
    if "max_loading_percent_nminus1" in net.line.columns
    else net.line["max_loading_percent"] if n_lines > 0 else [],
    net.trafo["max_loading_percent_nminus1"]
    if "max_loading_percent_nminus1" in net.trafo.columns
    else net.trafo["max_loading_percent"] if n_trafos > 0 else []]
    voltage_all = np.r_[net.bus.loc[net.line.from_bus.values, "vn_kv"].values if n_lines > 0 else [],
    net.trafo.vn_hv_kv if n_trafos > 0 else []]
    flows_all_mva = np.nan_to_num(kamps_all * voltage_all * np.sqrt(3))
    flows_limit_all = np.nan_to_num(max_loading_limit_all / 100 * max_i_ka_limit_all * voltage_all * np.sqrt(3))

    big_number = 1e6
    for element in ("line", "trafo"):
        if len(net[element]) == 0:
            continue
        if element == "line":
            kamps_element = kamps_all[:, 0:n_lines]
            kamps_element_cause = kamps_all[0:n_lines_cases, :]
            flows_element_mva = flows_all_mva[0:n_lines_cases, :]
            max_i_ka_limit = max_i_ka_limit_all[0:n_lines]
        else:
            kamps_element = kamps_all[:, n_lines:n_lines + n_trafos]
            kamps_element_cause = kamps_all[n_lines_cases:n_lines_cases + n_trafos_cases, :]
            flows_element_mva = flows_all_mva[n_lines_cases:n_lines_cases + n_trafos_cases, :]
            max_i_ka_limit = max_i_ka_limit_all[n_lines:n_lines + n_trafos]
        # max_i_ka = np.nanmax(kamps_line, where=~np.isnan(kamps_line), axis=0, initial=0)
        cause_index = np.nanargmax(kamps_element, axis=0)
        cause_element = np.where(cause_index < n_lines_cases, "line", "trafo")
        max_i_ka = np.nanmax(kamps_element, axis=0)
        net[f"res_{element}"]["max_loading_percent"] = max_i_ka / max_i_ka_limit * 100
        min_i_ka = np.nanmin(kamps_element, axis=0, where=kamps_element != 0,
                             initial=big_number)  # this is quite daring tbh
        min_i_ka[min_i_ka == big_number] = 0
        # min_i_ka = np.nanmin(kamps_line, axis=0)
        net[f"res_{element}"]["min_loading_percent"] = min_i_ka / max_i_ka_limit * 100
        causes_overloading = np.any(kamps_element_cause > max_loading_limit_all * max_i_ka_limit_all / 100, axis=1)
        net[f"res_{element}"]["causes_overloading"] = False
        if element in nminus1_cases:
            # order of n-1 cases is always sorted, so "vertical" sorting is different than "horizontal"
            net[f"res_{element}"].loc[net[element].index.values[
                np.sort(map_index[element])], "causes_overloading"] = causes_overloading
        cause_mask = cause_element == "line"
        if "line" in map_index:
            cause_index[cause_mask] = net.line.index.values[np.sort(map_index["line"])[cause_index[cause_mask]]]
        if "trafo" in map_index:
            cause_index[~cause_mask] = net.trafo.index.values[
                np.sort(map_index["trafo"])[cause_index[~cause_mask] - n_lines_cases]]
        net[f"res_{element}"]["cause_index"] = cause_index
        net[f"res_{element}"]["cause_element"] = cause_element

        congestion_mva = flows_element_mva - flows_limit_all
        congestion_mva = np.where(congestion_mva < 0, 0, congestion_mva)
        congestion_caused = congestion_mva.sum(axis=1)
        if element in nminus1_cases:
            # order of n-1 cases is always sorted, so "vertical" sorting is different than "horizontal"
            net[f"res_{element}"].loc[net[element].index.values[
                np.sort(map_index[element])], "congestion_caused_mva"] = congestion_caused


def _convert_trafo_phase_shifter(net):
    tap_phase_shifter = net.trafo.tap_phase_shifter.values.copy()
    # vn_hv_kv = net.trafo.vn_hv_kv.values.copy()
    shift_degree = net.trafo.shift_degree.values.copy()

    tap_pos = net.trafo.tap_pos.values
    tap_neutral = net.trafo.tap_neutral.values
    tap_diff = tap_pos - tap_neutral
    tap_step_degree = net.trafo.tap_step_degree.values.copy()

    net.trafo.loc[tap_phase_shifter, 'shift_degree'] += tap_diff[tap_phase_shifter] * tap_step_degree[tap_phase_shifter]
    net.trafo["tap_pos"] = 0
    net.trafo["tap_phase_shifter"] = False

    return tap_phase_shifter, tap_pos, shift_degree


def _update_contingency_results(net, contingency_results, result_variables, nminus1, cause_element=None,
                                cause_index=None):
    for element, vars in result_variables.items():
        for var in vars:
            val = net[f"res_{element}"][var].values
            if nminus1:
                if var == "loading_percent":
                    s = 'max_loading_percent_nminus1' \
                        if 'max_loading_percent_nminus1' in net[element].columns \
                        else 'max_loading_percent'
                    # this part with cause_mask and max_mask is not very efficient nor orderly
                    loading_limit = net[element].loc[contingency_results[element]["index"], s].values
                    cause_mask = val > loading_limit
                    if np.any(cause_mask):
                        contingency_results[cause_element]["causes_overloading"][
                            contingency_results[cause_element]["index"] == cause_index] = True
                    max_mask = val > contingency_results[element].get("max_loading_percent", np.full_like(val, -1))
                    if np.any(max_mask):
                        contingency_results[element]["cause_index"][max_mask] = cause_index
                        contingency_results[element]["cause_element"][max_mask] = cause_element
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

    INPUT
    ----------
    **net** - pandapowerNet

    OUTPUT
    -------
    **element_limits** - dict
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

    INPUT
    ----------
    **element_limits** - dict
    **contingency_results** - dict
    **nminus1** - bool
    **branch_tol** - float
        tolerance of the limit violation check for branch limits
    **bus_tol** - float
        tolerance of the limit violation check for bus limits

    OUTPUT
    -------
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


def _get_sub_index(index, sub_index):
    lookup_index = np.zeros(index.max() + 1, dtype=np.int64)
    lookup_index[index] = np.arange(len(index), dtype=np.int64)
    return lookup_index[sub_index]


def _get_iloc_index(index, sub_index):
    # index_dict = {k: i for i, k in enumerate(index)}
    # return np.array([index_dict[i] for i in sub_index], dtype=np.int64)
    # should be the same:
    continuous_index = np.arange(len(index), dtype=np.int64)
    return continuous_index[_get_sub_index(index, sub_index)]


def _log_violation(element, var, val, limit_index, mask):
    if np.any(mask):
        s = ' (N-1)' if 'max' in var else ''
        with np.printoptions(precision=3, suppress=True):
            logger.info(f"{element}: {var}{s} violation at index {limit_index[mask]} ({val[mask]})")


def report_contingency_results(element_limits, contingency_results, branch_tol=1e-3, bus_tol=1e-6):
    """
    Print log messages for elements with violations of limits

    INPUT
    ----------
    **element_limits** - dict
    **contingency_results** - dict
    **branch_tol** - float
        tolerance for branch results
    **bus_tol** - float
        tolerance for bus results
    """
    for element, results in contingency_results.items():
        limit = element_limits[element]
        index = _get_sub_index(results["index"], limit["index"])
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
            elif "cause" in var:
                continue
            else:
                mask_max = val[index] > limit['max_limit'] + tol
                _log_violation(element, var, val[index], limit["index"], mask_max)
                mask_min = val[index] < limit['min_limit'] - tol
                _log_violation(element, var, val[index], limit["index"], mask_min)
