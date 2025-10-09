import copy
import numpy as np
from typing import Optional
import multiprocessing as mp
from functools import partial
from pandapower.run import runpp
from pandapower.auxiliary import pandapowerNet

import logging

logger = logging.getLogger(__name__)

def _run_single_contingency(contingency_case, net, pf_options_nminus1, result_variables,
                            contingency_evaluation_function, raise_errors, **kwargs):
    """
    Worker function executed by each parallel process.
    It runs a single contingency analysis.
    """
    element, i = contingency_case
    # Create a shallow copy of the net object and a deep copy of the element to ensure process safety
    net_copy = copy.copy(net)
    net_copy[element] = copy.deepcopy(net[element])
    net_copy[element].at[i, 'in_service'] = False

    try:
        contingency_evaluation_function(net_copy, **pf_options_nminus1, **kwargs)
        # If successful, extract the relevant results
        result_pack = {
            "success": True,
            "case": contingency_case,
            "res_vals": {}
        }
        for res_element, res_vars in result_variables.items():
            result_pack["res_vals"][res_element] = {var: net_copy[f"res_{res_element}"][var].values for var in res_vars}
        return result_pack
    except Exception as err:
        logger.error(f"Contingency {element} {i} failed: {err}")
        if raise_errors:
            raise err
        # Return a pack indicating failure
        return {"success": False, "case": contingency_case}


def run_contingency_parallel(net: pandapowerNet,
                             nminus1_cases: dict,
                             pf_options: Optional[dict] = None,
                             pf_options_nminus1: Optional[dict] = None,
                             write_to_net: bool = True,
                             contingency_evaluation_function=runpp,
                             n_procs: int = 1,
                             **kwargs):
    """
    Obtain either loading (N-0) or max. loading (N-0 and all N-1 cases), and min/max bus voltage magnitude.
    This function can be run in parallel by setting n_procs.

    :param pandapowerNet net: The pandapower network
    :param dict nminus1_cases: describes all N-1 cases, e.g. {"line": {"index": [1, 2, 3]}, "trafo": {"index": [0]}}
    :param dict pf_options: options for power flow calculation in N-0 case
    :param dict pf_options_nminus1: options for power flow calculation in N-1 cases
    :param bool write_to_net: whether to write the results of contingency analysis to net (in `res_` tables).
    :param callable contingency_evaluation_function: function to use for power flow calculation, default pp.runpp
    :param int n_procs: Number of processors to use. If None, all available cores are used. If 1, runs sequentially.
    :return: contingency results dict of arrays per element for index, min/max result
    :rtype: dict
    """
    # --- Setup and Initialization ---
    raise_errors = kwargs.get("raise_errors", False)
    if "recycle" in kwargs:
        kwargs["recycle"] = False  # so that we can be sure it doesn't happen
    if pf_options is None:
        pf_options = net.user_pf_options.get("pf_options", net.user_pf_options)
    if pf_options_nminus1 is None:
        pf_options_nminus1 = net.user_pf_options.get("pf_options_nminus1", net.user_pf_options)
    if kwargs is not None:
        pf_options = {key: val for key, val in pf_options.items() if key not in kwargs.keys()}
        pf_options_nminus1 = {key: val for key, val in pf_options_nminus1.items() if key not in kwargs.keys()}

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

    # --- N-1 Contingency Analysis ---
    if n_procs is None:
        n_procs = mp.cpu_count()

    # --- Parallel Execution Path ---
    if n_procs > 1:
        tasks = []
        for element, val in nminus1_cases.items():
            for i in val["index"]:
                if net[element].at[i, "in_service"]:
                    tasks.append((element, i))

        worker_func = partial(_run_single_contingency, net=net, pf_options_nminus1=pf_options_nminus1,
                              result_variables=result_variables,
                              contingency_evaluation_function=contingency_evaluation_function,
                              raise_errors=raise_errors, **kwargs)

        with mp.Pool(processes=n_procs) as pool:
        #with cf.ProcessPoolExecutor(max_workers=n_procs) as pool:
            results_list = pool.map(worker_func, tasks)

        # Aggregate results from parallel runs
        for single_result in results_list:
            if single_result["success"]:
                _update_contingency_results_parallel(net, contingency_results, result_variables, nminus1=True,
                                                     cause_element=single_result["case"][0],
                                                     cause_index=single_result["case"][1],
                                                     parallel_results=single_result["res_vals"])

    # --- Sequential Execution Path ---
    else:
        for element, val in nminus1_cases.items():
            for i in val["index"]:
                if not net[element].at[i, "in_service"]:
                    continue
                net[element].at[i, 'in_service'] = False
                try:
                    contingency_evaluation_function(net, **pf_options_nminus1, **kwargs)
                    _update_contingency_results_parallel(net, contingency_results, result_variables, nminus1=True,
                                                         cause_element=element, cause_index=i)
                except Exception as err:
                    logger.error(f"{element} {i} causes {err}")
                    if raise_errors:
                        raise err
                finally:
                    net[element].at[i, 'in_service'] = True

    # --- N-0 (Base Case) Analysis ---
    contingency_evaluation_function(net, **pf_options, **kwargs)
    _update_contingency_results_parallel(net, contingency_results, result_variables, nminus1=False)

    # --- Write Results to Net ---
    if write_to_net:
        for element, element_results in contingency_results.items():
            index = element_results["index"]
            for var, val in element_results.items():
                if var == "index" or var in net[f"res_{element}"].columns.values:
                    continue
                net[f"res_{element}"].loc[index, var] = val

    return contingency_results


def _update_contingency_results_parallel(net, contingency_results, result_variables, nminus1, cause_element=None,
                                         cause_index=None, parallel_results=None):
    """
    Updates the main results dictionary with results from a single contingency run.
    This function is used by both the sequential and parallel execution paths.
    """
    for element, vars_list in result_variables.items():
        for var in vars_list:
            # Get result values from the net object (sequential) or the passed results (parallel)
            val = parallel_results[element][var] if parallel_results else net[f"res_{element}"][var].values

            if nminus1:
                if var == "loading_percent":
                    s = 'max_loading_percent_nminus1' if 'max_loading_percent_nminus1' in net[
                        element].columns else 'max_loading_percent'
                    loading_limit = net[element].loc[contingency_results[element]["index"], s].values
                    cause_mask = val > loading_limit
                    if np.any(cause_mask):
                        contingency_results[cause_element]["causes_overloading"][
                            contingency_results[cause_element]["index"] == cause_index] = True

                    # Check if this contingency causes a new maximum loading
                    current_max = contingency_results[element].get("max_loading_percent", np.full_like(val, -1.0))
                    max_mask = val > current_max
                    if np.any(max_mask):
                        contingency_results[element]["cause_index"][max_mask] = cause_index
                        contingency_results[element]["cause_element"][max_mask] = cause_element

                # Update min/max values for all variables
                for func, min_max in ((np.fmax, "max"), (np.fmin, "min")):
                    key = f"{min_max}_{var}"
                    # Initialize the array with NaNs if it doesn't exist
                    if key not in contingency_results[element]:
                        contingency_results[element][key] = np.full_like(val, np.nan, dtype=np.float64)

                    # Use in_service mask for sequential, assume all are valid for parallel (handled in worker)
                    where_mask = net[element]["in_service"].values if not parallel_results else ~np.isnan(val)

                    func(val, contingency_results[element][key], out=contingency_results[element][key],
                         where=where_mask & ~np.isnan(val))
            else:  # nminus1 is False (N-0 case)
                contingency_results[element][var] = val


if __name__ == '__main__':
    import time
    import pandapower as pp
    from pandapower.contingency import get_element_limits, check_elements_within_limits, report_contingency_results

    net = pp.networks.case2869pegase()
    nminus1_cases = {"line": {"index": net.line.index.values}}

    # res = pp.contingency.run_contingency(net, nminus1_cases)
    start = time.time()
    res = run_contingency_parallel(net, nminus1_cases, n_procs=8)
    parallel_time = time.time() - start

    print("Parallel time with 8 processes:", parallel_time)

    element_limits = get_element_limits(net)
    check_elements_within_limits(element_limits, res, True)
    report_contingency_results(element_limits, res)
