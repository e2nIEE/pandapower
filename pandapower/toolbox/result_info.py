# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import copy
from itertools import chain

import numpy as np
import pandas as pd
from pandapower.opf.validate_opf_input import _check_necessary_opf_parameters
from pandapower.toolbox import pp_elements

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def lf_info(net, numv=1, numi=2):  # pragma: no cover
    """
    Prints some basic information of the results in a net
    (max/min voltage, max trafo load, max line load).

    OPTIONAL:

        **numv** (integer, 1) - maximal number of printed maximal respectively minimal voltages

        **numi** (integer, 2) - maximal number of printed maximal loading at trafos or lines
    """
    logger.info("Max voltage in vm_pu:")
    for _, r in net.res_bus.sort_values("vm_pu", ascending=False).iloc[:numv].iterrows():
        logger.info("  %s at busidx %s (%s)", r.vm_pu, r.name, net.bus.name.at[r.name])
    logger.info("Min voltage in vm_pu:")
    for _, r in net.res_bus.sort_values("vm_pu").iloc[:numv].iterrows():
        logger.info("  %s at busidx %s (%s)", r.vm_pu, r.name, net.bus.name.at[r.name])
    logger.info("Max loading trafo in %:")
    if net.res_trafo is not None:
        for _, r in net.res_trafo.sort_values("loading_percent", ascending=False).iloc[
                    :numi].iterrows():
            logger.info("  %s loading at trafo %s (%s)", r.loading_percent, r.name,
                        net.trafo.name.at[r.name])
    logger.info("Max loading line in %:")
    for _, r in net.res_line.sort_values("loading_percent", ascending=False).iloc[:numi].iterrows():
        logger.info("  %s loading at line %s (%s)", r.loading_percent, r.name,
                    net.line.name.at[r.name])


def opf_task(net, delta_pq=1e-3, keep=False, log=True):
    """
    Collects some basic inforamtion of the optimal powerflow task und prints them.
    """
    if keep:
        net = copy.deepcopy(net)
    _check_necessary_opf_parameters(net, logger)

    opf_task_overview = {"flexibilities": dict(),
                         "network_constraints": dict(),
                         "flexibilities_without_costs": dict()}
    _determine_flexibilities_dict(net, opf_task_overview["flexibilities"], delta_pq)
    _determine_network_constraints_dict(net, opf_task_overview["network_constraints"])
    _determine_costs_dict(net, opf_task_overview)

    _check_overlapping_constraints(opf_task_overview)
    if log:
        _log_opf_task_overview(opf_task_overview)

    return opf_task_overview


def _determine_flexibilities_dict(net, data, delta_pq, **kwargs):
    """
    Determines which flexibilities exists in the net.

    INPUT:
        **net** - panpdapower net

        **data** (dict) - to store flexibilities information

        **delta_pq** (float) - if (abs(max - min) <= delta_pq) the variable is not assumed as
        flexible, since the range is as small as delta_pq (should be small, too).

    OPTIONAL:
        **kwargs**** - for comparing constraint columns with numpy.isclose(): rtol and atol
    """
    flex_elements = ["ext_grid", "gen", "dcline", "sgen", "load", "storage"]
    flex_tuple = tuple(zip(flex_elements, [True] * 3 + [False] * 3))

    for elm, controllable_default in flex_tuple:
        for power_type in ["P", "Q"]:
            key = power_type + elm
            if elm != "dcline":
                constraints = {"P": ["min_p_mw", "max_p_mw"],
                               "Q": ["min_q_mvar", "max_q_mvar"]}[power_type]
            else:
                constraints = {"P": ["max_p_mw"],
                               "Q": ["min_q_from_mvar", "max_q_from_mvar",
                                     "min_q_to_mvar", "max_q_to_mvar"]}[power_type]

            # determine indices of controllable elements, continue if no controllable element exists
            if elm in ["ext_grid", "dcline"]:
                controllables = net[elm].index
            elif "controllable" in net[elm].columns:
                controllables = net[elm].index[net[elm].controllable]
            elif controllable_default and net[elm].shape[0]:
                controllables = net[elm].index
            else:
                continue
            if not len(controllables):
                continue

            # consider delta_pq
            if len(constraints) >= 2 and pd.Series(constraints[:2]).isin(net[elm].columns).all():
                controllables = _find_idx_without_numerical_difference(
                    net[elm], constraints[0], constraints[1], delta_pq, idx=controllables,
                    equal_nan=False)
            if elm == "dcline" and power_type == "Q" and len(controllables) and \
                    pd.Series(constraints[2:4]).isin(net[elm].columns).all():
                controllables = _find_idx_without_numerical_difference(
                    net[elm], constraints[2], constraints[3], delta_pq, idx=controllables,
                    equal_nan=False)

            # add missing constraint columns
            for col_to_add in set(constraints) - set(net[elm].columns):
                net[elm][col_to_add] = np.nan

            data[key] = _cluster_same_floats(net[elm].loc[controllables], constraints, **kwargs)
            shorted = [col[:3] if col[:3] in ["min", "max"] else col for col in data[key].columns]
            if len(shorted) == len(set(shorted)):
                data[key].columns = shorted


def _find_idx_without_numerical_difference(df, column1, column2, delta, idx=None, equal_nan=False):
    """
    Returns indices where comlumn1 and column2 have a numerical difference bigger than delta.

    INPUT:
        **df** (DataFrame)

        **column1** (str) - name of first column within df to compare.
        The values of df[column1] must be numericals.

        **column2** (str) - name of second column within df to compare.
        The values of df[column2] must be numericals.

        **delta** (numerical) - value which defines whether indices are returned or not

    OPTIONAL:
        **idx** (iterable, None) - list of indices which should be considered only

        **equal_nan** (bool, False) - if False, indices are included where at least one value in
        df[column1] and df[column2] is NaN

    OUTPUT:
        **index** (pandas.Index) - index within idx where df[column1] and df[column2] deviates by
        at least delta or, if equal_na is True, one value is NaN
    """
    idx = idx if idx is not None else df.index
    idx_isnull = df.index[df[[column1, column2]].isnull().any(axis=1)]
    idx_without_null = idx.difference(idx_isnull)
    idx_no_delta = idx_without_null[(df.loc[idx_without_null, column1] - df.loc[
        idx_without_null, column2]).abs().values <= delta]

    if equal_nan:
        return idx_without_null.difference(idx_no_delta)
    else:
        return idx.difference(idx_no_delta)


def _determine_network_constraints_dict(net, data, **kwargs):
    """
    Determines which flexibilities exists in the net.

    INPUT:
        **net** - panpdapower net

        **data** (dict) - to store constraints information

    OPTIONAL:
        **kwargs**** - for comparing constraint columns with numpy.isclose(): rtol and atol
    """

    const_tuple = [("VMbus", "bus", ["min_vm_pu", "max_vm_pu"]),
                   ("LOADINGline", "line", ["max_loading_percent"]),
                   ("LOADINGtrafo", "trafo", ["max_loading_percent"]),
                   ("LOADINGtrafo3w", "trafo3w", ["max_loading_percent"])
                   ]
    for key, elm, constraints in const_tuple:
        missing_columns = set(constraints) - set(net[elm].columns)
        if net[elm].shape[0] and len(missing_columns) != len(constraints):

            # add missing constraint columns
            for col_to_add in missing_columns:
                net[elm][col_to_add] = np.nan

            data[key] = _cluster_same_floats(net[elm], constraints, **kwargs)
            shorted = [col[:3] if col[:3] in ["min", "max"] else col for col in data[key].columns]
            if len(shorted) == len(set(shorted)):
                data[key].columns = shorted


def _determine_costs_dict(net, opf_task_overview):
    """
    Determines which flexibilities do not have costs in the net. Each element is considered as one,
    i.e. if ext_grid 0, for instance,  is flexible in both, P and Q, and has one cost entry for P,
    it is not considered as 'flexibilities_without_costs'.

    INPUT:
        **net** - panpdapower net

        **opf_task_overview** (dict of dicts) - both, "flexibilities_without_costs" and
        "flexibilities" must be in opf_task_overview.keys()
    """

    cost_dfs = [df for df in ["poly_cost", "pwl_cost"] if net[df].shape[0]]
    if not len(cost_dfs):
        opf_task_overview["flexibilities_without_costs"] = "all"
        return

    flex_elements = ["ext_grid", "gen", "sgen", "load", "dcline", "storage"]

    for flex_element in flex_elements:

        # determine keys of opf_task_overview["flexibilities"] ending with flex_element
        keys = [power_type + flex_element for power_type in ["P", "Q"] if (
                power_type + flex_element) in opf_task_overview["flexibilities"].keys()]

        # determine indices of all flexibles
        idx_without_cost = set()
        for key in keys:
            idx_without_cost |= set(chain(*opf_task_overview["flexibilities"][key]["index"]))
            # simple alternative without itertools.chain():
        #            idx_without_cost |= {idx for idxs in opf_task_overview["flexibilities"][key][
        #                "index"] for idx in idxs}

        for cost_df in cost_dfs:
            idx_with_cost = set(net[cost_df].element[net[cost_df].et == flex_element].astype(np.int64))
            if len(idx_with_cost - idx_without_cost):
                logger.warning("These " + flex_element + "s have cost data but aren't flexible or" +
                               " have both, poly_cost and pwl_cost: " +
                               str(sorted(idx_with_cost - idx_without_cost)))
            idx_without_cost -= idx_with_cost

        if len(idx_without_cost):
            opf_task_overview["flexibilities_without_costs"][flex_element] = list(idx_without_cost)


def _cluster_same_floats(df, subset=None, **kwargs):
    """
    Clusters indices with close values. The values of df[subset] must be numericals.

    INPUT:
        **df** (DataFrame)

    OPTIONAL:
        **subset** (iterable, None) - list of columns of df which should be considered to cluster

        **kwargs**** - for numpy.isclose(): rtol and atol

    OUTPUT:
        **cluster_df** (DataFrame) - table of clustered values and corresponding lists of indices
    """
    if df.index.duplicated().any():
        logger.error("There are duplicated indices in df. Clusters will be determined but remain " +
                     "ambiguous.")
    subset = subset if subset is not None else df.select_dtypes(include=[
        np.number]).columns.tolist()
    uniq = ~df.duplicated(subset=subset).values

    # prepare cluster_df
    cluster_df = pd.DataFrame(np.empty((sum(uniq), len(subset) + 1)), columns=["index"] + subset)
    cluster_df["index"] = cluster_df["index"].astype(object)
    cluster_df[subset] = df.loc[uniq, subset].values

    if sum(uniq) == df.shape[0]:  # fast return if df has no duplicates
        for i1, idx in enumerate(df.index):
            cluster_df.at[i1, "index"] = [idx]
    else:  # determine index clusters
        i2 = 0
        for i1, uni in enumerate(uniq):
            if uni:
                cluster_df.at[i2, "index"] = list(df.index[np.isclose(
                    df[subset].values.astype(float),
                    df[subset].iloc[[i1]].values.astype(float),
                    equal_nan=True, **kwargs).all(axis=1)])
                i2 += 1

    return cluster_df


def _check_overlapping_constraints(opf_task_overview):
    """
    Logs variables where the minimum constraint is bigger than the maximum constraint.
    """
    overlap = []
    for dict_key in ["flexibilities", "network_constraints"]:
        for key, df in opf_task_overview[dict_key].items():
            min_col = [col for col in df.columns if "min" in col]
            max_col = [col for col in df.columns if "max" in col]
            n_col = min(len(min_col), len(max_col))
            for i_col in range(n_col):
                assert min_col[i_col].replace("min", "") == max_col[i_col].replace("max", "")
                if (df[min_col[i_col]] > df[max_col[i_col]]).any():
                    overlap.append(key)
    if len(overlap):
        logger.error("At these variables, there is a minimum constraint exceeding the maximum " +
                     "constraint value: " + str(overlap))


def _log_opf_task_overview(opf_task_overview):
    """
    Logs OPF task information.
    """
    s = ""
    for dict_key, data in opf_task_overview.items():
        if isinstance(data, str):
            assert dict_key == "flexibilities_without_costs"
            s += "\n\n%s flexibilities without costs" % data
            continue
        else:
            assert isinstance(data, dict)
        heading_logged = False
        keys, elms = _get_keys_and_elements_from_opf_task_dict(data)
        for key, elm in zip(keys, elms):
            assert elm in key
            df = data[key]

            if dict_key in ["flexibilities", "network_constraints"]:
                if not df.shape[0]:
                    continue
                if not heading_logged:
                    s += "\n\n%s:" % dict_key
                    heading_logged = True

                # --- logging information
                len_idx = len(list(chain(*df["index"])))
                if df.shape[0] > 1:
                    s += "\n    %ix %s" % (len_idx, key)
                else:
                    if not len(set(df.columns).symmetric_difference({"index", "min", "max"})):
                        s += "\n    %g <= %ix %s (all) <= %g" % (
                            df.loc[0, "min"], len_idx, key, df.loc[0, "max"])
                    else:
                        s += "\n    %ix %s (all) with these constraints:" % (len_idx, key)
                        for col in set(df.columns) - {"index"}:
                            s += " %s=%g" % (col, df.loc[0, col])
            elif dict_key == "flexibilities_without_costs":
                if not heading_logged:
                    s += "\n\n%s:" % dict_key
                    heading_logged = True
                s += "\n%ix %s" % (len(df), key)
            else:
                raise NotImplementedError("Key %s is unknown to this code." % dict_key)
    logger.info(s + "\n")


def _get_keys_and_elements_from_opf_task_dict(dict_):
    keys = list(dict_.keys())
    elms = ["".join(c for c in key if not c.isupper()) for key in keys]
    keys = list(np.array(keys)[np.argsort(elms)])
    elms = sorted(elms)
    return keys, elms


def switch_info(net, sidx):  # pragma: no cover
    """
    Prints what buses and elements are connected by a certain switch.
    """
    switch_type = net.switch.at[sidx, "et"]
    bidx = net.switch.at[sidx, "bus"]
    bus_name = net.bus.at[bidx, "name"]
    eidx = net.switch.at[sidx, "element"]
    if switch_type == "b":
        bus2_name = net.bus.at[eidx, "name"]
        logger.info("Switch %u connects bus %u (%s) with bus %u (%s)" % (sidx, bidx, bus_name,
                                                                         eidx, bus2_name))
    elif switch_type == "l":
        line_name = net.line.at[eidx, "name"]
        logger.info("Switch %u connects bus %u (%s) with line %u (%s)" % (sidx, bidx, bus_name,
                                                                          eidx, line_name))
    elif switch_type == "t":
        trafo_name = net.trafo.at[eidx, "name"]
        logger.info("Switch %u connects bus %u (%s) with trafo %u (%s)" % (sidx, bidx, bus_name,
                                                                           eidx, trafo_name))


def overloaded_lines(net, max_load=100):
    """
    Returns the results for all lines with loading_percent > max_load or None, if
    there are none.
    """
    if net.converged:
        return net["res_line"].index[net["res_line"]["loading_percent"] > max_load]
    else:
        raise UserWarning("The last loadflow terminated erratically, results are invalid!")


def violated_buses(net, min_vm_pu, max_vm_pu):
    """
    Returns all bus indices where vm_pu is not within min_vm_pu and max_vm_pu or returns None, if
    there are none of those buses.
    """
    if net.converged:
        return net["bus"].index[(net["res_bus"]["vm_pu"] < min_vm_pu) |
                                (net["res_bus"]["vm_pu"] > max_vm_pu)]
    else:
        raise UserWarning("The last loadflow terminated erratically, results are invalid!")


def clear_result_tables(net):
    """
    Clears all ``res_`` DataFrames in net.
    """
    for key in net.keys():
        if isinstance(net[key], pd.DataFrame) and key[:3] == "res" and net[key].shape[0]:
            net[key] = net[key].drop(net[key].index)


def res_power_columns(element_type, side=0):
    """Returns columns names of result tables for active and reactive power

    Parameters
    ----------
    element_type : str
        name of element table, e.g. "gen"
    side : typing.Union[int, str], optional
        Defines for branch elements which branch side is considered, by default 0

    Returns
    -------
    list[str]
        columns names of result tables for active and reactive power

    Examples
    --------
    >>> res_power_columns("gen")
    ["p_mw", "q_mvar"]
    >>> res_power_columns("line", "from")
    ["p_from_mw", "q_from_mvar"]
    >>> res_power_columns("line", 0)
    ["p_from_mw", "q_from_mvar"]
    >>> res_power_columns("line", "all")
    ["p_from_mw", "q_from_mvar", "p_to_mw", "q_to_mvar"]
    """
    if element_type in pp_elements(branch_elements=False, other_elements=False):
        return ["p_mw", "q_mvar"]
    elif element_type in pp_elements(bus=False, bus_elements=False, other_elements=False):
        if isinstance(side, int):
            if element_type == "trafo":
                side_options = {0: "hv", 1: "lv"}
            elif element_type == "trafo3w":
                side_options = {0: "hv", 1: "mv", 2: "lv"}
            else:
                side_options = {0: "from", 1: "to"}
            side = side_options[side]
        if side != "all":
            return [f"p_{side}_mw", f"q_{side}_mvar"]
        else:
            cols = res_power_columns(element_type, side=0) + \
                res_power_columns(element_type, side=1)
            if element_type == "trafo3w":
                cols += res_power_columns(element_type, side=2)
            return cols
    else:
        raise ValueError(f'{element_type=} cannot be considered by res_power_columns().')