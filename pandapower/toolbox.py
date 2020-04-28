# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import copy
import gc
from collections import defaultdict
from collections.abc import Iterable
from itertools import chain

import numpy as np
import pandas as pd
from packaging import version

from pandapower.auxiliary import get_indices, pandapowerNet, _preserve_dtypes
from pandapower.create import create_switch, create_line_from_parameters, \
    create_impedance, create_empty_network, create_gen, create_ext_grid, \
    create_load, create_shunt, create_bus, create_sgen
from pandapower.opf.validate_opf_input import _check_necessary_opf_parameters
from pandapower.run import runpp

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


# --- general issues
def element_bus_tuples(bus_elements=True, branch_elements=True, res_elements=False):
    """
    Utility function
    Provides the tuples of elements and corresponding columns for buses they are connected to
    :param bus_elements: whether tuples for bus elements e.g. load, sgen, ... are included
    :param branch_elements: whether branch elements e.g. line, trafo, ... are included
    :return: set of tuples with element names and column names
    """
    ebts = set()
    if bus_elements:
        ebts.update([("sgen", "bus"), ("load", "bus"), ("ext_grid", "bus"), ("gen", "bus"),
                     ("ward", "bus"), ("xward", "bus"), ("shunt", "bus"),
                     ("storage", "bus")])
    if branch_elements:
        ebts.update([("line", "from_bus"), ("line", "to_bus"), ("impedance", "from_bus"),
                     ("switch", "bus"), ("impedance", "to_bus"), ("trafo", "hv_bus"),
                     ("trafo", "lv_bus"), ("trafo3w", "hv_bus"), ("trafo3w", "mv_bus"),
                     ("trafo3w", "lv_bus"), ("dcline", "from_bus"), ("dcline", "to_bus")])
    if res_elements:
        elements_without_res = ["switch", "measurement"]
        ebts.update(
            [("res_" + ebt[0], ebt[1]) for ebt in ebts if ebt[0] not in elements_without_res])
    return ebts


def pp_elements(bus=True, bus_elements=True, branch_elements=True, other_elements=True,
                res_elements=False):
    """
    Returns the list of pandapower elements.
    """
    pp_elms = set(["bus"]) if bus else set()
    pp_elms |= set([el[0] for el in element_bus_tuples(
        bus_elements=bus_elements, branch_elements=branch_elements, res_elements=res_elements)])
    if other_elements:
        pp_elms |= {"measurement"}
    return pp_elms


def branch_element_bus_dict(include_switch=False):
    """
    Returns a dict with keys of branch elements and values of bus column names as list.
    """
    ebts = element_bus_tuples(bus_elements=False, branch_elements=True, res_elements=False)
    branch_elements = {ebt[0] for ebt in ebts}
    bebd = {elm: [] for elm in branch_elements}
    for elm, bus in ebts:
        bebd[elm].append(bus)
    if not include_switch:
        del bebd["switch"]
    return bebd


# def pq_from_cosphi(s, cosphi, qmode, pmode):
#    """
#    Calculates P/Q values from rated apparent power and cosine(phi) values.
#       - s: rated apparent power
#       - cosphi: cosine phi of the
#       - qmode: "ind" for inductive or "cap" for capacitive behaviour
#       - pmode: "load" for load or "gen" for generation
#    As all other pandapower functions this function is based on the consumer viewpoint. For active
#    power, that means that loads are positive and generation is negative. For reactive power,
#    inductive behaviour is modeled with positive values, capacitive behaviour with negative values.
#    """
#    s = np.array(ensure_iterability(s))
#    cosphi = np.array(ensure_iterability(cosphi, len(s)))
#    qmode = np.array(ensure_iterability(qmode, len(s)))
#    pmode = np.array(ensure_iterability(pmode, len(s)))
#
#    # qmode consideration
#    unknown_qmode = set(qmode) - set(["ind", "cap", "ohm"])
#    if len(unknown_qmode):
#        raise ValueError("Unknown qmodes: " + str(list(unknown_qmode)))
#    qmode_is_ohm = qmode == "ohm"
#    if any(cosphi[qmode_is_ohm] != 1):
#        raise ValueError("qmode cannot be 'ohm' if cosphi is not 1.")
#    qsign = np.ones(qmode.shape)
#    qsign[qmode == "cap"] = -1
#
#    # pmode consideration
#    unknown_pmode = set(pmode) - set(["load", "gen"])
#    if len(unknown_pmode):
#        raise ValueError("Unknown pmodes: " + str(list(unknown_pmode)))
#    psign = np.ones(pmode.shape)
#    psign[pmode == "gen"] = -1
#
#    # calculate p and q
#    p = psign * s * cosphi
#    q = qsign * np.sqrt(s ** 2 - p ** 2)
#
#    if len(p) > 1:
#        return p, q
#    else:
#        return p[0], q[0]


def pq_from_cosphi(s, cosphi, qmode, pmode):
    """
    Calculates P/Q values from rated apparent power and cosine(phi) values.

       - s: rated apparent power
       - cosphi: cosine phi of the
       - qmode: "ind" for inductive or "cap" for capacitive behaviour
       - pmode: "load" for load or "gen" for generation

    As all other pandapower functions this function is based on the consumer viewpoint. For active
    power, that means that loads are positive and generation is negative. For reactive power,
    inductive behaviour is modeled with positive values, capacitive behaviour with negative values.
    """
    if hasattr(s, "__iter__"):
        s = ensure_iterability(s)
        cosphi = ensure_iterability(cosphi, len(s))
        qmode = ensure_iterability(qmode, len(s))
        pmode = ensure_iterability(pmode, len(s))
        p, q = [], []
        for s_, cosphi_, qmode_, pmode_ in zip(s, cosphi, qmode, pmode):
            p_, q_ = _pq_from_cosphi(s_, cosphi_, qmode_, pmode_)
            p.append(p_)
            q.append(q_)
        return np.array(p), np.array(q)
    else:
        return _pq_from_cosphi(s, cosphi, qmode, pmode)


def _pq_from_cosphi(s, cosphi, qmode, pmode):
    if qmode == "ind":
        qsign = 1 if pmode == "load" else -1
    elif qmode == "cap":
        qsign = -1 if pmode == "load" else 1
    else:
        raise ValueError("Unknown mode %s - specify 'ind' or 'cap'" % qmode)

    p = s * cosphi
    q = qsign * np.sqrt(s ** 2 - p ** 2)
    return p, q


# def cosphi_from_pq(p, q):
#    """
#    Analog to pq_from_cosphi, but other way around.
#    In consumer viewpoint (pandapower): cap=overexcited and ind=underexcited
#    """
#    p = np.array(ensure_iterability(p))
#    q = np.array(ensure_iterability(q, len(p)))
#    if len(p) != len(q):
#        raise ValueError("p and q must have the same length.")
#    p_is_zero = np.array(p == 0)
#    cosphi = np.empty(p.shape)
#    if sum(p_is_zero):
#        cosphi[p_is_zero] = np.nan
#        logger.warning("A cosphi from p=0 is undefined.")
#    cosphi[~p_is_zero] = np.cos(np.arctan(q[~p_is_zero] / p[~p_is_zero]))
#    s = (p ** 2 + q ** 2) ** 0.5
#    pmode = np.array(["undef", "load", "gen"])[np.sign(p).astype(int)]
#    qmode = np.array(["ohm", "ind", "cap"])[np.sign(q).astype(int)]
#    if len(p) > 1:
#        return cosphi, s, qmode, pmode
#    else:
#        return cosphi[0], s[0], qmode[0], pmode[0]


def cosphi_from_pq(p, q):
    if hasattr(p, "__iter__"):
        assert len(p) == len(q)
        s, cosphi, qmode, pmode = [], [], [], []
        for p_, q_ in zip(p, q):
            cosphi_, s_, qmode_, pmode_ = _cosphi_from_pq(p_, q_)
            s.append(s_)
            cosphi.append(cosphi_)
            qmode.append(qmode_)
            pmode.append(pmode_)
        return np.array(cosphi), np.array(s), np.array(qmode), np.array(pmode)
    else:
        return _cosphi_from_pq(p, q)


def _cosphi_from_pq(p, q):
    """
    Analog to pq_from_cosphi, but other way around.
    In consumer viewpoint (pandapower): cap=overexcited and ind=underexcited
    """
    if p == 0:
        cosphi = np.nan
        logger.warning("A cosphi from p=0 is undefined.")
    else:
        cosphi = np.cos(np.arctan(q / p))
    s = (p ** 2 + q ** 2) ** 0.5
    pmode = ["undef", "load", "gen"][int(np.sign(p))]
    qmode = ["ohm", "ind", "cap"][int(np.sign(q))]
    return cosphi, s, qmode, pmode


def dataframes_equal(x_df, y_df, tol=1.e-14, ignore_index_order=True):
    """
    Returns a boolean whether the nets are equal or not.
    """
    if ignore_index_order:
        x_df.sort_index(axis=1, inplace=True)
        y_df.sort_index(axis=1, inplace=True)
        x_df.sort_index(axis=0, inplace=True)
        y_df.sort_index(axis=0, inplace=True)
    # eval if two DataFrames are equal, with regard to a tolerance
    if x_df.shape == y_df.shape:
        if x_df.shape[0]:
            # we use numpy.allclose to grant a tolerance on numerical values
            numerical_equal = np.allclose(x_df.select_dtypes(include=[np.number]),
                                          y_df.select_dtypes(include=[np.number]),
                                          atol=tol, equal_nan=True)
        else:
            numerical_equal = True
        # ... use pandas .equals for the rest, which also evaluates NaNs to be equal
        rest_equal = x_df.select_dtypes(exclude=[np.number]).equals(
            y_df.select_dtypes(exclude=[np.number]))

        return numerical_equal & rest_equal
    else:
        return False


def compare_arrays(x, y):
    """
    Returns an array of bools whether array x is equal to array y. Strings are allowed in x
    or y. NaN values are assumed as equal.
    """
    if x.shape == y.shape:
        # (x != x) is like np.isnan(x) - but works also for strings
        return np.equal(x, y) | ((x != x) & (y != y))
    else:
        raise ValueError("x and y needs to have the same shape.")


def ensure_iterability(var, len_=None):
    """
    Ensures iterability of a variable (and optional length).
    """
    if hasattr(var, "__iter__") and not isinstance(var, str):
        if isinstance(len_, int) and len(var) != len_:
            raise ValueError("Length of variable differs from %i." % len_)
    else:
        len_ = len_ or 1
        var = [var] * len_
    return var


# --- Information
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
            idx_with_cost = set(net[cost_df].element[net[cost_df].et == flex_element].astype(int))
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


def nets_equal(net1, net2, check_only_results=False, exclude_elms=None, **kwargs):
    """
    Compares the DataFrames of two networks. The networks are considered equal
    if they share the same keys and values, except of the
    'et' (elapsed time) entry which differs depending on
    runtime conditions and entries stating with '_'.
    """
    eq = isinstance(net1, pandapowerNet) and isinstance(net2, pandapowerNet)
    exclude_elms = [] if exclude_elms is None else list(exclude_elms)
    exclude_elms += ["res_" + ex for ex in exclude_elms]
    not_equal = []

    if eq:
        # for two networks make sure both have the same keys that do not start with "_"...
        net1_keys = [key for key in net1.keys() if not (key.startswith("_") or key in exclude_elms)]
        net2_keys = [key for key in net2.keys() if not (key.startswith("_") or key in exclude_elms)]
        keys_to_check = set(net1_keys) & set(net2_keys)
        key_difference = set(net1_keys) ^ set(net2_keys)

        if len(key_difference) > 0:
            logger.info("Networks entries mismatch at: %s" % key_difference)
            if not check_only_results:
                return False

        # ... and then iter through the keys, checking for equality for each table
        for df_name in list(keys_to_check):
            # skip 'et' (elapsed time) and entries starting with '_' (internal vars)
            if (df_name != 'et' and not df_name.startswith("_")):
                if check_only_results and not df_name.startswith("res_"):
                    continue  # skip anything that is not a result table

                if isinstance(net1[df_name], pd.DataFrame) and isinstance(net2[df_name],
                                                                          pd.DataFrame):
                    frames_equal = dataframes_equal(net1[df_name], net2[df_name], **kwargs)
                    eq &= frames_equal

                    if not frames_equal:
                        not_equal.append(df_name)

    if len(not_equal) > 0:
        logger.error("Networks do not match in DataFrame(s): %s" % (', '.join(not_equal)))

    return eq


def clear_result_tables(net):
    """
    Clears all 'res_...' DataFrames in net.
    """
    for key in net.keys():
        if isinstance(net[key], pd.DataFrame) and key[:3] == "res" and net[key].shape[0]:
            net[key].drop(net[key].index, inplace=True)


# --- Simulation setup and preparations
def add_column_from_node_to_elements(net, column, replace, elements=None, branch_bus=None,
                                     verbose=True):
    """
    Adds column data to elements, inferring them from the column data of buses they are
    connected to.

    INPUT:
        **net** (pandapowerNet) - the pandapower net that will be changed

        **column** (string) - name of column that should be copied from the bus table to the element
            table

        **replace** (boolean) - if True, an existing column in the element table will be overwritten

        **elements** (list) - list of elements that should get the column values from the bus table

        **branch_bus** (list) - defines which bus should be considered for branch elements.
            'branch_bus' must have the length of 2. One entry must be 'from_bus' or 'to_bus', the
            other 'hv_bus' or 'lv_bus'

    EXAMPLE:
        compare to add_zones_to_elements()
    """
    branch_bus = ["from_bus", "hv_bus"] if branch_bus is None else branch_bus
    if column not in net.bus.columns:
        raise ValueError("%s is not in net.bus.columns" % column)
    elements = elements if elements is not None else pp_elements(bus=False, other_elements=False)
    elements_to_replace = elements if replace else [
        el for el in elements if column not in net[el].columns or net[el][column].isnull().all()]
    # bus elements
    for element, bus_type in element_bus_tuples(bus_elements=True, branch_elements=False):
        if element in elements_to_replace:
            net[element][column] = net["bus"][column].loc[net[element][bus_type]].values
    # branch elements
    to_validate = {}
    for element, bus_type in element_bus_tuples(bus_elements=False, branch_elements=True):
        if element in elements_to_replace:
            if bus_type in (branch_bus + ["bus"]):  # copy data, append branch_bus for switch.bus
                net[element][column] = net["bus"][column].loc[net[element][bus_type]].values
            else:  # save data for validation
                to_validate[element] = net["bus"][column].loc[net[element][bus_type]].values
    # validate branch elements, but do not validate double and switches at all
    already_validated = ["switch"]
    for element, bus_type in element_bus_tuples(bus_elements=False, branch_elements=True):
        if (element in elements_to_replace) & (element not in already_validated):
            already_validated += [element]
            crossing = sum(~compare_arrays(net[element][column].values, to_validate[element]))
            if crossing > 0:
                if verbose:
                    logger.warning("There have been %i %ss with different " % (crossing, element) +
                                   "%s data at from-/hv- and to-/lv-bus" % column)
                else:
                    logger.debug("There have been %i %ss with different " % (crossing, element) +
                                 "%s data at from-/hv- and to-/lv-bus" % column)


def add_column_from_element_to_elements(net, column, replace, elements=None,
                                        continue_on_missing_column=True):
    """
    Adds column data to elements, inferring them from the column data of the elements linked by the
    columns "element" and "element_type" or "et".

    INPUT:
        **net** (pandapowerNet) - the pandapower net that will be changed

        **column** (string) - name of column that should be copied from the tables of the elements.

        **replace** (boolean) - if True, an existing column will be overwritten

        **elements** (list) - list of elements that should get the column values from the linked
            element tables. If None, all elements with the columns "element" and "element_type" or
            "et" are considered (these are currently "measurement" and "switch").

        **continue_on_missing_column** (Boolean, True) - If False, a error will be raised in case of
            an element table has no column 'column' although this element is refered in 'elements'.
            E.g. 'measurement' is in 'elements' and in net.measurement is a trafo measurement but
            in net.trafo there is no column 'name' although column=='name' - ni this case
            'continue_on_missing_column' acts.

    EXAMPLE:
        import pandapower as pp
        import pandapower.networks as pn
        net = pn.create_cigre_network_mv()
        pp.create_measurement(net, "i", "trafo", 5, 3, 0, side="hv")
        pp.create_measurement(net, "i", "line", 5, 3, 0, side="to")
        pp.create_measurement(net, "p", "bus", 5, 3, 2)
        print(net.measurement.name.values, net.switch.name.values)
        pp.add_column_from_element_to_elements(net, "name", True)
        print(net.measurement.name.values, net.switch.name.values)
    """
    elements = elements if elements is not None else pp_elements()
    elements_with_el_and_et_column = [el for el in elements if "element" in net[el].columns and (
            "element_type" in net[el].columns or "et" in net[el].columns)]
    elements_to_replace = elements_with_el_and_et_column if replace else [
        el for el in elements_with_el_and_et_column if column not in net[el].columns or net[el][
            column].isnull().all()]
    for el in elements_to_replace:
        et_col = "element_type" if "element_type" in net[el].columns else "et"
        element_type = net[el][et_col]
        for short, complete in [("t", "trafo"), ("t3", "trafo3w"), ("l", "line"), ("s", "switch"),
                                ("b", "bus")]:
            element_type.loc[element_type == short] = complete
        element_types_without_column = [et for et in set(element_type) if column not in
                                        net[et].columns]
        if len(element_types_without_column):
            message = "%s is not in net[et].columns with et in " % column + str(
                element_types_without_column)
            if not continue_on_missing_column:
                raise KeyError(message)
            else:
                logger.debug(message)
        for et in list(set(element_type) - set(element_types_without_column)):
            idx_et = element_type.index[element_type == et]
            net[el].loc[idx_et, column] = net[et][column].loc[net[el].element[idx_et]].values


def add_zones_to_elements(net, replace=True, elements=None, **kwargs):
    """
    Adds zones to elements, inferring them from the zones of buses they are connected to.
    """
    elements = ["line", "trafo", "ext_grid", "switch"] if elements is None else elements
    add_column_from_node_to_elements(net, "zone", replace=replace, elements=elements, **kwargs)


def reindex_buses(net, bus_lookup):
    """
    Changes the index of net.bus and considers the new bus indices in all other pandapower element
    tables.

    INPUT:
      **net** - pandapower network

      **bus_lookup** (dict) - the keys are the old bus indices, the values the new bus indices
    """
    not_fitting_bus_lookup_keys = set(bus_lookup.keys()) - set(net.bus.index)
    if len(not_fitting_bus_lookup_keys):
        logger.error("These bus indices are unknown to net. Thus, they cannot be reindexed: " +
                     str(not_fitting_bus_lookup_keys))

    missing_bus_indices = sorted(set(net.bus.index) - set(bus_lookup.keys()))
    if len(missing_bus_indices):
        bus_lookup.update({b: b for b in missing_bus_indices})

    net.bus.index = get_indices(net.bus.index, bus_lookup)
    net.res_bus.index = get_indices(net.res_bus.index, bus_lookup)

    for element, value in element_bus_tuples():
        net[element][value] = get_indices(net[element][value], bus_lookup)
    net["bus_geodata"].set_index(get_indices(net["bus_geodata"].index, bus_lookup), inplace=True)
    bb_switches = net.switch[net.switch.et == "b"]
    net.switch.loc[bb_switches.index, "element"] = get_indices(bb_switches.element, bus_lookup)
    bus_meas = net.measurement.element_type == "bus"
    net.measurement.loc[bus_meas, "element"] = get_indices(net.measurement.loc[bus_meas, "element"],
                                                           bus_lookup)
    side_meas = pd.to_numeric(net.measurement.side, errors="coerce").notnull()
    net.measurement.loc[side_meas, "side"] = get_indices(net.measurement.loc[side_meas, "side"],
                                                         bus_lookup)
    return bus_lookup


def create_continuous_bus_index(net, start=0, store_old_index=False):
    """
    Creates a continuous bus index starting at 'start' and replaces all
    references of old indices by the new ones.

    INPUT:
      **net** - pandapower network

    OPTIONAL:
      **start** - index begins with "start"

      **store_old_index** - if True, stores the old index in net.bus["old_index"]

    OUTPUT:
      **bus_lookup** - mapping of old to new index
    """
    net.bus.sort_index(inplace=True)
    if store_old_index:
        net.bus["old_index"] = net.bus.index.values
    new_bus_idxs = list(np.arange(start, len(net.bus) + start))
    bus_lookup = dict(zip(net["bus"].index.values, new_bus_idxs))
    reindex_buses(net, bus_lookup)
    return bus_lookup


def reindex_elements(net, element, new_indices, old_indices=None):
    """
    Changes the index of net[element].

    INPUT:
      **net** - pandapower network

      **element** (str) - name of the element table

      **new_indices** (iterable) - list of new indices

    OPTIONAL:
      **old_indices** (iterable) - list of old/previous indices which will be replaced.
          If None, all indices are considered.
    """
    old_indices = old_indices if old_indices is not None else net[element].index
    if not len(new_indices) or not net[element].shape[0]:
        return
    assert len(new_indices) == len(old_indices)
    lookup = dict(zip(old_indices, new_indices))

    if element == "bus":
        reindex_buses(net, lookup)
        return

    # --- reindex
    net[element]["index"] = net[element].index
    net[element].loc[old_indices, "index"] = get_indices(old_indices, lookup)
    net[element].set_index("index", inplace=True)

    # --- adapt measurement link
    if element in ["line", "trafo", "trafo3w"]:
        affected = net.measurement[(net.measurement.element_type == element) &
                                   (net.measurement.element.isin(old_indices))]
        if len(affected):
            net.measurement.loc[affected.index, "element"] = get_indices(affected.element, lookup)

    # --- adapt switch link
    if element in ["line", "trafo"]:
        affected = net.switch[(net.switch.et == element[0]) &
                              (net.switch.element.isin(old_indices))]
        if len(affected):
            net.switch.loc[affected.index, "element"] = get_indices(affected.element, lookup)

    # --- adapt line_geodata index
    if element == "line" and "line_geodata" in net and net["line_geodata"].shape[0]:
        net["line_geodata"]["index"] = net["line_geodata"].index
        net["line_geodata"].loc[old_indices, "index"] = get_indices(old_indices, lookup)
        net["line_geodata"].set_index("index", inplace=True)

    # adapt index in cost dataframes
    for cost_df in ["pwl_cost", "poly_cost"]:
        element_in_cost_df = net[cost_df].et == element
        if sum(element_in_cost_df):
            net[cost_df].element.loc[element_in_cost_df] = get_indices(net[cost_df].element[
                element_in_cost_df], lookup)


def create_continuous_elements_index(net, start=0, add_df_to_reindex=set()):
    """
    Creating a continuous index for all the elements, starting at zero and replaces all references
    of old indices by the new ones.

    INPUT:
      **net** - pandapower network with unodered indices

    OPTIONAL:
      **start** - index begins with "start"

      **add_df_to_reindex** - by default all useful pandapower elements for power flow will be
          selected. Customized DataFrames can also be considered here.

    OUTPUT:
      **net** - pandapower network with odered and continuous indices

    """
    elements = pp_elements(res_elements=True)

    # create continuous bus index
    create_continuous_bus_index(net, start=start)
    elements -= {"bus", "bus_geodata", "res_bus"}

    elements |= add_df_to_reindex

    # run reindex_elements() for all elements
    for elm in list(elements):
        net[elm].sort_index(inplace=True)
        new_index = list(np.arange(start, len(net[elm]) + start))

        if elm in net and isinstance(net[elm], pd.DataFrame):
            if elm in ["bus_geodata", "line_geodata"]:
                logger.info(elm + " don't need to bo included to 'add_df_to_reindex'. It is " +
                            "already included by elm=='" + elm.split("_")[0] + "'.")
            else:
                reindex_elements(net, elm, new_index)
        else:
            logger.debug("No indices could be changed for element '%s'." % elm)

    return net


def set_scaling_by_type(net, scalings, scale_load=True, scale_sgen=True):
    """
    Sets scaling of loads and/or sgens according to a dictionary
    mapping type to a scaling factor. Note that the type-string is case
    sensitive.
    E.g. scaling = {"pv": 0.8, "bhkw": 0.6}

    :param net:
    :param scalings: A dictionary containing a mapping from element type to
    :param scale_load:
    :param scale_sgen:
    """
    if not isinstance(scalings, dict):
        raise UserWarning("The parameter scaling has to be a dictionary, "
                          "see docstring")

    def scaleit(what):
        et = net[what]
        et["scaling"] = [scale[t] if scale[t] is not None else s for t, s in
                         zip(et.type.values, et.scaling.values)]

    scale = defaultdict(lambda: None, scalings)
    if scale_load:
        scaleit("load")
    if scale_sgen:
        scaleit("sgen")


def set_data_type_of_columns_to_default(net):
    """
    Overwrites dtype of DataFrame columns of PandapowerNet elements to default dtypes defined in
    pandapower. The function "convert_format" does that authomatically for nets saved with
    pandapower versions below 1.6. If this is required for versions starting with 1.6, it should be
    done manually with this function.

    INPUT:
      **net** - pandapower network with unodered indices

    OUTPUT:
      No output; the net passed as input has pandapower-default dtypes of columns in element tables.

    """
    new_net = create_empty_network()
    for key, item in net.items():
        if isinstance(item, pd.DataFrame):
            for col in item.columns:
                if key in new_net and col in new_net[key].columns:
                    if set(item.columns) == set(new_net[key]):
                        if version.parse(pd.__version__) < version.parse("0.21"):
                            net[key] = net[key].reindex_axis(new_net[key].columns, axis=1)
                        else:
                            net[key] = net[key].reindex(new_net[key].columns, axis=1)
                    if version.parse(pd.__version__) < version.parse("0.20.0"):
                        net[key][col] = net[key][col].astype(new_net[key][col].dtype,
                                                             raise_on_error=False)
                    else:
                        net[key][col] = net[key][col].astype(new_net[key][col].dtype,
                                                             errors="ignore")


# --- Modify topology

def close_switch_at_line_with_two_open_switches(net):
    """
    Finds lines that have opened switches at both ends and closes one of them.
    Function is usually used when optimizing section points to
    prevent the algorithm from ignoring isolated lines.
    """
    closed_switches = set()
    nl = net.switch[(net.switch.et == 'l') & (net.switch.closed == 0)]
    for _, switch in nl.groupby("element"):
        if len(switch.index) > 1:  # find all lines that have open switches at both ends
            # and close on of them
            net.switch.at[switch.index[0], "closed"] = 1
            closed_switches.add(switch.index[0])
    if len(closed_switches) > 0:
        logger.info('closed %d switches at line with 2 open switches (switches: %s)' % (
            len(closed_switches), closed_switches))


def fuse_buses(net, b1, b2, drop=True):
    """
    Reroutes any connections to buses in b2 to the given bus b1. Additionally drops the buses b2,
    if drop=True (default).
    """
    b2 = set(b2) - {b1} if isinstance(b2, Iterable) else [b2]

    for element, value in element_bus_tuples():
        i = net[element][net[element][value].isin(b2)].index
        net[element].loc[i, value] = b1

    i = net["switch"][(net["switch"]["et"] == 'b') & (
        net["switch"]["element"].isin(b2))].index
    net["switch"].loc[i, "element"] = b1
    net["switch"].drop(net["switch"][(net["switch"]["bus"] == net["switch"]["element"]) &
                                     (net["switch"]["et"] == "b")].index, inplace=True)
    bus_meas = net.measurement.loc[net.measurement.element_type == "bus"]
    bus_meas = bus_meas.index[bus_meas.element.isin(b2)]
    net.measurement.loc[bus_meas, "element"] = b1
    if drop:
        # drop_elements=False because the elements must be connected to new buses now
        drop_buses(net, b2, drop_elements=False)
        # if there were measurements at b1 and b2, these can be duplicated at b1 now -> drop
        drop_duplicated_measurements(net, buses=[b1])
    return net


def drop_buses(net, buses, drop_elements=True):
    """
    Drops specified buses, their bus_geodata and by default drops all elements connected to
    them as well.
    """
    net["bus"].drop(buses, inplace=True)
    net["bus_geodata"].drop(set(buses) & set(net["bus_geodata"].index), inplace=True)
    res_buses = net.res_bus.index.intersection(buses)
    net["res_bus"].drop(res_buses, inplace=True)
    if drop_elements:
        drop_elements_at_buses(net, buses)


def drop_switches_at_buses(net, buses):
    i = net["switch"][(net["switch"]["bus"].isin(buses)) |
                      ((net["switch"]["element"].isin(buses)) & (net["switch"]["et"] == "b"))].index
    net["switch"].drop(i, inplace=True)
    logger.info("dropped %d switches" % len(i))


def drop_elements_at_buses(net, buses, bus_elements=True, branch_elements=True):
    """
    drop elements connected to given buses
    """
    for element, column in element_bus_tuples(bus_elements, branch_elements, res_elements=False):
        if element == "switch":
            drop_switches_at_buses(net, buses)

        elif any(net[element][column].isin(buses)):
            eid = net[element][net[element][column].isin(buses)].index
            if element == 'line':
                drop_lines(net, eid)
            elif element == 'trafo' or element == 'trafo3w':
                drop_trafos(net, eid, table=element)
            else:
                n_el = net[element].shape[0]
                net[element].drop(eid, inplace=True)
                # res_element
                res_element = "res_" + element
                if res_element in net.keys() and isinstance(net[res_element], pd.DataFrame):
                    res_eid = net[res_element].index.intersection(eid)
                    net[res_element].drop(res_eid, inplace=True)
                if net[element].shape[0] < n_el:
                    logger.info("dropped %d %s elements" % (n_el - net[element].shape[0], element))


def drop_trafos(net, trafos, table="trafo"):
    """
    Deletes all trafos and in the given list of indices and removes
    any switches connected to it.
    """
    if table not in ('trafo', 'trafo3w'):
        raise UserWarning("parameter 'table' must be 'trafo' or 'trafo3w'")
    # drop any switches
    num_switches = 0
    if table == 'trafo':  # remove as soon as the trafo3w switches are implemented
        i = net["switch"].index[(net["switch"]["element"].isin(trafos)) &
                                (net["switch"]["et"] == "t")]
        net["switch"].drop(i, inplace=True)
        num_switches = len(i)

    # drop the trafos
    net[table].drop(trafos, inplace=True)
    res_trafos = net["res_" + table].index.intersection(trafos)
    net["res_" + table].drop(res_trafos, inplace=True)
    logger.info("dropped %d %s elements with %d switches" % (len(trafos), table, num_switches))


def drop_lines(net, lines):
    """
    Deletes all lines and their geodata in the given list of indices and removes
    any switches connected to it.
    """
    # drop connected switches
    i = net["switch"][(net["switch"]["element"].isin(lines)) & (net["switch"]["et"] == "l")].index
    net["switch"].drop(i, inplace=True)

    # drop lines and geodata
    net["line"].drop(lines, inplace=True)
    net["line_geodata"].drop(set(lines) & set(net["line_geodata"].index), inplace=True)
    res_lines = net.res_line.index.intersection(lines)
    net["res_line"].drop(res_lines, inplace=True)
    logger.info("dropped %d lines with %d line switches" % (len(lines), len(i)))


def drop_duplicated_measurements(net, buses=None, keep="first"):
    """
    Drops duplicated measurements at given set buses. If buses is None, all buses are considered.
    """
    buses = buses if buses is not None else net.bus.index
    # only analyze measurements at given buses
    bus_meas = net.measurement.loc[net.measurement.element_type == "bus"]
    analyzed_meas = bus_meas.loc[net.measurement.element.isin(buses).fillna("nan")]
    # drop duplicates
    if not analyzed_meas.duplicated(subset=[
       "measurement_type", "element_type", "side", "element"], keep=keep).empty:
        idx_to_drop = analyzed_meas.index[analyzed_meas.duplicated(subset=[
            "measurement_type", "element_type", "side", "element"], keep=keep)]
        net.measurement.drop(idx_to_drop, inplace=True)


def set_element_status(net, buses, in_service):
    """
    Sets buses and all elements connected to them in or out of service.
    """
    net.bus.loc[buses, "in_service"] = in_service

    for element in net.keys():
        if element not in ['bus'] and isinstance(net[element], pd.DataFrame) \
                and "in_service" in net[element].columns:
            try:
                idx = get_connected_elements(net, element, buses)
                net[element].loc[idx, 'in_service'] = in_service
            except:
                pass


def set_isolated_areas_out_of_service(net, respect_switches=True):
    """
    Set all isolated buses and all elements connected to isolated buses out of service.
    """
    from pandapower.topology import unsupplied_buses
    closed_switches = set()
    unsupplied = unsupplied_buses(net, respect_switches=respect_switches)
    logger.info("set %d of %d unsupplied buses out of service" % (
        len(net.bus.loc[unsupplied].query('~in_service')), len(unsupplied)))
    set_element_status(net, unsupplied, False)

    # TODO: remove this loop after unsupplied_buses are fixed
    for tr3w in net.trafo3w.index.values:
        tr3w_buses = net.trafo3w.loc[tr3w, ['hv_bus', 'mv_bus', 'lv_bus']].values
        if not all(net.bus.loc[tr3w_buses, 'in_service'].values):
            net.trafo3w.loc[tr3w, 'in_service'] = False

    for element in ["line", "trafo"]:
        oos_elements = net.line[~net.line.in_service].index
        oos_switches = net.switch[(net.switch.et == element[0]) &
                                  (net.switch.element.isin(oos_elements))].index

        closed_switches.update([i for i in oos_switches.values if not net.switch.at[i, 'closed']])
        net.switch.loc[oos_switches, "closed"] = True

        for idx, bus in net.switch[
          ~net.switch.closed & (net.switch.et == element[0])][["element", "bus"]].values:
            if not net.bus.in_service.at[next_bus(net, bus, idx, element)]:
                net[element].at[idx, "in_service"] = False
    if len(closed_switches) > 0:
        logger.info('closed %d switches: %s' % (len(closed_switches), closed_switches))


def drop_out_of_service_elements(net):
    # removes inactive lines and its switches and geodata
    inactive_lines = net.line[~net.line.in_service].index
    drop_lines(net, inactive_lines)
    inactive_res_lines = net.res_line.index.intersection(inactive_lines)
    net.res_line.drop(inactive_res_lines, inplace=True)

    inactive_trafos = net.trafo[~net.trafo.in_service].index
    drop_trafos(net, inactive_trafos, table='trafo')
    inactive_res_trafos = net.res_trafo.index.intersection(inactive_trafos)
    net.res_trafo.drop(inactive_res_trafos, inplace=True)

    inactive_trafos3w = net.trafo3w[~net.trafo3w.in_service].index
    drop_trafos(net, inactive_trafos3w, table='trafo3w')
    inactive_res_trafos3w = net.res_trafo3w.index.intersection(inactive_trafos3w)
    net.res_trafo3w.drop(inactive_res_trafos3w, inplace=True)

    do_not_delete = set(net.line.from_bus.values) | set(net.line.to_bus.values) | \
        set(net.trafo.hv_bus.values) | set(net.trafo.lv_bus.values) | \
        set(net.trafo3w.hv_bus.values) | set(net.trafo3w.mv_bus.values) | \
        set(net.trafo3w.lv_bus.values)

    # removes inactive buses safely
    inactive_buses = set(net.bus[~net.bus.in_service].index) - do_not_delete
    drop_buses(net, inactive_buses, drop_elements=True)
    inactive_res_buses = net.res_bus.index.intersection(inactive_buses)
    net.res_bus.drop(inactive_res_buses, inplace=True)

    # removes inactive elements other than buses, trafos and lines
    for element in net.keys():
        if element not in ["bus", "trafo", "trafo3w", "line", "_equiv_trafo3w"] \
                and isinstance(net[element], pd.DataFrame) \
                and "in_service" in net[element].columns:
            drop_idx = net[element].query("not in_service").index
            net[element].drop(drop_idx, inplace=True)

            # res_element
            res_element = "res_" + element
            if res_element in net.keys() and isinstance(net[res_element], pd.DataFrame):
                drop_res_idx = net[res_element].index.intersection(drop_idx)
                net[res_element].drop(drop_res_idx, inplace=True)

            if len(drop_idx) > 0:
                logger.debug("dropped %d %s elements!" % (len(drop_idx), element))


def drop_inactive_elements(net, respect_switches=True):
    """
    Drops any elements not in service AND any elements connected to inactive
    buses.
    """
    set_isolated_areas_out_of_service(net, respect_switches=respect_switches)
    drop_out_of_service_elements(net)


def _select_cost_df(net, p2, cost_type):
    isin = np.array([False]*net[cost_type].shape[0])
    for et in net[cost_type].et.unique():
        isin_et = net[cost_type].element.isin(p2[et].index)
        is_et = net[cost_type].et == et
        isin |= isin_et & is_et
    p2[cost_type] = net[cost_type].loc[isin]


def select_subnet(net, buses, include_switch_buses=False, include_results=False,
                  keep_everything_else=False):
    """
    Selects a subnet by a list of bus indices and returns a net with all elements
    connected to them.
    """
    buses = set(buses)
    if include_switch_buses:
        # we add both buses of a connected line, the one selected is not switch.bus

        # for all line switches
        for _, s in net["switch"].query("et=='l'").iterrows():
            # get from/to-bus of the connected line
            fb = net["line"]["from_bus"].at[s["element"]]
            tb = net["line"]["to_bus"].at[s["element"]]
            # if one bus of the line is selected and its not the switch-bus, add the other bus
            if fb in buses and s["bus"] != fb:
                buses.add(tb)
            if tb in buses and s["bus"] != tb:
                buses.add(fb)

    if keep_everything_else:
        p2 = copy.deepcopy(net)
        if not include_results:
            clear_result_tables(p2)
        # include_results = True  # assumption: the user doesn't want to old results without selection
    else:
        p2 = create_empty_network(add_stdtypes=False)
        p2["std_types"] = copy.deepcopy(net["std_types"])

    net_parameters = ["name", "f_hz"]
    for net_parameter in net_parameters:
        if net_parameter in net.keys():
            p2[net_parameter] = net[net_parameter]

    p2.bus = net.bus.loc[buses]
    for elm in pp_elements(bus=False, bus_elements=True, branch_elements=False,
                           other_elements=False, res_elements=False):
        p2[elm] = net[elm][net[elm].bus.isin(buses)]

    p2.line = net.line[(net.line.from_bus.isin(buses)) & (net.line.to_bus.isin(buses))]
    p2.dcline = net.dcline[(net.dcline.from_bus.isin(buses)) & (net.dcline.to_bus.isin(buses))]
    p2.trafo = net.trafo[(net.trafo.hv_bus.isin(buses)) & (net.trafo.lv_bus.isin(buses))]
    p2.trafo3w = net.trafo3w[(net.trafo3w.hv_bus.isin(buses)) & (net.trafo3w.mv_bus.isin(buses)) &
                             (net.trafo3w.lv_bus.isin(buses))]
    p2.impedance = net.impedance[(net.impedance.from_bus.isin(buses)) &
                                 (net.impedance.to_bus.isin(buses))]
    p2.measurement = net.measurement[((net.measurement.element_type == "bus") &
                                      (net.measurement.element.isin(buses))) |
                                     ((net.measurement.element_type == "line") &
                                      (net.measurement.element.isin(p2.line.index))) |
                                     ((net.measurement.element_type == "trafo") &
                                      (net.measurement.element.isin(p2.trafo.index))) |
                                     ((net.measurement.element_type == "trafo3w") &
                                      (net.measurement.element.isin(p2.trafo3w.index)))]

    _select_cost_df(net, p2, "poly_cost")
    _select_cost_df(net, p2, "pwl_cost")

    if include_results:
        for table in net.keys():
            if net[table] is None or (isinstance(net[table], pd.DataFrame) and not
               net[table].shape[0]):
                continue
            elif table == "res_bus":
                p2[table] = net[table].loc[buses]
            elif table.startswith("res_") and table != "res_cost":
                p2[table] = net[table].loc[p2[table.split("res_")[1]].index]
    if "bus_geodata" in net:
        p2["bus_geodata"] = net["bus_geodata"].loc[net["bus_geodata"].index.isin(buses)]
    if "line_geodata" in net:
        lines = p2.line.index
        p2["line_geodata"] = net["line_geodata"].loc[net["line_geodata"].index.isin(lines)]

    # switches
    si = [i for i, s in net["switch"].iterrows()
          if s["bus"] in buses and
          ((s["et"] == "b" and s["element"] in p2["bus"].index) or
           (s["et"] == "l" and s["element"] in p2["line"].index) or
           (s["et"] == "t" and s["element"] in p2["trafo"].index))]
    p2["switch"] = net["switch"].loc[si]

    return pandapowerNet(p2)


def merge_nets(net1, net2, validate=True, merge_results=True, tol=1e-9, **kwargs):
    """
    Function to concatenate two nets into one data structure. All element tables get new,
    continuous indizes in order to avoid duplicates.
    """
    net = copy.deepcopy(net1)
    net1 = copy.deepcopy(net1)
    net2 = copy.deepcopy(net2)
    create_continuous_bus_index(net2, start=net1.bus.index.max() + 1)
    if validate:
        runpp(net1, **kwargs)
        runpp(net2, **kwargs)

    def adapt_element_idx_references(net, element, element_type, offset=0):
        """
        used for switch and measurement
        """
        # element_type[0] == "l" for "line", etc.:
        et = element_type[0] if element == "switch" else element_type
        et_col = "et" if element == "switch" else "element_type"
        elements = net[element][net[element][et_col] == et]
        new_index = [net[element_type].index.get_loc(ix) + offset for ix in elements.element.values]
        if len(new_index):
            net[element].loc[elements.index, "element"] = new_index

    for element, table in net.items():
        if element.startswith("_") or element == "dtypes" or (element.startswith("res") and (
                validate or not merge_results)):
            continue
        if isinstance(table, pd.DataFrame) and (len(table) > 0 or len(net2[element]) > 0):
            if element in ["switch", "measurement"]:
                adapt_element_idx_references(net2, element, "line", offset=len(net1.line))
                adapt_element_idx_references(net1, element, "line")
                adapt_element_idx_references(net2, element, "trafo", offset=len(net1.trafo))
                adapt_element_idx_references(net1, element, "trafo")
            if element == "line_geodata":
                ni = [net1.line.index.get_loc(ix) for ix in net1["line_geodata"].index]
                net1.line_geodata.set_index(np.array(ni), inplace=True)
                ni = [net2.line.index.get_loc(ix) + len(net1.line)
                      for ix in net2["line_geodata"].index]
                net2.line_geodata.set_index(np.array(ni), inplace=True)
            ignore_index = element not in ("bus", "res_bus", "bus_geodata", "line_geodata")
            dtypes = net1[element].dtypes
            try:
                net[element] = pd.concat([net1[element], net2[element]], ignore_index=ignore_index,
                                         sort=False)
            except:
                # pandas legacy < 0.21
                net[element] = pd.concat([net1[element], net2[element]], ignore_index=ignore_index)
            _preserve_dtypes(net[element], dtypes)
    # update standard types of net by data of net2
    for type_ in net.std_types.keys():
        # net2.std_types[type_].update(net1.std_types[type_])  # if net1.std_types have priority
        net.std_types[type_].update(net2.std_types[type_])
    if validate:
        runpp(net, **kwargs)
        dev1 = max(abs(net.res_bus.loc[net1.bus.index].vm_pu.values - net1.res_bus.vm_pu.values))
        dev2 = max(abs(net.res_bus.iloc[len(net1.bus.index):].vm_pu.values -
                       net2.res_bus.vm_pu.values))
        if dev1 > tol or dev2 > tol:
            raise UserWarning("Deviation in bus voltages after merging: %.10f" % max(dev1, dev2))
    return net


def create_replacement_switch_for_branch(net, element, idx):
    """
    Creates a switch parallel to a branch, connecting the same buses as the branch.
    The switch is closed if the branch is in service and open if the branch is out of service.
    The in_service status of the original branch is not affected and should be set separately,
    if needed.

    :param net: pandapower network
    :param element: element table e. g. 'line', 'impedance'
    :param idx: index of the branch e. g. 0
    :return: None
    """
    bus_i = net[element].from_bus.at[idx]
    bus_j = net[element].to_bus.at[idx]
    in_service = net[element].in_service.at[idx]
    if element in ['line', 'trafo']:
        is_closed = all(
            net.switch.loc[(net.switch.element == idx) & (net.switch.et == element[0]), 'closed'])
        is_closed = is_closed and in_service
    else:
        is_closed = in_service

    switch_name = 'REPLACEMENT_%s_%d' % (element, idx)
    sid = create_switch(net, name=switch_name, bus=bus_i, element=bus_j, et='b', closed=is_closed,
                        type='CB')
    logger.debug('created switch %s (%d) as replacement for %s %s' %
                 (switch_name, sid, element, idx))
    return sid


def replace_zero_branches_with_switches(net, elements=('line', 'impedance'), zero_length=True,
                                        zero_impedance=True, in_service_only=True, min_length_km=0,
                                        min_r_ohm_per_km=0, min_x_ohm_per_km=0, min_c_nf_per_km=0,
                                        min_rft_pu=0, min_xft_pu=0, min_rtf_pu=0, min_xtf_pu=0,
                                        drop_affected=False):
    """
    Creates a replacement switch for branches with zero impedance (line, impedance) and sets them
    out of service.

    :param net: pandapower network
    :param elements: a tuple of names of element tables e. g. ('line', 'impedance') or (line)
    :param zero_length: whether zero length lines will be affected
    :param zero_impedance: whether zero impedance branches will be affected
    :param in_service_only: whether the branches that are not in service will be affected
    :param drop_affected: wheter the affected branch elements are dropped
    :param min_length_km: threshhold for line length for a line to be considered zero line
    :param min_r_ohm_per_km: threshhold for line R' value for a line to be considered zero line
    :param min_x_ohm_per_km: threshhold for line X' value for a line to be considered zero line
    :param min_c_nf_per_km: threshhold for line C' for a line to be considered zero line
    :param min_rft_pu: threshhold for R from-to value for impedance to be considered zero impedance
    :param min_xft_pu: threshhold for X from-to value for impedance to be considered zero impedance
    :param min_rtf_pu: threshhold for R to-from value for impedance to be considered zero impedance
    :param min_xtf_pu: threshhold for X to-from value for impedance to be considered zero impedance
    :return:
    """

    if not isinstance(elements, tuple):
        raise TypeError(
            'input parameter "elements" must be a tuple, e.g. ("line", "impedance") or ("line")')

    replaced = dict()
    for elm in elements:
        branch_zero = set()
        if elm == 'line' and zero_length:
            branch_zero.update(net[elm].loc[net[elm].length_km <= min_length_km].index.tolist())

        if elm == 'line' and zero_impedance:
            branch_zero.update(net[elm].loc[(net[elm].r_ohm_per_km <= min_r_ohm_per_km) &
                                            (net[elm].x_ohm_per_km <= min_x_ohm_per_km) &
                                            (net[elm].c_nf_per_km <= min_c_nf_per_km)
                                            ].index.tolist())

        if elm == 'impedance' and zero_impedance:
            branch_zero.update(net[elm].loc[(net[elm].rft_pu <= min_rft_pu) &
                                            (net[elm].xft_pu <= min_xft_pu) &
                                            (net[elm].rtf_pu <= min_rtf_pu) &
                                            (net[elm].xtf_pu <= min_xtf_pu)].index.tolist())

        affected_elements = set()
        for b in branch_zero:
            if in_service_only and ~net[elm].in_service.at[b]:
                continue
            create_replacement_switch_for_branch(net, element=elm, idx=b)
            net[elm].loc[b, 'in_service'] = False
            affected_elements.add(b)

        replaced[elm] = net[elm].loc[affected_elements]

        if drop_affected:
            if elm == 'line':
                drop_lines(net, affected_elements)
            else:
                net[elm].drop(affected_elements, inplace=True)

            logger.info('replaced %d %ss by switches' % (len(affected_elements), elm))
        else:
            logger.info('set %d %ss out of service' % (len(affected_elements), elm))

    return replaced


def replace_impedance_by_line(net, index=None, only_valid_replace=True, sn_as_max=False):
    """
    Creates lines by given impedances data, while the impedances are dropped.
    INPUT:
        **net** - pandapower net

    OPTIONAL:
        **index** (index, None) - Index of all impedances to be replaced. If None, all impedances
            will be replaced.

        **only_valid_replace** (bool, True) - If True, impedances will only replaced, if a
            replacement leads to equal power flow results. If False, unsymmetric impedances will
            be replaced by symmetric lines.

        **sn_as_max** (bool, False) - Flag to set whether sn_kva of impedances should be assumed
            for max_i_ka of lines.
    """
    index = index or net.impedance.index
    new_index = []
    for _, imp in net.impedance.loc[index].iterrows():
        if imp.rft_pu != imp.rtf_pu or imp.xft_pu != imp.xtf_pu:
            if only_valid_replace:
                continue
            logger.error("impedance differs in from or to bus direction. lines always " +
                         "parameters always pertain in both direction. only from_bus to " +
                         "to_bus parameters are considered.")
        vn = net.bus.vn_kv.at[imp.from_bus]
        Zni = vn ** 2 / imp.sn_mva
        max_i_ka = imp.sn_kva / vn / np.sqrt(3) if sn_as_max else np.nan
        new_index.append(create_line_from_parameters(
            net, imp.from_bus, imp.to_bus, 1, imp.rft_pu * Zni, imp.xft_pu * Zni, 0, max_i_ka,
            name=imp.name, in_service=imp.in_service))
    net.impedance.drop(index, inplace=True)
    return new_index


def replace_line_by_impedance(net, index=None, sn_mva=None, only_valid_replace=True):
    """
    Creates impedances by given lines data, while the lines are dropped.
    INPUT:
        **net** - pandapower net

    OPTIONAL:
        **index** (index, None) - Index of all lines to be replaced. If None, all lines
            will be replaced.

        **sn_kva** (list or array, None) - Values of sn_kva for creating the impedances. If None,
            the net.sn_kva is assumed

        **only_valid_replace** (bool, True) - If True, lines will only replaced, if a replacement
            leads to equal power flow results. If False, capacitance and dielectric conductance will
            be neglected.
    """
    index = index or net.line.index
    sn_mva = sn_mva or net.sn_mva
    sn_mva = sn_mva if sn_mva != "max_i_ka" else net.line.max_i_ka.loc[index]
    sn_mva = sn_mva if hasattr(sn_mva, "__iter__") else [sn_mva] * len(index)
    if len(sn_mva) != len(index):
        raise ValueError("index and sn_mva must have the same length.")
    i = 0
    new_index = []
    for idx, line_ in net.line.loc[index].iterrows():
        if line_.c_nf_per_km or line_.g_us_per_km:
            if only_valid_replace:
                continue
            logger.error("Capacitance and dielectric conductance of line %i cannot be " % idx +
                         "converted to impedances, which do not model such parameters.")
        vn = net.bus.vn_kv.at[line_.from_bus]
        Zni = vn ** 2 / sn_mva[i]
        new_index.append(create_impedance(
            net, line_.from_bus, line_.to_bus, line_.r_ohm_per_km * line_.length_km / Zni,
            line_.x_ohm_per_km * line_.length_km / Zni, sn_mva[i], name=line_.name,
            in_service=line_.in_service))
        i += 1
    drop_lines(net, index)
    return new_index


def replace_ext_grid_by_gen(net, ext_grids=None, gen_indices=None, slack=False, cols_to_keep=None,
                            add_cols_to_keep=None):
    """
    Replaces external grids by generators.

    INPUT:
        **net** - pandapower net

    OPTIONAL:
        **ext_grids** (iterable) - indices of external grids which should be replaced

        **gen_indices** (iterable) - required indices of new generators

        **slack** (bool, False) - indicates which value is set to net.gen.slack for the new
            generators

        **cols_to_keep** (list, None) - list of column names which should be kept while replacing
            ext_grids. If None these columns are kept if values exist: "max_p_mw", "min_p_mw",
            "max_q_mvar", "min_q_mvar". However cols_to_keep is given, these columns are alway set:
            "bus", "vm_pu", "p_mw", "name", "in_service", "controllable"

        **add_cols_to_keep** (list, None) - list of column names which should be added to
            'cols_to_keep' to be kept while replacing ext_grids.
    """
    # --- determine ext_grid index
    if ext_grids is None:
        ext_grids = net.ext_grid.index
    else:
        ext_grids = ensure_iterability(ext_grids)
    if gen_indices is None:
        gen_indices = [None]*len(ext_grids)
    elif len(gen_indices) != len(ext_grids):
        raise ValueError("The length of 'gen_indices' must be the same as 'ext_grids' but is " +
                         "%i instead of %i" % (len(gen_indices), len(ext_grids)))

    # --- determine which columns should be kept while replacing
    cols_to_keep = cols_to_keep if cols_to_keep is not None else [
        "max_p_mw", "min_p_mw", "max_q_mvar", "min_q_mvar"]
    if isinstance(add_cols_to_keep, list) and len(add_cols_to_keep):
        cols_to_keep += add_cols_to_keep
    elif add_cols_to_keep is not None:
        raise ValueError("'add_cols_to_keep' must be a list or None but is a %s" % str(type(
            add_cols_to_keep)))
    cols_to_keep = list(set(cols_to_keep) - {"bus", "vm_pu", "p_mw", "name", "in_service",
                                             "controllable"})

    existing_cols_to_keep = net.ext_grid.loc[ext_grids].dropna(axis=1).columns.intersection(
        cols_to_keep)
    # add missing columns to net.gen which should be kept
    missing_cols_to_keep = existing_cols_to_keep.difference(net.gen.columns)
    for col in missing_cols_to_keep:
        net.gen[col] = np.nan

    # --- create gens
    new_idx = []
    for ext_grid, index in zip(net.ext_grid.loc[ext_grids].itertuples(), gen_indices):
        p_mw = 0 if ext_grid.Index not in net.res_ext_grid.index else net.res_ext_grid.at[
            ext_grid.Index, "p_mw"]
        idx = create_gen(net, ext_grid.bus, vm_pu=ext_grid.vm_pu, p_mw=p_mw, name=ext_grid.name,
                         in_service=ext_grid.in_service, controllable=True, index=index)
        new_idx.append(idx)
    net.gen.slack.loc[new_idx] = slack
    net.gen.loc[new_idx, existing_cols_to_keep] = net.ext_grid.loc[
        ext_grids, existing_cols_to_keep].values

    # --- drop replaced ext_grids
    net.ext_grid.drop(ext_grids, inplace=True)

    # --- adapt cost data
    for table in ["pwl_cost", "poly_cost"]:
        if net[table].shape[0]:
            to_change = net[table].index[(net[table].et == "ext_grid") &
                                         (net[table].element.isin(ext_grids))]
            if len(to_change):
                net[table].et.loc[to_change] = "gen"
                net[table].element.loc[to_change] = new_idx

    # --- result data
    if net.res_ext_grid.shape[0]:
        to_add = net.res_ext_grid.loc[ext_grids]
        to_add.index = new_idx
        if version.parse(pd.__version__) < version.parse("0.23"):
            net.res_gen = pd.concat([net.res_gen, to_add])
        else:
            net.res_gen = pd.concat([net.res_gen, to_add], sort=True)
        net.res_ext_grid.drop(ext_grids, inplace=True)
    return new_idx


def replace_gen_by_ext_grid(net, gens=None, ext_grid_indices=None, cols_to_keep=None,
                            add_cols_to_keep=None):
    """
    Replaces generators by external grids.

    INPUT:
        **net** - pandapower net

    OPTIONAL:
        **gens** (iterable) - indices of generators which should be replaced

        **ext_grid_indices** (iterable) - required indices of new external grids

        **cols_to_keep** (list, None) - list of column names which should be kept while replacing
            gens. If None these columns are kept if values exist: "max_p_mw", "min_p_mw",
            "max_q_mvar", "min_q_mvar". However cols_to_keep is given, these columns are alway set:
            "bus", "vm_pu", "va_degree", "name", "in_service"

        **add_cols_to_keep** (list, None) - list of column names which should be added to
            'cols_to_keep' to be kept while replacing gens.
    """
    # --- determine gen index
    if gens is None:
        gens = net.gen.index
    else:
        gens = ensure_iterability(gens)
    if ext_grid_indices is None:
        ext_grid_indices = [None]*len(gens)
    elif len(ext_grid_indices) != len(gens):
        raise ValueError("The length of 'ext_grid_indices' must be the same as 'gens' but is " +
                         "%i instead of %i" % (len(ext_grid_indices), len(gens)))

    # --- determine which columns should be kept while replacing
    cols_to_keep = cols_to_keep if cols_to_keep is not None else [
        "max_p_mw", "min_p_mw", "max_q_mvar", "min_q_mvar"]
    if isinstance(add_cols_to_keep, list) and len(add_cols_to_keep):
        cols_to_keep += add_cols_to_keep
    elif add_cols_to_keep is not None:
        raise ValueError("'add_cols_to_keep' must be a list or None but is a %s" % str(type(
            add_cols_to_keep)))
    cols_to_keep = list(set(cols_to_keep) - {"bus", "vm_pu", "va_degree", "name", "in_service"})

    existing_cols_to_keep = net.gen.loc[gens].dropna(axis=1).columns.intersection(
        cols_to_keep)
    # add missing columns to net.ext_grid
    missing_cols_to_keep = existing_cols_to_keep.difference(net.ext_grid.columns)
    for col in missing_cols_to_keep:
        net.ext_grid[col] = np.nan

    # --- create ext_grids
    new_idx = []
    for gen, index in zip(net.gen.loc[gens].itertuples(), ext_grid_indices):
        va_degree = 0. if gen.bus not in net.res_bus.index else net.res_bus.va_degree.at[gen.bus]
        idx = create_ext_grid(net, gen.bus, vm_pu=gen.vm_pu, va_degree=va_degree, name=gen.name,
                              in_service=gen.in_service, index=index)
        new_idx.append(idx)
    net.ext_grid.loc[new_idx, existing_cols_to_keep] = net.gen.loc[
        gens, existing_cols_to_keep].values

    # --- drop replaced gens
    net.gen.drop(gens, inplace=True)

    # --- adapt cost data
    for table in ["pwl_cost", "poly_cost"]:
        if net[table].shape[0]:
            to_change = net[table].index[(net[table].et == "gen") & (net[table].element.isin(gens))]
            if len(to_change):
                net[table].et.loc[to_change] = "ext_grid"
                net[table].element.loc[to_change] = new_idx

    # --- result data
    if net.res_gen.shape[0]:
        to_add = net.res_gen.loc[gens]
        to_add.index = new_idx
        if version.parse(pd.__version__) < version.parse("0.23"):
            net.res_ext_grid = pd.concat([net.res_ext_grid, to_add])
        else:
            net.res_ext_grid = pd.concat([net.res_ext_grid, to_add], sort=True)
        net.res_gen.drop(gens, inplace=True)
    return new_idx


def replace_gen_by_sgen(net, gens=None, sgen_indices=None, cols_to_keep=None,
                        add_cols_to_keep=None):
    """
    Replaces generators by static generators.

    INPUT:
        **net** - pandapower net

    OPTIONAL:
        **gens** (iterable) - indices of generators which should be replaced

        **sgen_indices** (iterable) - required indices of new static generators

        **cols_to_keep** (list, None) - list of column names which should be kept while replacing
            gens. If None these columns are kept if values exist: "max_p_mw", "min_p_mw",
            "max_q_mvar", "min_q_mvar". However cols_to_keep is given, these columns are alway set:
            "bus", "p_mw", "q_mvar", "name", "in_service", "controllable"

        **add_cols_to_keep** (list, None) - list of column names which should be added to
            'cols_to_keep' to be kept while replacing gens.
    """
    # --- determine gen index
    if gens is None:
        gens = net.gen.index
    else:
        gens = ensure_iterability(gens)
    if sgen_indices is None:
        sgen_indices = [None]*len(gens)
    elif len(sgen_indices) != len(gens):
        raise ValueError("The length of 'sgen_indices' must be the same as 'gens' but is " +
                         "%i instead of %i" % (len(sgen_indices), len(gens)))

    # --- determine which columns should be kept while replacing
    cols_to_keep = cols_to_keep if cols_to_keep is not None else [
        "max_p_mw", "min_p_mw", "max_q_mvar", "min_q_mvar"]
    if isinstance(add_cols_to_keep, list) and len(add_cols_to_keep):
        cols_to_keep += add_cols_to_keep
    elif add_cols_to_keep is not None:
        raise ValueError("'add_cols_to_keep' must be a list or None but is a %s" % str(type(
            add_cols_to_keep)))
    cols_to_keep = list(set(cols_to_keep) - {"bus", "p_mw", "q_mvar", "name", "in_service",
                                             "controllable"})

    existing_cols_to_keep = net.gen.loc[gens].dropna(axis=1).columns.intersection(
        cols_to_keep)
    # add missing columns to net.gen which should be kept
    missing_cols_to_keep = existing_cols_to_keep.difference(net.sgen.columns)
    for col in missing_cols_to_keep:
        net.sgen[col] = np.nan

    # --- create sgens
    new_idx = []
    for gen, index in zip(net.gen.loc[gens].itertuples(), sgen_indices):
        q_mvar = 0. if gen.Index not in net.res_gen.index else net.res_gen.at[gen.Index, "q_mvar"]
        controllable = True if "controllable" not in net.gen.columns else gen.controllable
        idx = create_sgen(net, gen.bus, p_mw=gen.p_mw, q_mvar=q_mvar, name=gen.name,
                          in_service=gen.in_service, controllable=controllable, index=index)
        new_idx.append(idx)
    net.sgen.loc[new_idx, existing_cols_to_keep] = net.gen.loc[
        gens, existing_cols_to_keep].values

    # --- drop replaced gens
    net.gen.drop(gens, inplace=True)

    # --- adapt cost data
    for table in ["pwl_cost", "poly_cost"]:
        if net[table].shape[0]:
            to_change = net[table].index[(net[table].et == "gen") & (net[table].element.isin(gens))]
            if len(to_change):
                net[table].et.loc[to_change] = "sgen"
                net[table].element.loc[to_change] = new_idx

    # --- result data
    if net.res_gen.shape[0]:
        to_add = net.res_gen.loc[gens]
        to_add.index = new_idx
        if version.parse(pd.__version__) < version.parse("0.23"):
            net.res_sgen = pd.concat([net.res_sgen, to_add])
        else:
            net.res_sgen = pd.concat([net.res_sgen, to_add], sort=True)
        net.res_gen.drop(gens, inplace=True)
    return new_idx


def replace_sgen_by_gen(net, sgens=None, gen_indices=None, cols_to_keep=None,
                        add_cols_to_keep=None):
    """
    Replaces static generators by generators.

    INPUT:
        **net** - pandapower net

    OPTIONAL:
        **sgens** (iterable) - indices of static generators which should be replaced

        **gen_indices** (iterable) - required indices of new generators

        **cols_to_keep** (list, None) - list of column names which should be kept while replacing
            sgens. If None these columns are kept if values exist: "max_p_mw", "min_p_mw",
            "max_q_mvar", "min_q_mvar". However cols_to_keep is given, these columns are alway set:
            "bus", "vm_pu", "p_mw", "name", "in_service", "controllable"

        **add_cols_to_keep** (list, None) - list of column names which should be added to
            'cols_to_keep' to be kept while replacing sgens.
    """
    # --- determine sgen index
    if sgens is None:
        sgens = net.sgen.index
    else:
        sgens = ensure_iterability(sgens)
    if gen_indices is None:
        gen_indices = [None]*len(sgens)
    elif len(gen_indices) != len(sgens):
        raise ValueError("The length of 'gen_indices' must be the same as 'sgens' but is " +
                         "%i instead of %i" % (len(gen_indices), len(sgens)))

    # --- determine which columns should be kept while replacing
    cols_to_keep = cols_to_keep if cols_to_keep is not None else [
        "max_p_mw", "min_p_mw", "max_q_mvar", "min_q_mvar"]
    if isinstance(add_cols_to_keep, list) and len(add_cols_to_keep):
        cols_to_keep += add_cols_to_keep
    elif add_cols_to_keep is not None:
        raise ValueError("'add_cols_to_keep' must be a list or None but is a %s" % str(type(
            add_cols_to_keep)))
    cols_to_keep = list(set(cols_to_keep) - {"bus", "vm_pu", "p_mw", "name", "in_service",
                                             "controllable"})

    existing_cols_to_keep = net.sgen.loc[sgens].dropna(axis=1).columns.intersection(
        cols_to_keep)
    # add missing columns to net.gen which should be kept
    missing_cols_to_keep = existing_cols_to_keep.difference(net.gen.columns)
    for col in missing_cols_to_keep:
        net.gen[col] = np.nan

    # --- create gens
    new_idx = []
    log_warning = False
    for sgen, index in zip(net.sgen.loc[sgens].itertuples(), gen_indices):
        if sgen.bus in net.res_bus.index:
            vm_pu = net.res_bus.at[sgen.bus, "vm_pu"]
        else:  # no result information to get vm_pu -> use net.gen.vm_pu or net.ext_grid.vm_pu or
            # set 1.0
            if sgen.bus in net.gen.bus.values:
                vm_pu = net.gen.vm_pu.loc[net.gen.bus == sgen.bus].values[0]
            elif sgen.bus in net.ext_grid.bus.values:
                vm_pu = net.ext_grid.vm_pu.loc[net.ext_grid.bus == sgen.bus].values[0]
            else:
                vm_pu = 1.0
                log_warning = True
        controllable = False if "controllable" not in net.sgen.columns else sgen.controllable
        idx = create_gen(net, sgen.bus, vm_pu=vm_pu, p_mw=sgen.p_mw, name=sgen.name,
                         in_service=sgen.in_service, controllable=controllable, index=index)
        new_idx.append(idx)
    net.gen.loc[new_idx, existing_cols_to_keep] = net.sgen.loc[
        sgens, existing_cols_to_keep].values

    if log_warning:
        logger.warning("In replace_sgen_by_gen(), for some generator 'vm_pu' is assumed as 1.0 " +
                       "since no power flow results were available.")

    # --- drop replaced sgens
    net.sgen.drop(sgens, inplace=True)

    # --- adapt cost data
    for table in ["pwl_cost", "poly_cost"]:
        if net[table].shape[0]:
            to_change = net[table].index[(net[table].et == "sgen") &
                                         (net[table].element.isin(sgens))]
            if len(to_change):
                net[table].et.loc[to_change] = "gen"
                net[table].element.loc[to_change] = new_idx

    # --- result data
    if net.res_sgen.shape[0]:
        to_add = net.res_sgen.loc[sgens]
        to_add.index = new_idx
        if version.parse(pd.__version__) < version.parse("0.23"):
            net.res_gen = pd.concat([net.res_gen, to_add])
        else:
            net.res_gen = pd.concat([net.res_gen, to_add], sort=True)
        net.res_sgen.drop(sgens, inplace=True)
    return new_idx


def replace_ward_by_internal_elements(net, wards=None):
    """
    Replaces wards by loads and shunts
    INPUT:
        **net** - pandapower net

    OPTIONAL:
        **wards** (iterable) - indices of xwards which should be replaced

    OUTPUT:
        No output - the given wards in pandapower net are replaced by loads and shunts

    """
    # --- determine wards index
    if wards is None:
        wards = net.ward.index
    else:
        wards = ensure_iterability(wards)

    # --- create loads and shunts
    new_load_idx = []
    new_shunt_idx = []
    for ward in net.ward.loc[wards].itertuples():
        load_idx = create_load(net, ward.bus, ward.ps_mw, ward.qs_mvar,
                               in_service=ward.in_service, name=ward.name)
        shunt_idx = create_shunt(net, ward.bus, q_mvar=ward.qz_mvar, p_mw=ward.pz_mw,
                                 in_service=ward.in_service, name=ward.name)
        new_load_idx.append(load_idx)
        new_shunt_idx.append(shunt_idx)

    # --- result data
    if net.res_ward.shape[0]:
        sign_in_service = np.multiply(net.ward.in_service.values[wards], 1)
        sign_not_isolated = np.multiply(net.res_ward.vm_pu.values[wards] != 0, 1)
        to_add_load = net.res_ward.loc[wards, ["p_mw", "q_mvar"]]
        to_add_load.index = new_load_idx
        to_add_load.p_mw = net.ward.ps_mw[wards].values*sign_in_service*sign_not_isolated
        to_add_load.q_mvar = net.ward.qs_mvar[wards].values*sign_in_service*sign_not_isolated
        net.res_load = pd.concat([net.res_load, to_add_load])

        to_add_shunt = net.res_ward.loc[wards, ["p_mw", "q_mvar", "vm_pu"]]
        to_add_shunt.index = new_shunt_idx
        to_add_shunt.p_mw = net.res_ward.vm_pu[wards].values**2*net.ward.pz_mw[wards].values * \
            sign_in_service*sign_not_isolated
        to_add_shunt.q_mvar = net.res_ward.vm_pu[wards].values**2*net.ward.qz_mvar[wards].values * \
            sign_in_service*sign_not_isolated
        to_add_shunt.vm_pu = net.res_ward.vm_pu[wards].values
        net.res_shunt = pd.concat([net.res_shunt, to_add_shunt])

        net.res_ward.drop(wards, inplace=True)

    # --- drop replaced wards
    net.ward.drop(wards, inplace=True)


def replace_xward_by_internal_elements(net, xwards=None):
    """
    Replaces xward by loads, shunts, impedance and generators
    INPUT:
        **net** - pandapower net

    OPTIONAL:
        **xwards** (iterable) - indices of xwards which should be replaced

    OUTPUT:
        No output - the given xwards in pandapower are replaced by buses, loads, shunts, impadance
        and generators

    """
    # --- determine xwards index
    if xwards is None:
        xwards = net.xward.index
    else:
        xwards = ensure_iterability(xwards)

    # --- create buses, loads, shunts, gens and impedances
    for xward in net.xward.loc[xwards].itertuples():
        bus_v = net.bus.vn_kv[xward.bus]
        bus_idx = create_bus(net, net.bus.vn_kv[xward.bus], in_service=xward.in_service,
                             name=xward.name)
        create_load(net, xward.bus, xward.ps_mw, xward.qs_mvar,
                    in_service=xward.in_service, name=xward.name)
        create_shunt(net, xward.bus, q_mvar=xward.qz_mvar, p_mw=xward.pz_mw,
                     in_service=xward.in_service, name=xward.name)
        create_gen(net, bus_idx, 0, xward.vm_pu, in_service=xward.in_service,
                   name=xward.name)
        create_impedance(net, xward.bus, bus_idx, xward.r_ohm/(bus_v**2), xward.x_ohm/(bus_v**2),
                         net.sn_mva, in_service=xward.in_service, name=xward.name)

    # --- result data
    if net.res_xward.shape[0]:
        logger.debug("The implementation to move xward results to new internal elements is " +
                     "missing.")
        net.res_xward.drop(xwards, inplace=True)

    # --- drop replaced wards
    net.xward.drop(xwards, inplace=True)


# --- item/element selections

def get_element_index(net, element, name, exact_match=True):
    """
    Returns the element(s) identified by a name or regex and its element-table.

    INPUT:
      **net** - pandapower network

      **element** - Table to get indices from ("line", "bus", "trafo" etc.)

      **name** - Name of the element to match.

    OPTIONAL:
      **exact_match** (boolean, True) - True: Expects exactly one match, raises
                                                UserWarning otherwise.
                                        False: returns all indices containing the name

    OUTPUT:
      **index** - The indices of matching element(s).
    """
    if exact_match:
        idx = net[element][net[element]["name"] == name].index
        if len(idx) == 0:
            raise UserWarning("There is no %s with name %s" % (element, name))
        if len(idx) > 1:
            raise UserWarning("Duplicate %s names for %s" % (element, name))
        return idx[0]
    else:
        return net[element][net[element]["name"].str.contains(name)].index


def get_element_indices(net, element, name, exact_match=True):
    """
    Returns a list of element(s) identified by a name or regex and its element-table -> Wrapper
    function of get_element_index()

    INPUT:
      **net** - pandapower network

      **element** (str or iterable of strings) - Element table to get indices from ("line", "bus",
            "trafo" etc.)

      **name** (str or iterable of strings) - Name of the element to match.

    OPTIONAL:
      **exact_match** (boolean, True) - True: Expects exactly one match, raises
                                                UserWarning otherwise.
                                        False: returns all indices containing the name

    OUTPUT:
      **index** (list) - List of the indices of matching element(s).

    EXAMPLE:
        import pandapower.networks as pn
        import pandapower as pp
        net = pn.example_multivoltage()
        idx1 = pp.get_element_indices(net, "bus", ["Bus HV%i" % i for i in range(1, 4)])
        idx2 = pp.get_element_indices(net, ["bus", "line"], "HV", exact_match=False)
        idx3 = pp.get_element_indices(net, ["bus", "line"], ["Bus HV3", "MV Line6"])
    """
    if isinstance(element, str) and isinstance(name, str):
        element = [element]
        name = [name]
    else:
        element = element if not isinstance(element, str) else [element] * len(name)
        name = name if not isinstance(name, str) else [name] * len(element)
    if len(element) != len(name):
        raise ValueError("'element' and 'name' must have the same length.")
    idx = []
    for elm, nam in zip(element, name):
        idx += [get_element_index(net, elm, nam, exact_match=exact_match)]
    return idx


def next_bus(net, bus, element_id, et='line', **kwargs):
    """
    Returns the index of the second bus an element is connected to, given a
    first one. E.g. the from_bus given the to_bus of a line.
    """
    if et == 'line':
        bc = ["from_bus", "to_bus"]
    elif et == 'trafo':
        bc = ["hv_bus", "lv_bus"]
    elif et == "switch" and list(net[et].loc[element_id, ["et"]].values) == [
            'b']:  # Raises error if switch is not a bus-bus switch
        bc = ["bus", "element"]
    else:
        raise Exception("unknown element type")
    nb = list(net[et].loc[element_id, bc].values)
    nb.remove(bus)
    return nb[0]


def get_connected_elements(net, element, buses, respect_switches=True, respect_in_service=False):
    """
     Returns elements connected to a given bus.

     INPUT:
        **net** (pandapowerNet)

        **element** (string, name of the element table)

        **buses** (single integer or iterable of ints)

     OPTIONAL:
        **respect_switches** (boolean, True)    - True: open switches will be respected
                                                  False: open switches will be ignored
        **respect_in_service** (boolean, False) - True: in_service status of connected lines will be
                                                        respected
                                                  False: in_service status will be ignored
     OUTPUT:
        **connected_elements** (set) - Returns connected elements.

    """

    if not hasattr(buses, "__iter__"):
        buses = [buses]

    if element in ["line", "l"]:
        element = "l"
        element_table = net.line
        connected_elements = set(net.line.index[net.line.from_bus.isin(buses) |
                                                net.line.to_bus.isin(buses)])

    elif element in ["dcline"]:
        element_table = net.dcline
        connected_elements = set(net.dcline.index[net.dcline.from_bus.isin(buses) |
                                                  net.dcline.to_bus.isin(buses)])

    elif element in ["trafo"]:
        element = "t"
        element_table = net.trafo
        connected_elements = set(net["trafo"].index[(net.trafo.hv_bus.isin(buses)) |
                                                    (net.trafo.lv_bus.isin(buses))])
    elif element in ["trafo3w", "t3w"]:
        element = "t3w"
        element_table = net.trafo3w
        connected_elements = set(net["trafo3w"].index[(net.trafo3w.hv_bus.isin(buses)) |
                                                      (net.trafo3w.mv_bus.isin(buses)) |
                                                      (net.trafo3w.lv_bus.isin(buses))])
    elif element == "impedance":
        element_table = net.impedance
        connected_elements = set(net["impedance"].index[(net.impedance.from_bus.isin(buses)) |
                                                        (net.impedance.to_bus.isin(buses))])
    elif element in ["gen", "ext_grid", "xward", "shunt", "ward", "sgen", "load", "storage"]:
        element_table = net[element]
        connected_elements = set(element_table.index[(element_table.bus.isin(buses))])
    elif element == "measurement":
        connected_elements = set(net.measurement.index[(net.measurement.element.isin(buses)) |
                                                       (net.measurement.element_type == "bus")])
    elif element in ['_equiv_trafo3w']:
        # ignore '_equiv_trafo3w'
        return {}
    else:
        raise UserWarning("Unknown element! ", element)

    if respect_switches and element in ["l", "t", "t3w"]:
        open_switches = get_connected_switches(net, buses, consider=element, status="open")
        if open_switches:
            open_and_connected = net.switch.loc[net.switch.index.isin(open_switches) &
                                                net.switch.element.isin(connected_elements)].index
            connected_elements -= set(net.switch.element[open_and_connected])

    if respect_in_service:
        connected_elements -= set(element_table[~element_table.in_service].index)

    return connected_elements


def get_connected_buses(net, buses, consider=("l", "s", "t", "t3", "i"), respect_switches=True,
                        respect_in_service=False):
    """
     Returns buses connected to given buses. The source buses will NOT be returned.

     INPUT:
        **net** (pandapowerNet)

        **buses** (single integer or iterable of ints)

     OPTIONAL:
        **respect_switches** (boolean, True)        - True: open switches will be respected
                                                      False: open switches will be ignored
        **respect_in_service** (boolean, False)     - True: in_service status of connected buses
                                                            will be respected
                                                            False: in_service status will be
                                                            ignored
        **consider** (iterable, ("l", "s", "t"))    - Determines, which types of connections will
                                                      be will be considered.
                                                      l: lines
                                                      s: switches
                                                      t: trafos
                                                      t3: trafo3ws
                                                      i: impedances
     OUTPUT:
        **cl** (set) - Returns connected buses.

    """
    if not hasattr(buses, "__iter__"):
        buses = [buses]

    cb = set()
    if "l" in consider:
        in_service_constr = net.line.in_service if respect_in_service else True
        opened_lines = set(net.switch.loc[(~net.switch.closed) & (net.switch.et == "l")
                                          ].element.unique()) if respect_switches else {}
        connected_fb_lines = set(net.line.index[
                                     (net.line.from_bus.isin(buses)) & ~net.line.index.isin(
                                         opened_lines) &
                                     (in_service_constr)])
        connected_tb_lines = set(net.line.index[
                                     (net.line.to_bus.isin(buses)) & ~net.line.index.isin(
                                         opened_lines) &
                                     (in_service_constr)])
        cb |= set(net.line[net.line.index.isin(connected_tb_lines)].from_bus)
        cb |= set(net.line[net.line.index.isin(connected_fb_lines)].to_bus)

    if "s" in consider:
        cs = get_connected_switches(net, buses, consider='b',
                                    status="closed" if respect_switches else "all")
        cb |= set(net.switch[net.switch.index.isin(cs)].element)
        cb |= set(net.switch[net.switch.index.isin(cs)].bus)

    if "t" in consider:
        in_service_constr = net.trafo.in_service if respect_in_service else True
        opened_trafos = set(net.switch.loc[(~net.switch.closed) & (net.switch.et == "t")
                                           ].element.unique()) if respect_switches else {}
        connected_hvb_trafos = set(net.trafo.index[
                                       (net.trafo.hv_bus.isin(buses)) & ~net.trafo.index.isin(
                                           opened_trafos) &
                                       (in_service_constr)])
        connected_lvb_trafos = set(net.trafo.index[
                                       (net.trafo.lv_bus.isin(buses)) & ~net.trafo.index.isin(
                                           opened_trafos) &
                                       (in_service_constr)])
        cb |= set(net.trafo.loc[connected_lvb_trafos].hv_bus.values)
        cb |= set(net.trafo.loc[connected_hvb_trafos].lv_bus.values)

    # Gives the lv mv and hv buses of a 3 winding transformer
    if "t3" in consider:
        ct3 = get_connected_elements(net, "trafo3w", buses, respect_switches, respect_in_service)
        cb |= set(net.trafo3w.loc[ct3].hv_bus.values)
        cb |= set(net.trafo3w.loc[ct3].mv_bus.values)
        cb |= set(net.trafo3w.loc[ct3].lv_bus.values)

    if "i" in consider:
        in_service_constr = net.impedance.in_service if respect_in_service else True
        connected_fb_impedances = set(net.impedance.index[
                                     (net.impedance.from_bus.isin(buses)) & (in_service_constr)])
        connected_tb_impedances = set(net.impedance.index[
                                     (net.impedance.to_bus.isin(buses)) & (in_service_constr)])
        cb |= set(net.impedance[net.impedance.index.isin(connected_tb_impedances)].from_bus)
        cb |= set(net.impedance[net.impedance.index.isin(connected_fb_impedances)].to_bus)

    if respect_in_service:
        cb -= set(net.bus[~net.bus.in_service].index)

    return cb - set(buses)


def get_connected_buses_at_element(net, element, et, respect_in_service=False):
    """
     Returns buses connected to a given line, switch or trafo. In case of a bus switch, two buses
     will be returned, else one.

     INPUT:
        **net** (pandapowerNet)

        **element** (integer)

        **et** (string)                             - Type of the source element:
                                                      l: line
                                                      s: switch
                                                      t: trafo

     OPTIONAL:
        **respect_in_service** (boolean, False)     - True: in_service status of connected buses
                                                            will be respected
                                                      False: in_service status will be ignored
     OUTPUT:
        **cl** (set) - Returns connected switches.

    """

    cb = set()
    if et == 'l':
        cb.add(net.line.from_bus.at[element])
        cb.add(net.line.to_bus.at[element])

    elif et == 's':
        cb.add(net.switch.bus.at[element])
        if net.switch.et.at[element] == 'b':
            cb.add(net.switch.element.at[element])
    elif et == 't':
        cb.add(net.trafo.hv_bus.at[element])
        cb.add(net.trafo.lv_bus.at[element])

    if respect_in_service:
        cb -= set(net.bus[~net.bus.in_service].index)

    return cb


def get_connected_switches(net, buses, consider=('b', 'l', 't'), status="all"):
    """
    Returns switches connected to given buses.

    INPUT:
        **net** (pandapowerNet)

        **buses** (single integer or iterable of ints)

    OPTIONAL:
        **consider** (iterable, ("l", "s", "t"))    - Determines, which types of connections
                                                      will be considered.
                                                      l: lines
                                                      b: bus-bus-switches
                                                      t: trafos

        **status** (string, ("all", "closed", "open"))    - Determines, which switches will
                                                            be considered
    OUTPUT:
       **cl** (set) - Returns connected switches.
    """

    if not hasattr(buses, "__iter__"):
        buses = [buses]

    if status == "closed":
        switch_selection = net.switch.closed
    elif status == "open":
        switch_selection = ~net.switch.closed
    elif status == "all":
        switch_selection = np.full(len(net.switch), True, dtype=bool)
    else:
        logger.warning("Unknown switch status \"%s\" selected! "
                       "Selecting all switches by default." % status)

    cs = set()
    if 'b' in consider:
        cs |= set(net['switch'].index[
                      (net['switch']['bus'].isin(buses) | net['switch']['element'].isin(buses)) &
                      (net['switch']['et'] == 'b') & switch_selection])
    if 'l' in consider:
        cs |= set(net['switch'].index[(net['switch']['bus'].isin(buses)) & (
                net['switch']['et'] == 'l') & switch_selection])

    if 't' in consider:
        cs |= set(net['switch'].index[net['switch']['bus'].isin(buses) & (
                net['switch']['et'] == 't') & switch_selection])

    return cs


def get_connected_elements_dict(
        net, buses, respect_switches=True, respect_in_service=False, include_empty_lists=False,
        connected_buses=True, connected_bus_elements=True, connected_branch_elements=True,
        connected_other_elements=True):
    """Returns a dict of lists of connected elements."""
    pp_elms = pp_elements(
        bus=connected_buses, bus_elements=connected_bus_elements,
        branch_elements=connected_branch_elements, other_elements=connected_other_elements,
        res_elements=False)
    connected = dict()
    for elm in pp_elms:
        if elm == "bus":
            conn = get_connected_buses(net, buses, respect_switches=respect_switches,
                                       respect_in_service=respect_in_service)
        elif elm == "switch":
            conn = get_connected_switches(net, buses)
        else:
            conn = get_connected_elements(
                net, elm, buses, respect_switches=respect_switches,
                respect_in_service=respect_in_service)
        if include_empty_lists or len(conn):
            connected[elm] = list(conn)
    return connected


def get_gc_objects_dict():
    """
    This function is based on the code in mem_top module
    Summarize object types that are tracket by the garbage collector in the moment.
    Useful to test if there are memoly leaks.
    :return: dictionary with keys corresponding to types and values to the number of objects of the type
    """
    objs = gc.get_objects()
    nums_by_types = dict()

    for obj in objs:
        _type = type(obj)
        nums_by_types[_type] = nums_by_types.get(_type, 0) + 1
    return nums_by_types