# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import copy
import gc
from collections import defaultdict
from collections.abc import Iterable
from itertools import chain
import warnings

import networkx as nx
import numpy as np
import pandas as pd
import pandas.testing as pdt
import numbers
from pandapower.auxiliary import get_indices, pandapowerNet, _preserve_dtypes, ensure_iterability
from pandapower.create import create_switch, create_line_from_parameters, \
    create_impedance, create_empty_network, create_gen, create_ext_grid, \
    create_load, create_shunt, create_bus, create_sgen, create_storage
from pandapower.opf.validate_opf_input import _check_necessary_opf_parameters
from pandapower.run import runpp
from pandapower.std_types import change_std_type

try:
    from networkx.utils.misc import graphs_equal
    GRAPHS_EQUAL_POSSIBLE = True
except ImportError:
    GRAPHS_EQUAL_POSSIBLE = False

try:
    import pandaplan.core.pplog as logging
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
                     ("storage", "bus"), ("asymmetric_load", "bus"), ("asymmetric_sgen", "bus"),
                     ("motor", "bus")])
    if branch_elements:
        ebts.update([("line", "from_bus"), ("line", "to_bus"), ("impedance", "from_bus"),
                     ("switch", "bus"), ("impedance", "to_bus"), ("trafo", "hv_bus"),
                     ("trafo", "lv_bus"), ("trafo3w", "hv_bus"), ("trafo3w", "mv_bus"),
                     ("trafo3w", "lv_bus"), ("dcline", "from_bus"), ("dcline", "to_bus")])
    if res_elements:
        elements_without_res = ["switch", "measurement", "asymmetric_load", "asymmetric_sgen"]
        ebts.update(
            [("res_" + ebt[0], ebt[1]) for ebt in ebts if ebt[0] not in elements_without_res])
    return ebts


def pp_elements(bus=True, bus_elements=True, branch_elements=True, other_elements=True,
                cost_tables=False, res_elements=False):
    """
    Returns a set of pandapower elements.
    """
    pp_elms = set()
    if bus:
        pp_elms |= {"bus"}
        if res_elements:
            pp_elms |= {"res_bus"}
    pp_elms |= set([el[0] for el in element_bus_tuples(
        bus_elements=bus_elements, branch_elements=branch_elements, res_elements=res_elements)])
    if other_elements:
        pp_elms |= {"measurement"}
    if cost_tables:
        pp_elms |= {"poly_cost", "pwl_cost"}
    return pp_elms


def branch_element_bus_dict(include_switch=False, sort=False):
    """
    Returns a dict with keys of branch elements and values of bus column names as list.
    """
    ebts = element_bus_tuples(bus_elements=False, branch_elements=True, res_elements=False)
    bebd = dict()
    for elm, bus in ebts:
        if elm in bebd.keys():
            bebd[elm].append(bus)
        else:
            bebd[elm] = [bus]
    if not include_switch:
        del bebd["switch"]
    if sort:
        bebd = {elm: sorted(buses) for elm, buses in bebd.items()}
    return bebd


def signing_system_value(elm):
    """
    Returns a 1 for all bus elements using the consumver viewpoint and a -1 for all bus elements
    using the generator viewpoint.
    """
    generator_viewpoint_elms = ["ext_grid", "gen", "sgen"]
    if elm in generator_viewpoint_elms:
        return -1
    elif elm in pp_elements(bus=False, branch_elements=False, other_elements=False):
        return 1
    else:
        raise ValueError("This function is defined for bus elements, not for '%s'." % str(elm))


def pq_from_cosphi(s, cosphi, qmode, pmode):
    """
    Calculates P/Q values from rated apparent power and cosine(phi) values.

       - s: rated apparent power
       - cosphi: cosine phi of the
       - qmode: "underexcited" (Q absorption, decreases voltage) or "overexcited" (Q injection, increases voltage)
       - pmode: "load" for load or "gen" for generation

    As all other pandapower functions this function is based on the consumer viewpoint. For active
    power, that means that loads are positive and generation is negative. For reactive power,
    underexcited behavior (Q absorption, decreases voltage) is modeled with positive values,
    overexcited behavior (Q injection, increases voltage) with negative values.
    """
    if hasattr(s, "__iter__"):
        len_ = len(s)
    elif hasattr(cosphi, "__iter__"):
        len_ = len(cosphi)
    elif not isinstance(qmode, str) and hasattr(qmode, "__iter__"):
        len_ = len(qmode)
    elif not isinstance(pmode, str) and hasattr(pmode, "__iter__"):
        len_ = len(pmode)
    else:
        return _pq_from_cosphi(s, cosphi, qmode, pmode)
    return _pq_from_cosphi_bulk(s, cosphi, qmode, pmode, len_=len_)


def _pq_from_cosphi(s, cosphi, qmode, pmode):
    if qmode in ("ind", "cap"):
        logger.warning('capacitive or inductive behavior will be replaced by more clear terms ' +
                       '"underexcited" (Q absorption, decreases voltage) and "overexcited" ' +
                       '(Q injection, increases voltage). Please use "underexcited" ' +
                       'in place of "ind" and "overexcited" in place of "cap".')
    if qmode == "ind" or qmode == "underexcited":
        qsign = 1
    elif qmode == "cap" or qmode == "overexcited":
        qsign = -1
    else:
        raise ValueError('Unknown mode %s - specify "underexcited" (Q absorption, decreases voltage'
                         ') or "overexcited" (Q injection, increases voltage)' % qmode)

    if pmode == "load":
        psign = 1
    elif pmode == "gen":
        psign = -1
    else:
        raise ValueError('Unknown mode %s - specify "load" or "gen"' % pmode)

    p = s * cosphi
    q = psign * qsign * np.sqrt(s ** 2 - p ** 2)
    return p, q


def _pq_from_cosphi_bulk(s, cosphi, qmode, pmode, len_=None):
    if len_ is None:
        s = np.array(ensure_iterability(s))
        len_ = len(s)
    else:
        s = np.array(ensure_iterability(s, len_))
    cosphi = np.array(ensure_iterability(cosphi, len_))
    qmode = np.array(ensure_iterability(qmode, len_))
    pmode = np.array(ensure_iterability(pmode, len_))

    # "ind" -> "underexcited", "cap" -> "overexcited"
    is_ind = qmode == "ind"
    is_cap = qmode == "cap"
    if any(is_ind) or any(is_cap):
        logger.warning('capacitive or inductive behavior will be replaced by more clear terms ' +
                       '"underexcited" (Q absorption, decreases voltage) and "overexcited" ' +
                       '(Q injection, increases voltage). Please use "underexcited" ' +
                       'in place of "ind" and "overexcited" in place of "cap".')
    qmode[is_ind] = "underexcited"
    qmode[is_cap] = "overexcited"

    # qmode consideration
    unknown_qmode = set(qmode) - set(["underexcited", "overexcited"])
    if len(unknown_qmode):
        raise ValueError("Unknown qmodes: " + str(list(unknown_qmode)))
    qsign = np.ones(qmode.shape)
    qsign[qmode == "overexcited"] = -1

    # pmode consideration
    unknown_pmode = set(pmode) - set(["load", "gen"])
    if len(unknown_pmode):
        raise ValueError("Unknown pmodes: " + str(list(unknown_pmode)))
    psign = np.ones(pmode.shape)
    psign[pmode == "gen"] = -1

    # calculate p and q
    p = s * cosphi
    q = psign * qsign * np.sqrt(s ** 2 - p ** 2)

    return p, q


def cosphi_from_pq(p, q):
    """
    Analog to pq_from_cosphi, but the other way around.
    In consumer viewpoint (pandapower): "underexcited" (Q absorption, decreases voltage) and
    "overexcited" (Q injection, increases voltage)
    """
    if hasattr(p, "__iter__"):
        len_ = len(p)
    elif hasattr(q, "__iter__"):
        len_ = len(q)
    else:
        return _cosphi_from_pq(p, q)
    return _cosphi_from_pq_bulk(p, q, len_=len_)


def _cosphi_from_pq(p, q):
    if p == 0:
        cosphi = np.nan
        logger.warning("A cosphi from p=0 is undefined.")
    else:
        cosphi = np.cos(np.arctan(q / p))
    s = (p ** 2 + q ** 2) ** 0.5
    pmode = ["undef", "load", "gen"][int(np.sign(p))]
    qmode = ["underexcited", "underexcited", "overexcited"][int(np.sign(q))]
    return cosphi, s, qmode, pmode


def _cosphi_from_pq_bulk(p, q, len_=None):
    if len_ is None:
        p = np.array(ensure_iterability(p))
        len_ = len(p)
    else:
        p = np.array(ensure_iterability(p, len_))
    q = np.array(ensure_iterability(q, len_))
    p_is_zero = np.array(p == 0)
    cosphi = np.empty(p.shape)
    if sum(p_is_zero):
        cosphi[p_is_zero] = np.nan
        logger.warning("A cosphi from p=0 is undefined.")
    cosphi[~p_is_zero] = np.cos(np.arctan(q[~p_is_zero] / p[~p_is_zero]))
    s = (p ** 2 + q ** 2) ** 0.5
    pmode = np.array(["undef", "load", "gen"])[np.sign(p).astype(int)]
    qmode = np.array(["underexcited", "underexcited", "overexcited"])[np.sign(q).astype(int)]
    return cosphi, s, qmode, pmode


def dataframes_equal(df1, df2, ignore_index_order=True, **kwargs):
    """
    Returns a boolean whether the given two dataframes are equal or not.
    """
    if "tol" in kwargs:
        if "atol" in kwargs:
            raise ValueError("'atol' and 'tol' are given to dataframes_equal(). Don't use 'tol' "
                             "anymore.")
        logger.warning("in dataframes_equal() parameter 'tol' is deprecated. Use 'atol' instead.")
        kwargs["atol"] = kwargs.pop("tol")

    if ignore_index_order:
        df1 = df1.sort_index().sort_index(axis=1)
        df2 = df2.sort_index().sort_index(axis=1)

    # --- pandas implementation
    try:
        pdt.assert_frame_equal(df1, df2, **kwargs)
        return True
    except AssertionError:
        return False

    # --- alternative (old) implementation
    # if df1.shape == df2.shape:
    #     if df1.shape[0]:
    #         # we use numpy.allclose to grant a tolerance on numerical values
    #         numerical_equal = np.allclose(df1.select_dtypes(include=[np.number]),
    #                                       df2.select_dtypes(include=[np.number]),
    #                                       atol=tol, equal_nan=True)
    #     else:
    #         numerical_equal = True
    #     # ... use pandas .equals for the rest, which also evaluates NaNs to be equal
    #     rest_equal = df1.select_dtypes(exclude=[np.number]).equals(
    #         df2.select_dtypes(exclude=[np.number]))

    #     return numerical_equal & rest_equal
    # else:
    #     return False


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


# --- Information
def log_to_level(msg, passed_logger, level):
    if level == "error":
        passed_logger.error(msg)
    elif level == "warning":
        passed_logger.warning(msg)
    elif level == "info":
        passed_logger.info(msg)
    elif level == "debug":
        passed_logger.debug(msg)


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


def nets_equal(net1, net2, check_only_results=False, check_without_results=False, exclude_elms=None,
               name_selection=None, **kwargs):
    """
    Returns a boolean whether the two given pandapower networks are equal.

    pandapower net keys starting with "_" are ignored. Same for the key "et" (elapsed time).

    If the element tables contain JSONSerializableClass objects, they will also be compared:
    attributes are compared but not the addresses of the objects.

    INPUT:
        **net1** (pandapower net)

        **net2** (pandapower net)

    OPTIONAL:
        **check_only_results** (bool, False) - if True, only result tables (starting with "res_")
        are compared

        **check_without_results** (bool, False) - if True, result tables (starting with "res_")
        are ignored for comparison

        **exclude_elms** (list, None) - list of element tables which should be ignored in the
        comparison

        **name_selection** (list, None) - list of element tables which should be compared

        **kwargs** - key word arguments for dataframes_equal()
    """
    if not (isinstance(net1, pandapowerNet) and isinstance(net2, pandapowerNet)):
        logger.warning("At least one net is not of type pandapowerNet.")
        return False
    not_equal, not_checked_keys = _nets_equal_keys(
        net1, net2, check_only_results, check_without_results, exclude_elms, name_selection,
        **kwargs)
    if len(not_checked_keys) > 0:
        logger.warning("These keys were ignored by the comparison of the networks: %s" % (', '.join(
            not_checked_keys)))

    if len(not_equal) > 0:
        logger.warning("Networks do not match in DataFrame(s): %s" % (', '.join(not_equal)))
        return False
    else:
        return True


def _nets_equal_keys(net1, net2, check_only_results, check_without_results, exclude_elms,
                     name_selection, **kwargs):
    """ Returns a lists of keys which are 1) not equal and 2) not checked.
    Used within nets_equal(). """
    if check_without_results and check_only_results:
        raise UserWarning("Please provide only one of the options to check without results or to "
                          "exclude results in comparison.")

    exclude_elms = [] if exclude_elms is None else list(exclude_elms)
    exclude_elms += ["res_" + ex for ex in exclude_elms]
    not_equal = []

    # for two networks make sure both have the same keys
    if name_selection is not None:
        net1_keys = net2_keys = name_selection
    elif check_only_results:
        net1_keys = [key for key in net1.keys() if key.startswith("res_")
                     and key not in exclude_elms]
        net2_keys = [key for key in net2.keys() if key.startswith("res_")
                     and key not in exclude_elms]
    else:
        net1_keys = [key for key in net1.keys() if not (
            key.startswith("_") or key in exclude_elms or key == "et"
            or key.startswith("res_") and check_without_results)]
        net2_keys = [key for key in net2.keys() if not (
            key.startswith("_") or key in exclude_elms or key == "et"
            or key.startswith("res_") and check_without_results)]
    keys_to_check = set(net1_keys) & set(net2_keys)
    key_difference = set(net1_keys) ^ set(net2_keys)
    not_checked_keys = list()

    if len(key_difference) > 0:
        logger.warning(f"Networks entries mismatch at: {key_difference}")
        return key_difference, set()

    # ... and then iter through the keys, checking for equality for each table
    for key in list(keys_to_check):

        if isinstance(net1[key], pd.DataFrame):
            if not isinstance(net2[key], pd.DataFrame) or not dataframes_equal(
                    net1[key], net2[key], **kwargs):
                not_equal.append(key)

        elif isinstance(net1[key], np.ndarray):
            if not isinstance(net2[key], np.ndarray):
                not_equal.append(key)
            else:
                if not np.array_equal(net1[key], net2[key], equal_nan=True):
                    not_equal.append(key)

        elif isinstance(net1[key], int) or isinstance(net1[key], float) or \
                isinstance(net1[key], complex):
            if not np.isclose(net1[key], net2[key]):
                not_equal.append(key)

        elif isinstance(net1[key], nx.Graph):
            if GRAPHS_EQUAL_POSSIBLE:
                if not graphs_equal(net1[key], net2[key]):
                    not_equal.append(key)
            else:
                # Maybe there is a better way, but at least this could be checked
                if net1[key].nodes != net2[key].nodes or net1[key].edges != net2[key].edges:
                    not_equal.append(key)

        else:
            try:
                is_eq = net1[key] == net2[key]
                if not is_eq:
                    not_equal.append(key)
            except:
                not_checked_keys.append(key)
    return not_equal, not_checked_keys


def clear_result_tables(net):
    """
    Clears all ``res_`` DataFrames in net.
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

    # --- reindex buses
    net.bus.index = get_indices(net.bus.index, bus_lookup)
    net.res_bus.index = get_indices(net.res_bus.index, bus_lookup)

    # --- adapt link in bus elements
    for element, value in element_bus_tuples():
        net[element][value] = get_indices(net[element][value], bus_lookup)
    net["bus_geodata"].set_index(get_indices(net["bus_geodata"].index, bus_lookup), inplace=True)

    # --- adapt group link
    if net.group.shape[0]:
        for row in np.arange(net.group.shape[0], dtype=int)[
                (net.group.element_type == "bus").values & net.group.reference_column.isnull().values]:
            net.group.element.iat[row] = list(get_indices(net.group.element.iat[row], bus_lookup))

    # --- adapt measurement link
    bus_meas = net.measurement.element_type == "bus"
    net.measurement.loc[bus_meas, "element"] = get_indices(net.measurement.loc[bus_meas, "element"],
                                                           bus_lookup)
    side_meas = pd.to_numeric(net.measurement.side, errors="coerce").notnull()
    net.measurement.loc[side_meas, "side"] = get_indices(net.measurement.loc[side_meas, "side"],
                                                         bus_lookup)

    # --- adapt switch link
    bb_switches = net.switch[net.switch.et == "b"]
    net.switch.loc[bb_switches.index, "element"] = get_indices(bb_switches.element, bus_lookup)

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


def reindex_elements(net, element, new_indices=None, old_indices=None, lookup=None):

    """
    Changes the index of the DataFrame net[element].

    Parameters
    ----------
    net : pp.pandapowerNet
        net with elements to reindex
    element : str
        name of element type to rename, e.g. "gen" or "load"
    new_indices : typing.Union[list[int], pandas.Index[int]], optional
        new indices to set, by default None
    old_indices : typing.Union[list[int], pandas.Index[int]], optional
        old indices to be replaced. If not given, all indices are
        assumed in case of given new_indices, and all lookup keys are assumed in case of given
        lookup, by default None
    lookup : dict[int,int], optional
        lookup to assign new indices to old indices, by default None

    Notes
    -----
    Either new_indices or lookup must be given.
    old_indices can be given to limit the indices to be replaced. In case of given new_indices,
    both must have the same length.
    If element is "group", be careful to give new_indices without passing old_indices because group
    indices do not need to be unique.

    Examples
    --------
    >>> net = pp.create_empty_network()
    >>> idx0 = pp.create_bus(net, 110)
    >>> idx1 = 4
    >>> idx2 = 7

    Reindex using 'new_indices':
    >>> pp.reindex_elements(net, "bus", [idx1])  # passing old_indices=[idx0] is optional

    Reindex using 'lookup':
    >>> pp.reindex_elements(net, "bus", lookup={idx1: idx2})
    """
    if not net[element].shape[0]:
        return
    if new_indices is None and lookup is None:
        raise ValueError("Either new_indices or lookup must be given.")
    elif new_indices is not None and lookup is not None:
        raise ValueError("Only one can be considered, new_indices or lookup.")
    if new_indices is not None and not len(new_indices) or lookup is not None and not len(
            lookup.keys()):
        return

    if new_indices is not None:
        old_indices = old_indices if old_indices is not None else net[element].index
        assert len(new_indices) == len(old_indices)
        lookup = dict(zip(old_indices, new_indices))
    elif old_indices is None:
        old_indices = net[element].index.intersection(lookup.keys())

    if element == "bus":
        reindex_buses(net, lookup)
        return

    # --- reindex
    new_index = pd.Series(net[element].index, index=net[element].index)
    if element != "group":
        new_index.loc[old_indices] = get_indices(old_indices, lookup)
    else:
        new_index.loc[old_indices] = get_indices(new_index.loc[old_indices].values, lookup)
    net[element].set_index(pd.Index(new_index.values), inplace=True)

    # --- adapt group link
    if net.group.shape[0]:
        for row in np.arange(net.group.shape[0], dtype=int)[
                (net.group.element_type == element).values & net.group.reference_column.isnull().values]:
            net.group.element.iat[row] = list(get_indices(net.group.element.iat[row], lookup))

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

    # --- adapt index in cost dataframes
    for cost_df in ["pwl_cost", "poly_cost"]:
        element_in_cost_df = (net[cost_df].et == element) & net[cost_df].element.isin(old_indices)
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
                    if new_net[key][col].dtype == net[key][col].dtype:
                        continue
                    if set(item.columns) == set(new_net[key]):
                        net[key] = net[key].reindex(new_net[key].columns, axis=1)
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
            net.switch.at[switch.index[0], "closed"] = True
            closed_switches.add(switch.index[0])
    if len(closed_switches) > 0:
        logger.info('closed %d switches at line with 2 open switches (switches: %s)' % (
            len(closed_switches), closed_switches))


def fuse_buses(net, b1, b2, drop=True, fuse_bus_measurements=True):
    """
    Reroutes any connections to buses in b2 to the given bus b1. Additionally drops the buses b2,
    if drop=True (default).
    """
    b2 = set(b2) - {b1} if isinstance(b2, Iterable) else [b2]

    # --- reroute element connections from b2 to b1
    for element, value in element_bus_tuples():
        if net[element].shape[0]:
            net[element][value].loc[net[element][value].isin(b2)] = b1
    net["switch"]["element"].loc[(net["switch"]["et"] == 'b') & (
                                 net["switch"]["element"].isin(b2))] = b1

    # --- reroute bus measurements from b2 to b1
    if fuse_bus_measurements and net.measurement.shape[0]:
        bus_meas = net.measurement.loc[net.measurement.element_type == "bus"]
        bus_meas = bus_meas.index[bus_meas.element.isin(b2)]
        net.measurement.loc[bus_meas, "element"] = b1

    # --- drop b2
    if drop:
        # drop_elements=True is not needed because the elements must be connected to new buses now:
        drop_buses(net, b2, drop_elements=False)
        # branch elements which connected b1 with b2 are now connecting b1 with b1. these branch
        # can now be dropped:
        drop_inner_branches(net, buses=[b1])
        # if there were measurements at b1 and b2, these can be duplicated at b1 now -> drop
        if fuse_bus_measurements and net.measurement.shape[0]:
            drop_duplicated_measurements(net, buses=[b1])


def drop_buses(net, buses, drop_elements=True):
    """
    Drops specified buses, their bus_geodata and by default drops all elements connected to
    them as well.
    """
    drop_from_groups(net, "bus", buses)
    net["bus"].drop(buses, inplace=True)
    net["bus_geodata"].drop(set(buses) & set(net["bus_geodata"].index), inplace=True)
    res_buses = net.res_bus.index.intersection(buses)
    net["res_bus"].drop(res_buses, inplace=True)
    if drop_elements:
        drop_elements_at_buses(net, buses)
        drop_measurements_at_elements(net, "bus", idx=buses)


def drop_switches_at_buses(net, buses):
    i = net["switch"][(net["switch"]["bus"].isin(buses)) |
                      ((net["switch"]["element"].isin(buses)) & (net["switch"]["et"] == "b"))].index
    net["switch"].drop(i, inplace=True)
    logger.info("dropped %d switches" % len(i))


def drop_elements_at_buses(net, buses, bus_elements=True, branch_elements=True,
                           drop_measurements=True):
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
                drop_from_groups(net, element, eid)
                net[element].drop(eid, inplace=True)
                # res_element
                res_element = "res_" + element
                if res_element in net.keys() and isinstance(net[res_element], pd.DataFrame):
                    res_eid = net[res_element].index.intersection(eid)
                    net[res_element].drop(res_eid, inplace=True)
                if net[element].shape[0] < n_el:
                    logger.info("dropped %d %s elements" % (n_el - net[element].shape[0], element))
                # drop costs for the affected elements
                for cost_elm in ["poly_cost", "pwl_cost"]:
                    net[cost_elm].drop(net[cost_elm].index[(net[cost_elm].et == element) &
                                                           (net[cost_elm].element.isin(eid))], inplace=True)
    if drop_measurements:
        drop_measurements_at_elements(net, "bus", idx=buses)


def drop_trafos(net, trafos, table="trafo"):
    """
    Deletes all trafos and in the given list of indices and removes
    any switches connected to it.
    """
    if table not in ('trafo', 'trafo3w'):
        raise UserWarning("parameter 'table' must be 'trafo' or 'trafo3w'")
    # drop any switches
    et = "t" if table == 'trafo' else "t3"
    # remove any affected trafo or trafo3w switches
    i = net["switch"].index[(net["switch"]["element"].isin(trafos)) & (net["switch"]["et"] == et)]
    drop_from_groups(net, "switch", i)
    net["switch"].drop(i, inplace=True)
    num_switches = len(i)

    # drop measurements
    drop_measurements_at_elements(net, table, idx=trafos)

    # drop the trafos
    net[table].drop(trafos, inplace=True)
    drop_from_groups(net, table, trafos)
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
    drop_from_groups(net, "switch", i)
    net["switch"].drop(i, inplace=True)

    # drop measurements
    drop_measurements_at_elements(net, "line", idx=lines)

    # drop lines and geodata
    drop_from_groups(net, "line", lines)
    net["line"].drop(lines, inplace=True)
    net["line_geodata"].drop(set(lines) & set(net["line_geodata"].index), inplace=True)
    res_lines = net.res_line.index.intersection(lines)
    net["res_line"].drop(res_lines, inplace=True)
    logger.info("dropped %d lines with %d line switches" % (len(lines), len(i)))


def drop_measurements_at_elements(net, element_type, idx=None, side=None):
    """
    Drop measurements of given element_type and (if given) given elements (idx) and side.
    """
    idx = ensure_iterability(idx) if idx is not None else net[element_type].index
    bool1 = net.measurement.element_type == element_type
    bool2 = net.measurement.element.isin(idx)
    bool3 = net.measurement.side == side if side is not None else [True]*net.measurement.shape[0]
    to_drop = net.measurement.index[bool1 & bool2 & bool3]
    net.measurement.drop(to_drop, inplace=True)


def drop_duplicated_measurements(net, buses=None, keep="first"):
    """
    Drops duplicated measurements at given set of buses. If buses is None, all buses are considered.
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


def get_connecting_branches(net, buses1, buses2, branch_elements=None):
    """
    Gets/Drops branches that connects any bus of buses1 with any bus of buses2.
    """
    branch_dict = branch_element_bus_dict(include_switch=True)
    if branch_elements is not None:
        branch_dict = {key: branch_dict[key] for key in branch_elements}
    if "switch" in branch_dict:
        branch_dict["switch"].append("element")

    found = {elm: set() for elm in branch_dict.keys()}
    for elm, bus_types in branch_dict.items():
        for bus1 in bus_types:
            for bus2 in bus_types:
                if bus2 != bus1:
                    idx = net[elm].index[net[elm][bus1].isin(buses1) & net[elm][bus2].isin(buses2)]
                    if elm == "switch":
                        idx = idx.intersection(net[elm].index[net[elm].et == "b"])
                    found[elm] |= set(idx)
    return {key: val for key, val in found.items() if len(val)}


def _inner_branches(net, buses, task, branch_elements=None):
    """
    Drops or finds branches that connects buses within 'buses' at all branch sides (e.g. 'from_bus'
    and 'to_bus').
    """
    branch_dict = branch_element_bus_dict(include_switch=True)
    if branch_elements is not None:
        branch_dict = {key: branch_dict[key] for key in branch_elements}

    inner_branches = dict()
    for elm, bus_types in branch_dict.items():
        inner = pd.Series(True, index=net[elm].index)
        for bus_type in bus_types:
            inner &= net[elm][bus_type].isin(buses)
        if elm == "switch":
            inner &= net[elm]["element"].isin(buses)
            inner &= net[elm]["et"] == "b"  # bus-bus-switches

        if any(inner):
            if task == "drop":
                if elm == "line":
                    drop_lines(net, net[elm].index[inner])
                elif "trafo" in elm:
                    drop_trafos(net, net[elm].index[inner])
                else:
                    net[elm].drop(net[elm].index[inner], inplace=True)
            elif task == "get":
                inner_branches[elm] = net[elm].index[inner]
            else:
                raise NotImplementedError("task '%s' is unknown." % str(task))
    return inner_branches


def get_inner_branches(net, buses, branch_elements=None):
    """
    Returns indices of branches that connects buses within 'buses' at all branch sides (e.g.
    'from_bus' and 'to_bus').
    """
    return _inner_branches(net, buses, "get", branch_elements=branch_elements)


def drop_inner_branches(net, buses, branch_elements=None):
    """
    Drops branches that connects buses within 'buses' at all branch sides (e.g. 'from_bus' and
    'to_bus').
    """
    _inner_branches(net, buses, "drop", branch_elements=branch_elements)


def set_element_status(net, buses, in_service):
    """
    Sets buses and all elements connected to them in or out of service.
    """
    net.bus.loc[list(buses), "in_service"] = in_service

    for element in net.keys():
        if element not in ['bus'] and isinstance(net[element], pd.DataFrame) \
                and "in_service" in net[element].columns:
            try:
                idx = get_connected_elements(net, element, buses)
                net[element].loc[list(idx), 'in_service'] = in_service
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
        len(net.bus.loc[list(unsupplied)].query('~in_service')), len(unsupplied)))
    set_element_status(net, list(unsupplied), False)

    # TODO: remove this loop after unsupplied_buses are fixed
    for tr3w in net.trafo3w.index.values:
        tr3w_buses = net.trafo3w.loc[tr3w, ['hv_bus', 'mv_bus', 'lv_bus']].values
        if not all(net.bus.loc[tr3w_buses, 'in_service'].values):
            net.trafo3w.at[tr3w, 'in_service'] = False
        open_tr3w_switches = net.switch.loc[(net.switch.et == 't3') & ~net.switch.closed & (
            net.switch.element == tr3w)]
        if len(open_tr3w_switches) == 3:
            net.trafo3w.at[tr3w, 'in_service'] = False

    for element, et in zip(["line", "trafo"], ["l", "t"]):
        oos_elements = net[element].query("not in_service").index
        oos_switches = net.switch[(net.switch.et == et) & net.switch.element.isin(
            oos_elements)].index

        closed_switches.update([i for i in oos_switches.values if not net.switch.at[i, 'closed']])
        net.switch.loc[oos_switches, "closed"] = True

        for idx, bus in net.switch.loc[~net.switch.closed & (net.switch.et == et)][[
                "element", "bus"]].values:
            if not net.bus.in_service.at[next_bus(net, bus, idx, element)]:
                net[element].at[idx, "in_service"] = False
    if len(closed_switches) > 0:
        logger.info('closed %d switches: %s' % (len(closed_switches), closed_switches))


def drop_elements_simple(net, element, idx):
    """
    Drop elements and result entries from pandapower net.
    """
    idx = ensure_iterability(idx)
    drop_from_groups(net, element, idx)
    net[element].drop(idx, inplace=True)

    # res_element
    res_element = "res_" + element
    if res_element in net.keys() and isinstance(net[res_element], pd.DataFrame):
        drop_res_idx = net[res_element].index.intersection(idx)
        net[res_element].drop(drop_res_idx, inplace=True)

    # logging
    if len(idx) > 0:
        logger.debug("dropped %d %s elements!" % (len(idx), element))


def drop_out_of_service_elements(net):
    """
    Drop all elements (including corresponding dataframes such as switches, measurements,
    result tables, geodata) with "in_service" is False. Buses that are connected to in-service
    branches are not deleted.
    """

    # --- drop inactive branches
    inactive_lines = net.line[~net.line.in_service].index
    drop_lines(net, inactive_lines)

    inactive_trafos = net.trafo[~net.trafo.in_service].index
    drop_trafos(net, inactive_trafos, table='trafo')

    inactive_trafos3w = net.trafo3w[~net.trafo3w.in_service].index
    drop_trafos(net, inactive_trafos3w, table='trafo3w')

    other_branch_elms = pp_elements(bus=False, bus_elements=False, branch_elements=True,
                                    other_elements=False) - {"line", "trafo", "trafo3w", "switch"}
    for elm in other_branch_elms:
        drop_elements_simple(net, elm, net[elm][~net[elm].in_service].index)

    # --- drop inactive buses (safely)
    # do not delete buses connected to branches
    do_not_delete = set()
    for elm, bus_col in element_bus_tuples(bus_elements=False):
        if elm != "switch":
            do_not_delete |= set(net[elm][bus_col].values)

    # remove inactive buses (safely)
    inactive_buses = set(net.bus[~net.bus.in_service].index) - do_not_delete
    drop_buses(net, inactive_buses, drop_elements=True)

    # --- drop inactive elements other than branches and buses
    for elm in pp_elements(bus=False, bus_elements=True, branch_elements=False,
                           other_elements=True):
        if "in_service" not in net[elm].columns:
            if elm not in ["measurement", "switch"]:
                logger.info("Out-of-service elements cannot be dropped since 'in_service' is " +
                            "not in net[%s].columns" % elm)
        else:
            drop_elements_simple(net, elm, net[elm][~net[elm].in_service].index)


def drop_inactive_elements(net, respect_switches=True):
    """
    Drops any elements not in service AND any elements connected to inactive
    buses.
    """
    set_isolated_areas_out_of_service(net, respect_switches=respect_switches)
    drop_out_of_service_elements(net)


def drop_from_group(net, index, element_type, element_index):
    """Drops elements from the group of given index.
    No errors are raised if elements are passed to be drop from groups which alread don't have these
    elements as members.
    A reverse function is available -> pp.group.append_to_group().

    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    index : int
        Index of the group from which the element should be dropped
    element_type : str
        The element type of which elements should be dropped from the group(s), e.g. "bus"
    element_index : int or list of integers
        indices of the elements which should be dropped from the group
    """
    drop_from_groups(net, element_type, element_index, index=index)


def drop_from_groups(net, element_type, element_index, index=None):
    """Drops elements from one or multple groups, defined by 'index'.
    No errors are raised if elements are passed to be drop from groups which alread don't have these
    elements as members.
    A reverse function is available -> pp.group.append_to_group().

    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    element_type : str
        The element type of which elements should be dropped from the group(s), e.g. "bus"
    element_index : int or list of integers
        indices of the elements which should be dropped from the group
    index : int or list of integers, optional
        Indices of the group(s) from which the element should be dropped. If None, the elements are
        dropped from all groups, by default None
    """
    if index is None:
        index = net.group.index
    element_index = pd.Index(ensure_iterability(element_index), dtype=int)

    to_check = np.isin(net.group.index.values, index)
    to_check &= net.group.element_type.values == element_type
    keep = np.ones(net.group.shape[0], dtype=bool)

    for i in np.arange(len(to_check), dtype=int)[to_check]:
        rc = net.group.reference_column.iat[i]
        if rc is None or pd.isnull(rc):
            net.group.element.iat[i] = pd.Index(net.group.element.iat[i]).difference(
                element_index).tolist()
        else:
            net.group.element.iat[i] = pd.Index(net.group.element.iat[i]).difference(pd.Index(
                net[element_type][rc].loc[element_index.intersection(
                    net[element_type].index)])).tolist()

        if not len(net.group.element.iat[i]):
            keep[i] = False
    net.group = net.group.loc[keep]


def drop_group(net, index):
    """Drops the group of given index.

    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    index : int
        index of the group which should be dropped
    """
    net.group.drop(index, inplace=True)


def drop_group_and_elements(net, index):
    """
    Drops all elements of the group and in net.group the group itself.
    """
    # functions like drop_trafos, drop_lines, drop_buses are not considered since all elements
    # should be included in elements_dict
    for et in net.group.loc[[index], "element_type"].tolist():
        idx = group_element_index(net, index, et)
        net[et].drop(idx.intersection(net[et].index), inplace=True)
        res_et = "res_" + et
        if res_et in net.keys() and net[res_et].shape[0]:
            net[res_et].drop(net[res_et].index.intersection(idx), inplace=True)
    net.group.drop(index, inplace=True)


def _select_cost_df(net, p2, cost_type):
    isin = np.array([False] * net[cost_type].shape[0])
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
        buses_to_add = set()
        # for all line switches
        for s in net["switch"].query("et=='l'").itertuples():
            # get from/to-bus of the connected line
            fb = net["line"]["from_bus"].at[s.element]
            tb = net["line"]["to_bus"].at[s.element]
            # if one bus of the line is selected and its not the switch-bus, add the other bus
            if fb in buses and s.bus != fb:
                buses_to_add.add(tb)
            if tb in buses and s.bus != tb:
                buses_to_add.add(fb)
        buses |= buses_to_add

    if keep_everything_else:
        p2 = copy.deepcopy(net)
        if not include_results:
            clear_result_tables(p2)
    else:
        p2 = create_empty_network(add_stdtypes=False)
        p2["std_types"] = copy.deepcopy(net["std_types"])

        net_parameters = ["name", "f_hz"]
        for net_parameter in net_parameters:
            if net_parameter in net.keys():
                p2[net_parameter] = net[net_parameter]

    p2.bus = net.bus.loc[list(buses)]
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
    relevant_characteristics = set()
    for col in ("vk_percent_characteristic", "vkr_percent_characteristic"):
        if col in net.trafo.columns:
            relevant_characteristics |= set(net.trafo[~net.trafo[col].isnull(), col].values)
    for col in (f"vk_hv_percent_characteristic", f"vkr_hv_percent_characteristic",
                f"vk_mv_percent_characteristic", f"vkr_mv_percent_characteristic",
                f"vk_lv_percent_characteristic", f"vkr_lv_percent_characteristic"):
        if col in net.trafo3w.columns:
            relevant_characteristics |= set(net.trafo3w[~net.trafo3w[col].isnull(), col].values)
    p2.characteristic = net.characteristic.loc[list(relevant_characteristics)]

    _select_cost_df(net, p2, "poly_cost")
    _select_cost_df(net, p2, "pwl_cost")

    if include_results:
        for table in net.keys():
            if net[table] is None or not isinstance(net[table], pd.DataFrame) or not \
               net[table].shape[0] or not table.startswith("res_") or table[4:] not in \
               net.keys() or not isinstance(net[table[4:]], pd.DataFrame) or not \
               net[table[4:]].shape[0]:
                continue
            elif table == "res_bus":
                p2[table] = net[table].loc[list(buses.intersection(net[table].index))]
            else:
                p2[table] = net[table].loc[p2[table[4:]].index.intersection(net[table].index)]
    if "bus_geodata" in net:
        p2["bus_geodata"] = net.bus_geodata.loc[p2.bus.index.intersection(
            net.bus_geodata.index)]
    if "line_geodata" in net:
        p2["line_geodata"] = net.line_geodata.loc[p2.line.index.intersection(
            net.line_geodata.index)]

    # switches
    p2["switch"] = net.switch.loc[
        net.switch.bus.isin(p2.bus.index) & pd.concat([
            net.switch[net.switch.et == 'b'].element.isin(p2.bus.index),
            net.switch[net.switch.et == 'l'].element.isin(p2.line.index),
            net.switch[net.switch.et == 't'].element.isin(p2.trafo.index),
        ], sort=False)
        ]

    return pandapowerNet(p2)


def merge_nets(net1, net2, validate=True, merge_results=True, tol=1e-9, **kwargs):
    """Function to concatenate two nets into one data structure. The elements keep their indices
    unless both nets have the same indices. In that case, net2 elements get reindex. The reindex
    lookup of net2 elements can be retrieved by passing return_net2_reindex_lookup=True.

    Parameters
    ----------
    net1 : pp.pandapowerNet
        first net to concatenate
    net2 : pp.pandapowerNet
        second net to concatenate
    validate : bool, optional
        whether power flow results should be compared against the results of the input nets,
        by default True
    merge_results : bool, optional
        whether results tables should be concatenated, by default True
    tol : float, optional
        tolerance which is allowed to pass the results validate check (relevant if validate is
        True), by default 1e-9
    std_prio_on_net1 : bool, optional
        whether net1 standard type should be kept if net2 has types with same names, by default True
    return_net2_reindex_lookup : bool, optional
        if True, the merged net AND a dict of lookups is returned, by default False
    net2_reindex_log_level : str, optional
        logging level of the message which element types of net2 got reindexed elements. Options
        are, for example "debug", "info", "warning", "error", or None, by default "info"

    Returns
    -------
    pp.pandapowerNet
        net with concatenated element tables

    Raises
    ------
    UserWarning
        if validate is True and power flow results of the merged net deviate from input nets results
    """
    new_params = {"std_prio_on_net1", "return_net2_reindex_lookup", "net2_reindex_log_level"}
    msg_future_changes = f"In a future version merge_nets() will keep element indices and " + \
        "prioritize net1 standard types. To silence this warning and to use the future " + \
        f"functionality, explicitely pass at least one of the new parameters {new_params}."

    old_params_passed = not len(set(kwargs.keys()).intersection({
            "retain_original_indices_in_net1", "create_continuous_bus_indices"}))
    new_params_passed = len(set(kwargs.keys()).intersection(new_params))

    if new_params_passed:
        return _merge_nets(net1, net2, validate=validate, merge_results=merge_results, tol=tol,
                           **kwargs)
    else:
        warnings.warn(msg_future_changes, category=FutureWarning)
        return _merge_nets_deprecated(net1, net2, validate=validate, merge_results=merge_results, tol=tol,
                               **kwargs)


def _merge_nets(net1, net2, validate=True, merge_results=True, tol=1e-9,
                std_prio_on_net1=True, return_net2_reindex_lookup=False,
                net2_reindex_log_level="info", **runpp_kwargs):
    """Function to concatenate two nets into one data structure. The elements keep their indices
    unless both nets have the same indices. In that case, net2 elements get reindex. The reindex
    lookup of net2 elements can be retrieved by passing return_net2_reindex_lookup=True.
    """
    net = copy.deepcopy(net1)
    net2 = copy.deepcopy(net2)

    if validate:
        runpp(net, **runpp_kwargs)
        net1_res_bus = copy.deepcopy(net.res_bus)
        runpp(net2, **runpp_kwargs)

    # collect element types to copy from net2 to net (output)
    elm_types = [elm_type for elm_type, df in net2.items() if not elm_type.startswith("_") and \
        isinstance(df, pd.DataFrame) and df.shape[0] and elm_type != "dtypes" and \
            (not elm_type.startswith("res_") or (merge_results and not validate))]

    # reindex net2 elements if some indices already exist in net
    reindex_lookup = dict()
    for elm_type in elm_types:
        is_dupl = pd.Series(net2[elm_type].index).isin(net[elm_type].index)
        if any(is_dupl):
            start = max(net1[elm_type].index.max(), net2[elm_type].index[~is_dupl].max()) + 1
            old_indices = net2[elm_type].index[is_dupl]
            if elm_type == "group":
                old_indices = pd.Series(old_indices).loc[~pd.Series(old_indices).duplicated()].tolist()
            new_indices = range(start, start + len(old_indices))
            reindex_lookup[elm_type] = dict(zip(old_indices, new_indices))
            reindex_elements(net2, elm_type, lookup=reindex_lookup[elm_type])
    if len(reindex_lookup.keys()):
        log_to_level("net2 elements of these types has been reindexed by merge_nets() because " + \
            f"these exist already in net1: {list(reindex_lookup.keys())}", logger,
            net2_reindex_log_level)

    # copy dataframes from net2 to net (output)
    for elm_type in elm_types:
        dtypes = net[elm_type].dtypes
        net[elm_type] = pd.concat([net[elm_type], net2[elm_type]])
        _preserve_dtypes(net[elm_type], dtypes)

    # copy standard types of net by data of net2
    for type_ in net.std_types.keys():
        if std_prio_on_net1:
            net.std_types[type_] = {**net2.std_types[type_], **net.std_types[type_]}
        else:
            net.std_types[type_].update(net2.std_types[type_])

    # validate vm results
    if validate:
        runpp(net, **runpp_kwargs)
        dev1 = max(abs(net.res_bus.loc[net1.bus.index].vm_pu.values - net1_res_bus.vm_pu.values))
        dev2 = max(abs(net.res_bus.iloc[len(net1.bus.index):].vm_pu.values -
                       net2.res_bus.vm_pu.values))
        if dev1 > tol or dev2 > tol:
            raise UserWarning("Deviation in bus voltages after merging: %.10f" % max(dev1, dev2))

    if return_net2_reindex_lookup:
        return net, reindex_lookup
    else:
        return net


def _merge_nets_deprecated(net1, net2, validate=True, merge_results=True, tol=1e-9,
                    create_continuous_bus_indices=True,
                    retain_original_indices_in_net1=False, **kwargs):
    """
    Function to concatenate two nets into one data structure. All element tables get new,
    continuous indizes in order to avoid duplicates.

    Groups are not considered.
    """
    net = copy.deepcopy(net1)
    # net1 = copy.deepcopy(net1)  # commented to save time. net1 will not be changed (only by runpp)
    net2 = copy.deepcopy(net2)
    if create_continuous_bus_indices:
        create_continuous_bus_index(net2, start=net1.bus.index.max() + 1)
    if validate:
        runpp(net1, **kwargs)
        runpp(net2, **kwargs)

    def adapt_element_idx_references(net, element, element_type, offset=0):
        """
        used for switch, measurement, poly_cost and pwl_cost
        """
        # element_type[0] == "l" for "line", etc.:
        et = element_type[0] if element == "switch" else element_type
        et_col = "et" if element in ["switch", "poly_cost", "pwl_cost"] else "element_type"
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
                adapt_element_idx_references(net, element, "line")
                adapt_element_idx_references(net2, element, "trafo", offset=len(net1.trafo))
                adapt_element_idx_references(net, element, "trafo")
            if element in ["poly_cost", "pwl_cost"]:
                net[element]["element"] = [np.nan if row.element not in net1[row.et].index.values \
                                            else net1[row.et].index.get_loc(row.element) for row in
                                            net1[element].itertuples()]
                if net[element]["element"].isnull().any():
                    # this case could also be checked using get_false_links()
                    logger.warning(f"Some net1[{element}] does not link to an existing element."
                                   "These are dropped.")
                    net[element].drop(net[element].index[net[element].element.isnull()],
                                       inplace=True)
                    net[element]["element"] = net[element]["element"].astype(int)
                for et in ["gen", "sgen",  "ext_grid", "load", "dcline", "storage"]:
                    adapt_element_idx_references(net2, element, et, offset=len(net1[et]))
            if element == "line_geodata":
                ni = [net1.line.index.get_loc(ix) for ix in net1["line_geodata"].index]
                net.line_geodata.set_index(np.array(ni), inplace=True)
                ni = [net2.line.index.get_loc(ix) + len(net1.line)
                      for ix in net2["line_geodata"].index]
                net2.line_geodata.set_index(np.array(ni), inplace=True)
            elm_with_critical_index = element in ("bus", "res_bus", "bus_geodata", "line_geodata",
                                                  "group")
            ignore_index = not retain_original_indices_in_net1 and not elm_with_critical_index
            dtypes = net[element].dtypes
            net[element] = pd.concat([net[element], net2[element]], sort=False,
                                     ignore_index=ignore_index)
            if retain_original_indices_in_net1 and not elm_with_critical_index and \
                len(net1[element]):
                start = int(net1[element].index.max()) + 1
                net[element].index = net1[element].index.tolist() + \
                    list(range(start, len(net2[element]) + start))
            _preserve_dtypes(net[element], dtypes)
    # update standard types of net by data of net2
    for type_ in net.std_types.keys():
        net.std_types[type_].update(net2.std_types[type_])  # net2.std_types have priority
    if validate:
        runpp(net, **kwargs)
        dev1 = max(abs(net.res_bus.loc[net1.bus.index].vm_pu.values - net1.res_bus.vm_pu.values))
        dev2 = max(abs(net.res_bus.iloc[len(net1.bus.index):].vm_pu.values -
                       net2.res_bus.vm_pu.values))
        if dev1 > tol or dev2 > tol:
            raise UserWarning("Deviation in bus voltages after merging: %.10f" % max(dev1, dev2))
    return net


def repl_to_line(net, idx, std_type, name=None, in_service=False, **kwargs):
    """
    creates a power line in parallel to the existing power line based on the values of the new
    std_type. The new parallel line has an impedance value, which is chosen so that the resulting
    impedance of the new line and the already existing line is equal to the impedance of the
    replaced line. Or for electrical engineers:

    Z0 = impedance of the existing line
    Z1 = impedance of the replaced line
    Z2 = impedance of the created line

        --- Z2 ---
    ---|         |---   =  --- Z1 ---
       --- Z0 ---


    Parameters
    ----------
    net - pandapower net
    idx (int) - idx of the existing line
    std_type (str) - pandapower standard type
    name (str, None) - name of the new power line
    in_service (bool, False) - if the new power line is in service
    **kwargs - additional line parameters you want to set for the new line

    Returns
    -------
    new_idx (int) - index of the created power line

    """

    # impedance before changing the standard type
    r0 = net.line.at[idx, "r_ohm_per_km"]
    p0 = net.line.at[idx, "parallel"]
    x0 = net.line.at[idx, "x_ohm_per_km"]
    c0 = net.line.at[idx, "c_nf_per_km"]
    g0 = net.line.at[idx, "g_us_per_km"]
    i_ka0 = net.line.at[idx, "max_i_ka"]
    bak = net.line.loc[idx, :].values

    change_std_type(net, idx, std_type)

    # impedance after changing the standard type
    r1 = net.line.at[idx, "r_ohm_per_km"]
    x1 = net.line.at[idx, "x_ohm_per_km"]
    c1 = net.line.at[idx, "c_nf_per_km"]
    g1 = net.line.at[idx, "g_us_per_km"]
    i_ka1 = net.line.at[idx, "max_i_ka"]

    # complex resistance of the line parallel to the existing line
    y1 = 1 / complex(r1, x1)
    y0 = p0 / complex(r0, x0)
    z2 = 1 / (y1 - y0)

    # required parameters
    c_nf_per_km = c1 * 1 - c0 * p0
    r_ohm_per_km = z2.real
    x_ohm_per_km = z2.imag
    g_us_per_km = g1 * 1 - g0 * p0
    max_i_ka = i_ka1 - i_ka0
    name = "repl_" + str(idx) if name is None else name

    # if this line is in service to the existing line, the power flow result should be the same as
    # when replacing the existing line with the desired standard type
    new_idx = create_line_from_parameters(
        net, from_bus=net.line.at[idx, "from_bus"], to_bus=net.line.at[idx, "to_bus"],
        length_km=net.line.at[idx, "length_km"], r_ohm_per_km=r_ohm_per_km,
        x_ohm_per_km=x_ohm_per_km, c_nf_per_km=c_nf_per_km, max_i_ka=max_i_ka,
        g_us_per_km=g_us_per_km, in_service=in_service, name=name, **kwargs)
    # restore the previous line parameters before changing the standard type
    net.line.loc[idx, :] = bak

    # check switching state and add line switch if necessary:
    for bus in net.line.at[idx, "to_bus"], net.line.at[idx, "from_bus"]:
        if bus in net.switch[~net.switch.closed & (net.switch.element == idx) & (net.switch.et == "l")].bus.values:
            create_switch(net, bus=bus, element=new_idx, closed=False, et="l", type="LBS")

    return new_idx


def merge_parallel_line(net, idx):
    """
    Changes the impedances of the parallel line so that it equals a single line.
    Args:
        net: pandapower net
        idx: idx of the line to merge

    Returns:
        net

    Z0 = impedance of the existing parallel lines
    Z1 = impedance of the respective single line

        --- Z0 ---
    ---|         |---   =  --- Z1 ---
       --- Z0 ---

    """
    # impedance before changing the standard type

    r0 = net.line.at[idx, "r_ohm_per_km"]
    p0 = net.line.at[idx, "parallel"]
    x0 = net.line.at[idx, "x_ohm_per_km"]
    c0 = net.line.at[idx, "c_nf_per_km"]
    g0 = net.line.at[idx, "g_us_per_km"]
    i_ka0 = net.line.at[idx, "max_i_ka"]

    # complex resistance of the line to the existing line
    y0 = 1 / complex(r0, x0)
    y1 = p0*y0
    z1 = 1 / y1
    r1 = z1.real
    x1 = z1.imag

    g1 = p0*g0
    c1 = p0*c0
    i_ka1 = p0*i_ka0

    net.line.at[idx, "r_ohm_per_km"] = r1
    net.line.at[idx, "parallel"] = 1
    net.line.at[idx, "x_ohm_per_km"] = x1
    net.line.at[idx, "c_nf_per_km"] = c1
    net.line.at[idx, "g_us_per_km"] = g1
    net.line.at[idx, "max_i_ka"] = i_ka1

    return net


def merge_same_bus_generation_plants(net, add_info=True, error=True,
                                     gen_elms=["ext_grid", "gen", "sgen"]):
    """
    Merge generation plants connected to the same buses so that a maximum of one generation plants
    per node remains.

    ATTENTION:
        * gen_elms should always be given in order of slack (1.), PV (2.) and PQ (3.) elements.

    INPUT:
        **net** - pandapower net

    OPTIONAL:
        **add_info** (bool, True) - If True, the column 'includes_other_plants' is added to the
        elements dataframes. This column informs about which element table rows are the result of a
        merge of generation plants.

        **error** (bool, True) - If True, raises an Error, if vm_pu values differ with same buses.

        **gen_elms** (list, ["ext_grid", "gen", "sgen"]) - list of elements to be merged by same
        buses. Should be in order of slack (1.), PV (2.) and PQ (3.) elements.
    """
    if add_info:
        for elm in gen_elms:
            # adding column 'includes_other_plants' if missing or overwriting if its no bool column
            if "includes_other_plants" not in net[elm].columns or net[elm][
                    "includes_other_plants"].dtype != bool:
                net[elm]["includes_other_plants"] = False

    # --- construct gen_df with all relevant plants data
    limit_cols = ["min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"]
    cols = pd.Index(["bus", "vm_pu", "p_mw", "q_mvar"]+limit_cols)
    cols_dict = {elm: cols.intersection(net[elm].columns) for elm in gen_elms}
    gen_df = pd.concat([net[elm][cols_dict[elm]] for elm in gen_elms])
    gen_df["elm_type"] = np.repeat(gen_elms, [net[elm].shape[0] for elm in gen_elms])
    gen_df.reset_index(inplace=True)

    # --- merge data and drop duplicated rows - directly in the net tables
    something_merged = False
    for bus in gen_df["bus"].loc[gen_df["bus"].duplicated()].unique():
        idxs = gen_df.index[gen_df.bus == bus]
        if "vm_pu" in gen_df.columns and len(gen_df.vm_pu.loc[idxs].dropna().unique()) > 1:
            message = "Generation plants connected to bus %i have different vm_pu." % bus
            if error:
                raise ValueError(message)
            else:
                logger.error(message + " Only the first value is considered.")
        uniq_et = gen_df["elm_type"].at[idxs[0]]
        uniq_idx = gen_df.at[idxs[0], "index"]

        if add_info:  # add includes_other_plants information
            net[uniq_et].at[uniq_idx, "includes_other_plants"] = True

        # sum p_mw
        col = "p_mw" if uniq_et != "ext_grid" else "p_disp_mw"
        net[uniq_et].at[uniq_idx, col] = gen_df.loc[idxs, "p_mw"].sum()

        if "profiles" in net and col == "p_mw":
            elm = "gen" if "gen" in gen_df["elm_type"].loc[idxs[1:]].unique() else "sgen"
            net.profiles["%s.p_mw" % elm].loc[:, uniq_idx] = net.profiles["%s.p_mw" % elm].loc[
                :, gen_df["index"].loc[idxs]].sum(axis=1)
            net.profiles["%s.p_mw" % elm].drop(columns=gen_df["index"].loc[idxs[1:]], inplace=True)
            if elm == "gen":
                net.profiles["%s.vm_pu" % elm].drop(columns=gen_df["index"].loc[idxs[1:]],
                                                    inplace=True)

        # sum q_mvar (if available)
        if "q_mvar" in net[uniq_et].columns:
            net[uniq_et].at[uniq_idx, "q_mvar"] = gen_df.loc[idxs, "q_mvar"].sum()

        # sum limits
        for col in limit_cols:
            if col in gen_df.columns and not gen_df.loc[idxs, col].isnull().all():
                if col not in net[uniq_et].columns:
                    net[uniq_et][col] = np.nan
                net[uniq_et].at[uniq_idx, col] = gen_df.loc[idxs, col].sum()

        # drop duplicated elements
        for elm in gen_df["elm_type"].loc[idxs[1:]].unique():
            dupl_idx_elm = gen_df.loc[gen_df.index.isin(idxs[1:]) &
                                      (gen_df.elm_type == elm), "index"].values
            net[elm].drop(dupl_idx_elm, inplace=True)

        something_merged |= True
    return something_merged


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

        replaced[elm] = net[elm].loc[list(affected_elements)]

        if drop_affected:
            if elm == 'line':
                drop_lines(net, affected_elements)
            else:
                net[elm].drop(affected_elements, inplace=True)

            logger.info('replaced %d %ss by switches' % (len(affected_elements), elm))
        else:
            logger.info('set %d %ss out of service' % (len(affected_elements), elm))

    return replaced


def replace_impedance_by_line(net, index=None, only_valid_replace=True, max_i_ka=np.nan):
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

        **max_i_ka** (value(s), False) - Data/Information how to set max_i_ka. If 'imp.sn_mva' is
        given, the sn_mva values of the impedances are considered.
    """
    index = list(ensure_iterability(index)) if index is not None else list(net.impedance.index)
    max_i_ka = ensure_iterability(max_i_ka, len(index))
    new_index = []
    for (idx, imp), max_i in zip(net.impedance.loc[index].iterrows(), max_i_ka):
        if not np.isclose(imp.rft_pu, imp.rtf_pu) or not np.isclose(imp.xft_pu, imp.xtf_pu):
            if only_valid_replace:
                index.remove(idx)
                continue
            logger.error("impedance differs in from or to bus direction. lines always " +
                         "parameters always pertain in both direction. only from_bus to " +
                         "to_bus parameters are considered.")
        vn = net.bus.vn_kv.at[imp.from_bus]
        Zni = vn ** 2 / imp.sn_mva
        if max_i == 'imp.sn_mva':
            max_i = imp.sn_mva / vn / np.sqrt(3)
        new_index.append(create_line_from_parameters(

            net, imp.from_bus, imp.to_bus,
            length_km=1,
            r_ohm_per_km=imp.rft_pu * Zni,
            x_ohm_per_km=imp.xft_pu * Zni,
            c_nf_per_km=0,
            max_i_ka=max_i,
            r0_ohm_per_km=imp.rft0_pu * Zni if "rft0_pu" in net.impedance.columns else np.nan,
            x0_ohm_per_km=imp.xft0_pu * Zni if "xft0_pu" in net.impedance.columns else np.nan,
            c0_nf_per_km=0,
            parallel=1,
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
    index = list(ensure_iterability(index)) if index is not None else list(net.line.index)
    sn_mva = sn_mva or net.sn_mva
    sn_mva = sn_mva if sn_mva != "max_i_ka" else net.line.max_i_ka.loc[index]
    sn_mva = sn_mva if hasattr(sn_mva, "__iter__") else [sn_mva] * len(index)
    if len(sn_mva) != len(index):
        raise ValueError("index and sn_mva must have the same length.")

    parallel = net.line["parallel"].values

    i = 0
    new_index = []
    for idx, line_ in net.line.loc[index].iterrows():
        if line_.c_nf_per_km or line_.g_us_per_km:
            if only_valid_replace:
                index.remove(idx)
                continue
            logger.error(f"Capacitance and dielectric conductance of line {idx} cannot be "
                         "converted to impedances, which do not model such parameters.")
        vn = net.bus.vn_kv.at[line_.from_bus]
        Zni = vn ** 2 / sn_mva[i]
        par = parallel[idx]
        new_index.append(create_impedance(
            net, line_.from_bus, line_.to_bus,
            rft_pu=line_.r_ohm_per_km * line_.length_km / par / Zni,
            xft_pu=line_.x_ohm_per_km * line_.length_km / par / Zni,
            sn_mva=sn_mva[i],
            rft0_pu=line_.r0_ohm_per_km * line_.length_km / par / Zni if "r0_ohm_per_km" in net.line.columns else None,
            xft0_pu=line_.x0_ohm_per_km * line_.length_km / par / Zni if "x0_ohm_per_km" in net.line.columns else None,
            name=line_.name,
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
        "max_q_mvar", "min_q_mvar". However cols_to_keep is given, these columns are always set:
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
        gen_indices = [None] * len(ext_grids)
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
        in_res = pd.Series(ext_grids).isin(net["res_ext_grid"].index).values
        to_add = net.res_ext_grid.loc[pd.Index(ext_grids)[in_res]]
        to_add.index = pd.Index(new_idx)[in_res]
        net.res_gen = pd.concat([net.res_gen, to_add], sort=True)
        net.res_ext_grid.drop(pd.Index(ext_grids)[in_res], inplace=True)
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
        ext_grid_indices = [None] * len(gens)
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
        in_res = pd.Series(gens).isin(net["res_gen"].index).values
        to_add = net.res_gen.loc[pd.Index(gens)[in_res]]
        to_add.index = pd.Index(new_idx)[in_res]
        net.res_ext_grid = pd.concat([net.res_ext_grid, to_add], sort=True)
        net.res_gen.drop(pd.Index(gens)[in_res], inplace=True)
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
        "max_q_mvar", "min_q_mvar". However cols_to_keep is given, these columns are always set:
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
        sgen_indices = [None] * len(gens)
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
        in_res = pd.Series(gens).isin(net["res_gen"].index).values
        to_add = net.res_gen.loc[pd.Index(gens)[in_res]]
        to_add.index = pd.Index(new_idx)[in_res]
        net.res_sgen = pd.concat([net.res_sgen, to_add], sort=True)
        net.res_gen.drop(pd.Index(gens)[in_res], inplace=True)
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
        "max_q_mvar", "min_q_mvar". However cols_to_keep is given, these columns are always set:
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
        gen_indices = [None] * len(sgens)
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
    # add columns which should be kept from sgen but miss in gen to net.gen
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
        in_res = pd.Series(sgens).isin(net["res_sgen"].index).values
        to_add = net.res_sgen.loc[pd.Index(sgens)[in_res]]
        to_add.index = pd.Index(new_idx)[in_res]
        net.res_gen = pd.concat([net.res_gen, to_add], sort=True)
        net.res_sgen.drop(pd.Index(sgens)[in_res], inplace=True)
    return new_idx


def replace_pq_elmtype(net, old_elm, new_elm, old_indices=None, new_indices=None, cols_to_keep=None,
                       add_cols_to_keep=None):
    """
    Replaces e.g. static generators by loads or loads by storages and so forth.

    INPUT:
        **net** - pandapower net

        **old_elm** (str) - element type of which elements should be replaced. Should be in [
            "sgen", "load", "storage"]

        **new_elm** (str) - element type of which elements should be created. Should be in [
            "sgen", "load", "storage"]

    OPTIONAL:
        **old_indices** (iterable) - indices of the elements which should be replaced

        **new_indices** (iterable) - required indices of the new elements

        **cols_to_keep** (list, None) - list of column names which should be kept while replacing.
        If None these columns are kept if values exist: "max_p_mw", "min_p_mw",
        "max_q_mvar", "min_q_mvar". Independent whether cols_to_keep is given, these columns are
        always set: "bus", "p_mw", "q_mvar", "name", "in_service", "controllable"

        **add_cols_to_keep** (list, None) - list of column names which should be added to
        'cols_to_keep' to be kept while replacing.

    OUTPUT:
        **new_idx** (list) - list of indices of the new elements
    """
    if old_elm == new_elm:
        logger.warning("'old_elm' and 'new_elm' are both '%s'. No replacement is done." % old_elm)
        return old_indices
    if old_indices is None:
        old_indices = net[old_elm].index
    else:
        old_indices = ensure_iterability(old_indices)
    if not len(old_indices):
        return []
    if new_indices is None:
        new_indices = [None] * len(old_indices)
    elif len(new_indices) != len(old_indices):
        raise ValueError("The length of 'new_indices' must be the same as of 'old_indices' but " +
                         "is %i instead of %i" % (len(new_indices), len(old_indices)))

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

    existing_cols_to_keep = net[old_elm].loc[old_indices].dropna(axis=1).columns.intersection(
        cols_to_keep)
    # add missing columns to net[new_elm] which should be kept
    missing_cols_to_keep = existing_cols_to_keep.difference(net[new_elm].columns)
    for col in missing_cols_to_keep:
        net[new_elm][col] = np.nan

    # --- create new_elm
    already_considered_cols = set()
    new_idx = []
    for oelm, index in zip(net[old_elm].loc[old_indices].itertuples(), new_indices):
        controllable = False if "controllable" not in net[old_elm].columns else oelm.controllable
        sign = -1 if old_elm in ["sgen"] else 1
        args = dict()
        if new_elm == "load":
            fct = create_load
        elif new_elm == "sgen":
            fct = create_sgen
            sign *= -1
        elif new_elm == "storage":
            fct = create_storage
            already_considered_cols |= {"max_e_mwh"}
            args = {"max_e_mwh": 1 if "max_e_mwh" not in net[old_elm].columns else net[
                old_elm].max_e_kwh.loc[old_indices]}
        idx = fct(net, oelm.bus, p_mw=sign*oelm.p_mw, q_mvar=sign*oelm.q_mvar, name=oelm.name,
                  in_service=oelm.in_service, controllable=controllable, index=index, **args)
        new_idx.append(idx)

    if sign == -1:
        for col1, col2 in zip(["max_p_mw", "min_p_mw", "max_q_mvar", "min_q_mvar"],
                              ["min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"]):
            if col1 in existing_cols_to_keep:
                net[new_elm].loc[new_idx, col2] = sign * net[old_elm].loc[
                    old_indices, col1].values
                already_considered_cols |= {col1}
    net[new_elm].loc[new_idx, existing_cols_to_keep.difference(already_considered_cols)] = net[
        old_elm].loc[old_indices, existing_cols_to_keep.difference(already_considered_cols)].values

    # --- drop replaced old_indices
    net[old_elm].drop(old_indices, inplace=True)

    # --- adapt cost data
    for table in ["pwl_cost", "poly_cost"]:
        if net[table].shape[0]:
            to_change = net[table].index[(net[table].et == old_elm) &
                                         (net[table].element.isin(old_indices))]
            if len(to_change):
                net[table].et.loc[to_change] = new_elm
                net[table].element.loc[to_change] = new_idx

    # --- result data
    if net["res_" + old_elm].shape[0]:
        in_res = pd.Series(old_indices).isin(net["res_" + old_elm].index).values
        to_add = net["res_" + old_elm].loc[pd.Index(old_indices)[in_res]]
        to_add.index = pd.Index(new_idx)[in_res]
        net["res_" + new_elm] = pd.concat([net["res_" + new_elm], to_add], sort=True)
        net["res_" + old_elm].drop(pd.Index(old_indices)[in_res], inplace=True)
    return new_idx


def replace_ward_by_internal_elements(net, wards=None):
    """
    Replaces wards by loads and shunts.

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
        sign_in_service = np.multiply(net.ward.in_service.loc[wards].values, 1)
        sign_not_isolated = np.multiply(net.res_ward.vm_pu.loc[wards].values != 0, 1)
        to_add_load = net.res_ward.loc[wards, ["p_mw", "q_mvar"]]
        to_add_load.index = new_load_idx
        to_add_load.p_mw = net.ward.ps_mw.loc[wards].values * sign_in_service * sign_not_isolated
        to_add_load.q_mvar = net.ward.qs_mvar.loc[wards].values * sign_in_service * \
            sign_not_isolated
        net.res_load = pd.concat([net.res_load, to_add_load])

        to_add_shunt = net.res_ward.loc[wards, ["p_mw", "q_mvar", "vm_pu"]]
        to_add_shunt.index = new_shunt_idx
        to_add_shunt.p_mw = net.res_ward.vm_pu.loc[wards].values ** 2 * net.ward.pz_mw.loc[
            wards].values * sign_in_service * sign_not_isolated
        to_add_shunt.q_mvar = net.res_ward.vm_pu.loc[wards].values ** 2 * net.ward.qz_mvar.loc[
            wards].values * sign_in_service * sign_not_isolated
        to_add_shunt.vm_pu = net.res_ward.vm_pu.loc[wards].values
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
        create_impedance(net, xward.bus, bus_idx, xward.r_ohm / (bus_v ** 2),
                         xward.x_ohm / (bus_v ** 2), net.sn_mva, in_service=xward.in_service,
                         name=xward.name)

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
      **exact_match** (boolean, True) -
          True: Expects exactly one match, raises UserWarning otherwise.
          False: returns all indices containing the name

    OUTPUT:
      **index** - The indices of matching element(s).
    """
    if exact_match:
        idx = net[element][net[element]["name"] == name].index
        if len(idx) == 0:
            raise UserWarning(f"There is no {element} with name {name}")
        if len(idx) > 1:
            raise UserWarning(f"Duplicate {element} names for {name}")
        return idx[0]
    else:
        return net[element][net[element]["name"].str.contains(name)].index


def get_element_indices(net, element, name, exact_match=True):
    """
    Returns a list of element(s) identified by a name or regex and its element-table -> Wrapper
    function of get_element_index()

    INPUT:
      **net** - pandapower network

      **element** (str, string iterable) - Element table to get indices from
      ("line", "bus", "trafo" etc.).

      **name** (str) - Name of the element to match.

    OPTIONAL:
      **exact_match** (boolean, True)

          - True: Expects exactly one match, raises UserWarning otherwise.
          - False: returns all indices containing the name

    OUTPUT:
      **index** (list) - List of the indices of matching element(s).

    EXAMPLE:
        >>> import pandapower.networks as pn
        >>> import pandapower as pp
        >>> net = pn.example_multivoltage()
        >>> idx1 = pp.get_element_indices(net, "bus", ["Bus HV%i" % i for i in range(1, 4)])
        >>> idx2 = pp.get_element_indices(net, ["bus", "line"], "HV", exact_match=False)
        >>> idx3 = pp.get_element_indices(net, ["bus", "line"], ["Bus HV3", "MV Line6"])
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
    # todo: what to do with trafo3w?
    if et == 'line' or et == 'l':
        bc = ["from_bus", "to_bus"]
    elif et == 'trafo' or et == 't':
        bc = ["hv_bus", "lv_bus"]
    elif et == "switch" and list(net[et].loc[element_id, ["et"]].values) == ['b']:
        # Raises error if switch is not a bus-bus switch
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
        **respect_switches** (boolean, True)

            - True: open switches will be respected
            - False: open switches will be ignored

        **respect_in_service** (boolean, False)

            - True: in_service status of connected lines will be respected
            - False: in_service status will be ignored

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
    elif element == "measurement":
        element_table = net[element]
        connected_elements = set(net.measurement.index[(net.measurement.element.isin(buses)) |
                                                       (net.measurement.element_type == "bus")])
    elif element in pp_elements(bus=False, branch_elements=False):
        element_table = net[element]
        connected_elements = set(element_table.index[(element_table.bus.isin(buses))])
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

    if respect_in_service and "in_service" in element_table and not element_table.empty:
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
        **respect_switches** (boolean, True)

            - True: open switches will be respected
            - False: open switches will be ignored

        **respect_in_service** (boolean, False)

            - True: in_service status of connected buses will be respected
            - False: in_service status will be ignored

        **consider** (iterable, ("l", "s", "t", "t3", "i")) - Determines, which types of
        connections will be considered.

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
    if "l" in consider or 'line' in consider:
        in_service_constr = net.line.in_service if respect_in_service else True
        opened_lines = set(net.switch.loc[(~net.switch.closed) & (
            net.switch.et == "l")].element.unique()) if respect_switches else set()
        connected_fb_lines = set(net.line.index[(
            net.line.from_bus.isin(buses)) & ~net.line.index.isin(opened_lines) &
            in_service_constr])
        connected_tb_lines = set(net.line.index[(
            net.line.to_bus.isin(buses)) & ~net.line.index.isin(opened_lines) & in_service_constr])
        cb |= set(net.line[net.line.index.isin(connected_tb_lines)].from_bus)
        cb |= set(net.line[net.line.index.isin(connected_fb_lines)].to_bus)

    if "s" in consider or 'switch' in consider:
        cs = get_connected_switches(net, buses, consider='b',
                                    status="closed" if respect_switches else "all")
        cb |= set(net.switch[net.switch.index.isin(cs)].element)
        cb |= set(net.switch[net.switch.index.isin(cs)].bus)

    if "t" in consider or 'trafo' in consider:
        in_service_constr = net.trafo.in_service if respect_in_service else True
        opened_trafos = set(net.switch.loc[(~net.switch.closed) & (
            net.switch.et == "t")].element.unique()) if respect_switches else set()
        connected_hvb_trafos = set(net.trafo.index[(
            net.trafo.hv_bus.isin(buses)) & ~net.trafo.index.isin(opened_trafos) &
            in_service_constr])
        connected_lvb_trafos = set(net.trafo.index[(
            net.trafo.lv_bus.isin(buses)) & ~net.trafo.index.isin(opened_trafos) &
            in_service_constr])
        cb |= set(net.trafo.loc[net.trafo.index.isin(connected_lvb_trafos)].hv_bus.values)
        cb |= set(net.trafo.loc[net.trafo.index.isin(connected_hvb_trafos)].lv_bus.values)

    # Gives the lv mv and hv buses of a 3 winding transformer
    if "t3" in consider or 'trafo3w' in consider:
        in_service_constr3w = net.trafo3w.in_service if respect_in_service else True
        if respect_switches:
            opened_buses_hv = set(net.switch.loc[
                ~net.switch.closed & (net.switch.et == "t3") &
                net.switch.bus.isin(net.trafo3w.hv_bus)].bus.unique())
            opened_buses_mv = set(net.switch.loc[
                ~net.switch.closed & (net.switch.et == "t3") &
                net.switch.bus.isin(net.trafo3w.mv_bus)].bus.unique())
            opened_buses_lv = set(net.switch.loc[
                ~net.switch.closed & (net.switch.et == "t3") &
                net.switch.bus.isin(net.trafo3w.lv_bus)].bus.unique())
        else:
            opened_buses_hv = opened_buses_mv = opened_buses_lv = set()

        hvb_trafos3w = net.trafo3w.index[
            net.trafo3w.hv_bus.isin(buses) & ~net.trafo3w.hv_bus.isin(opened_buses_hv) &
            in_service_constr3w]
        mvb_trafos3w = net.trafo3w.index[
            net.trafo3w.mv_bus.isin(buses) & ~net.trafo3w.mv_bus.isin(opened_buses_mv) &
            in_service_constr3w]
        lvb_trafos3w = net.trafo3w.index[
            net.trafo3w.lv_bus.isin(buses) & ~net.trafo3w.lv_bus.isin(opened_buses_lv) &
            in_service_constr3w]

        cb |= (set(net.trafo3w.loc[hvb_trafos3w].mv_bus) | set(
            net.trafo3w.loc[hvb_trafos3w].lv_bus) - opened_buses_mv - opened_buses_lv)
        cb |= (set(net.trafo3w.loc[mvb_trafos3w].hv_bus) | set(
            net.trafo3w.loc[mvb_trafos3w].lv_bus) - opened_buses_hv - opened_buses_lv)
        cb |= (set(net.trafo3w.loc[lvb_trafos3w].hv_bus) | set(
            net.trafo3w.loc[lvb_trafos3w].mv_bus) - opened_buses_hv - opened_buses_mv)

    if "i" in consider or 'impedance' in consider:
        in_service_constr = net.impedance.in_service if respect_in_service else True
        connected_fb_impedances = set(net.impedance.index[
                                          (net.impedance.from_bus.isin(buses)) & in_service_constr])
        connected_tb_impedances = set(net.impedance.index[
                                          (net.impedance.to_bus.isin(buses)) & in_service_constr])
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

        **et** (string) - Type of the source element:

            l, line: line

            s, switch: switch

            t, trafo: trafo

            t3, trafo3w: trafo3w

            i, impedance: impedance

     OPTIONAL:
        **respect_in_service** (boolean, False)

        True: in_service status of connected buses will be respected

        False: in_service status will be ignored

     OUTPUT:
        **cl** (set) - Returns connected switches.

    """

    cb = set()
    if et == 'l' or et == 'line':
        cb.add(net.line.from_bus.at[element])
        cb.add(net.line.to_bus.at[element])
    elif et == 's' or et == 'switch':
        cb.add(net.switch.bus.at[element])
        if net.switch.et.at[element] == 'b':
            cb.add(net.switch.element.at[element])
    elif et == 't' or et == 'trafo':
        cb.add(net.trafo.hv_bus.at[element])
        cb.add(net.trafo.lv_bus.at[element])
    elif et == 't3' or et == 'trafo3w':
        cb.add(net.trafo3w.hv_bus.at[element])
        cb.add(net.trafo3w.mv_bus.at[element])
        cb.add(net.trafo3w.lv_bus.at[element])
    elif et == 'i' or et == 'impedance':
        cb.add(net.impedance.from_bus.at[element])
        cb.add(net.impedance.to_bus.at[element])

    if respect_in_service:
        cb -= set(net.bus[~net.bus.in_service].index)

    return cb


def get_connected_switches(net, buses, consider=('b', 'l', 't', 't3'), status="all"):
    """
    Returns switches connected to given buses.

    INPUT:
        **net** (pandapowerNet)

        **buses** (single integer or iterable of ints)

    OPTIONAL:
        **consider** (iterable, ("l", "s", "t", "t3))    - Determines, which types of connections
                                                      will be considered.
                                                      l: lines
                                                      b: bus-bus-switches
                                                      t: transformers
                                                      t3: 3W transformers

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
    else:
        switch_selection = np.full(len(net.switch), True, dtype=bool)
        if status != "all":
            logger.warning("Unknown switch status \"%s\" selected! "
                           "Selecting all switches by default." % status)

    cs = set()
    for et in consider:
        if et == 'b':
            cs |= set(net['switch'].index[
                          (net['switch']['bus'].isin(buses) |
                           net['switch']['element'].isin(buses)) &
                          (net['switch']['et'] == 'b') & switch_selection])
        else:
            cs |= set(net['switch'].index[(net['switch']['bus'].isin(buses)) &
                                          (net['switch']['et'] == et) & switch_selection])

    return cs


def get_connected_elements_dict(
        net, buses, respect_switches=True, respect_in_service=False, include_empty_lists=False,
        element_types=None, **kwargs):
    """Returns a dict of lists of connected elements.

    Parameters
    ----------
    net : _type_
        _description_
    buses : iterable of buses
        buses as origin to search for connected elements
    respect_switches : bool, optional
        _description_, by default True
    respect_in_service : bool, optional
        _description_, by default False
    include_empty_lists : bool, optional
        if True, the output doesn't have values of empty lists but may lack of element types as
        keys, by default False
    element_types : iterable of strings, optional
        types elements which are analyzed for connection. If not given, all pandapower element types
        are analyzed. That list of all element types can also be restricted by key word arguments
        "connected_buses", "connected_bus_elements", "connected_branch_elements" and
        "connected_other_elements", by default None

    Returns
    -------
    dict[str,list]
        elements connected to given buses
    """
    if element_types is None:
        element_types = pp_elements(
            bus=kwargs.get("connected_buses", True),
            bus_elements=kwargs.get("connected_bus_elements", True),
            branch_elements=kwargs.get("connected_branch_elements", True),
            other_elements=kwargs.get("connected_other_elements", True),
            cost_tables=False,
            res_elements=False)

    connected = dict()
    for et in element_types:
        if et == "bus":
            conn = get_connected_buses(net, buses, respect_switches=respect_switches,
                                       respect_in_service=respect_in_service)
        elif et == "switch":
            conn = get_connected_switches(net, buses)
        else:
            conn = get_connected_elements(
                net, et, buses, respect_switches=respect_switches,
                respect_in_service=respect_in_service)
        if include_empty_lists or len(conn):
            connected[et] = list(conn)
    return connected


def get_gc_objects_dict():
    """
    This function is based on the code in mem_top module
    Summarize object types that are tracket by the garbage collector in the moment.
    Useful to test if there are memoly leaks.
    :return: dictionary with keys corresponding to types and values to the number of objects of the
    type
    """
    objs = gc.get_objects()
    nums_by_types = dict()

    for obj in objs:
        _type = type(obj)
        nums_by_types[_type] = nums_by_types.get(_type, 0) + 1
    return nums_by_types


def false_elm_links(net, elm, col, target_elm):
    """
    Returns which indices have links to elements of other element tables which does not exist in the
    net.

    EXAMPLE 1:
        elm = "line"
        col = "to_bus"
        target_elm = "bus"

    EXAMPLE 2:
        elm = "poly_cost"
        col = "element"
        target_elm = net["poly_cost"]["et"]
    """
    if isinstance(target_elm, str):
        return net[elm][col].index[~net[elm][col].isin(net[target_elm].index)]
    else:  # target_elm is an iterable, e.g. a Series such as net["poly_cost"]["et"]
        df = pd.DataFrame({"element": net[elm][col].values, "et": target_elm,
                           "indices": net[elm][col].index.values})
        df = df.set_index("et")
        false_links = pd.Index([])
        for et in df.index:
            false_links = false_links.union(pd.Index(df.loc[et].indices.loc[
                ~df.loc[et].element.isin(net[et].index)]))
        return false_links


def false_elm_links_loop(net, elms=None):
    """
    Returns a dict of elements which indices have links to elements of other element tables which
    does not exist in the net.
    This function is an outer loop for get_false_links() applications.
    """
    false_links = dict()
    elms = elms if elms is not None else pp_elements(bus=False, cost_tables=True)
    bebd = branch_element_bus_dict(include_switch=True)
    for elm in elms:
        if net[elm].shape[0]:
            fl = pd.Index([])
            # --- define col and target_elm
            if elm in bebd.keys():
                for col in bebd[elm]:
                    fl = fl.union(false_elm_links(net, elm, col, "bus"))
            elif elm in {"poly_cost", "pwl_cost"}:
                fl = fl.union(false_elm_links(net, elm, "element", net[elm]["et"]))
            elif elm == "measurement":
                fl = fl.union(false_elm_links(net, elm, "element", net[elm]["element_type"]))
            else:
                fl = fl.union(false_elm_links(net, elm, "bus", "bus"))
            if len(fl):
                false_links[elm] = fl
    return false_links


def read_from_net(net, element, index, variable, flag='auto'):
    """
    Reads values from the specified element table at the specified index in the column according to the specified variable
    Chooses the method to read based on flag

    Parameters
    ----------
    net
    element : str
        element table in pandapower net; can also be a results table
    index : int or array_like
        index of the element table where values are read from
    variable : str
        column of the element table
    flag : str
        defines which underlying function to use, can be one of ['auto', 'single_index', 'all_index', 'loc', 'object']

    Returns
    -------
    values
        the values of the variable for the element table according to the index
    """
    if flag == "single_index":
        return _read_from_single_index(net, element, variable, index)
    elif flag == "all_index":
        return _read_from_all_index(net, element, variable)
    elif flag == "loc":
        return _read_with_loc(net, element, variable, index)
    elif flag == "object":
        return _read_from_object_attribute(net, element, variable, index)
    elif flag == "auto":
        auto_flag, auto_variable = _detect_read_write_flag(net, element, index, variable)
        return read_from_net(net, element, index, auto_variable, auto_flag)
    else:
        raise NotImplementedError("read: flag must be one of ['auto', 'single_index', 'all_index', 'loc', 'object']")


def write_to_net(net, element, index, variable, values, flag='auto'):
    """
    Writes values to the specified element table at the specified index in the column according to the specified variable
    Chooses the method to write based on flag

    Parameters
    ----------
    net
    element : str
        element table in pandapower net
    index : int or array_like
        index of the element table where values are written to
    variable : str
        column of the element table
    flag : str
        defines which underlying function to use, can be one of ['auto', 'single_index', 'all_index', 'loc', 'object']

    Returns
    -------
    None
    """
    # write functions faster, depending on type of element_index
    if flag == "single_index":
        _write_to_single_index(net, element, index, variable, values)
    elif flag == "all_index":
        _write_to_all_index(net, element, variable, values)
    elif flag == "loc":
        _write_with_loc(net, element, index, variable, values)
    elif flag == "object":
        _write_to_object_attribute(net, element, index, variable, values)
    elif flag == "auto":
        auto_flag, auto_variable = _detect_read_write_flag(net, element, index, variable)
        write_to_net(net, element, index, auto_variable, values, auto_flag)
    else:
        raise NotImplementedError("write: flag must be one of ['auto', 'single_index', 'all_index', 'loc', 'object']")


def _detect_read_write_flag(net, element, index, variable):
    if variable.startswith('object'):
        # write to object attribute
        return "object", variable.split(".")[1]
    elif isinstance(index, numbers.Number):
        # use .at if element_index is integer for speedup
        return "single_index", variable
    # commenting this out for now, see issue 609
    # elif net[element].index.equals(Index(index)):
    #     # use : indexer if all elements are in index
    #     return "all_index", variable
    else:
        # use common .loc
        return "loc", variable


# read functions:
def _read_from_single_index(net, element, variable, index):
    return net[element].at[index, variable]


def _read_from_all_index(net, element, variable):
    return net[element].loc[:, variable].values


def _read_with_loc(net, element, variable, index):
    return net[element].loc[index, variable].values


def _read_from_object_attribute(net, element, variable, index):
    if hasattr(index, '__iter__') and len(index) > 1:
        values = np.array(shape=index.shape)
        for i, idx in enumerate(index):
            values[i] = getattr(net[element]["object"].at[idx], variable)
    else:
        values = getattr(net[element]["object"].at[index], variable)
    return values


# write functions:
def _write_to_single_index(net, element, index, variable, values):
    net[element].at[index, variable] = values


def _write_to_all_index(net, element, variable, values):
    net[element].loc[:, variable] = values


def _write_with_loc(net, element, index, variable, values):
    net[element].loc[index, variable] = values


def _write_to_object_attribute(net, element, index, variable, values):
    if hasattr(index, '__iter__') and len(index) > 1:
        for idx, val in zip(index, values):
            setattr(net[element]["object"].at[idx], variable, val)
    else:
        setattr(net[element]["object"].at[index], variable, values)


# group function

def group_row(net, index, element_type):
    """Returns the row which consists the data of the requested group index and element type.
    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    index : int
        index of the group
    element_type : str
        element type (defines which row of the data of the group should be returned)
    Returns
    -------
    pandas.Series
        data of net.group, defined by the index of the group and the element type
    Raises
    ------
    KeyError
        Now row exist for the requested group and element type
    ValueError
        Multiple rows exist for the requested group and element type
    """
    group_df = net.group.loc[[index]].set_index("element_type")
    try:
        row = group_df.loc[element_type]
    except KeyError:
        raise KeyError(f"Group {index} has no {element_type}s.")
    if isinstance(row, pd.Series):
        return row
    elif isinstance(row, pd.DataFrame):
        raise ValueError(f"Multiple {element_type} rows for group {index}")
    else:
        raise ValueError(f"Returning row {element_type} for group {index} failed.")


def group_element_index(net, index, element_type):
    """Returns the indices of the elements of the group in the element table net[element_type]. This
    function considers net.group.reference_column.
    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    index : int
        Index of the group
    element_type : str
        name of the element table to which the returned indices of the elements of the group belong
        to
    Returns
    -------
    pd.Index
        indices of the elements of the group in the element table net[element_type]
    """
    if element_type not in net.group.loc[[index], "element_type"].values:
        return pd.Index([], dtype=int)

    row = group_row(net, index, element_type)
    element = row.at["element"]
    reference_column = row.at["reference_column"]

    if reference_column is None or pd.isnull(reference_column):
        return pd.Index(element, dtype=int)

    return net[element_type].index[net[element_type][reference_column].isin(element)]