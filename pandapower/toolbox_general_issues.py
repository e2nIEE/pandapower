# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
import pandas as pd
import pandas.testing as pdt
import networkx as nx

from pandapower.auxiliary import ensure_iterability, pandapowerNet

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


def count_elements(net, return_empties=False, **kwargs):
    """Counts how much elements of which element type exist in the pandapower net

    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    return_empties : bool, optional
        whether element types should be listed if no element exist, by default False

    Other Parameters
    ----------------
    kwargs : dict[str,bool], optional
        arguments (passed to pp_elements()) to narrow considered element types.
        If nothing is passed, an empty dict is passed to pp_elements(), by default None

    Returns
    -------
    pd.Series
        number of elements per element type existing in the net

    See also
    --------
    count_group_elements

    Examples
    --------
    >>> import pandapower as pp
    >>> import pandapower.networks as nw
    >>> pp.count_elements(nw.case9(), bus_elements=False)
    bus     9
    line    9
    dtype: int32
    """
    return pd.Series({et: net[et].shape[0] for et in pp_elements(**kwargs) if return_empties or \
        bool(net[et].shape[0])}, dtype=int)


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
        raise ValueError("x and y need to have the same shape.")


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
