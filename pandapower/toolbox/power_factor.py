# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np

from pandapower.auxiliary import ensure_iterability
from pandapower.toolbox.element_selection import pp_elements

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


def signing_system_value(element_type):
    """
    Returns a 1 for all bus elements using the consumver viewpoint and a -1 for all bus elements
    using the generator viewpoint.
    """
    generator_viewpoint_ets = ["ext_grid", "gen", "sgen"]
    if element_type in generator_viewpoint_ets:
        return -1
    elif element_type in pp_elements(bus=False, other_elements=False):
        return 1
    else:
        raise ValueError("This function is defined for bus and branch elements, not for "
                         f"'{element_type}'.")


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
    pmode = np.array(["undef", "load", "gen"])[np.sign(p).astype(np.int64)]
    qmode = np.array(["underexcited", "underexcited", "overexcited"])[np.sign(q).astype(np.int64)]
    return cosphi, s, qmode, pmode

# -------------------------------------------------------------------------------------------------
"""
Functions with numeric encoding of qmodes "underexcited" or "overexcited":
pos_neg: cosphi is positive for positive q values and negative otherwise
pos: cosphi is not in [0, 1] and [-1, -0] but in [0, 1] and [1, 2]
"""
# -------------------------------------------------------------------------------------------------


def cosphi_pos_neg_from_pq(p, q):
    """Returns the cosphi value(s) for given active and reactive power(s).
    Positive q values lead to positive cosphi values, negative to negative.
    If p, q come from generator view point, positive cosphi correspond to overexcited behaviour
    while in load view point positive cosphi correspond to underexcited behaviour.

    Parameters
    ----------
    p : float(s)
        active power value(s)
    q : float(s)
        reactive power values

    Examples
    --------
    >>> import numpy as np
    >>> from pandapower.toolbox import cosphi_pos_neg_from_pq
    >>> np.round(cosphi_pos_neg_from_pq(0.76, 0.25), 5)
    0.94993
    >>> np.round(cosphi_pos_neg_from_pq(0.76, -0.25), 5)
    -0.94993
    >>> np.round(cosphi_pos_neg_from_pq([0.76, 0.76, 0.76, 0.76], [0.25, -0.25, 0, 0.1]), 5)
    array([ 0.94993,  -0.94993,  1.     ,  0.99145])
    >>> np.round(cosphi_pos_neg_from_pq([0.76, 0.76, -0.76, 0.76, 0, 0.1],
    ...                                 [0.25, -0.25, 0.25, 0.1, 0.1, 0]), 5)
    array([ 0.94993,  -0.94993,  0.94993, 0.99145, nan, 1])

    Returns
    -------
    [float, np.array]
        cosphi values where

    See also
    --------
    pandapower.toolbox.cosphi_from_pq
    pandapower.toolbox.cosphi_to_pos
    """
    cosphis = cosphi_from_pq(p, q)[0]
    return np.copysign(cosphis, q)



def cosphi_to_pos(cosphi):
    """ Signed cosphi values are converted into positive cosphi values from 0 to 1 and 1 to 2.

    Examples
    --------
    >>> cosphi_to_pos(0.96)
    0.96
    >>> cosphi_to_pos(-0.94)
    1.06
    >>> cosphi_to_pos(-0.96)
    1.04
    >>> cosphi_to_pos([0.96, -0.94, -0.96])
    np.array([0.96, 1.06, 1.04])

    See also
    --------
    pandapower.toolbox.cosphi_from_pos
    pandapower.toolbox.cosphi_pos_neg_from_pq
    """
    multiple = not (isinstance(cosphi, float) or isinstance(cosphi, int))
    cosphi = np.array(ensure_iterability(cosphi))
    cosphi[cosphi < 0] += 2
    if multiple:
        return cosphi
    else:
        return cosphi[0]


def cosphi_from_pos(cosphi):
    """ All positive cosphi values are converted back to signed cosphi values

    Examples
    --------
    >>> cosphi_from_pos(0.96)
    0.96
    >>> cosphi_from_pos(1.06)
    -0.94
    >>> cosphi_from_pos(1.04)
    -0.96
    >>> cosphi_from_pos([0.96, 1.06, 1.04])
    np.array([0.96, -0.94, -0.96])

    See also
    --------
    pandapower.toolbox.cosphi_to_pos
    pandapower.toolbox.cosphi_pos_neg_from_pq
    """
    multiple = not (isinstance(cosphi, float) or isinstance(cosphi, int))
    cosphi = np.array(ensure_iterability(cosphi))
    cosphi[cosphi > 1] -= 2
    if multiple:
        return cosphi
    else:
        return cosphi[0]
