# -*- coding: utf-8 -*-

# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


"""Builds the B matrices and phase shift injections for DC power flow.
"""
import numpy as np
from numpy import ones, zeros_like, r_, pi, flatnonzero as find, real, int64, float64, divide, errstate
from numpy.typing import NDArray
from pandapower.pypower.idx_brch import F_BUS, T_BUS, BR_X, TAP, SHIFT, BR_STATUS
from pandapower.pypower.idx_brch_dc import DC_F_BUS, DC_T_BUS, DC_BR_STATUS, DC_BR_R, branch_dc_cols
from pandapower.pypower.idx_bus import BUS_I
from pandapower.pypower.idx_bus_dc import dc_bus_cols
from pandapower.pypower.idx_vsc import VSC_BUS, VSC_BUS_DC, VSC_STATUS, VSC_X, VSC_MODE_DC, VSC_MODE_DC_P, vsc_cols

from scipy.sparse import csr_matrix, csc_matrix

import logging

logger = logging.getLogger(__name__)


def makeBdc(bus: NDArray[float64],
            branch: NDArray[float64],
            bus_dc: NDArray[float64] | None = None,
            branch_dc: NDArray[float64] | None = None,
            vsc: NDArray[float64] | None = None,
            return_csr: bool = True):
    """
    Builds the B matrices and phase shift injections for DC power flow.

    Returns the B matrices and phase shift injection vectors needed for a DC power flow.
    The bus real power injections are related to bus voltage angles by::
        P = Bbus * Va + PBusinj
    The real power flows at the from end the lines are related to the bus voltage angles by:
        Pf = Bf * Va + Pfinj
    Does appropriate conversions to p.u.

    @see: L{dcpf}

    @author: Carlos E. Murillo-Sanchez (PSERC Cornell & Universidad
    Autonoma de Manizales)
    @author: Ray Zimmerman (PSERC Cornell)
    @author: Richard Lincoln

    :param bus: the ppci buses
    :param branch: the ppci branches
    :param bus_dc: the ppci dc buses
    :param branch_dc: the ppci dc branches
    :param vsc: the ppci vscs
    :param return_csr: return a csr matrix instead of a csc matrix
    """
    if bus_dc is None:
        bus_dc = np.empty((0, dc_bus_cols))
    if branch_dc is None:
        branch_dc = np.empty((0, branch_dc_cols))
    if vsc is None:
        vsc = np.empty((0, vsc_cols))
    # Select csc/csr B matrix
    sparse = csr_matrix if return_csr else csc_matrix

    # constants
    nb = bus.shape[0]           # number of buses
    nl = branch.shape[0]        # number of lines
    nb_dc = bus_dc.shape[0]     # number of dc buses
    nl_dc = branch_dc.shape[0]  # number of dc lines
    nvsc = vsc.shape[0]         # number of vscs

    # check that bus numbers are equal to indices to bus (one set of bus nums)
    if any(bus[:, BUS_I] != list(range(nb))):
        logger.error('makeBdc: buses must be numbered consecutively in bus matrix')

    # for each branch, compute the elements of the branch B matrix and the phase
    # shift "quiescent" injections, where
    #
    #      | Pf |   | Bff  Bft |   | Vaf |   | Pfinj |
    #      |    | = |          | * |     | + |       |
    #      | Pt |   | Btf  Btt |   | Vat |   | Ptinj |
    #
    b = calc_b_from_branch(branch, nl)
    b_vsc = calc_b_from_vsc_dc(vsc)
    b_dc = calc_b_from_branch_dc(branch_dc)

    # AC lines
    # build connection matrix Cft = Cf - Ct for line and from - to buses
    f = real(branch[:, F_BUS]).astype(int64)       # list of "from" buses
    t = real(branch[:, T_BUS]).astype(int64)       # list of "to" buses
    i = r_[range(nl), range(nl)]                   # double set of row indices
    # connection matrix
    Cft = sparse((r_[ones(nl), -ones(nl)], (i, r_[f, t])), (nl + nvsc +nl_dc, nb + nb_dc))

    # VSCs
    # build connection matrix Cft = Cf - Ct for VSC and AC - DC buses
    f_vsc = real(vsc[:, VSC_BUS]).astype(int64)              # list of "AC" buses
    t_vsc = real(vsc[:, VSC_BUS_DC]).astype(int64) + nb      # list of "DC" buses
    i_vsc = r_[range(nvsc), range(nvsc)] + nl                # double set of row indices
    # connection matrix
    Cft += sparse((r_[ones(nvsc), -ones(nvsc)], (i_vsc, r_[f_vsc, t_vsc])), (nl + nvsc + nl_dc, nb + nb_dc))
    # DC lines
    # build connection matrix Cft = Cf - Ct for DC line and from - to buses
    f_dc = real(branch_dc[:, DC_F_BUS]).astype(int64) + nb   # list of "from" buses
    t_dc = real(branch_dc[:, DC_T_BUS]).astype(int64) + nb   # list of "to" buses
    i_dc = r_[range(nl_dc), range(nl_dc)] + nl + nvsc        # double set of row indices
    # connection matrix
    Cft += sparse((r_[ones(nl_dc), -ones(nl_dc)], (i_dc, r_[f_dc, t_dc])), (nl + nvsc + nl_dc, nb + nb_dc))

    # build Bf such that Bf * Va is the vector of real branch powers injected at each branch's "from" bus
    Bf = sparse((r_[b, -b], (i, r_[f, t])), (nl + nvsc + nl_dc, nb + nb_dc))
    # VSC
    Bf += sparse((r_[b_vsc, -b_vsc], (i_vsc, r_[f_vsc, t_vsc])), (nl + nvsc + nl_dc, nb + nb_dc))
    # DC line
    Bf += sparse((r_[b_dc, -b_dc], (i_dc, r_[f_dc, t_dc])), (nl + nvsc + nl_dc, nb + nb_dc))

    # build Bbus
    Bbus = Cft.T * Bf

    # build phase shift injection vectors
    shift = np.concatenate([branch[:, SHIFT], np.zeros(nvsc + nl_dc)])
    Pfinj, Pbusinj = phase_shift_injection(np.concatenate([b, b_vsc, b_dc]), shift, Cft)

    return Bbus, Bf, Pbusinj, Pfinj, Cft


def phase_shift_injection(b, shift, Cft):
    # build phase shift injection vectors
    Pfinj = b * (-shift * pi / 180.)  # injected at the from bus
    Pbusinj = Cft.T * Pfinj
    return Pfinj, Pbusinj


# we set the numpy error handling for this function to raise error rather than issue a warning because
# otherwise the resulting nan values will propagate and case an error elsewhere, making the reason less obvious
@errstate(all="raise")
def calc_b_from_branch(branch, nl):
    stat = real(branch[:, BR_STATUS])  # ones at in-service branches
    br_x = real(branch[:, BR_X])  # ones at in-service branches
    b = zeros_like(stat, dtype=float64)
    # if some br_x values are 0 but the branches are not in service, we do not need to raise an error:
    # divide(x1=stat, x2=br_x, out=b, where=stat, dtype=float64)  ## series susceptance
    # however, we also work with ppci at this level, which only has in-service elements so we should just let it fail:
    divide(stat, br_x, out=b, dtype=float64)  ## series susceptance
    tap = ones(nl)  # default tap ratio = 1
    i = find(t := real(branch[:, TAP]))  # indices of non-zero tap ratios
    tap[i] = t[i]  # assign non-zero tap ratios
    b = b / tap
    return b


@errstate(all="raise")
def calc_b_from_branch_dc(branch):
    stat = real(branch[:, DC_BR_STATUS])  # ones at in-service branches
    br_x = real(branch[:, DC_BR_R])  # get the resistance from the branches
    b = zeros_like(stat, dtype=float64)
    divide(stat, br_x, out=b, dtype=float64)  # series susceptance
    return b


@errstate(all="raise")
def calc_b_from_vsc_dc(vsc):
    stat = real(vsc[:, VSC_STATUS])  # ones at in-service vscs
    br_x = real(vsc[:, VSC_X])  # get the resistance from the vscs
    b = zeros_like(stat, dtype=float64)
    divide(stat, br_x, out=b, dtype=float64)  # series susceptance
    b[vsc[:, VSC_MODE_DC] == VSC_MODE_DC_P] = 0.
    return b
