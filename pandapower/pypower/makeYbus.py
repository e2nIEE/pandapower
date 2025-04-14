# -*- coding: utf-8 -*-

# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


"""Builds the bus admittance matrix and branch admittance matrices.
"""

from numpy import ones, conj, nonzero, any, exp, pi, hstack, real, int64, errstate
from scipy.sparse import csr_matrix

from pandapower.pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, BR_G, BR_STATUS, SHIFT, TAP, BR_R_ASYM, \
    BR_X_ASYM, BR_G_ASYM, BR_B_ASYM
from pandapower.pypower.idx_bus import GS, BS


def makeYbus(baseMVA, bus, branch):
    """Builds the bus admittance matrix and branch admittance matrices.

    Returns the full bus admittance matrix (i.e. for all buses) and the
    matrices C{Yf} and C{Yt} which, when multiplied by a complex voltage
    vector, yield the vector currents injected into each line from the
    "from" and "to" buses respectively of each line. Does appropriate
    conversions to p.u.

    @see: L{makeSbus}

    @author: Ray Zimmerman (PSERC Cornell)
    @author: Richard Lincoln
    """
    ## constants
    nb = bus.shape[0]  ## number of buses
    nl = branch.shape[0]  ## number of lines

    ## for each branch, compute the elements of the branch admittance matrix where
    ##
    ##      | If |   | Yff  Yft |   | Vf |
    ##      |    | = |          | * |    |
    ##      | It |   | Ytf  Ytt |   | Vt |
    ##
    Ytt, Yff, Yft, Ytf = branch_vectors(branch, nl)
    ## compute shunt admittance
    ## if Psh is the real power consumed by the shunt at V = 1.0 p.u.
    ## and Qsh is the reactive power injected by the shunt at V = 1.0 p.u.
    ## then Psh - j Qsh = V * conj(Ysh * V) = conj(Ysh) = Gs - j Bs,
    ## i.e. Ysh = Psh + j Qsh, so ...
    ## vector of shunt admittances
    Ysh = (bus[:, GS] + 1j * bus[:, BS]) / baseMVA

    ## build connection matrices
    f = real(branch[:, F_BUS]).astype(int64)  ## list of "from" buses
    t = real(branch[:, T_BUS]).astype(int64)  ## list of "to" buses
    ## connection matrix for line & from buses
    Cf = csr_matrix((ones(nl), (range(nl), f)), (nl, nb))
    ## connection matrix for line & to buses
    Ct = csr_matrix((ones(nl), (range(nl), t)), (nl, nb))

    ## build Yf and Yt such that Yf * V is the vector of complex branch currents injected
    ## at each branch's "from" bus, and Yt is the same for the "to" bus end
    i = hstack([range(nl), range(nl)])  ## double set of row indices

    Yf = csr_matrix((hstack([Yff, Yft]), (i, hstack([f, t]))), (nl, nb))
    Yt = csr_matrix((hstack([Ytf, Ytt]), (i, hstack([f, t]))), (nl, nb))
    # Yf = spdiags(Yff, 0, nl, nl) * Cf + spdiags(Yft, 0, nl, nl) * Ct
    # Yt = spdiags(Ytf, 0, nl, nl) * Cf + spdiags(Ytt, 0, nl, nl) * Ct

    ## build Ybus
    Ybus = Cf.T * Yf + Ct.T * Yt + \
           csr_matrix((Ysh, (range(nb), range(nb))), (nb, nb))

    # for canonical format
    for Y in (Ybus, Yf, Yt):
        Y.eliminate_zeros()
        Y.sum_duplicates()
        Y.sort_indices()
        del Y._has_canonical_format

    return Ybus, Yf, Yt


@errstate(all="raise")
def branch_vectors(branch, nl):
    stat = branch[:, BR_STATUS]  # ones at in-service branches
    Ysf = stat / (branch[:, BR_R] + 1j * branch[:, BR_X])  # series admittance
    if any(branch[:, BR_R_ASYM]) or any(branch[:, BR_X_ASYM]):
        Yst = stat / (branch[:, BR_R] + branch[:, BR_R_ASYM] +
                      1j * (branch[:, BR_X] + branch[:, BR_X_ASYM]))
    else:
        Yst = Ysf

    Bcf = stat * (branch[:, BR_G] + 1j * branch[:, BR_B])  # branch charging admittance
    if any(branch[:, BR_G_ASYM]) or any(branch[:, BR_B_ASYM]):
        Bct = stat * (branch[:, BR_G] + branch[:, BR_G_ASYM] +
                      1j * (branch[:, BR_B] + branch[:, BR_B_ASYM]))
    else:
        Bct = Bcf

    tap = ones(nl)  # default tap ratio = 1
    i = nonzero(real(branch[:, TAP]))  # indices of non-zero tap ratios
    tap[i] = real(branch[i, TAP])  # assign non-zero tap ratios
    tap = tap * exp(1j * pi / 180 * branch[:, SHIFT])  # add phase shifters

    Ytt = Yst + Bct / 2
    Yff = (Ysf + Bcf / 2) / (tap * conj(tap))
    Yft = - Ysf / conj(tap)
    Ytf = - Yst / tap
    return Ytt, Yff, Yft, Ytf

