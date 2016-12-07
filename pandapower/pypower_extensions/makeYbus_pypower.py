# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Builds the bus admittance matrix and branch admittance matrices.
"""

from sys import stderr

from numpy import ones, any, r_, real
from scipy.sparse import csr_matrix

from pypower.idx_bus import BUS_I, GS, BS
from pypower.idx_brch import F_BUS, T_BUS
from pandapower.pypower_extensions.makeYbus import branch_vectors

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
    nb = bus.shape[0]          ## number of buses
    nl = branch.shape[0]       ## number of lines

    ## check that bus numbers are equal to indices to bus (one set of bus nums)
    if any(bus[:, BUS_I] != list(range(nb))):
        stderr.write('buses must appear in order by bus number\n')

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
    f = real(branch[:, F_BUS]).astype(int)                           ## list of "from" buses
    t = real(branch[:, T_BUS]).astype(int)                           ## list of "to" buses
    ## connection matrix for line & from buses
    Cf = csr_matrix((ones(nl), (range(nl), f)), (nl, nb))
    ## connection matrix for line & to buses
    Ct = csr_matrix((ones(nl), (range(nl), t)), (nl, nb))

    ## build Yf and Yt such that Yf * V is the vector of complex branch currents injected
    ## at each branch's "from" bus, and Yt is the same for the "to" bus end
    i = r_[range(nl), range(nl)]                   ## double set of row indices

    Yf = csr_matrix((r_[Yff, Yft], (i, r_[f, t])), (nl, nb))
    Yt = csr_matrix((r_[Ytf, Ytt], (i, r_[f, t])), (nl, nb))
    # Yf = spdiags(Yff, 0, nl, nl) * Cf + spdiags(Yft, 0, nl, nl) * Ct
    # Yt = spdiags(Ytf, 0, nl, nl) * Cf + spdiags(Ytt, 0, nl, nl) * Ct

    ## build Ybus
    Ybus = Cf.T * Yf + Ct.T * Yt + \
        csr_matrix((Ysh, (range(nb), range(nb))), (nb, nb))

    return Ybus, Yf, Yt
