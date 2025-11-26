# -*- coding: utf-8 -*-

# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

# Builds the DC PTDF matrix for a given choice of slack.


from sys import stderr

from numpy import zeros, arange, isscalar, dot, ix_, flatnonzero as find
import numpy as np
from numpy.linalg import solve
from scipy.sparse.linalg import spsolve, factorized

from .idx_bus import BUS_TYPE, REF, BUS_I
from .makeBdc import makeBdc


def makePTDF(baseMVA, bus, branch, slack=None,
             result_side=0, using_sparse_solver=False, branch_id=None, reduced=False):
    """Builds the DC PTDF matrix for a given choice of slack.
    Returns the DC PTDF matrix for a given choice of slack. The matrix is
    C{nbr x nb}, where C{nbr} is the number of branches and C{nb} is the
    number of buses. The C{slack} can be a scalar (single slack bus) or an
    C{nb x 1} column vector of weights specifying the proportion of the
    slack taken up at each bus. If the C{slack} is not specified the
    reference bus is used by default.
    For convenience, C{slack} can also be an C{nb x nb} matrix, where each
    column specifies how the slack should be handled for injections
    at that bus.
    To restrict the PTDF computation to a subset of branches, supply a list of ppci branch indices in C{branch_id}.
    If C{reduced==True}, the output is reduced to the branches given in C{branch_id}, otherwise the complement rows are set to NaN.
    @see: L{makeLODF}
    @author: Ray Zimmerman (PSERC Cornell)
    """
    if reduced and branch_id is None:
        raise ValueError("'reduced=True' is only valid if branch_id is not None")

    ## use reference bus for slack by default
    if slack is None:
        slack = find(bus[:, BUS_TYPE] == REF)
        slack = slack[0]

    ## set the slack bus to be used to compute initial PTDF
    if isscalar(slack):
        slack_bus = slack
    else:
        slack_bus = 0      ## use bus 1 for temp slack bus

    nb = bus.shape[0]
    nbr = branch.shape[0]
    noref = arange(1, nb)      ## use bus 1 for voltage angle reference
    noslack = find(arange(nb) != slack_bus)

    ## check that bus numbers are equal to indices to bus (one set of bus numbers)
    if any(bus[:, BUS_I] != arange(nb)):
        stderr.write('makePTDF: buses must be numbered consecutively')

    if reduced:
        H = zeros((len(branch_id), nb))
    else:
        H = zeros((nbr, nb))
    # compute PTDF for single slack_bus
    if using_sparse_solver:
        Bbus, Bf, *_ = makeBdc(bus, branch, return_csr=False)

        Bbus = Bbus.real
        if result_side == 1:
            Bf *= -1
        if branch_id is not None:
            Bf = Bf.real.toarray()
            if reduced:
                H[:, noslack] = spsolve(Bbus[ix_(noslack, noref)].T, Bf[ix_(branch_id, noref)].T).T
            else:
                H[ix_(branch_id, noslack)] = spsolve(Bbus[ix_(noslack, noref)].T, Bf[ix_(branch_id, noref)].T).T
        elif Bf.shape[0] < 2000:
            Bf = Bf.real.toarray()
            H[:, noslack] = spsolve(Bbus[ix_(noslack, noref)].T, Bf[:, noref].T).T
        else:
            # Use memory saving modus
            Bbus_fact = factorized(Bbus[ix_(noslack, noref)].T)
            for i in range(0, Bf.shape[0], 32):
                H[i:i+32, noslack] = Bbus_fact(Bf[i:i+32, noref].real.toarray().T).T
    else:
        Bbus, Bf, *_ = makeBdc(bus, branch)
        Bbus, Bf = np.real(Bbus.toarray()), np.real(Bf.toarray())
        if result_side == 1:
            Bf *= -1
        if branch_id is not None:
            if reduced:
                H[:, noslack] = solve(Bbus[ix_(noslack, noref)].T, Bf[ix_(branch_id, noref)].T).T
            else:
                H[ix_(branch_id, noslack)] = solve(Bbus[ix_(noslack, noref)].T, Bf[ix_(branch_id, noref)].T).T
        else:
            H[:, noslack] = solve(Bbus[ix_(noslack, noref)].T, Bf[:, noref].T).T

    ## distribute slack, if requested
    if not isscalar(slack):
        if len(slack.shape) == 1:  ## slack is a vector of weights
            slack = slack / sum(slack)   ## normalize weights

            ## conceptually, we want to do ...
            ##    H = H * (eye(nb, nb) - slack * ones((1, nb)))
            ## ... we just do it more efficiently
            v = dot(H, slack)
            for k in range(nb):
                H[:, k] = H[:, k] - v
        else:
            H = dot(H, slack)

    return H
