# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Solves the power flow using a fast decoupled method.
"""

import sys

from numpy import array, angle, exp, linalg, conj, r_, Inf
from scipy.sparse.linalg import splu

from pandapower.pypower.ppoption import ppoption


def fdpf(Ybus, Sbus, V0, Bp, Bpp, ref, pv, pq, ppopt=None):
    """Solves the power flow using a fast decoupled method.

    Solves for bus voltages given the full system admittance matrix (for
    all buses), the complex bus power injection vector (for all buses),
    the initial vector of complex bus voltages, the FDPF matrices B prime
    and B double prime, and column vectors with the lists of bus indices
    for the swing bus, PV buses, and PQ buses, respectively. The bus voltage
    vector contains the set point for generator (including ref bus)
    buses, and the reference angle of the swing bus, as well as an initial
    guess for remaining magnitudes and angles. C{ppopt} is a PYPOWER options
    vector which can be used to set the termination tolerance, maximum
    number of iterations, and output options (see L{ppoption} for details).
    Uses default options if this parameter is not given. Returns the
    final complex voltages, a flag which indicates whether it converged
    or not, and the number of iterations performed.

    @see: L{runpf}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    if ppopt is None:
        ppopt = ppoption()

    ## options
    tol     = ppopt['PF_TOL']
    max_it  = ppopt['PF_MAX_IT_FD']
    verbose = ppopt['VERBOSE']

    ## initialize
    converged = 0
    i = 0
    V = V0
    Va = angle(V)
    Vm = abs(V)

    ## set up indexing for updating V
    #npv = len(pv)
    #npq = len(pq)
    pvpq = r_[pv, pq]

    ## evaluate initial mismatch
    mis = (V * conj(Ybus * V) - Sbus) / Vm
    P = mis[pvpq].real
    Q = mis[pq].imag

    ## check tolerance
    normP = linalg.norm(P, Inf)
    normQ = linalg.norm(Q, Inf)
    if verbose > 1:
        sys.stdout.write('\niteration     max mismatch (p.u.)  ')
        sys.stdout.write('\ntype   #        P            Q     ')
        sys.stdout.write('\n---- ----  -----------  -----------')
        sys.stdout.write('\n  -  %3d   %10.3e   %10.3e' % (i, normP, normQ))
    if normP < tol and normQ < tol:
        converged = 1
        if verbose > 1:
            sys.stdout.write('\nConverged!\n')

    ## reduce B matrices
    Bp = Bp[array([pvpq]).T, pvpq].tocsc() # splu requires a CSC matrix
    Bpp = Bpp[array([pq]).T, pq].tocsc()

    ## factor B matrices
    Bp_solver = splu(Bp)
    Bpp_solver = splu(Bpp)

    ## do P and Q iterations
    while (not converged and i < max_it):
        ## update iteration counter
        i = i + 1

        ##-----  do P iteration, update Va  -----
        dVa = -Bp_solver.solve(P)

        ## update voltage
        Va[pvpq] = Va[pvpq] + dVa
        V = Vm * exp(1j * Va)

        ## evalute mismatch
        mis = (V * conj(Ybus * V) - Sbus) / Vm
        P = mis[pvpq].real
        Q = mis[pq].imag

        ## check tolerance
        normP = linalg.norm(P, Inf)
        normQ = linalg.norm(Q, Inf)
        if verbose > 1:
            sys.stdout.write("\n  %s  %3d   %10.3e   %10.3e" %
                             (type,i, normP, normQ))
        if normP < tol and normQ < tol:
            converged = 1
            if verbose:
                sys.stdout.write('\nFast-decoupled power flow converged in %d '
                    'P-iterations and %d Q-iterations.\n' % (i, i - 1))
            break

        ##-----  do Q iteration, update Vm  -----
        dVm = -Bpp_solver.solve(Q)

        ## update voltage
        Vm[pq] = Vm[pq] + dVm
        V = Vm * exp(1j * Va)

        ## evalute mismatch
        mis = (V * conj(Ybus * V) - Sbus) / Vm
        P = mis[pvpq].real
        Q = mis[pq].imag

        ## check tolerance
        normP = linalg.norm(P, Inf)
        normQ = linalg.norm(Q, Inf)
        if verbose > 1:
            sys.stdout.write('\n  Q  %3d   %10.3e   %10.3e' % (i, normP, normQ))
        if normP < tol and normQ < tol:
            converged = 1
            if verbose:
                sys.stdout.write('\nFast-decoupled power flow converged in %d '
                    'P-iterations and %d Q-iterations.\n' % (i, i))
            break

    if verbose:
        if not converged:
            sys.stdout.write('\nFast-decoupled power flow did not converge in '
                             '%d iterations.' % i)

    return V, converged, i
