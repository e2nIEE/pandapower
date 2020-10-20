# -*- coding: utf-8 -*-
# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Solves the power flow using a fast decoupled method.
"""
import warnings
from numpy import array, angle, exp, linalg, conj, r_, Inf, column_stack, real
from scipy.sparse.linalg import splu

from pandapower.pypower.makeSbus import makeSbus
from pandapower.pypower.makeB import makeB


def decoupledpf(Ybus, Sbus, V0, pv, pq, ppci, options):
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

    Modified to consider voltage_depend_loads
    """
    # old algortihm options to the new ones
    pp2pypower_algo = {'fdbx': 2, 'fdxb': 3}

    # options
    tol = options["tolerance_mva"]
    max_it = options["max_iteration"]
    # No use currently for numba. TODO: Check if can be applied in Bp and Bpp
    # numba = options["numba"]

    # NOTE: options["algorithm"] is either 'fdbx' or 'fdxb'. Otherwise, error
    algorithm = pp2pypower_algo[options["algorithm"]]

    voltage_depend_loads = options["voltage_depend_loads"]
    v_debug = options["v_debug"]

    baseMVA = ppci["baseMVA"]
    bus = ppci["bus"]
    branch = ppci["branch"]
    gen = ppci["gen"]

    # initialize
    i = 0
    V = V0
    Va = angle(V)
    Vm = abs(V)
    dVa, dVm = None, None

    if v_debug:
        Vm_it = Vm.copy()
        Va_it = Va.copy()
    else:
        Vm_it = None
        Va_it = None

    # set up indexing for updating V
    pvpq = r_[pv, pq]

    # evaluate initial mismatch
    P, Q = _evaluate_mis(Ybus, V, Sbus, pvpq, pq)

    # check tolerance
    converged = _check_for_convergence(P, Q, tol)

    # create and reduce B matrices
    Bp, Bpp = makeB(baseMVA, bus, real(branch), algorithm)
    # splu requires a CSC matrix
    Bp = Bp[array([pvpq]).T, pvpq].tocsc()
    Bpp = Bpp[array([pq]).T, pq].tocsc()

    # factor B matrices
    Bp_solver = splu(Bp)
    Bpp_solver = splu(Bpp)

    # do P and Q iterations
    while (not converged and i < max_it):
        # update iteration counter
        i = i + 1

        # -----  do P iteration, update Va  -----
        dVa = -Bp_solver.solve(P)

        # update voltage
        Va[pvpq] = Va[pvpq] + dVa
        V = Vm * exp(1j * Va)

        # evalute mismatch
        P, Q = _evaluate_mis(Ybus, V, Sbus, pvpq, pq)

        # check tolerance
        if _check_for_convergence(P, Q, tol):
            converged = True
            break

        # -----  do Q iteration, update Vm  -----
        dVm = -Bpp_solver.solve(Q)

        # update voltage
        Vm[pq] = Vm[pq] + dVm
        V = Vm * exp(1j * Va)

        if v_debug:
            Vm_it = column_stack((Vm_it, Vm))
            Va_it = column_stack((Va_it, Va))

        if voltage_depend_loads:
            Sbus = makeSbus(baseMVA, bus, gen, vm=Vm)

        # evalute mismatch
        P, Q = _evaluate_mis(Ybus, V, Sbus, pvpq, pq)

        # check tolerance
        if _check_for_convergence(P, Q, tol):
            converged = True
            break

    # the newtonpf/newtonpf funtion returns J. We are returning Bp and Bpp
    return V, converged, i, Bp, Bpp, Vm_it, Va_it


def _evaluate_mis(Ybus, V, Sbus, pvpq, pq):
    # evalute mis_p(x) and mis_q(x)
    mis = V * conj(Ybus * V) - Sbus
    mis_p, mis_q = mis[pvpq].real, mis[pq].imag
    return mis_p, mis_q


def _check_for_convergence(mis_p, mis_q, tol):
    # calc infinity norm
    return (
        (linalg.norm(mis_p, Inf) < tol) and (linalg.norm(mis_q, Inf) < tol)
    ) 
