# -*- coding: utf-8 -*-

# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

"""Solves the power flow using a full Newton's method.
"""

from numpy import array, angle, exp, linalg, conj, r_, Inf, arange, zeros, float64, empty, int32, max, complex128
from pandapower.pf.dSbus_dV_pypower import dSbus_dV
from scipy.sparse import hstack, vstack, csr_matrix as sparse
from scipy.sparse.linalg import spsolve

try:
    from pandapower.pf.create_J import create_J, create_J2
    from pandapower.pf.dSbus_dV_numba import dSbus_dV_numba_sparse
except ImportError:
    pass


def newtonpf(Ybus, Sbus, V0, pv, pq, options, Ibus=None):
    """Solves the power flow using a full Newton's method.

    Solves for bus voltages given the full system admittance matrix (for
    all buses), the complex bus power injection vector (for all buses),
    the initial vector of complex bus voltages, and column vectors with
    the lists of bus indices for the swing bus, PV buses, and PQ buses,
    respectively. The bus voltage vector contains the set point for
    generator (including ref bus) buses, and the reference angle of the
    swing bus, as well as an initial guess for remaining magnitudes and
    angles.

    @see: L{runpf}

    @author: Ray Zimmerman (PSERC Cornell)
    @author: Richard Lincoln

    Modified by University of Kassel (Florian Schaefer) to use numba
    """

    ## options
    tol = options['tolerance_kva'] * 1e-3
    max_it = options["max_iteration"]
    numba = options["numba"]

    ## initialize
    i = 0
    V = V0
    Va = angle(V)
    Vm = abs(V)

    Ibus = zeros(len(V)) if Ibus is None else Ibus

    ## set up indexing for updating V
    pvpq = r_[pv, pq]
    # generate lookup pvpq -> index pvpq (used in createJ)
    pvpq_lookup = zeros(max(Ybus.indices) + 1, dtype=int)
    pvpq_lookup[pvpq] = arange(len(pvpq))

    # import "numba enhanced" functions
    if numba:
        # check if pvpq is the same as pq. In this case a faster version of create_J can be used
        if len(pvpq) == len(pq):
            createJ = create_J2
        else:
            createJ = create_J

    npv = len(pv)
    npq = len(pq)
    j1 = 0
    j2 = npv  ## j1:j2 - V angle of pv buses
    j3 = j2
    j4 = j2 + npq  ## j3:j4 - V angle of pq buses
    j5 = j4
    j6 = j4 + npq  ## j5:j6 - V mag of pq buses

    ## evaluate F(x0)
    F = _evaluate_Fx(Ybus, V, Sbus, pv, pq, Ibus=Ibus)
    converged = _check_for_convergence(F, tol)

    Ybus = Ybus.tocsr()
    ## do Newton iterations
    while (not converged and i < max_it):
        ## update iteration counter
        i = i + 1

        # use numba if activated
        if numba:
            J = _create_J_with_numba(Ybus, V, pvpq, pq, createJ, pvpq_lookup, npv, npq, Ibus=Ibus)
        else:
            J = _create_J_without_numba(Ybus, V, pvpq, pq, Ibus=Ibus)

        dx = -1 * spsolve(J, F)
        ## update voltage
        if npv:
            Va[pv] = Va[pv] + dx[j1:j2]
        if npq:
            Va[pq] = Va[pq] + dx[j3:j4]
            Vm[pq] = Vm[pq] + dx[j5:j6]
        V = Vm * exp(1j * Va)
        Vm = abs(V)  ## update Vm and Va again in case
        Va = angle(V)  ## we wrapped around with a negative Vm

        F = _evaluate_Fx(Ybus, V, Sbus, pv, pq, Ibus=Ibus)
        converged = _check_for_convergence(F, tol)

    return V, converged, i


def _evaluate_Fx(Ybus, V, Sbus, pv, pq, Ibus=None):
    Ibus = zeros(len(V)) if Ibus is None else Ibus
    ## evalute F(x)
    mis = V * conj(Ybus * V - Ibus) - Sbus
    F = r_[mis[pv].real,
           mis[pq].real,
           mis[pq].imag]
    return F


def _create_J_with_numba(Ybus, V, pvpq, pq, createJ, pvpq_lookup, npv, npq, Ibus=None):

    Ibus = zeros(len(V), dtype=complex128) if Ibus is None else -Ibus
    # create Jacobian from fast calc of dS_dV
    dVm_x, dVa_x = dSbus_dV_numba_sparse(Ybus.data, Ybus.indptr, Ybus.indices, V, V / abs(V), Ibus)

    # data in J, space preallocated is bigger than acutal Jx -> will be reduced later on
    Jx = empty(len(dVm_x) * 4, dtype=float64)
    # row pointer, dimension = pvpq.shape[0] + pq.shape[0] + 1
    Jp = zeros(pvpq.shape[0] + pq.shape[0] + 1, dtype=int32)
    # indices, same with the preallocated space (see Jx)
    Jj = empty(len(dVm_x) * 4, dtype=int32)

    # fill Jx, Jj and Jp
    createJ(dVm_x, dVa_x, Ybus.indptr, Ybus.indices, pvpq_lookup, pvpq, pq, Jx, Jj, Jp)

    # resize before generating the scipy sparse matrix
    Jx.resize(Jp[-1], refcheck=False)
    Jj.resize(Jp[-1], refcheck=False)

    # generate scipy sparse matrix
    dimJ = npv + npq + npq
    J = sparse((Jx, Jj, Jp), shape=(dimJ, dimJ))

    return J


def _create_J_without_numba(Ybus, V, pvpq, pq, Ibus=None):
    Ibus = zeros(len(V)) if Ibus is None else Ibus
    # create Jacobian with standard pypower implementation.
    dS_dVm, dS_dVa = dSbus_dV(Ybus, V, I=Ibus)

    ## evaluate Jacobian
    J11 = dS_dVa[array([pvpq]).T, pvpq].real
    J12 = dS_dVm[array([pvpq]).T, pq].real
    J21 = dS_dVa[array([pq]).T, pvpq].imag
    J22 = dS_dVm[array([pq]).T, pq].imag

    J = vstack([
        hstack([J11, J12]),
        hstack([J21, J22])
    ], format="csr")

    return J


def _check_for_convergence(F, tol):
    # calc infinity norm
    return linalg.norm(F, Inf) < tol
