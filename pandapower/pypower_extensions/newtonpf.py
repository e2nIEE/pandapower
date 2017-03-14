# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

"""Solves the power flow using a full Newton's method.
"""

from numpy import array, angle, exp, linalg, conj, r_, Inf, arange, zeros, float64, empty, int64

from scipy.sparse import issparse, csr_matrix as sparse
from scipy.sparse import hstack, vstack
from scipy.sparse.linalg import spsolve


def newtonpf(Ybus, Sbus, V0, pv, pq, options, numba):
    """Solves the power flow using a full Newton's method.

    Solves for bus voltages given the full system admittance matrix (for
    all buses), the complex bus power injection vector (for all buses),
    the initial vector of complex bus voltages, and column vectors with
    the lists of bus indices for the swing bus, PV buses, and PQ buses,
    respectively. The bus voltage vector contains the set point for
    generator (including ref bus) buses, and the reference angle of the
    swing bus, as well as an initial guess for remaining magnitudes and
    angles. C{ppopt} is a PYPOWER options vector which can be used to
    set the termination tolerance, maximum number of iterations, and
    output options (see L{ppoption} for details). Uses default options if
    this parameter is not given. Returns the final complex voltages, a
    flag which indicates whether it converged or not, and the number of
    iterations performed.

    @see: L{runpf}

    @author: Ray Zimmerman (PSERC Cornell)
    @author: Richard Lincoln

    Modified by University of Kassel (Florian Schaefer) to use numba
    """

    ## default arguments
    global dSbus_dV_calc, dSbus_dV_calc

    ## options
    tol     = options['tolerance_kva'] * 1e-3
    max_it  = options["max_iteration"]

    ## initialize
    i = 0
    V = V0
    Va = angle(V)
    Vm = abs(V)

    ## set up indexing for updating V
    pvpq = r_[pv, pq]
    # generate lookup pvpq -> index pvpq (used in createJ)
    pvpq_lookup = zeros(pvpq[-1]+1, dtype=int)
    pvpq_lookup[pvpq] = arange(len(pvpq))

    # if numba is available import "numba enhanced" functions
    if numba:
        from pandapower.pypower_extensions.create_J import create_J, create_J2
        from pandapower.pypower_extensions.dSbus_dV import dSbus_dV_calc
        from pandapower.pypower_extensions.dSbus_dV import dSbus_dV

        # check if pvpq is the same as pq. In this case a faster version of create_J can be used
        if len(pvpq) == len(pq):
            createJ = create_J2
        else:
            createJ = create_J
    else:
        # numba == False -> Import pypower standard
        from pypower.dSbus_dV import dSbus_dV

    npv = len(pv)
    npq = len(pq)
    j1 = 0;         j2 = npv           ## j1:j2 - V angle of pv buses
    j3 = j2;        j4 = j2 + npq      ## j3:j4 - V angle of pq buses
    j5 = j4;        j6 = j4 + npq      ## j5:j6 - V mag of pq buses

    ## evaluate F(x0)
    mis = V * conj(Ybus * V) - Sbus
    F = r_[  mis[pv].real,
             mis[pq].real,
             mis[pq].imag  ]

    # check for convergence
    converged = _check_for_convergence(F, tol)

    Ybus = Ybus.tocsr()
    ## do Newton iterations
    while (not converged and i < max_it):
        ## update iteration counter
        i = i + 1

        # use numba if activated
        if numba:
            #create Jacobian from fast calc of dS_dV
            dVm_x, dVa_x = dSbus_dV_calc(Ybus.data, Ybus.indptr, Ybus.indices, V, V/abs(V))

            # data in J, space preallocated is bigger than acutal Jx -> will be reduced later on
            Jx = empty(len(dVm_x) * 4, dtype=float64)
            # row pointer, dimension = pvpq.shape[0] + pq.shape[0] + 1
            Jp = zeros(pvpq.shape[0] + pq.shape[0] + 1, dtype=int64)
            # indices, same with the preallocated space (see Jx)
            Jj = empty(len(dVm_x) * 4, dtype=int64)

            # fill Jx, Jj and Jp
            createJ(dVm_x, dVa_x, Ybus.indptr, Ybus.indices, pvpq_lookup, pvpq, pq, Jx, Jj, Jp)

            # resize before generating the scipy sparse matrix
            Jx.resize(Jp[-1], refcheck=False)
            Jj.resize(Jp[-1], refcheck=False)

            # generate scipy sparse matrix
            dimJ = npv+npq+npq
            J = sparse((Jx, Jj, Jp), shape=(dimJ, dimJ))

        else:
            # create Jacobian with standard pypower implementation.
            dS_dVm, dS_dVa = dSbus_dV(Ybus, V)

            ## evaluate Jacobian
            J11 = dS_dVa[array([pvpq]).T, pvpq].real
            J12 = dS_dVm[array([pvpq]).T, pq].real
            J21 = dS_dVa[array([pq]).T, pvpq].imag
            J22 = dS_dVm[array([pq]).T, pq].imag

            J = vstack([
                hstack([J11, J12]),
                hstack([J21, J22])
            ], format="csr")

        ## compute update step
        dx = -1 * spsolve(J, F)

        ## update voltage
        if npv:
            Va[pv] = Va[pv] + dx[j1:j2]
        if npq:
            Va[pq] = Va[pq] + dx[j3:j4]
            Vm[pq] = Vm[pq] + dx[j5:j6]
        V = Vm * exp(1j * Va)
        Vm = abs(V)            ## update Vm and Va again in case
        Va = angle(V)          ## we wrapped around with a negative Vm

        ## evalute F(x)
        mis = V * conj(Ybus * V) - Sbus
        F = r_[  mis[pv].real,
                 mis[pq].real,
                 mis[pq].imag  ]

        ## check for convergence
        converged = _check_for_convergence(F, tol)

    return V, converged, i

def _check_for_convergence(F, tol):
    # calc infinity norm
    normF = linalg.norm(F, Inf)

    if normF < tol:
        return True
    else:
        return False