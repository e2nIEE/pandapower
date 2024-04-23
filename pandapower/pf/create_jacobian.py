# -*- coding: utf-8 -*-

# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


from numpy import complex128, float64, int64, r_
from numpy.core.multiarray import zeros, empty, array
from scipy.sparse import csr_matrix as sparse, vstack, hstack

from pandapower.pypower.dSbus_dV import dSbus_dV

try:
    # numba functions
    from pandapower.pf.create_jacobian_numba import create_J, create_J2, create_J_ds
    from pandapower.pf.dSbus_dV_numba import dSbus_dV_numba_sparse
except ImportError:
    pass


def _create_J_with_numba(Ybus, V, refpvpq, pvpq, pq, createJ, pvpq_lookup, nref, npv, npq, slack_weights, dist_slack):
    Ibus = zeros(len(V), dtype=complex128)
    # create Jacobian from fast calc of dS_dV
    dVm_x, dVa_x = dSbus_dV_numba_sparse(Ybus.data, Ybus.indptr, Ybus.indices, V, V / abs(V), Ibus)
    # data in J, space preallocated is bigger than acutal Jx -> will be reduced later on
    Jx = empty(len(dVm_x) * 4, dtype=float64)
    # row pointer, dimension = pvpq.shape[0] + pq.shape[0] + 1
    if dist_slack:
        Jp = zeros(refpvpq.shape[0] + pq.shape[0] + 1, dtype=int64)
    else:
        Jp = zeros(pvpq.shape[0] + pq.shape[0] + 1, dtype=int64)
    # indices, same with the preallocated space (see Jx)
    Jj = empty(len(dVm_x) * 4, dtype=int64)

    # fill Jx, Jj and Jp
    createJ(dVm_x, dVa_x, Ybus.indptr, Ybus.indices, pvpq_lookup, refpvpq, pvpq, pq, Jx, Jj, Jp, slack_weights)

    # resize before generating the scipy sparse matrix
    Jx.resize(Jp[-1], refcheck=False)
    Jj.resize(Jp[-1], refcheck=False)

    # todo: why not replace npv by pv.shape[0] etc.?
    # generate scipy sparse matrix
    if dist_slack:
        dimJ = nref + npv + npq + npq
    else:
        dimJ = npv + npq + npq
    J = sparse((Jx, Jj, Jp), shape=(dimJ, dimJ))

    return J


def _create_J_without_numba(Ybus, V, ref, pvpq, pq, slack_weights, dist_slack):
    # create Jacobian with standard pypower implementation.
    dS_dVm, dS_dVa = dSbus_dV(Ybus, V)

    ## evaluate Jacobian

    if dist_slack:
        rows_pvpq = array(r_[ref, pvpq]).T
        cols_pvpq = r_[ref[1:], pvpq]
        J11 = dS_dVa[rows_pvpq, :][:, cols_pvpq].real
        J12 = dS_dVm[rows_pvpq, :][:, pq].real
    else:
        rows_pvpq = array([pvpq]).T
        cols_pvpq = pvpq
        J11 = dS_dVa[rows_pvpq, cols_pvpq].real
        J12 = dS_dVm[rows_pvpq, pq].real
    if len(pq) > 0 or dist_slack:
        J21 = dS_dVa[array([pq]).T, cols_pvpq].imag
        J22 = dS_dVm[array([pq]).T, pq].imag
        if dist_slack:
            J10 = sparse(slack_weights[rows_pvpq].reshape(-1,1))
            J20 = sparse(zeros(shape=(len(pq), 1)))
            J = vstack([
                hstack([J10, J11, J12]),
                hstack([J20, J21, J22])
            ], format="csr")
        else:
            J = vstack([
                hstack([J11, J12]),
                hstack([J21, J22])
            ], format="csr")
    else:
        J = vstack([
            hstack([J11, J12])
        ], format="csr")
    return J


def create_jacobian_matrix(Ybus, V, ref, refpvpq, pvpq, pq, createJ, pvpq_lookup, nref, npv, npq, numba, slack_weights, dist_slack):
    if numba:
        J = _create_J_with_numba(Ybus, V, refpvpq, pvpq, pq, createJ, pvpq_lookup, nref, npv, npq, slack_weights, dist_slack)
    else:
        J = _create_J_without_numba(Ybus, V, ref, pvpq, pq, slack_weights, dist_slack)
    return J


def get_fastest_jacobian_function(pvpq, pq, numba, dist_slack):
    if numba:
        if dist_slack:
            create_jacobian = create_J_ds
        elif len(pvpq) == len(pq):
            create_jacobian = create_J2
        else:
            create_jacobian = create_J
    else:
        create_jacobian = None
    return create_jacobian
