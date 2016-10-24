# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a 
# BSD-style license that can be found in the LICENSE file.

from numpy import conj, zeros, complex128, diag, asmatrix, asarray
from scipy.sparse import issparse, csr_matrix as sparse

from numba import jit, i4, c16
from numba.types import Tuple

#@jit(Tuple((c16[:], c16[:]))(c16[:], i4[:], i4[:], c16[:], c16[:]), nopython=True, cache=True)
@jit(nopython=True, cache=True)
def dSbus_dV_calc(Yx, Yp, Yj, V, Vnorm):
    """Computes partial derivatives of power injection w.r.t. voltage.

    Calculates faster with numba and sparse matrices.
    
    Input: Ybus in CSR sparse form (Yx = data, Yp = indptr, Yj = indices), V and Vnorm (= V / abs(V))
    
    Output: data from CSR form of dS_dVm, dS_dVa 
    (index pointer and indices are the same as the ones from Ybus)

    Translation of: dS_dVm = dS_dVm = diagV * conj(Ybus * diagVnorm) + conj(diagIbus) * diagVnorm
                             dS_dVa = 1j * diagV * conj(diagIbus - Ybus * diagV)
    """

    # transform input

    # init buffer vector
    buffer = zeros(len(V), dtype=complex128)
    Ibus = zeros(len(V), dtype=complex128)
    dS_dVm = Yx.copy()
    dS_dVa = Yx.copy()

    # iterate through sparse matrix
    for r in range(len(Yp) - 1):
        for k in range(Yp[r], Yp[r+1]):
            # Ibus = Ybus * V
            buffer[r] += Yx[k] * V[Yj[k]]
            # Ybus * diag(Vnorm)
            dS_dVm[k] *= Vnorm[Yj[k]]
            # Ybus * diag(V)
            dS_dVa[k] *= V[Yj[k]]

        Ibus[r] = buffer[r]
        # conj(diagIbus) * diagVnorm
        buffer[r] = conj(buffer[r]) * Vnorm[r]

    for r in range(len(Yp) - 1):
        for k in range(Yp[r], Yp[r+1]):
            if r == Yj[k]:
                dS_dVa[k] = -Ibus[r] + dS_dVa[k]

            dS_dVm[k] = conj(dS_dVm[k])
            dS_dVa[k] = conj(-dS_dVa[k])

            # diag(Vnorm) * conj(Ybus * diagVnorm)
            dS_dVm[k] *= V[r]
            # 1j * diagV * conj(diagIbus - Ybus * diagV)
            dS_dVa[k] *= (1j * V[r])
            if r == Yj[k]:
                # diagV * conj(Ybus * diagVnorm) + conj(diagIbus) * diagVnorm
                dS_dVm[k] += buffer[r]

    return dS_dVm, dS_dVa


def dSbus_dV(Ybus, V):
    """
    Calls functions to calculate dS/dV depending on whether Ybus is sparse or not
    """

    if issparse(Ybus):
        # calculates sparse data
        dS_dVm, dS_dVa = dSbus_dV_calc(Ybus.data, Ybus.indptr, Ybus.indices, V, V / abs(V))
        # generate sparse CSR matrices with computed data and return them
        return sparse((dS_dVm, Ybus.indices, Ybus.indptr)), sparse((dS_dVa, Ybus.indices, Ybus.indptr))
    else:
        # standard code from Pypower (slower than above)
        Ibus = Ybus * asmatrix(V).T

        diagV = asmatrix(diag(V))
        diagIbus = asmatrix(diag( asarray(Ibus).flatten() ))
        diagVnorm = asmatrix(diag(V / abs(V)))

        dS_dVm = diagV * conj(Ybus * diagVnorm) + conj(diagIbus) * diagVnorm
        dS_dVa = 1j * diagV * conj(diagIbus - Ybus * diagV)
        return dS_dVm, dS_dVa
