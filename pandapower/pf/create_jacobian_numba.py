# -*- coding: utf-8 -*-

# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

# DEBUG = False
# DEBUG = True

from numba import jit
# import numba as nb

# numpy is only used for debugging
# if DEBUG:
#     from numpy import zeros


# @jit(i8(c16[:], c16[:], i4[:], i4[:], i8[:], i8[:], f8[:], i8[:], i8[:]), nopython=True, cache=False)
@jit(nopython=True, cache=False)
def create_J(dVm_x, dVa_x, Yp, Yj, pvpq_lookup, refpvpq, pvpq, pq, Jx, Jj, Jp, slack_weights):  # pragma: no cover
    """Calculates Jacobian faster with numba and sparse matrices.

        Input: dS_dVa and dS_dVm in CSR sparse form (Yx = data, Yp = indptr, Yj = indices), pvpq, pq from pypower

        OUTPUT:  data from CSR form of Jacobian (Jx, Jj, Jp) and number of non zeros (nnz)

        @author: Florian Schaefer

        Calculate Jacobian entries

        J11 = dS_dVa[array([pvpq]).T, pvpq].real
        J12 = dS_dVm[array([pvpq]).T, pq].real
        J21 = dS_dVa[array([pq]).T, pvpq].imag
        J22 = dS_dVm[array([pq]).T, pq].imag

        Explanation of code:
        To understand the concept the CSR storage method should be known. See:
        https://de.wikipedia.org/wiki/Compressed_Row_Storage

        J has the shape
        | J11 | J12 |               | (pvpq, pvpq) | (pvpq, pq) |
        | --------- | = dimensions: | ------------------------- |
        | J21 | J22 |               |  (pq, pvpq)  |  (pq, pq)  |

        We first iterate the rows of J11 and J12 (for r in range lpvpq) and add the entries which are stored in dS_dV
        Then we iterate the rows of J21 and J22 (for r in range lpq) and add the entries from dS_dV

        Note: The row and column pointer of of dVm and dVa are the same as the one from Ybus
    """
    # Jacobi Matrix in sparse form
    # Jp, Jx, Jj equal J like:
    # J = zeros(shape=(ndim, ndim), dtype=float64)

    # get length of vectors
    lpvpq = len(pvpq)
    lpq = len(pq)
    lpv = lpvpq - lpq

    # nonzeros in J
    nnz = 0

    # iterate rows of J
    # first iterate pvpq (J11 and J12)
    for r in range(lpvpq):
        # nnzStar is necessary to calculate nonzeros per row
        nnzStart = nnz
        # iterate columns of J11 = dS_dVa.real at positions in pvpq
        # check entries in row pvpq[r] of dS_dV
        for c in range(Yp[pvpq[r]], Yp[pvpq[r] + 1]):
            # check if column Yj is in pvpq
            cc = pvpq_lookup[Yj[c]]
            # entries for J11 and J12
            if pvpq[cc] == Yj[c]:
                # entry found
                # equals entry of J11: J[r,cc] = dVa_x[c].real
                Jx[nnz] = dVa_x[c].real
                Jj[nnz] = cc
                nnz += 1
                # if entry is found in the "pq part" of pvpq = add entry of J12
                if cc >= lpv:
                    Jx[nnz] = dVm_x[c].real
                    Jj[nnz] = cc + lpq
                    nnz += 1
        # Jp: number of nonzeros per row = nnz - nnzStart (nnz at begging of loop - nnz at end of loop)
        Jp[r + 1] = nnz - nnzStart + Jp[r]
    # second: iterate pq (J21 and J22)
    for r in range(lpq):
        nnzStart = nnz
        # iterate columns of J21 = dS_dVa.imag at positions in pvpq
        for c in range(Yp[pq[r]], Yp[pq[r] + 1]):
            cc = pvpq_lookup[Yj[c]]
            if pvpq[cc] == Yj[c]:
                # entry found
                # equals entry of J21: J[r + lpvpq, cc] = dVa_x[c].imag
                Jx[nnz] = dVa_x[c].imag
                Jj[nnz] = cc
                nnz += 1
                if cc >= lpv:
                    # if entry is found in the "pq part" of pvpq = Add entry of J22
                    Jx[nnz] = dVm_x[c].imag
                    Jj[nnz] = cc + lpq
                    nnz += 1
        # Jp: number of nonzeros per row = nnz - nnzStart (nnz at begging of loop - nnz at end of loop)
        Jp[r + lpvpq + 1] = nnz - nnzStart + Jp[r + lpvpq]


# @jit(i8(c16[:], c16[:], i4[:], i4[:], i8[:], i8[:], f8[:], i8[:], i8[:]), nopython=True, cache=True)
@jit(nopython=True, cache=False)
def create_J2(dVm_x, dVa_x, Yp, Yj, pvpq_lookup, refpvpq, pvpq, pq, Jx, Jj, Jp, slack_weights):  # pragma: no cover
    """Calculates Jacobian faster with numba and sparse matrices. This version is similar to create_J except that
        if pvpq = pq (when no pv bus is available) some if statements are obsolete and J11 = J12 and J21 = J22

        Input: dS_dVa and dS_dVm in CSR sparse form (Yx = data, Yp = indptr, Yj = indices), pvpq, pq from pypower

        OUTPUT: data from CSR form of Jacobian (Jx, Jj, Jp) and number of non zeros (nnz)

        @author: Florian Schaefer
        @date: 30.08.2016

        see comments in create_J
    """
    # Jacobi Matrix in sparse form
    # Jp, Jx, Jj equal J like:
    # J = zeros(shape=(ndim, ndim), dtype=float64)

    # get info of vector
    lpvpq = len(pvpq)

    # nonzeros in J
    nnz = 0

    # iterate rows of J
    # first iterate pvpq (J11 and J12)
    for r in range(lpvpq):
        # nnzStart is necessary to calculate nonzeros per row
        nnzStart = nnz
        # iterate columns of J11 = dS_dVa.real at positions in pvpq
        # iterate columns of J12 = dS_dVm.real at positions in pq (=pvpq)
        for c in range(Yp[pvpq[r]], Yp[pvpq[r] + 1]):
            cc = pvpq_lookup[Yj[c]]
            if pvpq[cc] == Yj[c]:
                # entry found J11
                # J[r,cc] = dVa_x[c].real
                Jx[nnz] = dVa_x[c].real
                Jj[nnz] = cc
                nnz += 1
                # also entry in J12
                Jx[nnz] = dVm_x[c].real
                Jj[nnz] = cc + lpvpq
                nnz += 1
        # Jp: number of nonzeros per row = nnz - nnzStart (nnz at begging of loop - nnz at end of loop)
        Jp[r + 1] = nnz - nnzStart + Jp[r]
    # second: iterate pq (J21 and J22)
    for r in range(lpvpq):
        nnzStart = nnz
        # iterate columns of J21 = dS_dVa.imag at positions in pvpq
        # iterate columns of J22 = dS_dVm.imag at positions in pq (=pvpq)
        for c in range(Yp[pvpq[r]], Yp[pvpq[r] + 1]):
            cc = pvpq_lookup[Yj[c]]
            if pvpq[cc] == Yj[c]:
                # entry found J21
                # J[r + lpvpq, cc] = dVa_x[c].imag
                Jx[nnz] = dVa_x[c].imag
                Jj[nnz] = cc
                nnz += 1
                # also entry in J22
                Jx[nnz] = dVm_x[c].imag
                Jj[nnz] = cc + lpvpq
                nnz += 1
        # Jp: number of nonzeros per row = nnz - nnzStart (nnz at begging of loop - nnz at end of loop)
        Jp[r + lpvpq + 1] = nnz - nnzStart + Jp[r + lpvpq]


@jit(nopython=True, cache=False)
def create_J_ds(dVm_x, dVa_x, Yp, Yj, pvpq_lookup, refpvpq, pvpq, pq, Jx, Jj, Jp, slack_weights):  # pragma: no cover
    """Calculates Jacobian faster with numba and sparse matrices.

        Input: dS_dVa and dS_dVm in CSR sparse form (Yx = data, Yp = indptr, Yj = indices), pvpq, pq from pypower

        OUTPUT:  data from CSR form of Jacobian (Jx, Jj, Jp) and number of non zeros (nnz)

        @author: Roman Bolgaryn

        Calculate Jacobian entries

        J11 = dS_dVa[array([pvpq]).T, pvpq].real
        J12 = dS_dVm[array([pvpq]).T, pq].real
        J13 = slack_weights.reshape(-1,1)
        J21 = dS_dVa[array([pq]).T, pvpq].imag
        J22 = dS_dVm[array([pq]).T, pq].imag
        J23 = zeros(shape=(len(pq), 1))

        J has the shape
        | J11 | J12 | J13 |               | (ref+pvpq, pvpq) | (ref+pvpq, pq) | (ref+pvpq, 1) |
        | --------------- | = dimensions: | ------------------------------------------------- |
        | J21 | J22 | J23 |               |    (pq, pvpq)    |    (pq, pq)    |    (pq, 1)    |

        In addition to CreateJ, we add a row and a column
        We first iterate the rows of J11 and J12 (for r in range lpvpq) and add the entries which are stored in dS_dV
        We add the entries for slack weights (J13) if we encounter a positive slack weight in the row while iterating through J11 and J12
        Then we iterate the rows of J21 and J22 (for r in range lpq) and add the entries from dS_d
        The entries of J23 are always 0, so we do not do anything for J23 here (will be 0 because of the sparse matrix)

    """
    # Jacobi Matrix in sparse form
    # Jp, Jx, Jj equal J like:
    # J = zeros(shape=(ndim, ndim), dtype=float64)

    # Yp = Ybus.indptr (maps the elements of data and indices to the rows of the sparse matrix)
    # Yj = Ybus.indices (array mapping each element in data to its column in the sparse matrix)

    # get length of vectors
    lrefpvpq = len(refpvpq)
    lpvpq = len(pvpq)
    lpq = len(pq)
    lpv = lpvpq - lpq
    lrefpv = lrefpvpq - lpq
    lref = lrefpv - lpv

    # dense J matrix for debugging
    # if DEBUG:
    #     JJ = zeros(shape=(lrefpvpq+lpq, lrefpvpq+lpq))

    # nonzeros in J
    nnz = 0
    # iterate rows of J
    # first iterate pvpq (J11 and J12) -> the real part
    for r in range(lrefpvpq):
        # nnzStar is necessary to calculate nonzeros per row
        nnzStart = nnz
        # add slack weights to the first column
        if slack_weights[refpvpq[r]] > 0:
            Jx[nnz] = slack_weights[refpvpq[r]]
            Jj[nnz] = 0
            # if DEBUG:
            #     JJ[r, 0] = Jx[nnz]
            nnz += 1
        # iterate columns of J11 = dS_dVa.real at positions in pvpq
        # check entries in row pvpq[r] of dS_dV
        # r_start, r_end = Yp[refpvpq[r]], Yp[refpvpq[r] + 1]
        # for c in range(r_start, r_end):
        for c in range(Yp[refpvpq[r]], Yp[refpvpq[r] + 1]):
            # check if column Yj is in pvpq
            bus_idx = Yj[c]
            cc = pvpq_lookup[bus_idx]
            # entries for J11 and J12
            lookup_idx = refpvpq[cc]
            # skip if in the "ref" columns:
            skip = False
            for refcol in range(lref): # this loop is way faster
                if lookup_idx == refpvpq[refcol]:
                    skip = True
                    break
            if lookup_idx == bus_idx and not skip:# and lookup_idx in pvpq:  # too slow
                # entry found
                # equals entry of J11: J[r,cc] = dVa_x[c].real
                col = cc
                Jx[nnz] = dVa_x[c].real
                Jj[nnz] = col
                # if DEBUG:
                #     JJ[r, col] = Jx[nnz]
                nnz += 1
                # if entry is found in the "pq part" of pvpq = add entry of J12
                if cc >= lrefpv:
                    col = cc + lpq
                    Jx[nnz] = dVm_x[c].real
                    Jj[nnz] = col
                    # if DEBUG:
                    #     JJ[r, col] = Jx[nnz]
                    nnz += 1
        # Jp: number of nonzeros per row = nnz - nnzStart (nnz at begging of loop - nnz at end of loop)
        Jp[r + 1] = nnz - nnzStart + Jp[r]

    # second: iterate pq (J21 and J22) -> the imaginary part
    for r in range(lpq):
        nnzStart = nnz
        # iterate columns of J21 = dS_dVa.imag at positions in pvpq
        # r_start, r_end = Yp[pq[r]], Yp[pq[r] + 1]
        # for c in range(r_start, r_end):
        for c in range(Yp[pq[r]], Yp[pq[r] + 1]):
            bus_idx = Yj[c]
            cc = pvpq_lookup[bus_idx]
            lookup_idx = refpvpq[cc]
            skip = False
            for refcol in range(lref): # this loop is way faster
                if lookup_idx == refpvpq[refcol]:
                    skip = True
                    break
            if lookup_idx == bus_idx and not skip:# and lookup_idx in pvpq:   # too slow
                # entry found
                # equals entry of J21: J[r + lpvpq, cc] = dVa_x[c].imag
                col = cc
                Jx[nnz] = dVa_x[c].imag
                Jj[nnz] = col
                # if DEBUG:
                #     JJ[r+lrefpvpq, col] = Jx[nnz]
                nnz += 1
                # if entry is found in the "pq part" of pvpq = Add entry of J22
                if cc >= lrefpv:
                    col = cc + lpq
                    Jx[nnz] = dVm_x[c].imag
                    Jj[nnz] = col
                    # if DEBUG:
                    #     JJ[r+lrefpvpq, col] = Jx[nnz]
                    nnz += 1
        # Jp: number of nonzeros per row = nnz - nnzStart (nnz at begging of loop - nnz at end of loop)
        Jp[r + lrefpvpq + 1] = nnz - nnzStart + Jp[r + lrefpvpq]
