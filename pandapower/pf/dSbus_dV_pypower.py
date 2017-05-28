# -*- coding: utf-8 -*-

# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.


"""Computes partial derivatives of power injection w.r.t. voltage.
"""

from numpy import conj, diag, asmatrix, asarray, zeros
from scipy.sparse import issparse, csr_matrix as sparse


def dSbus_dV(Ybus, V, I=None):
    """Computes partial derivatives of power injection w.r.t. voltage.
    """

    I = zeros(len(V)) if I is None else I
    if issparse(Ybus):
        return dSbus_dV_sparse(Ybus, V, I)
    else:
        return dSbus_dV_dense(Ybus, V, I)


def dSbus_dV_sparse(Ybus, V, I):
    Ibus = Ybus * V - I
    ib = range(len(V))
    diagV = sparse((V, (ib, ib)))
    diagIbus = sparse((Ibus, (ib, ib)))
    diagVnorm = sparse((V / abs(V), (ib, ib)))
    dS_dVm = diagV * conj(Ybus * diagVnorm) + conj(diagIbus) * diagVnorm
    dS_dVa = 1j * diagV * conj(diagIbus - Ybus * diagV)
    return dS_dVm, dS_dVa


def dSbus_dV_dense(Ybus, V, I):
    # standard code from Pypower (slower than above)
    Ibus = Ybus * asmatrix(V).T - asmatrix(I).T

    diagV = asmatrix(diag(V))
    diagIbus = asmatrix(diag(asarray(Ibus).flatten()))
    diagVnorm = asmatrix(diag(V / abs(V)))

    dS_dVm = diagV * conj(Ybus * diagVnorm) + conj(diagIbus) * diagVnorm
    dS_dVa = 1j * diagV * conj(diagIbus - Ybus * diagV)
    return dS_dVm, dS_dVa
