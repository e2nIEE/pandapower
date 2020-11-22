# -*- coding: utf-8 -*-

# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

"""Builds the line outage distribution factor matrix.
"""

from numpy import ones, diag, r_, arange
import numpy as np
from scipy.sparse import csr_matrix as sparse
try:
    from numba import jit
except ImportError: # pragma: no cover
    from pandapower.pf.no_numba import jit

from .idx_brch import F_BUS, T_BUS


@jit
def update_LODF_diag(LODF): # pragma: no cover
    for ix in range(LODF.shape[0]):
        # To preserve the data type of diagnol elments
        LODF[ix, ix] -= (LODF[ix, ix] + 1.)


def makeLODF(branch, PTDF):
    """Builds the line outage distribution factor matrix.

    Returns the DC line outage distribution factor matrix for a given PTDF.
    The matrix is C{nbr x nbr}, where C{nbr} is the number of branches.

    Example::
        H = makePTDF(baseMVA, bus, branch)
        LODF = makeLODF(branch, H)

    @see: L{makePTDF}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    nl, nb = PTDF.shape
    f = np.real(branch[:, F_BUS])
    t = np.real(branch[:, T_BUS])
    Cft = sparse((r_[ones(nl), -ones(nl)],
                      (r_[f, t], r_[arange(nl), arange(nl)])), (nb, nl))

    H = PTDF * Cft
    h = diag(H, 0)
    # Avoid zero division error
    # Implies a N-1 contingency (No backup branch)
    den = (ones((nl, 1)) * h.T * -1 + 1.)

    # Silence warning caused by np.NaN
    with np.errstate(divide='ignore', invalid='ignore'):
        LODF = (H / den)

    # LODF = LODF - diag(diag(LODF)) - eye(nl, nl)
    update_LODF_diag(LODF)

    return LODF
