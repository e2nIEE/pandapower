# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Evaluates polynomial generator cost & derivatives.
"""

import sys

from numpy import zeros, arange, flatnonzero as find, int64

from pandapower.pypower.idx_cost import MODEL, NCOST, PW_LINEAR, COST


def polycost(gencost, Pg, der=0):
    """Evaluates polynomial generator cost & derivatives.

    C{f = polycost(gencost, Pg)} returns the vector of costs evaluated at C{Pg}

    C{df = polycost(gencost, Pg, 1)} returns the vector of first derivatives
    of costs evaluated at C{Pg}

    C{d2f = polycost(gencost, Pg, 2)} returns the vector of second derivatives
    of costs evaluated at C{Pg}

    C{gencost} must contain only polynomial costs
    C{Pg} is in MW, not p.u. (works for C{Qg} too)

    @author: Ray Zimmerman (PSERC Cornell)
    """
    if any(gencost[:, MODEL] == PW_LINEAR):
        sys.stderr.write('polycost: all costs must be polynomial\n')

    ng = len(Pg)
    maxN = max(gencost[:, NCOST].astype(int64))
    minN = min(gencost[:, NCOST].astype(int64))

    ## form coefficient matrix where 1st column is constant term, 2nd linear, etc.
    c = zeros((ng, maxN))
    for n in arange(minN, maxN + 1):
        k = find(gencost[:, NCOST] == n)   ## cost with n coefficients
        c[k, :n] = gencost[k, (COST + n - 1):COST - 1:-1]

    ## do derivatives
    for d in range(1, der + 1):
        if c.shape[1] >= 2:
            c = c[:, 1:maxN - d + 1]
        else:
            c = zeros((ng, 1))
            break

        for k in range(2, maxN - d + 1):
            c[:, k-1] = c[:, k-1] * k

    ## evaluate polynomial
    if len(c) == 0:
        f = zeros(Pg.shape)
    else:
        f = c[:, :1].flatten()  ## constant term
        for k in range(1, c.shape[1]):
            f = f + c[:, k] * Pg**k

    return f
