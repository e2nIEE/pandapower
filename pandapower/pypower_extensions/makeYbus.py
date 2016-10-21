# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a 
# BSD-style license that can be found in the LICENSE file.


from sys import stderr

from numpy import ones, conj, nonzero, any, exp, pi, r_, argsort, resize, empty, complex128, zeros, int64, array
from scipy.sparse import csr_matrix

from numba import jit

from pypower.idx_bus import BUS_I, GS, BS
from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, BR_STATUS, SHIFT, TAP

@jit(nopython=True, cache=True)
def gen_Ybus(Yf_x, Yt_x, Ysh, col_Y, f, t, f_sort, t_sort, nb, nl, r_nl):
    """
    Fast calculation of Ybus
    """

    r_nb = range(nb)

    # allocate data of Ybus in CSR format
    # Note: More space is allocated than needed with empty.
    #       The matrix size will be reduced afterwards
    Yx = empty(nb * 5, dtype=complex128) # data
    Yp = zeros(nb + 1, dtype=int64) # row pointer
    Yj = empty(nb * 5, dtype=int64) # colum indices

    # index iterators
    # a = iterator of f, b = iterator of t, curRow = current Row
    a, b, curRow = 0, 0, 0
    # number of nonzeros (total), number of nonzeros per row
    nnz, nnz_row = 0, 0
    # flag checks if diagonal entry was added
    YshAdded = False

    for curRow in r_nb:
        nnz_row = 0
        # iterate rows of Ybus

        # add entries from Yf
        while a < nl and f[f_sort[a]] == curRow:
            # Entries from f_sort[a] in current row of Ybus
            for col in (r_nl[f_sort[a]], r_nl[f_sort[a]] + nl):
                # 'Has entry at column in Yf: %i ' % col
                if col_Y[col] == curRow and not YshAdded:
                    # add Ysh and Yf_x (diagonal element). If not already added
                    curVal = Yf_x[col] + Ysh[curRow]
                    YshAdded = True
                else:
                    # add only Yf_x
                    curVal = Yf_x[col]

                for k in range(Yp[curRow], Yp[curRow] + nnz_row):
                    if col_Y[col] == Yj[k]:
                        # if entry at column already exists add value
                        Yx[k] += curVal
                        break
                else:
                    # new entry in Ybus
                    Yx[nnz] = curVal
                    Yj[nnz] = col_Y[col]
                    nnz += 1
                    nnz_row += 1
            a += 1

        # add entries from Yt
        while b < nl and t[t_sort[b]] == curRow:
            # Entries from t_sort[b] in current row of Ybus
            for col in (r_nl[t_sort[b]], r_nl[t_sort[b]] + nl):
                # 'Has entry at column in Yt: %i ' % col
                if col_Y[col] == curRow and not YshAdded:
                    # add Ysh and Yf_x (diagonal element). If not already added
                    curVal = Yt_x[col] + Ysh[curRow]
                    YshAdded = True
                else:
                    # add only Yt_x
                    curVal = Yt_x[col]

                for k in range(Yp[curRow], Yp[curRow] + nnz_row):
                    if col_Y[col] == Yj[k]:
                        # if entry at column already exists add value
                        Yx[k] += curVal
                        break
                else:
                    # new entry in Ybus
                    Yx[nnz] = curVal
                    Yj[nnz] = col_Y[col]
                    nnz += 1
                    nnz_row += 1
            b += 1

        if not YshAdded:
            # check if diagonal entry was added. If not -> add if not zero
            if Ysh[curRow]:
                Yx[nnz] = Ysh[curRow]
                Yj[nnz] = curRow
                nnz += 1
                nnz_row += 1

        YshAdded = False
        # add number of nonzeros in row to row pointer
        Yp[curRow + 1] = nnz_row + Yp[curRow]
        curRow += 1

    return Yx, Yj, Yp, nnz

def makeYbus(baseMVA, bus, branch):
    """Builds the bus admittance matrix and branch admittance matrices.

    Returns the full bus admittance matrix (i.e. for all buses) and the
    matrices C{Yf} and C{Yt} which, when multiplied by a complex voltage
    vector, yield the vector currents injected into each line from the
    "from" and "to" buses respectively of each line. Does appropriate
    conversions to p.u.

    @see: L{makeSbus}

    @author: Ray Zimmerman (PSERC Cornell)
    @author: Richard Lincoln

    modified by Florian Schaefer (to use numba) (florian.schaefer@uni-kassel.de)
    """
    ## constants
    nb = bus.shape[0]          ## number of buses
    nl = branch.shape[0]       ## number of lines

    ## check that bus numbers are equal to indices to bus (one set of bus nums)
    if any(bus[:, BUS_I] != list(range(nb))):
        stderr.write('buses must appear in order by bus number\n')

    ## for each branch, compute the elements of the branch admittance matrix where
    ##
    ##      | If |   | Yff  Yft |   | Vf |
    ##      |    | = |          | * |    |
    ##      | It |   | Ytf  Ytt |   | Vt |
    ##
    stat = branch[:, BR_STATUS]              ## ones at in-service branches
    Ys = stat / (branch[:, BR_R] + 1j * branch[:, BR_X])  ## series admittance
    Bc = stat * branch[:, BR_B]              ## line charging susceptance
    tap = ones(nl)                           ## default tap ratio = 1
    i = nonzero(branch[:, TAP])              ## indices of non-zero tap ratios
    tap[i] = branch[i, TAP]                  ## assign non-zero tap ratios
    tap = tap * exp(1j * pi / 180 * branch[:, SHIFT]) ## add phase shifters

    Ytt = Ys + 1j * Bc / 2
    Yff = Ytt / (tap * conj(tap))
    Yft = - Ys / conj(tap)
    Ytf = - Ys / tap

    ## compute shunt admittance
    ## if Psh is the real power consumed by the shunt at V = 1.0 p.u.
    ## and Qsh is the reactive power injected by the shunt at V = 1.0 p.u.
    ## then Psh - j Qsh = V * conj(Ysh * V) = conj(Ysh) = Gs - j Bs,
    ## i.e. Ysh = Psh + j Qsh, so ...
    ## vector of shunt admittances
    Ysh = (bus[:, GS] + 1j * bus[:, BS]) / baseMVA

    ## build connection matrices
    f = branch[:, F_BUS].astype(int)                           ## list of "from" buses
    t = branch[:, T_BUS].astype(int)                           ## list of "to" buses

    ## build Yf and Yt such that Yf * V is the vector of complex branch currents injected
    ## at each branch's "from" bus, and Yt is the same for the "to" bus end
    i = r_[range(nl), range(nl)]                   ## double set of row indices

    Yf_x = r_[Yff, Yft]
    Yt_x = r_[Ytf, Ytt]
    col_Y = r_[f, t]

    Yf = csr_matrix((Yf_x, (i, col_Y)), (nl, nb))
    Yt = csr_matrix((Yt_x, (i, col_Y)), (nl, nb))

    Yx, Yj, Yp, nnz = gen_Ybus(Yf_x, Yt_x, Ysh, col_Y, f, t, argsort(f), argsort(t), nb, nl,
                               array(range(nl), dtype=int64))
    Ybus = csr_matrix((resize(Yx, nnz), resize(Yj, nnz), Yp))


    return Ybus, Yf, Yt