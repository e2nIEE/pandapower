# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Checks for P-Q capability curve constraints.
"""

from sys import stderr

from numpy import any, zeros, nonzero

from pandapower.pypower.idx_gen import QMAX, QMIN, PMAX, PC1, PC2, QC1MIN, QC1MAX, QC2MIN, QC2MAX


def hasPQcap(gen, hilo='B'):
    """Checks for P-Q capability curve constraints.

    Returns a column vector of 1's and 0's. The 1's correspond to rows of
    the C{gen} matrix which correspond to generators which have defined a
    capability curve (with sloped upper and/or lower bound on Q) and require
    that additional linear constraints be added to the OPF.

    The C{gen} matrix in version 2 of the PYPOWER case format includes columns
    for specifying a P-Q capability curve for a generator defined as the
    intersection of two half-planes and the box constraints on P and Q. The
    two half planes are defined respectively as the area below the line
    connecting (Pc1, Qc1max) and (Pc2, Qc2max) and the area above the line
    connecting (Pc1, Qc1min) and (Pc2, Qc2min).

    If the optional 2nd argument is 'U' this function returns C{True} only for
    rows corresponding to generators that require the upper constraint on Q.
    If it is 'L', only for those requiring the lower constraint. If the 2nd
    argument is not specified or has any other value it returns true for rows
    corresponding to gens that require either or both of the constraints.

    It is smart enough to return C{True} only if the corresponding linear
    constraint is not redundant w.r.t the box constraints.

    @author: Ray Zimmerman (PSERC Cornell)
    """
    ## check for errors capability curve data
    if any( gen[:, PC1] > gen[:, PC2] ):
        stderr.write('hasPQcap: Pc1 > Pc2\n')
    if any( gen[:, QC2MAX] > gen[:, QC1MAX] ):
        stderr.write('hasPQcap: Qc2max > Qc1max\n')
    if any( gen[:, QC2MIN] < gen[:, QC1MIN] ):
        stderr.write('hasPQcap: Qc2min < Qc1min\n')

    L = zeros(gen.shape[0], bool)
    U = zeros(gen.shape[0], bool)
    k = nonzero( gen[:, PC1] != gen[:, PC2] )

    if hilo != 'U':       ## include lower constraint
        Qmin_at_Pmax = gen[k, QC1MIN] + (gen[k, PMAX] - gen[k, PC1]) * \
            (gen[k, QC2MIN] - gen[k, QC1MIN]) / (gen[k, PC2] - gen[k, PC1])
        L[k] = Qmin_at_Pmax > gen[k, QMIN]

    if hilo != 'L':       ## include upper constraint
        Qmax_at_Pmax = gen[k, QC1MAX] + (gen[k, PMAX] - gen[k, PC1]) * \
            (gen[k, QC2MAX] - gen[k, QC1MAX]) / (gen[k, PC2] - gen[k, PC1])
        U[k] = Qmax_at_Pmax < gen[k, QMAX]

    return L | U
