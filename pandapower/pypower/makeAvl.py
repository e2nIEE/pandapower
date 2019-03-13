# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Construct linear constraints for constant power factor var loads.
"""

from sys import stderr

from numpy import array, zeros, arange, sin, cos, arctan2, r_
from numpy import flatnonzero as find
from scipy.sparse import csr_matrix as sparse

from pypower.idx_gen import PG, QG, PMIN, QMIN, QMAX

from pypower.isload import isload


def makeAvl(baseMVA, gen):
    """Construct linear constraints for constant power factor var loads.

    Constructs parameters for the following linear constraint enforcing a
    constant power factor constraint for dispatchable loads::

         lvl <= Avl * [Pg, Qg] <= uvl

    C{ivl} is the vector of indices of generators representing variable loads.

    @author: Ray Zimmerman (PSERC Cornell)
    @author: Carlos E. Murillo-Sanchez (PSERC Cornell & Universidad
    Autonoma de Manizales)
    """
    ## data dimensions
    ng = gen.shape[0]      ## number of dispatchable injections
    Pg   = gen[:, PG] / baseMVA
    Qg   = gen[:, QG] / baseMVA
    Pmin = gen[:, PMIN] / baseMVA
    Qmin = gen[:, QMIN] / baseMVA
    Qmax = gen[:, QMAX] / baseMVA

    # Find out if any of these "generators" are actually dispatchable loads.
    # (see 'help isload' for details on what constitutes a dispatchable load)
    # Dispatchable loads are modeled as generators with an added constant
    # power factor constraint. The power factor is derived from the original
    # value of Pmin and either Qmin (for inductive loads) or Qmax (for
    # capacitive loads). If both Qmin and Qmax are zero, this implies a unity
    # power factor without the need for an additional constraint.

    ivl = find( isload(gen) & ((Qmin != 0) | (Qmax != 0)) )
    nvl = ivl.shape[0]  ## number of dispatchable loads

    ## at least one of the Q limits must be zero (corresponding to Pmax == 0)
    if any( (Qmin[ivl] != 0) & (Qmax[ivl] != 0) ):
        stderr.write('makeAvl: either Qmin or Qmax must be equal to zero for '
                     'each dispatchable load.\n')

    # Initial values of PG and QG must be consistent with specified power
    # factor This is to prevent a user from unknowingly using a case file which
    # would have defined a different power factor constraint under a previous
    # version which used PG and QG to define the power factor.
    Qlim = (Qmin[ivl] == 0) * Qmax[ivl] + (Qmax[ivl] == 0) * Qmin[ivl]
    if any( abs( Qg[ivl] - Pg[ivl] * Qlim / Pmin[ivl] ) > 1e-6 ):
        stderr.write('makeAvl: For a dispatchable load, PG and QG must be '
                     'consistent with the power factor defined by PMIN and '
                     'the Q limits.\n')

    # make Avl, lvl, uvl, for lvl <= Avl * [Pg Qg] <= uvl
    if nvl > 0:
        xx = Pmin[ivl]
        yy = Qlim
        pftheta = arctan2(yy, xx)
        pc = sin(pftheta)
        qc = -cos(pftheta)
        ii = r_[ arange(nvl), arange(nvl) ]
        jj = r_[ ivl, ivl + ng ]
        Avl = sparse((r_[pc, qc], (ii, jj)), (nvl, 2 * ng))
        lvl = zeros(nvl)
        uvl = lvl
    else:
        Avl = zeros((0, 2*ng))
        lvl = array([])
        uvl = array([])

    return Avl, lvl, uvl, ivl
