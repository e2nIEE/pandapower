# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Splits the gencost variable into two pieces if costs are given for Qg.
"""

from sys import stderr

from numpy import array, arange


def pqcost(gencost, ng, on=None):
    """Splits the gencost variable into two pieces if costs are given for Qg.

    Checks whether C{gencost} has cost information for reactive power
    generation (rows C{ng+1} to C{2*ng}). If so, it returns the first C{ng}
    rows in C{pcost} and the last C{ng} rows in C{qcost}. Otherwise, leaves
    C{qcost} empty. Also does some error checking.
    If C{on} is specified (list of indices of generators which are on line)
    it only returns the rows corresponding to these generators.

    @author: Ray Zimmerman (PSERC Cornell)
    """
    if on is None:
        on = arange(ng)

    if gencost.shape[0] == ng:
        pcost = gencost[on, :]
        qcost = array([])
    elif gencost.shape[0] == 2 * ng:
        pcost = gencost[on, :]
        qcost = gencost[on + ng, :]
    else:
        stderr.write('pqcost: gencost has wrong number of rows\n')

    return pcost, qcost
