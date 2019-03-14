# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Builds the FDPF matrices, B prime and B double prime.
"""

from numpy import ones, zeros, copy

from pandapower.pypower.idx_bus import BS
from pandapower.pypower.idx_brch import BR_B, BR_R, TAP, SHIFT
from pandapower.pypower.makeYbus import makeYbus


def makeB(baseMVA, bus, branch, alg):
    """Builds the FDPF matrices, B prime and B double prime.

    Returns the two matrices B prime and B double prime used in the fast
    decoupled power flow. Does appropriate conversions to p.u. C{alg} is the
    value of the C{PF_ALG} option specifying the power flow algorithm.

    @see: L{fdpf}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    ## constants
    nb = bus.shape[0]          ## number of buses
    nl = branch.shape[0]       ## number of lines

    ##-----  form Bp (B prime)  -----
    temp_branch = copy(branch)                 ## modify a copy of branch
    temp_bus = copy(bus)                       ## modify a copy of bus
    temp_bus[:, BS] = zeros(nb)                ## zero out shunts at buses
    temp_branch[:, BR_B] = zeros(nl)           ## zero out line charging shunts
    temp_branch[:, TAP] = ones(nl)             ## cancel out taps
    if alg == 2:                               ## if XB method
        temp_branch[:, BR_R] = zeros(nl)       ## zero out line resistance
    Bp = -1 * makeYbus(baseMVA, temp_bus, temp_branch)[0].imag

    ##-----  form Bpp (B double prime)  -----
    temp_branch = copy(branch)                 ## modify a copy of branch
    temp_branch[:, SHIFT] = zeros(nl)          ## zero out phase shifters
    if alg == 3:                               ## if BX method
        temp_branch[:, BR_R] = zeros(nl)    ## zero out line resistance
    Bpp = -1 * makeYbus(baseMVA, bus, temp_branch)[0].imag

    return Bp, Bpp
