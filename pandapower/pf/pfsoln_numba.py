# -*- coding: utf-8 -*-

# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


"""Updates bus, gen, branch data structures to match power flow soln.
"""

from numpy import pi, finfo, c_, real, flatnonzero as find, angle, conj, zeros, complex128

try:
    from numba import jit
except ImportError:
    from pandapower.pf.no_numba import jit

from pandapower.pypower.idx_brch import F_BUS, T_BUS, PF, PT, QF, QT
from pandapower.pypower.idx_bus import VM, VA, PD, QD
from pandapower.pypower.idx_gen import GEN_BUS, GEN_STATUS, PG, QG
from pandapower.pypower.pfsoln import _update_v, _update_q, _update_p

EPS = finfo(float).eps


def pfsoln(baseMVA, bus, gen, branch, Ybus, Yf, Yt, V, ref, ref_gens, Ibus=None):
    """Updates bus, gen, branch data structures to match power flow soln.

    @author: Ray Zimmerman (PSERC Cornell)
    @author: Richard Lincoln
    """
    ## generator info
    on = find(gen[:, GEN_STATUS] > 0)  ## which generators are on?
    gbus = gen[on, GEN_BUS].astype(int)  ## what buses are they at?

    ## compute total injected bus powers
    Ibus = zeros(len(V)) if Ibus is None else Ibus
    Sbus = V[gbus] * conj(Ybus[gbus, :] * V - Ibus[gbus])

    _update_v(bus, V)
    _update_q(baseMVA, bus, gen, gbus, Sbus, on)
    _update_p(baseMVA, bus, gen, ref, gbus, on, Sbus, ref_gens)

    ##----- update/compute branch power flows -----

    ## complex power at "from" bus
    Sf = V[real(branch[:, F_BUS]).astype(int)] * calc_branch_flows(Yf.data, Yf.indptr, Yf.indices, V, baseMVA,
                                                                   Yf.shape[0])
    ## complex power injected at "to" bus
    St = V[real(branch[:, T_BUS]).astype(int)] * calc_branch_flows(Yt.data, Yt.indptr, Yt.indices, V, baseMVA,
                                                                   Yt.shape[0])
    branch[:, [PF, QF, PT, QT]] = c_[Sf.real, Sf.imag, St.real, St.imag]
    return bus, gen, branch


def pf_solution_single_slack(baseMVA, bus, gen, branch, Ybus, Yf, Yt, V, ref, ref_gens, Ibus=None):
    """ experimental faster version of pfsoln.

        NOTE: this function is not used yet in standard pp, since there seems to be a problem with shunts
        NOTE: Do not use in combination with voltage dependend loads

    """

    ##----- update bus voltages -----
    bus[:, VM] = abs(V)
    bus[:, VA] = angle(V) * 180 / pi

    ##----- update/compute branch power flows -----

    ## complex power at "from" bus
    Sf = V[real(branch[:, F_BUS]).astype(int)] * calc_branch_flows(Yf.data, Yf.indptr, Yf.indices, V, baseMVA,
                                                                   Yf.shape[0])
    ## complex power injected at "to" bus
    St = V[real(branch[:, T_BUS]).astype(int)] * calc_branch_flows(Yt.data, Yt.indptr, Yt.indices, V, baseMVA,
                                                                   Yt.shape[0])

    branch[:, [PF, QF, PT, QT]] = c_[Sf.real, Sf.imag, St.real, St.imag]

    p_bus = bus[:, PD].sum()
    q_bus = bus[:, QD].sum()
    p_loss = branch[:, [PF, PT]].sum()
    q_loss = branch[:, [QF, QT]].sum()

    # slack p = sum of branch losses and p demand at all buses
    gen[:, PG] = p_loss.real + p_bus  # branch p losses + p demand
    gen[:, QG] = q_loss.real + q_bus  # branch q losses + q demand

    return bus, gen, branch


@jit(nopython=True, cache=True)
def calc_branch_flows(Yy_x, Yy_p, Yy_j, v, baseMVA, dim_x):  # pragma: no cover

    Sx = zeros(dim_x, dtype=complex128)

    # iterate through sparse matrix
    for r in range(len(Yy_p) - 1):
        for k in range(Yy_p[r], Yy_p[r + 1]):
            Sx[r] += conj(Yy_x[k] * v[Yy_j[k]]) * baseMVA

    return Sx
