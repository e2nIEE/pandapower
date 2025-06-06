# -*- coding: utf-8 -*-

# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


"""Updates bus, gen, branch data structures to match power flow soln.
"""

from numpy import conj, zeros, complex128, abs, float64, sqrt, real, isin, arange
from numpy import finfo, c_, flatnonzero as find, setdiff1d, r_, int64

from pandapower.pypower.idx_brch import F_BUS, T_BUS, PF, PT, QF, QT
from pandapower.pypower.idx_bus import PD, QD
from pandapower.pypower.idx_gen import GEN_BUS, GEN_STATUS, PG, QG
from pandapower.pypower.idx_ssc import SSC_Q
from pandapower.pypower.idx_svc import SVC_Q
from pandapower.pypower.idx_vsc import VSC_P, VSC_Q
from pandapower.pypower.idx_tcsc import TCSC_QF, TCSC_QT, TCSC_PF, TCSC_PT
from pandapower.pypower.pfsoln import _update_v, _update_q, _update_p
from pandapower.auxiliary import version_check

try:
    from numba import jit
    version_check('numba')
except ImportError:
    from pandapower.pf.no_numba import jit

EPS = finfo(float).eps


def pfsoln(baseMVA, bus, gen, branch, svc, tcsc, ssc, vsc, Ybus, Yf, Yt, V, ref, ref_gens, Ibus=None,
           limited_gens=None):
    """Updates bus, gen, branch data structures to match power flow soln.

    @author: Ray Zimmerman (PSERC Cornell)
    @author: Richard Lincoln
    """
    # generator info
    on = find(gen[:, GEN_STATUS] > 0)  # which generators are on?
    gbus = gen[on, GEN_BUS].astype(int64)  # what buses are they at?

    # xward: add ref buses that are not at the generators
    xbus = setdiff1d(ref, gbus)

    # compute total injected bus powers
    Ibus = zeros(len(V)) if Ibus is None else Ibus
    Sbus = V * conj(Ybus * V - Ibus)

    _update_v(bus, V)
    # update gen results
    _update_q(baseMVA, bus, gen, gbus, Sbus[gbus], on)

    if limited_gens is not None and len(limited_gens) > 0:
        on = find((gen[:, GEN_STATUS] > 0) | isin(arange(len(gen)), limited_gens))
        gbus = gen[on, GEN_BUS].astype(int64)

    _update_p(baseMVA, bus, gen, ref, gbus, Sbus, ref_gens)

    # ----- update/compute branch power flows -----
    branch = _update_branch_flows(Yf, Yt, V, baseMVA, branch)

    return bus, gen, branch


def pf_solution_single_slack(baseMVA, bus, gen, branch, svc, tcsc, ssc, vsc, Ybus, Yf, Yt, V, ref, ref_gens,
                             Ibus=None, limited_gens=None):
    """
    faster version of pfsoln for a grid with a single slack bus

    NOTE: Do not use in combination with shunts (check if ppc["bus"][:, GS/BS] are != 0.)
    NOTE: Do not use in combination with voltage dependend loads

    """

    # ----- update bus voltages -----
    _update_v(bus, V)

    # ----- update/compute branch power flows -----
    branch = _update_branch_flows(Yf, Yt, V, baseMVA, branch)

    p_bus = bus[:, PD].sum()
    q_bus = bus[:, QD].sum()
    p_loss = branch[:, [PF, PT]].sum()
    q_loss = branch[:, [QF, QT]].sum()

    # consider FACTS devices:
    q_facts = svc[:, SVC_Q].sum() + tcsc[:, [TCSC_QF, TCSC_QT]].sum() + ssc[:, SSC_Q].sum() + vsc[:, VSC_Q].sum()
    p_facts = tcsc[:, [TCSC_PF, TCSC_PT]].sum() + vsc[:, VSC_P].sum()  # now should all be 0 due to no losses

    # slack p = sum of branch losses and p demand at all buses
    gen[:, PG] = p_loss.real + p_bus + p_facts  # branch p losses + p demand
    gen[:, QG] = q_loss.real + q_bus + q_facts  # branch q losses + q demand

    return bus, gen, branch


def _update_branch_flows(Yf, Yt, V, baseMVA, branch):
    f_bus = real(branch[:, F_BUS]).astype(int64)
    t_bus = real(branch[:, T_BUS]).astype(int64)
    # complex power at "from" bus
    Sf = calc_branch_flows(Yf.data, Yf.indptr, Yf.indices, V, baseMVA, Yf.shape[0], f_bus)
    # complex power injected at "to" bus
    St = calc_branch_flows(Yt.data, Yt.indptr, Yt.indices, V, baseMVA, Yt.shape[0], t_bus)
    branch[:, [PF, QF, PT, QT]] = c_[Sf.real, Sf.imag, St.real, St.imag]
    return branch


@jit(nopython=True, cache=False)
def calc_branch_flows(Yy_x, Yy_p, Yy_j, v, baseMVA, dim_x, bus_ind):  # pragma: no cover

    Sx = zeros(dim_x, dtype=complex128)

    # iterate through sparse matrix and get Sx = conj(Y_kj* V[j])
    for r in range(len(Yy_p) - 1):
        for k in range(Yy_p[r], Yy_p[r + 1]):
            Sx[r] += conj(Yy_x[k] * v[Yy_j[k]]) * baseMVA

    # finally get Sx = V[k] * conj(Y_kj* V[j])
    Sx *= v[bus_ind]

    return Sx


@jit(nopython=True, cache=False)
def calc_branch_flows_batch(Yy_x, Yy_p, Yy_j, V, baseMVA, dim_x, bus_ind, base_kv):  # pragma: no cover
    """
    Function to get branch flows with a batch computation for the timeseries module

    Parameters
    ----------
    Yy_x, Yy_p, Yy_j - Yt or Yf CSR represenation
    V - complex voltage matrix results from time series
    baseMVA - base MVA from ppc
    dim_x - shape of Y
    bus_ind - f_bus or t_bus
    base_kv - pcci["bus"] BASE_KV values

    Returns
    ----------
    i_abs, s_abs - absolute branch currents and power flows. This is "i_ft" / "s_ft" in results_branch.py
    S - complex Sf / St values frpm ppci

    """

    S = zeros((V.shape[0], dim_x), dtype=complex128)
    s_abs = zeros((V.shape[0], dim_x), dtype=float64)
    i_abs = zeros((V.shape[0], dim_x), dtype=float64)
    sqrt_3 = sqrt(3)

    # iterate over entries in V (v= complex V result of each time step)
    for t in range(V.shape[0]):
        v = V[t]
        vm = abs(v)
        Sx = zeros(dim_x, dtype=complex128)

        # iterate through sparse matrix and get Sx = conj(Y_kj* V[j])
        for r in range(len(Yy_p) - 1):
            for k in range(Yy_p[r], Yy_p[r + 1]):
                Sx[r] += conj(Yy_x[k] * v[Yy_j[k]]) * baseMVA

        # finally get Sx = V[k] * conj(Y_kj* V[j])
        Sx *= v[bus_ind]
        S[t, :] = Sx
        s_abs[t, :] = abs(Sx)
        i_abs[t, :] = s_abs[t, :] / (vm[bus_ind] * base_kv[bus_ind]) / sqrt_3

    return S, s_abs, i_abs
