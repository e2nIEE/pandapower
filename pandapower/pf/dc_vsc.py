# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

"""
DC grid model of the Newton-Raphson power flow with VSC converters.

Every VSC k is modelled on the DC side as

    plus terminal f --- r_dc (g) --- internal node t === ideal converter === minus terminal m
                    |                                                       |
                    +------------------- no-load conductance (gl) ----------+

The minus terminal m is either ground (monopolar VSC, m = -1) or a DC bus (bipolar VSC, m >= 0). The converter
injects the current I_k = g * (V_t - V_f) at the internal node t and draws the same current from the minus
terminal m. For a bipolar VSC this return current is part of the KCL at bus m, so a metallic return (DMR) line
connected to m carries the unbalance current of the poles like any other DC line.

The KCL of the DC buses is formulated as a current mismatch (instead of a power mismatch), so that buses with
V = 0 p.u. (e.g. the neutral / DMR buses of a bipolar system) do not lead to a singular Jacobian.

Equations (one per relevant DC bus):
    * DC reference bus (source_dc):             V_i - V_set = 0
    * internal node t of VSC in DC P mode:      (V_f - V_m) * I_f - P_set = 0     (P_set flows from DC into VSC)
    * internal node t of VSC in DC V mode:      V_f - V_m - V_set = 0            (for m = ground: V_f - V_set = 0)
      if several VSC in DC V mode are connected to the same terminals, only the first one ("leader") controls the
      voltage, the others share the current equally:  I_k - I_leader = 0
    * internal node t of VSC in AC slack mode:  (V_t - V_m) * I_k - P_ac = 0      (P_ac: power from AC into VSC)
    * all other DC buses (KCL):                 (Y V)_i - P_dc_i / V_i = 0
"""

import warnings

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve, MatrixRankWarning

from pandapower.pf.makeYbus_facts import make_Ybus_facts


def _minus_terminal_voltage(V_dc, m):
    has_m = m >= 0
    return np.where(has_m, V_dc[np.where(has_m, m, 0)], 0.), has_m


def _parallel_v_control(f, m, ctrl_v):
    """
    Splits the VSC in DC V mode into leaders (control the voltage) and followers (same terminals as a leader,
    share the current of the leader). Returns the masks and, for every VSC, the index of its leader.
    """
    leader = np.arange(len(f))
    idx_v = np.flatnonzero(ctrl_v)
    if len(idx_v):
        _, first, inverse = np.unique(np.c_[f[idx_v], m[idx_v]], axis=0, return_index=True, return_inverse=True)
        leader[idx_v] = idx_v[first[inverse.ravel()]]
    follower = ctrl_v & (leader != np.arange(len(f)))
    return ctrl_v & ~follower, follower, leader


def makeYbus_dc(n, hvdc_fb, hvdc_tb, hvdc_y_pu, f, t, m, g, gl):
    """
    Builds the DC "admittance" matrices.

    Returns:
        Ybus_lines: DC lines only
        Ybus_vsc_dc: r_dc of the VSC, no-load conductance and the return current coupling of the minus terminals
        Ybus_dc: sum of both, (Ybus_dc @ V_dc)_i is the current flowing out of bus i into lines and VSC
    """
    Ybus_lines = make_Ybus_facts(hvdc_fb, hvdc_tb, hvdc_y_pu, n, dtype=np.float64)

    has_m = m >= 0
    gl_ground = np.where(has_m, 0., gl)
    # r_dc between plus terminal and internal node, no-load conductance to ground for monopolar VSC:
    Ybus_vsc_dc = make_Ybus_facts(f, t, g, n, ysf_pu=gl_ground, dtype=np.float64)
    if np.any(has_m):
        fm, tm, mm = f[has_m], t[has_m], m[has_m]
        # no-load conductance between plus and minus terminal for bipolar VSC:
        Ybus_vsc_dc = Ybus_vsc_dc + make_Ybus_facts(fm, mm, gl[has_m], n, dtype=np.float64)
        # converter current I_k = g * (V_t - V_f) is drawn from the minus terminal:
        rows = np.r_[mm, mm]
        cols = np.r_[tm, fm]
        data = np.r_[g[has_m], -g[has_m]]
        Ybus_vsc_dc = Ybus_vsc_dc + csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)

    return Ybus_lines, Ybus_vsc_dc, Ybus_lines + Ybus_vsc_dc


def calc_vsc_dc_quantities(V_dc, f, t, m, g, gl):
    """
    Returns:
        i_k: converter current, flowing out of the converter at the internal node t (p.u.)
        i_f: current flowing from the plus terminal bus f into the VSC (p.u.)
        v_fm: voltage between plus and minus terminal
        v_tm: voltage between internal node and minus terminal
        v_m: voltage of the minus terminal (0 for ground)
    """
    v_m, _ = _minus_terminal_voltage(V_dc, m)
    i_k = g * (V_dc[t] - V_dc[f])
    v_fm = V_dc[f] - v_m
    v_tm = V_dc[t] - v_m
    i_f = -i_k + gl * v_fm
    return i_k, i_f, v_fm, v_tm, v_m


def _row_order(dc_ref, dc_p):
    return np.r_[dc_ref, dc_p].astype(np.int64)


def evaluate_Fx_dc(V_dc, Ybus_dc, P_dc, dc_ref, dc_p, V_sl_dc, f, t, m, g, gl, ctrl_p, ctrl_v, ctrl_sl,
                   value_dc, p_ac_sl):
    """
    Mismatch of the DC grid, ordered as [dc_ref, dc_p] (same order as the DC variables in the Jacobian).

    ctrl_p, ctrl_v, ctrl_sl: boolean masks (one entry per VSC) for DC P mode, DC V mode and AC slack mode
    value_dc: DC set-point per VSC (p.u.)
    p_ac_sl: power flowing from the AC side into the VSC (p.u.), only used for ctrl_sl
    """
    i_bus = Ybus_dc.dot(V_dc)
    with np.errstate(divide="ignore", invalid="ignore"):
        i_load = np.where(P_dc != 0, P_dc / V_dc, 0.)
    F = i_bus - i_load

    i_k, i_f, v_fm, v_tm, _ = calc_vsc_dc_quantities(V_dc, f, t, m, g, gl)
    ctrl_v, follower, leader = _parallel_v_control(f, m, ctrl_v)
    F[t[ctrl_p]] = (v_fm * i_f)[ctrl_p] - value_dc[ctrl_p]
    F[t[ctrl_v]] = v_fm[ctrl_v] - value_dc[ctrl_v]
    F[t[follower]] = (i_k - i_k[leader])[follower]
    F[t[ctrl_sl]] = (v_tm * i_k)[ctrl_sl] - p_ac_sl[ctrl_sl]

    F[dc_ref] = V_dc[dc_ref] - V_sl_dc
    return F[_row_order(dc_ref, dc_p)]


def create_J_dc(V_dc, Ybus_dc, P_dc, dc_ref, dc_p, f, t, m, g, gl, ctrl_p, ctrl_v, ctrl_sl):
    """
    Jacobian of evaluate_Fx_dc w.r.t. V_dc, rows and columns ordered as [dc_ref, dc_p].
    The coupling between the AC and the DC side of the VSC is not included (same as for the AC part).
    """
    n = len(V_dc)
    i_k, i_f, v_fm, v_tm, _ = calc_vsc_dc_quantities(V_dc, f, t, m, g, gl)
    has_m = m >= 0
    ctrl_v, follower, leader = _parallel_v_control(f, m, ctrl_v)

    # KCL rows: d/dV (Y V - P / V) = Y + diag(P / V^2)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_load = np.where(P_dc != 0, P_dc / np.square(V_dc), 0.)
    kcl_row = np.ones(n, dtype=bool)
    kcl_row[dc_ref] = False
    kcl_row[t[ctrl_p | ctrl_v | follower | ctrl_sl]] = False
    J = diags(kcl_row.astype(np.float64)) @ (Ybus_dc + diags(d_load))

    rows, cols, data = [dc_ref], [dc_ref], [np.ones(len(dc_ref))]

    def _add(mask, col, value):
        rows.append(t[mask])
        cols.append(col[mask])
        data.append(value[mask])

    # DC P mode: F = (V_f - V_m) * i_f - P_set, with i_f = g (V_f - V_t) + gl (V_f - V_m)
    _add(ctrl_p, f, i_f + v_fm * (g + gl))
    _add(ctrl_p, t, -v_fm * g)
    _add(ctrl_p & has_m, m, -i_f - v_fm * gl)
    # DC V mode: F = V_f - V_m - V_set
    _add(ctrl_v, f, np.ones_like(g))
    _add(ctrl_v & has_m, m, -np.ones_like(g))
    # parallel VSC in DC V mode: F = i_k - i_k_leader, with i_k = g (V_t - V_f)
    _add(follower, t, g)
    _add(follower, f, g[leader] - g)
    _add(follower, t[leader], -g[leader])
    # AC slack mode: F = (V_t - V_m) * i_k - P_ac, with i_k = g (V_t - V_f)
    _add(ctrl_sl, t, i_k + v_tm * g)
    _add(ctrl_sl, f, -v_tm * g)
    _add(ctrl_sl & has_m, m, -i_k)

    J = J + csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    order = _row_order(dc_ref, dc_p)
    return J.tocsr()[order, :][:, order]


def init_V_dc(V_dc, Ybus_dc, dc_ref, V_sl_dc, f, t, m, g, ctrl_v, value_dc):
    """
    Initial DC voltages from a linear no-load solution: DC voltage set-points (also the pole-to-pole / pole-to-neutral
    set-points of bipolar VSC) and DC sources are enforced, all converter currents and loads are zero.
    This provides negative initial voltages for negative poles and ~0 for neutral buses.

    Raises a UserWarning if the DC system has no voltage reference (e.g. a bipolar system without any grounded bus).
    """
    n = len(V_dc)
    _check_voltage_reference(Ybus_dc, dc_ref, f, m, ctrl_v)
    ctrl_v, follower, leader = _parallel_v_control(f, m, ctrl_v)
    fixed_row = np.zeros(n, dtype=bool)
    fixed_row[dc_ref] = True
    fixed_row[t] = True
    A = diags((~fixed_row).astype(np.float64)) @ Ybus_dc
    b = np.zeros(n)

    has_m = m >= 0
    # leaders: V_f - V_m = V_set; followers: same current as the leader; all others: no current
    zero_i = ~ctrl_v & ~follower
    rows = [dc_ref, t[ctrl_v], t[ctrl_v & has_m], t[zero_i], t[zero_i],
            t[follower], t[follower], t[follower]]
    cols = [dc_ref, f[ctrl_v], m[ctrl_v & has_m], t[zero_i], f[zero_i],
            t[follower], f[follower], t[leader[follower]]]
    data = [np.ones(len(dc_ref)), np.ones(ctrl_v.sum()), -np.ones((ctrl_v & has_m).sum()), g[zero_i], -g[zero_i],
            g[follower], (g[leader] - g)[follower], -g[leader[follower]]]
    A = A + csr_matrix((np.concatenate(data), (np.concatenate(rows).astype(np.int64),
                                               np.concatenate(cols).astype(np.int64))), shape=(n, n))
    b[dc_ref] = V_sl_dc
    b[t[ctrl_v]] = value_dc[ctrl_v]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MatrixRankWarning)
        x = spsolve(A.tocsc(), b)
    if not np.all(np.isfinite(x)) or not np.allclose(A @ x, b, atol=1e-6):
        raise UserWarning(_NO_REFERENCE_MSG)
    V_dc[:] = x
    return V_dc


_NO_REFERENCE_MSG = ("The DC system with bipolar VSC has no voltage reference (it is floating). Ground exactly one bus "
                     "of each DC system, e.g. the neutral bus of one station: "
                     "create_source_dc(net, bus_dc=neutral_bus, vm_pu=0.)")


def _check_voltage_reference(Ybus_dc, dc_ref, f, m, ctrl_v):
    """
    Every connected DC system with bipolar VSC needs a voltage reference to ground: a DC source (e.g. a grounded
    neutral bus with vm_pu=0) or a monopolar VSC which controls the DC voltage.
    """
    adjacency = abs(Ybus_dc)
    _, labels = connected_components(adjacency + adjacency.T, directed=False)
    referenced = np.zeros(labels.max() + 1, dtype=bool)
    referenced[labels[dc_ref]] = True
    referenced[labels[f[ctrl_v & (m < 0)]]] = True
    if not np.all(referenced[labels[f[m >= 0]]]):
        raise UserWarning(_NO_REFERENCE_MSG)
