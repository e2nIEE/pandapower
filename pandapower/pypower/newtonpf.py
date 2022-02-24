# -*- coding: utf-8 -*-

# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Copyright (c) 2016-2022 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


"""Solves the power flow using a full Newton's method.
"""

from numpy import angle, exp, linalg, conj, r_, Inf, arange, zeros, max, zeros_like, column_stack, float64
from scipy.sparse.linalg import spsolve

from pandapower.pf.iwamoto_multiplier import _iwamoto_step
from pandapower.pypower.makeSbus import makeSbus
from pandapower.pf.create_jacobian import create_jacobian_matrix, get_fastest_jacobian_function
from pandapower.pypower.idx_gen import PG
from pandapower.pypower.idx_bus import PD, SL_FAC


def newtonpf(Ybus, Sbus, V0, ref, pv, pq, ppci, options):
    """Solves the power flow using a full Newton's method.
    Solves for bus voltages given the full system admittance matrix (for
    all buses), the complex bus power injection vector (for all buses),
    the initial vector of complex bus voltages, and column vectors with
    the lists of bus indices for the swing bus, PV buses, and PQ buses,
    respectively. The bus voltage vector contains the set point for
    generator (including ref bus) buses, and the reference angle of the
    swing bus, as well as an initial guess for remaining magnitudes and
    angles.
    @see: L{runpf}
    @author: Ray Zimmerman (PSERC Cornell)
    @author: Richard Lincoln
    Modified by University of Kassel (Florian Schaefer) to use numba
    """

    # options
    tol = options['tolerance_mva']
    max_it = options["max_iteration"]
    numba = options["numba"]
    iwamoto = options["algorithm"] == "iwamoto_nr"
    voltage_depend_loads = options["voltage_depend_loads"]
    dist_slack = options["distributed_slack"]
    v_debug = options["v_debug"]
    use_umfpack = options["use_umfpack"]
    permc_spec = options["permc_spec"]

    baseMVA = ppci['baseMVA']
    bus = ppci['bus']
    gen = ppci['gen']
    slack_weights = bus[:, SL_FAC].astype(float64)  ## contribution factors for distributed slack

    # initialize
    i = 0
    V = V0
    Va = angle(V)
    Vm = abs(V)
    dVa, dVm = None, None
    if iwamoto:
        dVm, dVa = zeros_like(Vm), zeros_like(Va)

    if v_debug:
        Vm_it = Vm.copy()
        Va_it = Va.copy()
    else:
        Vm_it = None
        Va_it = None

    # set up indexing for updating V
    if dist_slack and len(ref) > 1:
        pv = r_[ref[1:], pv]
        ref = ref[[0]]

    pvpq = r_[pv, pq]
    # reference buses are always at the top, no matter where they are in the grid (very confusing...)
    # so in the refpvpq, the indices must be adjusted so that ref bus(es) starts with 0
    # todo: is it possible to simplify the indices/lookups and make the code clearer?
    # for columns: columns are in the normal order in Ybus; column numbers for J are reduced by 1 internally
    refpvpq = r_[ref, pvpq]
    # generate lookup pvpq -> index pvpq (used in createJ):
    #   shows for a given row from Ybus, which row in J it becomes
    #   e.g. the first row in J is a PV bus. If the first PV bus in Ybus is in the row 2, the index of the row in Jbus must be 0.
    #   pvpq_lookup will then have a 0 at the index 2
    pvpq_lookup = zeros(max(Ybus.indices) + 1, dtype=int)
    if dist_slack:
        # slack bus is relevant for the function createJ_ds
        pvpq_lookup[refpvpq] = arange(len(refpvpq))
    else:
        pvpq_lookup[pvpq] = arange(len(pvpq))

    # get jacobian function
    createJ = get_fastest_jacobian_function(pvpq, pq, numba, dist_slack)

    nref = len(ref)
    npv = len(pv)
    npq = len(pq)
    j0 = 0
    j1 = nref if dist_slack else 0
    j2 = j1 + npv  # j1:j2 - V angle of pv buses
    j3 = j2
    j4 = j2 + npq  # j3:j4 - V angle of pq buses
    j5 = j4
    j6 = j4 + npq  # j5:j6 - V mag of pq buses

    # make initial guess for the slack
    slack = (gen[:, PG].sum() - bus[:, PD].sum()) / baseMVA
    # evaluate F(x0)
    F = _evaluate_Fx(Ybus, V, Sbus, ref, pv, pq, slack_weights, dist_slack, slack)
    converged = _check_for_convergence(F, tol)

    Ybus = Ybus.tocsr()
    J = None

    # do Newton iterations
    while (not converged and i < max_it):
        # update iteration counter
        i = i + 1

        J = create_jacobian_matrix(Ybus, V, ref, refpvpq, pvpq, pq, createJ, pvpq_lookup, nref, npv, npq, numba, slack_weights, dist_slack)

        dx = -1 * spsolve(J, F, permc_spec=permc_spec, use_umfpack=use_umfpack)
        # update voltage
        if dist_slack:
            slack = slack + dx[j0:j1]
        if npv and not iwamoto:
            Va[pv] = Va[pv] + dx[j1:j2]
        if npq and not iwamoto:
            Va[pq] = Va[pq] + dx[j3:j4]
            Vm[pq] = Vm[pq] + dx[j5:j6]

        # iwamoto multiplier to increase convergence
        if iwamoto:
            Vm, Va = _iwamoto_step(Ybus, J, F, dx, pq, npv, npq, dVa, dVm, Vm, Va, pv, j1, j2, j3, j4, j5, j6)

        V = Vm * exp(1j * Va)
        Vm = abs(V)  # update Vm and Va again in case
        Va = angle(V)  # we wrapped around with a negative Vm

        if v_debug:
            Vm_it = column_stack((Vm_it, Vm))
            Va_it = column_stack((Va_it, Va))

        if voltage_depend_loads:
            Sbus = makeSbus(baseMVA, bus, gen, vm=Vm)

        F = _evaluate_Fx(Ybus, V, Sbus, ref, pv, pq, slack_weights, dist_slack, slack)

        converged = _check_for_convergence(F, tol)

    return V, converged, i, J, Vm_it, Va_it


def _evaluate_Fx(Ybus, V, Sbus, ref, pv, pq, slack_weights=None, dist_slack=False, slack=None):
    # evalute F(x)
    if dist_slack:
        # we include the slack power (slack * contribution factors) in the mismatch calculation
        mis = V * conj(Ybus * V) - Sbus + slack_weights * slack
        F = r_[mis[ref].real, mis[pv].real, mis[pq].real, mis[pq].imag]
    else:
        mis = V * conj(Ybus * V) - Sbus
        F = r_[mis[pv].real, mis[pq].real, mis[pq].imag]
    return F


def _check_for_convergence(F, tol):
    # calc infinity norm
    return linalg.norm(F, Inf) < tol
