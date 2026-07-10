# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

"""
Voltage sensitivities from the Newton-Raphson Jacobian of the last powerflow.

After every AC powerflow, pandapower keeps the final Newton-Raphson Jacobian in
``net._ppc["internal"]["J"]`` (see pandapower/pf/run_newton_raphson_pf.py). Its state
ordering is ``x = [Va(pv+pq); Vm(pq)]`` with the mismatch rows ``[dP(pv+pq); dQ(pq)]``,
all in ppci-internal bus numbering (``net._pd2ppc_lookups["bus"]`` maps pandapower bus
indices into that numbering after the powerflow).

At the solution, F(x, Q_spec) = 0 with F_Q = Q_calc(x) - Q_spec, so a unit increase of the
reactive power injection at PQ bus b gives J * dx/dQ_b = e_{row_Q(b)} and therefore
dVm_m/dQ_b = (J^-1)[row_Vm(m), row_Q(b)]. The adjoint formulation solves
J^T y = e_{row_Vm(m)} once per *measured* bus, which is usually the smaller dimension.
"""

import logging

import numpy as np
from scipy.sparse.linalg import splu

logger = logging.getLogger(__name__)

# one-slot cache of the LU factorization of the current Jacobian. Deliberately NOT stored in
# net (_ppc or controllers): SuperLU objects cannot be deepcopied or serialized and would
# break net copying. Identity of the J matrix object decides validity; run_control triggers
# one powerflow per iteration, so all controllers of an iteration share one factorization.
_LU_CACHE = {"J": None, "lu": None}


def _factorized_jacobian(J):
    if _LU_CACHE["J"] is not J:
        _LU_CACHE["J"] = J
        _LU_CACHE["lu"] = splu(J.tocsc())
    return _LU_CACHE["lu"]


def calc_dvm_dq(net, q_bus_idx, vm_bus_idx):
    """Sensitivity of bus voltage magnitudes to reactive power injections.

    Parameters
    ----------
    net : pandapowerNet
        Net after a converged Newton-Raphson powerflow (runpp).
    q_bus_idx : array-like of int
        Pandapower bus indices where reactive power is injected (positive injection =
        generation, e.g. positive sgen q_mvar).
    vm_bus_idx : array-like of int
        Pandapower bus indices whose voltage magnitude response is wanted.

    Returns
    -------
    numpy.ndarray of shape (len(vm_bus_idx), len(q_bus_idx))
        dVm/dQ in pu per Mvar. Entries are NaN when either bus was not a PQ bus in the last
        powerflow (slack/PV voltages are fixed; their Q is balanced by the generator), or
        None if no Jacobian is available (e.g. no NR powerflow ran).
    """
    ppc = net.get("_ppc") if hasattr(net, "get") else None
    internal = ppc.get("internal") if ppc else None
    J = internal.get("J") if internal else None
    if J is None:
        return None
    pv, pq = internal["pv"], internal["pq"]
    base_mva = internal["baseMVA"]
    lookup = net["_pd2ppc_lookups"]["bus"]
    npvpq = len(pv) + len(pq)
    if J.shape[0] != npvpq + len(pq):
        # FACTS/extended formulations append state variables; not supported here
        logger.debug("calc_dvm_dq: Jacobian has extended state variables, skipping")
        return None
    pq_position = {int(b): i for i, b in enumerate(pq)}

    def block_row(pd_bus):
        internal_bus = int(lookup[int(pd_bus)])
        position = pq_position.get(internal_bus)
        return None if position is None else npvpq + position

    q_rows = [block_row(b) for b in q_bus_idx]
    vm_rows = [block_row(b) for b in vm_bus_idx]
    result = np.full((len(vm_rows), len(q_rows)), np.nan)
    try:
        lu = _factorized_jacobian(J)
    except RuntimeError:  # singular
        logger.debug("calc_dvm_dq: Jacobian factorization failed")
        return None
    n = J.shape[0]
    active_vm = [(i, row) for i, row in enumerate(vm_rows) if row is not None]
    if not active_vm:
        return result
    # one batched adjoint solve for all measured buses
    rhs = np.zeros((n, len(active_vm)))
    for column, (_, vm_row) in enumerate(active_vm):
        rhs[vm_row, column] = 1.0
    y = lu.solve(rhs, trans='T')
    active_q = [(j, row) for j, row in enumerate(q_rows) if row is not None]
    for column, (i, _) in enumerate(active_vm):
        for j, q_row in active_q:
            result[i, j] = y[q_row, column] / base_mva
    return result
