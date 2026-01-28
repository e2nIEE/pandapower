# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.



from time import perf_counter

import numpy as np
from numpy import pi, zeros, real, bincount, int64

from pandapower.pypower.idx_brch import PF, PT, QF, QT, SHIFT, TAP
from pandapower.pypower.idx_brch_dc import DC_PF, DC_PT, DC_IF, DC_IT
from pandapower.pypower.idx_bus import VA, GS
from pandapower.pypower.idx_gen import PG, GEN_BUS
from pandapower.pypower.idx_vsc import (VSC_BUS, VSC_BUS_DC, VSC_MODE_DC, VSC_MODE_DC_P, VSC_VALUE_DC, VSC_MODE_AC,
                                        VSC_MODE_AC_SL, VSC_Q, VSC_P, VSC_P_DC)
from pandapower.pypower.dcpf import dcpf
from pandapower.pypower.makeBdc import makeBdc, phase_shift_injection, calc_b_from_branch
from pandapower.pypower.makeSbus import makeSbus
from pandapower.pf.ppci_variables import _get_pf_variables_from_ppci, _store_results_from_pf_in_ppci


def _run_dc_pf(ppci, recycle: dict | bool = False):
    """
    Runs a decoupled (dc) powerflow to initialize all the values.
    :param ppci: the internal ppci structure
    :param recycle: Flag to use results from a previous powerflow

    Returns:

    """
    t0 = perf_counter()

    # if TAP is changed, the Bbus and Bf must be calculated from scratch
    # if SHIFT is changed, we can still save some time by only calculating the Pbusinj and Pfinj
    if isinstance(recycle, dict) and "Bbus" in ppci['internal'] and \
            np.array_equal(ppci['internal']['branch'][:, TAP], ppci["branch"][:, TAP]):
        baseMVA = ppci['baseMVA']
        bus = ppci['bus']
        gen = ppci['gen']
        branch = ppci['branch']
        branch_dc = ppci['branch_dc']
        vsc = ppci['internal']['vsc']
        ref = ppci['internal']['ref']
        pv = ppci['internal']['pv']
        pq = ppci['internal']['pq']
        ref_gens = ppci['internal']['ref_gens']
        B, Bf = ppci['internal']['Bbus'], ppci['internal']['Bf']
        # check if transformer phase shift has changed and update phase shift injections:
        if np.array_equal(ppci['internal']['shift'], branch[:, SHIFT]):
            Pbusinj, Pfinj = ppci['internal']['Pbusinj'], ppci['internal']['Pfinj']
        else:
            Cft = ppci['internal']['Cft']
            b = calc_b_from_branch(branch, branch.shape[0])
            Pfinj, Pbusinj = phase_shift_injection(b, branch[:, SHIFT], Cft)
            ppci['internal']['shift'] = branch[:, SHIFT]
            ppci['internal']['Pbusinj'] = Pbusinj
            ppci['internal']['Pfinj'] = Pfinj
    else:
        baseMVA, bus, gen, branch, svc, tcsc, ssc, vsc, ref, pv, pq, *_, ref_gens = _get_pf_variables_from_ppci(ppci, True)

        ppci['internal']['baseMVA'] = baseMVA
        ppci['internal']['bus'] = bus
        ppci['internal']['gen'] = gen
        ppci['internal']['branch'] = branch
        ppci['internal']['vsc'] = vsc
        ppci['internal']['ref'] = ref
        ppci['internal']['pv'] = pv
        ppci['internal']['pq'] = pq
        ppci['internal']['ref_gens'] = ref_gens
        branch_dc = ppci["branch_dc"]

        # build B matrices and phase shift injections
        B, Bf, Pbusinj, Pfinj, Cft = makeBdc(bus, branch, ppci["bus_dc"], branch_dc, vsc)

        # updates Bbus matrix
        ppci['internal']['Bbus'] = B
        ppci['internal']['Bf'] = Bf
        ppci['internal']['Pbusinj'] = Pbusinj
        ppci['internal']['Pfinj'] = Pfinj
        ppci['internal']['Cft'] = Cft
        ppci['internal']['shift'] = branch[:, SHIFT]

    # initial state
    va0 = bus[:, VA] * (pi / 180.)
    # append zeros for the DC nodes
    va0 = np.concatenate([va0, np.zeros(ppci["bus_dc"].shape[0])])

    # compute complex bus power injections [generation - load]
    # adjusted for phase shifters and real shunts
    Pbus = np.real(makeSbus(baseMVA, bus, gen)) - bus[:, GS] / baseMVA
    # append zeros for the DC nodes
    Pbus = np.concatenate([Pbus, np.zeros(ppci["bus_dc"].shape[0])])
    # select VSCs with mode DC p and not mode AC slack
    vsc_with_p = vsc[(vsc[:, VSC_MODE_DC] == VSC_MODE_DC_P) & (vsc[:, VSC_MODE_AC] != VSC_MODE_AC_SL)]
    ac_bus = vsc_with_p[:, VSC_BUS].astype(int64)
    dc_bus = vsc_with_p[:, VSC_BUS_DC].astype(int64) + bus.shape[0]
    value = vsc_with_p[:, VSC_VALUE_DC]
    # fix if multiple VSC with P mode are attached to the same AC node # todo add test case
    ac_bus, inv = np.unique(ac_bus, return_inverse=True)
    value_ac = np.zeros_like(ac_bus, dtype=value.dtype)
    np.add.at(value_ac, inv, value)
    # fix if multiple VSC with P mode are attached to the same DC node # todo test case + test case combo with AC node
    dc_bus, inv = np.unique(dc_bus, return_inverse=True)
    value_dc = np.zeros_like(dc_bus, dtype=value.dtype)
    np.add.at(value_dc, inv, value)
    Pbus[ac_bus] += value_ac
    Pbus[dc_bus] -= value_dc
    Pbus -= Pbusinj

    pq_with_dc = np.concatenate([pq, np.arange(bus.shape[0], bus.shape[0] + ppci["bus_dc"].shape[0])])
    # "run" the power flow
    Va = dcpf(B, Pbus, va0, ref, pv, pq_with_dc)
    ppci['internal']["V"] = Va

    # update data matrices with solution
    # the AC branches: Q = 0, PF and PT from the matrix
    branch[:, [QF, QT]] = zeros((branch.shape[0], 2))
    branch[:, PF] = (Bf * Va + Pfinj)[:branch.shape[0]] * baseMVA
    branch[:, PT] = -branch[:, PF]
    # the VSCs: Q = 0, P AC and P DC from the matrix
    vsc[:, VSC_Q] = zeros(vsc.shape[0])
    vsc[:, VSC_P] = (Bf * Va + Pfinj)[branch.shape[0]:branch.shape[0] + vsc.shape[0]] * baseMVA
    # add the P setpoints since the VSCs with P setpoint have no impedance and no P result
    vsc[(vsc[:, VSC_MODE_DC] == VSC_MODE_DC_P) & (vsc[:, VSC_MODE_AC] != VSC_MODE_AC_SL), VSC_P] = (
        -vsc)[(vsc[:, VSC_MODE_DC] == VSC_MODE_DC_P) & (vsc[:, VSC_MODE_AC] != VSC_MODE_AC_SL), VSC_VALUE_DC]
    vsc[:, VSC_P_DC] = -vsc[:, VSC_P]
    # the DC branches: DC_PF and DC_PT from the matrix
    branch_dc[:, DC_PF] = (Bf * Va + Pfinj)[branch.shape[0] + vsc.shape[0]:] * baseMVA
    branch_dc[:, DC_PT] = -branch_dc[:, DC_PF]
    # set the currents as the nominal power (the currents are calculated in results_branch)
    branch_dc[:, DC_IF] = branch_dc[:, DC_PF]
    branch_dc[:, DC_IT] = branch_dc[:, DC_PT]
    # on AC nodes considered for the angle
    bus[:, VA] = Va[:bus.shape[0]] * (180. / pi)

    # ext_grid (ref_gens) buses
    refgenbus=gen[ref_gens, GEN_BUS].astype(np.int64)
    # number of ext_grids (ref_gens) at those buses
    ext_grids_bus=bincount(refgenbus)
    gen[ref_gens, PG] = (
        real(gen[ref_gens, PG] + (B[refgenbus, :] * Va - Pbus[refgenbus]) * baseMVA / ext_grids_bus[refgenbus]))

    # store results from DC powerflow for AC powerflow
    ppci = _store_results_from_pf_in_ppci(ppci, bus, gen, branch, success=True, iterations=1, et=perf_counter() - t0)
    return ppci
