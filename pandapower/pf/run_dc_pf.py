# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.



from time import perf_counter

import numpy as np
from numpy import pi, zeros, real, bincount

from pandapower.pypower.idx_brch import PF, PT, QF, QT, SHIFT, BR_STATUS, BR_X, TAP
from pandapower.pypower.idx_bus import VA, GS
from pandapower.pypower.idx_gen import PG, GEN_BUS
from pandapower.pypower.dcpf import dcpf
from pandapower.pypower.makeBdc import makeBdc, phase_shift_injection, calc_b_from_branch
from pandapower.pypower.makeSbus import makeSbus
from pandapower.pf.ppci_variables import _get_pf_variables_from_ppci, _store_results_from_pf_in_ppci


def _run_dc_pf(ppci, recycle=False):
    t0 = perf_counter()

    # if TAP is changed, the Bbus and Bf must be calculated from scratch
    # if SHIFT is changed, we can still save some time by only calculating the Pbusinj and Pfinj
    if isinstance(recycle, dict) and "Bbus" in ppci["internal"] and \
            np.array_equal(ppci['internal']['branch'][:, TAP], ppci["branch"][:, TAP]):
        baseMVA = ppci['baseMVA']
        bus = ppci['bus']
        gen = ppci['gen']
        branch = ppci['branch']
        ref = ppci["internal"]['ref']
        pv = ppci["internal"]['pv']
        pq = ppci["internal"]['pq']
        ref_gens = ppci["internal"]['ref_gens']
        B, Bf = ppci["internal"]['Bbus'], ppci["internal"]['Bf']
        # check if transformer phase shift has changed and update phase shift injections:
        if np.array_equal(ppci['internal']['shift'], branch[:, SHIFT]):
            Pbusinj, Pfinj = ppci["internal"]['Pbusinj'], ppci["internal"]['Pfinj']
        else:
            Cft = ppci['internal']['Cft']
            b = calc_b_from_branch(branch, branch.shape[0])
            Pfinj, Pbusinj = phase_shift_injection(b, branch[:, SHIFT], Cft)
            ppci['internal']['shift'] = branch[:, SHIFT]
            ppci['internal']['Pbusinj'] = Pbusinj
            ppci['internal']['Pfinj'] = Pfinj
    else:
        baseMVA, bus, gen, branch, svc, tcsc, ssc, ref, pv, pq, *_, ref_gens = _get_pf_variables_from_ppci(ppci)

        ppci["internal"]['baseMVA'] = baseMVA
        ppci["internal"]['bus'] = bus
        ppci["internal"]['gen'] = gen
        ppci["internal"]['branch'] = branch
        ppci["internal"]['ref'] = ref
        ppci["internal"]['pv'] = pv
        ppci["internal"]['pq'] = pq
        ppci["internal"]['ref_gens'] = ref_gens

        ## build B matrices and phase shift injections
        B, Bf, Pbusinj, Pfinj, Cft = makeBdc(bus, branch)

        ## updates Bbus matrix
        ppci['internal']['Bbus'] = B
        ppci['internal']['Bf'] = Bf
        ppci['internal']['Pbusinj'] = Pbusinj
        ppci['internal']['Pfinj'] = Pfinj
        ppci['internal']['Cft'] = Cft
        ppci['internal']['shift'] = branch[:, SHIFT]

    ## initial state
    Va0 = bus[:, VA] * (pi / 180.)

    ## compute complex bus power injections [generation - load]
    ## adjusted for phase shifters and real shunts
    Pbus = makeSbus(baseMVA, bus, gen) - Pbusinj - bus[:, GS] / baseMVA

    ## "run" the power flow
    Va = dcpf(B, Pbus, Va0, ref, pv, pq)
    ppci["internal"]["V"] = Va

    ## update data matrices with solution
    branch[:, [QF, QT]] = zeros((branch.shape[0], 2))
    branch[:, PF] = (Bf * Va + Pfinj) * baseMVA
    branch[:, PT] = -branch[:, PF]
    bus[:, VA] = Va * (180. / pi)
    ## update Pg for slack generators
    ## (note: other gens at ref bus are accounted for in Pbus)
    ##      Pg = Pinj + Pload + Gs
    ##      newPg = oldPg + newPinj - oldPinj

    ## ext_grid (ref_gens) buses
    refgenbus=gen[ref_gens, GEN_BUS].astype(np.int64)
    ## number of ext_grids (ref_gens) at those buses
    ext_grids_bus=bincount(refgenbus)
    gen[ref_gens, PG] = real(gen[ref_gens, PG] + (B[refgenbus, :] * Va - Pbus[refgenbus]) * baseMVA / ext_grids_bus[refgenbus])

    # store results from DC powerflow for AC powerflow
    et = perf_counter() - t0
    success = True
    iterations = 1
    ppci = _store_results_from_pf_in_ppci(ppci, bus, gen, branch, success, iterations, et)
    return ppci
