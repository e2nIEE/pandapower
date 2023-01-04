# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.



from time import perf_counter

import numpy as np
from numpy import pi, zeros, real, bincount

from pandapower.pypower.idx_brch import PF, PT, QF, QT, SHIFT, BR_STATUS, BR_X
from pandapower.pypower.idx_bus import VA, GS
from pandapower.pypower.idx_gen import PG, GEN_BUS
from pandapower.pypower.dcpf import dcpf
from pandapower.pypower.makeBdc import makeBdc, phase_shift_injection
from pandapower.pypower.makeSbus import makeSbus
from pandapower.pf.ppci_variables import _get_pf_variables_from_ppci, _store_results_from_pf_in_ppci


def _run_dc_pf(ppci, recycle=False):
    t0 = perf_counter()

    if isinstance(recycle, dict) and not recycle["trafo"] and "Bbus" in ppci["internal"]:
        baseMVA = ppci['baseMVA']
        bus = ppci['bus']
        gen = ppci['gen']
        branch = ppci['branch']
        ref = ppci['ref']
        pv = ppci['pv']
        pq = ppci['pq']
        ref_gens = ppci['ref_gens']
        B, Bf = ppci["internal"]['Bbus'], ppci["internal"]['Bf']
        # check if transformer phase shift has changed and update phase shift injections:
        if np.array_equal(ppci['internal']['shift'], branch[:, SHIFT]):
            Pbusinj, Pfinj = ppci["internal"]['Pbusinj'], ppci["internal"]['Pfinj']
        else:
            Cft = ppci['internal']['Cft']
            Pfinj, Pbusinj = phase_shift_injection(branch[:, BR_STATUS] / branch[:, BR_X], branch[:, SHIFT], Cft)
    else:
        baseMVA, bus, gen, branch, ref, pv, pq, _, _, _, ref_gens = _get_pf_variables_from_ppci(ppci)

        ppci['baseMVA'] = baseMVA
        ppci['bus'] = bus
        ppci['gen'] = gen
        ppci['branch'] = branch
        ppci['ref'] = ref
        ppci['pv'] = pv
        ppci['pq'] = pq
        ppci['ref_gens'] = ref_gens

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
    refgenbus=gen[ref_gens, GEN_BUS].astype(int)
    ## number of ext_grids (ref_gens) at those buses
    ext_grids_bus=bincount(refgenbus)
    gen[ref_gens, PG] = real(gen[ref_gens, PG] + (B[refgenbus, :] * Va - Pbus[refgenbus]) * baseMVA / ext_grids_bus[refgenbus])

    # store results from DC powerflow for AC powerflow
    et = perf_counter() - t0
    success = True
    iterations = 1
    ppci = _store_results_from_pf_in_ppci(ppci, bus, gen, branch, success, iterations, et)
    return ppci
