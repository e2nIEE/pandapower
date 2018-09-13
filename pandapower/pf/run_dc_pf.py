# -*- coding: utf-8 -*-

# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.



from time import time

from numpy import flatnonzero as find, pi, zeros, real

from pandapower.idx_brch import PF, PT, QF, QT
from pandapower.idx_bus import VA, GS
from pandapower.idx_gen import PG
from pandapower.pf.dcpf import dcpf
from pandapower.pf.makeBdc import makeBdc
from pandapower.pf.makeSbus import makeSbus
from pandapower.pf.ppci_variables import _get_pf_variables_from_ppci, _store_results_from_pf_in_ppci

def _run_dc_pf(ppci):
    t0 = time()
    baseMVA, bus, gen, branch, ref, pv, pq, on, gbus, _, _ = _get_pf_variables_from_ppci(ppci)

    ## initial state
    Va0 = bus[:, VA] * (pi / 180.)

    ## build B matrices and phase shift injections
    B, Bf, Pbusinj, Pfinj = makeBdc(bus, branch)

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
    ## update Pg for slack generator (1st gen at ref bus)
    ## (note: other gens at ref bus are accounted for in Pbus)
    ##      Pg = Pinj + Pload + Gs
    ##      newPg = oldPg + newPinj - oldPinj

    refgen = zeros(len(ref), dtype=int)
    for k in range(len(ref)):
        temp = find(gbus == ref[k])
        refgen[k] = on[temp[0]]
    gen[refgen, PG] = real(gen[refgen, PG] + (B[ref, :] * Va - Pbus[ref]) * baseMVA)

    # store results from DC powerflow for AC powerflow
    et = time() - t0
    success = True
    iterations = 1
    ppci = _store_results_from_pf_in_ppci(ppci, bus, gen, branch, success, iterations, et)
    return ppci
