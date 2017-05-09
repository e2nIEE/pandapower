# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.


from time import time

from numpy import flatnonzero as find, pi, exp, zeros, ones, real

from pandapower.idx_brch import PF, PT, QF, QT
from pandapower.idx_bus import VM, VA, GS
from pandapower.idx_gen import PG, VG, GEN_STATUS, GEN_BUS
from pandapower.pf.bustypes import bustypes
from pandapower.pf.dcpf import dcpf
from pandapower.pf.makeBdc import makeBdc
from pandapower.pf.makeSbus import makeSbus


def _run_dc_pf(ppci):
    t0 = time()
    baseMVA, bus, gen, branch, ref, pv, pq, on, gbus, _ = _get_pf_variables_from_ppci(ppci)

    ppci["bus"][:, VM] = 1.0
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
    bus[:, VM] = ones(bus.shape[0])
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
    ppci = _store_results_from_pf_in_ppci(ppci, bus, gen, branch)

    ppci["et"] = time() - t0
    ppci["success"] = True

    return ppci


def _get_pf_variables_from_ppci(ppci):
    ## default arguments
    if ppci is None:
        ValueError('ppci is empty')
    # ppopt = ppoption(ppopt)

    # get data for calc
    baseMVA, bus, gen, branch = \
        ppci["baseMVA"], ppci["bus"], ppci["gen"], ppci["branch"]

    ## get bus index lists of each type of bus
    ref, pv, pq = bustypes(bus, gen)

    ## generator info
    on = find(gen[:, GEN_STATUS] > 0)  ## which generators are on?
    gbus = gen[on, GEN_BUS].astype(int)  ## what buses are they at?

    ## initial state
    # V0    = ones(bus.shape[0])            ## flat start
    V0 = bus[:, VM] * exp(1j * pi / 180. * bus[:, VA])
    V0[gbus] = gen[on, VG] / abs(V0[gbus]) * V0[gbus]

    return baseMVA, bus, gen, branch, ref, pv, pq, on, gbus, V0


def _store_results_from_pf_in_ppci(ppci, bus, gen, branch):
    ppci["bus"], ppci["gen"], ppci["branch"] = bus, gen, branch
    return ppci
