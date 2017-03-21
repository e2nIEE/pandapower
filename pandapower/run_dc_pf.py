from time import time

from numpy import zeros, ones, real
from numpy import flatnonzero as find
from numpy import pi, exp

from pypower.idx_bus import VM, VA, GS
from pypower.idx_brch import PF, PT, QF, QT
from pypower.idx_gen import PG, VG, GEN_STATUS, GEN_BUS
from pypower.makeSbus import makeSbus

from pandapower.pypower_extensions.dcpf import dcpf
from pandapower.pypower_extensions.makeBdc import makeBdc
from pandapower.pypower_extensions.bustypes import bustypes


# from pandapower.run_pf_algorithm import _store_results_from_pf_in_ppci, _get_pf_variables_from_ppci

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
    success = True

    # store results from DC powerflow for AC powerflow
    ppci = _store_results_from_pf_in_ppci(ppci, bus, gen, branch)

    ppci["et"] = time() - t0
    ppci["success"] = success

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
