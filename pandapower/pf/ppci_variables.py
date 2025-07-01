# -*- coding: utf-8 -*-
from pandapower.pypower.idx_brch import branch_cols
# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from pandapower.pypower.idx_bus import VM, VA
from pandapower.pypower.idx_gen import GEN_BUS, GEN_STATUS, VG
from pandapower.pypower.bustypes import bustypes
from numpy import flatnonzero as find, pi, exp, int64, hstack, zeros, float64

def _get_pf_variables_from_ppci(ppci, vsc_ref=False):
    ## default arguments
    if ppci is None:
        ValueError('ppci is empty')
    # ppopt = ppoption(ppopt)

    # get data for calc
    bus, gen, vsc = ppci["bus"], ppci["gen"], ppci["vsc"]

    # if ppc["branch"] comes from the pypower -> pandapower converter, it has fewer columns than ppc in pandapower
    # because it is lacking BR_R_ASYM, BR_X_ASYM, BR_G, BR_G_ASYM, BR_B_ASYM, and it is OK to use 0 as default values
    branch = ppci["branch"]
    br_shape = branch.shape
    if br_shape[1] < branch_cols:
        branch = hstack([branch, zeros(shape=(br_shape[0], branch_cols - br_shape[1]), dtype=float64)])

    ## get bus index lists of each type of bus
    ref, pv, pq = bustypes(bus, gen, vsc if vsc_ref else None)

    ## generator info
    on = find(gen[:, GEN_STATUS] > 0)  ## which generators are on?
    gbus = gen[on, GEN_BUS].astype(int64)  ## what buses are they at?

    ## initial state
    # V0    = ones(bus.shape[0])            ## flat start
    V0 = bus[:, VM] * exp(1j * pi / 180. * bus[:, VA])
    V0[gbus] = gen[on, VG] / abs(V0[gbus]) * V0[gbus]

    ref_gens = ppci["internal"]["ref_gens"]
    return ppci["baseMVA"], bus, gen, branch, ppci["svc"], ppci["tcsc"], ppci["ssc"], vsc, \
        ref, pv, pq, on, gbus, V0, ref_gens


def _store_results_from_pf_in_ppci(ppci, bus, gen, branch, success, iterations, et):
    ppci["bus"], ppci["gen"], ppci["branch"] = bus, gen, branch
    ppci["success"] = bool(success)
    ppci["iterations"] = iterations
    ppci["et"] = et
    return ppci
