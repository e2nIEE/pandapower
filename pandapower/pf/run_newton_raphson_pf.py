# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.



from time import time

from numpy import flatnonzero as find, r_, zeros, argmax, setdiff1d

from pandapower.pypower.idx_bus import PD, QD, BUS_TYPE, PQ
from pandapower.pypower.idx_gen import PG, QG, QMAX, QMIN, GEN_BUS, GEN_STATUS
from pandapower.pypower.bustypes import bustypes
from pandapower.pypower.makeSbus import makeSbus
from pandapower.pypower.makeYbus import makeYbus as makeYbus_pypower
from pandapower.pypower.newtonpf import newtonpf
from pandapower.pypower.pfsoln import pfsoln as pfsoln_pypower
from pandapower.pf.run_dc_pf import _run_dc_pf
from pandapower.pf.ppci_variables import _get_pf_variables_from_ppci, _store_results_from_pf_in_ppci

try:
    from pandapower.pf.makeYbus_numba import makeYbus as makeYbus_numba
    from pandapower.pf.pfsoln_numba import pfsoln as pfsoln_numba
except ImportError:
    pass

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def _run_newton_raphson_pf(ppci, options):
    """Runs a newton raphson power flow.
    """

    ##-----  run the power flow  -----


    t0 = time()
    if isinstance(options["init_va_degree"], str) and options["init_va_degree"] == "dc":
        ppci = _run_dc_pf(ppci)
    if options["enforce_q_lims"]:
        ppci, success, iterations, bus, gen, branch = _run_ac_pf_with_qlims_enforced(ppci, options)
    else:
        ppci, success, iterations, bus, gen, branch = _run_ac_pf_without_qlims_enforced(ppci, options)
    et = time() - t0
    ppci = _store_results_from_pf_in_ppci(ppci, bus, gen, branch, success, iterations, et)
    return ppci

def _get_Y_bus(ppci, options, makeYbus, baseMVA, bus, branch):
    recycle = options["recycle"]

    if recycle["Ybus"] and ppci["internal"]["Ybus"].size:
        Ybus, Yf, Yt = ppci["internal"]['Ybus'], ppci["internal"]['Yf'], ppci["internal"]['Yt']
    else:
        ## build admittance matrices
        Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
        ppci["internal"]['Ybus'], ppci["internal"]['Yf'], ppci["internal"]['Yt'] = Ybus, Yf, Yt

    return ppci, Ybus, Yf, Yt


def _get_numba_functions(ppci, options):
    """
    pfsoln from pypower maybe slow in some cases. This function chooses the fastest for the given pf calculation
    """
    if options["numba"]:
        makeYbus = makeYbus_numba
        pfsoln = pfsoln_numba
    else:
        makeYbus = makeYbus_pypower
        pfsoln = pfsoln_pypower
    return makeYbus, pfsoln


def _run_ac_pf_without_qlims_enforced(ppci, options):
    makeYbus, pfsoln = _get_numba_functions(ppci, options)

    baseMVA, bus, gen, branch, ref, pv, pq, _, _, V0, ref_gens = _get_pf_variables_from_ppci(ppci)

    ppci, Ybus, Yf, Yt = _get_Y_bus(ppci, options, makeYbus, baseMVA, bus, branch)

    ## compute complex bus power injections [generation - load]
    Sbus = makeSbus(baseMVA, bus, gen)

    ## run the newton power  flow

    V, success, iterations, ppci["internal"]["J"], ppci["internal"]["Vm_it"], ppci["internal"]["Va_it"] = newtonpf(Ybus, Sbus, V0, pv, pq, ppci, options)

    ## update data matrices with solution
    bus, gen, branch = pfsoln(baseMVA, bus, gen, branch, Ybus, Yf, Yt, V, ref, ref_gens)

    return ppci, success, iterations, bus, gen, branch


def _run_ac_pf_with_qlims_enforced(ppci, options):
    baseMVA, bus, gen, branch, ref, pv, pq, on, _, V0, ref_gens = _get_pf_variables_from_ppci(ppci)

    qlim = options["enforce_q_lims"]
    limited = []  ## list of indices of gens @ Q lims
    fixedQg = zeros(gen.shape[0])  ## Qg of gens at Q limits

    while True:
        ppci, success, iterations, bus, gen, branch = _run_ac_pf_without_qlims_enforced(ppci, options)

        ## find gens with violated Q constraints
        gen_status = gen[:, GEN_STATUS] > 0
        qg_max_lim = gen[:, QG] > gen[:, QMAX]
        qg_min_lim = gen[:, QG] < gen[:, QMIN]

        mx = setdiff1d(find(gen_status & qg_max_lim), ref_gens)
        mn = setdiff1d(find(gen_status & qg_min_lim), ref_gens)

        if len(mx) > 0 or len(mn) > 0:  ## we have some Q limit violations
            ## one at a time?
            if qlim == 2:  ## fix largest violation, ignore the rest
                k = argmax(r_[gen[mx, QG] - gen[mx, QMAX],
                              gen[mn, QMIN] - gen[mn, QG]])
                if k > len(mx):
                    mn = mn[k - len(mx)]
                    mx = []
                else:
                    mx = mx[k]
                    mn = []

            ## save corresponding limit values
            fixedQg[mx] = gen[mx, QMAX]
            fixedQg[mn] = gen[mn, QMIN]
            mx = r_[mx, mn].astype(int)

            ## convert to PQ bus
            gen[mx, QG] = fixedQg[mx]  ## set Qg to binding
            for i in range(len(mx)):  ## [one at a time, since they may be at same bus]
                gen[mx[i], GEN_STATUS] = 0  ## temporarily turn off gen,
                bi = gen[mx[i], GEN_BUS].astype(int)  ## adjust load accordingly,
                bus[bi, [PD, QD]] = (bus[bi, [PD, QD]] - gen[mx[i], [PG, QG]])

#            if len(ref) > 1 and any(bus[gen[mx, GEN_BUS].astype(int), BUS_TYPE] == REF):
#                raise ValueError('Sorry, pandapower cannot enforce Q '
#                                 'limits for slack buses in systems '
#                                 'with multiple slacks.')

            changed_gens = gen[mx, GEN_BUS].astype(int)
            bus[setdiff1d(changed_gens, ref), BUS_TYPE] = PQ  ## & set bus type to PQ

            ## update bus index lists of each type of bus
            ref, pv, pq = bustypes(bus, gen)

            limited = r_[limited, mx].astype(int)
        else:
            break  ## no more generator Q limits violated

    if len(limited) > 0:
        ## restore injections from limited gens [those at Q limits]
        gen[limited, QG] = fixedQg[limited]  ## restore Qg value,
        for i in range(len(limited)):  ## [one at a time, since they may be at same bus]
            bi = gen[limited[i], GEN_BUS].astype(int)  ## re-adjust load,
            bus[bi, [PD, QD]] = bus[bi, [PD, QD]] + gen[limited[i], [PG, QG]]
            gen[limited[i], GEN_STATUS] = 1  ## and turn gen back on

    return ppci, success, iterations, bus, gen, branch
