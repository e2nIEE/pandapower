# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

from time import time

from numpy import pi, exp
from numpy import r_, zeros, argmax
from numpy import flatnonzero as find

from pypower.makeSbus import makeSbus
from pypower.idx_bus import PD, QD, BUS_TYPE, PQ, REF, VM, VA
from pypower.idx_gen import PG, QG, QMAX, QMIN, GEN_BUS, GEN_STATUS, VG

from pandapower.pypower_extensions.pfsoln import pfsoln
from pandapower.pypower_extensions.newtonpf import newtonpf

from pandapower.pypower_extensions.bustypes import bustypes
from pandapower.run_dc_pf import _run_dc_pf


try:
    import pplog as logging
except:
    import logging

logger = logging.getLogger(__name__)


def _run_newton_raphson_pf(ppci, options):
    """Runs a newton raphson power flow.
    """

    ##-----  run the power flow  -----
    t0 = time()

    init = options["init"]

    if init == "dc":
        ppci = _run_dc_pf(ppci)

    ppci, success = _nr_ac_pf(ppci, options)

    ppci["et"] = time() - t0
    ppci["success"] = success

    return ppci


def _nr_ac_pf(ppci, options):
    if options["enforce_q_lims"]:
        ppci, success, bus, gen, branch = _run_ac_pf_with_qlims_enforced(ppci, options)

    else:
        ppci, success, bus, gen, branch = _run_ac_pf_without_qlims_enforced(ppci, options)

    ppci = _store_results_from_pf_in_ppci(ppci, bus, gen, branch)

    return ppci, success


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
    V0 = bus[:, VM] * exp(1j * pi / 180 * bus[:, VA])
    V0[gbus] = gen[on, VG] / abs(V0[gbus]) * V0[gbus]

    return baseMVA, bus, gen, branch, ref, pv, pq, on, gbus, V0


def _store_results_from_pf_in_ppci(ppci, bus, gen, branch):
    ppci["bus"], ppci["gen"], ppci["branch"] = bus, gen, branch
    return ppci


def _import_numba_extensions_if_flag_is_true(numba):
    ## check if numba is available and the corresponding flag
    if numba:
        try:
            from numba import _version as nb_version
            # get numba Version (in order to use it it must be > 0.25)
            nb_version = float(nb_version.version_version[:4])

            if nb_version < 0.25:
                logger.warning('Warning: Numba version too old -> Upgrade to a version > 0.25. Numba is disabled\n')
                numba = False

        except ImportError:
            # raise UserWarning('numba cannot be imported. Call runpp() with numba=False!')
            logger.warning('Warning: Numba cannot be imported. Numba is disabled. Call runpp() with Numba=False!\n')
            numba = False

    if numba:
        from pandapower.pypower_extensions.makeYbus import makeYbus
    else:
        from pandapower.pypower_extensions.makeYbus_pypower import makeYbus

    return numba, makeYbus


def _get_Y_bus(ppci, options, makeYbus, baseMVA, bus, branch):
    recycle = options["recycle"]

    if recycle["Ybus"] and ppci["internal"]["Ybus"].size:
        Ybus, Yf, Yt = ppci["internal"]['Ybus'], ppci["internal"]['Yf'], ppci["internal"]['Yt']
    else:
        ## build admittance matrices
        Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
        if recycle["Ybus"]:
            ppci["internal"]['Ybus'], ppci["internal"]['Yf'], ppci["internal"]['Yt'] = Ybus, Yf, Yt

    return ppci, Ybus, Yf, Yt


def _run_ac_pf_without_qlims_enforced(ppci, options):
    numba, makeYbus = _import_numba_extensions_if_flag_is_true(options["numba"])

    baseMVA, bus, gen, branch, ref, pv, pq, on, gbus, V0 = _get_pf_variables_from_ppci(ppci)

    ppci, Ybus, Yf, Yt = _get_Y_bus(ppci, options, makeYbus, baseMVA, bus, branch)

    ## compute complex bus power injections [generation - load]
    Sbus = makeSbus(baseMVA, bus, gen)

    ## run the newton power  flow
    V, success, _ = newtonpf(Ybus, Sbus, V0, pv, pq, options, numba)

    ## update data matrices with solution
    bus, gen, branch = pfsoln(baseMVA, bus, gen, branch, Ybus, Yf, Yt, V, ref, pv, pq)

    return ppci, success, bus, gen, branch


def _run_ac_pf_with_qlims_enforced(ppci, options):
    baseMVA, bus, gen, branch, ref, pv, pq, on, gbus, V0 = _get_pf_variables_from_ppci(ppci)

    qlim = options["enforce_q_lims"]
    limited = []  ## list of indices of gens @ Q lims
    fixedQg = zeros(gen.shape[0])  ## Qg of gens at Q limits

    while True:
        ppci, success, bus, gen, branch = _run_ac_pf_without_qlims_enforced(ppci, options)

        ## find gens with violated Q constraints
        gen_status = gen[:, GEN_STATUS] > 0
        qg_max_lim = gen[:, QG] > gen[:, QMAX]
        qg_min_lim = gen[:, QG] < gen[:, QMIN]

        mx = find(gen_status & qg_max_lim)
        mn = find(gen_status & qg_min_lim)

        if len(mx) > 0 or len(mn) > 0:  ## we have some Q limit violations
            # No PV generators
            if len(pv) == 0:
                success = 0
                break

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

            if len(ref) > 1 and any(bus[gen[mx, GEN_BUS].astype(int), BUS_TYPE] == REF):
                raise ValueError('Sorry, pandapower cannot enforce Q '
                                 'limits for slack buses in systems '
                                 'with multiple slacks.')

            bus[gen[mx, GEN_BUS].astype(int), BUS_TYPE] = PQ  ## & set bus type to PQ

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

    return ppci, success, bus, gen, branch
