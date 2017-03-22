# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

"""Runs a power flow.
"""

from time import time

from numpy import r_, zeros, pi, ones, exp, argmax, real
from numpy import flatnonzero as find

from pypower.ppoption import ppoption
from pypower.makeSbus import makeSbus
from pypower.fdpf import fdpf
from pypower.gausspf import gausspf
from pypower.makeB import makeB
from pypower.idx_bus import PD, QD, VM, VA, GS, BUS_TYPE, PQ, REF
from pypower.idx_brch import PF, PT, QF, QT
from pypower.idx_gen import PG, QG, VG, QMAX, QMIN, GEN_BUS, GEN_STATUS

from pandapower.pypower_extensions.makeBdc import makeBdc
from pandapower.pypower_extensions.pfsoln import pfsoln
from pandapower.pypower_extensions.dcpf import dcpf
from pandapower.pypower_extensions.bustypes import bustypes

try:
    import pplog as logging
except:
    import logging

logger = logging.getLogger(__name__)


def _runpf(ppci, options, **kwargs):
    """Runs a power flow.

    Similar to runpf() from pypower. See Pypower documentation for more information.

    Changes by University of Kassel (Florian Schaefer):
        numba can be used for pf calculations.
        Changes in structure (AC as well as DC PF can be calculated)
    """

    ##-----  run the power flow  -----
    t0 = time()
    # ToDo: Options should be extracted in every subfunction not here...
    init, ac, numba, recycle, ppopt = _get_options(options, **kwargs)

    if ac:  # AC formulation
        if init == "dc":
            ppci, success = _dc_runpf(ppci, ppopt)

        ppci, success = _ac_runpf(ppci, ppopt, numba, recycle)
    else:  ## DC formulation
        ppci, success = _dc_runpf(ppci, ppopt)

    ppci["et"] = time() - t0
    ppci["success"] = success

    return ppci, success


def _get_options(options, **kwargs):
    init = options["init"]
    ac = options["ac"]
    recycle = options["recycle"]
    numba = options["numba"]
    enforce_q_lims = options["enforce_q_lims"]
    tolerance_kva = options["tolerance_kva"]
    algorithm = options["algorithm"]
    max_iteration = options["max_iteration"]

    # algorithms implemented within pypower
    algorithm_pypower_dict = {'nr': 1, 'fdbx': 2, 'fdxb': 3, 'gs': 4}

    ppopt = ppoption(ENFORCE_Q_LIMS=enforce_q_lims, PF_TOL=tolerance_kva * 1e-3,
                     PF_ALG=algorithm_pypower_dict[algorithm], **kwargs)
    ppopt['PF_MAX_IT'] = max_iteration
    ppopt['PF_MAX_IT_GS'] = max_iteration
    ppopt['PF_MAX_IT_FD'] = max_iteration
    return init, ac, numba, recycle, ppopt


def _dc_runpf(ppci, ppopt):
    baseMVA, bus, gen, branch, ref, pv, pq, on, gbus, _ = _get_pf_variables_from_ppci(ppci)

    ppci["bus"][:, VM] = 1.0
    if ppopt["VERBOSE"]:
        print(' -- DC Power Flow\n')

    ## initial state
    Va0 = bus[:, VA] * (pi / 180)

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
    bus[:, VA] = Va * (180 / pi)
    ## update Pg for slack generator (1st gen at ref bus)
    ## (note: other gens at ref bus are accounted for in Pbus)
    ##      Pg = Pinj + Pload + Gs
    ##      newPg = oldPg + newPinj - oldPinj

    refgen = zeros(len(ref), dtype=int)
    for k in range(len(ref)):
        temp = find(gbus == ref[k])
        refgen[k] = on[temp[0]]
    gen[refgen, PG] = real(gen[refgen, PG] + (B[ref, :] * Va - Pbus[ref]) * baseMVA)
    success = 1

    # store results from DC powerflow for AC powerflow
    ppci = _store_results_from_pf_in_ppci(ppci, bus, gen, branch)

    return ppci, success


def _ac_runpf(ppci, ppopt, numba, recycle):
    numba, makeYbus = _import_numba_extensions_if_flag_is_true(numba)

    if ppopt["VERBOSE"] > 0:
        _print_info_about_solver(ppopt['PF_ALG'])

    if ppopt["ENFORCE_Q_LIMS"]:
        ppci, success, bus, gen, branch = _run_ac_pf_with_qlims_enforced(ppci, recycle, makeYbus, ppopt)

    else:
        ppci, success, bus, gen, branch = _run_ac_pf_without_qlims_enforced(ppci, recycle, makeYbus, ppopt)

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


def _print_info_about_solver(alg):
    if alg == 1:
        solver = 'Newton'
    elif alg == 2:
        solver = 'fast-decoupled, XB'
    elif alg == 3:
        solver = 'fast-decoupled, BX'
    elif alg == 4:
        solver = 'Gauss-Seidel'
    else:
        solver = 'unknown'
    logger.info(' -- AC Power Flow (%s)\n' % solver)


def _get_Y_bus(ppci, recycle, makeYbus, baseMVA, bus, branch):
    if recycle["Ybus"] and ppci["internal"]["Ybus"].size:
        Ybus, Yf, Yt = ppci["internal"]['Ybus'], ppci["internal"]['Yf'], ppci["internal"]['Yt']
    else:
        ## build admittance matrices
        Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
        if recycle["Ybus"]:
            ppci["internal"]['Ybus'], ppci["internal"]['Yf'], ppci["internal"]['Yt'] = Ybus, Yf, Yt

    return ppci, Ybus, Yf, Yt


def _run_ac_pf_without_qlims_enforced(ppci, recycle, makeYbus, ppopt):
    baseMVA, bus, gen, branch, ref, pv, pq, on, gbus, V0 = _get_pf_variables_from_ppci(ppci)

    ppci, Ybus, Yf, Yt = _get_Y_bus(ppci, recycle, makeYbus, baseMVA, bus, branch)

    ## compute complex bus power injections [generation - load]
    Sbus = makeSbus(baseMVA, bus, gen)

    ## run the power flow
    V, success = _call_power_flow_function(baseMVA, bus, branch, Ybus, Sbus, V0, ref, pv, pq, ppopt)

    ## update data matrices with solution
    bus, gen, branch = pfsoln(baseMVA, bus, gen, branch, Ybus, Yf, Yt, V, ref, pv, pq)

    return ppci, success, bus, gen, branch


def _run_ac_pf_with_qlims_enforced(ppci, recycle, makeYbus, ppopt):
    baseMVA, bus, gen, branch, ref, pv, pq, on, gbus, V0 = _get_pf_variables_from_ppci(ppci)

    qlim = ppopt["ENFORCE_Q_LIMS"]
    limited = []  ## list of indices of gens @ Q lims
    fixedQg = zeros(gen.shape[0])  ## Qg of gens at Q limits

    while True:
        ppci, success, bus, gen, branch = _run_ac_pf_without_qlims_enforced(ppci, recycle, makeYbus, ppopt)

        ## find gens with violated Q constraints
        gen_status = gen[:, GEN_STATUS] > 0
        qg_max_lim = gen[:, QG] > gen[:, QMAX]
        qg_min_lim = gen[:, QG] < gen[:, QMIN]

        mx = find(gen_status & qg_max_lim)
        mn = find(gen_status & qg_min_lim)

        if len(mx) > 0 or len(mn) > 0:  ## we have some Q limit violations
            # No PV generators
            if len(pv) == 0:
                if ppopt["VERBOSE"]:
                    if len(mx) > 0:
                        logger.info('Gen %d [only one left] exceeds upper Q limit : INFEASIBLE PROBLEM\n' % mx + 1)
                    else:
                        logger.info('Gen %d [only one left] exceeds lower Q limit : INFEASIBLE PROBLEM\n' % mn + 1)

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

            if ppopt["VERBOSE"] and len(mx) > 0:
                for i in range(len(mx)):
                    logger.info('Gen ' + str(mx[i] + 1) + ' at upper Q limit, converting to PQ bus\n')

            if ppopt["VERBOSE"] and len(mn) > 0:
                for i in range(len(mn)):
                    logger.info('Gen ' + str(mn[i] + 1) + ' at lower Q limit, converting to PQ bus\n')

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
            ref_temp = ref
            ref, pv, pq = bustypes(bus, gen)
            if ppopt["VERBOSE"] and ref != ref_temp:
                print('Bus %d is new slack bus\n' % ref)

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


def _call_power_flow_function(baseMVA, bus, branch, Ybus, Sbus, V0, ref, pv, pq, ppopt):
    alg = ppopt["PF_ALG"]
    # alg == 1 was deleted = nr -> moved as own pandapower solver
    if alg == 2 or alg == 3:
        Bp, Bpp = makeB(baseMVA, bus, real(branch), alg)
        V, success, _ = fdpf(Ybus, Sbus, V0, Bp, Bpp, ref, pv, pq, ppopt)
    elif alg == 4:
        V, success, _ = gausspf(Ybus, Sbus, V0, ref, pv, pq, ppopt)
    else:
        raise ValueError('Only Newton''s method, fast-decoupled, and '
                         'Gauss-Seidel power flow algorithms currently '
                         'implemented.\n')

    return V, success
