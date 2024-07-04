# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


from time import perf_counter

from numpy import flatnonzero as find, r_, zeros, argmax, setdiff1d, union1d, any, int32, \
    sum as np_sum, abs as np_abs, int64

from pandapower.pf.ppci_variables import _get_pf_variables_from_ppci, _store_results_from_pf_in_ppci
from pandapower.pf.run_dc_pf import _run_dc_pf
from pandapower.pypower.bustypes import bustypes
from pandapower.pypower.idx_bus import BUS_I, PD, QD, BUS_TYPE, PQ, PV, GS, BS, SL_FAC as SL_FAC_BUS
from pandapower.pypower.idx_gen import PG, QG, QMAX, QMIN, GEN_BUS, GEN_STATUS, SL_FAC
from pandapower.pypower.makeSbus import makeSbus
from pandapower.pypower.makeYbus import makeYbus as makeYbus_pypower
from pandapower.pypower.newtonpf import newtonpf
from pandapower.pypower.pfsoln import _update_v
from pandapower.pypower.pfsoln import pfsoln as pfsoln_pypower
from pandapower.auxiliary import version_check

try:
    from pandapower.pf.makeYbus_numba import makeYbus as makeYbus_numba
    from pandapower.pf.pfsoln_numba import pfsoln as pfsoln_numba, pf_solution_single_slack
    version_check('numba')
    numba_installed = True
except ImportError:
    numba_installed = False

try:
    from lightsim2grid.newtonpf import newtonpf_new as newton_ls
except ImportError:
    newton_ls = None


def _run_newton_raphson_pf(ppci, options):
    """
    Runs a Newton-Raphson power flow.

    INPUT
    ppci (dict) - the "internal" ppc (without out ot service elements and sorted elements)
    options(dict) - options for the power flow

    """
    t0 = perf_counter()
    # we cannot run DC pf before running newton with distributed slack because the slacks come pre-solved after the DC pf
    if isinstance(options["init_va_degree"], str) and options["init_va_degree"] == "dc":
        if options['distributed_slack']:
            pg_copy = ppci['gen'][:, PG].copy()
            pd_copy = ppci['bus'][:, PD].copy()
            ppci = _run_dc_pf(ppci, options["recycle"])
            ppci['gen'][:, PG] = pg_copy
            ppci['bus'][:, PD] = pd_copy
        else:
            ppci = _run_dc_pf(ppci, options["recycle"])
    if options["enforce_q_lims"]:
        ppci, success, iterations, bus, gen, branch = _run_ac_pf_with_qlims_enforced(ppci, options)
    else:
        ppci, success, iterations = _run_ac_pf_without_qlims_enforced(ppci, options)
        # update data matrices with solution store in ppci
        bus, gen, branch = ppci_to_pfsoln(ppci, options)
    et = perf_counter() - t0
    ppci = _store_results_from_pf_in_ppci(ppci, bus, gen, branch, success, iterations, et)
    return ppci


def ppci_to_pfsoln(ppci, options, limited_gens=None):
    internal = ppci["internal"]
    if options["only_v_results"]:
        # time series relevant hack which ONLY saves V from ppci
        _update_v(internal["bus"], internal["V"])
        return internal["bus"], internal["gen"], internal["branch"]
    else:
        # reads values from internal ppci storage to bus, gen, branch and returns it
        if options['distributed_slack']:
            # consider buses with non-zero slack weights as if they were slack buses,
            # and gens with non-zero slack weights as if they were reference machines
            # this way, the function pfsoln will extract results for distributed slack gens, too
            # also, the function pfsoln will extract results for the PQ buses for xwards
            gens_with_slack_weights = find(internal["gen"][:, SL_FAC] != 0)
            # gen_buses_with_slack_weights = internal["gen"][gens_with_slack_weights, GEN_BUS].astype(int32)
            buses_with_slack_weights = internal["bus"][find(internal["bus"][:, SL_FAC_BUS] != 0), BUS_I].astype(int32)
            # buses_with_slack_weights = union1d(gen_buses_with_slack_weights, buses_with_slack_weights)
            ref = union1d(internal["ref"], buses_with_slack_weights)
            ref_gens = union1d(internal["ref_gens"], gens_with_slack_weights)
        else:
            ref = internal["ref"]
            ref_gens = internal["ref_gens"]

        makeYbus, pfsoln = _get_numba_functions(ppci, options)

        # todo: this can be dropped if Ybus is returned from Newton and has the latest Ybus status:
        if options["tdpf"]:
            # needs to be updated to match the new R because of the temperature
            internal["Ybus"], internal["Yf"], internal["Yt"] = makeYbus(internal["baseMVA"], internal["bus"], internal["branch"])

        # todo: here Ybus_svc, Ybus_tcsc must be used (Ybus = Ybus + Ybus_svc + Ybus_tcsc)
        result_pfsoln = pfsoln(internal["baseMVA"], internal["bus"], internal["gen"], internal["branch"],
                               internal["svc"], internal["tcsc"], internal["ssc"],
                               internal["Ybus"], internal["Yf"], internal["Yt"], internal["V"],
                               ref, ref_gens, limited_gens=limited_gens)
        return result_pfsoln


def _get_Y_bus(ppci, options, makeYbus, baseMVA, bus, branch):
    recycle = options["recycle"]

    if isinstance(recycle, dict) and not recycle["trafo"] and ppci["internal"]["Ybus"].size:
        Ybus, Yf, Yt = ppci["internal"]['Ybus'], ppci["internal"]['Yf'], ppci["internal"]['Yt']
    else:
        # build admittance matrices
        Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)

    return ppci, Ybus, Yf, Yt


def _get_numba_functions(ppci, options):
    """
    pfsoln from pypower maybe slow in some cases. This function chooses the fastest for the given pf calculation
    """
    if options["numba"] and numba_installed:
        makeYbus = makeYbus_numba
        shunt_in_net = any(ppci["bus"][:, BS]) or any(ppci["bus"][:, GS])
        # faster pfsoln function if only one slack is in the grid and no gens
        pfsoln = pf_solution_single_slack if ppci["gen"].shape[0] == 1 \
                                             and not options["voltage_depend_loads"] \
                                             and not options['distributed_slack'] \
                                             and not shunt_in_net \
            else pfsoln_numba
    else:
        makeYbus = makeYbus_pypower
        pfsoln = pfsoln_pypower
    return makeYbus, pfsoln


def _store_internal(ppci, internal_storage):
    # internal storage is a dict with the variables to store in net["_ppc"]["internal"]
    for key, val in internal_storage.items():
        ppci["internal"][key] = val
    return ppci


def _get_Sbus(ppci, recycle=None):
    baseMVA, bus, gen = ppci["baseMVA"], ppci["bus"], ppci["gen"]
    if not isinstance(recycle, dict) or "Sbus" not in ppci["internal"]:
        return makeSbus(baseMVA, bus, gen)
    if recycle["bus_pq"] or recycle["gen"]:
        return makeSbus(baseMVA, bus, gen)
    return ppci["internal"]["Sbus"]


def _run_ac_pf_without_qlims_enforced(ppci, options):
    makeYbus, pfsoln = _get_numba_functions(ppci, options)

    baseMVA, bus, gen, branch, svc, tcsc, ssc, ref, pv, pq, *_, V0, ref_gens = _get_pf_variables_from_ppci(ppci)

    ppci, Ybus, Yf, Yt = _get_Y_bus(ppci, options, makeYbus, baseMVA, bus, branch)

    # compute complex bus power injections [generation - load]
    Sbus = _get_Sbus(ppci, options["recycle"])


    # run the newton power flow
    if options["lightsim2grid"]:
        V, success, iterations, J, Vm_it, Va_it = newton_ls(Ybus.tocsc(), Sbus, V0, ref, pv, pq, ppci, options)
        T = None
        r_theta_kelvin_per_mw = None
    else:
        V, success, iterations, J, Vm_it, Va_it, r_theta_kelvin_per_mw, T = newtonpf(Ybus, Sbus, V0, ref, pv, pq, ppci, options, makeYbus)
        # due to TPDF, SVC, TCSC, the Ybus matrices can be updated in the newtonpf and stored in ppci["internal"],
        # so we extract them here for later use:
        Ybus, Ybus_svc, Ybus_tcsc, Ybus_ssc = (ppci["internal"].get(key) for key in ("Ybus", "Ybus_svc", "Ybus_tcsc", "Ybus_ssc"))
        Ybus = Ybus + Ybus_svc + Ybus_tcsc + Ybus_ssc

    # keep "internal" variables in  memory / net["_ppc"]["internal"] -> needed for recycle.
    ppci = _store_internal(ppci, {"J": J, "Vm_it": Vm_it, "Va_it": Va_it, "bus": bus, "gen": gen, "branch": branch,
                                  "svc": svc, "tcsc": tcsc, "ssc": ssc, "baseMVA": baseMVA, "V": V,
                                  "pv": pv, "pq": pq, "ref": ref,
                                  "Sbus": Sbus, "ref_gens": ref_gens, "Ybus": Ybus, "Yf": Yf, "Yt": Yt,
                                  "r_theta_kelvin_per_mw": r_theta_kelvin_per_mw, "T": T})

    return ppci, success, iterations


def _run_ac_pf_with_qlims_enforced(ppci, options):
    baseMVA, bus, gen, branch, svc, tcsc, ssc, ref, pv, pq, on, *_, V0, ref_gens = _get_pf_variables_from_ppci(ppci)
    bus_backup_p_q = bus[:, [PD, QD]].copy()
    gen_backup_p = gen[:, PG].copy()

    qlim = options["enforce_q_lims"]
    limited = []  # list of indices of gens @ Q lims
    fixedQg = zeros(gen.shape[0])  # Qg of gens at Q limits

    while True:
        ppci, success, iterations = _run_ac_pf_without_qlims_enforced(ppci, options)
        gen[:, PG] = gen_backup_p
        bus[:, PD] = bus_backup_p_q[:, 0]
        bus, gen, branch = ppci_to_pfsoln(ppci, options, limited)

        # find gens with violated Q constraints
        gen_status = gen[:, GEN_STATUS] > 0
        qg_max_lim = gen[:, QG] > gen[:, QMAX]
        qg_min_lim = gen[:, QG] < gen[:, QMIN]

        mx = setdiff1d(find(gen_status & qg_max_lim), ref_gens)
        mn = setdiff1d(find(gen_status & qg_min_lim), ref_gens)

        if len(mx) > 0 or len(mn) > 0:  # we have some Q limit violations
            # one at a time?
            if qlim == 2:  # fix largest violation, ignore the rest
                k = argmax(r_[gen[mx, QG] - gen[mx, QMAX],
                              gen[mn, QMIN] - gen[mn, QG]])
                if k > len(mx):
                    mn = mn[k - len(mx)]
                    mx = []
                else:
                    mx = mx[k]
                    mn = []

            # save corresponding limit values
            fixedQg[mx] = gen[mx, QMAX]
            fixedQg[mn] = gen[mn, QMIN]
            mx = r_[mx, mn].astype(int64)

            # convert to PQ bus
            gen[mx, QG] = fixedQg[mx]  # set Qg to binding
            #            if len(ref) > 1 and any(bus[gen[mx, GEN_BUS].astype(np.int64), BUS_TYPE] == REF):
            #                raise ValueError('Sorry, pandapower cannot enforce Q '
            #                                 'limits for slack buses in systems '
            #                                 'with multiple slacks.')

            changed_gens = gen[mx, GEN_BUS].astype(int64)
            bus[setdiff1d(changed_gens, ref), BUS_TYPE] = PQ  # & set bus type to PQ

            # update bus index lists of each type of bus
            ref, pv, pq = bustypes(bus, gen)

            limited = r_[limited, mx].astype(int64)

            for i in range(len(limited)):  # [one at a time, since they may be at same bus]
                gen[limited[i], GEN_STATUS] = 0  # temporarily turn off gen,
                bi = gen[limited[i], GEN_BUS].astype(int64)  # adjust load accordingly,
                bus[bi, [PD, QD]] = (bus[bi, [PD, QD]] - gen[limited[i], [PG, QG]])
        else:
            break  # no more generator Q limits violated

    if len(limited) > 0:
        # restore injections from limited gens [those at Q limits]
        bus[setdiff1d(changed_gens, ref), BUS_TYPE] = PV  # & set bus type back to PV
        gen[limited, QG] = fixedQg[limited]  # restore Qg value,
        gen[limited, GEN_STATUS] = 1  # turn gens back on
        bus[:, [PD, QD]] = bus_backup_p_q
    return ppci, success, iterations, bus, gen, branch
