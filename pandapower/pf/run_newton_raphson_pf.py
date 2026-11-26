# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import logging
from time import perf_counter

from numpy import (
    any,
    argmax,
    array,
    inf,
    int32,
    int64,
    r_,
    setdiff1d,
    union1d,
    zeros,
)
from numpy import (
    flatnonzero as find,
)

from pandapower.auxiliary import version_check
from pandapower.pf.ppci_variables import _get_pf_variables_from_ppci, _store_results_from_pf_in_ppci
from pandapower.pf.run_dc_pf import _run_dc_pf
from pandapower.pypower.bustypes import bustypes
from pandapower.pypower.idx_bus import BS, BUS_I, BUS_TYPE, GS, PD, PQ, PV, QD
from pandapower.pypower.idx_bus import SL_FAC as SL_FAC_BUS
from pandapower.pypower.idx_gen import GEN_BUS, GEN_STATUS, PG, QG, QMAX, QMIN, SL_FAC
from pandapower.pypower.idx_svc import (
    SVC_CONTROLLABLE,
    SVC_MAX_FIRING_ANGLE,
    SVC_MIN_FIRING_ANGLE,
    SVC_STATUS,
    SVC_THYRISTOR_FIRING_ANGLE,
)
from pandapower.pypower.idx_tcsc import (
    TCSC_CONTROLLABLE,
    TCSC_MAX_FIRING_ANGLE,
    TCSC_MIN_FIRING_ANGLE,
    TCSC_STATUS,
    TCSC_THYRISTOR_FIRING_ANGLE,
)
from pandapower.pypower.make_s_bus import makeSbus
from pandapower.pypower.make_y_bus import makeYbus as makeYbus_pypower
from pandapower.pypower.newtonpf import newtonpf
from pandapower.pypower.pfsoln import _update_v
from pandapower.pypower.pfsoln import pfsoln as pfsoln_pypower

try:
    from pandapower.pf.make_y_bus_numba import makeYbus as makeYbus_numba
    from pandapower.pf.pfsoln_numba import pf_solution_single_slack
    from pandapower.pf.pfsoln_numba import pfsoln as pfsoln_numba

    version_check("numba")
    numba_installed = True
except ImportError:
    numba_installed = False

try:
    from lightsim2grid.newtonpf import newtonpf_new as newton_ls
except ImportError:
    newton_ls = None

logger = logging.getLogger(__name__)

# columns of the firing angle, its limits and the control flag in ppci["svc"] and ppci["tcsc"]
_FACTS_COLUMNS = {
    "svc": (SVC_THYRISTOR_FIRING_ANGLE, SVC_MIN_FIRING_ANGLE, SVC_MAX_FIRING_ANGLE, SVC_CONTROLLABLE),
    "tcsc": (TCSC_THYRISTOR_FIRING_ANGLE, TCSC_MIN_FIRING_ANGLE, TCSC_MAX_FIRING_ANGLE, TCSC_CONTROLLABLE),
}
# a firing angle this far outside its limits (in radians) counts as a violation
_FACTS_ANGLE_TOL = 1e-8


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
        if options["distributed_slack"]:
            pg_copy = ppci["gen"][:, PG].copy()
            pd_copy = ppci["bus"][:, PD].copy()
            ppci = _run_dc_pf(ppci, options["recycle"])
            ppci["gen"][:, PG] = pg_copy
            ppci["bus"][:, PD] = pd_copy
        else:
            ppci = _run_dc_pf(ppci, options["recycle"])
    if options["enforce_q_lims"] or options.get("enforce_facts_limits", False):
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
        if options["distributed_slack"]:
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
            internal["Ybus"], internal["Yf"], internal["Yt"] = makeYbus(
                internal["baseMVA"], internal["bus"], internal["branch"]
            )

        # todo: here Ybus_svc, Ybus_tcsc must be used (Ybus = Ybus + Ybus_svc + Ybus_tcsc)
        result_pfsoln = pfsoln(
            internal["baseMVA"],
            internal["bus"],
            internal["gen"],
            internal["branch"],
            internal["svc"],
            internal["tcsc"],
            internal["ssc"],
            internal["vsc"],
            internal["Ybus"],
            internal["Yf"],
            internal["Yt"],
            internal["V"],
            ref,
            ref_gens,
            limited_gens=limited_gens,
        )
        return result_pfsoln


def _get_Y_bus(ppci, options, makeYbus, baseMVA, bus, branch):
    recycle = options["recycle"]

    if isinstance(recycle, dict) and not recycle["trafo"] and ppci["internal"]["Ybus"].size:
        Ybus, Yf, Yt = ppci["internal"]["Ybus"], ppci["internal"]["Yf"], ppci["internal"]["Yt"]
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
        pfsoln = (
            pf_solution_single_slack
            if ppci["gen"].shape[0] == 1
            and not options["voltage_depend_loads"]
            and not options["distributed_slack"]
            and not shunt_in_net
            else pfsoln_numba
        )
    else:
        makeYbus = makeYbus_pypower
        pfsoln = pfsoln_pypower
    return makeYbus, pfsoln


def _get_Sbus(ppci, recycle=None):
    baseMVA, bus, gen = ppci["baseMVA"], ppci["bus"], ppci["gen"]
    if not isinstance(recycle, dict) or "Sbus" not in ppci["internal"]:
        return makeSbus(baseMVA, bus, gen)
    if recycle["bus_pq"] or recycle["gen"]:
        return makeSbus(baseMVA, bus, gen)
    return ppci["internal"]["Sbus"]


def _run_ac_pf_without_qlims_enforced(ppci, options):
    makeYbus, _ = _get_numba_functions(ppci, options)

    baseMVA, bus, gen, branch, svc, tcsc, ssc, vsc, ref, pv, pq, *_, V0, ref_gens = _get_pf_variables_from_ppci(
        ppci, True
    )

    ppci, Ybus, Yf, Yt = _get_Y_bus(ppci, options, makeYbus, baseMVA, bus, branch)

    # compute complex bus power injections [generation - load]
    Sbus = _get_Sbus(ppci, options["recycle"])

    # run the newton power flow
    if options["lightsim2grid"]:
        V, success, iterations, J, Vm_it, Va_it = newton_ls(Ybus.tocsc(), Sbus, V0, ref, pv, pq, ppci, options)
        T = None
        r_theta_kelvin_per_mw = None
    else:
        V, success, iterations, J, Vm_it, Va_it, r_theta_kelvin_per_mw, T = newtonpf(
            Ybus, Sbus, V0, ref, pv, pq, ppci, options, makeYbus
        )
        # due to TPDF, SVC, TCSC, the Ybus matrices can be updated in the newtonpf and stored in ppci["internal"],
        # so we extract them here for later use:
        Ybus = (
            ppci["internal"]["Ybus"]
            + ppci["internal"]["Ybus_svc"]
            + ppci["internal"]["Ybus_tcsc"]
            + ppci["internal"]["Ybus_ssc"]
            + ppci["internal"]["Ybus_vsc"]
        )

    # keep "internal" variables in  memory / net["_ppc"]["internal"] -> needed for recycle.
    ppci["internal"].update(
        {
            "J": J,
            "Vm_it": Vm_it,
            "Va_it": Va_it,
            "bus": bus,
            "gen": gen,
            "branch": branch,
            "svc": svc,
            "tcsc": tcsc,
            "ssc": ssc,
            "vsc": vsc,
            "baseMVA": baseMVA,
            "V": V,
            "pv": pv,
            "pq": pq,
            "ref": ref,
            "Sbus": Sbus,
            "ref_gens": ref_gens,
            "Ybus": Ybus,
            "Yf": Yf,
            "Yt": Yt,
            "r_theta_kelvin_per_mw": r_theta_kelvin_per_mw,
            "T": T,
        }
    )

    return ppci, success, iterations


def _facts_devices(ppci):
    """In-service SVCs and TCSCs that control a setpoint, as ("svc" | "tcsc", row) pairs."""
    devices = []
    for table, (_, _, _, ctrl) in _FACTS_COLUMNS.items():
        status = SVC_STATUS if table == "svc" else TCSC_STATUS
        rows = find((ppci[table][:, status].real > 0) & (ppci[table][:, ctrl].real > 0))
        devices += [(table, int(row)) for row in rows]
    return devices


def _set_facts_control(ppci, device, controllable, angle=None):
    table, row = device
    ang, _, _, ctrl = _FACTS_COLUMNS[table]
    ppci[table][row, ctrl] = controllable
    if angle is not None:
        ppci[table][row, ang] = angle


def _facts_angle_and_limits(ppci, device):
    table, row = device
    ang, lo, hi, _ = _FACTS_COLUMNS[table]
    return ppci[table][row, ang].real, ppci[table][row, lo].real, ppci[table][row, hi].real


def _largest_facts_violation(ppci, devices, held):
    """
    Returns the device whose firing angle is furthest outside its limits, relative to the width of its
    range, and the limit it crossed, or None if no device that is still controlling violates its limits.
    """
    largest = None
    for device in devices:
        if device in held:
            continue
        angle, lo, hi = _facts_angle_and_limits(ppci, device)
        excess = max(angle - hi, lo - angle)
        if excess > _FACTS_ANGLE_TOL:
            relative = excess / (hi - lo) if hi > lo else inf
            if largest is None or relative > largest[0]:
                largest = (relative, device, hi if angle > hi else lo)
    return None if largest is None else largest[1:]


def _release_facts_device(ppci, options, held):
    """
    Lets each device that is held at a firing angle limit control its setpoint again, one at a time. The first
    device whose firing angle stays within its limits is released and returned. All other devices are held at
    their limits again. Returns None if no device can be released.
    """
    for device in sorted(held):
        _set_facts_control(ppci, device, True)
        ppci, success, _ = _run_ac_pf_without_qlims_enforced(ppci, options)
        angle, lo, hi = _facts_angle_and_limits(ppci, device)
        if success and lo - _FACTS_ANGLE_TOL <= angle <= hi + _FACTS_ANGLE_TOL:
            del held[device]
            return device
        _set_facts_control(ppci, device, False, held[device])
    return None


def _run_ac_pf_with_qlims_enforced(ppci, options):
    """
    Runs the Newton-Raphson power flow in an outer loop that enforces the reactive power limits of generators
    (option enforce_q_lims) and the firing angle limits of SVCs and TCSCs (option enforce_facts_limits).

    Generators that violate their reactive power limits are fixed at the limit and are never released.

    With enforce_facts_limits, an SVC or TCSC whose firing angle leaves [min_angle, max_angle] is held at the
    limit it crossed, one device per round (the largest violation relative to the device's range first).
    Once no limit is violated, each held device is allowed to control its setpoint again; it is released if
    its firing angle then stays within its limits. If holding a device leads back to a combination of held
    devices that was already solved, the devices keep switching between limit and control, so the loop stops
    and warns.
    """
    _, bus, gen, branch, _, _, _, _, ref, *_, ref_gens = _get_pf_variables_from_ppci(ppci)
    bus_backup_p_q = bus[:, [PD, QD]].copy()
    gen_backup_p = gen[:, PG].copy()

    qlim = options["enforce_q_lims"]
    limited = []  # list of indices of gens @ Q lims
    fixedQg = zeros(gen.shape[0])  # Qg of gens at Q limits

    facts = _facts_devices(ppci) if options.get("enforce_facts_limits", False) else []
    held = {}  # FACTS device -> firing angle limit at which it is held
    seen = set()  # combinations of limited gens and held devices that have been solved
    release_checked = set()  # combinations for which no held device could be released
    max_rounds = 10 * (len(facts) + gen.shape[0]) + 10
    rounds = 0
    stop = False

    while True:
        ppci, success, iterations = _run_ac_pf_without_qlims_enforced(ppci, options)
        gen[:, PG] = gen_backup_p
        bus[:, PD] = bus_backup_p_q[:, 0]
        bus, gen, branch = ppci_to_pfsoln(ppci, options, limited)
        rounds += 1
        if stop:
            break
        if rounds > max_rounds:
            logger.warning(
                "The reactive power and firing angle limits were not settled after %d power flows; "
                "the result may violate some limits." % max_rounds
            )
            break

        # find gens with violated Q constraints
        if qlim:
            gen_status = gen[:, GEN_STATUS] > 0
            qg_max_lim = gen[:, QG] > gen[:, QMAX]
            qg_min_lim = gen[:, QG] < gen[:, QMIN]

            mx = setdiff1d(find(gen_status & qg_max_lim), ref_gens)
            mn = setdiff1d(find(gen_status & qg_min_lim), ref_gens)
        else:
            mx = mn = array([], dtype=int64)

        if len(mx) > 0 or len(mn) > 0:  # we have some Q limit violations
            # one at a time?
            if qlim == 2:  # fix largest violation, ignore the rest
                k = argmax(r_[gen[mx, QG] - gen[mx, QMAX], gen[mn, QMIN] - gen[mn, QG]])
                if k >= len(mx):
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
            ref, _, _ = bustypes(bus, gen)

            limited = r_[limited, mx].astype(int64)

            for i in range(len(limited)):  # [one at a time, since they may be at same bus]
                gen[limited[i], GEN_STATUS] = 0  # temporarily turn off gen,
                bi = gen[limited[i], GEN_BUS].astype(int64)  # adjust load accordingly,
                bus[bi, [PD, QD]] = bus[bi, [PD, QD]] - gen[limited[i], [PG, QG]]
        elif facts:
            state = (tuple(sorted(limited)), tuple(sorted(held)))
            seen.add(state)
            violation = _largest_facts_violation(ppci, facts, held)
            if violation is not None:
                device, limit = violation
                held[device] = limit
                _set_facts_control(ppci, device, False, limit)
                if (tuple(sorted(limited)), tuple(sorted(held))) in seen:
                    logger.warning(
                        "Some SVCs or TCSCs keep switching between their firing angle limits and controlling "
                        "their setpoints. The power flow stops with %d SVC(s) and %d TCSC(s) held at their "
                        "firing angle limits (see column at_limit of res_svc and res_tcsc); their setpoints "
                        "may not be met." % tuple(sum(1 for d in held if d[0] == t) for t in ("svc", "tcsc"))
                    )
                    stop = True  # solve once more for the current combination, then stop
            elif state in release_checked:
                break  # no limit violated and no held device can be released
            else:
                release_checked.add(state)
                # A released device controls its setpoint within its limits in the solution just found. If no
                # device is released, the next round solves again with all held devices at their limits.
                _release_facts_device(ppci, options, held)
        else:
            break  # no more generator Q limits violated

    for device in facts:  # restore the control flags; held devices keep their limit angles
        _set_facts_control(ppci, device, True)

    if len(limited) > 0:
        # restore injections from limited gens [those at Q limits]
        bus[setdiff1d(changed_gens, ref), BUS_TYPE] = PV  # & set bus type back to PV
        gen[limited, QG] = fixedQg[limited]  # restore Qg value,
        gen[limited, GEN_STATUS] = 1  # turn gens back on
        bus[:, [PD, QD]] = bus_backup_p_q
    return ppci, success, iterations, bus, gen, branch
