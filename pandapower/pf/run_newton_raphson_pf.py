# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import logging
from copy import deepcopy
from time import perf_counter

from numpy import (
    any,
    argmax,
    array,
    inf,
    int32,
    int64,
    isfinite,
    pi,
    r_,
    setdiff1d,
    union1d,
    zeros,
)
from numpy import (
    flatnonzero as find,
)
from numpy.linalg import LinAlgError

from pandapower.auxiliary import version_check
from pandapower.pf.make_y_bus_facts import calc_y_svc_pu
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
    TCSC_X_CVAR,
    TCSC_X_L,
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
    for device in devices:
        _, lo, hi = _facts_angle_and_limits(ppci, device)
        if not (isfinite(lo) and isfinite(hi) and pi / 2 <= lo <= hi <= pi):
            raise ValueError("Controllable SVC/TCSC firing angle limits must satisfy 90 <= min <= max <= 180 degrees")
        table, row = device
        if table == "tcsc":
            x_l, x_c = ppci[table][row, [TCSC_X_L, TCSC_X_CVAR]].real
            if calc_y_svc_pu(lo, x_l, x_c) * calc_y_svc_pu(hi, x_l, x_c) <= 0:
                raise ValueError("Controllable TCSC firing angle limits must not include its resonance angle")
    return sorted(devices, key=lambda device: _facts_sort_key(ppci, device))


def _facts_sort_key(ppci, device):
    # Use physical properties for ties and release order, independent of table row order.
    table, row = device
    angle_column = _FACTS_COLUMNS[table][0]
    return table, tuple(ppci[table][row, :angle_column].real), _facts_angle_and_limits(ppci, device)[1:]


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
    return largest


def _release_facts_device(ppci, held, solve):
    """Try releasing one held device without changing the accepted solution on failed trials."""
    for device in sorted(held, key=lambda device: _facts_sort_key(ppci, device)):
        trial = deepcopy(ppci)
        _set_facts_control(trial, device, True)
        try:
            trial, success, _, _, _, _ = solve(trial)
        except (FloatingPointError, LinAlgError):
            continue
        angle, lo, hi = _facts_angle_and_limits(trial, device)
        if success and lo - _FACTS_ANGLE_TOL <= angle <= hi + _FACTS_ANGLE_TOL:
            del held[device]
            return trial, device
    return ppci, None


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

    enforce_facts = options.get("enforce_facts_limits", False)
    facts = _facts_devices(ppci) if enforce_facts else []
    held = {}  # FACTS device -> firing angle limit at which it is held
    seen = set()  # combinations of limited gens and held devices that have been solved
    max_rounds = 10 * (len(facts) + gen.shape[0]) + 10
    rounds = 0
    # Limit enforcement changes generator injections even if the caller did not.
    # Recompute Sbus instead of recycling it across different control states.
    solve_options = dict(options, recycle=None) if enforce_facts else options

    def solve(case):
        # Each extra FACTS solve must start from the same loads. In particular, do not
        # repeatedly subtract the injections of generators already fixed at a Q limit.
        case_bus, case_gen = case["bus"], case["gen"]
        if enforce_facts:
            case_bus[:, [PD, QD]] = bus_backup_p_q
            case_gen[:, PG] = gen_backup_p
            case_gen[limited, QG] = fixedQg[limited]
            for index in limited:
                bi = int(case_gen[index, GEN_BUS])
                case_bus[bi, [PD, QD]] -= case_gen[index, [PG, QG]]
        case, success, iterations = _run_ac_pf_without_qlims_enforced(case, solve_options)
        case_gen[:, PG] = gen_backup_p
        case_bus[:, PD] = bus_backup_p_q[:, 0]
        case_bus, case_gen, case_branch = ppci_to_pfsoln(case, options, limited)
        if enforce_facts:
            case_bus[:, [PD, QD]] = bus_backup_p_q
        return case, success, iterations, case_bus, case_gen, case_branch

    while True:
        ppci, success, iterations, bus, gen, branch = solve(ppci)
        rounds += 1
        if enforce_facts:
            state = (tuple(sorted(limited)), tuple(sorted(held.items())))
            if success and state in seen:
                logger.warning(
                    "Some SVCs or TCSCs keep switching between their firing angle limits and controlling "
                    "their setpoints. The power flow stops with %d SVC(s) and %d TCSC(s) held at their "
                    "firing angle limits (see column at_limit of res_svc and res_tcsc); their setpoints "
                    "or other firing angle limits may not be met.",
                    *(sum(1 for d in held if d[0] == table) for table in ("svc", "tcsc")),
                )
                break
            if success:
                seen.add(state)
            if rounds >= max_rounds:
                logger.warning(
                    "The reactive power and firing angle limits were not settled after %d power flows; "
                    "the result may violate some limits.",
                    max_rounds,
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

        if enforce_facts:
            # Compare unlike constraints by their adjustable ranges, fixing only one
            # device per round. Preserve the existing Q-only algorithm when opted out.
            violation = _largest_facts_violation(ppci, facts, held)
            if len(mx) or len(mn):
                indices = r_[mx, mn].astype(int64)
                excess = r_[gen[mx, QG] - gen[mx, QMAX], gen[mn, QMIN] - gen[mn, QG]]
                widths = gen[indices, QMAX] - gen[indices, QMIN]
                relative = array([value / width if width > 0 else inf for value, width in zip(excess, widths)])
                k = int(argmax(relative))
                if violation is None or relative[k] >= violation[0]:
                    if k >= len(mx):
                        mn, mx = array([mn[k - len(mx)]]), array([], dtype=int64)
                    else:
                        mx, mn = array([mx[k]]), array([], dtype=int64)
                else:
                    mx = mn = array([], dtype=int64)
            if not (len(mx) or len(mn)):
                if violation is not None:
                    _, device, limit = violation
                    held[device] = limit
                    _set_facts_control(ppci, device, False, limit)
                    continue
                if not success:
                    break
                ppci, released = _release_facts_device(ppci, held, solve)
                if released is None:
                    break
                continue

        if len(mx) > 0 or len(mn) > 0:  # we have some Q limit violations
            # one at a time?
            if qlim == 2 and not enforce_facts:  # fix largest violation, ignore the rest
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

            for i in limited:  # one at a time, since generators may be at the same bus
                gen[i, GEN_STATUS] = 0
                if not enforce_facts:
                    bi = int(gen[i, GEN_BUS])
                    bus[bi, [PD, QD]] -= gen[i, [PG, QG]]
        else:
            break  # no more generator Q limits violated

    for device in facts:  # restore the control flags; held devices keep their limit angles
        _set_facts_control(ppci, device, True)

    if len(limited) > 0:
        # restore injections from limited gens [those at Q limits]
        restore_gens = gen[limited, GEN_BUS].astype(int64) if enforce_facts else changed_gens
        bus[setdiff1d(restore_gens, ref), BUS_TYPE] = PV  # restore PV buses
        gen[limited, QG] = fixedQg[limited]  # restore Qg value,
        gen[limited, GEN_STATUS] = 1  # turn gens back on
        bus[:, [PD, QD]] = bus_backup_p_q
    return ppci, success, iterations, bus, gen, branch
