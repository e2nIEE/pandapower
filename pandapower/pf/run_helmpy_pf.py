# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from time import perf_counter

import numpy as np
import pandas as pd

from pandapower.pypower.idx_bus import PD, QD, BUS_TYPE, GS, BS, BUS_I, VM, VA, SL_FAC as SL_FAC_BUS
from pandapower.pypower.idx_brch import F_BUS, T_BUS
from pandapower.pypower.idx_gen import PG, QMAX, QMIN, GEN_BUS, VG, MBASE, GEN_STATUS, SL_FAC
from pandapower.pf.run_newton_raphson_pf import ppci_to_pfsoln, _get_numba_functions, _get_Y_bus
from pandapower.pf.ppci_variables import _get_pf_variables_from_ppci, _store_results_from_pf_in_ppci

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def _build_helm_case(ppci):
    """Build a HELMpy :code:`CaseData` object from a pandapower internal ppc (ppci).

    The bus ordering of :code:`ppci` is preserved: HELMpy bus index :code:`i` corresponds
    to ppci bus row :code:`i`, so the resulting complex voltage profile can be written
    back to :code:`ppci` without any reordering.
    """
    from helmpy.core.classes import CaseData, process_branches  # type: ignore[import-not-found, import-untyped]

    buses = ppci['bus'].copy()
    generators = ppci['gen'].copy()
    branches = ppci['branch'].copy()

    # pandapower does not always set MBASE on the internal gen matrix
    generators[:, MBASE] = 100.

    # HELMpy follows MATPOWER's 1-based bus numbering and internally subtracts 1
    # (e.g. process_branches does ``int(branch[F_BUS]) - 1``). pandapower's internal
    # ppci uses 0-based, contiguous bus indices, so shift bus references by +1 to
    # match the convention HELMpy expects.
    buses[:, BUS_I] += 1
    generators[:, GEN_BUS] += 1
    branches[:, F_BUS] += 1
    branches[:, T_BUS] += 1

    N = len(buses)
    N_generators = len(generators)
    N_branches = len(branches)
    case = CaseData(name='pandapower', N=N, N_generators=N_generators)

    case.N_branches = N_branches
    case.Pd[:] = buses[:, PD] / 100
    case.Qd[:] = buses[:, QD] / 100

    # Bus shunt admittance. pandapower stores GS/BS in MW/MVAr at 1 p.u.; convert to
    # per-unit on the 100 MVA base that HELMpy assumes (matches the xlsx case builder,
    # which divides by 100, and pandapower's makeYbus, which divides by baseMVA).
    case.Shunt[:] = (buses[:, BS].copy() * 1j + buses[:, GS]) / 100
    case.Yshunt[:] = np.copy(case.Shunt)

    for i in range(N):
        case.Number_bus[int(buses[i][BUS_I]) - 1] = i
        if buses[i][BUS_TYPE] == 3:
            case.slack_bus = int(buses[i][BUS_I])
            case.slack = i

    pos = 0
    for i in range(N_generators):
        bus_i = case.Number_bus[int(generators[i][GEN_BUS]) - 1]
        if bus_i != case.slack:
            case.list_gen[pos] = bus_i
            pos += 1
        case.Buses_type[bus_i] = 'PVLIM'
        case.V[bus_i] = generators[i][VG]
        case.Pg[bus_i] = generators[i][PG] / 100
        case.Qgmax[bus_i] = generators[i][QMAX] / 100
        case.Qgmin[bus_i] = generators[i][QMIN] / 100

    case.Buses_type[case.slack] = 'Slack'
    case.Pg[case.slack] = 0

    process_branches(pd.DataFrame(branches), N_branches, case)

    for i in range(N):
        case.branches_buses[i].sort()    # Variable that saves the branches

    case.Y[:] = np.copy(case.Ytrans)
    for i in range(N):
        if case.Yshunt[i].real != 0:
            case.conduc_buses[i] = True
        case.Y[i, i] += case.Yshunt[i]
        if case.phase_barras[i]:
            for k in range(len(case.phase_dict[i][0])):
                case.Y[i, case.phase_dict[i][0][k]] += case.phase_dict[i][1][k]

    return case


def _get_slack_weights(ppci):
    """Collect pandapower's distributed-slack participation factors as a per-bus array.

    Weights come from in-service generators (:code:`gen[:, SL_FAC]`, accumulated on their bus)
    and from buses directly (:code:`bus[:, SL_FAC_BUS]`, e.g. xwards). The bus order matches
    the ppci bus rows, which is also HELMpy's internal bus order. Returns ``None`` when no
    weights are set so the caller lets HELMpy use its default behavior.
    """
    bus = ppci["bus"]
    gen = ppci["gen"]
    N = len(bus)
    weights = np.zeros(N, dtype=np.float64)

    weights += bus[:, SL_FAC_BUS]

    on = gen[:, GEN_STATUS] > 0
    gen_buses = gen[on, GEN_BUS].astype(np.int64)
    np.add.at(weights, gen_buses, gen[on, SL_FAC])

    if not np.any(weights):
        return None
    return weights


def _runpf_helmpy_pf(ppci, options, **kwargs):
    """Runs a HELM (Holomorphic Embedding Load flow Method) based power flow, provided by the optional HELMpy package.

    The converged complex voltage profile is routed through pandapower's regular
    ``pfsoln`` result extraction (the same one used by Newton-Raphson) so that
    generator reactive power, slack injections and branch flows are populated
    identically to the other algorithms.

    The HELM-specific option ``pv_bus_model`` (1 or 2, default 2) selects how PV
    (voltage-controlled) buses are embedded into the holomorphic equations:

    Parameters:
        ppci (dict): the "internal" ppc (without out of service elements and sorted elements)
        options (dict): options for the power flow

            pv_bus_model - 1 or 2

            - model 1:
              the real part of each PV-bus voltage coefficient is precomputed
              analytically from the |V| = const constraint, leaving only the imaginary part
              as a matrix unknown.
            - model 2:
              both real and imaginary parts stay unknowns and the |V| = const
              constraint is added as an explicit equation row.

            Both formulations converge to the same load-flow solution (verified to machine
            precision); they differ only in internal bookkeeping.

    Returns:
        ppci (dict)
    """
    try:
        from helmpy import helm  # type: ignore[import-not-found, import-untyped]
    except ImportError:
        raise ImportError("The HELM algorithm requires the optional 'helmpy' package. "
                          "Install it (pip install helmpy) to use algorithm='helm'.")

    t0 = perf_counter()

    max_coefficients = options['max_iteration']
    enforce_Q_limits = bool(options["enforce_q_lims"])
    DSB_model = bool(options['distributed_slack'])
    # HELMpy PV-bus embedding model (1 or 2). Both give the same solution; see module docstring.
    pv_bus_model = options.get('pv_bus_model', 2)

    # ---------------------------------------------------- run HELM ----------------------------------------------------
    case = _build_helm_case(ppci)

    # For distributed slack, pass pandapower's user-defined slack_weight factors so HELM
    # distributes the imbalance the same way as Newton-Raphson. If no weights are set,
    # K_factors stays None and HELM falls back to its generation-proportional default.
    K_factors = _get_slack_weights(ppci) if DSB_model else None

    run, series_large, flag_divergence = helm(
        case, detailed_run_print=False, mismatch=1e-8, scale=1,
        max_coefficients=max_coefficients, enforce_Q_limits=enforce_Q_limits,
        results_file_name=None, save_results=False, pv_bus_model=pv_bus_model,
        DSB_model=DSB_model, DSB_model_method=None, K_factors=K_factors,
    )

    success = not flag_divergence
    V = run.V_complex_profile.copy()

    # HELMpy fixes the slack bus angle at 0 degrees. pandapower references all angles
    # to the slack's va_degree setpoint (carried in ppci["bus"][slack, VA]), so rotate
    # the whole profile by that reference angle to align with the other algorithms.
    slack_va_rad = np.deg2rad(ppci["bus"][case.slack, VA])
    if slack_va_rad != 0:
        V *= np.exp(1j * slack_va_rad)

    # ------------------------------------------- result extraction via pfsoln -----------------------------------------
    # Store the converged voltage and admittance matrices in ppci["internal"] so that the
    # standard pfsoln-based extraction (shared with Newton-Raphson) computes all results.
    baseMVA, bus, gen, branch, svc, tcsc, ssc, vsc, ref, pv, pq, *_, ref_gens = \
        _get_pf_variables_from_ppci(ppci)

    makeYbus, _ = _get_numba_functions(ppci, options)
    ppci, Ybus, Yf, Yt = _get_Y_bus(ppci, options, makeYbus, baseMVA, bus, branch)

    # write voltage to the bus matrix as well (needed by only_v_results / time series hack)
    bus[:, VM] = np.abs(V)
    bus[:, VA] = np.angle(V, deg=True)

    internal = ppci["internal"]
    internal.update({"bus": bus, "gen": gen, "branch": branch, "svc": svc, "tcsc": tcsc,
                     "ssc": ssc, "vsc": vsc, "baseMVA": baseMVA, "V": V, "pv": pv, "pq": pq,
                     "ref": ref, "ref_gens": ref_gens, "Ybus": Ybus, "Yf": Yf, "Yt": Yt})

    if success:
        bus, gen, branch = ppci_to_pfsoln(ppci, options)

    et = perf_counter() - t0
    ppci = _store_results_from_pf_in_ppci(ppci, bus, gen, branch, success, series_large, et)

    return ppci
