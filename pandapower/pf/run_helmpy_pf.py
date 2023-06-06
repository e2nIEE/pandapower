from time import perf_counter
import numpy as np
from pandapower.pypower.idx_bus import PD, QD, BUS_TYPE, PQ, REF, GS, BS, BUS_I, VM, VA
from pandapower.pypower.idx_gen import PG, QG, QMAX, QMIN, GEN_BUS, GEN_STATUS, VG, MBASE
from pandapower.pypower.bustypes import bustypes
from pandapower.pypower.makeSbus import makeSbus
from pandapower.pf.ppci_variables import _get_pf_variables_from_ppci, _store_results_from_pf_in_ppci
import pandas as pd

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def convert_complex_to_polar_voltages(complex_voltage):
    """Separate each voltage value in magnitude and phase angle (degrees)"""
    polar_voltage = np.zeros((len(complex_voltage), 2), dtype=float)
    polar_voltage[:, 0] = np.absolute(complex_voltage)
    polar_voltage[:, 1] = np.angle(complex_voltage, deg=True)
    return polar_voltage


def _runpf_helmpy_pf(ppci, options: dict, **kwargs):
    """
    Runs a HELM based power flow.

    INPUT
    ppci (dict) - the "internal" ppc (without out ot service elements and sorted elements)
    options(dict) - options for the power flow

    """

    from helmpy import helm
    from helmpy.core.classes import CaseData, process_branches

    t0 = perf_counter()
    # we cannot run DC pf before running newton with distributed slack because the slacks come pre-solved after the DC pf
    # if isinstance(options["init_va_degree"], str) and options["init_va_degree"] == "dc":
    #     if options['distributed_slack']:
    #         pg_copy = ppci['gen'][:, PG].copy()
    #         pd_copy = ppci['bus'][:, PD].copy()
    #         ppci = _run_dc_pf(ppci, options["recycle"])
    #         ppci['gen'][:, PG] = pg_copy
    #         ppci['bus'][:, PD] = pd_copy
    #     else:
    #         ppci = _run_dc_pf(ppci, options["recycle"])

    max_coefficients = options['max_iteration']

    enforce_Q_limits = False
    if options["enforce_q_lims"]:
        enforce_Q_limits = True
    # else:
    #     ppci, success, iterations = _run_ac_pf_without_qlims_enforced(ppci, options)
    #     # update data matrices with solution store in ppci
    #     bus, gen, branch = ppci_to_pfsoln(ppci, options)

    DSB_model = False
    if options['distributed_slack']:
        DSB_model = True

# -------------------------------------------- Calculation Case preparation --------------------------------------------
    buses = ppci['bus'].copy()
    generators = ppci['gen'].copy()
    branches = ppci['branch'].copy()

    generators[:, MBASE] = 100.

    N = len(buses)
    N_generators = len(generators)
    N_branches = len(branches)
    case = CaseData(name='pandapower', N=N, N_generators=N_generators)

    case.N_branches = N_branches
    case.Pd[:] = buses[:, PD] / 100
    case.Qd[:] = buses[:, QD] / 100

    # weird but has to be done in this order. Otherwise, the values are wrong...
    case.Shunt[:] = (buses[:, BS].copy()*1j + buses[:, GS])
    case.Yshunt[:] = np.copy(case.Shunt)
    case.Shunt[:] = case.Shunt[:] / 100

    for i in range(N):
        case.Number_bus[buses[i][BUS_I]] = i
        if buses[i][BUS_TYPE] == 3:
            case.slack_bus = buses[i][BUS_I]
            case.slack = i

    pos = 0
    for i in range(N_generators):
        bus_i = case.Number_bus[generators[i][GEN_BUS]]
        if bus_i != case.slack:
            case.list_gen[pos] = bus_i
            pos += 1
        case.Buses_type[bus_i] = 'PVLIM'
        case.V[bus_i] = generators[i][VG]
        case.Pg[bus_i] = generators[i][PG]/100
        case.Qgmax[bus_i] = generators[i][QMAX]/100
        case.Qgmin[bus_i] = generators[i][QMIN]/100

    case.Buses_type[case.slack] = 'Slack'
    case.Pg[case.slack] = 0

    branches_df = pd.DataFrame(branches)

    process_branches(branches_df, N_branches, case)

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

    run, series_large, flag_divergence = helm(case, detailed_run_print=False, mismatch=1e-8, scale=1,
                                              max_coefficients=max_coefficients, enforce_Q_limits=enforce_Q_limits,
                                              results_file_name=None, save_results=False, pv_bus_model=1,
                                              DSB_model=DSB_model, DSB_model_method=None,)

    et = perf_counter() - t0
    complex_voltage = run.V_complex_profile.copy()
    polar_voltage = convert_complex_to_polar_voltages(complex_voltage)  # Calculate polar voltage

    buses[:, VM] = polar_voltage[:, 0]
    buses[:, VA] = polar_voltage[:, 1]

    ppci["bus"], ppci["gen"], ppci["branch"] = buses, generators, branches
    ppci["success"] = not flag_divergence
    ppci["internal"]["V"] = complex_voltage
    ppci["iterations"] = series_large
    ppci["et"] = et

    return ppci
