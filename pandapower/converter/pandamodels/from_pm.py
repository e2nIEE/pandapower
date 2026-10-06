# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import math
import numpy as np
import pandas as pd
from copy import deepcopy
from pandapower.auxiliary import _clean_up, pandapowerNet
from pandapower.pypower.idx_brch import PF, PT, QF, QT, BR_STATUS
from pandapower.pypower.idx_bus import VA, VM
from pandapower.pypower.idx_gen import PG, QG
from pandapower.results import _extract_results, _copy_results_ppci_to_ppc
from pandapower.optimal_powerflow import OPFNotConverged
from logging import getLogger

logger = getLogger('converter.pandamodels.from_pm')


def read_pm_results_to_net(net: pandapowerNet, ppc, ppci, result_pm):
    """
    reads power models results from result_pm to ppc / ppci and then to pandapower net
    """
    # read power models results from result_pm to result (== ppc with results)
    result, multinetwork = pm_results_to_ppc_results(net, ppc, ppci, result_pm)
    net["_pm_result"] = {} if multinetwork else result.copy()
    net["_pm_org_result"] = result_pm["solution"]
    if "ne_branch" in result_pm["solution"].keys():
        net["_pm_result"]["ne_branch"] = result_pm["solution"]["ne_branch"]
    net["_pm_result"]["solve_time"] = result_pm["solve_time"]

    success = ppc["success"]
    if success:
        if not multinetwork:
            # results are extracted from a single time step to pandapower dataframes
            _extract_results(net, result)
        else:
            net["res_ts_opt"] = _read_multinetwork_results(net, ppc, ppci, result_pm["solution"])
            # objective over all time steps
            net["res_cost"] = result_pm["objective"]
        _clean_up(net)
        net["OPF_converged"] = True
    else:
        _clean_up(net, res=False)
        logger.warning("OPF did not converge!")
        raise OPFNotConverged("PowerModels.jl OPF not converged")


def _read_multinetwork_results(net: pandapowerNet, ppc, ppci, sol):
    """
    Extracts the results of every time step and returns them as dict "res_<table>.<column>" ->
    DataFrame (index: time steps, columns: element indices), like the OutputWriter.
    """
    va_in_degrees = _va_in_degrees(sol)
    from_time_step = net._options.get("from_time_step") or 0
    # the net inputs are changed per time step -> work on one copy
    neti = deepcopy(net)
    res_tables: list[str] | None = None
    frames: dict[str, list[pd.DataFrame]] = {}
    time_steps: list[int] = []
    for nw in sorted(sol["nw"], key=int):
        time_step = from_time_step + int(nw) - 1
        soli = sol["nw"][nw]
        pm_results_to_ppc_results_one_time_step(ppci, soli, va_in_degrees)
        add_time_series_data_to_net(neti, net.controller, time_step)
        _extract_results(neti, _copy_results_ppci_to_ppc(ppci, ppc, net._options["mode"]))
        add_storage_results(neti, soli)
        if res_tables is None:
            res_tables = [key for key, val in neti.items() if key.startswith("res_") and
                          isinstance(val, pd.DataFrame) and len(val)]
        for table in res_tables:
            frames.setdefault(table, []).append(neti[table].copy())
        time_steps.append(time_step)

    res_ts_opt = {}
    for table, dfs in frames.items():
        stacked = pd.concat(dfs, keys=time_steps)
        for column in stacked.columns:
            res_ts_opt["%s.%s" % (table, column)] = stacked[column].unstack()
    return res_ts_opt


def add_storage_results(net: pandapowerNet, result_pmi):
    """writes the storage results of one time step to net.res_storage (pm index = pp index + 1)"""
    if "storage" not in result_pmi or not len(result_pmi["storage"]):
        return
    df = net.res_storage
    for column in ["soc_mwh", "soc_percent"]:
        if column not in df.columns:
            df[column] = np.nan
    pp_idx = [int(i) - 1 for i in result_pmi["storage"]]
    values = list(result_pmi["storage"].values())
    df.loc[pp_idx, "p_mw"] = [_float(v["ps"]) for v in values]
    df.loc[pp_idx, "q_mvar"] = [_float(v["qs"]) for v in values]
    soc_mwh = np.array([_float(v["se"]) for v in values])
    df.loc[pp_idx, "soc_mwh"] = soc_mwh
    capacity = (net.storage.max_e_mwh - net.storage.min_e_mwh).loc[pp_idx].values
    df.loc[pp_idx, "soc_percent"] = soc_mwh / capacity * 100.


def _float(value):
    # NaN is transferred as null
    return np.nan if value is None else value


def _va_in_degrees(sol):
    # PandaModels.jl >= 0.10 returns MW and degrees for data sent in MATPOWER units (per_unit=False),
    # older results are per unit on 1 MVA with angles in radians
    return "per_unit" in sol and not sol["per_unit"]


def add_time_series_data_to_net(net: pandapowerNet, controller, time_step):
    from pandapower.control import ConstControl
    for idx, content in controller.iterrows():
        if type(content["object"]) == ConstControl:
            element = content["object"].__dict__["matching_params"]["element"]
            variable = content["object"].__dict__["matching_params"]["variable"]
            elm_idxs = content["object"].__dict__["matching_params"]["element_index"]
            df = content["object"].data_source.df
            net[element].loc[elm_idxs, variable] = df.loc[time_step].values.astype(np.float64)


def pm_results_to_ppc_results(net: pandapowerNet, ppc, ppci, result_pm):
    options = net._options
    # status if result is from multiple grids
    multinetwork = False
    sol = result_pm["solution"]
    ppci["obj"] = result_pm["objective"]
    # the native power flows (pm_model "ACNative" / "DCNative") return a bool
    termination_status = result_pm["termination_status"]
    ppci["success"] = termination_status is True or "LOCALLY_SOLVED" in str(termination_status) or \
        "OPTIMAL" in str(termination_status)
    ppci["et"] = result_pm["solve_time"]
    ppci["f"] = result_pm["objective"]

    if "multinetwork" in sol and sol["multinetwork"]:
        # the time steps are read in _read_multinetwork_results
        multinetwork = True
        result = None
        ppc["obj"] = ppci["obj"]
        ppc["success"] = ppci["success"]
        ppc["et"] = ppci["et"]
        ppc["f"] = ppci["f"]
    else:
        if "bus" not in sol:
            ppci["success"] = False # PowerModels failed
        else:
            pm_results_to_ppc_results_one_time_step(ppci, sol, _va_in_degrees(sol))
            result = _copy_results_ppci_to_ppc(ppci, ppc, options["mode"])
    return result, multinetwork


def pm_results_to_ppc_results_one_time_step(ppci, sol, va_in_degrees=True):
    for i, bus in sol["bus"].items():
        bus_idx = int(i) - 1
        if "vm" in bus:
            ppci["bus"][bus_idx, VM] = _float(bus["vm"])
        if "va" in bus:
            # replace nans with 0.(in case of SOCWR model for example
            va = 0.0 if bus["va"] is None else bus["va"]
            ppci["bus"][bus_idx, VA] = va if va_in_degrees else math.degrees(va)
        if "w" in bus:
            # SOCWR model has only w = vm^2 instead of vm values
            ppci["bus"][bus_idx, VM] = math.sqrt(bus["w"])

    for i, gen in sol["gen"].items():
        gen_idx = int(i) - 1
        ppci["gen"][gen_idx, PG] = _float(gen["pg"])
        ppci["gen"][gen_idx, QG] = _float(gen["qg"])

    # read Q from branch results (if not DC calculation)
    if "branch" in sol:
        dc_results = sol["branch"]["1"]["qf"] is None or np.isnan(sol["branch"]["1"]["qf"])
        # read branch status results (OTS)
        branch_status = "br_status" in sol["branch"]["1"]
        for i, branch in sol["branch"].items():
            br_idx = int(i) - 1
            ppci["branch"][br_idx, PF] = _float(branch["pf"])
            ppci["branch"][br_idx, PT] = _float(branch["pt"])
            if not dc_results:
                ppci["branch"][br_idx, QF] = _float(branch["qf"])
                ppci["branch"][br_idx, QT] = _float(branch["qt"])
            if branch_status:
                ppci["branch"][br_idx, BR_STATUS] = branch["br_status"] > 0.5


def read_ots_results(net: pandapowerNet):
    """
    Reads the branch_status variable from ppc to pandapower net

    INPUT

        **net** - pandapower net
    """
    ppc = net._ppc
    for element, (f, t) in net._pd2ppc_lookups["branch"].items():
        # for trafo, line, trafo3w
        res = "res_" + element
        if "in_service" not in net[res]:
            # copy in service state from inputs
            net[res].loc[:, "in_service"] = None
            net[res].loc[:, "in_service"] = net[res].loc[:, "in_service"].values
        branch_status = ppc["branch"][f:t, BR_STATUS].real  # type: ignore[index]

        net[res].loc[:, "in_service"] = branch_status


def read_tnep_results(net: pandapowerNet):
    ne_branch = net._pm_result["ne_branch"]
    line_idx = net["res_ne_line"].index
    for pm_branch_idx, branch_data in ne_branch.items():
        # get pandapower index from power models index
        pp_idx = line_idx[int(pm_branch_idx) - 1]
        # built is a float, which is not exactly 1.0 or 0. sometimes
        net["res_ne_line"].loc[pp_idx, "built"] = branch_data["built"] > 0.5
