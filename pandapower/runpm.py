# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import json
import math
import os
import tempfile

import numpy as np
import pandas as pd

from pandapower import pp_dir
from pandapower.auxiliary import _add_ppc_options, _add_opf_options
from pandapower.auxiliary import _clean_up, _add_auxiliary_elements
from pandapower.build_branch import _calc_line_parameter
from pandapower.pd2ppc import _pd2ppc
from pandapower.pypower.idx_brch import branch_cols
from pandapower.results import _extract_results, reset_results, _copy_results_ppci_to_ppc

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

from pandapower.pypower.idx_gen import PG, QG, GEN_BUS, VG, QMAX, GEN_STATUS, QMIN, PMIN, PMAX
from pandapower.pypower.idx_bus import BUS_I, ZONE, BUS_TYPE, VMAX, VMIN, VA, VM, BASE_KV, PD, QD, GS, BS
from pandapower.pypower.idx_brch import BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, F_BUS, T_BUS, BR_STATUS, \
    ANGMIN, ANGMAX, TAP, SHIFT, PF, PT, QF, QT
from pandapower.pypower.idx_cost import MODEL, COST, NCOST

# this is only used by pm tnep
CONSTRUCTION_COST = 23


def runpm(net, julia_file, pp_to_pm_callback=None, calculate_voltage_angles=True,
          trafo_model="t", delta=0, trafo3w_losses="hv", check_connectivity=True):  # pragma: no cover
    """
    Runs a power system optimization using PowerModels.jl. with a custom julia file.
    
    Flexibilities, constraints and cost parameters are defined in the pandapower element tables.

    Flexibilities can be defined in net.sgen / net.gen /net.load
    net.sgen.controllable if a static generator is controllable. If False,
    the active and reactive power are assigned as in a normal power flow. If True, the following
    flexibilities apply:
        - net.sgen.min_p_mw / net.sgen.max_p_mw
        - net.sgen.min_q_mvar / net.sgen.max_q_mvar
        - net.load.min_p_mw / net.load.max_p_mw
        - net.load.min_q_mvar / net.load.max_q_mvar
        - net.gen.min_p_mw / net.gen.max_p_mw
        - net.gen.min_q_mvar / net.gen.max_q_mvar
        - net.ext_grid.min_p_mw / net.ext_grid.max_p_mw
        - net.ext_grid.min_q_mvar / net.ext_grid.max_q_mvar
        - net.dcline.min_q_to_mvar / net.dcline.max_q_to_mvar / net.dcline.min_q_from_mvar / net.dcline.max_q_from_mvar

    Controllable loads behave just like controllable static generators. It must be stated if they are controllable.
    Otherwise, they are not respected as flexibilities.
    Dc lines are controllable per default

    Network constraints can be defined for buses, lines and transformers the elements in the following columns:
        - net.bus.min_vm_pu / net.bus.max_vm_pu
        - net.line.max_loading_percent
        - net.trafo.max_loading_percent
        - net.trafo3w.max_loading_percent

    How these costs are combined into a cost function depends on the cost_function parameter.

    INPUT:
        **net** - The pandapower format network

    OPTIONAL:
        **julia_file** (str, None) - path to a custom julia optimization file

        **pp_to_pm_callback** (function, None) - callback function to add data to the PowerModels data structure

     """
    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode="opf", switch_rx_ratio=2, init_vm_pu="flat", init_va_degree="flat",
                     enforce_q_lims=True, recycle=dict(_is_elements=False, ppc=False, Ybus=False),
                     voltage_depend_loads=False, delta=delta, trafo3w_losses=trafo3w_losses)
    _add_opf_options(net, trafo_loading='power', ac=True, init="flat", numba=True,
                     pp_to_pm_callback=pp_to_pm_callback, julia_file=julia_file)
    _runpm(net)


def runpm_dc_opf(net, pp_to_pm_callback=None, calculate_voltage_angles=True,
                 trafo_model="t", delta=0, trafo3w_losses="hv", check_connectivity=True):  # pragma: no cover
    """
    Runs a linearized power system optimization using PowerModels.jl.
    
    Flexibilities, constraints and cost parameters are defined in the pandapower element tables.

    Flexibilities can be defined in net.sgen / net.gen /net.load
    net.sgen.controllable if a static generator is controllable. If False,
    the active and reactive power are assigned as in a normal power flow. If True, the following
    flexibilities apply:
        - net.sgen.min_p_mw / net.sgen.max_p_mw
        - net.sgen.min_q_mvar / net.sgen.max_q_mvar
        - net.load.min_p_mw / net.load.max_p_mw
        - net.load.min_q_mvar / net.load.max_q_mvar
        - net.gen.min_p_mw / net.gen.max_p_mw
        - net.gen.min_q_mvar / net.gen.max_q_mvar
        - net.ext_grid.min_p_mw / net.ext_grid.max_p_mw
        - net.ext_grid.min_q_mvar / net.ext_grid.max_q_mvar
        - net.dcline.min_q_to_mvar / net.dcline.max_q_to_mvar / net.dcline.min_q_from_mvar / net.dcline.max_q_from_mvar

    Controllable loads behave just like controllable static generators. It must be stated if they are controllable.
    Otherwise, they are not respected as flexibilities.
    Dc lines are controllable per default

    Network constraints can be defined for buses, lines and transformers the elements in the following columns:
        - net.bus.min_vm_pu / net.bus.max_vm_pu
        - net.line.max_loading_percent
        - net.trafo.max_loading_percent
        - net.trafo3w.max_loading_percent

    How these costs are combined into a cost function depends on the cost_function parameter.

    INPUT:
        **net** - The pandapower format network

    OPTIONAL:
        **pp_to_pm_callback** (function, None) - callback function to add data to the PowerModels data structure

     """
    julia_file = os.path.join(pp_dir, "opf", 'run_powermodels_dc.jl')

    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode="opf", switch_rx_ratio=2, init_vm_pu="flat", init_va_degree="flat",
                     enforce_q_lims=True, recycle=dict(_is_elements=False, ppc=False, Ybus=False),
                     voltage_depend_loads=False, delta=delta, trafo3w_losses=trafo3w_losses)
    _add_opf_options(net, trafo_loading='power', ac=True, init="flat", numba=True,
                     pp_to_pm_callback=pp_to_pm_callback, julia_file=julia_file)
    _runpm(net)


def runpm_ac_opf(net, pp_to_pm_callback=None, calculate_voltage_angles=True,
                 trafo_model="t", delta=0, trafo3w_losses="hv", check_connectivity=True):  # pragma: no cover
    """
    Runs a non-linear power system optimization using PowerModels.jl.

    Flexibilities, constraints and cost parameters are defined in the pandapower element tables.

    Flexibilities can be defined in net.sgen / net.gen /net.load
    net.sgen.controllable if a static generator is controllable. If False,
    the active and reactive power are assigned as in a normal power flow. If True, the following
    flexibilities apply:
        - net.sgen.min_p_mw / net.sgen.max_p_mw
        - net.sgen.min_q_mvar / net.sgen.max_q_mvar
        - net.load.min_p_mw / net.load.max_p_mw
        - net.load.min_q_mvar / net.load.max_q_mvar
        - net.gen.min_p_mw / net.gen.max_p_mw
        - net.gen.min_q_mvar / net.gen.max_q_mvar
        - net.ext_grid.min_p_mw / net.ext_grid.max_p_mw
        - net.ext_grid.min_q_mvar / net.ext_grid.max_q_mvar
        - net.dcline.min_q_to_mvar / net.dcline.max_q_to_mvar / net.dcline.min_q_from_mvar / net.dcline.max_q_from_mvar

    Controllable loads behave just like controllable static generators. It must be stated if they are controllable.
    Otherwise, they are not respected as flexibilities.
    Dc lines are controllable per default

    Network constraints can be defined for buses, lines and transformers the elements in the following columns:
        - net.bus.min_vm_pu / net.bus.max_vm_pu
        - net.line.max_loading_percent
        - net.trafo.max_loading_percent
        - net.trafo3w.max_loading_percent

    How these costs are combined into a cost function depends on the cost_function parameter.

    INPUT:
        **net** - The pandapower format network

    OPTIONAL:

        **pp_to_pm_callback** (function, None) - callback function to add data to the PowerModels data structure

     """
    julia_file = os.path.join(pp_dir, "opf", 'run_powermodels_ac.jl')

    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode="opf", switch_rx_ratio=2, init_vm_pu="flat", init_va_degree="flat",
                     enforce_q_lims=True, recycle=dict(_is_elements=False, ppc=False, Ybus=False),
                     voltage_depend_loads=False, delta=delta, trafo3w_losses=trafo3w_losses)
    _add_opf_options(net, trafo_loading='power', ac=True, init="flat", numba=True,
                     pp_to_pm_callback=pp_to_pm_callback, julia_file=julia_file)
    _runpm(net)


def runpm_tnep(net, pp_to_pm_callback=None, calculate_voltage_angles=True,
               trafo_model="t", delta=0, trafo3w_losses="hv", check_connectivity=True):  # pragma: no cover
    """
    Runs a non-linear transmission network extension planning (tnep) optimization using PowerModels.jl.

    see above

     """
    julia_file = os.path.join(pp_dir, "opf", 'run_powermodels_tnep.jl')

    if "ne_line" not in net:
        raise ValueError("ne_line DataFrame missing in net. Please define to run tnep")
    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode="opf", switch_rx_ratio=2, init_vm_pu="flat", init_va_degree="flat",
                     enforce_q_lims=True, recycle=dict(_is_elements=False, ppc=False, Ybus=False),
                     voltage_depend_loads=False, delta=delta, trafo3w_losses=trafo3w_losses)
    _add_opf_options(net, trafo_loading='power', ac=True, init="flat", numba=True,
                     pp_to_pm_callback=pp_to_pm_callback, julia_file=julia_file)
    _runpm(net)
    _read_tnep_results(net)


def runpm_ots(net, pp_to_pm_callback=None, calculate_voltage_angles=True,
              trafo_model="t", delta=0, trafo3w_losses="hv", check_connectivity=True):  # pragma: no cover
    """
    Runs a non-linear optimal transmission switching (OTS) optimization using PowerModels.jl.

    see above

     """
    julia_file = os.path.join(pp_dir, "opf", 'run_powermodels_ots.jl')

    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode="opf", switch_rx_ratio=2, init_vm_pu="flat", init_va_degree="flat",
                     enforce_q_lims=True, recycle=dict(_is_elements=False, ppc=False, Ybus=False),
                     voltage_depend_loads=False, delta=delta, trafo3w_losses=trafo3w_losses)
    _add_opf_options(net, trafo_loading='power', ac=True, init="flat", numba=True,
                     pp_to_pm_callback=pp_to_pm_callback, julia_file=julia_file)
    _runpm(net)
    _read_ots_results(net)


def _read_tnep_results(net):
    ne_branch = net._pm_result["solution"]["ne_branch"]
    line_idx = net["res_ne_line"].index
    for pm_branch_idx, branch_data in ne_branch.items():
        # get pandapower index from power models index
        pp_idx = line_idx[int(pm_branch_idx) - 1]
        # built is a float, which is not exactly 1.0 or 0. sometimes
        net["res_ne_line"].loc[pp_idx, "built"] = branch_data["built"] > 0.5


def _read_ots_results(net):
    # branch_lookup = net._pd2ppc_lookups["branch"]
    # ots_branch = net._pm_result["solution"]["branch"]
    # line_idx = net["res_line"].index
    # if "in_service" not in net["res_line"]:
    #     # copy in service state from inputs
    #     net["res_line"].loc[:, "in_service"] = net["line"].loc[:, "in_service"].values
    # if "in_service" not in net["res_trafo"]:
    #     # copy in service state from inputs
    #     net["res_trafo"].loc[:, "in_service"] = net["trafo"].loc[:, "in_service"].values
    #
    #
    # for pm_branch_idx, branch_data in ots_branch.items():
    #     # get pandapower index from power models index
    #     pp_idx = line_idx[int(pm_branch_idx) - 1]
    #     # the branch status from powermodels == in service status in pandapower
    #     net["res_line"].loc[pp_idx, "in_service"] = branch_data["br_status"] > 0.5

    ppc = net._ppc
    for element, (f, t) in net._pd2ppc_lookups["branch"].items():
        # for trafo, line, trafo3w
        res = "res_" + element
        if "in_service" not in net[res]:
            # copy in service state from inputs
            net[res].loc[:, "in_service"] = None
            net[res].loc[:, "in_service"] = net[res].loc[:, "in_service"].values
        # f, t = net._pd2ppc_lookups["branch"][element]
        branch_status = ppc["branch"][f:t, BR_STATUS].real

        net[res]["in_service"].values[:] = branch_status




def runpm_storage_opf(net, calculate_voltage_angles=True,
                      trafo_model="t", delta=0, trafo3w_losses="hv", check_connectivity=True,
                      n_timesteps=24, time_elapsed=1.0):  # pragma: no cover
    """
    Runs a non-linear power system optimization with storages and time series using PowerModels.jl.


    INPUT:
        **net** - The pandapower format network

    OPTIONAL:
        **n_timesteps** (int, 24) - number of time steps to optimize

        **time_elapsed** (float, 1.0) - time elapsed between time steps (1.0 = 1 hour)

     """
    julia_file = os.path.join(pp_dir, "opf", 'run_powermodels_mn_storage.jl')

    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode="opf", switch_rx_ratio=2, init_vm_pu="flat", init_va_degree="flat",
                     enforce_q_lims=True, recycle=dict(_is_elements=False, ppc=False, Ybus=False),
                     voltage_depend_loads=False, delta=delta, trafo3w_losses=trafo3w_losses)
    _add_opf_options(net, trafo_loading='power', ac=True, init="flat", numba=True,
                     pp_to_pm_callback=_add_storage_opf_settings, julia_file=julia_file)

    net._options["n_time_steps"] = n_timesteps
    net._options["time_elapsed"] = time_elapsed

    _runpm(net)
    storage_results = _read_pm_storage_results(net)
    return storage_results


def _add_storage_opf_settings(net, ppci, pm):
    # callback function to add storage settings. Must be called after initializing pm data structure since the
    # pm["storage"] dict is filled

    # n time steps to optimize (here 3 hours)
    pm["n_time_steps"] = net._options["n_time_steps"]
    # time step (here 1 hour)
    pm["time_elapsed"] = net._options["time_elapsed"]

    # add storage systems to pm
    # Todo: Some variables are not used and not included in pandapower as well (energy_rating, thermal_rating,
    # (efficiencies, r, x...)
    bus_lookup = net._pd2ppc_lookups["bus"]

    for idx in net["storage"].index:
        energy = (net["storage"].at[idx, "soc_percent"] * 1e-2 *
                  (net["storage"].at[idx, "max_e_mwh"] -
                   net["storage"].at[idx, "min_e_mwh"])) / pm["baseMVA"]
        qs = net["storage"].at[idx, "q_mvar"].item() / pm["baseMVA"]
        ps = net["storage"].at[idx, "p_mw"].item() / pm["baseMVA"]
        pm_idx = int(idx) + 1
        pm["storage"][str(pm_idx)] = {
            "energy_rating": 1.,
            "standby_loss": 0.,
            "x": 0.,
            "energy": energy,
            "r": 0.0,
            "qs": qs,
            "thermal_rating": 1.0,
            "status": int(net["storage"].at[idx, "in_service"]),
            "discharge_rating": ps,
            "storage_bus": bus_lookup[net["storage"].at[idx, "bus"]].item(),
            "charge_efficiency": 1.,
            "index": pm_idx,
            "ps": ps,
            "qmax": qs,
            "qmin": -qs,
            "charge_rating": ps,
            "discharge_efficiency": 1.0
        }


def _read_pm_storage_results(net):
    # reads the storage results from multiple time steps from the PowerModels optimization
    pm_result = net._pm_result
    # power model networks (each network represents the result of one time step)
    networks = pm_result["solution"]["nw"]
    storage_results = dict()
    n_timesteps = len(networks)
    timesteps = np.arange(n_timesteps)
    for idx in net["storage"].index:
        # read storage results for each storage from power models to a dataframe with rows = timesteps
        pm_idx = str(int(idx) + 1)
        res_storage = pd.DataFrame(data=None,
                                   index=timesteps,
                                   columns=["p_mw", "q_mvar", "soc_mwh", "soc_percent"],
                                   dtype=float)
        for t in range(n_timesteps):
            pm_storage = networks[str(t + 1)]["storage"][pm_idx]
            res_storage.at[t, "p_mw"] = pm_storage["ps"] * pm_result["solution"]["baseMVA"]
            res_storage.at[t, "q_mvar"] = pm_storage["qs"] * pm_result["solution"]["baseMVA"]
            res_storage.at[t, "soc_percent"] = pm_storage["se"] * 1e2
            res_storage.at[t, "soc_mwh"] = pm_storage["se"] * \
                                           pm_result["solution"]["baseMVA"] * \
                                           (net["storage"].at[idx, "max_e_mwh"] - net["storage"].at[idx, "min_e_mwh"])

        storage_results[idx] = res_storage

    # DEBUG print for storage result
    # for key, val in net._pm_result.items():
    #     if key == "solution":
    #         for subkey, subval in val.items():
    #             if subkey == "nw":
    #                 for i, nw in subval.items():
    #                     print("Network {}\n".format(i))
    #                     print(nw["storage"])
    #                     print("\n")

    return storage_results


def build_ne_branch(net, ppc):
    if "ne_line" in net:
        length = len(net["ne_line"])
        ppc["ne_branch"] = np.zeros(shape=(length, branch_cols + 1), dtype=np.complex128)
        ppc["ne_branch"][:, :13] = np.array([0, 0, 0, 0, 0, 250, 250, 250, 1, 0, 1, -360, 360])
        # create branch array ne_branch like the common branch array in the ppc
        net._pd2ppc_lookups["ne_branch"] = dict()
        net._pd2ppc_lookups["ne_branch"]["ne_line"] = (0, length)
        _calc_line_parameter(net, ppc, "ne_line", "ne_branch")
        ppc["ne_branch"][:, CONSTRUCTION_COST] = net["ne_line"].loc[:, "construction_cost"].values
    return ppc


def _runpm(net):  # pragma: no cover
    net["OPF_converged"] = False
    net["converged"] = False
    _add_auxiliary_elements(net)
    reset_results(net)
    ppc, ppci = _pd2ppc(net)
    ppci = build_ne_branch(net, ppci)
    net["_ppc_opf"] = ppci
    pm = ppc_to_pm(net, ppci)
    net._pm = pm
    if net._options["pp_to_pm_callback"] is not None:
        net._options["pp_to_pm_callback"](net, ppci, pm)
    result_pm = _call_powermodels(pm, net._options["julia_file"])
    result, multinetwork = pm_results_to_ppc_results(net, ppc, ppci, result_pm)
    net._pm_result = result_pm
    success = ppc["success"]
    if success:
        if not multinetwork:
            # results are extracted from a single time step to pandapower dataframes
            _extract_results(net, result)
        _clean_up(net)
        net["OPF_converged"] = True
    else:
        _clean_up(net, res=False)
        logger.warning("OPF did not converge!")


def _call_powermodels(pm, julia_file):  # pragma: no cover
    buffer_file = os.path.join(tempfile.gettempdir(), "pp_pm.json")
    logger.debug("writing PowerModels data structure to %s" % buffer_file)
    with open(buffer_file, 'w') as outfile:
        json.dump(pm, outfile)
    try:
        import julia
        from julia import Main
    except ImportError:
        raise ImportError("Please install pyjulia to run pandapower with PowerModels.jl")
    try:
        j = julia.Julia()
    except:
        raise UserWarning(
            "Could not connect to julia, please check that Julia is installed and pyjulia is correctly configured")

    Main.include(os.path.join(pp_dir, "opf", 'pp_2_pm.jl'))
    try:
        run_powermodels = Main.include(julia_file)
    except ImportError:
        raise UserWarning("File %s could not be imported" % julia_file)
    result_pm = run_powermodels(buffer_file)
    return result_pm


def _pp_element_to_pm(net, pm, element, pd_bus, qd_bus, load_idx):
    bus_lookup = net._pd2ppc_lookups["bus"]

    for idx in net[element].index:
        if "controllable" in net[element] and net[element].at[idx, "controllable"]:
            continue

        pp_bus = net[element].at[idx, "bus"]
        pm_bus = bus_lookup[pp_bus] + 1

        scaling = net[element].at[idx, "scaling"]
        if element == "sgen":
            pd = -net[element].at[idx, "p_mw"] * scaling
            qd = -net[element].at[idx, "q_mvar"] * scaling
        else:
            pd = net[element].at[idx, "p_mw"] * scaling
            qd = net[element].at[idx, "q_mvar"] * scaling
        in_service = net[element].at[idx, "in_service"]

        pm["load"][str(load_idx)] = {"pd": pd.item(), "qd": qd.item(), "load_bus": pm_bus.item(),
                                     "status": int(in_service), "index": load_idx}
        # print("Added load {} to pm".format(load_idx))
        # print(pm["load"][str(load_idx)])
        # print("PP element:")
        # print(net[element].loc[idx, ["bus", "p_mw"]])
        # print("\n")
        if pm_bus not in pd_bus:
            pd_bus[pm_bus] = pd
            qd_bus[pm_bus] = qd
        else:
            pd_bus[pm_bus] += pd
            qd_bus[pm_bus] += qd

        load_idx += 1
    return load_idx


def ppc_to_pm(net, ppc):  # pragma: no cover
    # create power models dict. Similar to matpower case file. ne_branch is for a tnep case
    pm = {"gen": dict(), "branch": dict(), "bus": dict(), "dcline": dict(), "load": dict(), "storage": dict(),
          "ne_branch": dict(),
          "baseMVA": ppc["baseMVA"], "source_version": "2.0.0", "shunt": dict(),
          "sourcetype": "matpower", "per_unit": True, "name": net.name}
    load_idx = 1
    shunt_idx = 1
    # PowerModels has a load model -> add loads and sgens to pm["load"]

    # temp dicts which hold the sum of p, q of loads + sgens
    pd_bus = dict()
    qd_bus = dict()
    load_idx = _pp_element_to_pm(net, pm, "load", pd_bus, qd_bus, load_idx)
    load_idx = _pp_element_to_pm(net, pm, "sgen", pd_bus, qd_bus, load_idx)

    for row in ppc["bus"]:
        bus = dict()
        idx = int(row[BUS_I]) + 1
        bus["index"] = idx
        bus["bus_i"] = idx
        bus["zone"] = int(row[ZONE])
        bus["bus_type"] = int(row[BUS_TYPE])
        bus["vmax"] = row[VMAX]
        bus["vmin"] = row[VMIN]
        bus["va"] = row[VA]
        bus["vm"] = row[VM]
        bus["base_kv"] = row[BASE_KV]

        pd = row[PD]
        qd = row[QD]

        # pd and qd are the PQ values in the ppci, if they are equal to the sum in load data is consistent
        if idx in pd_bus:
            pd -= pd_bus[idx]
            qd -= qd_bus[idx]
        # if not we have to add more loads wit the remaining value
        pq_mismatch = not np.allclose(pd, 0.) or not np.allclose(qd, 0.)
        if pq_mismatch:
            # Todo: This will be called if ppc PQ != sum at bus. -> Storages are within sum (which must be excluded I think in storage_optimization)
            print("PQ mismatch. Adding another load at idx {}".format(load_idx))
            pm["load"][str(load_idx)] = {"pd": pd, "qd": qd, "load_bus": idx,
                                         "status": True, "index": load_idx}
            load_idx += 1
        bs = row[BS]
        gs = row[GS]
        if pq_mismatch:
            pm["shunt"][str(shunt_idx)] = {"gs": gs, "bs": bs, "shunt_bus": idx,
                                           "status": True, "index": shunt_idx}
            shunt_idx += 1
        pm["bus"][str(idx)] = bus

    n_lines = net._pd2ppc_lookups["branch"]["line"][1]
    for idx, row in enumerate(ppc["branch"], start=1):
        branch = dict()
        branch["index"] = idx
        branch["transformer"] = idx > n_lines
        branch["br_r"] = row[BR_R].real
        branch["br_x"] = row[BR_X].real
        branch["g_fr"] = - row[BR_B].imag / 2.0
        branch["g_to"] = - row[BR_B].imag / 2.0
        branch["b_fr"] = row[BR_B].real / 2.0
        branch["b_to"] = row[BR_B].real / 2.0
        branch["rate_a"] = row[RATE_A].real if row[RATE_A] > 0 else row[RATE_B].real
        branch["rate_b"] = row[RATE_B].real
        branch["rate_c"] = row[RATE_C].real
        branch["f_bus"] = int(row[F_BUS].real) + 1
        branch["t_bus"] = int(row[T_BUS].real) + 1
        branch["br_status"] = int(row[BR_STATUS].real)
        branch["angmin"] = row[ANGMIN].real
        branch["angmax"] = row[ANGMAX].real
        branch["tap"] = row[TAP].real
        branch["shift"] = math.radians(row[SHIFT].real)
        pm["branch"][str(idx)] = branch

    for idx, row in enumerate(ppc["gen"], start=1):
        gen = dict()
        gen["pg"] = row[PG]
        gen["qg"] = row[QG]
        gen["gen_bus"] = int(row[GEN_BUS]) + 1
        gen["vg"] = row[VG]
        gen["qmax"] = row[QMAX]
        gen["gen_status"] = int(row[GEN_STATUS])
        gen["qmin"] = row[QMIN]
        gen["pmin"] = row[PMIN]
        gen["pmax"] = row[PMAX]
        gen["index"] = idx
        pm["gen"][str(idx)] = gen

    if "ne_branch" in ppc:
        for idx, row in enumerate(ppc["ne_branch"], start=1):
            branch = dict()
            branch["index"] = idx
            branch["transformer"] = False
            branch["br_r"] = row[BR_R].real
            branch["br_x"] = row[BR_X].real
            branch["g_fr"] = - row[BR_B].imag / 2.0
            branch["g_to"] = - row[BR_B].imag / 2.0
            branch["b_fr"] = row[BR_B].real / 2.0
            branch["b_to"] = row[BR_B].real / 2.0
            branch["rate_a"] = row[RATE_A].real if row[RATE_A] > 0 else row[RATE_B].real
            branch["rate_b"] = row[RATE_B].real
            branch["rate_c"] = row[RATE_C].real
            branch["f_bus"] = int(row[F_BUS].real) + 1
            branch["t_bus"] = int(row[T_BUS].real) + 1
            branch["br_status"] = int(row[BR_STATUS].real)
            branch["angmin"] = row[ANGMIN].real
            branch["angmax"] = row[ANGMAX].real
            branch["tap"] = row[TAP].real
            branch["shift"] = math.radians(row[SHIFT].real)
            branch["construction_cost"] = row[CONSTRUCTION_COST].real
            pm["ne_branch"][str(idx)] = branch

    if len(ppc["gencost"]) > len(ppc["gen"]):
        logger.warning("PowerModels.jl does not reactive power cost - costs are ignored")
        ppc["gencost"] = ppc["gencost"][:ppc["gen"].shape[0], :]
    for idx, row in enumerate(ppc["gencost"], start=1):
        gen = pm["gen"][str(idx)]
        gen["model"] = int(row[MODEL])
        if gen["model"] == 1:
            gen["ncost"] = int(row[NCOST])
            gen["cost"] = row[COST:COST + gen["ncost"] * 2].tolist()
        elif gen["model"] == 2:
            gen["ncost"] = 2
            gen["cost"] = [0] * 3
            costs = row[COST:]
            if len(costs) > 3:
                logger.info(costs)
                raise ValueError("Maximum quadratic cost function allowed")
            gen["cost"][-len(costs):] = costs

    # for key, val in pm["load"].items():
    #     print("load {}".format(key))
    #     print(val)
    #     print("\n")
    return pm


def pm_results_to_ppc_results(net, ppc, ppci, result_pm):  # pragma: no cover
    options = net._options
    # status if result is from multiple grids
    multinetwork = False
    sol = result_pm["solution"]
    ppci["obj"] = result_pm["objective"]
    ppci["success"] = "LOCALLY_SOLVED" in str(result_pm["termination_status"])
    ppci["et"] = result_pm["solve_time"]
    ppci["f"] = result_pm["objective"]

    if "multinetwork" in sol and sol["multinetwork"]:
        multinetwork = True
        ppc["obj"] = ppci["obj"]
        ppc["success"] = ppci["success"]
        ppc["et"] = ppci["et"]
        ppc["f"] = ppci["f"]
        return ppc, multinetwork

    for i, bus in sol["bus"].items():
        bus_idx = int(i) - 1
        ppci["bus"][bus_idx, VM] = bus["vm"]
        ppci["bus"][bus_idx, VA] = math.degrees(bus["va"])

    for i, gen in sol["gen"].items():
        gen_idx = int(i) - 1
        ppci["gen"][gen_idx, PG] = gen["pg"]
        ppci["gen"][gen_idx, QG] = gen["qg"]

    # read Q from branch results (if not DC calculation)
    dc_results = np.isnan(sol["branch"]["1"]["qf"])
    # read branch status results (OTS)
    branch_status = "br_status" in sol["branch"]["1"]
    for i, branch in sol["branch"].items():
        br_idx = int(i) - 1
        ppci["branch"][br_idx, PF] = branch["pf"]
        ppci["branch"][br_idx, PT] = branch["pt"]
        if not dc_results:
            ppci["branch"][br_idx, QF] = branch["qf"]
            ppci["branch"][br_idx, QT] = branch["qt"]
        if branch_status:
            ppci["branch"][br_idx, BR_STATUS] = branch["br_status"] > 0.5

    result = _copy_results_ppci_to_ppc(ppci, ppc, options["mode"])
    return result, multinetwork


def init_ne_line(net, new_line_index, construction_costs=None):
    """
    init function for new line dataframe, which specifies the possible new lines being built by power models opt

    Parameters
    ----------
    net - pp net
    new_line_index (list) - indices of new lines. These are copied to the new dataframe net["ne_line"] from net["line"]
    construction_costs (list, 0.) - costs of newly constructed lines

    Returns
    -------

    """
    # init dataframe
    net["ne_line"] = net["line"].loc[new_line_index, :]
    # add costs, if None -> init with zeros
    construction_costs = np.zeros(len(new_line_index)) if construction_costs is None else construction_costs
    net["ne_line"].loc[new_line_index, "construction_cost"] = construction_costs
    # set in service, but only in ne line dataframe
    net["ne_line"].loc[new_line_index, "in_service"] = True
    # init res_ne_line to save built status afterwards
    net["res_ne_line"] = pd.DataFrame(data=0, index=new_line_index, columns=["built"], dtype=int)
