# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
import tempfile
import math
import os
import json

from pandapower.auxiliary import _add_ppc_options, _add_opf_options
from pandapower.pd2ppc import _pd2ppc
from pandapower.results import _extract_results, reset_results, _copy_results_ppci_to_ppc
from pandapower.auxiliary import _clean_up, _add_auxiliary_elements
from pandapower import pp_dir

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


def runpm(net, julia_file, pp_to_pm_callback=None, calculate_voltage_angles=True,
          trafo_model="t", delta=0, trafo3w_losses="hv", check_connectivity=True): #pragma: no cover
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
                 trafo_model="t", delta=0, trafo3w_losses="hv", check_connectivity=True): #pragma: no cover
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
        **julia_file** (str, None) - path to a custom julia optimization file

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
                 trafo_model="t", delta=0, trafo3w_losses="hv", check_connectivity=True): #pragma: no cover
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
        **julia_file** (str, None) - path to a custom julia optimization file

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


def _runpm(net): #pragma: no cover
    net["OPF_converged"] = False
    net["converged"] = False
    _add_auxiliary_elements(net)
    reset_results(net)
    ppc, ppci = _pd2ppc(net)
    net["_ppc_opf"] = ppci
    pm = ppc_to_pm(net, ppci)
    net._pm = pm
    if net._options["pp_to_pm_callback"] is not None:
        net._options["pp_to_pm_callback"](net, ppci, pm)
    result_pm = _call_powermodels(pm, net._options["julia_file"])
    net._pm_res = result_pm
    result = pm_results_to_ppc_results(net, ppc, ppci, result_pm)
    net._pm_result = result_pm
    success = ppc["success"]
    if success:
        _extract_results(net, result)
        _clean_up(net)
        net["OPF_converged"] = True
    else:
        _clean_up(net, res=False)
        logger.warning("OPF did not converge!")


def _call_powermodels(pm, julia_file): #pragma: no cover
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


def ppc_to_pm(net, ppc): #pragma: no cover
    pm = {"gen": dict(), "branch": dict(), "bus": dict(), "dcline": dict(), "load": dict(), "storage": dict(),
          "baseMVA": ppc["baseMVA"], "source_version": "2.0.0", "shunt": dict(),
          "sourcetype": "matpower", "per_unit": True, "name": net.name}
    load_idx = 1
    shunt_idx = 1
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
        if pd != 0 or qd != 0:
            pm["load"][str(load_idx)] = {"pd": pd, "qd": qd, "load_bus": idx,
                                         "status": True, "index": load_idx}
            load_idx += 1
        bs = row[BS]
        gs = row[GS]
        if pd != 0 or qd != 0:
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
                print(costs)
                raise ValueError("Maximum quadratic cost function allowed")
            gen["cost"][-len(costs):] = costs
    return pm


def pm_results_to_ppc_results(net, ppc, ppci, result_pm): #pragma: no cover
    options = net._options
    sol = result_pm["solution"]
    for i, bus in sol["bus"].items():
        ppci["bus"][int(i) - 1, VM] = bus["vm"]
        ppci["bus"][int(i) - 1, VA] = math.degrees(bus["va"])

    for i, gen in sol["gen"].items():
        ppci["gen"][int(i) - 1, PG] = gen["pg"]
        ppci["gen"][int(i) - 1, QG] = gen["qg"]

    dc_results = np.isnan(sol["branch"]["1"]["qf"])
    for i, branch in sol["branch"].items():
        ppci["branch"][int(i) - 1, PF] = branch["pf"]
        ppci["branch"][int(i) - 1, PT] = branch["pt"]
        if not dc_results:
            ppci["branch"][int(i) - 1, QF] = branch["qf"]
            ppci["branch"][int(i) - 1, QT] = branch["qt"]

    ppc["obj"] = result_pm["objective"]
    ppci["success"] = result_pm["status"] == "LocalOptimal"
    ppci["et"] = result_pm["solve_time"]
    ppci["f"] = result_pm["objective"]
    result = _copy_results_ppci_to_ppc(ppci, ppc, options["mode"])
    return result
