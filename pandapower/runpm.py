# -*- coding: utf-8 -*-

# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from pandapower.auxiliary import _add_ppc_options, _add_opf_options
from pandapower.pd2ppc import _pd2ppc
from pandapower.pf.pfsoln import pfsoln
from pandapower.pf.makeYbus import makeYbus
from pandapower.pf.ppci_variables import _get_pf_variables_from_ppci
from pandapower.results import _extract_results_opf, reset_results, _copy_results_ppci_to_ppc
from pandapower.powerflow import _add_auxiliary_elements
from pandapower.auxiliary import _clean_up

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

import numpy as np
import tempfile
import os
import json

from pandapower.idx_gen import QC1MAX, PG,QC2MAX, RAMP_AGC, QG, GEN_BUS, RAMP_10,VG, MBASE, PC2, QMAX,GEN_STATUS, QMIN,QC1MIN, QC2MIN, PC1, PMIN, PMAX, RAMP_Q, RAMP_30, APF
from pandapower.idx_bus import BUS_I, ZONE, BUS_TYPE, VMAX, VMIN, VA, VM, BASE_KV, PD, QD, GS, BS
from pandapower.idx_brch import BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, F_BUS, T_BUS, BR_STATUS, ANGMIN, ANGMAX, TAP, SHIFT
from pandapower.idx_cost import MODEL, STARTUP, SHUTDOWN, COST

def runpm(net, julia_file=None, pp_to_pm_callback=None, calculate_voltage_angles=True,
          trafo_model="t", delta=0, trafo3w_losses="hv"):
    """
    Runs a power system optimization using PowerModels.jl.
    Flexibilities, constraints and cost parameters are defined in the pandapower element tables.

    Flexibilities can be defined in net.sgen / net.gen /net.load
    net.sgen.controllable if a static generator is controllable. If False,
    the active and reactive power are assigned as in a normal power flow. If True, the following
    flexibilities apply:
        - net.sgen.min_p_kw / net.sgen.max_p_kw
        - net.sgen.min_q_kvar / net.sgen.max_q_kvar
        - net.load.min_p_kw / net.load.max_p_kw
        - net.load.min_q_kvar / net.load.max_q_kvar
        - net.gen.min_p_kw / net.gen.max_p_kw
        - net.gen.min_q_kvar / net.gen.max_q_kvar
        - net.ext_grid.min_p_kw / net.ext_grid.max_p_kw
        - net.ext_grid.min_q_kvar / net.ext_grid.max_q_kvar
        - net.dcline.min_q_to_kvar / net.dcline.max_q_to_kvar / net.dcline.min_q_from_kvar / net.dcline.max_q_from_kvar

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
                     trafo_model=trafo_model, check_connectivity=False,
                     mode="opf", copy_constraints_to_ppc=True,
                     r_switch=0, init_vm_pu="flat", init_va_degree="flat",
                     enforce_q_lims=True, recycle=dict(_is_elements=False, ppc=False, Ybus=False),
                     voltage_depend_loads=False, delta=delta, trafo3w_losses=trafo3w_losses)
    _add_opf_options(net, trafo_loading='current', ac=True, init="flat", numba=True)
    _runpm(net, julia_file, pp_to_pm_callback)

def _runpm(net, julia_file=None, pp_to_pm_callback=None):
    net["OPF_converged"] = False
    net["converged"] = False
    _add_auxiliary_elements(net)
    reset_results(net)
    ppc, ppci = _pd2ppc(net)
    pm = ppc_to_pm(net, ppci)
    net._pm = pm
    if pp_to_pm_callback is not None:
        pp_to_pm_callback(net, ppci, pm)
    result_pm = _call_powermodels(pm, julia_file)
    net._result_pm = result_pm
    result = pm_results_to_ppc_results(ppc, ppci, result_pm)
    success = ppc["success"]
    net["_ppc_opf"] = ppci
    if success:
        _extract_results_opf(net, result)
        _clean_up(net)
        net["OPF_converged"] = True
    else:
        _clean_up(net)
        logger.warning("OPF did not converge!")

def _call_powermodels(pm, julia_file=None):
    buffer_file = os.path.join(tempfile.gettempdir(), "pp_pm.json")
    with open(buffer_file, 'w') as outfile:
        json.dump(pm, outfile)
    try:
        import julia
    except ImportError:
        raise ImportError("Please install pyjulia to run pandapower with PowerModels.jl")
    try:
        j = julia.Julia()
    except:
        raise UserWarning("Could not connect to julia, please check that Julia is installed and pyjulia is correctly configured")

    if julia_file is None:
        import pandapower.opf
        folder = os.path.abspath(os.path.dirname(pandapower.opf.__file__))
        julia_file = os.path.join(folder, 'run_powermodels.jl')
    try:
        run_powermodels = j.include(julia_file)
    except:
        raise UserWarning("File %s could not be imported"%julia_file)
    result_pm = run_powermodels(buffer_file)
    return result_pm

def ppc_to_pm(net, ppc):
    pm = {"gen": dict(), "branch": dict(), "bus": dict(), "dcline": dict(), "load": dict(),
          "baseMVA": ppc["baseMVA"],  "source_version": "2.0.0", "shunt": dict(),
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
        branch["shift"] = row[SHIFT].real
        pm["branch"][str(idx)] = branch

    for idx, row in enumerate(ppc["gen"], start=1):
        gen = dict()
        gen["qc1max"] = row[QC1MAX]
        gen["pg"] = row[PG]
        gen["qc2max"] = row[QC2MAX]
        gen["ramp_agc"] = row[RAMP_AGC]
        gen["qg"] = row[QG]
        gen["gen_bus"] = int(row[GEN_BUS]) + 1
        gen["ramp_10"] = row[RAMP_10]
        gen["vg"] = row[VG]
        gen["mbase"] = row[MBASE]
        gen["pc2"] = row[PC2]
        gen["qmax"] = row[QMAX]
        gen["gen_status"] = int(row[GEN_STATUS])
        gen["qmin"] = row[QMIN]
        gen["qc1min"] = row[QC1MIN]
        gen["qc2min"] = row[QC2MIN]
        gen["pc1"] = row[PC1]
        gen["ramp_q"] = row[RAMP_Q]
        gen["ramp_30"] = row[RAMP_30]
        gen["pmin"] = row[PMIN]
        gen["pmax"] = row[PMAX]
        gen["apf"] = row[APF]
        gen["index"] = idx
        pm["gen"][str(idx)] = gen

    for idx, row in enumerate(ppc["gencost"], start=1):
        gen = pm["gen"][str(idx)]
        gen["model"] = int(row[MODEL])
        gen["shutdown"] = row[SHUTDOWN]
        gen["startup"] = row[STARTUP]
        gen["ncost"] = 1#int(row[NCOST])
        gen["cost"] = [row[COST], row[COST +1], 0]
    return pm

def pm_results_to_ppc_results(ppc, ppci, result_pm):
    V = np.zeros(len(ppci["bus"]), dtype="complex")
    for i, bus in result_pm["solution"]["bus"].items():
        V[int(i)-1] = bus["vm"] * np.exp(1j*bus["va"])

    for i, gen in result_pm["solution"]["gen"].items():
        ppci["gen"][int(i)-1, PG] = gen["pg"]
        ppci["gen"][int(i)-1, QG] = gen["qg"]

    ppci["obj"] = result_pm["objective"]
    ppci["success"] = result_pm["status"] == "LocalOptimal"
    ppci["et"] = result_pm["solve_time"]
    ppci["f"] = result_pm["objective"]
    ppci["internal_gencost"] = result_pm["objective"]

    baseMVA, bus, gen, branch, ref, pv, pq, _, gbus, V0, ref_gens = _get_pf_variables_from_ppci(ppci)
    Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
    bus, gen, branch = pfsoln(baseMVA, bus, gen, branch, Ybus, Yf, Yt, V, ref, ref_gens)
    ppc["bus"][:, VM] = np.nan
    ppc["bus"][:, VA] = np.nan
    result = _copy_results_ppci_to_ppc(ppci, ppc, "opf")
    return result

if __name__ == '__main__':
    import pandapower as pp
    net = pp.create_empty_network()

    #create buses
    bus1 = pp.create_bus(net, vn_kv=220.)
    bus2 = pp.create_bus(net, vn_kv=110.)
    bus3 = pp.create_bus(net, vn_kv=110.)
    bus4 = pp.create_bus(net, vn_kv=110.)
    bus5 = pp.create_bus(net, vn_kv=110.)

    bus6 = pp.create_bus(net, vn_kv=110., in_service=False)


    #create 220/110 kV transformer
    pp.create_transformer3w_from_parameters(net, bus1, bus2, bus5, vn_hv_kv=220, vn_mv_kv=110,
                                            vn_lv_kv=110, vsc_hv_percent=10., vsc_mv_percent=10.,
                                            vsc_lv_percent=10., vscr_hv_percent=0.5,
                                            vscr_mv_percent=0.5, vscr_lv_percent=0.5, pfe_kw=100.,
                                            i0_percent=0.1, shift_mv_degree=0, shift_lv_degree=0,
                                            sn_hv_kva=100e3, sn_mv_kva=50e3, sn_lv_kva=50e3)
    net.trafo3w["max_loading_percent"] = 50

    #create 110 kV lines
    pp.create_line(net, bus2, bus3, length_km=70., std_type='149-AL1/24-ST1A 110.0')
    pp.create_line(net, bus3, bus4, length_km=50., std_type='149-AL1/24-ST1A 110.0')
    pp.create_line(net, bus4, bus2, length_km=40., std_type='149-AL1/24-ST1A 110.0')
    pp.create_line(net, bus4, bus5, length_km=30., std_type='149-AL1/24-ST1A 110.0')
    net.line["max_loading_percent"] = 30

    #create loads
    pp.create_load(net, bus2, p_kw=60e3, controllable = False)
    pp.create_load(net, bus3, p_kw=70e3, controllable = False)
    pp.create_load(net, bus4, p_kw=10e3, controllable = False)

    #create generators
    eg = pp.create_ext_grid(net, bus1, min_p_kw = -1e9, max_p_kw = 1e9)
    g0 = pp.create_gen(net, bus3, p_kw=-80*1e3, min_p_kw=-80e3, max_p_kw=0,vm_pu=1.01, controllable=True)
    g1 = pp.create_gen(net, bus4, p_kw=-100*1e3, min_p_kw=-100e3, max_p_kw=0, vm_pu=1.01, controllable=True)

    costeg = pp.create_polynomial_cost(net, 0, 'ext_grid', np.array([-1, 0]))
    costgen1 = pp.create_polynomial_cost(net, 0, 'gen', np.array([-1, 0]))
    costgen2 = pp.create_polynomial_cost(net, 1, 'gen', np.array([-1, 0]))

    runpm(net, "run_powermodels.jl")
    if net.OPF_converged:
        from pandapower.test.consistency_checks import consistency_checks
        consistency_checks(net)