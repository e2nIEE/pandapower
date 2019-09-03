import json
import math
import os
import tempfile
from math import pi

import numpy as np

from pandapower import pp_dir
from pandapower.auxiliary import _add_auxiliary_elements, _clean_up
from pandapower.opf.pm_tnep import build_ne_branch, CONSTRUCTION_COST
from pandapower.pd2ppc import _pd2ppc
from pandapower.pypower.idx_brch import BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, F_BUS, T_BUS, BR_STATUS, \
    ANGMIN, ANGMAX, TAP, SHIFT, PF, PT, QF, QT
from pandapower.pypower.idx_bus import BUS_I, ZONE, BUS_TYPE, VMAX, VMIN, VA, VM, BASE_KV, PD, QD, GS, BS
from pandapower.pypower.idx_cost import MODEL, COST, NCOST
from pandapower.pypower.idx_gen import PG, QG, GEN_BUS, VG, QMAX, GEN_STATUS, QMIN, PMIN, PMAX
from pandapower.results import _extract_results, _copy_results_ppci_to_ppc
from pandapower.results import reset_results

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def _runpm(net):  # pragma: no cover
    net["OPF_converged"] = False
    net["converged"] = False
    _add_auxiliary_elements(net)
    reset_results(net)
    ppc, ppci = _pd2ppc(net)
    ppci = build_ne_branch(net, ppci)
    net["_ppc_opf"] = ppci
    pm = ppc_to_pm(net, ppci)
    pm = add_pm_options(pm, net)
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
    temp_name = next(tempfile._get_candidate_names())
    buffer_file = os.path.join(tempfile.gettempdir(), "pp_to_pm_" + temp_name + ".json")
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
        if pm_bus not in pd_bus:
            pd_bus[pm_bus] = pd
            qd_bus[pm_bus] = qd
        else:
            pd_bus[pm_bus] += pd
            qd_bus[pm_bus] += qd

        load_idx += 1
    return load_idx


def get_branch_angles(row, correct_pm_network_data):
    angmin = row[ANGMIN].real
    angmax = row[ANGMAX].real
    # check if angles are too small for PowerModels OPF (recommendation from Carleton Coffrin himself)
    if correct_pm_network_data:
        if angmin < -60.:
            logger.debug("changed voltage angle minimum of branch {}, "
                           "to -60 from {} degrees".format(int(row[0].real), angmin))
            angmin = -60.
        if angmax > 60.:
            logger.debug("changed voltage angle maximum of branch {} to 60. "
                           "from {} degrees".format(int(row[0].real), angmax))
            angmax = 60.
    angmin = (angmin / 180.) * pi  # convert to p.u. as well
    angmax = (angmax / 180.) * pi  # convert to p.u. as well
    return angmin, angmax


def ppc_to_pm(net, ppci):  # pragma: no cover
    # create power models dict. Similar to matpower case file. ne_branch is for a tnep case
    pm = {"gen": dict(), "branch": dict(), "bus": dict(), "dcline": dict(), "load": dict(), "storage": dict(),
          "ne_branch": dict(), "switch": dict(),
          "baseMVA": ppci["baseMVA"], "source_version": "2.0.0", "shunt": dict(),
          "sourcetype": "matpower", "per_unit": True, "name": net.name}
    load_idx = 1
    shunt_idx = 1
    # PowerModels has a load model -> add loads and sgens to pm["load"]

    # temp dicts which hold the sum of p, q of loads + sgens
    pd_bus = dict()
    qd_bus = dict()
    load_idx = _pp_element_to_pm(net, pm, "load", pd_bus, qd_bus, load_idx)
    load_idx = _pp_element_to_pm(net, pm, "sgen", pd_bus, qd_bus, load_idx)
    load_idx = _pp_element_to_pm(net, pm, "storage", pd_bus, qd_bus, load_idx)
    correct_pm_network_data = net._options["correct_pm_network_data"]

    for row in ppci["bus"]:
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
            # This will be called if ppc PQ != sum at bus.
            logger.info("PQ mismatch. Adding another load at idx {}".format(load_idx))
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
    for idx, row in enumerate(ppci["branch"], start=1):
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
        branch["angmin"], branch["angmax"] = get_branch_angles(row, correct_pm_network_data)
        branch["tap"] = row[TAP].real
        branch["shift"] = math.radians(row[SHIFT].real)
        pm["branch"][str(idx)] = branch

    for idx, row in enumerate(ppci["gen"], start=1):
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

    if "ne_branch" in ppci:
        for idx, row in enumerate(ppci["ne_branch"], start=1):
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
            branch["angmin"], branch["angmax"] = get_branch_angles(row, correct_pm_network_data)
            branch["tap"] = row[TAP].real
            branch["shift"] = math.radians(row[SHIFT].real)
            branch["construction_cost"] = row[CONSTRUCTION_COST].real
            pm["ne_branch"][str(idx)] = branch

    if len(ppci["gencost"]) > len(ppci["gen"]):
        logger.warning("PowerModels.jl does not consider reactive power cost - costs are ignored")
        ppci["gencost"] = ppci["gencost"][:ppci["gen"].shape[0], :]
    for idx, row in enumerate(ppci["gencost"], start=1):
        gen = pm["gen"][str(idx)]
        gen["model"] = int(row[MODEL])
        if gen["model"] == 1:
            gen["ncost"] = int(row[NCOST])
            gen["cost"] = row[COST:COST + gen["ncost"] * 2].tolist()
        elif gen["model"] == 2:
            gen["ncost"] = 3
            gen["cost"] = [0] * 3
            costs = row[COST:]
            if len(costs) > 3:
                logger.info(costs)
                raise ValueError("Maximum quadratic cost function allowed")
            gen["cost"][-len(costs):] = costs
    return pm


def pm_results_to_ppc_results(net, ppc, ppci, result_pm):  # pragma: no cover
    options = net._options
    # status if result is from multiple grids
    multinetwork = False
    sol = result_pm["solution"]
    ppci["obj"] = result_pm["objective"]
    termination_status = str(result_pm["termination_status"])
    ppci["success"] = "LOCALLY_SOLVED" in termination_status or "OPTIMAL" in termination_status
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


def add_pm_options(pm, net):
    if "pm_solver" in net._options:
        pm["pm_solver"] = net._options["pm_solver"]
    if "pm_model" in net._options:
        pm["pm_model"] = net._options["pm_model"]
    pm["correct_pm_network_data"] = net._options["correct_pm_network_data"]
    return pm
