# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import json
import math
import os
import tempfile

from os import remove
from os.path import isfile
from collections import OrderedDict
import numpy as np
import pandas as pd

from pandapower.auxiliary import _add_ppc_options, _add_opf_options, _add_auxiliary_elements
from pandapower.build_branch import _calc_line_parameter
from pandapower.pd2ppc import _pd2ppc
from pandapower.pypower.idx_brch import ANGMIN, ANGMAX, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, \
    TAP, SHIFT, branch_cols, F_BUS, T_BUS, BR_STATUS
from pandapower.pypower.idx_bus import ZONE, VA, BASE_KV, BS, GS, BUS_I, BUS_TYPE, VMAX, VMIN, \
     VM, PD, QD
from pandapower.pypower.idx_cost import MODEL, NCOST, COST
from pandapower.pypower.idx_gen import PG, QG, GEN_BUS, VG, GEN_STATUS, QMAX, QMIN, PMIN, PMAX
from pandapower.results import init_results


# const value in branch for tnep
CONSTRUCTION_COST = 23
try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)):
            return None
        return json.JSONEncoder.default(self, obj)


def convert_pp_to_pm(net, pm_file_path=None, correct_pm_network_data=True,
                     calculate_voltage_angles=True,
                     ac=True, silence=True, trafo_model="t", delta=1e-8, trafo3w_losses="hv",
                     check_connectivity=True, pp_to_pm_callback=None, pm_model="ACPPowerModel",
                     pm_solver="ipopt",
                     pm_mip_solver="cbc", pm_nl_solver="ipopt", opf_flow_lim="S", pm_tol=1e-8,
                     voltage_depend_loads=False, from_time_step=None, to_time_step=None, **kwargs):
    """
    Converts a pandapower net to a PowerModels.jl datastructure and saves it to a json file
    INPUT:
        **net**  - pandapower net
    OPTIONAL:
        **pm_file_path** (str, None) - Specifiy the filename, under which the .json file for
        powermodels is stored. If you want to keep the file after optimization, you should also
        set delete_buffer_file to False!

        **correct_pm_network_data** (bool, True) - checks if network data is correct.
        If not tries to correct it

        **silence** (bool, True) - Suppresses information and warning messages output by PowerModels

        **pm_model** (str, "ACPPowerModel") - The PowerModels.jl model to use

        **pm_solver** (str, "ipopt") - The "main" power models solver

        **pm_mip_solver** (str, "cbc") - The mixed integer solver (when "main" solver == juniper)

        **pm_nl_solver** (str, "ipopt") - The nonlinear solver (when "main" solver == juniper)

        **pm_time_limits** (Dict, None) - Time limits in seconds for power models interface.
        To be set as a dict like
        {"pm_time_limit": 300., "pm_nl_time_limit": 300., "pm_mip_time_limit": 300.}

        **pm_log_level** (int, 0) - solver log level in power models

        **delete_buffer_file** (Bool, True) - If True, the .json file used by powermodels will be
        deleted after optimization.

        **pp_to_pm_callback** (function, None) - callback function to add data to the PowerModels
        data structure

        **opf_flow_lim** (str, "I") - Quantity to limit for branch flow constraints, in line with
        matpower's "opf.flowlim" parameter:

            "S" - apparent power flow (limit in MVA),

            "I" - current magnitude (limit in MVA at 1 p.u. voltage)

        **pm_tol** (float, 1e-8) - default desired convergence tolerance for solver to use.

        **voltage_depend_loads** (bool, False) - consideration of voltage-dependent loads.
        If False, net.load.const_z_percent and net.load.const_i_percent are not considered,
        i.e. net.load.p_mw and net.load.q_mvar are considered as constant-power loads.

    Returns
    -------
    """

    net._options = {}

    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode="opf", switch_rx_ratio=2, init_vm_pu="flat", init_va_degree="flat",
                     enforce_q_lims=True, recycle=dict(_is_elements=False, ppc=False, Ybus=False),
                     voltage_depend_loads=voltage_depend_loads, delta=delta,
                     trafo3w_losses=trafo3w_losses)
    _add_opf_options(net, trafo_loading='power', ac=ac, init="flat", numba=True,
                     pp_to_pm_callback=pp_to_pm_callback, pm_solver=pm_solver, pm_model=pm_model,
                     correct_pm_network_data=correct_pm_network_data, silence=silence,
                     pm_mip_solver=pm_mip_solver,
                     pm_nl_solver=pm_nl_solver, opf_flow_lim=opf_flow_lim, pm_tol=pm_tol)

    net, pm, ppc, ppci = convert_to_pm_structure(net, from_time_step=from_time_step, 
                                                 to_time_step=to_time_step)
    buffer_file = dump_pm_json(pm, pm_file_path)
    if pm_file_path is None and isfile(buffer_file):
        remove(buffer_file)
    return pm


logger = logging.getLogger(__name__)


def convert_to_pm_structure(net, opf_flow_lim="S", from_time_step=None, to_time_step=None, 
                            **kwargs):
    if net["_options"]["voltage_depend_loads"] and not (
            np.allclose(net.load.const_z_percent.values, 0) and
            np.allclose(net.load.const_i_percent.values, 0)):
        logger.error("pandapower optimal_powerflow does not support voltage depend loads.")
    net["OPF_converged"] = False
    net["converged"] = False
    _add_auxiliary_elements(net)
    init_results(net)
    ppc, ppci = _pd2ppc(net)
    ppci = build_ne_branch(net, ppci)
    net["_ppc_opf"] = ppci
    pm = ppc_to_pm(net, ppci)
    pm = add_pm_options(pm, net)
    pm = add_params_to_pm(net, pm)
    if from_time_step is not None and to_time_step is not None:
        pm = add_time_series_to_pm(net, pm, from_time_step, to_time_step)
    pm = allow_multi_ext_grids(net, pm)
    net._pm = pm
    return net, pm, ppc, ppci


def dump_pm_json(pm, buffer_file=None):
    # dump pm dict to buffer_file (*.json)
    if buffer_file is None:
        # if no buffer file is provided a random file name is generated
        temp_name = next(tempfile._get_candidate_names())
        buffer_file = os.path.join(tempfile.gettempdir(), "pp_to_pm_" + temp_name + ".json")
    logger.debug("writing PowerModels data structure to %s" % buffer_file)
    with open(buffer_file, 'w') as outfile:
        json.dump(pm, outfile, indent=4, sort_keys=True, cls=NumpyEncoder)
    return buffer_file


def _pp_element_to_pm(net, pm, element, pd_bus, qd_bus, load_idx):
    bus_lookup = net._pd2ppc_lookups["bus"]

    pm_lookup = np.ones(max(net[element].index) + 1, dtype=np.int64) * -1 if len(net[element].index) \
        else np.array([], dtype=np.int64)
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

        pm_lookup[idx] = load_idx
        load_idx += 1
    return load_idx, pm_lookup


def get_branch_angles(row, correct_pm_network_data):
    angmin = row[ANGMIN].real
    angmax = row[ANGMAX].real
    # check if angles are too small for PowerModels OPF (recommendation from Carleton Coffrin
    # himself)
    if correct_pm_network_data:
        if angmin < -60.:
            logger.debug("changed voltage angle minimum of branch {}, "
                         "to -60 from {} degrees".format(int(row[0].real), angmin))
            angmin = -60.
        if angmax > 60.:
            logger.debug("changed voltage angle maximum of branch {} to 60. "
                         "from {} degrees".format(int(row[0].real), angmax))
            angmax = 60.
    # convert to rad (per unit value)
    angmin = math.radians(angmin) #/ (360/(2*np.pi))
    angmax = math.radians(angmax) #/ (360/(2*np.pi))
    return angmin, angmax


def create_pm_lookups(net, pm_lookup):
    for key, val in net._pd2ppc_lookups.items():
        if isinstance(val, dict):
            # lookup is something like "branch" with dict as val -> iterate over the subdicts
            pm_val = dict()
            for subkey, subval in val.items():
                pm_val[subkey] = tuple((v + 1 for v in subval))
        elif isinstance(val, int) or isinstance(val, np.ndarray):
            # lookup is a numpy array
            # julia starts counting at 1 instead of 0
            pm_val = val + 1
            # restore -1 for not existing elements
            pm_val[pm_val == 0] = -1
        else:
            # val not supported
            continue
        pm_lookup[key] = pm_val
    net._pd2pm_lookups = pm_lookup
    return net


def ppc_to_pm(net, ppci):
    # create power models dict. Similar to matpower case file. ne_branch is for a tnep case
    # "per_unit == True" means that the grid data in PowerModels are per-unit values. In this
    # ppc-to-pm process, the grid data schould be transformed according to baseMVA = 1.
    pm = {"gen": dict(), "branch": dict(), "bus": dict(), "dcline": dict(), "load": dict(),
          "storage": dict(),
          "ne_branch": dict(), "switch": dict(),
          "baseMVA": ppci["baseMVA"], "source_version": "2.0.0", "shunt": dict(),
          "sourcetype": "matpower", "per_unit": True, "name": net.name}
    baseMVA = ppci["baseMVA"]
    load_idx = 1
    shunt_idx = 1
    # PowerModels has a load model -> add loads and sgens to pm["load"]

    # temp dicts which hold the sum of p, q of loads + sgens
    pd_bus = dict()
    qd_bus = dict()
    load_idx, load_lookup = _pp_element_to_pm(net, pm, "load", pd_bus, qd_bus, load_idx)
    load_idx, sgen_lookup = _pp_element_to_pm(net, pm, "sgen", pd_bus, qd_bus, load_idx)
    load_idx, storage_lookup = _pp_element_to_pm(net, pm, "storage", pd_bus, qd_bus, load_idx)
    pm_lookup = {"load": load_lookup, "sgen": sgen_lookup, "storage": storage_lookup}
    net = create_pm_lookups(net, pm_lookup)

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

        pd_value = row[PD]
        qd_value = row[QD]

        # pd and qd are the PQ values in the ppci, if they are equal to the sum in load data is
        # consistent
        if idx in pd_bus:
            pd_value -= pd_bus[idx]
            qd_value -= qd_bus[idx]
        # if not we have to add more loads wit the remaining value
        pq_mismatch = not np.allclose(pd_value, 0.) or not np.allclose(qd_value, 0.)
        if pq_mismatch:
            # This will be called if ppc PQ != sum at bus.
            logger.info("PQ mismatch. Adding another load at idx {}".format(load_idx))
            pm["load"][str(load_idx)] = {"pd": pd_value, "qd": qd_value, "load_bus": idx,
                                         "status": True, "index": load_idx}
            load_idx += 1
        # if bs or gs != 0. -> shunt element at this bus
        bs = row[BS] / baseMVA # to be validated
        gs = row[GS] / baseMVA # to be validated
        if not np.allclose(bs, 0.) or not np.allclose(gs, 0.):
            pm["shunt"][str(shunt_idx)] = {"gs": gs, "bs": bs, "shunt_bus": idx,
                                           "status": True, "index": shunt_idx}
            shunt_idx += 1
        pm["bus"][str(idx)] = bus

    n_lines = net.line.in_service.sum()
    for idx, row in enumerate(ppci["branch"], start=1):
        branch = dict()
        branch["index"] = idx
        branch["transformer"] = bool(idx > n_lines)
        branch["br_r"] = row[BR_R].real / baseMVA
        branch["br_x"] = row[BR_X].real / baseMVA
        branch["g_fr"] = - row[BR_B].imag / 2.0 / baseMVA
        branch["g_to"] = - row[BR_B].imag / 2.0 / baseMVA
        branch["b_fr"] = row[BR_B].real / 2.0 * baseMVA
        branch["b_to"] = row[BR_B].real / 2.0 * baseMVA

        if net._options["opf_flow_lim"] == "S":  # or branch["transformer"]:
            branch["rate_a"] = row[RATE_A].real if row[RATE_A] > 0 else row[RATE_B].real
            branch["rate_b"] = row[RATE_B].real
            branch["rate_c"] = row[RATE_C].real
        elif net._options["opf_flow_lim"] == "I":  # need to call _run_opf_cl from PowerModels
            f = net._pd2ppc_lookups["branch"]["line"][0]
            f = int(row[F_BUS].real)  # from bus of this line
            vr = ppci["bus"][f][BASE_KV]
            branch["c_rating_a"] = row[RATE_A].real if row[RATE_A] > 0 else row[RATE_B].real
            branch["c_rating_a"] = branch["c_rating_a"]
            branch["c_rating_b"] = row[RATE_B].real
            branch["c_rating_c"] = row[RATE_C].real
        else:
            logger.error("Branch flow limit %s not understood", net._options["opf_flow_lim"])

        branch["f_bus"] = int(row[F_BUS].real) + 1
        branch["t_bus"] = int(row[T_BUS].real) + 1
        branch["br_status"] = int(row[BR_STATUS].real)
        branch["angmin"], branch["angmax"] = get_branch_angles(row, correct_pm_network_data)
        branch["tap"] = row[TAP].real
        branch["shift"] = math.radians(row[SHIFT].real)
        pm["branch"][str(idx)] = branch

    #### create pm["gen"]
    gen_idxs_pm = [str(i+1) for i in range(len(ppci["gen"]))]
    gen_df = pd.DataFrame(index=gen_idxs_pm)
    gen_df["pg"] = ppci["gen"][:, PG]
    gen_df["qg"] = ppci["gen"][:, QG]
    gen_df["gen_bus"] = (ppci["gen"][:, GEN_BUS] + 1).astype(np.int64)
    gen_df["vg"] = ppci["gen"][:, VG]
    gen_df["qmax"] = ppci["gen"][:, QMAX]
    gen_df["gen_status"] = ppci["gen"][:, GEN_STATUS].astype(np.int64)
    gen_df["qmin"] = ppci["gen"][:, QMIN]
    gen_df["pmin"] = ppci["gen"][:, PMIN]
    gen_df["pmax"] = ppci["gen"][:, PMAX]
    gen_df["index"] = list(map(int, gen_idxs_pm))
    # add cost-parameters
    if len(ppci["gencost"]) > len(ppci["gen"]):
        logger.warning("PowerModels.jl does not consider reactive power cost - costs are ignored")
        ppci["gencost"] = ppci["gencost"][:ppci["gen"].shape[0], :]
    gen_df["startup"] = 0.0
    gen_df["shutdown"] = 0.0
    model_type = ppci["gencost"][:, MODEL].astype(np.int64)
    gen_df["model"] = model_type
    # calc ncost and cost
    ncost = np.array([0] * len(ppci["gen"]))
    cost = [[0, 0, 0] for i in gen_idxs_pm]
    ncost[model_type==1] = ppci["gencost"][:, NCOST][model_type==1]
    ncost[model_type==2] = 3
    for i in np.where(model_type==1)[0]:
        cost[i] = ppci["gencost"][i, COST:COST + ncost[i] * 2].tolist()
    for i in np.where(model_type==2)[0]:
        cost_value = ppci["gencost"][i, COST:].tolist()
        if len(cost_value) > 3:
            raise ValueError("Maximum quadratic cost function allowed")
        cost[i][-len(cost_value):] = cost_value
    gen_df["ncost"] = ncost
    gen_df["cost"] = cost
    # dataframe to dict
    pm["gen"] = gen_df.astype(object).T.to_dict()

    if "ne_branch" in ppci:
        for idx, row in enumerate(ppci["ne_branch"], start=1):
            branch = dict()
            branch["index"] = idx
            branch["transformer"] = False
            branch["br_r"] = row[BR_R].real / baseMVA
            branch["br_x"] = row[BR_X].real / baseMVA
            branch["g_fr"] = - row[BR_B].imag / 2.0 / baseMVA
            branch["g_to"] = - row[BR_B].imag / 2.0 / baseMVA
            branch["b_fr"] = row[BR_B].real / 2.0 * baseMVA
            branch["b_to"] = row[BR_B].real / 2.0 * baseMVA

            if net._options["opf_flow_lim"] == "S":  # --> Rate_a is always needed for the TNEP problem, right?
                branch["rate_a"] = row[RATE_A].real if row[RATE_A] > 0 else row[RATE_B].real
                branch["rate_b"] = row[RATE_B].real
                branch["rate_c"] = row[RATE_C].real
            elif net._options["opf_flow_lim"] == "I":
                f, t = net._pd2ppc_lookups["branch"]["line"]
                f = int(row[F_BUS].real)  # from bus of this line
                vr = ppci["bus"][f][BASE_KV]
                row[RATE_A] = row[RATE_A] / (vr * np.sqrt(3))

                branch["c_rating_a"] = row[RATE_A].real if row[RATE_A] > 0 else row[RATE_B].real
                branch["c_rating_b"] = row[RATE_B].real
                branch["c_rating_c"] = row[RATE_C].real

            branch["f_bus"] = int(row[F_BUS].real) + 1
            branch["t_bus"] = int(row[T_BUS].real) + 1
            branch["br_status"] = int(row[BR_STATUS].real)
            branch["angmin"], branch["angmax"] = get_branch_angles(row, correct_pm_network_data)
            branch["tap"] = row[TAP].real
            branch["shift"] = math.radians(row[SHIFT].real)
            branch["construction_cost"] = row[CONSTRUCTION_COST].real
            pm["ne_branch"][str(idx)] = branch

    return pm


def add_pm_options(pm, net):
    # read values from net_options if present else use default values
    pm["pm_solver"] = net._options["pm_solver"] if "pm_solver" in net._options else "ipopt"
    pm["pm_mip_solver"] = net._options["pm_mip_solver"] if "pm_mip_solver" in net._options else "cbc"
    pm["pm_nl_solver"] = net._options["pm_nl_solver"] if "pm_nl_solver" in net._options else "ipopt"
    pm["pm_model"] = net._options["pm_model"] if "pm_model" in net._options else "DCPPowerModel"
    pm["pm_log_level"] = net._options["pm_log_level"] if "pm_log_level" in net._options else 0
    pm["pm_tol"] = net._options["pm_tol"] if "pm_tol" in net._options else 1e-8
    if "pm_time_limits" in net._options and isinstance(net._options["pm_time_limits"], dict):
        # write time limits to power models data structure
        for key, val in net._options["pm_time_limits"].items():
            pm[key] = val
    else:
        pm["pm_time_limit"], pm["pm_nl_time_limit"], pm["pm_mip_time_limit"] = \
            np.inf, np.inf, np.inf
    pm["correct_pm_network_data"] = net._options["correct_pm_network_data"]
    pm["silence"] = net._options["silence"]
    pm["ac"] = net._options["ac"]
    return pm


def build_ne_branch(net, ppc):
    # this is only used by pm tnep
    if "ne_line" in net:
        length = len(net["ne_line"])
        ppc["ne_branch"] = np.zeros(shape=(length, branch_cols + 1), dtype=np.complex128)
        ppc["ne_branch"][:, :13] = np.array([0, 0, 0, 0, 0, 250, 250, 250, 1, 0, 1, -60, 60])
        # create branch array ne_branch like the common branch array in the ppc
        net._pd2ppc_lookups["ne_branch"] = dict()
        net._pd2ppc_lookups["ne_branch"]["ne_line"] = (0, length)
        _calc_line_parameter(net, ppc, "ne_line", "ne_branch")
        ppc["ne_branch"][:, CONSTRUCTION_COST] = net["ne_line"].loc[:, "construction_cost"].values
    return ppc


def init_ne_line(net, new_line_index, construction_costs=None):
    """
    init function for new line dataframe, which specifies the possible new lines being built by power models tnep opt
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
    construction_costs = np.zeros(len(new_line_index)) if construction_costs is None else \
        construction_costs
    net["ne_line"].loc[new_line_index, "construction_cost"] = construction_costs
    # set in service, but only in ne line dataframe
    net["ne_line"].loc[new_line_index, "in_service"] = True
    # init res_ne_line to save built status afterwards
    net["res_ne_line"] = pd.DataFrame(data=0, index=new_line_index, columns=["built"], dtype=np.int64)


def add_params_to_pm(net, pm):
    # add user defined parameters to pm
    pd_idxs_br = []
    pm_idxs_br = []
    br_elms = ["line", "trafo"]
    pm["user_defined_params"] = dict()
    for elm in ["bus", "line", "gen", "load", "trafo", "sgen"]:
        param_cols = [col for col in net[elm].columns if 'pm_param' in col]
        if not param_cols:
            continue
        params = [param_col.split("/")[-1] for param_col in param_cols]
        if "side" in params:
            params.remove("side")
            params.insert(len(params), "side")
            param_cols.remove("pm_param/side")
            param_cols.insert(len(params), "pm_param/side")
            
        br_param = list(set(params) - {'side'})
        for param, param_col in zip(params, param_cols):
            pd_idxs = net[elm].index[net[elm][param_col].notna()].tolist()
            target_values = net[elm][param_col][pd_idxs].values.tolist()
            if elm in br_elms and param in br_param:
                pd_idxs_br += net[elm].index[net[elm][param_col].notna()].tolist()
                target_values = net[elm][param_col][pd_idxs_br].values.tolist()
            if elm in ["line", "trafo"]:
                start, end = net._pd2pm_lookups["branch"][elm]
                pd_pos = [net[elm].index.tolist().index(p) for p in pd_idxs_br]
                pm_idxs = [int(v) + start for v in pd_pos]
            elif elm == "sgen":
                pm_idxs = [int(v) for v in net._pd2pm_lookups[elm+"_controllable"][pd_idxs]]
                elm = "gen"
            else:
                pm_idxs = [int(v) for v in net._pd2pm_lookups[elm][pd_idxs]]
            df = pd.DataFrame(index=pm_idxs) if elm not in ["line", "trafo"] else pd.DataFrame(index=pm_idxs_br)   
            df["element_index"] = pm_idxs
            df["element_pp_index"] = pd_idxs if elm not in ["line", "trafo"] else pd_idxs_br
            df["value"] = target_values
            df["element"] = elm
            pm["user_defined_params"][param] = df.to_dict(into=OrderedDict, orient="index")

        if elm in ["line", "trafo"]:
            for bp in br_param:
                for k in pm["user_defined_params"]["side"].keys():
                    side = pm["user_defined_params"]["side"][k]["value"]
                    side_bus_f = side + "_bus"
                    if elm == "line":
                        side_bus_t = "from_bus" if side == "to" else "to_bus"
                    if elm == "trafo":
                        side_bus_t = "hv_bus" if side == "lv" else "lv_bus"
                    pd_idx = pm["user_defined_params"]["side"][k]["element_pp_index"]
                    ppcidx = net._pd2pm_lookups["branch"][elm][0]-1+pd_idx   
                    
                    if side in ["from", "hv"]:
                        ppcrow_f = 0
                        ppcrow_t = 1
                    else:
                        ppcrow_f = 1
                        ppcrow_t = 0
                        assert side in ["to", "lv"]

                    pm["user_defined_params"][bp][k]["f_bus"] = \
                        int(net._ppc_opf["branch"][ppcidx, ppcrow_f].real) + 1
                    pm["user_defined_params"][bp][k]["t_bus"] = \
                        int(net._ppc_opf["branch"][ppcidx, ppcrow_t].real) + 1
                    # pm["user_defined_params"][bp][k]["f_bus"] = \
                    #     net._pd2pm_lookups["bus"][net[elm][side_bus_f][pd_idx]]
                    # pm["user_defined_params"][bp][k]["t_bus"] = \
                    #     net._pd2pm_lookups["bus"][net[elm][side_bus_t][pd_idx]]

    # add controllable sgen:
    dic = {}
    if "user_defined_params" in pm.keys():
        for elm in ["gen", "sgen_controllable"]:
            if elm in net._pd2pm_lookups.keys():
                pm_idxs = net._pd2pm_lookups[elm] 
                for k in pm_idxs[pm_idxs!=-1]:
                    dic[str(k)] = k
        if dic != {}:
            pm["user_defined_params"]["gen_and_controllable_sgen"] = dic
    
    # add objective factors for multi optimization
    if "obj_factors" in net.keys():
        assert type(net.obj_factors) == list
        assert sum(net.obj_factors) <= 1
        dic = {}
        for i, k in enumerate(net.obj_factors):
            dic["fac_"+str(i+1)] = k        
        pm["user_defined_params"]["obj_factors"] = dic

    return pm


def add_time_series_to_pm(net, pm, from_time_step, to_time_step):
    from pandapower.control import ConstControl
    if from_time_step is None or to_time_step is None:
        raise ValueError("please define 'from_time_step' " +
                         "and 'to_time_step' to call time-series optimizaiton ")
    tp_list = list(range(from_time_step, to_time_step))
    if len(net.controller):
        load_dict, gen_dict = {}, {}
        pm["time_series"] = {"load": load_dict, "gen": gen_dict,
                             "from_time_step": from_time_step+1, 
                             "to_time_step": to_time_step+1} 
        for idx, content in net.controller.iterrows():
            if not type(content["object"]) == ConstControl:
                continue
            else:
                element = content["object"].__dict__["matching_params"]["element"]
                variable = content["object"].__dict__["matching_params"]["variable"]
                elm_idxs = content["object"].__dict__["matching_params"]["element_index"]
                df = content["object"].data_source.df
                for pd_ei in elm_idxs:
                    if element == "sgen" and "controllable" in net[element].columns and net[element].controllable[pd_ei]:
                        pm_ei = net._pd2pm_lookups[element+"_controllable"][pd_ei]
                        pm_elm = "gen"
                    else:
                        pm_ei = net._pd2pm_lookups[element][pd_ei]
                        pm_elm = "load"
                    if str(pm_ei) not in list(pm["time_series"][pm_elm].keys()):
                        pm["time_series"][pm_elm][str(pm_ei)] = {}
                    target_ts = df[pd_ei][from_time_step:to_time_step].values
                    pm["time_series"][pm_elm][str(pm_ei)][variable] = \
                       {str(tp_list[m]):n for m, n in enumerate(list(-target_ts))} if (element!="load" and pm_elm not in element) \
                            else {str(tp_list[m]):n for m, n in enumerate(list(target_ts))} 
    return pm


def allow_multi_ext_grids(net, pm, ext_grids=None):
    ext_grids = net.ext_grid.index.tolist() if ext_grids is None else ext_grids
    target_pp_buses = net.ext_grid.bus[ext_grids]
    target_pm_buses = net._pd2pm_lookups["bus"][target_pp_buses]
    for b in target_pm_buses:
        pm["bus"][str(b)]["bus_type"] = 3
    return pm
