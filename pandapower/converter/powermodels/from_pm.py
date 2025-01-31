import math
import copy

import numpy as np

from pandapower.auxiliary import _clean_up
from pandapower.converter import logger
from pandapower.pypower.idx_brch import PF, PT, QF, QT, BR_STATUS
from pandapower.pypower.idx_bus import VA, VM, LAM_P
from pandapower.pypower.idx_gen import PG, QG
from pandapower.results import _extract_results, _copy_results_ppci_to_ppc
from pandapower.optimal_powerflow import OPFNotConverged

def read_pm_results_to_net(net, ppc, ppci, result_pm):
    """
    reads power models results from result_pm to ppc / ppci and then to pandapower net
    """
    # read power models results from result_pm to result (== ppc with results)
    net._pm_result_orig = result_pm
    result_pm = _deep_copy_pm_results(result_pm)
    result_pm = _convert_pm_units_to_pp_units(result_pm, net.sn_mva)
    net._pm_result = result_pm
    result, multinetwork = pm_results_to_ppc_results(net, ppc, ppci, result_pm)
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
        raise OPFNotConverged("PowerModels.jl OPF not converged")


def pm_results_to_ppc_results(net, ppc, ppci, result_pm):
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

    if "bus" not in sol:
        # PowerModels failed
        ppci["success"] = False
    else:
        for i, bus in sol["bus"].items():
            bus_idx = int(i) - 1
            if "vm" in bus:
                ppci["bus"][bus_idx, VM] = bus["vm"]
            if "va" in bus:
                # replace nans with 0.(in case of SOCWR model for example
                ppci["bus"][bus_idx, VA] = 0.0 if bus["va"] == None else math.degrees(bus["va"])
            if "w" in bus:
                # SOCWR model has only w instead of vm values
                ppci["bus"][bus_idx, VM] = bus["w"]
            # se comentar as próximas 2 linhas, muda o resultado
            if "lam_kcl_r" in bus:
                ppci["bus"][bus_idx, LAM_P] = bus["lam_kcl_r"]

        for i, gen in sol["gen"].items():
            gen_idx = int(i) - 1
            ppci["gen"][gen_idx, PG] = gen["pg"]
            ppci["gen"][gen_idx, QG] = gen["qg"]

        # read Q from branch results (if not DC calculation)
        if "branch" in sol:
            dc_results = sol["branch"]["1"]["qf"] is None or np.isnan(sol["branch"]["1"]["qf"])
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


def read_ots_results(net):
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
        branch_status = ppc["branch"][f:t, BR_STATUS].real

        net[res]["in_service"].values[:] = branch_status


def read_tnep_results(net):
    ne_branch = net._pm_result["solution"]["ne_branch"]
    line_idx = net["res_ne_line"].index
    for pm_branch_idx, branch_data in ne_branch.items():
        # get pandapower index from power models index
        pp_idx = line_idx[int(pm_branch_idx) - 1]
        # built is a float, which is not exactly 1.0 or 0. sometimes
        net["res_ne_line"].loc[pp_idx, "built"] = branch_data["built"] > 0.5


def _deep_copy_pm_results(result_pm):
    """Deep copy pm solution, to keep the original output of PowerModels."""
    
    pm = {key: (val if key != "solution" else copy.deepcopy(val))
          for key, val in result_pm.items()}

    return pm


def _convert_pm_units_to_pp_units(result_pm, sn_mva):

    rad2degree = lambda x: math.degrees(x) if x is not None else x
    pu2mva = lambda x: x * sn_mva
    
    sol = result_pm["solution"]

    # verifying if the solution is there
    if "bus" in sol:
        for i, bus in sol["bus"].items():
            sol["bus"][i]["va"] = rad2degree(bus["va"])
            # converting the unit of shadow prices (lagrange multiplier)
            if "lam_kcl_r" in bus:
                sol["bus"][i]["lam_kcl_r"] = -1 * bus["lam_kcl_r"] / sn_mva

        for i, gen in sol["gen"].items():
            for field in ["pg", "qg"]:
                sol["gen"][i][field] = pu2mva(gen[field])

        for element in ["branch", "dcline"]:
            if element in sol:
                for i, ele in sol[element].items():
                    for field in ["pf", "pt", "qf", "qt"]:
                        sol[element][i][field] = pu2mva(ele[field])

    result_pm["solution"] = sol

    return result_pm
