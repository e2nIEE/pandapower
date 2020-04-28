# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pandas as pd


from pandapower.results_branch import _get_branch_results, _get_branch_results_3ph
from pandapower.results_bus import _get_bus_results, _set_buses_out_of_service, \
    _get_shunt_results, _get_p_q_results, _get_bus_v_results, _get_bus_v_results_3ph, _get_p_q_results_3ph, \
    _get_bus_results_3ph
from pandapower.results_gen import _get_gen_results, _get_gen_results_3ph

suffix_mode = {"sc": "sc", "se": "se", "pf_3ph": "3ph"}

def _extract_results(net, ppc):
    _set_buses_out_of_service(ppc)
    bus_lookup_aranged = _get_aranged_lookup(net)
    _get_bus_v_results(net, ppc)
    bus_pq = _get_p_q_results(net, ppc, bus_lookup_aranged)
    _get_shunt_results(net, ppc, bus_lookup_aranged, bus_pq)
    _get_branch_results(net, ppc, bus_lookup_aranged, bus_pq)
    _get_gen_results(net, ppc, bus_lookup_aranged, bus_pq)
    _get_bus_results(net, ppc, bus_pq)
    if net._options["mode"] == "opf":
        _get_costs(net, ppc)


def _extract_results_3ph(net, ppc0, ppc1, ppc2):
    # reset_results(net, False)
    _set_buses_out_of_service(ppc0)
    _set_buses_out_of_service(ppc1)
    _set_buses_out_of_service(ppc2)
    bus_lookup_aranged = _get_aranged_lookup(net)

    _get_bus_v_results_3ph(net, ppc0, ppc1, ppc2)
    bus_pq = _get_p_q_results_3ph(net, bus_lookup_aranged)
    # _get_shunt_results(net, ppc, bus_lookup_aranged, bus_pq)
    _get_branch_results_3ph(net, ppc0, ppc1, ppc2, bus_lookup_aranged, bus_pq)
    _get_gen_results_3ph(net, ppc0, ppc1, ppc2, bus_lookup_aranged, bus_pq)
    _get_bus_results_3ph(net, bus_pq)
    

def _extract_results_se(net, ppc):
    _set_buses_out_of_service(ppc)
    bus_lookup_aranged = _get_aranged_lookup(net)
    _get_bus_v_results(net, ppc)
    bus_pq = np.zeros(shape=(len(net["bus"].index), 2), dtype=np.float)
    _get_branch_results(net, ppc, bus_lookup_aranged, bus_pq)


def _get_costs(net, ppc):
    net.res_cost = ppc['obj']


def _get_aranged_lookup(net):
    # generate bus_lookup net -> consecutive ordering
    maxBus = max(net["bus"].index.values)
    bus_lookup_aranged = -np.ones(maxBus + 1, dtype=int)
    bus_lookup_aranged[net["bus"].index.values] = np.arange(len(net["bus"].index.values))

    return bus_lookup_aranged


def verify_results(net, mode="pf"):
    elements = get_relevant_elements(mode)
    suffix = suffix_mode.get(mode, None)
    for element in elements:
        res_element, res_empty_element = get_result_tables(element, suffix)
        if len(net[element]) != len(net[res_element]):
            init_element(net, element)
            if element == "bus":
                net._options["init_vm_pu"] = "auto"
                net._options["init_va_degree"] = "auto"


def get_result_tables(element, suffix=None):
    res_element = "res_" + element
    if suffix is not None:
        res_element += "_%s"%suffix
    return res_element, "_empty_%s"%res_element


def empty_res_element(net, element, suffix=None):
    res_element, res_empty_element = get_result_tables(element, suffix)
    if res_empty_element in net:
        net[res_element] = net[res_empty_element].copy()
    else:
        net[res_element] = pd.DataFrame()


def init_element(net, element, suffix=None):
    res_element, res_empty_element = get_result_tables(element, suffix)
    index = net[element].index
    if len(index):
        # init empty dataframe
        if res_empty_element in net:
            columns = net[res_empty_element].columns
            net[res_element] = pd.DataFrame(np.nan, index=index,
                                            columns=columns, dtype='float')
        else:
            net[res_element] = pd.DataFrame(index=index, dtype='float')            
    else:
        empty_res_element(net, element, suffix)

def get_relevant_elements(mode="pf"):
    if mode == "pf" or mode == "opf":
        return ["bus", "line", "trafo", "trafo3w", "impedance", "ext_grid",
                "load", "sgen", "storage", "shunt", "gen", "ward", "xward",
                "dcline"]
    elif mode == "sc":
        return ["bus", "line", "trafo", "trafo3w", "ext_grid", "gen", "sgen"]
    elif mode == "se":
        return ["bus", "line", "trafo", "trafo3w", "impedance", "ext_grid",
                "load", "sgen", "storage", "shunt", "gen", "ward", "xward",
                "dcline", "measurement"]        
    elif mode == "pf_3ph":
        return ["bus", "line", "trafo", "ext_grid", "shunt",
                "load", "sgen", "storage", "asymmetric_load", "asymmetric_sgen"]  
        
def init_results(net, mode="pf"):
    elements = get_relevant_elements(mode)
    suffix = suffix_mode.get(mode, None)
    for element in elements:
        init_element(net, element, suffix)
        
def reset_results(net, mode="pf"):
    elements = get_relevant_elements(mode)
    suffix = suffix_mode.get(mode, None)
    for element in elements:
        empty_res_element(net, element, suffix)


def _ppci_bus_to_ppc(result, ppc):
    # result is the ppci (ppc without out of service buses)
    # busses are sorted (REF, PV, PQ, NONE) -> results are the first 3 types
    n_buses, bus_cols = np.shape(ppc['bus'])
    n_rows_result, bus_cols_result = np.shape(result['bus'])
    # create matrix of proper size
    updated_bus = np.empty((n_buses, bus_cols_result))
    # fill in results (first 3 types)
    updated_bus[:n_rows_result, :] = result['bus']
    if n_buses > n_rows_result:
        # keep rows for busses of type NONE
        updated_bus[n_rows_result:, :bus_cols] = ppc['bus'][n_rows_result:, :]
    ppc['bus'] = updated_bus


def _ppci_branch_to_ppc(result, ppc):
    # in service branches and gens are taken from 'internal'
    branch_cols = np.shape(ppc['branch'])[1]
    ppc['branch'][result["internal"]['branch_is'], :branch_cols] = result['branch'][:, :branch_cols]


def _ppci_gen_to_ppc(result, ppc):
    gen_cols = np.shape(ppc['gen'])[1]
    ppc['gen'][result["internal"]['gen_is'], :gen_cols] = result['gen'][:, :gen_cols]


def _ppci_other_to_ppc(result, ppc, mode):
    ppc['internal'] = result['internal']

    if mode != "sc" and mode != "se":
        ppc['success'] = result['success']
        ppc['et'] = result['et']

    if mode == 'opf':
        ppc['obj'] = result['f']
        ppc['internal_gencost'] = result['gencost']

    if "iterations" in result:
        ppc["iterations"] = result["iterations"]


def _copy_results_ppci_to_ppc(result, ppc, mode):
    """
    result contains results for all in service elements
    ppc gets the results for in- and out of service elements
    -> results must be copied

    ppc and ppci are structured as follows:
          [in_service elements]
    ppc = [out_of_service elements]
    result = [in_service elements]

    Parameters
    ----------
    result - ppci with results
    ppc - ppc without results
    mode - "pf","opf", "sc"...

    Returns
    -------
    ppc with results
    """

    # copy the results for bus, gen and branch and some additional values like "success"
    _ppci_bus_to_ppc(result, ppc)
    _ppci_branch_to_ppc(result, ppc)
    _ppci_gen_to_ppc(result, ppc)
    _ppci_other_to_ppc(result, ppc, mode)

    result = ppc
    return result
