# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pandas as pd

from pandapower.results_branch import _get_branch_results, _get_branch_results_3ph
from pandapower.results_bus import _get_bus_results, _get_bus_dc_results, _set_buses_out_of_service, \
    _get_shunt_results, _get_p_q_results, _get_bus_v_results, _get_bus_v_results_3ph, _get_p_q_results_3ph, \
    _get_bus_results_3ph, _get_bus_dc_v_results, _get_p_dc_results, _set_dc_buses_out_of_service
from pandapower.results_gen import _get_gen_results, _get_gen_results_3ph, _get_dc_slack_results

BRANCH_RESULTS_KEYS = ("branch_ikss_f", "branch_ikss_t",
                       "branch_ikss_angle_f", "branch_ikss_angle_t",
                       "branch_pkss_f", "branch_pkss_t",
                       "branch_qkss_f", "branch_qkss_t",
                       "branch_vkss_f", "branch_vkss_t",
                       "branch_vkss_angle_f", "branch_vkss_angle_t",
                       "branch_ip_f", "branch_ip_t",
                       "branch_ith_f", "branch_ith_t")

suffix_mode = {"sc": "sc", "se": "est", "pf_3ph": "3ph"}


def _extract_results(net, ppc):
    _set_buses_out_of_service(ppc)  # for NaN results in net.res_bus for inactive buses
    _set_dc_buses_out_of_service(ppc)  # for NaN results in net.res_bus_dc for inactive buses
    bus_lookup_aranged = _get_aranged_lookup(net)
    bus_dc_lookup_aranged = _get_aranged_lookup(net, "bus_dc")
    _get_bus_v_results(net, ppc)
    _get_bus_dc_v_results(net, ppc)
    bus_pq = _get_p_q_results(net, ppc, bus_lookup_aranged)
    _get_shunt_results(net, ppc, bus_lookup_aranged, bus_pq)
    _get_branch_results(net, ppc, bus_lookup_aranged, bus_pq)
    _get_gen_results(net, ppc, bus_lookup_aranged, bus_pq)
    _get_bus_results(net, ppc, bus_pq)
    bus_p_dc = _get_p_dc_results(net, ppc, bus_dc_lookup_aranged)
    _get_dc_slack_results(net, ppc, bus_dc_lookup_aranged, bus_p_dc)
    # _get_branch_dc_results(net, ppc, bus_dc_lookup_aranged, bus_p_dc) # not needed since it is calculated in _get_branch_results
    _get_bus_dc_results(net, bus_p_dc)
    _get_b2b_vsc_results(net)
    if net._options["mode"] == "opf":
        _get_costs(net, ppc)
    else:
        _remove_costs(net)


def _get_b2b_vsc_results(net):
    """
    Extract the results for the bipolar VSC from the monopolar VSC table.
    This done via the indexes, since we have created at the beginning two net.vsc entries, for every net.b2b_vsc entry.
    The idea is, to first lookup the indexes and recreate the naming scheme, then cut out the part needed form the
    net.res_vsc table and afterwards grouping and aggregating everything together.
    """
    if len(net["b2b_vsc"]) > 0:
        # create an index of the b2b_vsc's
        indices = net.b2b_vsc.index.values
        # naming scheme is b2b_0+, b2b_0-, b2b_1+, b2b_1-, ...
        naming_scheme = 'b2b_' + np.repeat(indices, 2).astype(str) + np.tile(['+', '-'], len(indices))
        vsc_idx = net.vsc[net.vsc['name'].isin(naming_scheme)].index
        res_vsc = net.res_vsc.loc[vsc_idx]

        # Add a grouping index to split rows into pairs (0,1), (2,3), etc.
        res_vsc['group'] = res_vsc.index // 2

        net.res_b2b_vsc = res_vsc.groupby('group').agg(
            p_mw=('p_mw', 'sum'),                               # p_mw gets summed since this is the AC power
            q_mvar=('q_mvar', 'sum'),                           # q_mvar also gets summed
            p_dc_mw_p=('p_dc_mw', 'first'),                     # MW of the plus DC bus
            p_dc_mw_m=('p_dc_mw', 'last'),                      # MW of the minus DC bus
            vm_internal_pu=('vm_internal_pu', 'mean'),          # The internal vm_pu is the average of both vsc
            va_internal_degree=('va_internal_degree', 'mean'),  # Same for the angle
            vm_pu=('vm_pu', 'mean'),                            # Mean of the vm_pu set point
            va_degree = ('va_degree', 'mean'),                  # Mean of the va_degree set point
            vm_internal_dc_pu_p=('vm_internal_dc_pu', 'first'), # Internal dc set point for the plus bus
            vm_internal_dc_pu_m=('vm_internal_dc_pu', 'last'),  # Same for the minus bus
            vm_dc_pu_p=('vm_dc_pu', 'first'),                   # And also for the external pu set point
            vm_dc_pu_m=('vm_dc_pu', 'last'),                    # also for the minus bus
        )

        # remove the b2b_vsc results from the res table
        net.res_vsc.drop(vsc_idx, axis=0, inplace=True)

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
    _get_bus_v_results(net, ppc, suffix="_est")
    bus_pq = np.zeros(shape=(len(net["bus"].index), 2), dtype=float)
    _get_branch_results(net, ppc, bus_lookup_aranged, bus_pq, suffix="_est")


def _get_costs(net, ppc):
    net.res_cost = ppc['obj']


def _remove_costs(net):
    if "res_cost" in net.keys():
        del net["res_cost"]


def _get_aranged_lookup(net, bus_table="bus"):
    # generate bus_lookup net -> consecutive ordering
    if len(net[bus_table]) == 0:
        return np.array([], dtype=np.int64)
    maxBus = max(net[bus_table].index.values)
    bus_lookup_aranged = -np.ones(maxBus + 1, dtype=np.int64)
    bus_lookup_aranged[net[bus_table].index.values] = np.arange(len(net[bus_table].index.values))

    return bus_lookup_aranged


def verify_results(net, mode="pf"):
    elements = get_relevant_elements(mode)
    suffix = suffix_mode.get(mode, None)
    for element in elements:
        res_element, res_empty_element = get_result_tables(element, suffix)

        index_equal = False if res_element not in net else net[element].index.equals(net[res_element].index)
        if not index_equal:
            if net["_options"]["init_results"] and element == "bus":
                # if the indices of bus and res_bus are not equal, but init_results is set, the voltage vector
                # is wrong. A UserWarning is raised in this case. For all other elements the result table is emptied.
                raise UserWarning("index of result table '{}' is not equal to the element table '{}'. The init result"
                                  " option may lead to a non-converged power flow.".format(res_element, element))
            # init result table for
            init_element(net, element)
            if element == "bus":
                net._options["init_vm_pu"] = "auto"
                net._options["init_va_degree"] = "auto"


def get_result_tables(element, suffix=None):
    res_element = "res_" + element
    res_element_with_suffix = res_element if suffix is None else res_element + "_%s" % suffix

    if suffix == suffix_mode.get("se", None):
        # State estimation used default result table
        return res_element_with_suffix, "_empty_%s" % res_element
    else:
        return res_element_with_suffix, "_empty_%s" % res_element_with_suffix


def empty_res_element(net, element, suffix=None):
    res_element, res_empty_element = get_result_tables(element, suffix)
    if res_empty_element in net:
        net[res_element] = net[res_empty_element].copy()
    else:
        net[res_element] = pd.DataFrame(columns=pd.Index([], dtype=object),
                                        index=pd.Index([], dtype=np.int64))


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
        return ["bus", "bus_dc", "line", "line_dc", "trafo", "trafo3w", "impedance", "ext_grid",
                "load", "load_dc", "motor", "sgen", "storage", "shunt", "gen", "ward",
                "xward", "dcline", "asymmetric_load", "asymmetric_sgen", "source_dc",
                "switch", "tcsc", "svc", "ssc", "vsc", "b2b_vsc"]
    elif mode == "sc":
        return ["bus", "line", "trafo", "trafo3w", "ext_grid", "gen", "sgen", "switch"]
    elif mode == "se":
        return ["bus", "line", "trafo", "trafo3w", "impedance", "switch", "shunt"]
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
    if "res_cost" in net.keys():
        del net["res_cost"]


def _ppci_bus_to_ppc(result, ppc):
    # result is the ppci (ppc without out-of-service buses)
    # buses are sorted (REF, PV, PQ, NONE) -> results are the first 3 types
    for bus_table in ("bus", "bus_dc"):
        n_buses, bus_cols = np.shape(ppc[bus_table])
        n_rows_result, bus_cols_result = np.shape(result[bus_table])
        # create matrix of proper size
        updated_bus = np.empty((n_buses, bus_cols_result))
        # fill in results (first 3 types)
        updated_bus[:n_rows_result, :] = result[bus_table]
        if n_buses > n_rows_result:
            # keep rows for busses of type NONE
            updated_bus[n_rows_result:, :bus_cols] = ppc[bus_table][n_rows_result:, :]
        ppc[bus_table] = updated_bus

    ppc['svc'][result["internal"]['svc_is'], :] = result['svc'][:, :]
    ppc['ssc'][result["internal"]['ssc_is'], :] = result['ssc'][:, :]
    ppc['vsc'][result["internal"]['vsc_is'], :] = result['vsc'][:, :]


def _ppci_branch_to_ppc(result, ppc):
    # in service branches and gens are taken from 'internal'
    branch_cols = np.shape(ppc['branch'])[1]
    ppc['branch'][result["internal"]['branch_is'], :branch_cols] = result['branch'][:, :branch_cols]

    ppc['tcsc'][result["internal"]['tcsc_is'], :] = result['tcsc'][:, :]

    ppc['branch_dc'][result["internal"]['branch_dc_is'], :] = result['branch_dc'][:, :]


def _ppci_gen_to_ppc(result, ppc):
    gen_cols = np.shape(ppc['gen'])[1]
    ppc['gen'][result["internal"]['gen_is'], :gen_cols] = result['gen'][:, :gen_cols]


def _ppci_other_to_ppc(result, ppc, mode):

    if mode != "sc" and mode != "se":
        ppc['success'] = result['success']
        ppc['et'] = result['et']

    if mode == 'opf':
        ppc['obj'] = result['f']
        ppc['internal_gencost'] = result['gencost']

    if "iterations" in result:
        ppc["iterations"] = result["iterations"]


def _ppci_internal_to_ppc(result, ppc):
    for key, value in result["internal"].items():
        # Only for sc calculation
        # if branch current matrices have been stored they need to include out of service elements
        if key in BRANCH_RESULTS_KEYS:

            # n_buses = np.shape(ppc['bus'])[0]
            n_branches = np.shape(ppc['branch'])[0]
            # n_rows_result = np.shape(result['bus'])[0]
            # update_matrix = np.empty((n_branches, n_buses)) * np.nan
            # update_matrix[result["internal"]['branch_is'], :n_rows_result] = result["internal"][key]

            # To select only required buses and pad one column of nan value for oos bus
            update_matrix = np.empty((n_branches, value.shape[1]+1)) * 0.0
            update_matrix[result["internal"]['branch_is'],
                          :value.shape[1]] = result["internal"][key]
            ppc['internal'][key] = update_matrix
            if "br_res_ks_ppci_bus" in result["internal"]:
                br_res_ks_ppci_bus = np.r_[result["internal"]["br_res_ks_ppci_bus"], [-1]]
            else:
                br_res_ks_ppci_bus = np.r_[np.arange(value.shape[1]), [-1]]
            ppc['internal'][key] = pd.DataFrame(data=update_matrix, columns=br_res_ks_ppci_bus)
        else:
            ppc["internal"][key] = value


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
    _ppci_internal_to_ppc(result, ppc)
    _ppci_other_to_ppc(result, ppc, mode)

    result = ppc
    return result
