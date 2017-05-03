# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import copy

import numpy as np

from pandapower.results_branch import _get_branch_results
from pandapower.results_bus import _get_bus_results, _get_p_q_results, _set_buses_out_of_service, \
    _get_shunt_results, _get_p_q_results_opf, _get_bus_v_results
from pandapower.results_gen import _get_gen_results


def _extract_results(net, ppc):
    _set_buses_out_of_service(ppc)
    bus_lookup_aranged = _get_aranged_lookup(net)

    _get_bus_v_results(net, ppc)
    bus_pq = _get_p_q_results(net, bus_lookup_aranged)
    _get_shunt_results(net, ppc, bus_lookup_aranged, bus_pq)
    _get_branch_results(net, ppc, bus_lookup_aranged, bus_pq)
    _get_gen_results(net, ppc, bus_lookup_aranged, bus_pq)
    _get_bus_results(net, ppc, bus_pq)


def _extract_results_opf(net, ppc):
    # get options
    bus_lookup_aranged = _get_aranged_lookup(net)

    _set_buses_out_of_service(ppc)
    bus_pq = _get_p_q_results_opf(net, ppc, bus_lookup_aranged)
    _get_shunt_results(net, ppc, bus_lookup_aranged, bus_pq)
    _get_branch_results(net, ppc, bus_lookup_aranged, bus_pq)
    _get_gen_results(net, ppc, bus_lookup_aranged, bus_pq)
    _get_bus_results(net, ppc, bus_pq)
    _get_costs(net, ppc)


def _get_costs(net, ppc):
    net.res_cost = ppc['obj']


def _get_aranged_lookup(net):
    # generate bus_lookup net -> consecutive ordering
    maxBus = max(net["bus"].index.values)
    bus_lookup_aranged = -np.ones(maxBus + 1, dtype=int)
    bus_lookup_aranged[net["bus"].index.values] = np.arange(len(net["bus"].index.values))

    return bus_lookup_aranged


def reset_results(net):
    net["res_bus"] = copy.copy(net["_empty_res_bus"])
    net["res_ext_grid"] = copy.copy(net["_empty_res_ext_grid"])
    net["res_line"] = copy.copy(net["_empty_res_line"])
    net["res_load"] = copy.copy(net["_empty_res_load"])
    net["res_sgen"] = copy.copy(net["_empty_res_sgen"])
    net["res_trafo"] = copy.copy(net["_empty_res_trafo"])
    net["res_trafo3w"] = copy.copy(net["_empty_res_trafo3w"])
    net["res_shunt"] = copy.copy(net["_empty_res_shunt"])
    net["res_impedance"] = copy.copy(net["_empty_res_impedance"])
    net["res_gen"] = copy.copy(net["_empty_res_gen"])
    net["res_ward"] = copy.copy(net["_empty_res_ward"])
    net["res_xward"] = copy.copy(net["_empty_res_xward"])
    net["res_dcline"] = copy.copy(net["_empty_res_dcline"])


def _copy_results_ppci_to_ppc(result, ppc, mode):
    '''
    result contains results for all in service elements
    ppc shall get the results for in- and out of service elements
    -> results must be copied

    ppc and ppci are structured as follows:

          [in_service elements]
    ppc = [out_of_service elements]

    result = [in_service elements]

    @author: fschaefer

    @param result:
    @param ppc:
    @return:
    '''

    # copy the results for bus, gen and branch
    # busses are sorted (REF, PV, PQ, NONE) -> results are the first 3 types
    n_busses, bus_cols = np.shape(ppc['bus'])
    n_rows_result, bus_cols_result = np.shape(result['bus'])
    # create matrix of proper size
    updated_bus = np.empty((n_busses, bus_cols_result))
    # fill in results (first 3 types)
    updated_bus[:n_rows_result, :] = result['bus']
    if n_busses > n_rows_result:
        # keep rows for busses of type NONE
        updated_bus[n_rows_result:,:bus_cols] = ppc['bus'][n_rows_result:,:]
    ppc['bus']= updated_bus

    if mode == "sc":
        ppc["bus"][:len(result['bus']), :bus_cols] = result["bus"][:len(result['bus']), :bus_cols]
    # in service branches and gens are taken from 'internal'
    branch_cols = np.shape(ppc['branch'])[1]
    ppc['branch'][result["internal"]['branch_is'], :branch_cols] = result['branch'][:, :branch_cols]

    gen_cols = np.shape(ppc['gen'])[1]
    ppc['gen'][result["internal"]['gen_is'], :gen_cols] = result['gen'][:, :gen_cols]

    ppc['internal'] = result['internal']

    if mode != "sc" and mode != "se":
        ppc['success'] = result['success']
        ppc['et'] = result['et']

    if mode == 'opf':
        ppc['obj'] = result['f']
        ppc['internal_gencost'] = result['gencost']

    result = ppc
    return result
