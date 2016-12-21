# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

from pandapower.results import _set_buses_out_of_service, _get_shunt_results, _get_branch_results, \
                                                                _get_gen_results, _get_bus_results
from numpy import zeros, array, float, hstack, invert, ones, arange, searchsorted
from pandapower.build_bus import _sum_by_group
from pypower.idx_gen import PG, QG, GEN_BUS


def _extract_results_opf(net, ppc, is_elems, bus_lookup, trafo_loading, return_voltage_angles,
                         ac):
    eg_is = is_elems['ext_grid']
    gen_is = is_elems['gen']
    bus_is = is_elems['bus']

    # generate bus_lookup net -> consecutive ordering
    maxBus = max(net["bus"].index.values)
    bus_lookup_aranged = -ones(maxBus + 1, dtype=int)
    bus_lookup_aranged[net["bus"].index.values] = arange(len(net["bus"].index.values))

    _set_buses_out_of_service(ppc)
    len_gen = len(eg_is) + len(gen_is)
    bus_pq = _get_p_q_results_opf(net, ppc, is_elems, bus_lookup, bus_lookup_aranged, len_gen)
    _get_shunt_results(net, ppc, bus_lookup, bus_lookup_aranged, bus_pq, bus_is, ac)
    _get_branch_results(net, ppc, bus_lookup_aranged, bus_pq, trafo_loading, ac)
    _get_gen_results(net, ppc, is_elems, bus_lookup, bus_lookup_aranged, bus_pq, 
                     return_voltage_angles, ac)
    _get_bus_results(net, ppc, bus_lookup, bus_pq, return_voltage_angles, ac)


def _get_p_q_results_opf(net, ppc, is_elems, bus_lookup, bus_lookup_aranged, gen_end):
    bus_pq = zeros(shape=(len(net["bus"].index), 2), dtype=float)
    b, p, q = array([]), array([]), array([])

    l = net["load"]
    if len(l) > 0:
        load_is = is_elems["load"]
        scaling = l["scaling"].values
        pl = l["p_kw"].values * scaling * load_is
        net["res_load"]["p_kw"] = pl
        p = hstack([p, pl])
        # q results
        ql = l["q_kvar"].values * scaling * load_is
        net["res_load"]["q_kvar"] = ql
        q = hstack([q, ql])

        b = hstack([b, l["bus"].values])
        net["res_load"].index = net["load"].index

    sg = net["sgen"]
    if len(sg) > 0:
        sgen_is = is_elems["sgen"]
        sgen_ctrl = sg["controllable"].values
        scaling = sg["scaling"].values
        psg = sg["p_kw"].values * scaling * sgen_is * invert(sgen_ctrl)
        qsg = sg["q_kvar"].values * scaling * sgen_is * invert(sgen_ctrl)
        # get gen index in ppc
        gidx_ppc = searchsorted(ppc['gen'][:, GEN_BUS], bus_lookup[net['sgen'][sgen_is].bus.values])
        psg[sgen_is & sgen_ctrl] = - ppc["gen"][gidx_ppc, PG] * 1000
        qsg[sgen_is & sgen_ctrl] = - ppc["gen"][gidx_ppc, QG] * 1000

        net["res_sgen"]["p_kw"] = psg
        net["res_sgen"]["q_kvar"] = qsg
        q = hstack([q, qsg])
        p = hstack([p, psg])
        b = hstack([b, sg["bus"].values])
        net["res_sgen"].index = net["sgen"].index

    b_pp, vp, vq = _sum_by_group(b.astype(int), p, q)
    b_ppc = bus_lookup_aranged[b_pp]
    bus_pq[b_ppc, 0] = vp
    bus_pq[b_ppc, 1] = vq
    return bus_pq