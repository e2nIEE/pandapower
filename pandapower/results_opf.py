# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

from pandapower.results import _set_buses_out_of_service, _get_shunt_results, _get_branch_results, \
                                                                _get_gen_results, _get_bus_results
from numpy import zeros, array, float, hstack
from pandapower.build_bus import _sum_by_group
import pandapower as pp
from pypower.idx_gen import PG, QG


def _extract_results_opf(net, ppc, is_elems, bus_lookup, trafo_loading, return_voltage_angles):
    eg_is = is_elems['eg']
    gen_is = is_elems['gen']
    bus_is = is_elems['bus']

    _set_buses_out_of_service(ppc)
    bus_pq = _get_p_q_results_opf(net, ppc, bus_lookup, len(eg_is) + len(gen_is))
    _get_shunt_results(net, ppc, bus_lookup, bus_pq, bus_is)
    _get_branch_results(net, ppc, bus_lookup, bus_pq, trafo_loading)
    _get_gen_results(net, ppc, is_elems, bus_lookup, bus_pq, return_voltage_angles)
    _get_bus_results(net, ppc, bus_lookup, bus_pq, return_voltage_angles)


def _get_p_q_results_opf(net, ppc, bus_lookup, gen_end):
    bus_pq = zeros(shape=(len(net["bus"].index), 2), dtype=float)
    b, p, q = array([]), array([]), array([])

    l = net["load"]
    if len(l) > 0:
        load_is = l["in_service"].values
        scaling = l["scaling"].values
        pl = l["p_kw"].values * scaling * load_is
        ql = l["q_kvar"].values * scaling * load_is
        net["res_load"]["p_kw"] = pl
        net["res_load"]["q_kvar"] = ql
        b = hstack([b, l["bus"].values])
        p = hstack([p, pl])
        q = hstack([q, ql])
        net["res_load"].index = net["load"].index

    sg = net["sgen"]
    if len(sg) > 0:
        sgen_is = sg["in_service"].values
        scaling = sg["scaling"].values
        psg = sg["p_kw"].values * scaling * sgen_is
        qsg = sg["q_kvar"].values * scaling * sgen_is
        net["res_sgen"]["p_kw"] = psg
        net["res_sgen"]["q_kvar"] = qsg
        b = hstack([b, sg["bus"].values])
        p = hstack([p, psg])
        q = hstack([q, qsg])
        net["res_sgen"].index = net["sgen"].index

        if net.sgen.controllable.any():
            net.res_sgen.p_kw[net.sgen.controllable] = - ppc["gen"][gen_end:, PG] * 1000
            net.res_sgen.q_kvar[net.sgen.controllable] = - ppc["gen"][gen_end:, QG] * 1000

    b_pp, vp, vq = _sum_by_group(b.astype(int), p, q)
    b_ppc = pp.get_indices(b_pp, bus_lookup, fused_indices=False)
    bus_pq[b_ppc, 0] = vp
    bus_pq[b_ppc, 1] = vq
    return bus_pq
