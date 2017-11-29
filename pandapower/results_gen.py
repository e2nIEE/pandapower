# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from pandapower.idx_bus import VM, VA
from pandapower.idx_gen import PG, QG

from pandapower.auxiliary import _sum_by_group


def _get_gen_results(net, ppc, bus_lookup_aranged, pq_bus):
    ac = net["_options"]["ac"]

    eg_end = len(net['ext_grid'])
    gen_end = eg_end + len(net['gen'])

    b, p, q = _get_ext_grid_results(net, ppc)

    # get results for gens
    if gen_end > eg_end:
        b, p, q = _get_pp_gen_results(net, ppc, b, p, q)

    if len(net.dcline) > 0:
        _get_dcline_results(net)
        b = np.hstack([b, net.dcline[["from_bus", "to_bus"]].values.flatten()])
        p = np.hstack([p, -net.res_dcline[["p_from_kw", "p_to_kw"]].values.flatten()])
        q = np.hstack([q, -net.res_dcline[["q_from_kvar", "q_to_kvar"]].values.flatten()])

    if not ac:
        q = np.zeros(len(p))
    b_sum, p_sum, q_sum = _sum_by_group(b, p, q)
    b = bus_lookup_aranged[b_sum]
    pq_bus[b, 0] += p_sum
    pq_bus[b, 1] += q_sum


def _get_ext_grid_results(net, ppc):
    ac = net["_options"]["ac"]

    # get results for external grids
    eg_is_mask = net["_is_elements"]['ext_grid']
    ext_grid_lookup = net["_pd2ppc_lookups"]["ext_grid"]

    n_res_eg = len(net['ext_grid'])
    # indices of in service gens in the ppc
    eg_is_idx = net["ext_grid"].index.values[eg_is_mask]
    gen_idx_ppc = ext_grid_lookup[eg_is_idx]
#    # mask for indices of in service gens in net['res_gen']
#    idx_eg = np.in1d(net['ext_grid'].index.values, eg_is.index.values)
    # read results from ppc for these buses
    p = np.zeros(n_res_eg)
    q = np.zeros(n_res_eg)
    p[eg_is_mask] = -ppc["gen"][gen_idx_ppc, PG] * 1e3
    # store result in net['res']
    net["res_ext_grid"]["p_kw"] = p

    # if ac PF q results are also available
    if ac:
        q[eg_is_mask] = -ppc["gen"][gen_idx_ppc, QG] * 1e3
        net["res_ext_grid"]["q_kvar"] = q

    # get bus values for pq_bus
    b = net['ext_grid'].bus.values
    # copy index for results
    net["res_ext_grid"].index = net['ext_grid'].index

    return b, p, q


def _get_pp_gen_results(net, ppc, b, p, q):
    ac = net["_options"]["ac"]

    _is_elements = net["_is_elements"]

    gen_is_mask = _is_elements['gen']
    gen_lookup = net["_pd2ppc_lookups"]["gen"]
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    # bus index of in service gens
    n_res_gen = len(net['gen'])

    b = np.hstack([b, net['gen'].bus.values])

    gen_is_idx = net["gen"].index[gen_is_mask]
    # indices of in service gens in the ppc
    if np.any(_is_elements["gen"]):
        gen_idx_ppc = gen_lookup[gen_is_idx]
    else:
        gen_idx_ppc = []

    # read results from ppc for these buses
    p_gen = np.zeros(n_res_gen)
    p_gen[gen_is_mask] = -ppc["gen"][gen_idx_ppc, PG] * 1e3
    if ac:
        q_gen = np.zeros(n_res_gen)
        q_gen[gen_is_mask] = -ppc["gen"][gen_idx_ppc, QG] * 1e3

    bus_idx_ppc = bus_lookup[net["gen"]["bus"].values[gen_is_mask]]

    v_pu = np.zeros(n_res_gen)
    v_pu[gen_is_mask] = ppc["bus"][bus_idx_ppc][:, VM]
    net["res_gen"]["vm_pu"] = v_pu

    # voltage angles
    v_a = np.zeros(n_res_gen)
    v_a[gen_is_mask] = ppc["bus"][bus_idx_ppc][:, VA]
    net["res_gen"]["va_degree"] = v_a

    net["res_gen"].index = net['gen'].index

    # store result in net['res']
    p = np.hstack([p, p_gen])
    net["res_gen"]["p_kw"] = p_gen
    if ac:
        q = np.hstack([q, q_gen])
        net["res_gen"]["q_kvar"] = q_gen

    return b, p, q


def _get_dcline_results(net):
    dc_gens = net.gen.index[(len(net.gen) - len(net.dcline) * 2):]
    from_gens = net.res_gen.loc[dc_gens[1::2]]
    to_gens = net.res_gen.loc[dc_gens[::2]]

    net.res_dcline.p_from_kw = from_gens.p_kw.values
    net.res_dcline.p_to_kw = to_gens.p_kw.values
    net.res_dcline.pl_kw = from_gens.p_kw.values + to_gens.p_kw.values

    net.res_dcline.q_from_kvar = from_gens.q_kvar.values
    net.res_dcline.q_to_kvar = to_gens.q_kvar.values

    net.res_dcline.vm_from_pu = from_gens.vm_pu.values
    net.res_dcline.vm_to_pu = to_gens.vm_pu.values
    net.res_dcline.va_from_degree = from_gens.va_degree.values
    net.res_dcline.va_to_degree = to_gens.va_degree.values

    net.res_dcline.index = net.dcline.index
