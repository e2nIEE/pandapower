# -*- coding: utf-8 -*-

# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
from pandapower.pf.bustypes import bustypes

from pandapower.idx_bus import VM, VA, GS, BS
from pandapower.idx_gen import PG, QG

from pandapower.auxiliary import _sum_by_group, sequence_to_phase, _sum_by_group_nvals, combine_X012, \
    I_from_SV, S_from_VI, I0_from_V012, I1_from_V012, I2_from_V012
from pandapower.pf.makeYbus import makeYbus


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


def _get_gen_results_3ph(net, ppc0, ppc1, ppc2, bus_lookup_aranged, pq_bus):
    ac = net["_options"]["ac"]

    eg_end = len(net['ext_grid'])
    gen_end = eg_end + len(net['gen'])

    b, pA, qA, pB, qB, pC, qC = _get_ext_grid_results_3ph(net, ppc0, ppc1, ppc2)

    # get results for gens
    if gen_end > eg_end:
        b, pA, qA, pB, qB, pC, qC = _get_pp_gen_results_3ph(net, ppc0, ppc1, ppc2, b, pA, qA, pB, qB, pC, qC)

    if len(net.dcline) > 0:
        _get_dcline_results(net)
        b = np.hstack([b, net.dcline[["from_bus", "to_bus"]].values.flatten()])
        pDC = -net.res_dcline[["p_from_kw", "p_to_kw"]].values.flatten() / 3
        pA = np.hstack([pA, pDC])
        pB = np.hstack([pB, pDC])
        pC = np.hstack([pC, pDC])
        qDC = -net.res_dcline[["q_from_kvar", "q_to_kvar"]].values.flatten() / 3
        qA = np.hstack([qA, qDC])
        qB = np.hstack([qB, qDC])
        qC = np.hstack([qC, qDC])

    if not ac:
        qA, qB, qC = np.copy((np.zeros(len(pA)),)*3)

    b_pp, pA_sum, qA_sum, pB_sum, qB_sum, pC_sum, qC_sum = _sum_by_group_nvals(b.astype(int), pA, qA, pB, qB, pC, qC)
    b_ppc = bus_lookup_aranged[b_pp]
    pq_bus[b_ppc, 0] += pA_sum
    pq_bus[b_ppc, 1] += qA_sum
    pq_bus[b_ppc, 2] += pB_sum
    pq_bus[b_ppc, 3] += qB_sum
    pq_bus[b_ppc, 4] += pC_sum
    pq_bus[b_ppc, 5] += qC_sum


def _get_ext_grid_results(net, ppc):
    ac = net["_options"]["ac"]

    # get results for external grids
    eg_is_mask = net["_is_elements"]['ext_grid']
    ext_grid_lookup = net["_pd2ppc_lookups"]["ext_grid"]

    n_res_eg = len(net['ext_grid'])
    # indices of in service gens in the ppc
    eg_is_idx = net["ext_grid"].index.values[eg_is_mask]
    gen_idx_ppc = ext_grid_lookup[eg_is_idx]

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


def _get_ext_grid_results_3ph(net, ppc0, ppc1, ppc2):
    ac = net["_options"]["ac"]

    # get results for external grids
    eg_is_mask = net["_is_elements"]['ext_grid']
    ext_grid_lookup = net["_pd2ppc_lookups"]["ext_grid"]

    n_res_eg = len(net['ext_grid'])
    # indices of in service gens in the ppc
    eg_is_idx = net["ext_grid"].index.values[eg_is_mask]
    gen_idx_ppc = ext_grid_lookup[eg_is_idx]

    # read results from ppc for these buses
    V012 = np.matrix(np.zeros((3, n_res_eg), dtype=complex))
    V012[:, gen_idx_ppc] = np.matrix([ppc["bus"][gen_idx_ppc, VM]
                                      * np.exp(1j * np.deg2rad(ppc["bus"][gen_idx_ppc, VA]))
                                      for ppc in [ppc0, ppc1, ppc2]])

    S012 = np.matrix(np.zeros((3, n_res_eg), dtype=complex))
    S012[:, gen_idx_ppc] = np.matrix([-(ppc["gen"][gen_idx_ppc, PG] + 1j * ppc["gen"][gen_idx_ppc, QG]) for ppc in [ppc0, ppc1, ppc2]])
    I012 = np.matrix(np.zeros((3, n_res_eg), dtype=complex))
    I012[:, gen_idx_ppc] = I_from_SV(S012[:, gen_idx_ppc], V012[:, gen_idx_ppc])

    Vabc = sequence_to_phase(V012[:, gen_idx_ppc])
    Iabc = sequence_to_phase(I012[:, gen_idx_ppc])
    Sabc = S_from_VI(Vabc, Iabc) * 1e3
    pA, pB, pC = map(lambda x: x.A1, np.real(Sabc))
    qA, qB, qC = map(lambda x: x.A1, np.imag(Sabc))

    # store result in net['res']
    net["res_ext_grid_3ph"]["p_kw_A"] = pA
    net["res_ext_grid_3ph"]["p_kw_B"] = pB
    net["res_ext_grid_3ph"]["p_kw_C"] = pC
    net["res_ext_grid"]["q_kvar_A"] = qA
    net["res_ext_grid"]["q_kvar_B"] = qB
    net["res_ext_grid"]["q_kvar_C"] = qC

    # get bus values for pq_bus
    b = net['ext_grid'].bus.values
    # copy index for results
    net["res_ext_grid_3ph"].index = net['ext_grid'].index

    return b, pA, qA, pB, qB, pC, qC


def _get_p_q_gen_resuts(net, ppc):
    _is_elements = net["_is_elements"]
    gen_is_mask = _is_elements['gen']
    gen_lookup = net["_pd2ppc_lookups"]["gen"]
    gen_is_idx = net["gen"].index[gen_is_mask]
    # indices of in service gens in the ppc
    if np.any(_is_elements["gen"]):
        gen_idx_ppc = gen_lookup[gen_is_idx]
    else:
        gen_idx_ppc = []

    # read results from ppc for these buses
    n_res_gen = len(net['gen'])
    p_gen = np.zeros(n_res_gen)
    p_gen[gen_is_mask] = -ppc["gen"][gen_idx_ppc, PG] * 1e3
    q_gen = None
    if net["_options"]["ac"]:
        q_gen = np.zeros(n_res_gen)
        q_gen[gen_is_mask] = -ppc["gen"][gen_idx_ppc, QG] * 1e3
        net["res_gen"]["q_kvar"] = q_gen

    net["res_gen"]["p_kw"] = p_gen
    return p_gen, q_gen


def _get_p_q_gen_results_3ph(net, ppc0, ppc1, ppc2):
    _is_elements = net["_is_elements"]
    ac = net["_options"]["ac"]
    gen_is_mask = _is_elements['gen']
    gen_lookup = net["_pd2ppc_lookups"]["gen"]
    gen_is_idx = net["gen"].index[gen_is_mask]
    # indices of in service gens in the ppc
    if np.any(_is_elements["gen"]):
        gen_idx_ppc = gen_lookup[gen_is_idx]
    else:
        gen_idx_ppc = []

    # read results from ppc for these buses
    n_res_gen = len(net['gen'])

    V012 = np.matrix(np.zeros((3, n_res_gen), dtype=complex))
    V012[:, gen_idx_ppc] = np.matrix([ppc["bus"][gen_idx_ppc, VM]
                                      * np.exp(1j * np.deg2rad(ppc["bus"][gen_idx_ppc, VA]))
                                      for ppc in [ppc0, ppc1, ppc2]])

    S012 = np.matrix(np.zeros((3, n_res_gen), dtype=complex))
    S012[:, gen_idx_ppc] = np.matrix(
        [-(ppc["gen"][gen_idx_ppc, PG] + 1j * ppc["gen"][gen_idx_ppc, QG]) for ppc in [ppc0, ppc1, ppc2]])
    I012 = np.matrix(np.zeros((3, n_res_gen), dtype=complex))
    I012[:, gen_idx_ppc] = I_from_SV(S012[:, gen_idx_ppc], V012[:, gen_idx_ppc])

    Vabc = sequence_to_phase(V012)
    Iabc = sequence_to_phase(I012)
    Sabc = S_from_VI(Vabc, Iabc) * 1e3
    pA, pB, pC = map(lambda x: x.A1, np.real(Sabc))
    qA, qB, qC = map(lambda x: x.A1, np.imag(Sabc))

    net["res_gen_3ph"]["pA_kw"] = pA
    net["res_gen_3ph"]["pB_kw"] = pB
    net["res_gen_3ph"]["pC_kw"] = pC
    net["res_gen_3ph"]["qA_kvar"] = qA
    net["res_gen_3ph"]["qB_kvar"] = qB
    net["res_gen_3ph"]["qC_kvar"] = qC

    return pA, qA, pB, qB, pC, qC


def _get_v_gen_resuts(net, ppc):
    # lookups for ppc
    bus_lookup = net["_pd2ppc_lookups"]["bus"]

    # in service gens
    gen_is_mask = net["_is_elements"]['gen']
    bus_idx_ppc = bus_lookup[net["gen"]["bus"].values[gen_is_mask]]

    n_res_gen = len(net['gen'])

    # voltage magnitudes
    v_pu = np.zeros(n_res_gen)
    v_pu[gen_is_mask] = ppc["bus"][bus_idx_ppc][:, VM]

    # voltage angles
    v_a = np.zeros(n_res_gen)
    v_a[gen_is_mask] = ppc["bus"][bus_idx_ppc][:, VA]

    net["res_gen"]["vm_pu"] = v_pu
    net["res_gen"]["va_degree"] = v_a
    return v_pu, v_a


def _get_v_gen_results_3ph(net, ppc0, ppc1, ppc2):
    # lookups for ppc
    bus_lookup = net["_pd2ppc_lookups"]["bus"]

    # in service gens
    gen_is_mask = net["_is_elements"]['gen']
    bus_idx_ppc = bus_lookup[net["gen"]["bus"].values[gen_is_mask]]

    n_res_gen = len(net['gen'])

    V012 = np.matrix(np.zeros((3, n_res_gen), dtype=complex))
    V012[:, gen_is_mask] = np.matrix([ppc["bus"][gen_is_mask, VM]
                                      * np.exp(1j * np.deg2rad(ppc["bus"][gen_is_mask, VA]))
                                      for ppc in [ppc0, ppc1, ppc2]])
    VABC = sequence_to_phase(V012)

    # voltage magnitudes
    vA_pu, vB_pu, vC_pu = np.copy((np.zeros(n_res_gen),) * 3)
    vA_pu[gen_is_mask] = np.abs(VABC[0, :])
    vB_pu[gen_is_mask] = np.abs(VABC[1, :])
    vC_pu[gen_is_mask] = np.abs(VABC[2, :])

    # voltage angles
    vA_a, vB_a, vC_a = np.copy((np.zeros(n_res_gen),) * 3)
    vA_a[gen_is_mask] = np.rad2deg(np.angle(VABC[0, :]))
    vB_a[gen_is_mask] = np.rad2deg(np.angle(VABC[1, :]))
    vC_a[gen_is_mask] = np.rad2deg(np.angle(VABC[2, :]))

    net["res_gen_3ph"]["vmA_pu"] = vA_pu
    net["res_gen_3ph"]["vmB_pu"] = vB_pu
    net["res_gen_3ph"]["vmC_pu"] = vC_pu
    net["res_gen_3ph"]["vaA_degree"] = vA_a
    net["res_gen_3ph"]["vaB_degree"] = vB_a
    net["res_gen_3ph"]["vaC_degree"] = vC_a
    return vA_pu, vA_a, vB_pu, vB_a, vC_pu, vC_a


def _get_pp_gen_results(net, ppc, b, p, q):
    p_gen, q_gen = _get_p_q_gen_resuts(net, ppc)
    _get_v_gen_resuts(net, ppc)

    net["res_gen"].index = net['gen'].index
    b = np.hstack([b, net['gen'].bus.values])

    p = np.hstack([p, p_gen])
    if net["_options"]["ac"]:
        q = np.hstack([q, q_gen])

    return b, p, q


def _get_pp_gen_results_3ph(net, ppc0, ppc1, ppc2, b, pA, qA, pB, qB, pC, qC):
    pA_gen, qA_gen, pB_gen, qB_gen, pC_gen, qC_gen = _get_p_q_gen_results_3ph(net, ppc0, ppc1, ppc2)
    _get_v_gen_results_3ph(net, ppc0, ppc1, ppc2)

    ac = net["_options"]["ac"]

    net["res_gen_3ph"].index = net['gen'].index
    b = np.hstack([b, net['gen'].bus.values])

    pA = np.hstack([pA, pA_gen])
    pB = np.hstack([pB, pB_gen])
    pC = np.hstack([pC, pC_gen])
    if ac:
        qA = np.hstack([qA, qA_gen])
        qB = np.hstack([qB, qB_gen])
        qC = np.hstack([qC, qC_gen])

    return b, pA, qA, pB, qB, pC, qC


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
