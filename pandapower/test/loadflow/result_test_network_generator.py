# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import pandapower as pp
from pandapower.test.toolbox import add_grid_connection, create_test_line
from pandapower.toolbox import nets_equal


def result_test_network_generator(sn_kva=1e3, skip_test_impedance=False):
    """ This is a generator for the result_test_network
        It is structured like this so it can be tested for consistency at
        different stages of adding elements
    """
    net = pp.create_empty_network(sn_kva=sn_kva)
    yield add_test_line(net)
    yield add_test_load_sgen(net)
    yield add_test_load_sgen_split(net)
    yield add_test_ext_grid(net)
    yield add_test_trafo(net)
    yield add_test_single_load_single_eg(net)
    yield add_test_ward(net)
    yield add_test_ward_split(net)
    yield add_test_xward(net)
    yield add_test_xward_combination(net)
    yield add_test_gen(net)
    yield add_test_ext_grid_gen_switch(net)
    yield add_test_enforce_qlims(net)
    yield add_test_trafo3w(net)
    if not skip_test_impedance:
        yield add_test_impedance(net)
    yield add_test_bus_bus_switch(net)
    yield add_test_oos_bus_with_is_element(net)
    yield add_test_shunt(net)
    yield add_test_shunt_split(net)
    yield add_test_two_open_switches_on_deactive_line(net)


def result_test_network_generator_dcpp(sn_kva=1e3):
    """ This is a generator for the result_test_network
        It is structured like this so it can be tested for consistency at
        different stages of adding elements
    """
    # ToDo: Uncommented tests fail in rundcpp -> Check why and correct it

    net = pp.create_empty_network(sn_kva=sn_kva)
    yield add_test_line(net)
    yield add_test_load_sgen(net)
    yield add_test_load_sgen_split(net)
    # yield add_test_ext_grid(net)
    # yield add_test_trafo(net)
    yield add_test_single_load_single_eg(net)
    yield add_test_ward(net)
    yield add_test_ward_split(net)
    yield add_test_xward(net)
    yield add_test_xward_combination(net)
    # yield add_test_gen(net)
    # yield add_test_ext_grid_gen_switch(net)
    # yield add_test_enforce_qlims(net)
    # yield add_test_trafo3w(net)
    # yield add_test_impedance(net)
    yield add_test_bus_bus_switch(net)
    # yield add_test_oos_bus_with_is_element(net)
    yield add_test_shunt(net)
    yield add_test_shunt_split(net)
    # yield add_test_two_open_switches_on_deactive_line(net)


def add_test_line(net):
    b1, b2, l1 = add_grid_connection(net, zone="test_line")
    net.line.parallel.at[l1] = 2
    pp.create_load(net, b2, p_kw=1200, q_kvar=1100)
    l2 = create_test_line(net, b1, b2)
    pp.create_switch(net, b2, l2, et="l", closed=False)
    create_test_line(net, b1, b2, in_service=False)
    net.last_added_case = "test_line"
    return net


def add_test_ext_grid(net):
    b1, b2, ln = add_grid_connection(net, zone="test_ext_grid")
    b3 = pp.create_bus(net, vn_kv=20., zone="test_ext_grid")
    create_test_line(net, b2, b3)
    pp.create_ext_grid(net, b3, vm_pu=1.02, va_degree=3.)
    return net


def add_test_ext_grid_gen_switch(net):
    zone = "test_ext_grid_gen_switch"
    vn_kv = 20.
    b1 = pp.create_bus(net, vn_kv=vn_kv, zone=zone)
    b2 = pp.create_bus(net, vn_kv=vn_kv, zone=zone)
    b3 = pp.create_bus(net, vn_kv=vn_kv, zone=zone)
    b4 = pp.create_bus(net, vn_kv=vn_kv, zone=zone)
    b5 = pp.create_bus(net, vn_kv=vn_kv, zone=zone)

    pp.create_switch(net, bus=b1, element=b2, et="b")
    create_test_line(net, b2, b3)
    create_test_line(net, b3, b4)
    pp.create_switch(net, bus=b4, element=b5, et="b")

    pp.create_ext_grid(net, bus=b1, vm_pu=1.01)
    pp.create_gen(net, bus=b5, vm_pu=1.015, p_kw=-300)
    return net


def add_test_load_sgen(net):
    b1, b2, ln = add_grid_connection(net, zone="test_load_sgen")
    pl = 1200
    ql = 1100
    ps = -500
    qs = 100
    # load and sgen at one bus
    pp.create_load(net, b2, p_kw=pl, q_kvar=ql)
    pp.create_sgen(net, b2, p_kw=ps, q_kvar=qs)
    # adding out of serivce loads and sgens should not change the result
    pp.create_load(net, b2, p_kw=pl, q_kvar=ql, in_service=False,
                   index=pp.get_free_id(net.load) + 1)
    pp.create_sgen(net, b2, p_kw=ps, q_kvar=qs, in_service=False,
                   index=pp.get_free_id(net.sgen) + 1)
    net.last_added_case = "test_load_sgen"
    return net


def add_test_load_sgen_split(net):
    b1, b2, ln = add_grid_connection(net, zone="test_load_sgen_split")
    nr = 2
    pl = 1200
    ql = 1100
    ps = -500
    qs = 100
    for _ in list(range(nr)):
        pp.create_load(net, b2, p_kw=pl, q_kvar=ql, scaling=1. / nr)
        pp.create_sgen(net, b2, p_kw=ps, q_kvar=qs, scaling=1. / nr)
    net.last_added_case = "test_load_sgen_split"
    return net


def add_test_trafo(net):
    b1, b2, ln = add_grid_connection(net, zone="test_trafo")
    b3 = pp.create_bus(net, vn_kv=0.4, zone="test_trafo")
    pp.create_transformer_from_parameters(net, b2, b3, vsc_percent=5., vscr_percent=2.,
                                          i0_percent=.4, pfe_kw=2, sn_kva=400, vn_hv_kv=22,
                                          vn_lv_kv=0.42, tp_max=10, tp_mid=5, tp_min=0,
                                          tp_st_percent=1.25, tp_pos=3, shift_degree=150,
                                          tp_side="hv", parallel=2)
    t2 = pp.create_transformer_from_parameters(net, b2, b3, vsc_percent=5., vscr_percent=2.,
                                               i0_percent=.4, pfe_kw=2, sn_kva=400, vn_hv_kv=22,
                                               vn_lv_kv=0.42, tp_max=10, tp_mid=5, tp_min=0,
                                               tp_st_percent=1.25, tp_pos=3, tp_side="hv",
                                               shift_degree=150, index=pp.get_free_id(net.trafo) + 1)
    pp.create_switch(net, b3, t2, et="t", closed=False)
    pp.create_transformer_from_parameters(net, b2, b3, vsc_percent=5., vscr_percent=2.,
                                          i0_percent=1., pfe_kw=20, sn_kva=400, vn_hv_kv=20,
                                          vn_lv_kv=0.4, in_service=False)
    pp.create_load(net, b3, p_kw=200, q_kvar=50)
    net.last_added_case = "test_trafo"
    return net


def add_test_single_load_single_eg(net):
    b1 = pp.create_bus(net, vn_kv=20., zone="test_single_load_single_eg")
    pp.create_ext_grid(net, b1)
    pp.create_load(net, b1, p_kw=100, q_kvar=100)
    net.last_added_case = "test_single_load_single_eg"
    return net


def add_test_ward(net):
    b1, b2, ln = add_grid_connection(net, zone="test_ward")

    pz = 1200
    qz = 1100
    ps = 500
    qs = 200
    # one shunt at a bus
    pp.create_ward(net, b2, pz_kw=pz, qz_kvar=qz, ps_kw=ps, qs_kvar=qs)
    # add out of service ward shuold not change the result
    pp.create_ward(net, b2, pz_kw=pz, qz_kvar=qz, ps_kw=ps, qs_kvar=qs, in_service=False,
                   index=pp.get_free_id(net.ward) + 1)
    net.last_added_case = "test_ward"
    return net


def add_test_ward_split(net):
    # splitting up the wards should not change results
    pz = 1200
    qz = 1100
    ps = 500
    qs = 200
    b1, b2, ln = add_grid_connection(net, zone="test_ward_split")
    pp.create_ward(net, b2, pz_kw=pz / 2, qz_kvar=qz / 2, ps_kw=ps / 2, qs_kvar=qs / 2)
    pp.create_ward(net, b2, pz_kw=pz / 2, qz_kvar=qz / 2, ps_kw=ps / 2, qs_kvar=qs / 2)
    net.last_added_case = "test_ward_split"
    return net


def add_test_xward(net):
    b1, b2, ln = add_grid_connection(net, zone="test_xward")

    pz = 1200
    qz = 1100
    ps = 500
    qs = 200
    vm_pu = 1.06
    r_ohm = 50
    x_ohm = 70
    # one xward at a bus
    pp.create_xward(net, b2, pz_kw=pz, qz_kvar=qz, ps_kw=ps, qs_kvar=qs,
                    vm_pu=vm_pu, x_ohm=x_ohm, r_ohm=r_ohm)
    # add out of service xward should not change the result
    pp.create_xward(net, b2, pz_kw=pz, qz_kvar=qz, ps_kw=ps, qs_kvar=qs, vm_pu=vm_pu,
                    x_ohm=x_ohm, r_ohm=r_ohm, in_service=False,
                    index=pp.get_free_id(net.xward) + 1)
    net.last_added_case = "test_xward"
    return net


def add_test_xward_combination(net):
    b1, b2, ln = add_grid_connection(net, zone="test_xward_combination")

    pz = 1200
    qz = 1100
    ps = 500
    qs = 200
    vm_pu = 1.06
    r_ohm = 50
    x_ohm = 70
    # one xward at a bus
    pp.create_xward(net, b2, pz_kw=pz, qz_kvar=qz, ps_kw=ps, qs_kvar=qs,
                    vm_pu=vm_pu, x_ohm=x_ohm, r_ohm=r_ohm)
    # add out of service xward should not change the result
    pp.create_xward(net, b2, pz_kw=pz, qz_kvar=qz, ps_kw=ps, qs_kvar=qs, vm_pu=vm_pu,
                    x_ohm=x_ohm, r_ohm=r_ohm, in_service=False)
    # add second xward at the bus
    pp.create_xward(net, b2, pz_kw=pz, qz_kvar=qz, ps_kw=ps, qs_kvar=qs,
                    vm_pu=vm_pu, x_ohm=x_ohm, r_ohm=r_ohm)
    net.last_added_case = "test_xward_combination"
    return net


def add_test_gen(net):
    b1, b2, ln = add_grid_connection(net, zone="test_gen")
    pl = 1200
    ql = 1100
    ps = -500
    u_set = 1.0

    b3 = pp.create_bus(net, zone="test_gen", vn_kv=.4)
    pp.create_line_from_parameters(net, b2, b3, 12.2, r_ohm_per_km=0.08, x_ohm_per_km=0.12,
                                   c_nf_per_km=300, max_i_ka=.2, df=.8)

    pp.create_load(net, b3, p_kw=pl, q_kvar=ql)
    pp.create_gen(net, b3, p_kw=ps, vm_pu=u_set)
    # adding out of serivce gens should not change the result
    pp.create_gen(net, b2, p_kw=ps, vm_pu=u_set, in_service=False, index=pp.get_free_id(net.gen) + 1)

    net.last_added_case = "test_gen"
    return net


def add_test_enforce_qlims(net):
    b1, b2, ln = add_grid_connection(net, zone="test_enforce_qlims")
    pl = 1200
    ql = 1100
    ps = -500
    qmax = 200.
    u_set = 1.0

    b3 = pp.create_bus(net, zone="test_enforce_qlims", vn_kv=.4)
    pp.create_line_from_parameters(net, b2, b3, 12.2, r_ohm_per_km=0.08, x_ohm_per_km=0.12,
                                   c_nf_per_km=300, max_i_ka=.2, df=.8)

    pp.create_load(net, b3, p_kw=pl, q_kvar=ql)
    pp.create_gen(net, b3, p_kw=ps, vm_pu=u_set, max_q_kvar=qmax)

    net.last_added_case = "test_enforce_qlims"
    return net


def add_test_trafo3w(net):
    b1, b2, ln = add_grid_connection(net, zone="test_trafo3w")
    b3 = pp.create_bus(net, vn_kv=0.6, zone="test_trafo3w")
    pp.create_load(net, b3, p_kw=200, q_kvar=0)
    b4 = pp.create_bus(net, vn_kv=0.4, zone="test_trafo3w")
    pp.create_load(net, b4, p_kw=100, q_kvar=0)

    pp.create_transformer3w_from_parameters(net, hv_bus=b2, mv_bus=b3, lv_bus=b4, vn_hv_kv=22,
                                            vn_mv_kv=.64, vn_lv_kv=.42, sn_hv_kva=1000,
                                            sn_mv_kva=700, sn_lv_kva=300, vsc_hv_percent=1.,
                                            vscr_hv_percent=.03, vsc_mv_percent=.5,
                                            vscr_mv_percent=.02, vsc_lv_percent=.25,
                                            vscr_lv_percent=.01, pfe_kw=.5, i0_percent=0.1,
                                            name="test", index=pp.get_free_id(net.trafo3w) + 1,
                                            tp_side="hv", tp_pos=2, tp_st_percent=1.25,
                                            tp_min=-5, tp_mid=0, tp_max=5)
    # adding out of service 3w trafo should not change results
    pp.create_transformer3w_from_parameters(net, hv_bus=b2, mv_bus=b3, lv_bus=b4, vn_hv_kv=20,
                                            vn_mv_kv=.6, vn_lv_kv=.4, sn_hv_kva=1000, sn_mv_kva=700,
                                            sn_lv_kva=300, vsc_hv_percent=2., vscr_hv_percent=.3,
                                            vsc_mv_percent=1., vscr_mv_percent=.2,
                                            vsc_lv_percent=.5, vscr_lv_percent=.1, pfe_kw=50.,
                                            i0_percent=1., name="test", in_service=False,
                                            index=pp.get_free_id(net.trafo3w) + 1)
    net.last_added_case = "test_trafo3w"
    return net


def add_test_impedance(net):
    b1, b2, ln = add_grid_connection(net, zone="test_impedance")
    b3 = pp.create_bus(net, vn_kv=220., zone="test_impedance")
    rij = 0.02
    xij = 0.01
    rji = 0.03
    xji = 0.005
    s = 2000

    pl = 1000
    ql = 500

    pp.create_impedance(net, b2, b3, rft_pu=rij, xft_pu=xij, rtf_pu=rji, xtf_pu=xji,
                        sn_kva=s, index=pp.get_free_id(net.impedance) + 1)
    pp.create_load(net, b3, p_kw=pl, q_kvar=ql)
    net.last_added_case = "test_impedance"
    return net


def add_test_bus_bus_switch(net):
    b1, b2, ln = add_grid_connection(net, zone="test_bus_bus_switch")
    b3 = pp.create_bus(net, vn_kv=20., zone="test_bus_bus_switch")
    pp.create_switch(net, b2, b3, et="b")

    pl = 1000
    ql = 500

    psg = -500
    qsg = 100

    pz = 1200
    qz = 1100
    ps = 500
    qs = 200

    vm_pu = 1.06
    r_ohm = 50
    x_ohm = 70

    pp.create_load(net, b2, p_kw=pl, q_kvar=ql)
    pp.create_load(net, b3, p_kw=pl, q_kvar=ql, scaling=0.5)

    pp.create_sgen(net, b2, p_kw=psg, q_kvar=qsg)
    pp.create_sgen(net, b3, p_kw=psg, q_kvar=qsg, scaling=0.5)

    pp.create_ward(net, b2, pz_kw=pz, qz_kvar=qz, ps_kw=ps, qs_kvar=qs)
    pp.create_ward(net, b3, pz_kw=0.5 * pz, qz_kvar=0.5 * qz, ps_kw=0.5 * ps, qs_kvar=0.5 * qs)

    pp.create_xward(net, b3, pz_kw=0.5 * pz, qz_kvar=0.5 * qz, ps_kw=0.5 * ps, qs_kvar=0.5 * qs,
                    vm_pu=vm_pu, x_ohm=x_ohm, r_ohm=r_ohm)
    pp.create_xward(net, b2, pz_kw=pz, qz_kvar=qz, ps_kw=ps, qs_kvar=qs,
                    vm_pu=vm_pu, x_ohm=x_ohm, r_ohm=r_ohm)
    net.last_added_case = "test_bus_bus_switch"
    return net


def add_test_oos_bus_with_is_element(net):
    b1, b2, ln = add_grid_connection(net, zone="test_oos_bus_with_is_element")

    pl = 1200
    ql = 1100
    ps = -500
    u_set = 1.0

    pz = 1200
    qz = 1100
    qs = 200

    vm_pu = 1.06
    r_ohm = 50
    x_ohm = 70

    # OOS buses
    b3 = pp.create_bus(net, zone="test_oos_bus_with_is_element", vn_kv=0.4, in_service=False)
    b4 = pp.create_bus(net, zone="test_oos_bus_with_is_element", vn_kv=0.4, in_service=False)
    b5 = pp.create_bus(net, zone="test_oos_bus_with_is_element", vn_kv=0.4, in_service=False)

    pp.create_line_from_parameters(net, b2, b3, 12.2, r_ohm_per_km=0.08, x_ohm_per_km=0.12,
                                   c_nf_per_km=300, max_i_ka=.2, df=.8)
    pp.create_line_from_parameters(net, b2, b4, 12.2, r_ohm_per_km=0.08, x_ohm_per_km=0.12,
                                   c_nf_per_km=300, max_i_ka=.2, df=.8)
    pp.create_line_from_parameters(net, b2, b5, 12.2, r_ohm_per_km=0.08, x_ohm_per_km=0.12,
                                   c_nf_per_km=300, max_i_ka=.2, df=.8)

    # in service elements
    pp.create_load(net, b3, p_kw=pl, q_kvar=ql)
    pp.create_gen(net, b4, p_kw=ps, vm_pu=u_set)
    pp.create_sgen(net, b5, p_kw=ps, q_kvar=ql)
    pp.create_ward(net, b3, pz_kw=pz, qz_kvar=qz, ps_kw=ps, qs_kvar=qs)
    pp.create_xward(net, b4, pz_kw=0.5 * pz, qz_kvar=0.5 * qz, ps_kw=0.5 * ps, qs_kvar=0.5 * qs,
                    vm_pu=vm_pu, x_ohm=x_ohm, r_ohm=r_ohm)
    pp.create_shunt(net, b5, q_kvar=-800, p_kw=0)

    net.last_added_case = "test_oos_bus_with_is_element"
    return net


def add_test_shunt(net):
    b1, b2, ln = add_grid_connection(net, zone="test_shunt")
    pz = 120
    qz = -1200
    # one shunt at a bus
    pp.create_shunt_as_capacitor(net, b2, q_kvar=1200, loss_factor=0.1, vn_kv=22., step=2)
    # add out of service shunt shuold not change the result
    pp.create_shunt(net, b2, p_kw=pz, q_kvar=qz, in_service=False)
    return net


def add_test_shunt_split(net):
    b1, b2, ln = add_grid_connection(net, zone="test_shunt_split")
    pz = 120
    qz = -1200
    # one shunt at a bus
    pp.create_shunt(net, b2, p_kw=pz / 2, q_kvar=qz / 2)
    pp.create_shunt(net, b2, p_kw=pz / 2, q_kvar=qz / 2)
    return net


def add_test_two_open_switches_on_deactive_line(net):
    b1, b2, l1 = add_grid_connection(net, zone="two_open_switches_on_deactive_line")
    b3 = pp.create_bus(net, vn_kv=20.)
    l2 = create_test_line(net, b2, b3, in_service=False)
    create_test_line(net, b3, b1)
    pp.create_switch(net, b2, l2, et="l", closed=False)
    pp.create_switch(net, b3, l2, et="l", closed=False)
    return net


if __name__ == '__main__':
    net = pp.create_empty_network()
    b1, b2, ln = add_grid_connection(net, zone="test_shunt")
    pz = 120
    qz = -1200
    # one shunt at a bus
    pp.create_shunt_as_capacitor(net, b2, q_kvar=1200, loss_factor=0.1, vn_kv=20., step=1)
    # add out of service shunt shuold not change the result
    pp.create_shunt(net, b2, p_kw=pz, q_kvar=qz, in_service=False)
    # add out of service shunt shuold not change the result
    pp.runpp(net)