# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a 
# BSD-style license that can be found in the LICENSE file.

import pandapower as pp
import pytest

from pandapower.test.consistency_checks import runpp_with_consistency_checks
from pandapower.test.result_test_network_generator import add_test_enforce_qlims, add_test_gen

def test_line(result_test_network):
    net = result_test_network
    buses = net.bus[net.bus.zone == "test_line"]
    lines = [x for x in net.line.index if net.line.from_bus[x] in buses.index]
    l1 = lines[0]
    l2 = lines[1]
    l3 = lines[2]
    b2 = buses.index[1]

    # result values from powerfactory
    load1 = 14.578
    load2 = 8.385

    ika1 = 0.04665
    ika2 = 0.0134

    p_from1 = 1202.21
    p_from2 = 0.132

    q_from1 = 167.390
    q_from2 = -469.371

    p_to1 = -1200.000
    p_to2 = 0.000

    q_to1 = -1100.000
    q_to2 = 0.0000

    v = 1.007395422

    # line 1
    assert abs(net.res_line.loading_percent.at[l1] - load1) < 1e-2
    assert abs(net.res_line.i_ka.at[l1] - ika1) < 1e-2
    assert abs(net.res_line.p_from_kw.at[l1] - p_from1) < 1e-2
    assert abs(net.res_line.q_from_kvar.at[l1] - q_from1) < 1e-2
    assert abs(net.res_line.p_to_kw.at[l1] - p_to1) < 1e-2
    assert abs(net.res_line.q_to_kvar.at[l1] - q_to1) < 1e-2

    # line2 (open switch line)
    assert abs(net.res_line.loading_percent.at[l2] - load2) < 1e-2
    assert abs(net.res_line.i_ka.at[l2] - ika2) < 1e-2
    assert abs(net.res_line.p_from_kw.at[l2] - p_from2) < 1e-2
    assert abs(net.res_line.q_from_kvar.at[l2] - q_from2) < 1e-2
    assert abs(net.res_line.p_to_kw.at[l2] - p_to2) < 1e-2
    assert abs(net.res_line.q_to_kvar.at[l2] - q_to2) < 1e-2

    assert abs(net.res_bus.vm_pu.at[b2] - v) < 1e-8

    # line3 (of out of service line)
    assert abs(net.res_line.loading_percent.at[l3] - 0) < 1e-2
    assert abs(net.res_line.i_ka.at[l3] - 0) < 1e-2
    assert abs(net.res_line.p_from_kw.at[l3] - 0) < 1e-2
    assert abs(net.res_line.q_from_kvar.at[l3] - 0) < 1e-2
    assert abs(net.res_line.p_to_kw.at[l3] - 0) < 1e-2
    assert abs(net.res_line.q_to_kvar.at[l3] - 0) < 1e-2


def test_load_sgen(result_test_network):
     net = result_test_network
     buses = net.bus[net.bus.zone == "test_load_sgen"]
     loads = [x for x in net.load.index if net.load.bus[x] in buses.index]
     sgens = [x for x in net.sgen.index if net.sgen.bus[x] in buses.index]
     l1 = loads[0]
     sg1 = sgens[0]
     b2 = buses.index[1]
     # result values from powerfactory
     pl1 = 1200.000
     ql1 = 1100.000

     qs1 = -100.000
     ps1 = 500.000

     u = 1.00477465

     assert abs(net.res_load.p_kw.at[l1] - pl1) < 1e-2
     assert abs(net.res_load.q_kvar.at[l1] - ql1) < 1e-2
     # pf uses generator system
     assert abs(net.res_sgen.p_kw.at[sg1] - (- ps1)) < 1e-2
     # pf uses generator system
     assert abs(net.res_sgen.q_kvar.at[sg1] - (-qs1)) < 1e-2
     assert abs(net.res_bus.vm_pu.at[b2] - u) < 1e-8


def test_load_sgen_split(result_test_network):

     # splitting up the load/sgen should not change the result
     net = result_test_network
     buses = net.bus[net.bus.zone == "test_load_sgen_split"]
     b2 = buses.index[1]

     u = 1.00477465

     assert abs(net.res_bus.vm_pu.at[b2] - u) < 1e-8


def test_trafo(result_test_network):
     net = result_test_network
     buses = net.bus[net.bus.zone == "test_trafo"]
     trafos = [x for x in net.trafo.index if net.trafo.hv_bus[x] in buses.index]
     t1 = trafos[0]
     t2 = trafos[1]
     t3 = trafos[2]
     b2 = buses.index[1]
     b3 = buses.index[2]
 #     powerfactory results (to check t-equivalent circuit model)
     runpp_with_consistency_checks(net, trafo_model="t", trafo_loading="current")

     load1 = 56.7348
     load2 = 5.0478

     ph1 = 222.4211
     ph2 = 20.3943

     qh1 = 55.4248
     qh2 = 0.0362

     pl1 = -199.9981
     pl2 = 0

     ql1 = -49.9957
     ql2 = 0

     ih1 = 0.006551
     ih2 = 0.000583

     il1 = 0.299500
     il2 = 0

     v2 = 1.01006174
     v3 = 0.99350859


     assert abs(net.res_trafo.loading_percent.at[t1] - load1) < 1e-1
     assert abs(net.res_trafo.p_hv_kw.at[t1] - ph1) < 1e-1
     assert abs(net.res_trafo.q_hv_kvar.at[t1] - qh1) < 1e-1
     assert abs(net.res_trafo.p_lv_kw.at[t1] - pl1) < 1e-1
     assert abs(net.res_trafo.q_lv_kvar.at[t1] - ql1) < 1e-1
     assert abs(net.res_trafo.i_hv_ka.at[t1] - ih1) < 1e-1
     assert abs(net.res_trafo.i_lv_ka.at[t1] - il1) < 1e-1

     assert abs(net.res_trafo.loading_percent.at[t2] - load2) < 1e-1
     assert abs(net.res_trafo.p_hv_kw.at[t2] - ph2) < 1e-1
     assert abs(net.res_trafo.q_hv_kvar.at[t2] - qh2) < 1e-1
     assert abs(net.res_trafo.p_lv_kw.at[t2] - pl2) < 1e-1
     assert abs(net.res_trafo.q_lv_kvar.at[t2] - ql2) < 1e-1
     assert abs(net.res_trafo.i_hv_ka.at[t2] - ih2) < 1e-1
     assert abs(net.res_trafo.i_lv_ka.at[t2] - il2) < 1e-1

     assert abs(net.res_trafo.loading_percent.at[t3] - 0) < 1e-1
     assert abs(net.res_trafo.p_hv_kw.at[t3] - 0) < 1e-1
     assert abs(net.res_trafo.q_hv_kvar.at[t3] - 0) < 1e-1
     assert abs(net.res_trafo.p_lv_kw.at[t3] - 0) < 1e-1
     assert abs(net.res_trafo.q_lv_kvar.at[t3] - 0) < 1e-1
     assert abs(net.res_trafo.i_hv_ka.at[t3] - 0) < 1e-1
     assert abs(net.res_trafo.i_lv_ka.at[t3] - 0) < 1e-1

     assert abs(net.res_bus.vm_pu.at[b2] - v2) < 1e-6
     assert abs(net.res_bus.vm_pu.at[b3] - v3) < 1e-6

 #    # sincal results (to check pi-equivalent circuit model)
     runpp_with_consistency_checks(net, trafo_model="pi", trafo_loading="current")

     load1 = 56.76
     load2 = 5.049

     v2 = 1.010061887962
     v3 = 0.9935012394385


     assert abs(net.res_trafo.loading_percent.at[t1] - load1) < 1e-1
     assert abs(net.res_trafo.loading_percent.at[t2] - load2) < 1e-1

     assert abs(net.res_bus.vm_pu.at[b2] - v2) < 1e-6
     assert abs(net.res_bus.vm_pu.at[b3] - v3) < 1e-6

     runpp_with_consistency_checks(net, trafo_model="pi", trafo_loading="power")

     load1 = 57.307
     load2 = 5.10

     assert abs(net.res_trafo.loading_percent.at[t1] - load1) < 1e-1
     assert abs(net.res_trafo.loading_percent.at[t2] - load2) < 1e-1


def test_trafo_tap(result_test_network):
     net = result_test_network
     runpp_with_consistency_checks(net, trafo_model="t", trafo_loading="current")

     buses = net.bus[net.bus.zone == "test_trafo_tap"]
     b2 = buses.index[1]
     b3 = buses.index[2]

     assert (1.010114175 - net.res_bus.vm_pu.at[b2]) < 1e-6
     assert (0.924072090 - net.res_bus.vm_pu.at[b3]) < 1e-6


#def test_shunt(net):
#     b1, b2, ln = add_grid_connection(net)
#     pz = 1200
#     qz = 1100
#     # one shunt at a bus
#     pp.create_shunt(net, b2, p_kw=pz, q_kvar=qz)
#     runpp_with_consistency_checks(net)
# 
# #    u =  0.99061732759039389
# #    assert abs(net.res_bus.vm_pu.loc[b2] - u) < 1e-6
# 
#     # add out of service shunt shuold not change the result
#     pp.create_shunt(net, b2, p_kw=pz, q_kvar=qz, in_service=False)
#     runpp_with_consistency_checks(net)
# 
# #    assert abs(net.res_bus.vm_pu.loc[b2] - u) < 1e-6
# 
#     # splitting up the shunts should not change results
#     b1, b2, ln = add_grid_connection(net)
#     pp.create_shunt(net, b2, p_kw=pz/2, q_kvar=qz/2)
#     pp.create_shunt(net, b2, p_kw=pz/2, q_kvar=qz/2)
#     runpp_with_consistency_checks(net)
# #    assert abs(net.res_bus.vm_pu.loc[b2] - u) < 1e-6
 #
 
def test_ext_grid(result_test_network):
     net = result_test_network
     buses = net.bus[net.bus.zone == "test_ext_grid"]
     ext_grids = [
         x for x in net.ext_grid.index if net.ext_grid.bus[x] in buses.index]
     eg1 = ext_grids[0]
     eg2 = ext_grids[1]
     # results from powerfactory
     p1 = -1273.6434
     q1 = -2145.0519

     p2 = 1286.2537
     q2 = 1690.1253

     assert abs(net.res_ext_grid.p_kw.at[eg1] - (-p1))
     assert abs(net.res_ext_grid.q_kvar.at[eg1] - (-q1))

     assert abs(net.res_ext_grid.p_kw.at[eg2] - (-p2))
     assert abs(net.res_ext_grid.q_kvar.at[eg2] - (-q2))


def test_ward(result_test_network):
     net = result_test_network
     buses = net.bus[net.bus.zone == "test_ward"]
     wards = [x for x in net.ward.index if net.ward.bus[x] in buses.index]
     b2 = buses.index[1]
     w1 = wards[0]
     # powerfactory results
     pw = -1704.6146
     qw = -1304.2294
     u = 1.00192121

     assert abs(net.res_bus.vm_pu.loc[b2] - u) < 1e-6
     assert abs(net.res_ward.p_kw.loc[w1] - (-pw)) < 1e-1
     assert abs(net.res_ward.q_kvar.loc[w1] - (-qw)) < 1e-1


def test_ward_split(result_test_network):
     net = result_test_network
     buses = net.bus[net.bus.zone == "test_ward_split"]
     wards = [x for x in net.ward.index if net.ward.bus[x] in buses.index]
     b2 = buses.index[1]
     w1 = wards[0]
     w2 = wards[1]
     # powerfactory results
     pw = -1704.6146
     qw = -1304.2294
     u = 1.00192121

     assert abs(net.res_bus.vm_pu.at[b2] - u)
     assert abs(net.res_ward.p_kw.loc[[w1, w2]].sum() - (-pw))
     assert abs(net.res_ward.q_kvar.loc[[w1, w2]].sum() - (-qw))
 #


def test_xward(result_test_network):
     net = result_test_network
     buses = net.bus[net.bus.zone == "test_xward"]
     xwards = [x for x in net.xward.index if net.xward.bus[x] in buses.index]
     b2 = buses.index[1]
     xw1 = xwards[0]
     xw2 = xwards[1]  # Out of servic xward
 #    powerfactory result for 1 xward
     u = 1.00308684
     pxw = -1721.0343
     qxw = -975.9919
 #
     assert abs(net.res_bus.vm_pu.at[b2] - u) < 1e-2
     assert abs(net.res_xward.p_kw.at[xw1] - (-pxw)) < 1e-2
     assert abs(net.res_xward.q_kvar.at[xw1] - (-qxw)) < 1e-2

     assert abs(net.res_bus.vm_pu.at[b2] - u) < 1e-2
     assert abs(net.res_xward.p_kw.loc[[xw1, xw2]].sum() - (-pxw)) < 1e-2
     assert abs(net.res_xward.q_kvar.loc[[xw1, xw2]].sum() - (-qxw)) < 1e-2


def test_xward_combination(result_test_network):
     net = result_test_network
     buses = net.bus[net.bus.zone == "test_xward_combination"]
     xwards = [x for x in net.xward.index if net.xward.bus[x] in buses.index]
     b2 = buses.index[1]
     xw1 = xwards[0]
     xw3 = xwards[2]

     # powerfactory result for 2 active xwards
     u = 0.99568034
     pxw1 = -1707.1063
     pxw3 = -1707.1063

     qxw1 = -918.7192
     qxw3 = -918.7192

     assert abs(net.res_bus.vm_pu.at[b2] - u) < 1e-2
     assert abs(net.res_xward.p_kw.at[xw1] - (-pxw1)) < 1e-1
     assert abs(net.res_xward.q_kvar.at[xw1] - (-qxw1)) < 1e-1

     assert abs(net.res_xward.p_kw.at[xw3] - (-pxw3)) < 1e-1
     assert abs(net.res_xward.q_kvar.at[xw3] - (-qxw3)) < 1e-1


def test_gen(result_test_network):
     net = result_test_network
     buses = net.bus[net.bus.zone == "test_gen"]
     gens = [x for x in net.gen.index if net.gen.bus[x] in buses.index]
     b2 = buses.index[1]
     b3 = buses.index[2]
     g1 = gens[0]
     # powerfactory results
     q = -260.660
     u2 = 1.00584636
     u_set = 1.0

     assert abs(net.res_bus.vm_pu.at[b2] - u2) < 1e-8
     assert abs(net.res_bus.vm_pu.at[b3] - u_set) < 1e-8
     assert abs(net.res_gen.q_kvar.at[g1] - (-q)) < 1e-1


def test_enforce_qlims(result_test_network):
     net = result_test_network
     buses = net.bus[net.bus.zone == "test_enforce_qlims"]
     gens = [x for x in net.gen.index if net.gen.bus[x] in buses.index]
     b2 = buses.index[1]
     b3 = buses.index[2]
     g1 = gens[0]


 #    enforce reactive power limits
     runpp_with_consistency_checks(net, enforce_q_lims=True)

     # powerfactory results
     u2 = 1.00607194
     u3 = 1.00045091

     assert abs(net.res_bus.vm_pu.at[b2] - u2) < 1e-2
     assert abs(net.res_bus.vm_pu.at[b3] - u3) < 1e-2
     assert abs(net.res_gen.q_kvar.at[g1] - net.gen.max_q_kvar.at[g1]) < 1e-2
 #
 #


def test_trafo3w(result_test_network):
     net = result_test_network
     buses = net.bus[net.bus.zone == "test_trafo3w"]
     trafos = [x for x in net.trafo3w.index if net.trafo3w.hv_bus[
         x] in buses.index]
     runpp_with_consistency_checks(net, trafo_model="t")
     b2 = buses.index[1]
     b3 = buses.index[2]
     b4 = buses.index[3]
     t3 = trafos[0]

     uhv = 1.00895246
     umv = 1.00440765
     ulv = 1.00669961

     load = 68.261
     qhv = 154.60
     qmv = -100.00
     qlv = -50.00

     phv = 551.43
     pmv = -300.00
     plv = -200.00

     assert abs((net.res_bus.vm_pu.at[b2] - uhv)) < 1e-4
     assert abs((net.res_bus.vm_pu.at[b3] - umv)) < 1e-4
     assert abs((net.res_bus.vm_pu.at[b4] - ulv)) < 1e-4

     assert abs((net.res_trafo3w.loading_percent.at[t3] - load)) < 1e-2

     assert abs((net.res_trafo3w.p_hv_kw.at[t3] - phv)) < 1
     assert abs((net.res_trafo3w.p_mv_kw.at[t3] - pmv)) < 1
     assert abs((net.res_trafo3w.p_lv_kw.at[t3] - plv)) < 1

     assert abs((net.res_trafo3w.q_hv_kvar.at[t3] - qhv)) < 1
     assert abs((net.res_trafo3w.q_mv_kvar.at[t3] - qmv)) < 1
     assert abs((net.res_trafo3w.q_lv_kvar.at[t3] - qlv)) < 1

     # power transformer loading
     runpp_with_consistency_checks(net, trafo_model="t", trafo_loading="power")
     load_p = 68.718
     assert abs((net.res_trafo3w.loading_percent.at[t3] - load_p)) < 1e-2


def test_impedance(result_test_network):
     net = result_test_network
     buses = net.bus[net.bus.zone == "test_impedance"]
     impedances = [
         x for x in net.impedance.index if net.impedance.from_bus[x] in buses.index]
     runpp_with_consistency_checks(net, trafo_model="t")
     b2 = buses.index[1]
     b3 = buses.index[2]
     imp1 = impedances[0]

     # powerfactory results
     ifrom = 0.0325
     ito = 0.0030

     pfrom = 1012.6480
     qfrom = 506.3231

     pto = -999.9960
     qto = -499.9971

     u2 = 1.00654678
     u3 = 0.99397101

     assert abs(net.res_impedance.p_from_kw.at[imp1] - pfrom) < 1e-1
     assert abs(net.res_impedance.p_to_kw.at[imp1] - pto) < 1e-1
     assert abs(net.res_impedance.q_from_kvar.at[imp1] - qfrom) < 1e-1
     assert abs(net.res_impedance.q_to_kvar.at[imp1] - qto) < 1e-1
     assert abs(net.res_impedance.i_from_ka.at[imp1] - ifrom) < 1e-1
     assert abs(net.res_impedance.i_to_ka.at[imp1] - ito) < 1e-1

     assert abs(net.res_bus.vm_pu.at[b2] - u2) < 1e-6
     assert abs(net.res_bus.vm_pu.at[b3] - u3) < 1e-6


def test_bus_bus_switch(result_test_network):
     net = result_test_network
     buses = net.bus[net.bus.zone == "test_bus_bus_switch"]
     b2 = buses.index[1]
     b3 = buses.index[2]

     # powerfactory voltage
     u = 0.982265380
     assert abs(net.res_bus.vm_pu.at[b2] - u) < 1e-5
     assert abs(net.res_bus.vm_pu.at[b3] - u) < 1e-5
     assert abs(net.res_bus.vm_pu.at[b2] == net.res_bus.vm_pu.at[b2])


def test_enforce_q_lims():
    """ Test for enforce_q_lims loadflow option
    """
#    net = pp.test.create_test_network()
#    net.gen.max_q_kvar = 1000
#    net.gen.min_q_kvar = -1000
#    pp.runpp(net, enforce_q_lims=True)
    net = pp.create_empty_network()
    # test_gen
    net = add_test_gen(net)
    pp.runpp(net)
    buses = net.bus[net.bus.zone == "test_gen"]
    gens = [x for x in net.gen.index if net.gen.bus[x] in buses.index]
#    b1=buses.index[0]
    b2 = buses.index[1]
    b3 = buses.index[2]
    g1 = gens[0]
    q = -260.660
    u2 = 1.00584636
    u_set = 1.0
    assert abs(net.res_bus.vm_pu.at[b2] - u2) < 1e-8
    assert abs(net.res_bus.vm_pu.at[b3] - u_set) < 1e-8
    assert abs(net.res_gen.q_kvar.at[g1] - (-q)) < 1e-1

    # test_enforce_qlims
    net = add_test_enforce_qlims(net)

    pp.runpp(net, enforce_q_lims=True)
    buses = net.bus[net.bus.zone == "test_enforce_qlims"]
    gens = [x for x in net.gen.index if net.gen.bus[x] in buses.index]
    b2 = buses.index[1]
    b3 = buses.index[2]
    g1 = gens[0]
    u2 = 1.00607194
    u3 = 1.00045091
    assert abs(net.res_bus.vm_pu.at[b2] - u2) < 1e-2
    assert abs(net.res_bus.vm_pu.at[b3] - u3) < 1e-2
    assert abs(net.res_gen.q_kvar.at[g1] - net.gen.max_q_kvar.at[g1]) < 1e-2

def test_shunt(result_test_network):
    net = result_test_network
    buses = net.bus[net.bus.zone == "test_shunt"]
    b2 = buses.index[1]
    shunts = [x for x in net.shunt.index if net.shunt.bus[x] in buses.index]
    s1 = shunts[0]

    u =  1.015007
    p = 123.628741 
    q = -1236.287413
    
    assert abs(net.res_bus.vm_pu.loc[b2] - u) < 1e-6
    assert abs(net.res_shunt.p_kw.loc[s1] - p) < 1e-6
    assert abs(net.res_shunt.q_kvar.loc[s1] - q) < 1e-6

def test_shunt_split(result_test_network):
    net = result_test_network
    buses = net.bus[net.bus.zone == "test_shunt_split"]
    b2 = buses.index[1]
    shunts = [x for x in net.shunt.index if net.shunt.bus[x] in buses.index]
    s1 = shunts[0]

    u =  1.015007
    p = 123.628741 
    q = -1236.287413

    assert abs(net.res_bus.vm_pu.loc[b2] - u) < 1e-6
    assert abs(net.res_shunt.p_kw.loc[s1] - p/2) < 1e-6
    assert abs(net.res_shunt.q_kvar.loc[s1] - q/2) < 1e-6

 

# def test_trafo3w_tap(net):
# TODO

if __name__ == "__main__":
    pytest.main(["test_results.py", "-s"])

