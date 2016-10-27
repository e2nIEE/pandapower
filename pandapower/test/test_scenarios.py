# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a 
# BSD-style license that can be found in the LICENSE file.

import pytest
import pandapower as pp
from pandapower.test.toolbox import create_test_network2
from pandapower.test.consistency_checks import runpp_with_consistency_checks
from pandapower.test.result_test_network_generator import add_test_bus_bus_switch
import numpy as np

#TODO: 2 gen 2 ext_grid missing

def test_2gen_1ext_grid():
    net = create_test_network2()
    net.shunt.q_kvar *= -1
    pp.create_gen(net, 2, p_kw=-100)
    net.trafo.shift_degree = 150
    pp.runpp(net, init='dc', calculate_voltage_angles=True)

    assert np.allclose(net.res_gen.p_kw.values, [-100., -100.])
    assert np.allclose(net.res_gen.q_kvar.values, [447.397232056, 
                                                   51.8152713776])
    assert np.allclose(net.res_gen.va_degree.values, [0.242527288986, 
                                                      -143.558157703])
    assert np.allclose(net.res_gen.vm_pu.values, [1.0, 1.0])

    assert np.allclose(net.res_bus.vm_pu, [1.000000, 0.956422, 1.000000, 
                                           1.000000])
    assert np.allclose(net.res_bus.va_degree, [0.000000, -145.536429154, 
                                               -143.558157703, 0.242527288986])
    assert np.allclose(net.res_bus.p_kw, [61.87173, 30.00000, -100.00000,
                                          0.00000])
    assert np.allclose(net.res_bus.q_kvar, [-470.929980278, 2.000000, 
                                            21.8152713776, 447.397232056])
    assert np.allclose(net.res_ext_grid.p_kw.values, [61.87173])
    assert np.allclose(net.res_ext_grid.q_kvar, [-470.927898])


def test_0gen_2ext_grid():
    # testing 2 ext grid and 0 gen, both EG on same trafo side
    net = create_test_network2()
    net.shunt.q_kvar *= -1
    pp.create_ext_grid(net, 1)
    net.gen = net.gen.drop(0)
    net.trafo.shift_degree = 150
    net.ext_grid.in_service.at[1] = False
    pp.create_ext_grid(net, 3)

    pp.runpp(net, init='dc', calculate_voltage_angles=True)
    assert np.allclose(net.res_bus.p_kw.values, [-0.000000, 30.000000, 
                                                 0.000000, -32.993015])
    assert np.allclose(net.res_bus.q_kvar.values, [4.08411026001, 2.000000,
                                                   -28.6340014753, 27.437210083])
    assert np.allclose(net.res_bus.va_degree.values, [0.000000, -155.719283,
                                                      -153.641832, 0.000000])
    assert np.allclose(net.res_bus.vm_pu.values,  [1.000000, 0.932225, 
                                                   0.976965, 1.000000])
    
    assert np.allclose(net.res_ext_grid.p_kw.values, [-0.000000, 0.000000, -132.993015])
    assert np.allclose(net.res_ext_grid.q_kvar, [4.08411026001, 0.000000, 27.437210083])


def test_0gen_2ext_grid_decoupled():
    net = create_test_network2()
    net.gen = net.gen.drop(0)
    net.shunt.q_kvar *= -1
    pp.create_ext_grid(net, 1)
    net.ext_grid.in_service.at[1] = False
    pp.create_ext_grid(net, 3)
    net.ext_grid.in_service.at[2] = False
    auxbus = pp.create_bus(net, name="bus1", vn_kv=10.)
    net.trafo.shift_degree = 150
    pp.create_std_type(net, {"type": "cs", "r_ohm_per_km": 0.876,  "q_mm2": 35.0,
                             "endtmp_deg": 160.0, "c_nf_per_km": 260.0,
                             "imax_ka": 0.123, "x_ohm_per_km": 0.1159876}, 
                             name="NAYSEY 3x35rm/16 6/10kV" , element="line")
    pp.create_line(net, 0, auxbus, 1, name="line_to_decoupled_grid",
                   std_type="NAYSEY 3x35rm/16 6/10kV") #NAYSEY 3x35rm/16 6/10kV
    pp.create_ext_grid(net, auxbus)
    pp.create_switch(net, auxbus, 2, et="l", closed=0, type="LS")
    pp.runpp(net, init='dc', calculate_voltage_angles=True)

    assert np.allclose(net.res_bus.p_kw.values, [-133.158732, 30.000000, 
                                             0.000000, 100.000000, 0.000000])
    assert np.allclose(net.res_bus.q_kvar.values, [39.5843982697, 2.000000, 
                                           -28.5636406913, 0.000000, 0.000000])
    assert np.allclose(net.res_bus.va_degree.values, [0.000000, -155.752225311,
                                                      -153.669395244, 
                                                      -0.0225931152895, 0.0])
    assert np.allclose(net.res_bus.vm_pu.values,  [1.000000, 0.930961, 
                                                   0.975764, 0.998865, 1.0])
    
    assert np.allclose(net.res_ext_grid.p_kw.values, [-133.158732, 0.000000, 0.000000, -0.000000])
    assert np.allclose(net.res_ext_grid.q_kvar, [39.5843982697, 0.000000, 0.000000, -0.000000])


def test_bus_bus_switch_at_eg():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, name="bus1", vn_kv=.4)
    b2 = pp.create_bus(net, name="bus2", vn_kv=.4)
    b3 = pp.create_bus(net, name="bus3", vn_kv=.4)

    pp.create_ext_grid(net, b1)

    pp.create_switch(net, b1, et="b", element=1)
    pp.create_line(net, b2, b3, 1, name="line1",
                   std_type="NAYY 4x150 SE")

    pp.create_load(net, b3, p_kw=10, q_kvar=0, name="load1")

    runpp_with_consistency_checks(net)


def test_bb_switch():
    net = pp.create_empty_network()
    net = add_test_bus_bus_switch(net)
    runpp_with_consistency_checks(net)


if __name__ == "__main__":
    pytest.main(["test_scenarios.py"])
