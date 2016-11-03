# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a 
# BSD-style license that can be found in the LICENSE file.

import pytest
import pandas as pd
import numpy as np

import pandapower as pp
from pandapower.test.toolbox import add_grid_connection, create_test_line
from pandapower.test.result_test_network_generator import result_test_network_generator
from pandapower.test.consistency_checks import runpp_with_consistency_checks

def test_runpp_init():
    net = pp.create_empty_network()
    b1, b2, l1 = add_grid_connection(net)
    b3 = pp.create_bus(net, vn_kv=0.4)
    tidx = pp.create_transformer(net, hv_bus=b2, lv_bus=b3, std_type="0.25 MVA 20/0.4 kV")
    net.trafo.shift_degree.at[tidx] = 70
    pp.runpp(net)
    va = net.res_bus.va_degree.at[4]   
    pp.runpp(net, calculate_voltage_angles=True, init="dc")
    assert np.allclose(va - net.trafo.shift_degree.at[tidx], net.res_bus.va_degree.at[4])
    pp.runpp(net, calculate_voltage_angles=True, init="results")
    assert np.allclose(va - net.trafo.shift_degree.at[tidx], net.res_bus.va_degree.at[4])

def test_runpp_init_auxiliary_buses():
    net = pp.create_empty_network()
    b1, b2, l1 = add_grid_connection(net, vn_kv=110.)
    b3 = pp.create_bus(net, vn_kv=20.)
    b4 = pp.create_bus(net, vn_kv=10.)
    tidx = pp.create_transformer3w(net, b2, b3, b4, std_type='63/25/38 MVA 110/20/10 kV')
    pp.create_load(net, b3, 5e3)
    pp.create_load(net, b4, 5e3)
    pp.create_xward(net, b4, 1000, 1000, 1000, 1000, 0.1, 0.1, 1.0)    
    net.trafo3w.shift_lv_degree.at[tidx] = 120
    net.trafo3w.shift_mv_degree.at[tidx] = 80
    pp.runpp(net)
    va = net.res_bus.va_degree.at[b2]
    pp.runpp(net, calculate_voltage_angles=True, init="dc")
    assert np.allclose(va - net.trafo3w.shift_mv_degree.at[tidx], net.res_bus.va_degree.at[b3],
                       atol=2)
    assert np.allclose(va - net.trafo3w.shift_lv_degree.at[tidx], net.res_bus.va_degree.at[b4],
                       atol=2)
    pp.runpp(net, calculate_voltage_angles=True, init="results")
    assert np.allclose(va - net.trafo3w.shift_mv_degree.at[tidx], net.res_bus.va_degree.at[b3],
                       atol=2)
    assert np.allclose(va - net.trafo3w.shift_lv_degree.at[tidx], net.res_bus.va_degree.at[b4], 
                       atol=2)
    
def test_result_iter():
    for net in result_test_network_generator():
#        try:
        runpp_with_consistency_checks(net, enforce_q_lims=True)
#        except (AssertionError, pp.LoadflowNotConverged) :
#            pytest.fail(["Error after adding_", net.last_added_case])
#            raise UserWarning(")

def test_bus_bus_switches():
    net = pp.create_empty_network()
    add_grid_connection(net)
    for _u in range(4):
        pp.create_bus(net, vn_kv=.4)
    pp.create_load(net, 5, p_kw=10)
    pp.create_switch(net, 3, 6, et="b")
    pp.create_switch(net, 4, 5, et="b")
    pp.create_switch(net, 6, 5, et="b")
    pp.create_switch(net, 0, 7, et="b")
    create_test_line(net, 4, 7)
    pp.create_load(net, 4, p_kw=10)
    pp.runpp(net)
    assert net.res_bus.vm_pu.at[3] == net.res_bus.vm_pu.at[4] == net.res_bus.vm_pu.at[5] == \
            net.res_bus.vm_pu.at[6]
    assert net.res_bus.vm_pu.at[0] == net.res_bus.vm_pu.at[7]

    net.bus.in_service.at[5] = False   
    pp.runpp(net)
    assert net.res_bus.vm_pu.at[3] == net.res_bus.vm_pu.at[6]
    assert net.res_bus.vm_pu.at[0] == net.res_bus.vm_pu.at[7]
    assert pd.isnull(net.res_bus.vm_pu.at[5])
    assert net.res_bus.vm_pu.at[6] != net.res_bus.vm_pu.at[4]

def test_two_open_switches():
    net = pp.create_empty_network()
    b1, b2, l1 = add_grid_connection(net)
    b3 = pp.create_bus(net, vn_kv=20.)
    l2 = create_test_line(net, b2, b3)
    create_test_line(net, b3, b1)    
    pp.create_switch(net, b2, l2, et="l", closed=False)
    pp.create_switch(net, b3, l2, et="l", closed=False)
    pp.runpp(net)
    assert net.res_line.i_ka.at[l2] == 0.

if __name__ == "__main__":
    pytest.main(["test_runpp.py", "-xs"])

