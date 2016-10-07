    # -*- coding: utf-8 -*-
from __future__ import print_function
import pytest
import pandapower as pp
import pandas as pd
from pandapower.test.toolbox import assert_res_out_of_service, add_grid_connection, create_test_line
from numpy import array_equal, array, allclose
import numpy as np
from pandapower.test.result_test_network_generator import result_test_network_generator, add_test_gen, add_test_enforce_qlims
from pandapower.test.consistency_checks import runpp_with_consistency_checks
from pandapower.build_bus import _build_bus_mpc

def test_voltage_angles():
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
        pp.create_bus(net)
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

if __name__ == "__main__":
    nets = [net for net in result_test_network_generator()]
#    net = nets[3]
#    pp.runpp(net)
#        try:
#        runpp_with_consistency_checks(net, enforce_q_lims=True)

#     net = pp.create_empty_network()
#     b1, b2, l1 = add_grid_connection(net)
#     b3 = pp.create_bus(net, vn_kv=0.4)
##     b4 = pp.create_bus(net, vn_kv=0.4)
##     b5 = pp.create_bus(net, vn_kv=0.4)
#    
#     tidx = pp.create_transformer(net, hv_bus=b2, lv_bus=b3, std_type="0.25 MVA 20/0.4 kV")
#     net.trafo.shift_degree.at[tidx] = 50
#     pp.create_load(net, b3, p_kw=1)
##     create_test_line(net, b3, b4)
##     ln = create_test_line(net, b4, b5)
##     create_test_line(net, b5, b3)
##     pp.create_switch(net, bus=b4, element=ln, et="l", closed=False)
#     pp.runpp(net, calculate_voltage_angles=True, init="dc")
#     pp.runpp(net, calculate_voltage_angles=True, init="results")
    # _build_bus_mpc(net)
    
    

    pytest.main(["test_runpp.py", "-xs"])
#
