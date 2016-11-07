# -*- coding: utf-8 -*-


import pandapower as pp
from result_test_network_generator import add_test_oos_bus_with_is_element
from consistency_checks import runpp_with_consistency_checks
import pytest


def test_oos_bus():
    net = pp.create_empty_network()
    add_test_oos_bus_with_is_element(net)
    assert runpp_with_consistency_checks(net)
    
#    test for pq-node result
    pp.create_shunt(net, 6, q_kvar = -800)
    assert runpp_with_consistency_checks(net)
    
#   1test for pv-node result
    pp.create_gen(net, 4, p_kw = -500)
    assert runpp_with_consistency_checks(net)
    
    
if __name__ == '__main__':
    pytest.main(["test_oos_bus.py", "-s"])