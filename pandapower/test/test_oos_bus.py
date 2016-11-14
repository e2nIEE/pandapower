# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a 
# BSD-style license that can be found in the LICENSE file.

import pandapower as pp
from pandapower.test.result_test_network_generator import add_test_oos_bus_with_is_element
from pandapower.test.consistency_checks import runpp_with_consistency_checks
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