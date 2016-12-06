# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a 
# BSD-style license that can be found in the LICENSE file.

import pandapower as pp

from pandapower.test.consistency_checks import runpp_with_consistency_checks
from pandapower.test.result_test_network_generator import add_test_impedance, add_test_trafo_tap

net = pp.create_empty_network()
add_test_trafo_tap(net)
runpp_with_consistency_checks(net, trafo_model="t", trafo_loading="current", numba=True)

buses = net.bus[net.bus.zone == "test_trafo_tap"]
b2 = buses.index[1]
b3 = buses.index[2]

assert (1.010114175 - net.res_bus.vm_pu.at[b2]) < 1e-6
assert (0.924072090 - net.res_bus.vm_pu.at[b3]) < 1e-6