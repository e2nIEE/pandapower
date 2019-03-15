# -*- coding: utf-8 -*-

# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import os

import numpy as np
import pytest

import pandapower as pp
import pandapower.networks as nw
from pandapower.estimation import chi2_analysis, remove_bad_data, estimate
from copy import deepcopy


def test_2bus():
    # 1. Create network
    net = pp.create_empty_network()
    pp.create_bus(net, name="bus1", vn_kv=1.)
    pp.create_bus(net, name="bus2", vn_kv=1.)
    pp.create_load(net, 1, p_mw=0.0111, q_mvar=0.06)
    pp.create_ext_grid(net, 0, vm_pu=1.038)
    pp.create_line_from_parameters(net, 0, 1, 1, r_ohm_per_km=1,x_ohm_per_km=0.5, c_nf_per_km=0,
                                   max_i_ka=1)

    pp.create_measurement(net, "p", "line", -0.0111, 0.003, 0, 1)  # p12
    pp.create_measurement(net, "q", "line", -0.06, 0.003, 0, 1)    # q12

    pp.create_measurement(net, "v", "bus", 1.038, 0.0001, 0)  # u1
    pp.create_measurement(net, "v", "bus", 1.02, 0.1, 1)   # u2

    # 2. Do state estimation
    success = estimate(net, init='flat', algorithm="lav")
    assert success

if __name__ == '__main__':
    pytest.main([__file__, "-xs"])
    
    net = nw.case14()
