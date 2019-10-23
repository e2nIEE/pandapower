# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import os

import numpy as np
import pytest

import pandapower as pp
import pandapower.networks as nw
from pandapower.estimation import chi2_analysis, remove_bad_data, estimate
from pandapower.estimation.util import add_virtual_meas_from_loadflow
from pandapower.estimation.observability_analysis import add_virtual_meas_for_unobserved_bus
from copy import deepcopy

np.random.seed(14)

def test_case14():
    # 1. Create network and add virtual measurements
    net = nw.case14()
    pp.runpp(net)
    add_virtual_meas_from_loadflow(net, 0.001, 0.001, 0.001)
    
    # 2. Reducing available measurements
    net.measurement = net.measurement.iloc[np.random.choice(net.measurement.shape[0], size=20), :]
    
    # 3. Estimate without and with virtual measurement
    try:
        estimate(net)
    except:
        add_virtual_meas_for_unobserved_bus(net)
        success = estimate(net)
    assert success
    assert np.allclose(net.res_bus_power_flow.vm_pu, net.res_bus_est.vm_pu, atol=1e-2)
    assert np.allclose(net.res_bus_power_flow.va_degree, net.res_bus_est.va_degree, atol=5e-1)


def test_cigre_mv():
    # 1. Create network and add virtual measurements
    net = nw.create_cigre_network_mv(with_der="all")
    pp.runpp(net)
    add_virtual_meas_from_loadflow(net, 0.001, 0.001, 0.001)
    
    # 2. Reducing available measurements
    net.measurement = net.measurement.iloc[np.random.choice(net.measurement.shape[0], size=35), :]
    
    # 3. Estimate without and with virtual measurement
    try:
        estimate(net)
    except:
        add_virtual_meas_for_unobserved_bus(net)
        success = estimate(net)
    assert success
    assert np.allclose(net.res_bus_power_flow.vm_pu, net.res_bus_est.vm_pu, atol=1e-2)
    assert np.allclose(net.res_bus_power_flow.va_degree, net.res_bus_est.va_degree, atol=5e-1)


if __name__ == '__main__':
    pytest.main([__file__, "-xs"])
