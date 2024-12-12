# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
import pytest

from pandapower.run import runpp
from pandapower.networks.power_system_test_cases import case30
from pandapower.estimation import StateEstimation
from pandapower.estimation.util import add_virtual_meas_from_loadflow


def test_recycle_case30():
    net = case30()
    runpp(net)
    add_virtual_meas_from_loadflow(net)
    se = StateEstimation(net, recycle=True)
    se.estimate()
    assert np.allclose(net.res_bus.vm_pu, net.res_bus_est.vm_pu, atol=1e-2)
    assert np.allclose(net.res_bus.va_degree, net.res_bus_est.va_degree, atol=5e-1)  

    # Run SE again
    net.load.p_mw -= 10
    runpp(net)
    add_virtual_meas_from_loadflow(net)
    assert se.estimate()
    assert np.allclose(net.res_bus.vm_pu, net.res_bus_est.vm_pu, atol=1e-2)
    assert np.allclose(net.res_bus.va_degree, net.res_bus_est.va_degree, atol=1)  

if __name__ == '__main__':
    pytest.main([__file__, "-xs"])
