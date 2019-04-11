# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import os

import numpy as np
import pytest

import pandapower as pp
import pandapower.networks as nw
from pandapower.estimation import estimate
from pandapower.estimation.toolbox import add_virtual_meas_from_loadflow
from copy import deepcopy

def test_irwls_comp_wls():
    for net in [nw.case14(), nw.case57(), nw.case118()]:
        pp.runpp(net)
        add_virtual_meas_from_loadflow(net)

        success = estimate(net, init='flat', algorithm="irwls", estimator='wls')
        assert success

        net_wls = deepcopy(net)
        estimate(net_wls)
        assert np.allclose(net_wls.res_bus_est.vm_pu, net.res_bus_est.vm_pu, 1e-6)
        assert np.allclose(net_wls.res_bus_est.va_degree, net.res_bus_est.va_degree, 1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
