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

if __name__ == "__main__":
    net = nw.case14()
    pp.runpp(net)
    add_virtual_meas_from_loadflow(net)

    success = estimate(net, init='flat', algorithm="irwls", estimator='wls', maximum_iterations=50)
    assert success

    net_wls = deepcopy(net)
    estimate(net_wls)
    assert np.allclose(net_wls.res_bus_est.vm_pu, net.res_bus_est.vm_pu, rtol=0.1)
    assert np.allclose(net_wls.res_bus_est.va_degree, net.res_bus_est.va_degree, rtol=0.1)


