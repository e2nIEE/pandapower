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


def test_lp_simple_network():
    pass
    



if __name__ == "__main__":
    
    import pplog as logging
    logger = logging.getLogger()
    #    logging.basicConfig()
    #    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
#    test_irwls_comp_wls()
#    net = nw.simple_four_bus_system()
#    net = nw.case118()
    net = nw.case57()
#    net = nw.create_cigre_network_mv(with_der="pv_wind")
#    net = nw.mv_oberrhein()
    pp.runpp(net)
    add_virtual_meas_from_loadflow(net)

    success = estimate(net, init='flat', algorithm="lp", tolerance=1e-5)

    
#    success = estimate(net, init='flat', algorithm="opt", estimator="lav", max_iterations=20)


#    net_wls = deepcopy(net)
#    estimate(net_wls)
#    assert np.allclose(net_wls.res_bus_est.vm_pu, net.res_bus_est.vm_pu, 1e-6)
#    assert np.allclose(net_wls.res_bus_est.va_degree, net.res_bus_est.va_degree, 1e-6)


