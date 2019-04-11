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
#    pytest.main([__file__, "-xs"])
    from pandapower.estimation.algorithm.optimization import OptAlgorithm
    from pandapower.estimation.algorithm.wls import WLSAlgorithm
    from pandapower.estimation.algorithm.irwls import IRWLSAlgorithm
    from pandapower.estimation.ppc_conversion import pp2eppci
    from pandapower.estimation.results import eppci2pp

    net = nw.case14()
    pp.runpp(net)
    add_virtual_meas_from_loadflow(net)
    
    net_wls = deepcopy(net)
    estimate(net_wls)
    
    net_lav = deepcopy(net)
    estimate(net_lav, algorithm='lp')

    net, ppc, eppci = pp2eppci(net)
#    estimation_wls = WLSAlgorithm(1e-3, 3)
    estimation_ir = IRWLSAlgorithm(1e-6, 15)

#    eppci = estimation_wls.estimate(eppci)
    eppci = estimation_ir.estimate(eppci, algorithm="irwls", estimator='ql', a=1)
    assert estimation_ir.successful
    net = eppci2pp(net, ppc, eppci)

    
