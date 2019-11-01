# -*- coding: utf-8 -*-

# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
import pytest

import pandapower as pp
import pandapower.networks as nw
from pandapower.estimation import estimate
from pandapower.estimation.util import add_virtual_meas_from_loadflow

from pandapower.estimation.algorithm.optimization import OptAlgorithm
from pandapower.estimation.algorithm.base import WLSAlgorithm
from pandapower.estimation.ppc_conversion import pp2eppci
from pandapower.estimation.results import eppci2pp
from copy import deepcopy


def test_simple_net_with_pmu():
    net = pp.create_empty_network()
    
    b1 = pp.create_bus(net, name="bus 1", vn_kv=1., index=1)
    b2 = pp.create_bus(net, name="bus 2", vn_kv=1., index=2)
    b3 = pp.create_bus(net, name="bus 3", vn_kv=1., index=3)
    
    pp.create_load(net, bus=2, p_mw=0.6, q_mvar=1.2)
    pp.create_load(net, bus=3, p_mw=1, q_mvar=-0.7)
    
    pp.create_ext_grid(net, b1, vm_pu=1.006)     # set the slack to bus 1
    
    l1 = pp.create_line_from_parameters(net, 1, 2, 1, r_ohm_per_km=.01, x_ohm_per_km=.03, c_nf_per_km=0., max_i_ka=1)
    l2 = pp.create_line_from_parameters(net, 1, 3, 1, r_ohm_per_km=.02, x_ohm_per_km=.05, c_nf_per_km=0., max_i_ka=1)
    l3 = pp.create_line_from_parameters(net, 2, 3, 2, r_ohm_per_km=.03, x_ohm_per_km=.08, c_nf_per_km=0., max_i_ka=1)
    pp.runpp(net)
    
    # bus voltages
    pp.create_measurement(net, "v", "bus", 1.006, 0.0004, b1)
    pp.create_measurement(net, "v", "bus", .969, 0.0004, b2)
    pp.create_measurement(net, "v", "bus", 1.007, 0.0004, b3)
    
    # bus voltage_angle
    pp.create_measurement(net, "va", "bus", np.deg2rad(-0.74), 0.001, b2)
    pp.create_measurement(net, "va", "bus", np.deg2rad(-2.94), 0.001, b3)
    
    # bus, p, q
    pp.create_measurement(net, "p", "bus", -0.6, 1., b1)
    pp.create_measurement(net, "q", "bus", -1.2, 1., b1)
    pp.create_measurement(net, "p", "bus", -1, 1., b2)
    pp.create_measurement(net, "q", "bus", -0.7, 1., b2)
    
    # line
    pp.create_measurement(net, "p", "line", 0.74, 8., l1, side=1)
    pp.create_measurement(net, "q", "line", 0.97, 8., l1, side=1)
    pp.create_measurement(net, "p", "line", 0.89, 8., l2, side="from")
    pp.create_measurement(net, "q", "line", -0.36, 8., l2, side="from")
    pp.create_measurement(net, "p", "line", 0.13, 8., l3, side="from")
    pp.create_measurement(net, "q", "line", -0.27, 8., l3, side="from")

    
#    # line degree
    pp.create_measurement(net, "i", "line", 0.702, 8., l2, side=1)
#    pp.create_measurement(net, "ia", "line", , 8., l2, side=1)
    
    estimate(net)

    assert np.allclose(net.res_bus_est.vm_pu, net.res_bus.vm_pu, atol=1e-1)
    assert np.allclose(net.res_bus_est.va_degree, net.res_bus.va_degree, atol=1e-1)


if __name__ == '__main__':
    pytest.main([__file__, "-xs"])
