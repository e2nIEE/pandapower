# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 15:25:07 2019

@author: Zhenqi Wang
"""

import numpy as np
import pytest

import pandapower as pp
import pandapower.networks as nw
from pandapower.estimation import chi2_analysis, remove_bad_data, estimate
from pandapower.estimation.toolbox import add_virtual_meas_from_loadflow
from copy import deepcopy


#def test_four_bus():
#    net = nw.simple_four_bus_system()
#    pp.runpp(net)
#    add_virtual_meas_from_loadflow(net)
#    vm, va = net.res_bus.vm_pu, net.res_bus.va_degree
#
#    success = estimate(net, algorithm="opt", estimator="lav")
#    assert success
#    
#    success = estimate(net, algorithm="opt", estimator="ql", a=20)
#    assert success

if __name__ == "__main__":
    net = nw.case57()
    pp.runpp(net)
    add_virtual_meas_from_loadflow(net)
    vm, va = net.res_bus.vm_pu, net.res_bus.va_degree

    success = estimate(net, algorithm="torch")
    assert success
    
    net_wls = deepcopy(net)
    success = estimate(net_wls, algorithm="wls")
    
    print()
    
