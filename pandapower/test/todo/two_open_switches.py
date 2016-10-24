# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 16:05:30 2016

@author: thurner
"""

from pandapower.test.toolbox import add_grid_connection, create_test_line
import pandapower as pp

def test_two_open_switches():
    net = pp.create_empty_network()
    b1, b2, l1 = add_grid_connection(net)
    b3 = pp.create_bus(net, vn_kv=20.)
    l2 = create_test_line(net, b2, b3)
    create_test_line(net, b3, b1)    
    pp.create_switch(net, b2, l2, et="l", closed=False)
    pp.create_switch(net, b3, l2, et="l", closed=False)
    pp.runpp(net)
    assert net.res_line.i_ka.at[l2] == 0.