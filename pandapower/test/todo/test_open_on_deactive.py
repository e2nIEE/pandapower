# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a 
# BSD-style license that can be found in the LICENSE file.

import pytest

import pandapower as pp
from pandapower.test.toolbox import add_grid_connection, create_test_line

def test_two_open_switches_on_deactive_line():
    net = pp.create_empty_network()
    b1, b2, l1 = add_grid_connection(net)
    b3 = pp.create_bus(net, vn_kv=20.)
    l2 = create_test_line(net, b2, b3, in_service=False)
    create_test_line(net, b3, b1)
    pp.create_switch(net, b2, l2, et="l", closed=False)
    pp.create_switch(net, b3, l2, et="l", closed=False)
    pp.runpp(net)
    assert net.res_line.i_ka.at[l2] == 0.

if __name__ == "__main__":
    test_two_open_switches_on_deactive_line()

