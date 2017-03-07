# -*- coding: utf-8 -*-

__author__ = 'lthurner'

import pytest
import pandapower.shortcircuit as sc
import pandapower as pp
import os

@pytest.fixture
def one_line_one_generator():
    folder = os.path.abspath(os.path.dirname(pp.__file__))
    net = pp.from_pickle(os.path.join(folder, "test", "shortcircuit", "sc_test_gen.p"))
    bid = pp.create_bus(net, vn_kv=10.)
    pp.create_switch(net, bid, net.gen.bus.iloc[0], et="b")
    net.gen.bus.iloc[0] = bid
    pp.create_bus(net, vn_kv=0.4, in_service=False)
    return net    
    
def test_max_gen(one_line_one_generator):
    net = one_line_one_generator
    sc.runsc(net, case="max")
    assert abs(net.res_bus_sc.ikss_max_ka.at[0] - 1.5395815) < 1e-7
    assert abs(net.res_bus_sc.ikss_max_ka.at[1] - 1.5083952) < 1e-7

def test_min_gen(one_line_one_generator):
    net = one_line_one_generator
    sc.runsc(net, case="min")
    assert abs(net.res_bus_sc.ikss_min_ka.at[0] - 1.3996195) < 1e-7
    assert abs(net.res_bus_sc.ikss_min_ka.at[1] - 1.3697407) < 1e-7    

if __name__ == '__main__':   
    pytest.main(['-xs'])
