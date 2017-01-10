# -*- coding: utf-8 -*-

__author__ = 'lthurner'

import pytest
import pandapower.shortcircuit as sc
import pandapower as pp
import os

@pytest.fixture
def one_line_one_transformer():
    folder = os.path.abspath(os.path.dirname(pp.__file__))
    net = pp.from_pickle(os.path.join(folder, "test", "shortcircuit", "sc_test_one_line_one_transformer.p"))
    bid = pp.create_bus(net, vn_kv=10.) #add a bus bus switch to test switch compatibility
    pp.create_switch(net, net.ext_grid.bus.iloc[0], bid, et="b")
    net.ext_grid.bus.at[0] = bid
    pp.create_bus(net, vn_kv=0.4, in_service=False) #add out of service bus to test oos indexing
    return net
  
def test_max_10_one_line_one_transformer(one_line_one_transformer):
    net = one_line_one_transformer
    sc.runsc(net, case='max', ip=True, ith=True, lv_tol_percent= 10.)
    assert (abs(net.res_bus_sc.ikss_max_ka.at[0] - 5.773503) <1e-5)
    assert (abs(net.res_bus_sc.ikss_max_ka.at[1] - 14.82619) <1e-5)
    assert (abs(net.res_bus_sc.ikss_max_ka.at[2] - 1.099453) <1e-5)        
    assert (abs(net.res_bus_sc.ip_max_ka.at[0] - 14.25605) <1e-5)
    assert (abs(net.res_bus_sc.ip_max_ka.at[1] - 33.7513) <1e-5)
    assert (abs(net.res_bus_sc.ip_max_ka.at[2] - 1.588343) <1e-5)    
    assert (abs(net.res_bus_sc.ith_max_ka.at[0] - 5.871191) <1e-5)
    assert (abs(net.res_bus_sc.ith_max_ka.at[1] - 14.97527) <1e-5)
    assert (abs(net.res_bus_sc.ith_max_ka.at[2] - 1.100885) <1e-5)

def test_max_6_one_line_one_transformer(one_line_one_transformer):
    net = one_line_one_transformer
    sc.runsc(net, case='max', ip=True, ith=True, lv_tol_percent = 6.)
    assert (abs(net.res_bus_sc.ikss_max_ka.at[0] - 5.773503) <1e-5)
    assert (abs(net.res_bus_sc.ikss_max_ka.at[1] - 14.75419) <1e-5)
    assert (abs(net.res_bus_sc.ikss_max_ka.at[2] - 1.051297) <1e-5)  
      
    assert (abs(net.res_bus_sc.ip_max_ka.at[0] - 14.25605) <1e-5)
    assert (abs(net.res_bus_sc.ip_max_ka.at[1] - 33.59996) <1e-5)
    assert (abs(net.res_bus_sc.ip_max_ka.at[2] - 1.51868) <1e-5)  
    
    assert (abs(net.res_bus_sc.ith_max_ka.at[0] - 5.871191) <1e-5)
    assert (abs(net.res_bus_sc.ith_max_ka.at[1] - 14.90284) <1e-5)
    assert (abs(net.res_bus_sc.ith_max_ka.at[2] - 1.052665) <1e-5)
    
def test_min_10_one_line_one_transformer(one_line_one_transformer):
    net = one_line_one_transformer
    sc.runsc(net, case='min', ip=True, ith=True, lv_tol_percent= 10.)
    assert (abs(net.res_bus_sc.ikss_min_ka.at[0] - 2.309401) <1e-5)
    assert (abs(net.res_bus_sc.ikss_min_ka.at[1] - 11.3267) <1e-5)
    assert (abs(net.res_bus_sc.ikss_min_ka.at[2] - 0.6444859) <1e-5)
        
    assert (abs(net.res_bus_sc.ip_min_ka.at[0] - 5.702418) <1e-5)
    assert (abs(net.res_bus_sc.ip_min_ka.at[1] - 26.01655) <1e-5)
    assert (abs(net.res_bus_sc.ip_min_ka.at[2] - 0.9297185) <1e-5)    
    
    assert (abs(net.res_bus_sc.ith_min_ka.at[0] - 2.348476) <1e-5)
    assert (abs(net.res_bus_sc.ith_min_ka.at[1] - 11.44622) <1e-5)
    assert (abs(net.res_bus_sc.ith_min_ka.at[2] - 0.6453097) <1e-5)

def test_min_6_one_line_one_transformer(one_line_one_transformer):
    net = one_line_one_transformer
    sc.runsc(net, case='min', ip=True, ith=True, lv_tol_percent = 6.)
    assert (abs(net.res_bus_sc.ikss_min_ka.at[0] - 2.309401) <1e-5)
    assert (abs(net.res_bus_sc.ikss_min_ka.at[1] - 11.75072) <1e-5)
    assert (abs(net.res_bus_sc.ikss_min_ka.at[2] - 0.6450874) <1e-5)
        
    assert (abs(net.res_bus_sc.ip_min_ka.at[0] - 5.702418) <1e-5)
    assert (abs(net.res_bus_sc.ip_min_ka.at[1] - 27.00861) <1e-5)
    assert (abs(net.res_bus_sc.ip_min_ka.at[2] - 0.9305832) <1e-5)    
    
    assert (abs(net.res_bus_sc.ith_min_ka.at[0] - 2.348476) <1e-5)
    assert (abs(net.res_bus_sc.ith_min_ka.at[1] - 11.87518) <1e-5)
    assert (abs(net.res_bus_sc.ith_min_ka.at[2] - 0.6459119) <1e-5) 

if __name__ == '__main__':   
    pytest.main(['-xs'])
