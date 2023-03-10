# -*- coding: utf-8 -*-
"""
This file included various test network for modelling and simulation different pandapower modules

"""
import pandas as pd
import pandapower as pp
import numpy as np
pd.Series(dtype='float64')

def three_radial_bus_net():
    
        import pandapower as pp
        net = pp.create_empty_network()  # create an empty network
        pp.create_buses(net, nr_buses=16, vn_kv=20, index=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], name=None, type="n", 
                        geodata=[(0,0), (0, -1), (0, -2), (-2, -3), (-2, -4), (-2, -5), (0, -6),
                                 (0, -4.5), (0, -5), (0, -6), (0, -7), (0, -8), (2, -8), (2, -9), (2, -10), (2, -12)])

        # external grids
        pp.create_ext_grid(net, 0, vm_pu=1.0, va_degree=0, s_sc_max_mva=100, s_sc_min_mva=50, rx_max=0.1, rx_min=0.1)
        
        pp.create_lines(net, from_buses=[0,1,2,3,4,5,2,7,8,9,10,9,12,13,14], to_buses=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], length_km=[5,4,4,0.5,0.5,1,2,4,4,5,6,5,3,2,1],std_type="NAYY 4x50 SE",
                     name=None, index=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], df=1., parallel=1)

        net.line["endtemp_degree"] = 250
        
        pp.create_switches(net, buses =  [0,1,2,3,4,5,2,7,8,9,10,9,12,13,14], elements =
                           [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], et = 'l', type ="CB_IDTOC")


        # Define load
        pp.create_loads(net, buses=[5,11,15], p_mw=[5,5,4], q_mvar=[1,1,1], const_z_percent=0, const_i_percent=0, sn_mva=None,
                         name=None, scaling=1., index=[0,1,2])


        return net
  

def dtoc_relay_net(open_loop=True):
    net = pp.create_empty_network()  # create an empty network
    
    #create buses
    pp.create_buses(net, nr_buses=7, vn_kv=20, index=[0,1,2,3,4,5,6], name=None, type="n", 
                    geodata=[(0,0), (0, -1), (-2, -2), (-2, -4), (2, -2), (2, -3), (2, -4)])

    # external grids
    pp.create_ext_grid(net, 0, vm_pu=1.0, va_degree=0, s_sc_max_mva=100, s_sc_min_mva=50, rx_max=0.1, rx_min=0.1)
    
    pp.create_lines(net, from_buses=[0,1,2,1,4,5,3], to_buses=[1,2,3,4,5,6,6], length_km=[2,5,4,4,0.5,0.5,1],std_type="NAYY 4x50 SE",
                 name=None, index=[0,1,2,3,4,5,6], df=1., parallel=1)

    net.line["endtemp_degree"] = 250
    
    pp.create_switches(net, buses =  [0,1,2,3,4,5], elements =
                       [0,1,2,3,4,5], et = 'l', type ="CB_DTOC")
    # Define switches
    
    if open_loop:
        
        pp.create_switches(net, buses =  [3,6], elements =
                           [6,6], et = 'l', type ="CB_DTOC",closed=False)
    else:
        pp.create_switches(net, buses =  [3,6], elements =
                           [6,6], et = 'l', type ="CB_DTOC",closed=True)
        
        
    #define load
    pp.create_loads(net, buses=[3,6], p_mw=[5,2], q_mvar=[1,1], const_z_percent=0, const_i_percent=0, sn_mva=None,
                     name=None, scaling=1., index=[0,1])

    return net

def idmt_relay_net(open_loop=True):
    net = pp.create_empty_network()  # create an empty network
    
    #create buses
    pp.create_buses(net, nr_buses=7, vn_kv=20, index=[0,1,2,3,4,5,6], name=None, type="n", 
                    geodata=[(0,0), (0, -1), (-2, -2), (-2, -4), (2, -2), (2, -3), (2, -4)])

    # external grids
    pp.create_ext_grid(net, 0, vm_pu=1.0, va_degree=0, s_sc_max_mva=100, s_sc_min_mva=50, rx_max=0.1, rx_min=0.1)
    
    pp.create_lines(net, from_buses=[0,1,2,1,4,5,3], to_buses=[1,2,3,4,5,6,6], length_km=[2,5,4,4,0.5,0.5,1],std_type="NAYY 4x50 SE",
                 name=None, index=[0,1,2,3,4,5,6], df=1., parallel=1)

    net.line["endtemp_degree"] = 250
    
    pp.create_switches(net, buses =  [0,1,2,3,4,5], elements =
                       [0,1,2,3,4,5], et = 'l', type ="CB_IDMT")
    # Define switches
    if open_loop:
        
        pp.create_switches(net, buses =  [3,6], elements =
                           [6,6], et = 'l', type ="CB_IDMT",closed=False)
    else:
        pp.create_switches(net, buses =  [3,6], elements =
                           [6,6], et = 'l', type ="CB_IDMT",closed=True)
        
    #define load
    
    pp.create_loads(net, buses=[3,6], p_mw=[5,2], q_mvar=[1,1], const_z_percent=0, const_i_percent=0, sn_mva=None,
                     name=None, scaling=1., index=[0,1])
    return net

def idtoc_relay_net(open_loop=True):
    net = pp.create_empty_network()  # create an empty network
    
    #create buses
    pp.create_buses(net, nr_buses=7, vn_kv=20, index=[0,1,2,3,4,5,6], name=None, type="n", 
                    geodata=[(0,0), (0, -1), (-2, -2), (-2, -4), (2, -2), (2, -3), (2, -4)])

    # external grids
    pp.create_ext_grid(net, 0, vm_pu=1.0, va_degree=0, s_sc_max_mva=100, s_sc_min_mva=50, rx_max=0.1, rx_min=0.1)
    
    pp.create_lines(net, from_buses=[0,1,2,1,4,5,3], to_buses=[1,2,3,4,5,6,6], length_km=[2,5,4,4,0.5,0.5,1],std_type="NAYY 4x50 SE",
                 name=None, index=[0,1,2,3,4,5,6], df=1., parallel=1)

    net.line["endtemp_degree"] = 250
    
    pp.create_switches(net, buses =  [0,1,2,3,4,5], elements =
                       [0,1,2,3,4,5], et = 'l', type ="CB_IDTOC")
    # Define switches
    
    if open_loop:
        
        pp.create_switches(net, buses =  [3,6], elements =
                           [6,6], et = 'l', type ="CB_IDTOC",closed=False)
    else:
        pp.create_switches(net, buses =  [3,6], elements =
                           [6,6], et = 'l', type ="CB_IDTOC",closed=True)
    #define load
    pp.create_loads(net, buses=[3,6], p_mw=[5,2], q_mvar=[1,1], const_z_percent=0, const_i_percent=0, sn_mva=None,
                     name=None, scaling=1., index=[0,1])
    return net