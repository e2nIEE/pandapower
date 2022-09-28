# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 14:31:23 2021

@author: plyta
"""
import pandas as pd
import pandapower as pp
import numpy as np


def load_4bus_net(open_loop = False):
    net = pp.create_empty_network() #create an empty network
    
    
    #busbars and nodes
    bus0 = pp.create_bus(net,name = "Bus_extgrid", vn_kv = 20, type = "b")
    bus1 = pp.create_bus(net,name = "Bus_line1", vn_kv = 20, type = "b")
    bus2 = pp.create_bus(net,name = "Bus_load", vn_kv = 20, type = "b")
    bus3 = pp.create_bus(net,name = "Bus_line2", vn_kv = 20, type = "b")
    
    
    #external grids
    pp.create_ext_grid(net, bus0, vm_pu = 1.0, va_degree = 0, s_sc_max_mva = 100, s_sc_min_mva = 50, rx_max = 0.1, rx_min = 0.1)
    
    #lines
    #line1 = pp.create_line(net, bus0, bus1, length_km = 5, std_type = "NA2XS2Y 1x185 RM/25 12/20 kV", index = 1)
    #line2 = pp.create_line(net, bus1, bus2, length_km = 4, std_type = "NA2XS2Y 1x185 RM/25 12/20 kV", index = 2)
    #line3 = pp.create_line(net, bus1, bus3, length_km = 4, std_type = "NA2XS2Y 1x185 RM/25 12/20 kV", index = 3)
    
    line1 = pp.create_line_from_parameters(net, bus0, bus1, length_km = 5, index = 1, r_ohm_per_km = 0.169, x_ohm_per_km = 0.118438, c_nf_per_km = 273, max_i_ka = 0.361)
    line2 = pp.create_line_from_parameters(net, bus1, bus2, length_km = 4, index = 2, r_ohm_per_km = 0.256, x_ohm_per_km = 0.126606, c_nf_per_km = 235, max_i_ka = 0.286)
    line3 = pp.create_line_from_parameters(net, bus1, bus3, length_km = 4, index = 3, r_ohm_per_km = 0.256, x_ohm_per_km = 0.126606, c_nf_per_km = 235, max_i_ka = 0.286)
    line4 = pp.create_line_from_parameters(net, bus3, bus2, length_km = 0.5, index =4, r_ohm_per_km = 0.256, x_ohm_per_km = 0.126606, c_nf_per_km = 235, max_i_ka = 0.286)
    
    #line1 = pp.create_line(net, bus0, bus1, length_km = 5, std_type = "NA2XS2Y 1x185 RM/25 12/20 kV", index = 1)
    
    net.line["endtemp_degree"] = 250
    
    
    #switches
    sw1 = pp.create_switch(net, bus0, line1, et="l", type = "CB_dir", closed = True, index = 0)
    sw2 = pp.create_switch(net, bus1, line2, et="l", type = "CB_dir", closed = True, index = 1)
    sw3 = pp.create_switch(net, bus1, line3, et="l", type = "CB_dir", closed = True, index = 2)
    # sw6 = pp.create_switch(net, bus3, line3, et="l", type = "CB", closed = True)
    
    if open_loop:
        sw4 = pp.create_switch(net, bus2, line4, et="l", type = "CB_dir", closed = False, index = 3)
        sw5 = pp.create_switch(net, bus3, line4, et="l", type = "CB_dir", closed = False, index = 4)
    else:
        sw4 = pp.create_switch(net, bus2, line4, et="l", type = "CB_dir", closed = True, index = 3)
        sw5 = pp.create_switch(net, bus3, line4, et="l", type = "CB_dir", closed = True, index = 4)
    
    #load
    pp.create_load(net, bus2, p_mw = 5, q_mvar= 0, scaling = 1, name="load 1")
    #pp.create_sgen(net, bus3, p_mw =2, q_mvar=0, sn_mva =2)
            
    #geodata Zeilen initialisieren
    net.bus_geodata.loc[0] = None
    net.bus_geodata.loc[1] = None
    net.bus_geodata.loc[2] = None
    net.bus_geodata.loc[3] = None
    
    
    #Knoten neue Koordinaten für Plot zuweisen
    net.bus_geodata.x.at[0] = 0
    net.bus_geodata.x.at[1] = 0
    net.bus_geodata.x.at[2] = -1
    net.bus_geodata.x.at[3] = 1
    
    
    net.bus_geodata.y.at[0] = 1
    net.bus_geodata.y.at[1] = 0
    net.bus_geodata.y.at[2] = -1
    net.bus_geodata.y.at[3] = -1
    
    return net
def load_4bus_net_oc_doc(open_loop = False):
    net = pp.create_empty_network() #create an empty network
    
    
    #busbars and nodes
    bus0 = pp.create_bus(net,name = "Bus_extgrid", vn_kv = 20, type = "b")
    bus1 = pp.create_bus(net,name = "Bus_line1", vn_kv = 20, type = "b")
    bus2 = pp.create_bus(net,name = "Bus_load", vn_kv = 20, type = "b")
    bus3 = pp.create_bus(net,name = "Bus_line2", vn_kv = 20, type = "b")
    
    
    #external grids
    pp.create_ext_grid(net, bus0, vm_pu = 1.0, va_degree = 0, s_sc_max_mva = 100, s_sc_min_mva = 50, rx_max = 0.1, rx_min = 0.1)
    
    #lines
    #line1 = pp.create_line(net, bus0, bus1, length_km = 5, std_type = "NA2XS2Y 1x185 RM/25 12/20 kV", index = 1)
    #line2 = pp.create_line(net, bus1, bus2, length_km = 4, std_type = "NA2XS2Y 1x185 RM/25 12/20 kV", index = 2)
    #line3 = pp.create_line(net, bus1, bus3, length_km = 4, std_type = "NA2XS2Y 1x185 RM/25 12/20 kV", index = 3)
    
    line1 = pp.create_line_from_parameters(net, bus0, bus1, length_km = 5, index = 1, r_ohm_per_km = 0.169, x_ohm_per_km = 0.118438, c_nf_per_km = 273, max_i_ka = 0.361)
    line2 = pp.create_line_from_parameters(net, bus1, bus2, length_km = 4, index = 2, r_ohm_per_km = 0.256, x_ohm_per_km = 0.126606, c_nf_per_km = 235, max_i_ka = 0.286)
    line3 = pp.create_line_from_parameters(net, bus1, bus3, length_km = 4, index = 3, r_ohm_per_km = 0.256, x_ohm_per_km = 0.126606, c_nf_per_km = 235, max_i_ka = 0.286)
    line4 = pp.create_line_from_parameters(net, bus3, bus2, length_km = 0.5, index =4, r_ohm_per_km = 0.256, x_ohm_per_km = 0.126606, c_nf_per_km = 235, max_i_ka = 0.286)
    
    #line1 = pp.create_line(net, bus0, bus1, length_km = 5, std_type = "NA2XS2Y 1x185 RM/25 12/20 kV", index = 1)
    
    net.line["endtemp_degree"] = 250
    
    
    #switches
    sw1 = pp.create_switch(net, bus0, line1, et="l", type = "CB_non_dir", closed = True, index = 0)
    sw2 = pp.create_switch(net, bus1, line2, et="l", type = "CB_non_dir", closed = True, index = 1)
    sw3 = pp.create_switch(net, bus1, line3, et="l", type = "CB_non_dir", closed = True, index = 2)
    # sw6 = pp.create_switch(net, bus3, line3, et="l", type = "CB", closed = True)
    
    if open_loop:
        sw4 = pp.create_switch(net, bus2, line4, et="l", type = "CB_dir", closed = False, index = 3)
        sw5 = pp.create_switch(net, bus3, line4, et="l", type = "CB_dir", closed = False, index = 4)
    else:
        sw4 = pp.create_switch(net, bus2, line4, et="l", type = "CB_dir", closed = True, index = 3)
        sw5 = pp.create_switch(net, bus3, line4, et="l", type = "CB_dir", closed = True, index = 4)
    
    #load
    pp.create_load(net, bus2, p_mw = 5, q_mvar= 0, scaling = 1, name="load 1")
            
    #geodata Zeilen initialisieren
    net.bus_geodata.loc[0] = None
    net.bus_geodata.loc[1] = None
    net.bus_geodata.loc[2] = None
    net.bus_geodata.loc[3] = None
    
    
    #Knoten neue Koordinaten für Plot zuweisen
    net.bus_geodata.x.at[0] = 0
    net.bus_geodata.x.at[1] = 0
    net.bus_geodata.x.at[2] = -1
    net.bus_geodata.x.at[3] = 1
    
    
    net.bus_geodata.y.at[0] = 1
    net.bus_geodata.y.at[1] = 0
    net.bus_geodata.y.at[2] = -1
    net.bus_geodata.y.at[3] = -1
    
    return net

def load_distance_test_grid():
    net = pp.create_empty_network() #create an empty network
    
    bus0 = pp.create_bus(net,name = "Bus0", vn_kv = 110, type = "b")
    bus1 = pp.create_bus(net,name = "Bus1", vn_kv = 110, type = "b")
    bus2 = pp.create_bus(net,name = "Bus2", vn_kv = 110, type = "b")
    bus3 = pp.create_bus(net,name = "Bus3", vn_kv = 110, type = "b")
    bus4 = pp.create_bus(net,name = "Bus4", vn_kv = 110, type = "b")
    
    pp.create_ext_grid(net, bus0, vm_pu = 1.0, va_degree = 0, s_sc_max_mva = 2500, s_sc_min_mva = 2000, rx_max = 0.1, rx_min = 0.1)
    
    line1 = pp.create_line_from_parameters(net, bus0, bus1, length_km = 5, index = 1, r_ohm_per_km = 0.1021, x_ohm_per_km = 0.157079, c_nf_per_km = 130, max_i_ka = 0.461)
    line2 = pp.create_line_from_parameters(net, bus1, bus3, length_km = 5, index = 2, r_ohm_per_km = 0.1021, x_ohm_per_km = 0.157079, c_nf_per_km = 130, max_i_ka = 0.461)
    line3 = pp.create_line_from_parameters(net, bus0, bus2, length_km = 5, index = 3, r_ohm_per_km = 0.1021, x_ohm_per_km = 0.157079, c_nf_per_km = 130, max_i_ka = 0.461)
    line4 = pp.create_line_from_parameters(net, bus2, bus4, length_km = 5, index = 4, r_ohm_per_km = 0.1021, x_ohm_per_km = 0.157079, c_nf_per_km = 130, max_i_ka = 0.461)
    
    pp.create_switch(net, bus0, line1, et="l", type = "CB", closed = True, index = 1)
    pp.create_switch(net, bus1, line2, et="l", type = "CB", closed = True, index = 2)
    pp.create_switch(net, bus0, line3, et="l", type = "CB", closed = True, index = 3)
    pp.create_switch(net, bus2, line4, et="l", type = "CB", closed = True, index = 4)
    pp.create_switch(net, bus3, bus4, et="b", type = "DS", closed = True, index = 5)
    
    pp.create_load(net, bus3, p_mw = 10, q_mvar= 0, scaling = 1, name="load 1")
    pp.create_load(net, bus4, p_mw = 10, q_mvar= 0, scaling = 1, name="load 2")
    
    net.line['endtemp_degree'] = 250
    #geodata Zeilen initialisieren
    #net.bus_geodata.loc[0] = None
    #net.bus_geodata.loc[1] = None
    #net.bus_geodata.loc[2] = None
    #net.bus_geodata.loc[3] = None
    #net.bus_geodata.loc[4] = None
    
    
    #Knoten neue Koordinaten für Plot zuweisen
    net.bus_geodata.x.at[0] = 0
    net.bus_geodata.x.at[1] = -1
    net.bus_geodata.x.at[2] = 1
    net.bus_geodata.x.at[3] = -1
    net.bus_geodata.x.at[4] = 1
    
    net.bus_geodata.y.at[0] = 0
    net.bus_geodata.y.at[1] = -1
    net.bus_geodata.y.at[2] = -1
    net.bus_geodata.y.at[3] = -2
    net.bus_geodata.y.at[4] = -2
    
    return net

def load_grid_with_infeed_110kv(infeed_s_sc_max_mva = 100, infeed_s_sc_min_mva = 100):
    net = pp.create_empty_network() #create an empty network

    bus0 = pp.create_bus(net,name = "Bus0", vn_kv = 110, type = "b")
    bus1 = pp.create_bus(net,name = "Bus1", vn_kv = 110, type = "b")
    bus2 = pp.create_bus(net,name = "Bus2", vn_kv = 110, type = "b")
    
    pp.create_ext_grid(net, bus0, vm_pu = 1.0, va_degree = 0, s_sc_max_mva = 500, s_sc_min_mva = 500, rx_max = 0.1, rx_min = 0.1)
    pp.create_ext_grid(net, bus1, vm_pu = 1.0, va_degree = 0, s_sc_max_mva = infeed_s_sc_max_mva, s_sc_min_mva = infeed_s_sc_min_mva, rx_max = 0.1, rx_min = 0.1)
    
    line0 = pp.create_line_from_parameters(net, bus0, bus1, length_km = 5, index = 0, r_ohm_per_km = 0.1021, x_ohm_per_km = 0.157079, c_nf_per_km = 130, max_i_ka = 0.461)
    line1 = pp.create_line_from_parameters(net, bus1, bus2, length_km = 5, index = 1, r_ohm_per_km = 0.1021, x_ohm_per_km = 0.157079, c_nf_per_km = 130, max_i_ka = 0.461)
    
    pp.create_switch(net, bus0, line0, et="l", type = "CB", closed = True, index = 0)
    pp.create_switch(net, bus1, line0, et="l", type = "CB", closed = True, index = 1)
    pp.create_switch(net, bus1, line1, et="l", type = "CB", closed = True, index = 2)
    
    pp.create_load(net, bus2, p_mw = 10, q_mvar= 0, scaling = 1, name="load 1")
    
    net.line['endtemp_degree'] = 250
    
    #geodata Zeilen initialisieren
    net.bus_geodata.loc[0] = None
    net.bus_geodata.loc[1] = None
    net.bus_geodata.loc[2] = None

    net.bus_geodata.x.at[0] = 0
    net.bus_geodata.x.at[1] = 1
    net.bus_geodata.x.at[2] = 0
    
    net.bus_geodata.y.at[0] = 0
    net.bus_geodata.y.at[1] = -1
    net.bus_geodata.y.at[2] = -2

    return net

def load_mv_oberrhein_grid():
    import pandapower.networks as nw

    net = nw.mv_oberrhein()
    net.ext_grid['s_sc_max_mva'] = 5000
    net.ext_grid['s_sc_min_mva'] = 5000
    net.ext_grid['rx_max'] = 0.1
    net.ext_grid['rx_min'] = 0.1
    net.line['endtemp_degree'] = 80
    
    net.sgen['k'] = 1.2  # needs to reconsider
    
    net.switch.type = "CB"
    
    net.sgen['rx'] = 0.1
    return net

def load_balzer_grid(current_angle = 90, k = 1.2, Ikss_ext_grid_ka = 20, with_sgens = True):
    net = pp.create_empty_network() #create an empty network
    
    vn_kv_grid = 110
    
    bus1 = pp.create_bus(net,name = "Bus1", vn_kv = vn_kv_grid, type = "b", index = 1)
    bus2 = pp.create_bus(net,name = "Bus2", vn_kv = vn_kv_grid, type = "b", index = 2)
    bus3 = pp.create_bus(net,name = "Bus3", vn_kv = vn_kv_grid, type = "b", index = 3)
    bus4 = pp.create_bus(net,name = "Bus4", vn_kv = vn_kv_grid, type = "b", index = 4)
    
    S_sc = np.sqrt(3) * vn_kv_grid * Ikss_ext_grid_ka
    pp.create_ext_grid(net, bus1, vm_pu = 1.0, va_degree = 0, s_sc_max_mva = S_sc, s_sc_min_mva = S_sc, rx_max = 0.1, rx_min = 0.1)
    
    R_l_per_km = 0.120
    X_l_per_km = 0.393
    
    l12 = 100
    l13 = 50
    l23 = 50
    l34 = 25
    
    line12 = pp.create_line_from_parameters(net, bus1, bus2, length_km = l12, index = 12, r_ohm_per_km = R_l_per_km, x_ohm_per_km = X_l_per_km, c_nf_per_km = 0, max_i_ka = 0.531)
    line13 = pp.create_line_from_parameters(net, bus1, bus3, length_km = l13, index = 13, r_ohm_per_km = R_l_per_km, x_ohm_per_km = X_l_per_km, c_nf_per_km = 0, max_i_ka = 0.531)
    line23 = pp.create_line_from_parameters(net, bus2, bus3, length_km = l23, index = 23, r_ohm_per_km = R_l_per_km, x_ohm_per_km = X_l_per_km, c_nf_per_km = 0, max_i_ka = 0.531)
    line34 = pp.create_line_from_parameters(net, bus3, bus4, length_km = l34, index = 34, r_ohm_per_km = R_l_per_km, x_ohm_per_km = X_l_per_km, c_nf_per_km = 0, max_i_ka = 0.531)
    net.line['endtemp_degree'] = 180
    
    sw1 = pp.create_switch(net, bus1, line12, et="l", type = "CB", closed = True, index = 1)
    sw2 = pp.create_switch(net, bus2, line12, et="l", type = "CB", closed = True, index = 2)
    sw3 = pp.create_switch(net, bus1, line13, et="l", type = "CB", closed = True, index = 3)
    sw4 = pp.create_switch(net, bus3, line13, et="l", type = "CB", closed = True, index = 4)
    sw5 = pp.create_switch(net, bus3, line23, et="l", type = "CB", closed = True, index = 5)
    sw6 = pp.create_switch(net, bus2, line23, et="l", type = "CB", closed = True, index = 6)
    sw7 = pp.create_switch(net, bus3, line34, et="l", type = "CB", closed = True, index = 7)
    sw8 = pp.create_switch(net, bus4, line34, et="l", type = "CB", closed = True, index = 8)
    
    if with_sgens:
        Pu2 = 100
        Pu3 = 50
        Pu4 = 50
        k_sgen = k
        
        pp.create_sgen(net, bus2, p_mw = Pu2, sn_mva = Pu2, k = k_sgen, current_source=True)
        pp.create_sgen(net, bus3, p_mw = Pu3, sn_mva = Pu3, k = k_sgen, current_source=True)
        pp.create_sgen(net, bus4, p_mw = Pu4, sn_mva = Pu4, k = k_sgen, current_source=True)
        
        net.sgen["current_angle"] = current_angle
        
        net.sgen["s_sc_max_mva"] = net.sgen.k*net.sgen.sn_mva
    
    net.bus_geodata.loc[1] = None
    net.bus_geodata.loc[2] = None
    net.bus_geodata.loc[3] = None
    net.bus_geodata.loc[4] = None
    
    net.bus_geodata.x.at[1] = 0
    net.bus_geodata.x.at[2] = 0.5
    net.bus_geodata.x.at[3] = 0.25
    net.bus_geodata.x.at[4] = 0
    
    net.bus_geodata.y.at[1] = 0
    net.bus_geodata.y.at[2] = 0
    net.bus_geodata.y.at[3] = -0.5
    net.bus_geodata.y.at[4] = -0.5
    
    return net


#test grid for non directional

def load_6bus_net_non_directional(open_loop=False):
    import pandapower as pp
    net = pp.create_empty_network()  # create an empty network

    # busbars and nodes
    bus0 = pp.create_bus(net, name="Bus_extgrid", vn_kv=20, type="b")
    bus1 = pp.create_bus(net, name="Bus_line0", vn_kv=20, type="n")
    bus2 = pp.create_bus(net, name="Bus_line1", vn_kv=20, type="n")
    bus3 = pp.create_bus(net, name="Bus_line2", vn_kv=20, type="n")
    bus4 = pp.create_bus(net, name="Bus_load1", vn_kv=20, type="n")
    bus5 = pp.create_bus(net, name="Bus_line3", vn_kv=20, type="n")
    bus6 = pp.create_bus(net, name="Bus_load2", vn_kv=20, type="n")
    
    #new bus test
    bus7 = pp.create_bus(net, name="Bus_line4", vn_kv=20, type="n")
    bus8 = pp.create_bus(net, name="Bus_line5", vn_kv=20, type="n")
    
    #bus9 = pp.create_bus(net, name="load1", vn_kv=20, type="n")


    # external grids
    pp.create_ext_grid(net, bus0, vm_pu=1.0, va_degree=0, s_sc_max_mva=100, s_sc_min_mva=50, rx_max=0.1, rx_min=0.1)

    # lines
    # line1 = pp.create_line(net, bus0, bus1, length_km = 5, std_type = "NA2XS2Y 1x185 RM/25 12/20 kV", index = 1)
    # line2 = pp.create_line(net, bus1, bus2, length_km = 4, std_type = "NA2XS2Y 1x185 RM/25 12/20 kV", index = 2)
    # line3 = pp.create_line(net, bus1, bus3, length_km = 4, std_type = "NA2XS2Y 1x185 RM/25 12/20 kV", index = 3)

    line0 = pp.create_line_from_parameters(net, bus0, bus1, length_km=2, index=0, r_ohm_per_km=0.169,
                                           x_ohm_per_km=0.118438, c_nf_per_km=273, max_i_ka=0.361)
    line1 = pp.create_line_from_parameters(net, bus1, bus2, length_km=5, index=1, r_ohm_per_km=0.169,
                                           x_ohm_per_km=0.118438, c_nf_per_km=273, max_i_ka=0.361)

    line2 = pp.create_line_from_parameters(net, bus2, bus3, length_km=4, index=2, r_ohm_per_km=0.256,
                                           x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)

    line3 = pp.create_line_from_parameters(net, bus1, bus4, length_km=4, index=3, r_ohm_per_km=0.256,
                                           x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)

    line4 = pp.create_line_from_parameters(net, bus4, bus5, length_km=0.5, index=4, r_ohm_per_km=0.256,
                                           x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)

    line5 = pp.create_line_from_parameters(net, bus5, bus6, length_km=0.5, index=5, r_ohm_per_km=0.256,
                                           x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)
        
    # new line test
    line6 = pp.create_line_from_parameters(net, bus3, bus7, length_km=4, index=6, r_ohm_per_km=0.256,
                                           x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)
    
    line7 = pp.create_line_from_parameters(net, bus6, bus8, length_km=4, index=7, r_ohm_per_km=0.256,
                                           x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)
    
    line8 = pp.create_line_from_parameters(net, bus7, bus8, length_km=4, index=8, r_ohm_per_km=0.256,
                                           x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)
    

    net.line["endtemp_degree"] = 250

    # switches
    swo=pp.create_switch(net, bus0, line0, et="l", type="CB_non_dir", closed=True, index=0)
    sw1 = pp.create_switch(net, bus1, line1, et="l", type="CB_non_dir", closed=True, index=1)

    sw2 = pp.create_switch(net, bus2, line2, et="l", type = "CB_non_dir", closed = True, index=2)

    sw3 = pp.create_switch(net, bus1, line3, et="l", type="CB_non_dir", closed=True, index=3)

    sw4 = pp.create_switch(net, bus4, line4, et="l", type = "CB_non_dir", closed = True, index=4)

    sw5 = pp.create_switch(net, bus5, line5, et="l", type="CB_non_dir", closed=True, index=5)
    
    sw6 = pp.create_switch(net, bus3, line6, et="l", type="CB_non_dir", closed=True, index=6)
    
    sw7 = pp.create_switch(net, bus6, line7, et="l", type="CB_non_dir", closed=True, index=7)
    
    # bus bar connection
    if open_loop:
        
        sw8 = pp.create_switch(net, bus7, line8, et="l", type="CB_non_dir", closed=False, index=8)
        sw9 = pp.create_switch(net, bus8, line8, et="l", type="CB_non_dir", closed=False, index=9)
    else:
        sw8 = pp.create_switch(net, bus7, line8, et="l", type="CB_non_dir", closed=True, index=8)
        sw9 = pp.create_switch(net, bus8, line8, et="l", type="CB_non_dir", closed=True, index=9)
    
    
    pp.create_load(net, bus7, p_mw=5, q_mvar=1, scaling=1, name= "load 1")


    # Define load


#modifies load
    #pp.create_load(net, bus8, p_mw=4, q_mvar=1, scaling=1, name="load 2")

    # geodata Zeilen initialisieren
    net.bus_geodata.loc[0] = None
    net.bus_geodata.loc[1] = None
    net.bus_geodata.loc[2] = None
    net.bus_geodata.loc[3] = None
    net.bus_geodata.loc[4] = None
    net.bus_geodata.loc[5] = None
    net.bus_geodata.loc[6] = None
    
    #test
    net.bus_geodata.loc[7] = None
    net.bus_geodata.loc[8] = None

    

    # Knoten neue Koordinaten für Plot zuweisen
    net.bus_geodata.x.at[0] = 0
    net.bus_geodata.x.at[1] = 0
    net.bus_geodata.x.at[2] = -2
    net.bus_geodata.x.at[3] = -2
    
    # test
    net.bus_geodata.x.at[7] = -2
    net.bus_geodata.x.at[8] = 2

    
    net.bus_geodata.x.at[4] = 2
    net.bus_geodata.x.at[5] = 2
    net.bus_geodata.x.at[6] = 2

################################
    net.bus_geodata.y.at[0] = 0
    net.bus_geodata.y.at[1] = -1
    net.bus_geodata.y.at[2] = -2
    net.bus_geodata.y.at[3] = -4
    
    #test
    net.bus_geodata.y.at[7] = -6
    net.bus_geodata.y.at[8] =  -6


    net.bus_geodata.y.at[4] = -2
    net.bus_geodata.y.at[5] = -3
    net.bus_geodata.y.at[6] = -4


    return net
###  Test grid developed by Arjun:
    
    
def load_6bus_net_directional(open_loop=False):
    import pandapower as pp
    net = pp.create_empty_network()  # create an empty network

    # busbars and nodes
    bus0 = pp.create_bus(net, name="Bus_extgrid", vn_kv=20, type="b")
    bus1 = pp.create_bus(net, name="Bus_line0", vn_kv=20, type="n")
    bus2 = pp.create_bus(net, name="Bus_line1", vn_kv=20, type="n")
    bus3 = pp.create_bus(net, name="Bus_line2", vn_kv=20, type="n")
    bus4 = pp.create_bus(net, name="Bus_load1", vn_kv=20, type="n")
    bus5 = pp.create_bus(net, name="Bus_line3", vn_kv=20, type="n")
    bus6 = pp.create_bus(net, name="Bus_load2", vn_kv=20, type="n")
    
    #new bus test
    bus7 = pp.create_bus(net, name="Bus_line4", vn_kv=20, type="n")
    bus8 = pp.create_bus(net, name="Bus_line5", vn_kv=20, type="n")
    
    #bus9 = pp.create_bus(net, name="load1", vn_kv=20, type="n")


    # external grids
    pp.create_ext_grid(net, bus0, vm_pu=1.0, va_degree=0, s_sc_max_mva=100, s_sc_min_mva=50, rx_max=0.1, rx_min=0.1)

    # lines
    # line1 = pp.create_line(net, bus0, bus1, length_km = 5, std_type = "NA2XS2Y 1x185 RM/25 12/20 kV", index = 1)
    # line2 = pp.create_line(net, bus1, bus2, length_km = 4, std_type = "NA2XS2Y 1x185 RM/25 12/20 kV", index = 2)
    # line3 = pp.create_line(net, bus1, bus3, length_km = 4, std_type = "NA2XS2Y 1x185 RM/25 12/20 kV", index = 3)

    line0 = pp.create_line_from_parameters(net, bus0, bus1, length_km=2, index=0, r_ohm_per_km=0.169,
                                           x_ohm_per_km=0.118438, c_nf_per_km=273, max_i_ka=0.361)
    line1 = pp.create_line_from_parameters(net, bus1, bus2, length_km=5, index=1, r_ohm_per_km=0.169,
                                           x_ohm_per_km=0.118438, c_nf_per_km=273, max_i_ka=0.361)

    line2 = pp.create_line_from_parameters(net, bus2, bus3, length_km=4, index=2, r_ohm_per_km=0.256,
                                           x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)

    line3 = pp.create_line_from_parameters(net, bus1, bus4, length_km=4, index=3, r_ohm_per_km=0.256,
                                           x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)

    line4 = pp.create_line_from_parameters(net, bus4, bus5, length_km=5, index=4, r_ohm_per_km=0.256,
                                           x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)

    line5 = pp.create_line_from_parameters(net, bus5, bus6, length_km=0.5, index=5, r_ohm_per_km=0.256,
                                           x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)

    #bus bar connection line
    #line6 = pp.create_line_from_parameters(net, bus7, bus8, length_km=2, index=6, r_ohm_per_km=0.256,
    #                                       x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)
    
    
    # new line test
    line6 = pp.create_line_from_parameters(net, bus3, bus7, length_km=4, index=6, r_ohm_per_km=0.256,
                                           x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)
    
    line7 = pp.create_line_from_parameters(net, bus6, bus8, length_km=4, index=7, r_ohm_per_km=0.256,
                                           x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)
    
    line8 = pp.create_line_from_parameters(net, bus7, bus8, length_km=4, index=8, r_ohm_per_km=0.256,
                                           x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)
    

    net.line["endtemp_degree"] = 250

    # switches
    swo=pp.create_switch(net, bus0, line0, et="l", type="CB_non_dir", closed=True, index=0)
    sw1 = pp.create_switch(net, bus1, line1, et="l", type="CB_non_dir", closed=True, index=1)

    sw2 = pp.create_switch(net, bus2, line2, et="l", type = "CB_non_dir", closed = True, index=2)

    sw3 = pp.create_switch(net, bus1, line3, et="l", type="CB_non_dir", closed=True, index=3)

    sw4 = pp.create_switch(net, bus4, line4, et="l", type = "CB_non_dir", closed = True, index=4)

    sw5 = pp.create_switch(net, bus5, line5, et="l", type="CB_non_dir", closed=True, index=5)
    
    sw6 = pp.create_switch(net, bus3, line6, et="l", type="CB_non_dir", closed=True, index=6)
    
    sw7 = pp.create_switch(net, bus6, line7, et="l", type="CB_non_dir", closed=True, index=7)
    
    # bus bar connection
    if open_loop:
        
        sw8 = pp.create_switch(net, bus7, line8, et="l", type="CB_dir", closed=False, index=8)
        sw9 = pp.create_switch(net, bus8, line8, et="l", type="CB_dir", closed=False, index=9)
    else:
        sw8 = pp.create_switch(net, bus7, line8, et="l", type="CB_dir", closed=True, index=8)
        sw9 = pp.create_switch(net, bus8, line8, et="l", type="CB_dir", closed=True, index=9)
    
    #sw9 = pp.create_switch(net, bus6, line8, et="l", type="CB_non_dir", closed=True, index=8)
    
    
    pp.create_load(net, bus7, p_mw=5, q_mvar=1, scaling=1, name= "load 1")
    pp.create_load(net, bus8, p_mw=10, q_mvar=1, scaling=1, name= "load 2")


    # Define load

    

#modifies load
    #pp.create_load(net, bus8, p_mw=4, q_mvar=1, scaling=1, name="load 2")

    # geodata Zeilen initialisieren
    net.bus_geodata.loc[0] = None
    net.bus_geodata.loc[1] = None
    net.bus_geodata.loc[2] = None
    net.bus_geodata.loc[3] = None
    net.bus_geodata.loc[4] = None
    net.bus_geodata.loc[5] = None
    net.bus_geodata.loc[6] = None
    
    #test
    net.bus_geodata.loc[7] = None
    net.bus_geodata.loc[8] = None

    

    # Knoten neue Koordinaten für Plot zuweisen
    net.bus_geodata.x.at[0] = 0
    net.bus_geodata.x.at[1] = 0
    net.bus_geodata.x.at[2] = -2
    net.bus_geodata.x.at[3] = -2
    
    # test
    net.bus_geodata.x.at[7] = -2
    net.bus_geodata.x.at[8] = 2

    
    net.bus_geodata.x.at[4] = 2
    net.bus_geodata.x.at[5] = 2
    net.bus_geodata.x.at[6] = 2

################################
    net.bus_geodata.y.at[0] = 0
    net.bus_geodata.y.at[1] = -1
    net.bus_geodata.y.at[2] = -2
    net.bus_geodata.y.at[3] = -4
    
    #test
    net.bus_geodata.y.at[7] = -6
    net.bus_geodata.y.at[8] =  -6


    net.bus_geodata.y.at[4] = -2
    net.bus_geodata.y.at[5] = -3
    net.bus_geodata.y.at[6] = -4


    return net

###  Test grid developed by Arjun:
def three_radial_bus_net():
    
        import pandapower as pp
        net = pp.create_empty_network()  # create an empty network

        # busbars and nodes
        bus0 = pp.create_bus(net, name="Bus_extgrid", vn_kv=20, type="b")
        bus1 = pp.create_bus(net, name="Bus_line1", vn_kv=20, type="n")
        bus2 = pp.create_bus(net, name="Bus_line2", vn_kv=20, type="n")
        bus3 = pp.create_bus(net, name="Bus_line3", vn_kv=20, type="n")
        bus4 = pp.create_bus(net, name="Bus_line4", vn_kv=20, type="n")
        bus5 = pp.create_bus(net, name="Bus_line5", vn_kv=20, type="n")
        bus6 = pp.create_bus(net, name="Bus_load1", vn_kv=20, type="n")
        
        bus7 = pp.create_bus(net, name="Bus_line6", vn_kv=20, type="n")
        bus8 = pp.create_bus(net, name="Bus_line7", vn_kv=20, type="n")
        bus9 = pp.create_bus(net, name="Bus_line8", vn_kv=20, type="n")
        bus10= pp.create_bus(net, name="Bus_line9", vn_kv=20, type="n")
        bus11= pp.create_bus(net, name="Bus_load2", vn_kv=20, type="n")
        
        bus12= pp.create_bus(net, name="Bus_line10", vn_kv=20, type="n")
        bus13= pp.create_bus(net, name="Bus_line11", vn_kv=20, type="n")
        bus14= pp.create_bus(net, name="Bus_line12", vn_kv=20, type="n")
        bus15= pp.create_bus(net, name="Bus_load3", vn_kv=20, type="n")

        


        # external grids
        pp.create_ext_grid(net, bus0, vm_pu=1.0, va_degree=0, s_sc_max_mva=100, s_sc_min_mva=50, rx_max=0.1, rx_min=0.1)

        # lines


        line1 = pp.create_line_from_parameters(net, bus0, bus1, length_km=5, index=1, r_ohm_per_km=0.169,
                                               x_ohm_per_km=0.118438, c_nf_per_km=273, max_i_ka=0.361)

        line2 = pp.create_line_from_parameters(net, bus1, bus2, length_km=4, index=2, r_ohm_per_km=0.256,
                                               x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)

        line3 = pp.create_line_from_parameters(net, bus2, bus3, length_km=4, index=3, r_ohm_per_km=0.256,
                                               x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)

        line4 = pp.create_line_from_parameters(net, bus3, bus4, length_km=0.5, index=4, r_ohm_per_km=0.256,
                                               x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)

        line5 = pp.create_line_from_parameters(net, bus4, bus5, length_km=0.5, index=5, r_ohm_per_km=0.256,
                                               x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)
        
        line6 = pp.create_line_from_parameters(net, bus5, bus6, length_km=5, index=6, r_ohm_per_km=0.169,
                                               x_ohm_per_km=0.118438, c_nf_per_km=273, max_i_ka=0.361)
        


        line7 = pp.create_line_from_parameters(net, bus2, bus7, length_km=4, index=7, r_ohm_per_km=0.256,
                                               x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)

        line8 = pp.create_line_from_parameters(net, bus7, bus8, length_km=4, index=8, r_ohm_per_km=0.256,
                                               x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)

        line9 = pp.create_line_from_parameters(net, bus8, bus9, length_km=0.5, index=9, r_ohm_per_km=0.256,
                                               x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)

        line10 = pp.create_line_from_parameters(net, bus9, bus10, length_km=0.5, index=10, r_ohm_per_km=0.256,
                                               x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)
        
        line11 = pp.create_line_from_parameters(net, bus10, bus11, length_km=0.5, index=11, r_ohm_per_km=0.256,
                                               x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)
        
        
        line12 = pp.create_line_from_parameters(net, bus9, bus12, length_km=4, index=12, r_ohm_per_km=0.256,
                                               x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)

        line13 = pp.create_line_from_parameters(net, bus12, bus13, length_km=4, index=13, r_ohm_per_km=0.256,
                                               x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)

        line14 = pp.create_line_from_parameters(net, bus13, bus14, length_km=0.5, index=14, r_ohm_per_km=0.256,
                                               x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)

        line15 = pp.create_line_from_parameters(net, bus14, bus15, length_km=0.5, index=15, r_ohm_per_km=0.256,
                                               x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)


        net.line["endtemp_degree"] = 250

        # switches
        swo=pp.create_switch(net, bus0, line1, et="l", type="CB", closed=True, index=0)
        
        sw1 = pp.create_switch(net, bus1, line2, et="l", type="CB", closed=True, index=1)

        sw2 = pp.create_switch(net, bus2, line3, et="l", type = "CB", closed = True, index=2)

        sw3 = pp.create_switch(net, bus3, line4, et="l", type="CB", closed=True, index=3)

        sw4 = pp.create_switch(net, bus4, line5, et="l", type = "CB", closed = True, index=4)

        sw5 = pp.create_switch(net, bus5, line6, et="l", type="CB", closed=True, index=5)
        
        
        # 2nd radial connection
        sw6 = pp.create_switch(net, bus2, line7, et="l", type="CB", closed=True, index=6)
        
        sw7 = pp.create_switch(net, bus7, line8, et="l", type="CB", closed=True, index=7)

        sw8 = pp.create_switch(net, bus8, line9, et="l", type = "CB", closed = True, index=8)

        sw9 = pp.create_switch(net, bus9, line10, et="l", type="CB", closed=True, index=9)
        
        sw10 = pp.create_switch(net, bus10, line11, et="l", type = "CB", closed = True, index=10)

        #3rd radial
        sw11 = pp.create_switch(net, bus9, line12, et="l", type = "CB", closed = True, index=11)

        sw12 = pp.create_switch(net, bus12, line13, et="l", type="CB", closed=True, index=12)
        
        sw13 = pp.create_switch(net, bus13, line14, et="l", type = "CB", closed = True, index=13)

        sw14 = pp.create_switch(net, bus14, line15, et="l", type="CB", closed=True, index=14)
        
        


        # Define load

        pp.create_load(net, bus6, p_mw=5, q_mvar=1, scaling=1, name="load1")
        
        pp.create_load(net, bus11, p_mw=5, q_mvar=1, scaling=1, name="load2")

        pp.create_load(net, bus15, p_mw=4, q_mvar=1, scaling=1, name="load3")
        

        # geodata Zeilen initialisieren
        net.bus_geodata.loc[0] = None
        net.bus_geodata.loc[1] = None
        net.bus_geodata.loc[2] = None
        net.bus_geodata.loc[3] = None
        net.bus_geodata.loc[4] = None
        net.bus_geodata.loc[5] = None
        net.bus_geodata.loc[6] = None
        net.bus_geodata.loc[7] = None
        net.bus_geodata.loc[8] = None
        net.bus_geodata.loc[9] = None
        net.bus_geodata.loc[10] = None
        net.bus_geodata.loc[11] = None
        net.bus_geodata.loc[12] = None
        net.bus_geodata.loc[13] = None
        net.bus_geodata.loc[14] = None
        net.bus_geodata.loc[15] = None

        

        # Knoten neue Koordinaten für Plot zuweisen
        net.bus_geodata.x.at[0] =  0
        net.bus_geodata.x.at[1] =  0
        net.bus_geodata.x.at[2] =  0
        net.bus_geodata.x.at[3] = -2
        net.bus_geodata.x.at[4] = -2
        net.bus_geodata.x.at[5] = -2
        net.bus_geodata.x.at[6] = -2
        
        net.bus_geodata.x.at[7] =  0
        net.bus_geodata.x.at[8] =  0
        net.bus_geodata.x.at[9] =  0
        net.bus_geodata.x.at[9] =  0
        net.bus_geodata.x.at[10] = 0
        
        net.bus_geodata.x.at[11] = 0
        net.bus_geodata.x.at[12] = 2
        net.bus_geodata.x.at[13] = 2
        net.bus_geodata.x.at[14] = 2
        net.bus_geodata.x.at[15] = 2

        

    ################################
        net.bus_geodata.y.at[0] = 0
        net.bus_geodata.y.at[1] = -1
        net.bus_geodata.y.at[2] = -2
        net.bus_geodata.y.at[3] = -3
        net.bus_geodata.y.at[4] = -4
        net.bus_geodata.y.at[5] = -5
        net.bus_geodata.y.at[6] = -6
        
        net.bus_geodata.y.at[7] = -4.5
        net.bus_geodata.y.at[8] = -5
        net.bus_geodata.y.at[9] = -6
        net.bus_geodata.y.at[10] = -7
        net.bus_geodata.y.at[11] = -8
        
        net.bus_geodata.y.at[12] = -8
        net.bus_geodata.y.at[13] = -9
        net.bus_geodata.y.at[14] = -10
        net.bus_geodata.y.at[15] = -12

        
        return net
    
###  Test grid developed by Arjun:
def radial_bus_net_branch():
    
        import pandapower as pp
        net = pp.create_empty_network()  # create an empty network

        # busbars and nodes
        bus0 = pp.create_bus(net, name="Bus_extgrid", vn_kv=20, type="b")
        bus1 = pp.create_bus(net, name="Bus_line1", vn_kv=20, type="n")
        bus2 = pp.create_bus(net, name="Bus_line2", vn_kv=20, type="n")
        bus3 = pp.create_bus(net, name="Bus_line3", vn_kv=20, type="n")
        bus4 = pp.create_bus(net, name="Bus_line4", vn_kv=20, type="n")
        bus5 = pp.create_bus(net, name="Bus_line5", vn_kv=20, type="n")
        bus6 = pp.create_bus(net, name="Bus_line6", vn_kv=20, type="n")
        bus7 = pp.create_bus(net, name="Bus_load1", vn_kv=20, type="n")
        
        
        bus8 = pp.create_bus(net, name="Bus_line7", vn_kv=20, type="n")
        bus9 = pp.create_bus(net, name="Bus_line8", vn_kv=20, type="n")
        bus10= pp.create_bus(net, name="Bus_load2", vn_kv=20, type="n")
        
        
        
        bus11= pp.create_bus(net, name="Bus_line9", vn_kv=20, type="n")
        bus12= pp.create_bus(net, name="Bus_line10", vn_kv=20, type="n")
        bus13= pp.create_bus(net, name="Bus_line11", vn_kv=20, type="n")
        bus14= pp.create_bus(net, name="Bus_line12", vn_kv=20, type="n")
        bus15= pp.create_bus(net, name="Bus_load3", vn_kv=20, type="n")

        bus16 = pp.create_bus(net, name="Bus_line13", vn_kv=20, type="n")
        bus17 = pp.create_bus(net, name="Bus_load4", vn_kv=20, type="n")
        
        bus18 = pp.create_bus(net, name="Bus_line14", vn_kv=20, type="n")
        bus19 = pp.create_bus(net, name="Bus_line15", vn_kv=20, type="n")
        bus20 = pp.create_bus(net, name="Bus_load5", vn_kv=20, type="n")



        # external grids
        pp.create_ext_grid(net, bus0, vm_pu=1.0, va_degree=0, s_sc_max_mva=100, s_sc_min_mva=50, rx_max=0.1, rx_min=0.1)

        # lines
        

        line0 = pp.create_line_from_parameters(net, bus0, bus1, length_km=5, index=0, r_ohm_per_km=0.169,
                                               x_ohm_per_km=0.118438, c_nf_per_km=273, max_i_ka=0.361)

        line1 = pp.create_line_from_parameters(net, bus1, bus2, length_km=4, index=1, r_ohm_per_km=0.256,
                                               x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)

        line2 = pp.create_line_from_parameters(net, bus2, bus3, length_km=4, index=2, r_ohm_per_km=0.256,
                                               x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)

        line3 = pp.create_line_from_parameters(net, bus3, bus4, length_km=0.5, index=3, r_ohm_per_km=0.256,
                                               x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)

        line4 = pp.create_line_from_parameters(net, bus4, bus5, length_km=0.5, index=4, r_ohm_per_km=0.256,
                                               x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)
        
        line5 = pp.create_line_from_parameters(net, bus5, bus6, length_km=5, index=5, r_ohm_per_km=0.169,
                                               x_ohm_per_km=0.118438, c_nf_per_km=273, max_i_ka=0.361)
        
        line6 = pp.create_line_from_parameters(net, bus6, bus7, length_km=4, index=6, r_ohm_per_km=0.256,
                                               x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)
        
        
        

        line7 = pp.create_line_from_parameters(net, bus4, bus8, length_km=4, index=7, r_ohm_per_km=0.256,
                                               x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)

        line8 = pp.create_line_from_parameters(net, bus8, bus9, length_km=0.5, index=8, r_ohm_per_km=0.256,
                                               x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)

        line9 = pp.create_line_from_parameters(net, bus9, bus10, length_km=0.5, index=9, r_ohm_per_km=0.256,
                                               x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)
        
        
        
        line10 = pp.create_line_from_parameters(net, bus4, bus11, length_km=0.5, index=10, r_ohm_per_km=0.256,
                                               x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)
        
        
        line11 = pp.create_line_from_parameters(net, bus11, bus12, length_km=4, index=11, r_ohm_per_km=0.256,
                                               x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)

        line12 = pp.create_line_from_parameters(net, bus12, bus13, length_km=4, index=12, r_ohm_per_km=0.256,
                                               x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)

        line13 = pp.create_line_from_parameters(net, bus13, bus14, length_km=0.5, index=13, r_ohm_per_km=0.256,
                                               x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)

        line14 = pp.create_line_from_parameters(net, bus14, bus15, length_km=0.5, index=14, r_ohm_per_km=0.256,
                                                x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)



        line15 = pp.create_line_from_parameters(net, bus1, bus16, length_km=4, index=15, r_ohm_per_km=0.256,
                                               x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)
        
        line16 = pp.create_line_from_parameters(net, bus16, bus17, length_km=4, index=16, r_ohm_per_km=0.256,
                                               x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)
        
        
        
        line17 = pp.create_line_from_parameters(net, bus1, bus18, length_km=4, index=17, r_ohm_per_km=0.256,
                                               x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)
        
        line18 = pp.create_line_from_parameters(net, bus18, bus19, length_km=4, index=18, r_ohm_per_km=0.256,
                                               x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)
        
        line19 = pp.create_line_from_parameters(net, bus19, bus20, length_km=4, index=19, r_ohm_per_km=0.256,
                                               x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)
    
        net.line["endtemp_degree"] = 250

        # switches
        
        #first radial
        swo = pp.create_switch(net, bus0, line0, et="l", type="CB", closed=True, index=0)
        
        sw1 = pp.create_switch(net, bus1, line1, et="l", type="CB", closed=True, index=1)

        sw2 = pp.create_switch(net, bus2, line2, et="l", type = "CB", closed = True, index=2)

        sw3 = pp.create_switch(net, bus3, line3, et="l", type="CB", closed=True, index=3)

        sw4 = pp.create_switch(net, bus4, line4, et="l", type = "CB", closed = True, index=4)

        sw5 = pp.create_switch(net, bus5, line5, et="l", type="CB", closed=True, index=5)
        
        sw6 = pp.create_switch(net, bus6, line6, et="l", type="CB", closed=True, index=6)

    
        sw7 = pp.create_switch(net, bus4, line7, et="l", type="CB", closed=True, index=7)
        
        sw8 = pp.create_switch(net, bus8, line8, et="l", type="CB", closed=True, index=8)

        sw9 = pp.create_switch(net, bus9, line9, et="l", type = "CB", closed = True, index=9)
        
        
        sw10 = pp.create_switch(net, bus4, line10, et="l", type = "CB", closed = True, index=10)

        sw11 = pp.create_switch(net, bus11, line11, et="l", type = "CB", closed = True, index=11)

        sw12 = pp.create_switch(net, bus12, line12, et="l", type="CB", closed=True, index=12)
        
        sw13 = pp.create_switch(net, bus13, line13, et="l", type = "CB", closed = True, index=13)

        sw14 = pp.create_switch(net, bus14, line14, et="l", type="CB", closed=True, index=14)
        
        #2nd radial
        
        sw15 = pp.create_switch(net, bus1, line15, et="l", type="CB", closed=True, index=15)

        sw16 = pp.create_switch(net, bus16, line16, et="l", type = "CB", closed = True, index=16)
        
        #3rd radial

        sw17 = pp.create_switch(net, bus1, line17, et="l", type="CB", closed=True, index=17)

        sw18 = pp.create_switch(net, bus18, line18, et="l", type = "CB", closed = True, index=18)

        sw19 = pp.create_switch(net, bus19, line19, et="l", type="CB", closed=True, index=19)
        
   


        # Define load

        pp.create_load(net, bus7, p_mw=5, q_mvar=1, scaling=1, name="load1")
        
        pp.create_load(net, bus10, p_mw=5, q_mvar=1, scaling=1, name="load2")

        pp.create_load(net, bus15, p_mw=4, q_mvar=1, scaling=1, name="load3")
        
        pp.create_load(net, bus17, p_mw=4, q_mvar=1, scaling=1, name="load4")
        
        pp.create_load(net, bus20, p_mw=4, q_mvar=1, scaling=1, name="load5")


        

        # geodata Zeilen initialisieren
        net.bus_geodata.loc[0] = None
        net.bus_geodata.loc[1] = None
        net.bus_geodata.loc[2] = None
        net.bus_geodata.loc[3] = None
        net.bus_geodata.loc[4] = None
        net.bus_geodata.loc[5] = None
        net.bus_geodata.loc[6] = None
        net.bus_geodata.loc[7] = None
        net.bus_geodata.loc[8] = None
        net.bus_geodata.loc[9] = None
        net.bus_geodata.loc[10] = None
        net.bus_geodata.loc[11] = None
        net.bus_geodata.loc[12] = None
        net.bus_geodata.loc[13] = None
        net.bus_geodata.loc[14] = None
        net.bus_geodata.loc[15] = None
        net.bus_geodata.loc[16] = None
        net.bus_geodata.loc[17] = None
        net.bus_geodata.loc[18] = None
        net.bus_geodata.loc[19] = None
        net.bus_geodata.loc[20] = None

        

        # Knoten neue Koordinaten für Plot zuweisen
        net.bus_geodata.x.at[0] =  0
        net.bus_geodata.x.at[1] =  0
        net.bus_geodata.x.at[2] =  -2
        net.bus_geodata.x.at[3] = -2
        net.bus_geodata.x.at[4] = -2
        net.bus_geodata.x.at[5] = -3
        net.bus_geodata.x.at[6] = -3
        net.bus_geodata.x.at[7] =  -3
        
        net.bus_geodata.x.at[8] =  -2
        net.bus_geodata.x.at[9] =  -2
        net.bus_geodata.x.at[10] = -2
        
        net.bus_geodata.x.at[11] = 3
        net.bus_geodata.x.at[12] = 3
        net.bus_geodata.x.at[13] = 3
        net.bus_geodata.x.at[14] = 3
        net.bus_geodata.x.at[15] = 3
        
        net.bus_geodata.x.at[16]= 0
        net.bus_geodata.x.at[17] = 0
        
        net.bus_geodata.x.at[18] = 2
        net.bus_geodata.x.at[19] = 2
        net.bus_geodata.x.at[20] = 2

        

    ################################
        net.bus_geodata.y.at[0] = 0
        net.bus_geodata.y.at[1] = -1
        net.bus_geodata.y.at[2] = -1
        net.bus_geodata.y.at[3] = -3
        net.bus_geodata.y.at[4] = -5
        net.bus_geodata.y.at[5] = -5
        net.bus_geodata.y.at[6] = -7
        net.bus_geodata.y.at[7] = -8
        
        net.bus_geodata.y.at[8] = -6
        net.bus_geodata.y.at[9] = -8
        net.bus_geodata.y.at[10] = -9
        
        net.bus_geodata.y.at[11] = -5
        net.bus_geodata.y.at[12] = -7
        net.bus_geodata.y.at[13] = -8
        net.bus_geodata.y.at[14] = -10
        net.bus_geodata.y.at[15] = -11
        
        net.bus_geodata.y.at[16] = -3
        net.bus_geodata.y.at[17] = -4
        
        net.bus_geodata.y.at[18] = -1
        net.bus_geodata.y.at[19] = -3
        net.bus_geodata.y.at[20] = -4

        
        return net
    

def oc_relay_net(open_loop=True):
    import pandapower as pp
    net = pp.create_empty_network()  # create an empty network

    # busbars and nodes
    bus0 = pp.create_bus(net, name="Bus_extgrid", vn_kv=20, type="b")
    bus1 = pp.create_bus(net, name="Bus_line0", vn_kv=20, type="n")
    bus2 = pp.create_bus(net, name="Bus_line1", vn_kv=20, type="n")
    bus3 = pp.create_bus(net, name="Bus_line2", vn_kv=20, type="n")
    bus4 = pp.create_bus(net, name="Bus_load1", vn_kv=20, type="n")
    bus5 = pp.create_bus(net, name="Bus_line3", vn_kv=20, type="n")
    bus6 = pp.create_bus(net, name="Bus_load2", vn_kv=20, type="n")
    

    # external grids
    pp.create_ext_grid(net, bus0, vm_pu=1.0, va_degree=0, s_sc_max_mva=100, s_sc_min_mva=50, rx_max=0.1, rx_min=0.1)


    line0 = pp.create_line_from_parameters(net, bus0, bus1, length_km=2, index=0, r_ohm_per_km=0.169,
                                           x_ohm_per_km=0.118438, c_nf_per_km=273, max_i_ka=0.361)
    line1 = pp.create_line_from_parameters(net, bus1, bus2, length_km=5, index=1, r_ohm_per_km=0.169,
                                           x_ohm_per_km=0.118438, c_nf_per_km=273, max_i_ka=0.361)

    line2 = pp.create_line_from_parameters(net, bus2, bus3, length_km=4, index=2, r_ohm_per_km=0.256,
                                           x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)

    line3 = pp.create_line_from_parameters(net, bus1, bus4, length_km=4, index=3, r_ohm_per_km=0.256,
                                           x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)

    line4 = pp.create_line_from_parameters(net, bus4, bus5, length_km=0.5, index=4, r_ohm_per_km=0.256,
                                           x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)

    line5 = pp.create_line_from_parameters(net, bus5, bus6, length_km=0.5, index=5, r_ohm_per_km=0.256,
                                           x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)
        
    line6 = pp.create_line_from_parameters(net, bus3, bus6, length_km=4, index=6, r_ohm_per_km=0.256,
                                           x_ohm_per_km=0.126606, c_nf_per_km=235, max_i_ka=0.286)
    

    net.line["endtemp_degree"] = 250

    # switches
    swo=pp.create_switch(net, bus0, line0, et="l", type="CB_non_dir", closed=True, index=0)
    sw1 = pp.create_switch(net, bus1, line1, et="l", type="CB_non_dir", closed=True, index=1)

    sw2 = pp.create_switch(net, bus2, line2, et="l", type = "CB_non_dir", closed = True, index=2)

    sw3 = pp.create_switch(net, bus1, line3, et="l", type="CB_non_dir", closed=True, index=3)

    sw4 = pp.create_switch(net, bus4, line4, et="l", type = "CB_non_dir", closed = True, index=4)

    sw5 = pp.create_switch(net, bus5, line5, et="l", type="CB_non_dir", closed=True, index=5)
    
    # bus bar connection
    if open_loop:
        
        sw6 = pp.create_switch(net, bus3, line6, et="l", type="CB_non_dir", closed=False, index=6)
        sw7 = pp.create_switch(net, bus6, line6, et="l", type="CB_non_dir", closed=False, index=7)
    else:
        sw6 = pp.create_switch(net, bus3, line6, et="l", type="CB_non_dir", closed=True, index=6)
        sw7 = pp.create_switch(net, bus6, line6, et="l", type="CB_non_dir", closed=True, index=7)
    
    #define load
    pp.create_load(net, bus3, p_mw=5, q_mvar=1, scaling=1, name= "load 1")
    pp.create_load(net, bus6, p_mw=2, q_mvar=1, scaling=1, name= "load 2")

    # initialise geo coordinates
    net.bus_geodata.loc[0] = None
    net.bus_geodata.loc[1] = None
    net.bus_geodata.loc[2] = None
    net.bus_geodata.loc[3] = None
    net.bus_geodata.loc[4] = None
    net.bus_geodata.loc[5] = None
    net.bus_geodata.loc[6] = None
    
    
    #input geo coordinates
    net.bus_geodata.x.at[0] = 0
    net.bus_geodata.x.at[1] = 0
    net.bus_geodata.x.at[2] = -2
    net.bus_geodata.x.at[3] = -2

    
    net.bus_geodata.x.at[4] = 2
    net.bus_geodata.x.at[5] = 2
    net.bus_geodata.x.at[6] = 2

    net.bus_geodata.y.at[0] = 0
    net.bus_geodata.y.at[1] = -1
    net.bus_geodata.y.at[2] = -2
    net.bus_geodata.y.at[3] = -4


    net.bus_geodata.y.at[4] = -2
    net.bus_geodata.y.at[5] = -3
    net.bus_geodata.y.at[6] = -4


    return net