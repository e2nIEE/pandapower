# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 16:34:19 2015

@author: lthurner
"""
import pandapower as pp


def create_example_simple():
    """
    The following example contains all basic elements that are supported by the pandapower format:
    """
    net = pp.create_empty_network()

    # create busses
    bus1 = pp.create_bus(net, name="HV Busbar", vn_kv=110, type="b")
    bus2 = pp.create_bus(net, name="HV Busbar 2", vn_kv=110, type="b")
    bus3 = pp.create_bus(net, name="HV Transformer Bus", vn_kv=110, type="n")
    bus4 = pp.create_bus(net, name="MV Transformer Bus", vn_kv=20, type="n")
    bus5 = pp.create_bus(net, name="MV Station 1", vn_kv=20, type="b")
    bus6 = pp.create_bus(net, name="MV Station 2", vn_kv=20, type="b")
    bus7 = pp.create_bus(net, name="MV Station 3", vn_kv=20, type="b")
    bus8 = pp.create_bus(net, name="MV Station 4", vn_kv=20, type="b")

    # create external grid
    pp.create_ext_grid(net, bus1, va_degree=20)

    # create transformer
    pp.create_transformer(net, bus3, bus4, name="110kV/20kV transformer",
                          std_type="25 MVA 110/20 kV")

    # create lines
    line1 = pp.create_line(net, bus1, bus2, 0.225, std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV",
                           name="Line 1")
    line2 = pp.create_line(net, bus5, bus6, 0.075, std_type="NA2XS2Y 1x240 RM/25 12/20 kV",
                           name="Line 2")
    line3 = pp.create_line(net, bus5, bus7, 0.125, std_type="NA2XS2Y 1x240 RM/25 12/20 kV",
                           name="Line 3")
    line4 = pp.create_line(net, bus5, bus8, 0.175, std_type="NA2XS2Y 1x240 RM/25 12/20 kV",
                           name="Line 4")

    # create switches
    # (Circuit breaker)
    pp.create_switch(net, bus2, bus3, et="b", type="CB")
    pp.create_switch(net, bus4, bus5, et="b", type="CB")
    # (Load break switches)
    pp.create_switch(net, bus5, line2, et="l", type="LBS")
    pp.create_switch(net, bus6, line2, et="l", type="LBS")
    pp.create_switch(net, bus5, line3, et="l", type="LBS")
    pp.create_switch(net, bus7, line3, et="l", type="LBS")
    pp.create_switch(net, bus5, line4, et="l", type="LBS")
    pp.create_switch(net, bus8, line4, et="l", type="LBS")

    # create generator
    pp.create_gen(net, bus6, p_kw=-6000, vm_pu=1.05)

    # create static generator
    pp.create_sgen(net, bus7, p_kw=-2000)

    # Last mit 20MV bei scaling 0.6 ~ 12MV -> Industrie / Verbaucher am MS Netz
    # create load
    pp.create_load(net, bus8, p_kw=20000, q_kvar=4000, scaling=0.6)

    # create shunt
    pp.create_shunt(net, bus3, p_kw=0, q_kvar=-960, name='Shunt')

    # run power flow and generate result tables
    pp.runpp(net)

    return net
