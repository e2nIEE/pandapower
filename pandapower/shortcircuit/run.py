# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

import pandas as pd
from pandapower.shortcircuit.currents import calc_ikss, calc_ip, calc_ith
from pandapower.shortcircuit.impedance import calc_equiv_sc_impedance
from pandapower.shortcircuit.kappa import calc_kappa
try:
    import pplog as logging
except:
    import logging
logger = logging.getLogger(__name__)

def runsc(net, case='max', lv_tol_percent=10, network_structure="auto", ip=False, ith=False, 
          tk_s=1.):
    
    """
    Calculates minimal or maximal symmetrical short-circuit currents.  
    The calculation is based on the method of the equivalent voltage source
    according to DIN/IEC EN 60909.
    The initial short-circuit alternating current *ikss* is the basis of the short-circuit
    calculation and is therefore always calculated.
    Other short-circuit currents can be calculated from *ikss* with the conversion factors defined
    in DIN/IEC EN 60909.
    
    The output is stored in the net.res_bus_sc table as a short_circuit current
    for each bus.

    INPUT:
        **net** (PandaPowerNet) Pandapower Network
        
        **case** (str) 'max' / 'min' for maximal / minimal current calculation
        
        **lv_tol_percent** (int) voltage tolerance band in the low voltage grid,  can be either 6% or 10% according to IEC 60909
            
        **ip** (bool) if True, calculate aperiodic short-circuit current 
        
        **Ith** (bool) if True, calculate equivalent thermical short-circuit current Ith

        **meshing** (str) define option for meshing (only relevant for ip and ith)
        
            "meshed" - it is assumed all buses are supplied over multiple paths
            
            "radial" - it is assumed all buses are supplied over exactly one path
            
            "auto" - topology check for each bus is performed to see if it is supplied over multiple paths (might be computationally expensive)

        **Tk_s** (float) failure clearing time in seconds (only relevant for ith)

    OUTPUT:
    
    EXAMPLE:
        runsc(net)

        print(net.res_bus_sc)
    """
    if ip and len(net.gen) > 0:
        raise NotImplementedError("aperiodic short-circuit current not implemented for short circuits close to generators")

    if ith and len(net.gen) > 0:
        raise NotImplementedError("thermical short-circuit current not implemented for short circuits close to generators")

    if case not in ['max', 'min']:
        raise ValueError('case can only be "min" or "max" for minimal or maximal short "\
                                "circuit current')
    if network_structure not in ["meshed", "radial", "auto"]:
        raise ValueError('specify network structure as "meshed", "radial" or "auto"')        
            
    if len(net.ext_grid) > 0:
        if  not "s_sc_%s_mva"%case in net.ext_grid or any(pd.isnull(net.ext_grid["s_sc_%s_mva"%case])):
            raise ValueError("s_sc_%s is not defined for all ext_grids" %case)
        if  not "rx_%s"%case in net.ext_grid or any(pd.isnull(net.ext_grid["rx_%s"%case])):
            raise ValueError("rx_%s is not defined for all ext_grids" %case)

    setup_sc(net, lv_tol_percent)
    calc_equiv_sc_impedance(net, case)
    calc_ikss(net, case)
    
    if ip or ith:
        calc_kappa(net, network_structure)

    if ip:
        calc_ip(net, case, network_structure)

    if ith:
        calc_ith(net, case, tk_s, network_structure)
    
    clean_up(net)

def setup_sc(net, lv_tol_percent):
    if lv_tol_percent==10:
        c_ns = 1.1
    elif lv_tol_percent==6:
        c_ns = 1.05
    else:
        raise ValueError("Voltage tolerance in the low voltage grid has" \
                                    " to be either 6% or 10% according to IEC 60909")

    lv_buses = net.bus[net.bus.vn_kv < 1.].index
    net.bus["c_max"] = 1.1
    net.bus.c_max.loc[lv_buses] = c_ns
    net.bus["c_min"] = 1.
    net.bus.c_min.loc[lv_buses] = .95
    net.bus["kappa_max"] = 2.
    net.bus.kappa_max.loc[lv_buses] = 1.8
    
def clean_up(net):
    for var in ["kappa_max", "z_equiv", "kappa_korr", "kappa", "c_max", "c_min", "x", "r"]:
        for element in "line", "ext_grid", "trafo":
            if var in net[element]:            
                net[element].drop(var, axis=1, inplace=True)