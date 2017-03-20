# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 13:21:26 2017

@author: thurner
"""
import pandas as pd
from numpy import real, conj, max, dot, diag

from pandapower.shortcircuit.idx_bus import IKSS, IP, ITH, R_EQUIV, X_EQUIV
from pandapower.shortcircuit.idx_brch import IKSS_F, IKSS_T, IP_F, IP_T, ITH_F, ITH_T

def _extract_results(net, ppc):
    _initialize_result_tables(net)
    _get_bus_results(net, ppc)
    _get_line_results(net, ppc)
    _get_trafo_results(net, ppc)
    _get_trafo3w_results(net, ppc)
        
def _initialize_result_tables(net):
    net.res_bus_sc = pd.DataFrame(index=net.bus.index)
    net.res_line_sc = pd.DataFrame(index=net.line.index)
    net.res_trafo_sc = pd.DataFrame(index=net.trafo.index)
    net.res_trafo3w_sc = pd.DataFrame(index=net.trafo3w.index)
    
def _get_bus_results(net, ppc):
    bus_lookup = net._pd2ppc_lookups["bus"]
    ppc_index = bus_lookup[net.bus.index]
    net.res_bus_sc["ikss_ka"] = ppc["bus_sc"][ppc_index, IKSS]
    if net._options["ip"]:
        net.res_bus_sc["ip_ka"] = ppc["bus_sc"][ppc_index, IP]
    if net._options["ith"]:
        net.res_bus_sc["ith_ka"] = ppc["bus_sc"][ppc_index, ITH]
        
def _get_line_results(net, ppc):
    branch_lookup = net._pd2ppc_lookups["branch"]
    if "line" in branch_lookup:
        f, t = branch_lookup["line"]
        net.res_line_sc["ikss_ka"] = ppc["branch_sc"][f:t, IKSS_F].real
        
def _get_trafo_results(net, ppc):
    branch_lookup = net._pd2ppc_lookups["branch"]
    if "trafo" in branch_lookup:
        f, t = branch_lookup["trafo"]
        net.res_trafo_sc["ikss_lv_ka"] = ppc["branch_sc"][f:t, IKSS_F].real
        net.res_trafo_sc["ikss_hv_ka"] = ppc["branch_sc"][f:t, IKSS_T].real
        
def _get_trafo3w_results(net, ppc):
    branch_lookup = net._pd2ppc_lookups["branch"]
    if "trafo3w" in branch_lookup:
        f, t = net._pd2ppc_lookups["branch"]["trafo3w"]
        hv = int(f + (t - f) / 3)
        mv = int(f + 2 * (t - f) / 3)
        lv = t    
        net.res_trafo3w_sc["ikss_hv_ka"] = ppc["branch"][f:hv, IKSS_F].real * 1e3
        net.res_trafo3w_sc = ppc["branch"][hv:mv, IKSS_T].real * 1e3
        net.res_trafo3w_sc = ppc["branch"][mv:lv, IKSS_T].real * 1e3