# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 13:21:26 2017

@author: thurner
"""
import pandas as pd
from numpy import real, conj, max, dot, diag

def _extract_results(net, ppc, ppci, zbus, Yf):
    bus = net._is_elems["bus"]
    currents = net._options["currents"]
    branch_lookup = net._pd2ppc_lookups["branch"]
    bus_lookup = net._pd2ppc_lookups["bus"]
    net.res_bus_sc = pd.DataFrame(index=net.bus.index, data=bus[currents])
    ikss = bus["ikss_pu"].values
    V = dot(zbus, diag(1 / ikss[bus_lookup]))
#    print(V)
    fb = real(ppc["branch"][:,0]).astype(int)
#    S_all = abs(V[fb] * conj(Yf.dot(V)) * ppc["baseMVA"])
    S_all = abs(V[fb] * conj(Yf.dot(V)) * ppc["baseMVA"])
    print(S_all)
    S = max(S_all, axis=1)
    f, t = branch_lookup["line"]
    net.res_line_sc = pd.DataFrame(index=net.line.index, data=S[f:t], columns=["s_sc_kva"])
    