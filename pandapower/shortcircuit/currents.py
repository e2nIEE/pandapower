# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

from pandapower.shortcircuit.idx_bus import C_MIN, C_MAX, KAPPA, R_EQUIV, IKSS, IP, ITH, X_EQUIV
from pandapower.shortcircuit.idx_brch import IKSS_F, IKSS_T, IP_F, IP_T, ITH_F, ITH_T
from pypower.idx_bus import BASE_KV
import numpy as np

def _calc_ikss(net, ppc):
    fault = net._options["fault"]
    case = net._options["case"]
    c = ppc["bus_sc"][:, C_MIN] if case == "min" else ppc["bus_sc"][:, C_MAX]
    ppc["internal"]["baseI"] = ppc["bus"][:, BASE_KV] * np.sqrt(3) / ppc["baseMVA"]
    z_equiv = abs(ppc["bus_sc"][:, R_EQUIV] + ppc["bus_sc"][:, X_EQUIV] *1j)
    if fault == "3ph":
        ppc["bus_sc"][:, IKSS] = c / z_equiv  / ppc["bus"][:, BASE_KV] / np.sqrt(3) * ppc["baseMVA"]
    elif fault == "2ph":
        ppc["bus_sc"][:, IKSS] = c / z_equiv / ppc["bus"][:, BASE_KV] / 2 * ppc["baseMVA"]
    ppc["branch_sc"][:, [IKSS_F, IKSS_T]] = _branch_currents_from_bus(ppc, ppc["bus_sc"][:, IKSS])
    
def _calc_ip(ppc):
    ppc["bus_sc"][:, IP] = ppc["bus_sc"][:, KAPPA] * np.sqrt(2) * ppc["bus_sc"][:, IKSS]
    ppc["branch_sc"][:, [IP_F, IP_T]] = _branch_currents_from_bus(ppc, ppc["bus_sc"][:, IP])
    
def _calc_ith(net, ppc):
    tk_s = net["_options"]["tk_s"]
    kappa = ppc["bus_sc"][:, KAPPA]
    f = 50
    n = 1
    m = (np.exp(4 * f * tk_s * np.log(kappa - 1)) - 1) / (2 * f * tk_s * np.log(kappa - 1))
    m[np.where(kappa > 1.99)] = 0
    ppc["bus_sc"][:, ITH] = ppc["bus_sc"][:, IKSS] * np.sqrt(m + n)
    ppc["branch_sc"][:, [ITH_F, ITH_T]] = _branch_currents_from_bus(ppc, ppc["bus_sc"][:, ITH])

def _branch_currents_from_bus(ppc, current):
    zbus = ppc["internal"]["zbus"]
    Yf = ppc["internal"]["Yf"]
    Yt = ppc["internal"]["Yf"]
    baseI = ppc["internal"]["baseI"]
    V = np.dot(zbus, np.diag(current * baseI))
    fb = np.real(ppc["branch"][:,0]).astype(int)
    tb = np.real(ppc["branch"][:,1]).astype(int)
    i_all_f = abs(np.conj(Yf.dot(V)))
    i_all_t = abs(np.conj(Yt.dot(V)))
    current_from = np.max(i_all_f, axis=1) / baseI[fb] 
    current_to = np.max(i_all_t, axis=1) / baseI[tb]
    return np.c_[current_from, current_to]