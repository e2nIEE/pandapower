# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
from pandapower.shortcircuit.idx_bus import BASE_KV, C_MIN, C_MAX, KAPPA, R_EQUIV, IKSS, IP, ITH,\
                                            X_EQUIV
import numpy as np

def calc_ikss(net, ppc):
    case = net._options["case"]
    c = ppc["bus"][:, C_MIN] if case == "min" else ppc["bus"][:, C_MAX]
    z_equiv = abs(ppc["bus"][:, R_EQUIV] + ppc["bus"][:, X_EQUIV] *1j)
    ppc["bus"][:, IKSS] = c / z_equiv / np.sqrt(3) / ppc["bus"][:, BASE_KV] * ppc["baseMVA"]
    
def calc_ip(ppc):
    ppc["bus"][:, IP] = ppc["bus"][:, KAPPA] * np.sqrt(2) * ppc["bus"][:, IKSS]
    
def calc_ith(net, ppc):
    tk_s = net["_options"]["tk_s"]
    kappa = ppc["bus"][:, KAPPA]
    f = 50
    n = 1
    m = (np.exp(4 * f * tk_s * np.log(kappa - 1)) - 1) / (2 * f * tk_s * np.log(kappa - 1))
    m[np.where(kappa > 1.99)] = 0
    ppc["bus"][:, ITH] = ppc["bus"][:, IKSS] * np.sqrt(m + n)
    
#def pu_to_ka(i_pu, vn_kv, sn_kva):
#    in_ka = sn_kva / vn_kv 
#    return i_pu * in_ka