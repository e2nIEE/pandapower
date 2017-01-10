# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

import pandas as pd
import numpy as np

def calc_ikss(net, case):
    ikss_pu = net.bus["c_%s"%case] / abs(net.bus.z_equiv)
    net.res_bus_sc = pd.DataFrame(data=pu_to_kA(ikss_pu, net.bus.vn_kv),
                                  index=net.bus.index, columns=["ikss_%s_ka"%case])
    
def calc_ip(net, case, network_structure):
    ikss = net.res_bus_sc["ikss_%s_ka"%case]
    net.res_bus_sc["ip_%s_ka"%case] = net.bus.kappa * np.sqrt(2) * ikss
    
def calc_ith(net, case, tk_s, network_structure):
    f = 50
    n = 1
    m = (np.exp(4 * f * tk_s * np.log(net.bus.kappa.values - 1)) - 1) / \
             (2 * f * tk_s * np.log(net.bus.kappa.values - 1))
    m[np.where(net.bus.kappa > 1.99)] = 0
    net.res_bus_sc["ith_%s_ka"%case] = net.res_bus_sc["ikss_%s_ka"%case]*np.sqrt(m + n)
    
def pu_to_kA(i_pu, Ur_kV, Sr_MVA=1.):
    Ir=Sr_MVA/Ur_kV/np.sqrt(3)
    return i_pu*Ir