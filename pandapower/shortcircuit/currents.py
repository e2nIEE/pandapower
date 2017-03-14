# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

import numpy as np

def calc_ikss(net):
    bus = net._is_elems["bus"]
    case = net["_options"]["case"]
    name = "ikss_%s_ka"%case
    ikss_pu = bus["c_%s"%case] / abs(bus.z_equiv)
    ikss_ka = pu_to_ka(ikss_pu, bus.vn_kv, net.sn_kva)
    bus[name] = ikss_ka
    net._options["currents"].append(name)
    
def calc_ip(net):
    bus = net._is_elems["bus"]
    case = net["_options"]["case"]
    name = "ip_%s_ka"%case
    ikss = bus["ikss_%s_ka"%case]
    bus[name] = bus.kappa * np.sqrt(2) * ikss
    net._options["currents"].append(name)
    
def calc_ith(net):
    bus = net._is_elems["bus"]
    case = net["_options"]["case"]
    name = "ith_%s_ka"%case
    tk_s = net["_options"]["tk_s"]
    f = 50
    n = 1
    m = (np.exp(4 * f * tk_s * np.log(bus.kappa.values - 1)) - 1) / \
             (2 * f * tk_s * np.log(bus.kappa.values - 1))
    m[np.where(bus.kappa > 1.99)] = 0
    bus[name] = bus["ikss_%s_ka"%case]*np.sqrt(m + n)
    net._options["currents"].append(name)
    
def pu_to_ka(i_pu, vn_kv, sn_kva):
    in_ka = sn_kva / vn_kv / np.sqrt(3) * 1e-3
    return i_pu * in_ka