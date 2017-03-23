# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import pandas as pd

import pandapower as pp

pd.set_option('display.width', 1500)
pd.set_option('display.precision', 4)

#initialize datastructure
net = pp.create_empty_network()

#create_buses
b1 = pp.create_bus(net, vn_kv=20., name="Bus 1")
b2 = pp.create_bus(net, vn_kv=0.4, name="Bus 2")
b3 = pp.create_bus(net, vn_kv=0.4, name="Bus 3")

#create slack
pp.create_ext_grid(net, bus=b1, vm_pu=1.02, name="Grid Connection")

#create_load
pp.create_load(net, bus=b3, p_kw=100, q_kvar=50, name="Load")

tid = pp.create_transformer(net, hv_bus=b1, lv_bus=b2, std_type="0.4 MVA 20/0.4 kV")
#tid = pp.create_transformer_from_parameters(net,
#                                            hv_bus=b1,
#                                            lv_bus=b2,
#                                            sn_kva=400.,
#                                            vn_hv_kv=20.,
#                                            vn_lv_kv=0.4,
#                                            vsc_percent=6.,
#                                            vscr_percent=1.425,
#                                            i0_percent=0.3375,
#                                            pfe_kw=1.35)
#create line by standard type
lid = pp.create_line(net, from_bus=b2, to_bus=b3, length_km=0.1,
                     std_type="NAYY 4x50 SE", name="Line")     

#run loadflow
pp.runpp(net)


#print element tables
print("-------------------")
print("  ELEMENT TABLES   ")
print("-------------------")

print("net.bus")
print(net.bus)

print("\n net.trafo")
print(net.trafo)

print("\n net.line")
print(net.line)

print("\n net.load")
print(net.load)

print("\n net.ext_grid")
print(net.ext_grid)

#print result tables
print("\n-------------------")
print("   RESULT TABLES   ")
print("-------------------")

print("net.res_bus")
print(net.res_bus)

print("\n net.res_trafo")
print(net.res_trafo)

print("\n net.res_line")
print(net.res_line)

print("\n net.res_load")
print(net.res_load)

print("\n net.res_ext_grid")
print(net.res_ext_grid)