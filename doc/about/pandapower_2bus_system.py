# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a 
# BSD-style license that can be found in the LICENSE file.

import pandapower as pp
import pandas as pd
pd.set_option('display.width', 800)
pd.set_option('display.precision', 2)

#initialize datastructure
net = pp.create_empty_network()

#create_buses
b1 = pp.create_bus(net, vn_kv=0.4, name="Bus 1")
b2 = pp.create_bus(net, vn_kv=0.4, name="Bus 2")

#create slack
pp.create_ext_grid(net, bus=b1, vm_pu=1.02, name="Grid Connection")

#create_load
pp.create_load(net, bus=b2, p_kw=100, q_kvar=50, name="Load")

#create line by standard type
lid = pp.create_line(net, from_bus=b1, to_bus=b2, length_km=0.1,
                     std_type="NAYY 4x50 SE", name="Line")     

#run loadflow
pp.runpp(net)


#print element tables
print("-------------------")
print("  ELEMENT TABLES   ")
print("-------------------")

print("net.bus")
print(net.bus)

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
print("\n net.res_line")
print(net.res_line)
print("\n net.res_load")
print(net.res_load)
print("\n net.res_ext_grid")
print(net.res_ext_grid)