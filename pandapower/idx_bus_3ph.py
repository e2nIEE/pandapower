# -*- coding: utf-8 -*-

# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


# define the indices
from pandapower.idx_bus import bus_cols as start

VM_A = start + 0    # Vm, voltage magnitude (p.u.)
VA_A = start + 1    # Va, voltage angle (degrees)
VM_B = start + 2    # Vm, voltage magnitude (p.u.)
VA_B = start + 3    # Va, voltage angle (degrees)
VM_C = start + 4    # Vm, voltage magnitude (p.u.)
VA_C = start + 5    # Va, voltage angle (degrees)

bus_cols_3ph = 6