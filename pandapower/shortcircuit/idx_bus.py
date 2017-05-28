# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

# define the indices
from pandapower.idx_bus import bus_cols as start

R_EQUIV     = start + 0
X_EQUIV     = start + 1
KAPPA       = start + 2
M           = start + 3
C_MIN       = start + 4
C_MAX       = start + 5
IKSS1       = start + 6
IKSS2       = start + 7
IKCV        = start + 8
IP          = start + 9
IB          = start + 10
ITH         = start + 11
IK          = start + 12

bus_cols_sc = 13