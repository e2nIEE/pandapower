# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


# define the indices
from pandapower.pypower.idx_bus import bus_cols as start

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
R_EQUIV_OHM = start + 13
X_EQUIV_OHM = start + 14
K_G         = start + 15
K_SG        = start + 16
V_G         = start + 17
PS_TRAFO_IX = start + 18
GS_P        = start + 19
BS_P        = start + 20
GS_GEN      = start + 21
BS_GEN      = start + 22
SKSS        = start + 23

PHI_IKSS1_DEGREE = start + 24
PHI_IKSS2_DEGREE = start + 25
PHI_IKCV_DEGREE  = start + 26

bus_cols_sc = 27