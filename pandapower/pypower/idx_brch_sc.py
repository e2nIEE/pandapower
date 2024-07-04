# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from pandapower.pypower.idx_brch import branch_cols as start

# todo: add comments to describe the variables!

IKSS_F    = start + 0
IKSS_T    = start + 1
ITH_F     = start + 2
ITH_T     = start + 3
IP_F      = start + 4
IP_T      = start + 5
IK_F      = start + 6
IK_T      = start + 7
IB_F      = start + 8
IB_T      = start + 9
K_T       = start + 10
K_ST       = start + 11


PKSS_F    = start + 12
QKSS_F    = start + 13
PKSS_T    = start + 14
QKSS_T    = start + 15
VKSS_MAGN_F    = start + 16
VKSS_MAGN_T    = start + 17
VKSS_ANGLE_F = start + 18
VKSS_ANGLE_T = start + 19
IKSS_ANGLE_F = start + 20
IKSS_ANGLE_T = start + 21

branch_cols_sc = 22
