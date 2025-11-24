# -*- coding: utf-8 -*-

# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

"""
Defines constants for named column indices to BI_VSC matrix.

Some examples of usage, after defining the constants using the lines above,
are::

    BI_VSC[3, BI_VSC_STATUS] = 0                  # take BI_VSC 4 out of service
    BI_VSC_Q = BI_VSC[:, BI_VSC_Q]                   # extract reactive power vector

The index, name, and meaning of each column of the BI_VSC matrix are given below:

columns 0-12 must be included in the input matrix (BI_VSC in ppc dict)
    0.  C{BI_VSC_BUS}                      'from' AC bus number
    1.  C{BI_VSC_INTERNAL_BUS}             Internal bus number for BI_VSC
    2.  C{BI_VSC_INTERNAL_BUS_DC}          Internal DC bus number for BI_VSC
    3.  C{BI_VSC_BUS_DC}                   'to' DC bus number
    4.  C{BI_VSC_R}                        BI_VSC resistance (p.u.)
    5.  C{BI_VSC_X}                        BI_VSC reactance (p.u.)
    6.  C{BI_VSC_R_DC}                     BI_VSC dc-side resistance (p.u.)
    7.  C{BI_VSC_MODE_AC}                  AC mode selector, 0 - vm_pu, 1 - q_mvar, 2 - slack
    8.  C{BI_VSC_VALUE_AC}                 AC mode value (p.u.)
    9.  C{BI_VSC_MODE_DC}                  DC mode selector, 0 - vm_pu, 1 - p_mw
    10  C{BI_VSC_VALUE_DC}                 DC mode value (p.u.)
    11. C{BI_VSC_STATUS}                   Initial BI_VSC status, 1 - in service, 0 - out of service
    12. C{BI_VSC_CONTROLLABLE}             Controllability flag for BI_VSC
    13. C{BI_VSC_P}                        AC bus resultant active power (MW)
    14. C{BI_VSC_Q}                        AC bus resultant reactive power (MVAr)
    15. C{BI_VSC_P_DC}                     DC bus resultant active power (MW)

"""
# define AC modes
BI_VSC_MODE_AC_V = 0
BI_VSC_MODE_AC_Q = 1
BI_VSC_MODE_AC_SL = 2

# define DC modes
BI_VSC_MODE_DC_V = 0
BI_VSC_MODE_DC_P = 1

# define the indices
BI_VSC_BUS = 0  # f, from bus number
BI_VSC_INTERNAL_BUS = 1
BI_VSC_INTERNAL_BUS_DC = 2
BI_VSC_BUS_DC_P = 3  # f, from bus number
BI_VSC_BUS_DC_M = 4  # f, from bus number
BI_VSC_R = 5  # (p.u.)
BI_VSC_X = 6  # (p.u.)
BI_VSC_R_DC = 7  # (p.u.)
BI_VSC_PL_DC = 8  # p.u., specifies the no-load losses (initially input as MW and converted to p.u. for ppc)
BI_VSC_MODE_AC = 9  # 0 - vm_pu, 1 - q_mvar, 2 - slack
BI_VSC_VALUE_AC = 10
BI_VSC_MODE_DC = 11  # 0 - vm_pu, 1 - p_mw
BI_VSC_VALUE_DC = 12
BI_VSC_STATUS = 13  # initial branch status, 1 - in service, 0 - out of service
BI_VSC_CONTROLLABLE = 14

# for results:
BI_VSC_P = 15  # result for P
BI_VSC_Q = 16  # result for Q
BI_VSC_P_DC = 17  # result for P


BI_VSC_cols = 18
