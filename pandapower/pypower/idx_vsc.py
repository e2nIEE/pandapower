# -*- coding: utf-8 -*-

# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

"""
Defines constants for named column indices to VSC matrix.

Some examples of usage, after defining the constants using the lines above,
are::

    vsc[3, VSC_STATUS] = 0                  # take VSC 4 out of service
    VSC_Q = vsc[:, VSC_Q]                   # extract reactive power vector

The index, name, and meaning of each column of the VSC matrix are given below:

columns 0-12 must be included in the input matrix (vsc in ppc dict)
    0.  C{VSC_BUS}                      'from' AC bus number
    1.  C{VSC_INTERNAL_BUS}             Internal bus number for VSC
    2.  C{VSC_INTERNAL_BUS_DC}          Internal DC bus number for VSC
    3.  C{VSC_BUS_DC}                   'to' DC bus number
    4.  C{VSC_R}                        VSC resistance (p.u.)
    5.  C{VSC_X}                        VSC reactance (p.u.)
    6.  C{VSC_R_DC}                     VSC dc-side resistance (p.u.)
    7.  C{VSC_MODE_AC}                  AC mode selector, 0 - vm_pu, 1 - q_mvar
    8.  C{VSC_VALUE_AC}                 AC mode value (p.u.)
    9.  C{VSC_MODE_DC}                  DC mode selector, 0 - vm_pu, 1 - p_mw
    10  C{VSC_VALUE_DC}                 DC mode value (p.u.)
    11. C{VSC_STATUS}                   Initial VSC status, 1 - in service, 0 - out of service
    12. C{VSC_CONTROLLABLE}             Controllability flag for VSC
    13. C{VSC_P}                        AC bus resultant active power (MW)
    14. C{VSC_Q}                        AC bus resultant reactive power (MVAr)
    15. C{VSC_P_DC}                     DC bus resultant active power (MW)

"""

VSC_MODE_AC_V = 0
VSC_MODE_AC_Q = 1
VSC_MODE_AC_SL = 2
VSC_MODE_DC_V = 0
VSC_MODE_DC_P = 1

# define the indices
VSC_BUS = 0  # f, from bus number
VSC_INTERNAL_BUS = 1
VSC_INTERNAL_BUS_DC = 2
VSC_BUS_DC = 3  # f, from bus number
VSC_R = 4  # (p.u.)
VSC_X = 5  # (p.u.)
VSC_R_DC = 6  # (p.u.)
VSC_PL_DC = 7  # p.u., specifies the no-load losses (initially input as MW and converted to p.u. for ppc)
VSC_MODE_AC = 8  # 0 - vm_pu, 1 - q_mvar, 2 - slack
VSC_VALUE_AC = 9
VSC_MODE_DC = 10  # 0 - vm_pu, 1 - p_mw
VSC_VALUE_DC = 11
VSC_STATUS = 12  # initial branch status, 1 - in service, 0 - out of service
VSC_CONTROLLABLE = 13

# for results:
VSC_P = 14  # result for P
VSC_Q = 15  # result for Q
VSC_P_DC = 16  # result for P


vsc_cols = 17
