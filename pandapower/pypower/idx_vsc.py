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
    1.. C{VSC_INTERNAL_BUS}             Internal bus number for VSC
    2.  C{VSC_BUS_DC}                   'to' DC bus number
    3.  C{VSC_R}                        VSC resistance (p.u.)
    4.  C{VSC_X}                        VSC reactance (p.u.)
    5.  C{VSC_MODE_AC}                  AC mode selector, 0 - vm_pu, 1 - q_mvar
    6.  C{VSC_VALUE_AC}                 AC mode value (p.u.)
    7.  C{VSC_MODE_DC}                  DC mode selector, 0 - vm_pu, 1 - p_mw
    8.  C{VSC_VALUE_DC}                 DC mode value (p.u.)
    9.  C{VSC_STATUS}                   Initial VSC status, 1 - in service, 0 - out of service
    10  C{VSC_CONTROLLABLE}             Controllability flag for VSC
    11. C{VSC_P}                        AC bus resultant active power (MW)
    12. C{VSC_Q}                        AC bus resultant reactive power (MVAr)
    13. C{VSC_P_DC}                     DC bus resultant active power (MW)

"""

# define the indices
VSC_BUS = 0  # f, from bus number
VSC_INTERNAL_BUS = 1
VSC_BUS_DC = 2  # f, from bus number
VSC_R = 3  # (p.u.)
VSC_X = 4  # (p.u.)
VSC_MODE_AC = 5  # 0 - vm_pu, 1 - q_mvar
VSC_VALUE_AC = 6
VSC_MODE_DC = 7  # 0 - vm_pu, 1 - p_mw
VSC_VALUE_DC = 8
VSC_STATUS = 9  # initial branch status, 1 - in service, 0 - out of service
VSC_CONTROLLABLE = 10

# for results:
VSC_P = 11  # result for P
VSC_Q = 12  # result for Q
VSC_P_DC = 13  # result for P


vsc_cols = 14
