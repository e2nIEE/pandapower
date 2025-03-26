# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

"""
Defines constants for named column indices to SSC matrix.

Some examples of usage, after defining the constants using the lines above,
are::

    ssc[3, SSC_STATUS] = 0                  # take SSC 4 out of service
    SSC_Q = ssc[:, SSC_Q]                   # extract reactive power vector

The index, name, and meaning of each column of the SSC matrix are given below:

columns 0-9 must be included in the input matrix (ssc in ppc dict)
    0.  C{SSC_BUS}                      SSC bus number
    1.  C{SSC_INTERNAL_BUS}             Internal bus number for SSC
    2.  C{SSC_R}                        SSC coupling transformer resistance (p.u.)
    3.  C{SSC_X}                        SSC coupling transformer reactance (p.u.)
    4.  C{SSC_SET_VM_PU}                SSC bus voltage magnitude set point (p.u.)
    5.  C{SSC_STATUS}                   Initial SSC status, 1 - in service, 0 - out of service
    6.  C{SSC_CONTROLLABLE}             Controllability flag for SSC
    7.  C{SSC_X_CONTROL_VA}             SSC internal voltage angle (radians)
    8.  C{SSC_X_CONTROL_VM}             SSC internal voltage magnitude (p.u.)
    9.  C{SSC_Q}                        reactive power absorbed (+) or injected (-) by SSC (MVAr)

"""

# define the indices
SSC_BUS = 0  # f, from bus number
SSC_INTERNAL_BUS = 1
SSC_R = 2  # (p.u.)
SSC_X = 3  # (p.u.)
SSC_SET_VM_PU = 4
SSC_STATUS = 5  # initial branch status, 1 - in service, 0 - out of service
SSC_CONTROLLABLE = 6
SSC_X_CONTROL_VA = 7 # va degrees ## check with roman
SSC_X_CONTROL_VM = 8 # (p.u)  vm
SSC_Q = 9 # result for Q

ssc_cols  = 10