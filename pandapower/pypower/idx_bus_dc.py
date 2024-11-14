# -*- coding: utf-8 -*-

# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

"""
Defines constants for named column indices to DC bus matrix.

Some examples of usage, after defining the constants using the line above,
are::

    Pd = bus_dc[3, DC_PD]     # get the real power demand at DC bus 4
    bus_dc[:, DC_VMIN] = 0.95 # set the min voltage magnitude to 0.95 at all DC buses

The index, name, and meaning of each column of the DC bus matrix is given below:

columns 0-10 must be included in the input matrix (bus_dc in ppc dict)
    0.  C{DC_BUS_I}       DC bus number (1 to 29997)
    1.  C{DC_BUS_TYPE}    DC bus type (0 = reference, 1 = P, 2 = isolated)
    2.  C{DC_PD}          real power demand (MW)
    3.  C{DC_GS}          shunt conductance (MW at V = 1.0 p.u.)
    4.  C{DC_BUS_AREA}    area number, 1-100
    5.  C{DC_VM}          voltage magnitude (p.u.)
    6.  C{DC_BASE_KV}     base voltage (kV)
    7.  C{DC_ZONE}        loss zone (1-999)
    8.  C{DC_VMAX}        maximum voltage magnitude (p.u.)
    9.  C{DC_VMIN}        minimum voltage magnitude (p.u.)
    10. C{DC_SL_FAC}      scaling factor for special purposes

additional constants, used to assign/compare values in the C{DC_BUS_TYPE} column
    0.  C{DC_REF}   reference DC bus
    1.  C{DC_P}     DC bus with active power demand
    2.  C{DC_NONE}  DC bus that is out of service

"""

# define DC bus types
DC_REF = 0
DC_B2B = 1
DC_P = 2
DC_NONE = 3

# define the indices
DC_BUS_I = 0    # DC bus number (1 to 29997)
DC_BUS_TYPE: int = 1    # DC bus type
DC_PD = 2    # Pd, real power demand (MW)
DC_GS = 3    # Gs, shunt conductance (MW at V = 1.0 p.u.)
DC_BUS_AREA = 4    # area number, 1-100
DC_VM = 5    # Vm, voltage magnitude (p.u.)
DC_BASE_KV = 6    # baseKV, base voltage (kV)
DC_ZONE = 7   # zone, loss zone (1-999)
DC_VMAX = 8   # maxVm, maximum voltage magnitude (p.u.)
DC_VMIN = 9   # minVm, minimum voltage magnitude (p.u.)
DC_SL_FAC = 10   # scaling factor for special purposes

dc_bus_cols = 11
