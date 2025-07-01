# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


"""Defines constants for named column indices to TCSC matrix.

Some examples of usage, after defining the constants using the line above,
are:

    tcsc[3, TCSC_STATUS] = 0              # take TCSC 4 out of service
    Pset = tcsc[:, TCSC_SET_P]            # extract TCSC power set point vector

The index, name and meaning of each column of the TCSC matrix is given
below:

columns 0-9 must be included in input matrix (tcsc in ppc dict)
    0.  C{TCSC_F_BUS}                       TCSC from bus number
    1.  C{TCSC_T_BUS}                       TCSC to bus number
    2.  C{TCSC_X_L}                         inductive reactance (p.u.)
    3.  C{TCSC_X_CVAR}                      capacitive reactance (p.u.)
    4.  C{TCSC_SET_P}                       TCSC active power set point (p.u.)
    5.  C{TCSC_THYRISTOR_FIRING_ANGLE}      TCSC thyristor firing angle (radians)
    6.  C{TCSC_STATUS}                      initial TCSC status, 1 - in service, 0 - out of service
    7.  C{TCSC_CONTROLLABLE}                controllability flag for TCSC
    8.  C{TCSC_MIN_FIRING_ANGLE}            minimum thyristor firing angle (radians)
    9.  C{TCSC_MAX_FIRING_ANGLE}            maximum thyristor firing angle (radians)

columns 10-16 are filled with values after power flow

    10. C{PF}                               real power injected at "from" bus end (MW)
    11. C{QF}                               reactive power injected at "from" bus end (MVAr)
    12. C{PT}                               real power injected at "to" bus end (MW)
    13. C{QT}                               reactive power injected at "to" bus end (MVAr)
    14. C{IF}                               current injected at "from" bus end (kA)
    15. C{IT}                               current injected at "to" bus end (kA)
    16. C{TCSC_X_PU}                        reactance in per unit (p.u.)

"""

# define the indices
TCSC_F_BUS = 0  # f, from bus number
TCSC_T_BUS = 1  # t, to bus number
TCSC_X_L = 2  # (p.u.)
TCSC_X_CVAR = 3  # (p.u.)
TCSC_SET_P = 4  # (p.u.)
TCSC_THYRISTOR_FIRING_ANGLE = 5
TCSC_STATUS = 6  # initial branch status, 1 - in service, 0 - out of service
TCSC_CONTROLLABLE = 7
TCSC_MIN_FIRING_ANGLE = 8
TCSC_MAX_FIRING_ANGLE = 9

TCSC_PF = 10
TCSC_QF = 11
TCSC_PT = 12
TCSC_QT = 13

TCSC_IF = 14
TCSC_IT = 15
TCSC_X_PU = 16

tcsc_cols = 17
