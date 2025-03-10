# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

"""Defines constants for named column indices to SVC matrix.

Some examples of usage, after defining the constants using the lines above,
are::

    svc[3, SVC_STATUS] = 0              # take SVC 4 out of service
    SVC_Q = svc[:, SVC_Q]               # extract reactive power vector

The index, name, and meaning of each column of the SVC matrix are given below:

columns 0-10 must be included in the input matrix (in case file)
    0.  C{SVC_BUS}                          SVC bus number
    1.  C{SVC_X_L}                          inductive reactance (p.u.)
    2.  C{SVC_X_CVAR}                       capacitive reactance (p.u.)
    3.  C{SVC_SET_VM_PU}                    SVC bus voltage magnitude set point (p.u.)
    4.  C{SVC_THYRISTOR_FIRING_ANGLE}       SVC thyristor firing angle (radians)
    5.  C{SVC_STATUS}                       initial SVC status, 1 - in service, 0 - out of service
    6.  C{SVC_CONTROLLABLE}                 controllability flag for SVC
    7.  C{SVC_MIN_FIRING_ANGLE}             minimum thyristor firing angle (radians)
    8.  C{SVC_MAX_FIRING_ANGLE}             maximum thyristor firing angle (radians)
    9.  C{SVC_Q}                            reactive power absorbed (+) or injected (-) by SVC (MVAr)
    10. C{SVC_X_PU}                         reactance in per unit (p.u.)

"""


# define the indices
SVC_BUS = 0  # f, from bus number
SVC_X_L = 1  # (p.u.)
SVC_X_CVAR = 2  # (p.u.)
SVC_SET_VM_PU = 3  # (p.u.)
SVC_THYRISTOR_FIRING_ANGLE = 4
SVC_STATUS = 5  # initial branch status, 1 - in service, 0 - out of service
SVC_CONTROLLABLE = 6
SVC_MIN_FIRING_ANGLE = 7
SVC_MAX_FIRING_ANGLE = 8

SVC_Q = 9
SVC_X_PU = 10

svc_cols = 11
