# -*- coding: utf-8 -*-

# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


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
