# -*- coding: utf-8 -*-

# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


# define the indices
VSC_BUS = 0  # f, from bus number
VSC_BUS_DC = 1  # f, from bus number
VSC_R = 2  # (p.u.)
VSC_X = 3  # (p.u.)
VSC_MODE_AC = 4  # 0 - vm_pu, 1 - q_mvar
VSC_VALUE_AC = 5
VSC_MODE_DC = 6  # 0 - vm_pu, 1 - p_mw
VSC_VALUE_DC = 7
VSC_STATUS = 8  # initial branch status, 1 - in service, 0 - out of service
VSC_CONTROLLABLE = 9

# for results:
VSC_X_CONTROL_VA = 10  # va degrees
VSC_X_CONTROL_VM = 11  # (p.u)  vm
VSC_P = 12  # result for P
VSC_Q = 13  # result for Q
VSC_P_DC = 14  # result for P
VSC_INTERNAL_BUS = 15

vsc_cols = 16
