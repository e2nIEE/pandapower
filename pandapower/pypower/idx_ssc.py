# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.



# define the indices
SSC_BUS = 0  # f, from bus number
SSC_R = 1  # (p.u.)
SSC_X = 2  # (p.u.)
SSC_SET_VM_PU = 3
SSC_STATUS = 4  # initial branch status, 1 - in service, 0 - out of service
SSC_CONTROLLABLE = 5
SSC_X_CONTROL_VA = 6 # va degrees
SSC_X_CONTROL_VM = 7 # (p.u)  vm

SSC_Q = 8 # result for Q
SSC_INTERNAL_BUS = 9

ssc_cols  = 10