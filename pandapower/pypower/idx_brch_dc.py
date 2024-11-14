# -*- coding: utf-8 -*-

# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


"""Defines constants for named column indices to branch matrix.

Some examples of usage, after defining the constants using the line above,
are::

    branch[3, BR_STATUS] = 0              # take branch 4 out of service
    Ploss = branch[:, PF] + branch[:, PT] # compute real power loss vector

The index, name and meaning of each column of the branch matrix is given
below:

columns 0-10 must be included in input matrix (in case file)
    0.  C{F_BUS}       from bus number
    1.  C{T_BUS}       to bus number
    2.  C{BR_R}        resistance (p.u.)
    3.  C{BR_X}        reactance (p.u.)
    4.  C{BR_B}        total line charging susceptance (p.u.)
    5.  C{RATE_A}      MVA rating A (long term rating)
    6.  C{RATE_B}      MVA rating B (short term rating)
    7.  C{RATE_C}      MVA rating C (emergency rating)
    8.  C{TAP}         transformer off nominal turns ratio
    9.  C{SHIFT}       transformer phase shift angle (degrees)
    10. C{BR_STATUS}   initial branch status, 1 - in service, 0 - out of service
    11. C{ANGMIN}      minimum angle difference, angle(Vf) - angle(Vt) (degrees)
    12. C{ANGMAX}      maximum angle difference, angle(Vf) - angle(Vt) (degrees)

columns 13-16 are added to matrix after power flow or OPF solution
they are typically not present in the input matrix
    13. C{PF}          real power injected at "from" bus end (MW)
    14. C{QF}          reactive power injected at "from" bus end (MVAr)
    15. C{PT}          real power injected at "to" bus end (MW)
    16. C{QT}          reactive power injected at "to" bus end (MVAr)

columns 17-18 are added to matrix after OPF solution
they are typically not present in the input matrix

(assume OPF objective function has units, C{u})
    17. C{MU_SF}       Kuhn-Tucker multiplier on MVA limit at "from" bus (u/MVA)
    18. C{MU_ST}       Kuhn-Tucker multiplier on MVA limit at "to" bus (u/MVA)

columns 19-20 are added to matrix after SCOPF solution
they are typically not present in the input matrix

(assume OPF objective function has units, C{u})
    19. C{MU_ANGMIN}   Kuhn-Tucker multiplier lower angle difference limit
    20. C{MU_ANGMAX}   Kuhn-Tucker multiplier upper angle difference limit

@author: Ray Zimmerman (PSERC Cornell)
@author: Richard Lincoln
"""

# define the indices
DC_F_BUS       = 0    # f, from bus number
DC_T_BUS       = 1    # t, to bus number
DC_BR_R        = 2    # r, resistance (p.u.)
DC_BR_G        = 3    # b, total line charging susceptance (p.u.)  # todo Roman check if necessary
DC_RATE_A      = 4    # rateA, MVA rating A (long term rating)
DC_RATE_B      = 5    # rateB, MVA rating B (short term rating)
DC_RATE_C      = 6    # rateC, MVA rating C (emergency rating)
DC_BR_STATUS   = 7   # initial branch status, 1 - in service, 0 - out of service

# included in power flow solution, not necessarily in input
DC_PF          = 8   # real power injected at "from" bus end (MW)
DC_IF          = 9   # current injected at "from" bus end (p.u.)
DC_PT          = 10   # real power injected at "to" bus end (MW)
DC_IT          = 11   # current injected at "to" bus end (p.u.)

DC_BR_R_ASYM = 12   # todo Roman check if necessary
DC_BR_X_ASYM = 13   # todo Roman check if necessary

DC_TDPF = 14  ### TDPF not implemented for DC lines

branch_dc_cols = 15
