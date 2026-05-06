# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

""" Defines constants for named column indices to gen matrix.

The index, name and meaning of each column of the gen matrix is given
below:

columns 0-20 must be included in input matrix (in case file)
    0.  C{SOURCE_DC_BUS}    bus number
    1.  C{PG}               real power output (MW)
    2.  C{VG}               voltage magnitude setpoint (p.u.)
    3.  C{MBASE}            total MVA base of machine, defaults to baseMVA
    4.  C{SOURCE_DC_STATUS} 1 - in service, 0 - out of service
    5.  C{PMAX}             maximum real power output (MW)
    6.  C{PMIN}             minimum real power output (MW)
    7.  C{RAMP_AGC}         ramp rate for load following/AGC (MW/min)
    8.  C{RAMP_10}          ramp rate for 10 minute reserves (MW)
    9.  C{RAMP_30}          ramp rate for 30 minute reserves (MW)
    10. C{APF}              area participation factor

columns 21-24 are added to matrix after OPF solution
they are typically not present in the input matrix

(assume OPF objective function has units, u)
    11. C{MU_PMAX}     Kuhn-Tucker multiplier on upper Pg limit (u/MW)
    12. C{MU_PMIN}     Kuhn-Tucker multiplier on lower Pg limit (u/MW)

@author: Mike Vogt
"""

# define the indices
SOURCE_DC_BUS       = 0    # bus number
SOURCE_DC_PG        = 1    # Pg, real power output (MW)
SOURCE_DC_VG        = 2    # Vg, voltage magnitude setpoint (p.u.)
SOURCE_DC_MBASE     = 3    # mBase, total MVA base of this machine, defaults to baseMVA
SOURCE_DC_STATUS    = 4    # status, 1 - machine in service, 0 - machine out of service
SOURCE_DC_PMAX      = 5    # Pmax, maximum real power output (MW)
SOURCE_DC_PMIN      = 6    # Pmin, minimum real power output (MW)
SOURCE_DC_RAMP_AGC  = 7    # ramp rate for load following/AGC (MW/min)
SOURCE_DC_RAMP_10   = 8    # ramp rate for 10 minute reserves (MW)
SOURCE_DC_RAMP_30   = 9    # ramp rate for 30 minute reserves (MW)
SOURCE_DC_APF       = 10   # area participation factor

# included in opf solution, not necessarily in input
# assume objective function has units, u
SOURCE_DC_MU_PMAX   = 11   # Kuhn-Tucker multiplier on upper Pg limit (u/MW)
SOURCE_DC_MU_PMIN   = 12   # Kuhn-Tucker multiplier on lower Pg limit (u/MW)

# Additional added by pandapower
SOURCE_SL_FAC       = 13   # Slack contribution factor

source_dc_cols = 14
