# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from pandapower.pypower.idx_brch import branch_cols as start

# columns for TDPF

TDPF 						= start + 0   # bool flag (True or False) of whether the branch must be considered in the TDPF power flow
BR_R_REF_OHM_PER_KM 		= start + 1   # reference R is needed because BR_R in ppc can already be adjusted by temperature
BR_LENGTH_KM 				= start + 3   # to calculate R_ref in p.u.
RATE_I_KA 					= start + 4   # rated I limit in kA (max_i_ka)
T_START_C 					= start + 5   # starting temperature of the conductor in °C
T_REF_C 					= start + 6   # reference temperature for calculating updated R (optional)
T_AMBIENT_C 				= start + 7   # temperature of the surrounding air
ALPHA 						= start + 8   # thermal coefficient of resistance (4.03e-3 for Al)
WIND_SPEED_MPS 				= start + 9   # Wind speed in meters per second
WIND_ANGLE_DEGREE 			= start + 10  # Angle of attack, angle between wind direction and conductor
SOLAR_RADIATION_W_PER_SQ_M 	= start + 11  # Solar radiation in W/m²
GAMMA 						= start + 12  # absorption albedo, solar absorptivity
EPSILON 					= start + 13  # radiative albedo, emissivity
R_THETA 					= start + 14  # Thermal resistance R_Theta in K/MW (Optional)
OUTER_DIAMETER_M 			= start + 15  # outer diameter of the conductor
MC_JOULE_PER_M_K 			= start + 16  # thermal inertia parameter of the conductor

branch_cols_tdpf 			= 17
