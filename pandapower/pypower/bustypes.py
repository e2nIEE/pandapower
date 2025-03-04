# -*- coding: utf-8 -*-

# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


"""Builds index lists of each type of bus.
"""

from numpy import ones, flatnonzero as find, isin
from pandapower.pypower.idx_bus import BUS_TYPE, REF, PV, PQ, BUS_I
from pandapower.pypower.idx_bus_dc import DC_BUS_TYPE, DC_REF, DC_P, DC_B2B
from pandapower.pypower.idx_gen import GEN_BUS, GEN_STATUS
from scipy.sparse import csr_matrix as sparse

from pandapower.pypower.idx_vsc import VSC_MODE_AC, VSC_MODE_AC_SL, VSC_BUS


def bustypes(bus, gen, vsc=None):
    """Builds index lists of each type of bus (C{REF}, C{PV}, C{PQ}).

    Generators with "out-of-service" status are treated as L{PQ} buses with
    zero generation (regardless of C{Pg}/C{Qg} values in gen). Expects C{bus}
    and C{gen} have been converted to use internal consecutive bus numbering.

    @param bus: bus data
    @param gen: generator data
    @return: index lists of each bus type

    @author: Ray Zimmerman (PSERC Cornell)
    @author: Richard Lincoln

    changes by Uni Kassel (Florian Schaefer): If new ref bus is chosen -> Init as numpy array
    """
    # get generator status
#    nb = bus.shape[0]
#    ng = gen.shape[0]
    # gen connection matrix, element i, j is 1 if, generator j at bus i is ON
    #Cg = sparse((gen[:, GEN_STATUS] > 0,
#                 (gen[:, GEN_BUS], range(ng))), (nb, ng))
    # number of generators at each bus that are ON
    #bus_gen_status = (Cg * ones(ng, int)).astype(bool)

    # form index lists for slack, PV, and PQ buses
    if vsc is None:
        ref = find((bus[:, BUS_TYPE] == REF)) # ref bus index
        pv = find((bus[:, BUS_TYPE] == PV))  # PV bus indices
        pq = find((bus[:, BUS_TYPE] == PQ))  # PQ bus indices
    else:
        vsc_ref_bus = vsc[vsc[:, VSC_MODE_AC] == VSC_MODE_AC_SL, VSC_BUS]
        vsc_ref_mask = isin(bus[:, BUS_I], vsc_ref_bus)
        ref = find((bus[:, BUS_TYPE] == REF) & ~vsc_ref_mask)  # ref bus index
        pv  = find((bus[:, BUS_TYPE] == PV)) # PV bus indices
        pq  = find((bus[:, BUS_TYPE] == PQ) | vsc_ref_mask) # PQ bus indices
    return ref, pv, pq


def bustypes_dc(bus_dc):
    """Builds index lists of each type of bus (DC_REF, DC_P).

    Expects bus_dc have been converted to use internal consecutive bus numbering.

    @param bus: bus data
    @return: index lists of each bus type
    """
    # form index lists for slack, PV, and PQ buses
    ref = find((bus_dc[:, DC_BUS_TYPE] == DC_REF))  # ref bus index
    b2b = find((bus_dc[:, DC_BUS_TYPE] == DC_B2B))  # P bus indices
    p = find((bus_dc[:, DC_BUS_TYPE] == DC_P))  # P bus indices
    return ref, b2b, p
