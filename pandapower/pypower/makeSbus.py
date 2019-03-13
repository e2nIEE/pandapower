# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Builds the vector of complex bus power injections.
"""

from numpy import ones, flatnonzero as find
from pandapower.pypower.idx_bus import PD, QD
from pandapower.pypower.idx_gen import GEN_BUS, PG, QG, GEN_STATUS
from scipy.sparse import csr_matrix as sparse

from pandapower.pypower.idx_bus import CID, CZD


def _get_Sbus(baseMVA, bus, gen_on, Cg, vm=None):
    # power injected by gens plus power injected by loads converted to p.u.
    S_load = _get_Sload(bus, vm)
    Sbus = (Cg * (gen_on[:, PG] + 1j * gen_on[:, QG])
            - S_load) / baseMVA
    return Sbus


def _get_Sload(bus, vm):
    S_load = bus[:, PD] + 1j * bus[:, QD]
    if vm is not None:
        ci = bus[:, CID]
        cz = bus[:, CZD]
        cp = (1 - ci - cz)
        volt_depend = cp + ci * vm + cz * vm ** 2
        S_load *= volt_depend
    return S_load


def _get_Cg(gen_on, bus):
    gbus = gen_on[:, GEN_BUS]  ## what buses are they at?

    ## form net complex bus power injection vector
    nb = bus.shape[0]
    ngon = gen_on.shape[0]
    return sparse((ones(ngon), (gbus, range(ngon))), (nb, ngon))


def makeSbus(baseMVA, bus, gen, vm=None):
    """Builds the vector of complex bus power injections.

    Returns the vector of complex bus power injections, that is, generation
    minus load. Power is expressed in per unit.

    @see: L{makeYbus}

    @author: Ray Zimmerman (PSERC Cornell)
    @author: Richard Lincoln
    """
    ## generator info
    on = find(gen[:, GEN_STATUS] > 0)  ## which generators are on?
    gen_on = gen[on, :]

    ## connection matrix, element i, j is 1 if gen on(j) at bus i is ON
    Cg = _get_Cg(gen_on, bus)

    ## power injected by gens plus power injected by loads converted to p.u.
    Sbus = _get_Sbus(baseMVA, bus, gen_on, Cg, vm)

    return Sbus
