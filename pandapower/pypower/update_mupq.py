# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Updates values of generator limit shadow prices.
"""

from pypower.idx_gen import MU_PMAX, MU_PMIN, MU_QMAX, MU_QMIN


def update_mupq(baseMVA, gen, mu_PQh, mu_PQl, data):
    """Updates values of generator limit shadow prices.

    Updates the values of C{MU_PMIN}, C{MU_PMAX}, C{MU_QMIN}, C{MU_QMAX} based
    on any shadow prices on the sloped portions of the generator
    capability curve constraints.

    @param mu_PQh: shadow prices on upper sloped portion of capability curves
    @param mu_PQl: shadow prices on lower sloped portion of capability curves
    @param data: "data" dict returned by L{makeApq}

    @see: C{makeApq}.

    @author: Ray Zimmerman (PSERC Cornell)
    @author: Carlos E. Murillo-Sanchez (PSERC Cornell & Universidad
    Autonoma de Manizales)
    """
    ## extract the constraint parameters
    ipqh, ipql, Apqhdata, Apqldata = \
        data['ipqh'], data['ipql'], data['h'], data['l']

    ## combine original limit multipliers into single value
    muP = gen[:, MU_PMAX] - gen[:, MU_PMIN]
    muQ = gen[:, MU_QMAX] - gen[:, MU_QMIN]

    ## add P and Q components of multipliers on upper sloped constraint
    muP[ipqh] = muP[ipqh] - mu_PQh * Apqhdata[:, 0] / baseMVA
    muQ[ipqh] = muQ[ipqh] - mu_PQh * Apqhdata[:, 1] / baseMVA

    ## add P and Q components of multipliers on lower sloped constraint
    muP[ipql] = muP[ipql] - mu_PQl * Apqldata[:, 0] / baseMVA
    muQ[ipql] = muQ[ipql] - mu_PQl * Apqldata[:, 1] / baseMVA

    # split back into upper and lower multipliers based on sign
    gen[:, MU_PMAX] = (muP > 0) *  muP
    gen[:, MU_PMIN] = (muP < 0) * -muP
    gen[:, MU_QMAX] = (muQ > 0) *  muQ
    gen[:, MU_QMIN] = (muQ < 0) * -muQ

    return gen
