# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Construct constraints for branch angle difference limits.
"""

from numpy import array, ones, zeros, r_, Inf, pi, arange
from numpy import flatnonzero as find
from scipy.sparse import csr_matrix as sparse

from pypower.idx_brch import F_BUS, T_BUS, ANGMIN, ANGMAX


def makeAang(baseMVA, branch, nb, ppopt):
    """Construct constraints for branch angle difference limits.

    Constructs the parameters for the following linear constraint limiting
    the voltage angle differences across branches, where C{Va} is the vector
    of bus voltage angles. C{nb} is the number of buses::

        lang <= Aang * Va <= uang

    C{iang} is the vector of indices of branches with angle difference limits.

    @author: Ray Zimmerman (PSERC Cornell)
    @author: Carlos E. Murillo-Sanchez (PSERC Cornell & Universidad
    Autonoma de Manizales)
    """
    ## options
    ignore_ang_lim = ppopt['OPF_IGNORE_ANG_LIM']

    if ignore_ang_lim:
        Aang  = zeros((0, nb))
        lang  = array([])
        uang  = array([])
        iang  = array([])
    else:
        iang = find(((branch[:, ANGMIN] != 0) & (branch[:, ANGMIN] > -360)) |
                    ((branch[:, ANGMAX] != 0) & (branch[:, ANGMAX] <  360)))
        iangl = find(branch[iang, ANGMIN])
        iangh = find(branch[iang, ANGMAX])
        nang = len(iang)

        if nang > 0:
            ii = r_[arange(nang), arange(nang)]
            jj = r_[branch[iang, F_BUS], branch[iang, T_BUS]]
            Aang = sparse((r_[ones(nang), -ones(nang)],
                           (ii, jj)), (nang, nb))
            uang = Inf * ones(nang)
            lang = -uang
            lang[iangl] = branch[iang[iangl], ANGMIN] * pi / 180
            uang[iangh] = branch[iang[iangh], ANGMAX] * pi / 180
        else:
            Aang  = zeros((0, nb))
            lang  = array([])
            uang  = array([])

    return Aang, lang, uang, iang
