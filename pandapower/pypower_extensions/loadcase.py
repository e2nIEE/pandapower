# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

"""Loads a PYPOWER case dictionary. Uni Kassel: Deleted unncessary deepcopies and checks
"""

import sys

from os.path import basename, splitext, exists

from numpy import array, zeros, ones, c_

from scipy.io import loadmat

from pypower._compat import PY2
from pypower.idx_gen import PMIN, MU_PMAX, MU_PMIN, MU_QMAX, MU_QMIN, APF
from pypower.idx_brch import PF, QF, PT, QT, MU_SF, MU_ST, BR_STATUS


if not PY2:
    basestring = str


def loadcase(casefile,
        return_as_obj=True, expect_gencost=True, expect_areas=True):
    """Checks if ppc is empty.
    If yes: Print Error
    If not: return ppc as it is
    """
    if return_as_obj == True:
        expect_gencost = False
        expect_areas = False

    # read data into case object

    s = casefile
    # check contents of dict
    # check for required keys
    if (s['baseMVA'] is None or s['bus'] is None \
        or s['gen'] is None or s['branch'] is None) or \
        (expect_gencost and s['gencost'] is None) or \
        (expect_areas and s['areas'] is None):

        ValueError('Syntax error or undefined data '
              'matrix(ices) in the ppc file\n')
    else:
        ## remove empty areas if not needed
        if hasattr(s, 'areas') and (len(s['areas']) == 0) and (not expect_areas):
            del s['areas']

    ## all fields present, copy to ppc
    ppc = s
    if not hasattr(ppc, 'version'):  ## hmm, struct with no 'version' field
        if ppc['gen'].shape[1] < 21:    ## version 2 has 21 or 25 cols
            ppc['version'] = '1'
        else:
            ppc['version'] = '2'

    if (ppc['version'] == '1'):
        # convert from version 1 to version 2
        ppc['gen'], ppc['branch'] = ppc_1to2(ppc['gen'], ppc['branch']);
        ppc['version'] = '2'


    if return_as_obj:
        return ppc
    else:
        result = [ppc['baseMVA'], ppc['bus'], ppc['gen'], ppc['branch']]
        if expect_gencost:
            if expect_areas:
                result.extend([ppc['areas'], ppc['gencost']])
            else:
                result.extend([ppc['gencost']])
        return result



def ppc_1to2(gen, branch):
    ##-----  gen  -----
    ## use the version 1 values for column names
    if gen.shape[1] >= APF:
        sys.stderr.write('ppc_1to2: gen matrix appears to already be in '
                         'version 2 format\n')
        return gen, branch

    shift = MU_PMAX - PMIN - 1
    tmp = array([MU_PMAX, MU_PMIN, MU_QMAX, MU_QMIN]) - shift
    mu_Pmax, mu_Pmin, mu_Qmax, mu_Qmin = tmp

    ## add extra columns to gen
    tmp = zeros((gen.shape[0], shift))
    if gen.shape[1] >= mu_Qmin:
        gen = c_[ gen[:, 0:PMIN + 1], tmp, gen[:, mu_Pmax:mu_Qmin] ]
    else:
        gen = c_[ gen[:, 0:PMIN + 1], tmp ]

    ##-----  branch  -----
    ## use the version 1 values for column names
    shift = PF - BR_STATUS - 1
    tmp = array([PF, QF, PT, QT, MU_SF, MU_ST]) - shift
    Pf, Qf, Pt, Qt, mu_Sf, mu_St = tmp

    ## add extra columns to branch
    tmp = ones((branch.shape[0], 1)) * array([-360, 360])
    tmp2 = zeros((branch.shape[0], 2))
    if branch.shape[1] >= mu_St - 1:
        branch = c_[ branch[:, 0:BR_STATUS + 1], tmp, branch[:, PF - 1:MU_ST + 1], tmp2 ]
    elif branch.shape[1] >= QT - 1:
        branch = c_[ branch[:, 0:BR_STATUS + 1], tmp, branch[:, PF - 1:QT + 1] ]
    else:
        branch = c_[ branch[:, 0:BR_STATUS + 1], tmp ]

    return gen, branch
