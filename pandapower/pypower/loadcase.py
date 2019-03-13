# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Loads a PYPOWER case dictionary.
"""

import sys

from os.path import basename, splitext, exists

from copy import deepcopy

from numpy import array, zeros, ones, c_

from scipy.io import loadmat

from pypower._compat import PY2
from pypower.idx_gen import PMIN, MU_PMAX, MU_PMIN, MU_QMAX, MU_QMIN, APF
from pypower.idx_brch import PF, QF, PT, QT, MU_SF, MU_ST, BR_STATUS


if not PY2:
    basestring = str


def loadcase(casefile,
        return_as_obj=True, expect_gencost=True, expect_areas=True):
    """Returns the individual data matrices or an dict containing them
    as values.

    Here C{casefile} is either a dict containing the keys C{baseMVA}, C{bus},
    C{gen}, C{branch}, C{areas}, C{gencost}, or a string containing the name
    of the file. If C{casefile} contains the extension '.mat' or '.py', then
    the explicit file is searched. If C{casefile} containts no extension, then
    L{loadcase} looks for a '.mat' file first, then for a '.py' file.  If the
    file does not exist or doesn't define all matrices, the function returns
    an exit code as follows:

        0.  all variables successfully defined
        1.  input argument is not a string or dict
        2.  specified extension-less file name does not exist
        3.  specified .mat file does not exist
        4.  specified .py file does not exist
        5.  specified file fails to define all matrices or contains syntax
            error

    If the input data is not a dict containing a 'version' key, it is
    assumed to be a PYPOWER case file in version 1 format, and will be
    converted to version 2 format.

    @author: Carlos E. Murillo-Sanchez (PSERC Cornell & Universidad
    Autonoma de Manizales)
    @author: Ray Zimmerman (PSERC Cornell)
    """
    if return_as_obj == True:
        expect_gencost = False
        expect_areas = False

    info = 0

    # read data into case object
    if isinstance(casefile, basestring):
        # check for explicit extension
        if casefile.endswith(('.py', '.mat')):
            rootname, extension = splitext(casefile)
            fname = basename(rootname)
        else:
            # set extension if not specified explicitly
            rootname = casefile
            if exists(casefile + '.mat'):
                extension = '.mat'
            elif exists(casefile + '.py'):
                extension = '.py'
            else:
                info = 2
            fname = basename(rootname)

        lasterr = ''

        ## attempt to read file
        if info == 0:
            if extension == '.mat':       ## from MAT file
                try:
                    d = loadmat(rootname + extension, struct_as_record=True)
                    if 'ppc' in d or 'mpc' in d:    ## it's a MAT/PYPOWER dict
                        if 'ppc' in d:
                            struct = d['ppc']
                        else:
                            struct = d['mpc']
                        val = struct[0, 0]

                        s = {}
                        for a in val.dtype.names:
                            s[a] = val[a]
                    else:                 ## individual data matrices
                        d['version'] = '1'

                        s = {}
                        for k, v in d.items():
                            s[k] = v

                    s['baseMVA'] = s['baseMVA'][0]  # convert array to float

                except IOError as e:
                    info = 3
                    lasterr = str(e)
            elif extension == '.py':      ## from Python file
                try:
                    if PY2:
                        execfile(rootname + extension)
                    else:
                        exec(compile(open(rootname + extension).read(),
                                     rootname + extension, 'exec'))

                    try:                      ## assume it returns an object
                        s = eval(fname)()
                    except ValueError as e:
                        info = 4
                        lasterr = str(e)
                    ## if not try individual data matrices
                    if info == 0 and not isinstance(s, dict):
                        s = {}
                        s['version'] = '1'
                        if expect_gencost:
                            try:
                                s['baseMVA'], s['bus'], s['gen'], s['branch'], \
                                s['areas'], s['gencost'] = eval(fname)()
                            except IOError as e:
                                info = 4
                                lasterr = str(e)
                        else:
                            if return_as_obj:
                                try:
                                    s['baseMVA'], s['bus'], s['gen'], \
                                        s['branch'], s['areas'], \
                                        s['gencost'] = eval(fname)()
                                except ValueError as e:
                                    try:
                                        s['baseMVA'], s['bus'], s['gen'], \
                                            s['branch'] = eval(fname)()
                                    except ValueError as e:
                                        info = 4
                                        lasterr = str(e)
                            else:
                                try:
                                    s['baseMVA'], s['bus'], s['gen'], \
                                        s['branch'] = eval(fname)()
                                except ValueError as e:
                                    info = 4
                                    lasterr = str(e)

                except IOError as e:
                    info = 4
                    lasterr = str(e)


                if info == 4 and exists(rootname + '.py'):
                    info = 5
                    err5 = lasterr

    elif isinstance(casefile, dict):
        s = deepcopy(casefile)
    else:
        info = 1

    # check contents of dict
    if info == 0:
        # check for required keys
        if (s['baseMVA'] is None or s['bus'] is None \
            or s['gen'] is None or s['branch'] is None) or \
            (expect_gencost and s['gencost'] is None) or \
            (expect_areas and s['areas'] is None):
            info = 5  ## missing some expected fields
            err5 = 'missing data'
        else:
            ## remove empty areas if not needed
            if hasattr(s, 'areas') and (len(s['areas']) == 0) and (not expect_areas):
                del s['areas']

            ## all fields present, copy to ppc
            ppc = deepcopy(s)
            if not hasattr(ppc, 'version'):  ## hmm, struct with no 'version' field
                if ppc['gen'].shape[1] < 21:    ## version 2 has 21 or 25 cols
                    ppc['version'] = '1'
                else:
                    ppc['version'] = '2'

            if (ppc['version'] == '1'):
                # convert from version 1 to version 2
                ppc['gen'], ppc['branch'] = ppc_1to2(ppc['gen'], ppc['branch']);
                ppc['version'] = '2'

    if info == 0:  # no errors
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
    else:  # error encountered
        if info == 1:
            sys.stderr.write('Input arg should be a case or a string '
                             'containing a filename\n')
        elif info == 2:
            sys.stderr.write('Specified case not a valid file\n')
        elif info == 3:
            sys.stderr.write('Specified MAT file does not exist\n')
        elif info == 4:
            sys.stderr.write('Specified Python file does not exist\n')
        elif info == 5:
            sys.stderr.write('Syntax error or undefined data '
                             'matrix(ices) in the file\n')
        else:
            sys.stderr.write('Unknown error encountered loading case.\n')

        sys.stderr.write(lasterr + '\n')

        return info


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
