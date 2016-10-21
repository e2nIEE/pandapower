# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.


"""Runs a DC power flow.
"""

from os.path import dirname, join

from pypower.ppoption import ppoption
from pandapower.runpf import runpf


def rundcpf(casedata=None, ppopt=None, fname='', solvedcase=''):
    """Runs a DC power flow.

    @see: L{runpf}

    @author: Ray Zimmerman (PSERC Cornell)
    @author: Richard Lincoln

    Changes by University of Kassel: Different runpf is imported
    """
    ## default arguments
    if casedata is None:
        casedata = join(dirname(__file__), 'case9')
    ppopt = ppoption(ppopt, PF_DC=True)

    return runpf(casedata, ppopt, fname, solvedcase)
