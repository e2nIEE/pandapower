# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

import scipy.io
import numpy as np


def mpc_to_ppc(mpc_file):
    mpc = scipy.io.loadmat(mpc_file, squeeze_me=True, struct_as_record=False)

    ppc = dict()
    # generate ppc
    ppc['version'] = mpc['mpc'].version
    ppc["baseMVA"] = mpc['mpc'].baseMVA
    ppc["bus"] = mpc['mpc'].bus
    ppc["gen"] = mpc['mpc'].gen
    ppc["branch"] = mpc['mpc'].branch

    # ppc must start at 0 rather than 1 (matlab)
    ppc["bus"][:, 0] -= 1
    ppc["branch"][:, 0] -= 1
    ppc["branch"][:, 1] -= 1
    ppc["gen"][:, 0] -= 1

    # adjust for the matpower converter -> taps should be 0 when there is no transformer, but are 1
    ppc["branch"][np.where(ppc["branch"][:, 7] == 0), 7] = 1

    return ppc
