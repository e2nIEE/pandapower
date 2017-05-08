# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import warnings

import numpy as np
from scipy.sparse.linalg import inv as inv_sparse
from scipy.linalg import inv


from pandapower.shortcircuit.idx_bus import R_EQUIV, X_EQUIV
from pandapower.idx_bus import BASE_KV
try:
    from pandapower.pf.makeYbus import makeYbus
except ImportError:
    from pandapower.pf.makeYbus_pypower import makeYbus


def _calc_rx(net, ppc):
    Zbus = ppc["internal"]["Zbus"]
    r_fault = net["_options"]["r_fault_ohm"]
    x_fault = net["_options"]["x_fault_ohm"]
    if r_fault > 0 or x_fault > 0:
        base_r = np.square(ppc["bus"][:, BASE_KV]) / ppc["baseMVA"]
        fault_impedance = (r_fault + x_fault * 1j) / base_r
        np.fill_diagonal(Zbus, Zbus.diagonal() + fault_impedance)
    z_equiv = np.diag(Zbus)
    ppc["bus"][:, R_EQUIV] = z_equiv.real
    ppc["bus"][:, X_EQUIV] = z_equiv.imag

def _calc_ybus(ppc):
    Ybus, Yf, Yt = makeYbus(ppc["baseMVA"], ppc["bus"],  ppc["branch"])
    ppc["internal"]["Yf"] = Yf
    ppc["internal"]["Yt"] = Yt
    ppc["internal"]["Ybus"] = Ybus

def _calc_zbus(ppc):
    Ybus = ppc["internal"]["Ybus"]
    sparsity = Ybus.nnz / Ybus.shape[0]**2
    if sparsity < 0.002:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ppc["internal"]["Zbus"] = inv_sparse(Ybus).toarray()
    else:
        ppc["internal"]["Zbus"] = inv(Ybus.toarray())
