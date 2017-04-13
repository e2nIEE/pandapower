# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import warnings

import numpy as np
from scipy.sparse.linalg import inv

from pandapower.shortcircuit.idx_bus import R_EQUIV, X_EQUIV
from pypower.idx_bus import BASE_KV
try:
    from pandapower.pypower_extensions.makeYbus import makeYbus
except ImportError:
    from pypower.makeYbus import makeYbus


def _calc_equiv_sc_impedance(net, ppc):
    r_fault = net["_options"]["r_fault_ohm"]
    x_fault = net["_options"]["x_fault_ohm"]
    zbus = _calc_zbus(ppc)
    if r_fault > 0 or x_fault > 0:
        base_r = np.square(ppc["bus"][:, BASE_KV]) / ppc["baseMVA"]
        fault_impedance = (r_fault + x_fault * 1j) / base_r
        np.fill_diagonal(zbus, zbus.diagonal() + fault_impedance)
    z_equiv = np.diag(zbus)
    ppc["bus_sc"][:, R_EQUIV] = z_equiv.real
    ppc["bus_sc"][:, X_EQUIV] = z_equiv.imag
    ppc["internal"]["zbus"] = zbus

def _calc_zbus(ppc):
    Ybus, Yf, Yt = makeYbus(ppc["baseMVA"], ppc["bus"],  ppc["branch"])
    ppc["internal"]["Yf"] = Yf
    ppc["internal"]["Yt"] = Yt
    ppc["internal"]["Ybus"] = Ybus
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return inv(Ybus).toarray()