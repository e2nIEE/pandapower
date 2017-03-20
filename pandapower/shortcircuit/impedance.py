# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 10:03:45 2017

@author: thurner
"""
import warnings
from pandapower.pypower_extensions.makeYbus import makeYbus
from pandapower.shortcircuit.idx_bus import R_EQUIV, X_EQUIV
from pypower.idx_bus import BASE_KV

import numpy as np
from scipy.sparse.linalg import inv
from scipy.sparse import diags

def _calc_equiv_sc_impedance(net, ppc):
    r_fault = net["_options"]["r_fault_ohm"]
    x_fault = net["_options"]["x_fault_ohm"]
    zbus = _calc_zbus(ppc)
    if r_fault > 0 or x_fault > 0:
        base_r = np.square(ppc["bus"][:, BASE_KV]) / ppc["baseMVA"]
        fault = diags((r_fault + x_fault * 1j) / base_r)
        zbus += fault
    zbus = zbus.toarray()
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
        return inv(Ybus)