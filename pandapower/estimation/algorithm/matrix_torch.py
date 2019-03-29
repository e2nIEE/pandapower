# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import warnings
import numpy as np
from scipy.sparse import csr_matrix, vstack, hstack, diags

from pandapower.pypower.idx_brch import F_BUS, T_BUS
from pandapower.pypower.idx_bus import BUS_TYPE
from pandapower.pypower.makeYbus import makeYbus

class BaseAlgebraTorch:
    def __init__(self, ppci, non_nan_meas_mask, z):
        np.seterr(divide='ignore', invalid='ignore')
        self.ppci = ppci
        self.z = z
        self.fb = self.ppci["branch"][:, F_BUS].real.astype(int)
        self.tb = self.ppci["branch"][:, T_BUS].real.astype(int)
        self.n_bus = self.ppci['bus'].shape[0]
        self.n_branch = self.ppci['branch'].shape[0]
        self.Y_bus = None
        self.Yf = None
        self.Yt = None
        self.G = None
        self.B = None
        # TODO CHECK!
        # Double the size of the bus to get mask for v_a and v_m
        self.non_nan_meas_mask = non_nan_meas_mask
        self.non_slack_bus_mask = (self.ppci['bus'][:, BUS_TYPE] != 3).ravel()
        self.delta_v_bus_mask = np.r_[self.non_slack_bus_mask,
                                      np.ones(self.non_slack_bus_mask.shape[0], dtype=bool)].ravel()
        self.createY()

    # Function which builds a node admittance matrix out of the topology data
    # In addition, it provides the series admittances of lines as G_series and B_series
    def createY(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Ybus, Yf, Yt = makeYbus(self.ppci["baseMVA"], self.ppci["bus"], self.ppci["branch"])
            self.ppci['internal']['Yf'], self.ppci['internal']['Yt'],\
                self.ppci['internal']['Ybus'] = Yf, Yt, Ybus

        # create relevant matrices
        self.Ybus, self.Yf, self.Yt = Ybus, Yf, Yt
        self.G = self.Ybus.real
        self.B = self.Ybus.imag
        self.Gshunt = None
        self.Bshunt = None
        
    def create_rx(self, v, delta):
        hx = self.create_hx(v, delta)
        return (self.z - hx).ravel()
    
    def create_hx(self, v, delta):
        f_bus, t_bus = self.fb, self.tb
        V = v * np.exp(1j * delta)
        Sfe = V[f_bus] * np.conj(self.Yf * V)
        Ste = V[t_bus] * np.conj(self.Yt * V)
        Sbuse = V * np.conj(self.Ybus * V)
        Ife = np.abs(Sfe)/v[f_bus]
        Ite = np.abs(Ste)/v[t_bus]
        hx = np.r_[np.real(Sbuse),
                   np.real(Sfe),
                   np.real(Ste),
                   np.imag(Sbuse),
                   np.imag(Sfe),
                   np.imag(Ste),
                   v,
                   Ife,
                   Ite]
        return hx[self.non_nan_meas_mask]
    
if __name__ == "__main__":
    from nevergrad.optimization import optimizerlib
    
    
    def square(x):
        return sum((x-.5)**2)
    
    optimizer = optimizerlib.OnePlusOne(dimension=1, budget=300)
    
    recommendation = optimizer.optimize(square)
    
    
    
        
    