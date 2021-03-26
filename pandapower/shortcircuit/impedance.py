# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import warnings

import numpy as np
from scipy.sparse.linalg import inv as inv_sparse
from scipy.linalg import inv


from pandapower.shortcircuit.idx_bus import R_EQUIV, X_EQUIV
from pandapower.pypower.idx_bus import BASE_KV
from pandapower.auxiliary import _clean_up
try:
    from pandapower.pf.makeYbus_numba import makeYbus
except ImportError:
    from pandapower.pypower.makeYbus import makeYbus


def _calc_rx(net, ppc, bus):
    # Vectorized for multiple bus
    if bus is None:
        # Slice(None) is equal to select all
        bus_idx = slice(None)
    else:
        bus_idx = net._pd2ppc_lookups["bus"][bus] #bus where the short-circuit is calculated (j)

    r_fault = net["_options"]["r_fault_ohm"]
    x_fault = net["_options"]["x_fault_ohm"]
    if r_fault > 0 or x_fault > 0:
        base_r = np.square(ppc["bus"][bus_idx, BASE_KV]) / ppc["baseMVA"]
        fault_impedance = (r_fault + x_fault * 1j) / base_r
    else:
        fault_impedance = 0 + 0j

    if net["_options"]["inverse_y"]:
        Zbus = ppc["internal"]["Zbus"]
        z_equiv = np.diag(Zbus)[bus_idx] + fault_impedance
    else:
        z_equiv = _calc_zbus_diag(net, ppc, bus) + fault_impedance
    ppc["bus"][bus_idx, R_EQUIV] = z_equiv.real
    ppc["bus"][bus_idx, X_EQUIV] = z_equiv.imag


def _calc_ybus(ppc):
    Ybus, Yf, Yt = makeYbus(ppc["baseMVA"], ppc["bus"],  ppc["branch"])
    if np.isnan(Ybus.data).any():
        raise ValueError("nan value detected in Ybus matrix - check calculation parameters for nan values")
    ppc["internal"]["Yf"] = Yf
    ppc["internal"]["Yt"] = Yt
    ppc["internal"]["Ybus"] = Ybus


def _calc_zbus(net, ppc):
    try:
        Ybus = ppc["internal"]["Ybus"]
        sparsity = Ybus.nnz / Ybus.shape[0]**2
        if sparsity < 0.002:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ppc["internal"]["Zbus"] = inv_sparse(Ybus).toarray()
        else:
            ppc["internal"]["Zbus"] = inv(Ybus.toarray())
    except Exception as e:
        _clean_up(net, res=False)
        raise (e)


def _calc_zbus_diag(net, ppc, bus=None):
    ybus_fact = ppc["internal"]["ybus_fact"]
    n_bus = ppc["bus"].shape[0]

    if bus is None:
        diagZ = np.zeros(n_bus, dtype=np.complex)
        for i in range(ppc["bus"].shape[0]):
            b = np.zeros(n_bus, dtype=np.complex)
            b[i] = 1 + 0j
            diagZ[i] = ybus_fact(b)[i]
        ppc["internal"]["diagZ"] = diagZ
        return diagZ
    else:
        if isinstance(bus, int):
            bus = np.array([bus])
        diagZ = np.zeros(np.shape(bus)[0], dtype=np.complex)
        for ix, b in enumerate(bus):
            bus_idx = net._pd2ppc_lookups["bus"][b] #bus where the short-circuit is calculated (j)
            b = np.zeros(n_bus, dtype=np.complex)
            b[bus_idx] = 1 + 0j
            diagZ[ix] = ybus_fact(b)[bus_idx]
        return diagZ

    # if bus is None:
    #     bus = net.bus.index

    # diagZ = np.zeros(np.shape(bus)[0], dtype=np.complex)
    # ix = 0

    # # Use windows size 32 to calculate Zbus
    # while ix < np.shape(bus)[0]:
    #     ix_end = min(ix+32, np.shape(bus)[0])
    #     bus_idx = net._pd2ppc_lookups["bus"][bus[ix: ix_end]]
    #     b = np.zeros((n_bus, (ix_end-ix)), dtype=np.complex)
    #     for this_ix, this_bus_ix in enumerate(bus_idx):
    #         b[this_bus_ix, this_ix] = 1 + 0j
    #     res = ybus_fact(b)
    #     for this_ix, this_bus_ix in enumerate(bus_idx):
    #         diagZ[ix] = res[this_bus_ix, this_ix]
    #     ix += 32
    # return diagZ