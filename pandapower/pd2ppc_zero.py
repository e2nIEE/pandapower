# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 11:45:31 2017

@author: Lu
"""

import math
import numpy as np

from pandapower.pd2ppc import _pd2ppc, _ppc2ppci
from pandapower.idx_bus import BASE_KV, BS, GS
from pandapower.auxiliary import _sum_by_group
from pandapower.idx_brch import BR_B, BR_R, BR_X

def _pd2ppc_zero(net):
    """
        Input: 
        **net** - the pandapower format network
        Output:
        **ppc_0** - the matpower format zero sequence network
        **ppci_0** - the "internal" pypower format zero sequence network
    """

    ppc_0, ppci_0 = _pd2ppc(net)
    _add_ext_grid_sc_impedance_zero(net, ppc_0)
    _calc_line_parameter_zero(net, ppc_0)
    ppci_0 = _ppc2ppci(ppc_0, ppci_0, net)
    return ppc_0, ppci_0



def _add_ext_grid_sc_impedance_zero(net, ppc):
    """
        fills the zero sequence impedance of external grid in ppc
    """
    from pandapower.shortcircuit.idx_bus import C_MAX, C_MIN
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    case = net._options["case"]
    eg = net["ext_grid"][net._is_elements["ext_grid"]]
    if len(eg) == 0:
        return
    eg_buses = eg.bus.values
    eg_buses_ppc  = bus_lookup[eg_buses]

    c = ppc["bus"][eg_buses_ppc, C_MAX] if case == "max" else ppc["bus"][eg_buses_ppc, C_MIN]
    if not "s_sc_%s_mva"%case in eg:
        raise ValueError("short circuit apparent power s_sc_%s_mva needs to be specified for external grid"%case)
    s_sc = eg["s_sc_%s_mva"%case].values
    if not "rx_%s"%case in eg:
        raise ValueError("short circuit R/X rate rx_%s needs to be specified for external grid"%case)
    rx = eg["rx_%s"%case].values
    # ext_grid impedance positive sequence
    z_grid = c / s_sc
    x_grid = z_grid / np.sqrt(rx**2 + 1)
    r_grid = rx * x_grid
    # ext_grid impedance zero sequence
    x0_grid = net.ext_grid["x0x"] * x_grid                     #x0x: ratio of the ext_grid reactance between zero seq and positiv seq
    r0_grid = net.ext_grid["r0x0"] * x0_grid                   #r0x0: ratio of the ext_grid resistance and reactance at zero sequence
    y0_grid = 1 / (r0_grid + x0_grid*1j)
    buses, gs, bs = _sum_by_group(eg_buses_ppc, y0_grid.real, y0_grid.imag)
    ppc["bus"][buses, GS] = gs
    ppc["bus"][buses, BS] = bs
    


def _calc_line_parameter_zero(net, ppc):
    """
        calculates the line parameters in zero sequence network.
        fills the values in ppc.
    """
    line = net["line"]
    lookup = net._pd2ppc_lookups["branch"]
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    length = line["length_km"].values
    parallel = line["parallel"].values
    fb = bus_lookup[line["from_bus"].values]
    baseR = np.square(ppc["bus"][fb, BASE_KV]) / net.sn_kva * 1e3
    if "line" in lookup:
        f, t = lookup["line"]
        ppc["branch"][f:t, BR_B] = (2 * net.f_hz * math.pi * line["c0_nf_per_km"].values * 1e-9 * baseR *
                     length * parallel)
        ppc["branch"][f:t, BR_R] = line["r0_ohm_per_km"].values * length / baseR / parallel
        ppc["branch"][f:t, BR_X] = line["x0_ohm_per_km"].values * length / baseR / parallel
        








