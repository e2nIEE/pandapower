# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

from scipy.sparse.linalg import inv
import numpy as np
import warnings
import pandas as pd


from pandapower.pypower_extensions.makeYbus import makeYbus

from pandapower.pd2ppc import _pd2ppc
from pypower.idx_bus import GS, BS

def calc_equiv_sc_impedance(net, case):
    ppc, ppci = _pd2ppc(net)
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
#    if len(net._is_elems["ext_grid"]) > 0:
#        add_ext_grid_admittance_to_ppc(net, ppci, bus_lookup)
#    if len(net._is_elems["gen"]) > 0:
#        add_generator_admittance_to_ppc(net, ppci, bus_lookup)
#    if len(net._is_elems["sgen"]) > 0:
#        add_sgen_admittance_to_ppc(net, ppci, bus_lookup)
    zbus = calc_zbus(ppci)
    z_equiv = np.diag(zbus.toarray())
    net.bus["z_equiv"] = np.nan + np.nan *1j
    ppc_index = bus_lookup[net._is_elems["bus"].index]
    net.bus["z_equiv"].loc[net._is_elems["bus"].index] = z_equiv[ppc_index]


def end_temperature_correction_factor(net):
    if "endtemp_degree" not in net.line:
        raise UserWarning("Specify end temperature for lines in net.endtemp_degree")
    return (1 + .004 * (net.line.endtemp_degree.values.astype(float) - 20)) #formula from standard

def add_ext_grid_admittance_to_ppc(net, ppc, bus_lookup):
    case = net._options["sc_case"]
    eg = net._is_elems["ext_grid"]
    eg_buses = eg.bus.values
    c_grid = net.bus["c_%s"%case].loc[eg_buses].values
    s_sc = eg["s_sc_%s_mva"%case].values
    rx = eg["rx_%s"%case].values

    z_grid = c_grid / s_sc
    x_grid = np.sqrt(z_grid**2 / (rx**2 + 1))
    r_grid = np.sqrt(z_grid**2 - x_grid**2)
    eg["r"] = r_grid
    eg["x"] = x_grid

    y_grid = 1 / (r_grid + x_grid*1j)
    eg_bus_idx = bus_lookup[eg_buses]
    ppc["bus"][eg_bus_idx, GS] = y_grid.real
    ppc["bus"][eg_bus_idx, BS] = y_grid.imag

def add_generator_admittance_to_ppc(net, ppc, bus_lookup):
    gen = net._is_elems["gen"]
    gen_buses = gen.bus.values
    vn_net = net.bus.vn_kv.loc[gen_buses].values
    cmax = net.bus["c_max"].loc[gen_buses].values
    phi_gen = np.arccos(gen.cos_phi)

    vn_gen = gen.vn_kv.values
    sn_gen = gen.sn_kva.values

    z_r = vn_net**2 / sn_gen * 1e3
    x_gen = gen.xdss.values / 100 * z_r
    r_gen = gen.rdss.values / 100 * z_r

    kg = generator_correction_factor(vn_net, vn_gen, cmax, phi_gen, gen.xdss)
    y_gen = 1 / ((r_gen + x_gen*1j) * kg)

    gen_bus_idx = bus_lookup[gen_buses]
    ppc["bus"][gen_bus_idx, GS] = y_gen.real
    ppc["bus"][gen_bus_idx, BS] = y_gen.imag

def add_sgen_admittance_to_ppc(net, ppc, bus_lookup):
    sgen = net._is_elems["sgen"]
    if any(pd.isnull(sgen.sn_kva)):
        raise UserWarning("sn_kva needs to be specified for all sgens in net.sgen.sn_kva")
    sgen_buses = sgen.bus.values

    z_sgen = 1 / (sgen.sn_kva.values * 1e-3) / 3 #1 us reference voltage in pu
    x_sgen = np.sqrt(z_sgen**2 / (0.1**2 + 1))
    r_sgen = np.sqrt(z_sgen**2 - x_sgen**2)
    y_sgen = 1 / (r_sgen + x_sgen*1j)
   
    gen_bus_idx = bus_lookup[sgen_buses]
    ppc["bus"][gen_bus_idx, GS] = y_sgen.real
    ppc["bus"][gen_bus_idx, BS] = y_sgen.imag

def transformer_correction_factor(vsc, vscr, sn, cmax):
    sn = sn / 1000.
    zt = vsc / 100 / sn
    rt = vscr / 100 / sn
    xt = np.sqrt(zt**2 - rt**2)
    kt = 0.95 * cmax / (1 + .6 * xt * sn)
    return kt

def generator_correction_factor(vn_net, vn_gen, cmax, phi_gen, xdss):
    kg = vn_gen / vn_net * cmax / (1 + xdss * np.sin(phi_gen))
    return kg

def calc_zbus(ppc):
    Ybus, Yf, Yt = makeYbus(ppc["baseMVA"], ppc["bus"],  ppc["branch"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return inv(Ybus)
