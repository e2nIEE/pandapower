# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

from scipy.sparse.linalg import inv
import numpy as np
import warnings


from pandapower.pypower_extensions.makeYbus import makeYbus

from pandapower.pd2ppc import _pd2ppc
from pandapower.auxiliary import _select_is_elements
from pandapower.build_branch import _calc_tap_from_dataframe

from pypower.idx_bus import GS, BS

def calc_equiv_sc_impedance(net, case):
    is_elems = _select_is_elements(net)
    ppc, ppci, bus_lookup = _pd2ppc(net, is_elems)
    correct_branch_impedances(net, case, ppci, bus_lookup)
    if len(net.ext_grid) > 0:
        add_ext_grid_admittance_to_ppc(net, case, ppci, bus_lookup)
    if len(net.gen) > 0:
        add_generator_admittance_to_ppc(net, ppci, bus_lookup)

    zbus = calc_zbus(ppci)
    z_equiv = np.diag(zbus.toarray())
    net.bus["z_equiv"] = np.nan + np.nan *1j
    ppc_index = bus_lookup[is_elems["bus"].index]
    net.bus["z_equiv"].loc[is_elems["bus"].index] = z_equiv[ppc_index]


def consider_line_end_temperature(net, ppc):
    if "endtemp_degree" not in net.line:
        raise UserWarning("Specify end temperature for lines in net.endtemp")
    endtemp = net.line.endtemp_degree.values.astype(float)
    ppc["branch"][:len(net.line), 2] *= (1 + .004 * (endtemp - 20)) #formula from standard


def correct_branch_impedances(net, case, ppc, bus_lookup):
    ppc["branch"][:, 4] = 0
    if case == "min":
        consider_line_end_temperature(net, ppc)

    kt = transformer_correction_factor(net)
    ppc["branch"][len(net.line):, 3] *= kt
    ppc["branch"][len(net.line):, 2] *= kt
    ratio = _calc_tap_from_dataframe(ppc, net.trafo, net.trafo.vn_hv_kv, net.trafo.vn_lv_kv,
                                     bus_lookup)
    ppc["branch"][len(net.line):, 3] /= (ratio**2)
    ppc["branch"][len(net.line):, 2] /= (ratio**2)

    net.line["r"] = ppc["branch"][:len(net.line), 2].real
    net.line["x"] = ppc["branch"][:len(net.line):, 3].real
    net.trafo["r"] = ppc["branch"][len(net.line):, 2].real
    net.trafo["x"] = ppc["branch"][len(net.line):, 3].real

def add_ext_grid_admittance_to_ppc(net, case, ppc, bus_lookup):
    eg_buses = net.ext_grid.bus.values
    c_grid = net.bus["c_%s"%case].loc[eg_buses].values
    s_sc = net.ext_grid["s_sc_%s_mva"%case].values
    rx = net.ext_grid["rx_%s"%case].values

    z_grid = c_grid / s_sc
    x_grid = np.sqrt(z_grid**2 / (rx**2 + 1))
    r_grid = np.sqrt(z_grid**2 - x_grid**2)
    net.ext_grid["r"] = r_grid
    net.ext_grid["x"] = x_grid

    y_grid = 1 / (r_grid + x_grid*1j)
    eg_bus_idx = bus_lookup[eg_buses]
    ppc["bus"][eg_bus_idx, GS] = y_grid.real
    ppc["bus"][eg_bus_idx, BS] = y_grid.imag

def add_generator_admittance_to_ppc(net, ppc, bus_lookup):
    gen_buses = net.gen.bus.values
    vn_net = net.bus.vn_kv.loc[gen_buses].values
    cmax = net.bus["c_max"].loc[gen_buses].values
    phi_gen = np.arccos(net.gen.cos_phi)

    vn_gen = net.gen.vn_kv.values
    sn_gen = net.gen.sn_kva.values

    z_r = vn_net**2 / sn_gen * 1e3
    x_gen = net.gen.xdss.values / 100 * z_r
    r_gen = net.gen.rdss.values / 100 * z_r

    kg = generator_correction_factor(vn_net, vn_gen, cmax, phi_gen, net.gen.xdss)
    y_gen = 1 / ((r_gen + x_gen*1j) * kg)

    gen_bus_idx = bus_lookup[gen_buses]
    ppc["bus"][gen_bus_idx, GS] = y_gen.real
    ppc["bus"][gen_bus_idx, BS] = y_gen.imag

def transformer_correction_factor(net):
    uk = net.trafo.vsc_percent.values
    ukr = net.trafo.vscr_percent.values
    sn = net.trafo.sn_kva.values/1000        
    zt = uk / 100 / sn
    rt = ukr / 100 / sn
    xt = np.sqrt(zt**2 - rt**2)
    kt = 0.95 * net.bus.c_max.loc[net.trafo.lv_bus.values] / (1 + .6 * xt * sn)
    return kt

def generator_correction_factor(vn_net, vn_gen, cmax, phi_gen, xdss):
    kg = vn_gen / vn_net * cmax / (1 + xdss * np.sin(phi_gen))
    return kg

def calc_zbus(ppc):
    Ybus, Yf, Yt = makeYbus(ppc["baseMVA"], ppc["bus"],  ppc["branch"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return inv(Ybus)
