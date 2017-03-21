# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

import numpy as np
import pandas as pd
from itertools import chain
from collections import defaultdict

from pypower.idx_bus import BUS_I, BASE_KV, PD, QD, GS, BS, VMAX, VMIN, BUS_TYPE, NONE, VM, VA

from pandapower.auxiliary import _sum_by_group

class DisjointSet(dict):
    def add(self, item):
        self[item] = item

    def find(self, item):
        parent = self[item]
        if self[parent] != parent:
            parent = self.find(parent)
            self[item] = parent
        return parent

    def union(self, item1, item2):
        p1 = self.find(item1)
        p2 = self.find(item2)
        self[p1] = p2


def _build_bus_ppc(net, ppc):
    """
    Generates the ppc["bus"] array and the lookup pandapower indices -> ppc indices
    """
    copy_constraints_to_ppc = net["_options"]["copy_constraints_to_ppc"]
    r_switch = net["_options"]["r_switch"]
    init = net["_options"]["init"]
    mode = net["_options"]["mode"]

    # get bus indices
    bus_index = net["bus"].index.values
    n_bus = len(bus_index)
    # get in service elements
    is_elems = net["_is_elems"]
    eg_is = is_elems['ext_grid']
    gen_is = is_elems['gen']
    bus_is = is_elems['bus']

    # create a mapping from arbitrary pp-index to a consecutive index starting at zero (ppc-index)
    consec_buses = np.arange(n_bus)
    # bus_lookup as dict:
    # bus_lookup = dict(zip(bus_index, consec_buses))

    # bus lookup as mask from pandapower -> pypower
    bus_lookup = -np.ones(max(bus_index) + 1, dtype=int)
    bus_lookup[bus_index] = consec_buses

    # if there are any closed bus-bus switches update those entries
    slidx = (net["switch"]["closed"].values == 1) & (net["switch"]["et"].values == "b") & \
            (net["switch"]["bus"].isin(bus_is.index).values) & (
                net["switch"]["element"].isin(bus_is.index).values)
    net._closed_bb_switches = slidx
    
    if r_switch == 0 and slidx.any():
        # Note: this might seem a little odd - first constructing a pp to ppc mapping without
        # fused busses and then update the entries. The alternative (to construct the final
        # mapping at once) would require to determine how to fuse busses and which busses
        # are not part of any opened bus-bus switches first. It turns out, that the latter takes
        # quite some time in the average usecase, where #busses >> #bus-bus switches.

        # Find PV / Slack nodes -> their bus must be kept when fused with a PQ node
        if len(net["xward"]) > 0:
            # add xwards if available
            pv_ref = set(np.r_[eg_is["bus"].values, gen_is["bus"].values, net["xward"][
                net["xward"].in_service == 1]["ad_bus"].values].flatten())
        else:
            pv_ref = set(np.r_[eg_is["bus"].values, gen_is["bus"].values].flatten())

        # get the pp-indices of the buses which are connected to a switch
        fbus = net["switch"]["bus"].values[slidx]
        tbus = net["switch"]["element"].values[slidx]

        # create a mapping to map each bus to itself at frist ...
        ds = DisjointSet({e: e for e in chain(fbus, tbus)})

        # ... to follow each bus along a possible chain of switches to its final bus and update the map
        for f, t in zip(fbus, tbus):
            ds.union(f, t)

        # now we can find out how to fuse each bus by looking up the final bus of the chain they are connected to
        v = defaultdict(set)
        for a in ds:
            v[ds.find(a)].add(a)
        # build sets of buses which will be fused
        disjoint_sets = [e for e in v.values() if len(e) > 1]

        # check if PV buses need to be fused
        # if yes: the sets with PV buses must be found (which is slow)
        # if no: the check can be omitted
        if any(i in fbus or i in tbus for i in pv_ref):
            # for every disjoint set
            for dj in disjoint_sets:
                # check if pv buses are in the disjoint set dj
                pv_buses_in_set = pv_ref & dj
                nr_pv_bus = len(pv_buses_in_set)
                if nr_pv_bus == 0:
                    # no pv buses. Use any bus in dj
                    map_to = bus_lookup[dj.pop()]
                elif nr_pv_bus == 1:
                    # one pv bus. Get bus from pv_buses_in_set
                    map_to = bus_lookup[pv_buses_in_set.pop()]
                else:
                    raise UserWarning("Can't fuse two PV buses")
                for bus in dj:
                    # update lookup
                    bus_lookup[bus] = map_to
        else:
            # no PV buses in set
            for dj in disjoint_sets:
                # use any bus in set
                map_to = bus_lookup[dj.pop()]
                for bus in dj:
                    # update bus lookup
                    bus_lookup[bus] = map_to

    # init ppc with empty values
    ppc["bus"] = np.zeros(shape=(n_bus, 13), dtype=float)
    ppc["bus"][:] = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1.1, 0.9])
    # apply consecutive bus numbers
    ppc["bus"][:, BUS_I] = consec_buses

    # init voltages from net
    ppc["bus"][:n_bus, BASE_KV] = net["bus"]["vn_kv"].values
    # set buses out of service (BUS_TYPE == 4)
    ppc["bus"][bus_lookup[net["bus"].index.values[~net["bus"]["in_service"].values.astype(bool)]],
        BUS_TYPE] = NONE

    if init == "results" and len(net["res_bus"]) > 0:
        # init results (= voltages) from previous power flow
        ppc["bus"][:n_bus, VM] = net["res_bus"]["vm_pu"].values
        ppc["bus"][:n_bus, VA] = net["res_bus"].va_degree.values
        
    if mode == "sc":
        ppc["bus_sc"] = np.empty(shape=(n_bus, 10), dtype=float)
        ppc["bus_sc"].fill(np.nan)
        _add_c_to_ppc(net, ppc)
        

    if copy_constraints_to_ppc:
        if "max_vm_pu" in net.bus:
            ppc["bus"][:n_bus, VMAX] = net["bus"].max_vm_pu.values
        else:
            ppc["bus"][:n_bus, VMAX] = 10
        if "min_vm_pu" in net.bus:
            ppc["bus"][:n_bus, VMIN] = net["bus"].min_vm_pu.values
        else:
            ppc["bus"][:n_bus, VMIN] = 0

    net["_pd2ppc_lookups"]["bus"] = bus_lookup


def _calc_loads_and_add_on_ppc(net, ppc):
    '''
    wrapper function to call either the PF or the OPF version
    '''
    mode = net["_options"]["mode"]

    if mode=="opf":
        _calc_loads_and_add_on_ppc_opf(net, ppc)
    else:
        _calc_loads_and_add_on_ppc_pf(net, ppc)


def _calc_loads_and_add_on_ppc_pf(net, ppc):
    # init values
    b, p, q = np.array([], dtype=int), np.array([]), np.array([])

    # get in service elements
    is_elems = net["_is_elems"]

    l = net["load"]
    # element_is = check if element is at a bus in service & element is in service
    if len(l) > 0:
        vl = is_elems["load"] * l["scaling"].values.T / np.float64(1000.)
        q = np.hstack([q, l["q_kvar"].values * vl])
        p = np.hstack([p, l["p_kw"].values * vl])
        b = np.hstack([b, l["bus"].values])

    s = net["sgen"]
    if len(s) > 0:
        vl = is_elems["sgen"] * s["scaling"].values.T / np.float64(1000.)
        q = np.hstack([q, s["q_kvar"].values * vl])
        p = np.hstack([p, s["p_kw"].values * vl])
        b = np.hstack([b, s["bus"].values])

    w = net["ward"]
    if len(w) > 0:
        vl = is_elems["ward"] / np.float64(1000.)
        q = np.hstack([q, w["qs_kvar"].values * vl])
        p = np.hstack([p, w["ps_kw"].values * vl])
        b = np.hstack([b, w["bus"].values])

    xw = net["xward"]
    if len(xw) > 0:
        vl = is_elems["xward"] / np.float64(1000.)
        q = np.hstack([q, xw["qs_kvar"].values * vl])
        p = np.hstack([p, xw["ps_kw"].values * vl])
        b = np.hstack([b, xw["bus"].values])

    # if array is not empty
    if b.size:
        bus_lookup = net["_pd2ppc_lookups"]["bus"]
        b = bus_lookup[b]
        b, vp, vq = _sum_by_group(b, p, q)

        ppc["bus"][b, PD] = vp
        ppc["bus"][b, QD] = vq


def _calc_loads_and_add_on_ppc_opf(net, ppc):
    """ we need to exclude controllable sgens from the bus table
    """
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    # get in service elements
    is_elems = net["_is_elems"]

    l = net["load"]
    if not l.empty:
        l["controllable"] = _controllable_to_bool(l["controllable"])
        vl = (is_elems["load"] & ~l["controllable"]) * l["scaling"].values.T / np.float64(1000.)
        lp = l["p_kw"].values * vl
        lq = l["q_kvar"].values * vl
    else: 
        lp = []
        lq = []

    sgen = net["sgen"]
    if not sgen.empty:
        sgen["controllable"] = _controllable_to_bool(sgen["controllable"])
        vl = (is_elems["sgen"] & ~sgen["controllable"]) * sgen["scaling"].values.T / \
             np.float64(1000.)
        sp = sgen["p_kw"].values * vl
        sq = sgen["q_kvar"].values * vl
    else:
        sp = []
        sq = []

    b = bus_lookup[np.hstack([l["bus"].values, sgen["bus"].values])]
    b, vp, vq = _sum_by_group(b, np.hstack([lp, sp]), np.hstack([lq, sq]))

    ppc["bus"][b, PD] = vp
    ppc["bus"][b, QD] = vq


def _calc_shunts_and_add_on_ppc(net, ppc):
    # init values
    b, p, q = np.array([], dtype=int), np.array([]), np.array([])
    # get in service elements
    is_elems = net["_is_elems"]

    s = net["shunt"]
    if len(s) > 0:
        vl = is_elems["shunt"] / np.float64(1000.)
        q = np.hstack([q, s["q_kvar"].values * vl])
        p = np.hstack([p, s["p_kw"].values * vl])
        b = np.hstack([b, s["bus"].values])

    w = net["ward"]
    if len(w) > 0:
        vl = is_elems["ward"] / np.float64(1000.)
        q = np.hstack([q, w["qz_kvar"].values * vl])
        p = np.hstack([p, w["pz_kw"].values * vl])
        b = np.hstack([b, w["bus"].values])

    xw = net["xward"]
    if len(xw) > 0:
        vl = is_elems["xward"] / np.float64(1000.)
        q = np.hstack([q, xw["qz_kvar"].values * vl])
        p = np.hstack([p, xw["pz_kw"].values * vl])
        b = np.hstack([b, xw["bus"].values])

    # if array is not empty
    if b.size:
        bus_lookup = net["_pd2ppc_lookups"]["bus"]
        b = bus_lookup[b]
        b, vp, vq = _sum_by_group(b, p, q)

        ppc["bus"][b, GS] = vp
        ppc["bus"][b, BS] = -vq
        
def _controllable_to_bool(ctrl):
    ctrl_bool = []
    for val in ctrl:
        ctrl_bool.append(val if not np.isnan(val) else False)
    return np.array(ctrl_bool, dtype=bool)
        
def _add_gen_impedances_ppc(net, ppc):
    _add_ext_grid_sc_impedance(net, ppc)
    _add_gen_sc_impedance(net, ppc)
    if net._options["consider_sgens"] and net._options["fault"] != "2ph":
        _add_sgen_sc_impedance(net, ppc)

def _add_ext_grid_sc_impedance(net, ppc):
    from pandapower.shortcircuit.idx_bus import C_MAX, C_MIN
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    case = net._options["case"]
    eg = net._is_elems["ext_grid"]
    if len(eg) == 0:
        return
    eg_buses = eg.bus.values
    eg_buses_ppc  = bus_lookup[eg_buses]

    c = ppc["bus_sc"][eg_buses_ppc, C_MAX] if case == "max" else ppc["bus_sc"][eg_buses_ppc, C_MIN]
    s_sc = eg["s_sc_%s_mva"%case].values
    rx = eg["rx_%s"%case].values

    z_grid = c / s_sc
    x_grid = np.sqrt(z_grid**2 / (rx**2 + 1))
    r_grid = np.sqrt(z_grid**2 - x_grid**2)
    eg["r"] = r_grid
    eg["x"] = x_grid

    y_grid = 1 / (r_grid + x_grid*1j)
    buses, gs, bs = _sum_by_group(eg_buses_ppc, y_grid.real, y_grid.imag)
    ppc["bus"][buses, GS] = gs
    ppc["bus"][buses, BS] = bs

def _add_gen_sc_impedance(net, ppc):
    from pandapower.shortcircuit.idx_bus import C_MAX
    gen = net._is_elems["gen"]
    if len(gen) == 0:
        return
    gen_buses = gen.bus.values
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    gen_buses_ppc = bus_lookup[gen_buses]
    
    vn_net = ppc["bus"][gen_buses_ppc, BASE_KV]
    cmax = ppc["bus_sc"][gen_buses_ppc, C_MAX]
    phi_gen = np.arccos(gen.cos_phi)

    vn_gen = gen.vn_kv.values
    sn_gen = gen.sn_kva.values

    z_r = vn_net**2 / sn_gen * 1e3
    x_gen = gen.xdss.values / 100 * z_r
    r_gen = gen.rdss.values / 100 * z_r

    kg = _generator_correction_factor(vn_net, vn_gen, cmax, phi_gen, gen.xdss)
    y_gen = 1 / ((r_gen + x_gen*1j) * kg)

    buses, gs, bs = _sum_by_group(gen_buses_ppc, y_gen.real, y_gen.imag)
    ppc["bus"][buses, GS] = gs
    ppc["bus"][buses, BS] = bs

def _add_sgen_sc_impedance(net, ppc):
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    sgen = net.sgen[net._is_elems["sgen"]]
    if len(sgen) == 0:
        return
    if any(pd.isnull(sgen.sn_kva)):
        raise UserWarning("sn_kva needs to be specified for all sgens in net.sgen.sn_kva")
    sgen_buses = sgen.bus.values
    sgen_buses_ppc = bus_lookup[sgen_buses]

    z_sgen = 1 / (sgen.sn_kva.values * 1e-3) / 3 #1 us reference voltage in pu
    x_sgen = np.sqrt(z_sgen**2 / (0.1**2 + 1))
    r_sgen = np.sqrt(z_sgen**2 - x_sgen**2)
    y_sgen = 1 / (r_sgen + x_sgen*1j)
   
    buses, gs, bs = _sum_by_group(sgen_buses_ppc, y_sgen.real, y_sgen.imag)
    ppc["bus"][buses, GS] = gs
    ppc["bus"][buses, BS] = bs

def _generator_correction_factor(vn_net, vn_gen, cmax, phi_gen, xdss):
    kg = vn_gen / vn_net * cmax / (1 + xdss * np.sin(phi_gen))
    return kg
    
def _add_c_to_ppc(net, ppc):
    from pandapower.shortcircuit.idx_bus import C_MAX, C_MIN
    ppc["bus_sc"][:, C_MAX] = 1.1
    ppc["bus_sc"][:, C_MIN] = 1.
    lv_buses = np.where(ppc["bus"][:, BASE_KV] < 1.)
    if len(lv_buses) > 0:
        lv_tol_percent = net["_options"]["lv_tol_percent"]
        if lv_tol_percent == 10:
            c_ns = 1.1
        elif lv_tol_percent == 6:
            c_ns = 1.05
        else:
            raise ValueError("Voltage tolerance in the low voltage grid has" \
                                        " to be either 6% or 10% according to IEC 60909")
        ppc["bus_sc"][lv_buses, C_MAX] = c_ns
        ppc["bus_sc"][lv_buses, C_MIN] = .95