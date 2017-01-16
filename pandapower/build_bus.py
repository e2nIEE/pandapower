# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

import numpy as np
from itertools import chain
from collections import defaultdict

from pypower.idx_bus import BUS_I, BASE_KV, PD, QD, GS, BS, VMAX, VMIN, BUS_TYPE

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


def _build_bus_ppc(net, ppc, is_elems, init_results=False, copy_constraints_to_ppc=False):
    """
    Generates the ppc["bus"] array and the lookup pandapower indices -> ppc indices
    """

    # add additional xward and trafo3w buses
    if len(net["trafo3w"]) > 0:
        # TODO: include directly in pd2ppc so that buses are only in ppc, not in pandapower. LT
        _create_trafo3w_buses(net, init_results)
    if len(net["xward"]) > 0:
        # TODO: include directly in pd2ppc so that buses are only in ppc, not in pandapower. LT
        _create_xward_buses(net, init_results)

    # get buses as set
    bus_list = net["bus"].index.values
    n_bus = len(bus_list)
    # get in service elements
    eg_is = is_elems['ext_grid']
    gen_is = is_elems['gen']
    bus_is = is_elems['bus']

    # create a mapping from arbitrary pp-index to a consecutive index starting at zero (ppc-index)
    consec_buses = np.arange(n_bus)
    # bus_lookup as dict:
    # bus_lookup = dict(zip(bus_list, consec_buses))

    # bus lookup as mask from pandapower -> pypower
    bus_lookup = -np.ones(max(bus_list) + 1, dtype=int)
    bus_lookup[bus_list] = consec_buses

    # if there are any opened bus-bus switches update those entries
    slidx = (net["switch"]["closed"].values == 1) & (net["switch"]["et"].values == "b") & \
            (net["switch"]["bus"].isin(bus_is.index).values) & (
                net["switch"]["element"].isin(bus_is.index).values)

    if slidx.any():
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
    ppc["bus"][bus_lookup[net["bus"].index.values[~net["bus"]["in_service"].values.astype(bool)]], BUS_TYPE] = 4

    if init_results is True and len(net["res_bus"]) > 0:
        # init results (= voltages) from previous power flow
        ppc["bus"][:n_bus, 7] = net["res_bus"]["vm_pu"].values
        ppc["bus"][:n_bus, 8] = net["res_bus"].va_degree.values

    if copy_constraints_to_ppc:
        if "max_vm_pu" in net.bus:
            ppc["bus"][:n_bus, VMAX] = net["bus"].max_vm_pu.values
        else:
            ppc["bus"][:n_bus, VMAX] = 2
        if "min_vm_pu" in net.bus:
            ppc["bus"][:n_bus, VMIN] = net["bus"].min_vm_pu.values
        else:
            ppc["bus"][:n_bus, VMIN] = 0

    return bus_lookup


def _calc_loads_and_add_on_ppc(net, ppc, is_elems, bus_lookup, opf=False):
    '''
    wrapper function to call either the PF or the OPF version
    '''

    if opf:
        _calc_loads_and_add_on_ppc_opf(net, ppc, is_elems, bus_lookup)
    else:
        _calc_loads_and_add_on_ppc_pf(net, ppc, is_elems, bus_lookup)


def _calc_loads_and_add_on_ppc_pf(net, ppc, is_elems, bus_lookup):
    # init values
    b, p, q = np.array([], dtype=int), np.array([]), np.array([])

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
    if len(w) > 0:
        vl = is_elems["xward"] / np.float64(1000.)
        q = np.hstack([q, xw["qs_kvar"].values * vl])
        p = np.hstack([p, xw["ps_kw"].values * vl])
        b = np.hstack([b, xw["bus"].values])

    # if array is not empty
    if b.size:
        b = bus_lookup[b]
        b, vp, vq = _sum_by_group(b, p, q)

        ppc["bus"][b, PD] = vp
        ppc["bus"][b, QD] = vq


def _calc_loads_and_add_on_ppc_opf(net, ppc, is_elems, bus_lookup):
    """ we need to exclude controllable sgens from the bus table
    """

    l = net["load"]
    vl = is_elems["load"] * l["scaling"].values.T / np.float64(1000.)
    lp = l["p_kw"].values * vl
    lq = l["q_kvar"].values * vl

    sgen = net["sgen"]
    if not sgen.empty:
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


def _calc_shunts_and_add_on_ppc(net, ppc, is_elems, bus_lookup):
    # init values
    b, p, q = np.array([], dtype=int), np.array([]), np.array([])
    # get in service elements

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
        b = bus_lookup[b]
        b, vp, vq = _sum_by_group(b, p, q)

        ppc["bus"][b, GS] = vp
        ppc["bus"][b, BS] = -vq


def _create_xward_buses(net, init_results):
    from pandapower.create import create_buses
    main_buses = net.bus.loc[net.xward.bus.values]
    bid = create_buses(net, nr_buses=len(main_buses),
                       vn_kv=main_buses.vn_kv.values,
                       in_service=net["xward"]["in_service"].values)
    net.xward["ad_bus"] = bid
    if init_results:
        # TODO: this is probably slow, but the whole auxiliary bus creation should be included in
        #      pd2ppc anyways. LT
        for hv_bus, aux_bus in zip(main_buses.index, bid):
            net.res_bus.loc[aux_bus] = net.res_bus.loc[hv_bus].values


def _create_trafo3w_buses(net, init_results):
    from pandapower.create import create_buses
    hv_buses = net.bus.loc[net.trafo3w.hv_bus.values]
    bid = create_buses(net, nr_buses=len(net["trafo3w"]),
                       vn_kv=hv_buses.vn_kv.values,
                       in_service=net.trafo3w.in_service.values)
    net.trafo3w["ad_bus"] = bid
    if init_results:
        # TODO: this is probably slow, but the whole auxiliary bus creation should be included in
        #      pd2ppc anyways. LT
        for hv_bus, aux_bus in zip(hv_buses.index, bid):
            net.res_bus.loc[aux_bus] = net.res_bus.loc[hv_bus].values
