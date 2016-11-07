# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.


import numpy as np
from itertools import chain
from collections import defaultdict

from pypower.idx_bus import BUS_I, BASE_KV, PD, QD, GS, BS, VMAX, VMIN

from pandapower.auxiliary import get_indices, _sum_by_group


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


def _build_bus_ppc(net, ppc, is_elems, init_results=False, set_opf_constraints=False):
    """
    """
    if len(net["trafo3w"]) > 0:
        # TODO: include directly in pd2ppc so that buses are only in ppc, not in pandapower. LT
        _create_trafo3w_buses(net, init_results)
    if len(net["xward"]) > 0:
        # TODO: include directly in pd2ppc so that buses are only in ppc, not in pandapower. LT
        _create_xward_buses(net, init_results)

    # get buses as set
    bus_list = set(net["bus"].index.values)
    # get in service elements
    eg_is = is_elems['eg']
    gen_is = is_elems['gen']
    bus_is = is_elems['bus']

    # create a mapping from arbitrary pp-index to a consecutive index starting at zero (ppc-index)
    # To sort the array first, so that PV comes first, three steps are necessary:

    # 1. Find PV / Slack nodes and place them first (necessary for fast generation of Jacobi-Matrix)
    # get indices of PV (and ref) buses
    if len(net["xward"]) > 0:
        # add xwards if available
        pv_ref = set((np.r_[eg_is["bus"].values, gen_is["bus"].values, net["xward"][
                     net["xward"].in_service == 1]["ad_bus"].values]).flatten())
    else:
        pv_ref = set(np.r_[eg_is["bus"].values, gen_is["bus"].values].flatten())

    # 2. Add PQ buses without switches
    slidx = (net["switch"]["closed"].values == 1) & (net["switch"]["et"].values == "b") &\
            (net["switch"]["bus"].isin(bus_is.index).values) & (
                net["switch"]["element"].isin(bus_is.index).values)

    # get buses with switches
    switch_buses = set((np.r_[net["switch"]["bus"].values[slidx], net[
                       "switch"]["element"].values[slidx]]).flatten())
    pq_buses_without_switches = (bus_list - switch_buses) - pv_ref

    # consecutive values for pv, ref, and non switch pq buses
    npArange = np.arange(len(pq_buses_without_switches) + len(pv_ref))
    # buses in PandaPower
    PandaBusses = sorted(pv_ref) + sorted(pq_buses_without_switches)
    # generate bus_lookup PandaPower -> [PV, PQ(without switches)]
    bus_lookup = dict(zip(PandaBusses, npArange))

    # 3. Add PQ buses with switches and fuse them
    v = defaultdict(set)

    # get the pp-indices of the buses for those switches
    fbus = net["switch"]["bus"].values[slidx]
    tbus = net["switch"]["element"].values[slidx]

    # create a mapping to map each bus to itself at frist ...
    ds = DisjointSet({e: e for e in chain(fbus, tbus)})
    for f, t in zip(fbus, tbus):
        ds.union(f, t)

    for a in ds:
        v[ds.find(a)].add(a)
    disjoint_sets = [e for e in v.values() if len(e) > 1]

    i = npArange[-1]

    # check if PV buses need to be fused
    # if yes: the sets with PV buses must be found (which is slow)
    # if no: the check can be omitted
    if any(i in fbus or i in tbus for i in pv_ref):
        for dj in disjoint_sets:
            pv_buses_in_set = pv_ref & dj
            nr_pv_bus = len(pv_buses_in_set)
            if nr_pv_bus == 0:
                i += 1
                map_to = i
                bus = dj.pop()
                PandaBusses.append(bus)
                bus_lookup[bus] = map_to
            elif nr_pv_bus == 1:
                map_to = bus_lookup[pv_buses_in_set.pop()]
            else:
                raise UserWarning("Can't fuse two PV buses")
            for bus in dj:
                bus_lookup[bus] = map_to
    else:
        for dj in disjoint_sets:
            # new bus to map to
            i += 1
            map_to = i
            # get bus ID and append to Panda bus list
            bus = dj.pop()
            PandaBusses.append(bus)
            bus_lookup[bus] = map_to
            for bus in dj:
                bus_lookup[bus] = map_to

    # init ppc with zeros
    ppc["bus"] = np.zeros(shape=(i + 1, 13), dtype=float)
    # fill ppc with init values
    ppc["bus"][:] = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1.1, 0.9])
    ppc["bus"][:, BUS_I] = np.arange(i + 1)

    # change the voltages of the buses to the values in net
    ppc["bus"][:, BASE_KV] = net["bus"].vn_kv.ix[PandaBusses]

    if init_results is True and len(net["res_bus"]) > 0:
        int_index = get_indices(net["bus"].index.values, bus_lookup)
        ppc["bus"][int_index, 7] = net["res_bus"]["vm_pu"].values
        ppc["bus"][int_index, 8] = net["res_bus"].va_degree.values

    if set_opf_constraints:
        if "max_vm_pu" in net.bus:
            ppc["bus"][:, VMAX] = net["bus"].max_vm_pu.loc[PandaBusses]
        else:
            ppc["bus"][:, VMAX] = 10
        if "min_vm_pu" in net.bus:
            ppc["bus"][:, VMIN] = net["bus"].min_vm_pu.loc[PandaBusses]
        else:
            ppc["bus"][:, VMIN] = 0

    return bus_lookup


def _calc_loads_and_add_on_ppc(net, ppc, is_elems, bus_lookup):
    # get in service elements
    bus_is = is_elems['bus']

    l = net["load"]
    # element_is = check if element is at a bus in service & element is in service
    load_is = np.in1d(l.bus.values, bus_is.index) \
        & l.in_service.values.astype(bool)
    vl = load_is * l["scaling"].values.T / np.float64(1000.)
    lp = l["p_kw"].values * vl
    lq = l["q_kvar"].values * vl

    s = net["sgen"]
    sgen_is = np.in1d(s.bus.values, bus_is.index) \
        & s.in_service.values.astype(bool)
    vl = sgen_is * s["scaling"].values.T / np.float64(1000.)
    sp = s["p_kw"].values * vl
    sq = s["q_kvar"].values * vl

    w = net["ward"]
    ward_is = np.in1d(w.bus.values, bus_is.index) \
        & w.in_service.values.astype(bool)
    vl = ward_is / np.float64(1000.)
    wp = w["ps_kw"].values * vl
    wq = w["qs_kvar"].values * vl

    xw = net["xward"]
    xward_is = np.in1d(xw.bus.values, bus_is.index) \
        & xw.in_service.values.astype(bool)
    vl = xward_is / np.float64(1000.)
    xwp = xw["ps_kw"].values * vl
    xwq = xw["qs_kvar"].values * vl

    b = get_indices(np.hstack([l["bus"].values, s["bus"].values, w["bus"].values, xw["bus"].values]
                              ), bus_lookup)
    b, vp, vq = _sum_by_group(b, np.hstack([lp, sp, wp, xwp]), np.hstack([lq, sq, wq, xwq]))

    ppc["bus"][b, PD] = vp
    ppc["bus"][b, QD] = vq


def _calc_shunts_and_add_on_ppc(net, ppc, is_elems, bus_lookup):
    # get in service elements
    bus_is = is_elems['bus']

    s = net["shunt"]
    shunt_is = np.in1d(s.bus.values, bus_is.index) \
        & s.in_service.values.astype(bool)
    vl = shunt_is / np.float64(1000.)
    sp = s["p_kw"].values * vl
    sq = s["q_kvar"].values * vl

    w = net["ward"]
    ward_is = np.in1d(w.bus.values, bus_is.index) \
        & w.in_service.values.astype(bool)
    vl = ward_is / np.float64(1000.)
    wp = w["pz_kw"].values * vl
    wq = w["qz_kvar"].values * vl

    xw = net["xward"]
    xward_is = np.in1d(xw.bus.values, bus_is.index) \
        & xw.in_service.values.astype(bool)
    vl = xward_is / np.float64(1000.)
    xwp = xw["pz_kw"].values * vl
    xwq = xw["qz_kvar"].values * vl

    b = get_indices(np.hstack([s["bus"].values, w["bus"].values, xw["bus"].values]), bus_lookup)
    b, vp, vq = _sum_by_group(b, np.hstack([sp, wp, xwp]), np.hstack([sq, wq, xwq]))

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
