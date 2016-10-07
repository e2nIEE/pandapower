# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from builtins import zip
__author__ = 'scheidler, jdollichon, lthurner, tdess'

from collections import defaultdict

import numpy as np
import pandas as pd
from itertools import chain

from pypower.idx_bus import BUS_I, BASE_KV, BUS_TYPE, PD, QD, GS, BS

from pandapower.auxiliary import get_indices
from .auxiliary import _sum_by_group


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

# @profile
def _build_bus_mpc(net, mpc, calculate_voltage_angles, gen_is, eg_is, init_results=False):
    """
    """
    if len(net["trafo3w"]) > 0:
        _create_trafo3w_busses(net)
    if len(net["xward"]) > 0:
        _create_xward_busses(net)  

    bus_list = set(net["bus"].index.values)
    n_bus = len(bus_list)

    # create a mapping from arbitrary pp-index to a consecutive index starting at zero (mpc-index)
    # To sort the array first, so that PV comes first, three steps are necessary:

    # 1. Find PV / Slack nodes and place them first (necessary for fast generation of Jacobi-Matrix)
    # get indices of PV (and ref) busses
    if len(net["xward"]) > 0:
        # add xwards if available
        pv_ref = set((np.r_[eg_is["bus"].values\
                     , gen_is["bus"].values\
                     , net["xward"][net["xward"].in_service == 1]["ad_bus"].values]).flatten())
    else:
        pv_ref = set(np.r_[eg_is["bus"].values, gen_is["bus"].values].flatten())

    # 2. Add PQ busses without switches
    in_service = net.bus[net["bus"].in_service==True].index
    slidx = (net["switch"]["closed"].values == 1) & (net["switch"]["et"].values == "b") &\
            (net["switch"]["bus"].isin(in_service).values) & (net["switch"]["element"].isin(in_service).values)

    # get busses with switches
    switch_busses = set((np.r_[net["switch"]["bus"].values[slidx], net["switch"]["element"].values[slidx]]).flatten())
    pq_busses_without_switches = (bus_list - switch_busses) - pv_ref

    # consecutive values for pv, ref, and non switch pq busses
    npArange = np.arange(len(pq_busses_without_switches) + len(pv_ref))
    # busses in PandaPower
    PandaBusses = sorted(pv_ref) + sorted(pq_busses_without_switches)
    # generate bus_lookup PandaPower -> [PV, PQ(without switches)]
    bus_lookup = dict(zip(PandaBusses, npArange))

    # 3. Add PQ busses with switches and fuse them
    v = defaultdict(set)

    # get the pp-indices of the busses for those switches
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

    # check if PV busses need to be fused
    # if yes: the sets with PV busses must be found (which is slow)
    # if no: the check can be omitted
    if any(i in fbus or i in tbus for i in pv_ref):
        for dj in disjoint_sets:
            pv_busses_in_set = pv_ref & dj
            nr_pv_bus = len(pv_busses_in_set)
            if nr_pv_bus == 0:
                i += 1
                map_to = i
                bus = dj.pop()
                PandaBusses.append(bus)
                bus_lookup[bus] = map_to
            elif nr_pv_bus == 1:
                map_to = bus_lookup[pv_busses_in_set.pop()]
            else:
                raise UserWarning("Can't fuse two PV busses")
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

    # init mpc with zeros
    mpc["bus"] = np.zeros(shape=(i + 1, 13), dtype=float)
    # fill mpc with init values
    mpc["bus"][:] = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1.1, 0.9])
    mpc["bus"][:, BUS_I] = np.arange(i + 1)

    # change the voltages of the busses to the values in net
    mpc["bus"][:, BASE_KV] = net["bus"].vn_kv.loc[PandaBusses]

    # add lookup with indices before any busses were fused
    bus_lookup["before_fuse"] = dict(zip(net["bus"].index.values, np.arange(n_bus)))

    if init_results is True and len(net["res_bus"]) > 0:
        int_index = get_indices(net["bus"].index.values, bus_lookup)
        mpc["bus"][int_index, 7] = net["res_bus"]["vm_pu"].values
        mpc["bus"][int_index, 8] = net["res_bus"].va_degree.values

    return bus_lookup

def _calc_loads_and_add_on_mpc(net, mpc, bus_lookup):
    l = net["load"]
    vl = l["in_service"].values * l["scaling"].values.T / np.float64(1000.)
    lp = l["p_kw"].values * vl
    lq = l["q_kvar"].values * vl

    s = net["sgen"]
    vl = s["in_service"].values * s["scaling"].values.T / np.float64(1000.)
    sp = s["p_kw"].values * vl
    sq = s["q_kvar"].values * vl

    w = net["ward"]
    vl = w["in_service"].values / np.float64(1000.)
    wp = w["ps_kw"].values * vl
    wq = w["qs_kvar"].values * vl

    xw = net["xward"]
    vl = xw["in_service"].values / np.float64(1000.)
    xwp = xw["ps_kw"].values * vl
    xwq = xw["qs_kvar"].values * vl

    b = get_indices(np.hstack([l["bus"].values, s["bus"].values, w["bus"].values, xw["bus"].values]\
                              ), bus_lookup)
    b, vp, vq = _sum_by_group(b, np.hstack([lp, sp, wp, xwp]), np.hstack([lq, sq, wq, xwq]))

    mpc["bus"][b, PD] = vp
    mpc["bus"][b, QD] = vq

def _calc_shunts_and_add_on_mpc(net, mpc, bus_lookup):
    s = net["shunt"]
    vl = s["in_service"].values / np.float64(1000.)
    sp = s["p_kw"].values * vl
    sq = s["q_kvar"].values * vl    
    
    w = net["ward"]
    vl = w["in_service"].values / np.float64(1000.)
    wp = w["pz_kw"].values * vl
    wq = w["qz_kvar"].values * vl

    xw = net["xward"]
    vl = xw["in_service"].values / np.float64(1000.)
    xwp = xw["pz_kw"].values * vl
    xwq = xw["qz_kvar"].values * vl

    b = get_indices(np.hstack([s["bus"].values, w["bus"].values, xw["bus"].values]), bus_lookup)
    b, vp, vq = _sum_by_group(b, np.hstack([sp, wp, xwp]), np.hstack([sq, wq, xwq]))

    mpc["bus"][b, GS] = vp
    mpc["bus"][b, BS] = -vq

def _create_xward_busses(net):
    from pandapower.create import create_busses

    bid = create_busses(net, nr_busses=len(net["xward"]),
                        vn_kv=net.bus.vn_kv.loc[net.xward.bus.values].values,
                        in_service=net.xward.in_service.values)
    net.xward["ad_bus"] = bid

def _create_trafo3w_busses(net):
    from pandapower.create import create_busses

    bid = create_busses(net, nr_busses=len(net["trafo3w"]),
                        vn_kv=net.bus.vn_kv.loc[net.trafo3w.hv_bus.values].values,
                        in_service=net.trafo3w.in_service.values)
    net.trafo3w["ad_bus"] = bid
        
if __name__ == "__main__":
    import numpy as np

#    b = np.array(list(range(100, 1000)))
#    l = np.random.choice(b, 10000)
#    d = {k:i for i, k in enumerate(b)}
#    v = [d[w] for w in l]