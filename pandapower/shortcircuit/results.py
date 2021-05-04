# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pandas as pd

from pandapower.shortcircuit.idx_brch import IKSS_F, IKSS_T, IP_F, IP_T, ITH_F, ITH_T
from pandapower.shortcircuit.idx_bus import IKSS1, IP, ITH, IKSS2
from pandapower.pypower.idx_bus import VM, VA, BUS_TYPE
from pandapower.results_bus import _get_bus_idx, _set_buses_out_of_service
from pandapower.results import _get_aranged_lookup, _get_branch_results
from pandapower.shortcircuit.idx_bus import C_MIN, C_MAX

def _get_bus_ppc_idx_for_br_all_results(net, ppc, bus):
    bus_lookup = net._pd2ppc_lookups["bus"]
    if bus is None:
        bus = net.bus.index

    ppc_index = np.arange(np.shape(bus)[0])
    ppc_index[ppc["bus"][bus_lookup[ppc_index], BUS_TYPE] == 4] = -1
    return bus, ppc_index

def _extract_results(net, ppc, ppc_0, bus):
    _get_bus_results(net, ppc, ppc_0, bus)
    if net._options["branch_results"]:
        if net._options['return_all_currents']:
            _get_line_all_results(net, ppc, bus)
            _get_trafo_all_results(net, ppc, bus)
            _get_trafo3w_all_results(net, ppc, bus)
        else:
            _get_line_results(net, ppc)
            _get_trafo_results(net, ppc)
            _get_trafo3w_results(net, ppc)

def _extract_single_results(net, ppc):
    for element in ["line", "trafo"]:
        net["res_%s_sc"%element] = pd.DataFrame(np.nan, index=net[element].index,
                                                columns=net["_empty_res_%s"%element].columns,
                                                dtype='float')
    _get_single_bus_results(net, ppc)
    net["_options"]["ac"] = True
    net["_options"]["trafo_loading"] = "current"
    bus_lookup_aranged = _get_aranged_lookup(net)
    bus_pq = np.zeros(shape=(len(net["bus"].index), 2), dtype=np.float)
    _get_branch_results(net, ppc, bus_lookup_aranged, bus_pq, suffix="_sc")


def _get_single_bus_results(net, ppc):
    _set_buses_out_of_service(ppc)
    bus_idx = _get_bus_idx(net)
    case = net._options["case"]
    c = ppc["bus"][bus_idx, C_MIN] if case == "min" else ppc["bus"][bus_idx, C_MAX]
    net["res_bus"]["vm_pu"] = np.nan
    net["res_bus_sc"]["vm_pu"] = c - ppc["bus"][bus_idx, VM]
    net["res_bus_sc"]["va_degree"] = ppc["bus"][bus_idx, VA]


def _get_bus_results(net, ppc, ppc_0, bus):
    if bus is None:
        bus = slice(None)

    bus_lookup = net._pd2ppc_lookups["bus"]
    ppc_index = bus_lookup[net.bus.index]

    if net["_options"]["fault"] == "1ph":
        net.res_bus_sc["ikss_ka"] = ppc_0["bus"][ppc_index, IKSS1] + ppc["bus"][ppc_index, IKSS2]
    else:
        net.res_bus_sc["ikss_ka"] = ppc["bus"][ppc_index, IKSS1] + ppc["bus"][ppc_index, IKSS2]
    if net._options["ip"]:
        net.res_bus_sc["ip_ka"] = ppc["bus"][ppc_index, IP]
    if net._options["ith"]:
        net.res_bus_sc["ith_ka"] = ppc["bus"][ppc_index, ITH]

    net.res_bus_sc = net.res_bus_sc.loc[bus, :]


def _get_line_results(net, ppc):
    branch_lookup = net._pd2ppc_lookups["branch"]
    case = net._options["case"]
    if "line" in branch_lookup:
        f, t = branch_lookup["line"]
        minmax = np.max if case == "max" else np.min
        net.res_line_sc["ikss_ka"] = minmax(ppc["branch"][f:t, [IKSS_F, IKSS_T]].real, axis=1)
        if net._options["ip"]:
            net.res_line_sc["ip_ka"] = minmax(ppc["branch"][f:t, [IP_F, IP_T]].real, axis=1)
        if net._options["ith"]:
            net.res_line_sc["ith_ka"] = minmax(ppc["branch"][f:t, [ITH_F, ITH_T]].real, axis=1)


def _get_line_all_results(net, ppc, bus):
    case = net._options["case"]

    bus, ppc_index = _get_bus_ppc_idx_for_br_all_results(net, ppc, bus)
    branch_lookup = net._pd2ppc_lookups["branch"]

    multindex = pd.MultiIndex.from_product([net.res_line_sc.index, bus], names=['line','bus'])
    net.res_line_sc = net.res_line_sc.reindex(multindex)

    if "line" in branch_lookup:
        f, t = branch_lookup["line"]
        minmax = np.maximum if case == "max" else np.minimum

        net.res_line_sc["ikss_ka"] = minmax(ppc["internal"]["branch_ikss_f"][f:t, ppc_index].real.reshape(-1, 1),
                                            ppc["internal"]["branch_ikss_t"][f:t, ppc_index].real.reshape(-1, 1))
        if net._options["ip"]:
            net.res_line_sc["ip_ka"] = minmax(ppc["internal"]["branch_ip_f"][f:t, ppc_index].real.reshape(-1, 1),
                                              ppc["internal"]["branch_ip_t"][f:t, ppc_index].real.reshape(-1, 1))
        if net._options["ith"]:
            net.res_line_sc["ith_ka"] = minmax(ppc["internal"]["branch_ith_f"][f:t, ppc_index].real.reshape(-1, 1),
                                               ppc["internal"]["branch_ith_t"][f:t, ppc_index].real.reshape(-1, 1))

def _get_trafo_results(net, ppc):
    branch_lookup = net._pd2ppc_lookups["branch"]
    if "trafo" in branch_lookup:
        f, t = branch_lookup["trafo"]
        net.res_trafo_sc["ikss_hv_ka"] = ppc["branch"][f:t, IKSS_F].real
        net.res_trafo_sc["ikss_lv_ka"] = ppc["branch"][f:t, IKSS_T].real


def _get_trafo_all_results(net, ppc, bus):
    bus, ppc_index = _get_bus_ppc_idx_for_br_all_results(net, ppc, bus)
    branch_lookup = net._pd2ppc_lookups["branch"]

    multindex = pd.MultiIndex.from_product([net.res_trafo_sc.index, bus], names=['trafo', 'bus'])
    net.res_trafo_sc = net.res_trafo_sc.reindex(multindex)

    if "trafo" in branch_lookup:
        f, t = branch_lookup["trafo"]
        net.res_trafo_sc["ikss_hv_ka"] = ppc["internal"]["branch_ikss_f"][f:t, ppc_index].real.reshape(-1, 1)
        net.res_trafo_sc["ikss_lv_ka"] = ppc["internal"]["branch_ikss_t"][f:t, ppc_index].real.reshape(-1, 1)


def _get_trafo3w_results(net, ppc):
    branch_lookup = net._pd2ppc_lookups["branch"]
    if "trafo3w" in branch_lookup:
        f, t = net._pd2ppc_lookups["branch"]["trafo3w"]
        hv = int(f + (t - f) / 3)
        mv = int(f + 2 * (t - f) / 3)
        lv = t
        net.res_trafo3w_sc["ikss_hv_ka"] = ppc["branch"][f:hv, IKSS_F].real
        net.res_trafo3w_sc["ikss_mv_ka"] = ppc["branch"][hv:mv, IKSS_T].real
        net.res_trafo3w_sc["ikss_lv_ka"] = ppc["branch"][mv:lv, IKSS_T].real


def _get_trafo3w_all_results(net, ppc, bus):
    bus, ppc_index = _get_bus_ppc_idx_for_br_all_results(net, ppc, bus)
    branch_lookup = net._pd2ppc_lookups["branch"]

    multindex = pd.MultiIndex.from_product([net.res_trafo3w_sc.index, bus], names=['trafo3w', 'bus'])
    net.res_trafo3w_sc = net.res_trafo3w_sc.reindex(multindex)

    if "trafo3w" in branch_lookup:
        f, t = branch_lookup["trafo3w"]
        hv = int(f + (t - f) / 3)
        mv = int(f + 2 * (t - f) / 3)
        lv = t
        net.res_trafo3w_sc["ikss_hv_ka"] = ppc["internal"]["branch_ikss_f"][f:hv, ppc_index].real.reshape(-1, 1)
        net.res_trafo3w_sc["ikss_mv_ka"] = ppc["internal"]["branch_ikss_t"][hv:mv, ppc_index].real.reshape(-1, 1)
        net.res_trafo3w_sc["ikss_lv_ka"] = ppc["internal"]["branch_ikss_t"][mv:lv, ppc_index].real.reshape(-1, 1)