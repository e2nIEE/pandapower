# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import numpy as np
import pandas as pd

from pandapower.shortcircuit.idx_brch import IKSS_F, IKSS_T, IP_F, IP_T, ITH_F, ITH_T
from pandapower.shortcircuit.idx_bus import IKSS1, IP, ITH, IKSS2

def _extract_results(net, ppc):
    _initialize_result_tables(net)
    _get_bus_results(net, ppc)
    if net._options["branch_results"]:
        _get_line_results(net, ppc)
        _get_trafo_results(net, ppc)
        _get_trafo3w_results(net, ppc)

def _initialize_result_tables(net):
    net.res_bus_sc = pd.DataFrame(index=net.bus.index)
    net.res_line_sc = pd.DataFrame(index=net.line.index)
    net.res_trafo_sc = pd.DataFrame(index=net.trafo.index)
    net.res_trafo3w_sc = pd.DataFrame(index=net.trafo3w.index)

def _get_bus_results(net, ppc):
    bus_lookup = net._pd2ppc_lookups["bus"]
    ppc_index = bus_lookup[net.bus.index]
    net.res_bus_sc["ikss_ka"] = ppc["bus"][ppc_index, IKSS1] +  ppc["bus"][ppc_index, IKSS2]
    if net._options["ip"]:
        net.res_bus_sc["ip_ka"] = ppc["bus"][ppc_index, IP]
    if net._options["ith"]:
        net.res_bus_sc["ith_ka"] = ppc["bus"][ppc_index, ITH]

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

def _get_trafo_results(net, ppc):
    branch_lookup = net._pd2ppc_lookups["branch"]
    if "trafo" in branch_lookup:
        f, t = branch_lookup["trafo"]
        net.res_trafo_sc["ikss_hv_ka"] = ppc["branch"][f:t, IKSS_F].real
        net.res_trafo_sc["ikss_lv_ka"] = ppc["branch"][f:t, IKSS_T].real

def _get_trafo3w_results(net, ppc):
    branch_lookup = net._pd2ppc_lookups["branch"]
    if "trafo3w" in branch_lookup:
        f, t = net._pd2ppc_lookups["branch"]["trafo3w"]
        hv = int(f + (t - f) / 3)
        mv = int(f + 2 * (t - f) / 3)
        lv = t
        net.res_trafo3w_sc["ikss_hv_ka"] = ppc["branch"][f:hv, IKSS_F].real * 1e3
        net.res_trafo3w_sc = ppc["branch"][hv:mv, IKSS_T].real * 1e3
        net.res_trafo3w_sc = ppc["branch"][mv:lv, IKSS_T].real * 1e3