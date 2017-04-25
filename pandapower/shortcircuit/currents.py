# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from pandapower.idx_bus import BASE_KV
import pandas as pd

from pandapower.shortcircuit.idx_brch import IKSS_F, IKSS_T, IP_F, IP_T, ITH_F, ITH_T
from pandapower.shortcircuit.idx_bus import C_MIN, C_MAX, KAPPA, R_EQUIV, IKSS, IP, ITH, X_EQUIV, IKSSCV
from pandapower.auxiliary import _sum_by_group

def _calc_ikss(net, ppc):
    fault = net._options["fault"]
    case = net._options["case"]
    c = ppc["bus_sc"][:, C_MIN] if case == "min" else ppc["bus_sc"][:, C_MAX]
    ppc["internal"]["baseI"] = ppc["bus"][:, BASE_KV] * np.sqrt(3) / ppc["baseMVA"]
    z_equiv = abs(ppc["bus_sc"][:, R_EQUIV] + ppc["bus_sc"][:, X_EQUIV] *1j)
    if fault == "3ph":
        ppc["bus_sc"][:, IKSS] = c / z_equiv  / ppc["bus"][:, BASE_KV] / np.sqrt(3) * ppc["baseMVA"]
    elif fault == "2ph":
        ppc["bus_sc"][:, IKSS] = c / z_equiv / ppc["bus"][:, BASE_KV] / 2 * ppc["baseMVA"]
    _current_source_current(net, ppc)
    ikss = ppc["bus_sc"][:, IKSS] + ppc["bus_sc"][:, IKSSCV]
    if net._options["branch_results"]:
        ppc["branch_sc"][:, [IKSS_F, IKSS_T]] = _branch_currents_from_bus(ppc, ikss, case)

def _current_source_current(net, ppc):
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    if not "motor" in net.sgen.type.values:
        sgen = net.sgen[net._is_elements["sgen"]]
    else:
        sgen = net.sgen[(net._is_elements["sgen"]) & (net.sgen.type != "motor")]
    if len(sgen) == 0:
        ppc["bus_sc"][:, IKSSCV] = 0.
        return
    if any(pd.isnull(sgen.sn_kva)):
        raise UserWarning("sn_kva needs to be specified for all sgens in net.sgen.sn_kva")
    sgen_buses = sgen.bus.values
    sgen_buses_ppc = bus_lookup[sgen_buses]
    Zbus = ppc["internal"]["Zbus"]
    i_sgen_pu = sgen.sn_kva.values / net.sn_kva * sgen.k.values * -1j
    buses, i, _ = _sum_by_group(sgen_buses_ppc, i_sgen_pu, i_sgen_pu)
    i_sgen_bus_pu = np.zeros(ppc["bus"].shape[0], dtype=complex)
    i_sgen_bus_pu[buses] = i
    ppc["bus_sc"][:, IKSSCV] = abs(1 / np.diag(Zbus) * np.dot(Zbus, i_sgen_bus_pu) / ppc["bus"][:, BASE_KV] / np.sqrt(3) * ppc["baseMVA"])

def _calc_ip(net, ppc):
    case = net._options["case"]
    ikss = ppc["bus_sc"][:, IKSS] + ppc["bus_sc"][:, IKSSCV]
    ip = np.sqrt(2) * (ppc["bus_sc"][:, KAPPA] * ikss)
    ppc["bus_sc"][:, IP] = ip
    if net._options["branch_results"]:
        ppc["branch_sc"][:, [IP_F, IP_T]] = _branch_currents_from_bus(ppc, ip, case)

def _calc_ith(net, ppc):
    case = net._options["case"]
    tk_s = net["_options"]["tk_s"]
    kappa = ppc["bus_sc"][:, KAPPA]
    f = 50
    n = 1
    m = (np.exp(4 * f * tk_s * np.log(kappa - 1)) - 1) / (2 * f * tk_s * np.log(kappa - 1))
    m[np.where(kappa > 1.99)] = 0
    ith = (ppc["bus_sc"][:, IKSS] + ppc["bus_sc"][:, IKSSCV])  * np.sqrt(m + n)
    ppc["bus_sc"][:, ITH] = ith
    if net._options["branch_results"]:
        ppc["branch_sc"][:, [ITH_F, ITH_T]] = _branch_currents_from_bus(ppc, ith, case)

def _branch_currents_from_bus(ppc, current, case):
    Zbus = ppc["internal"]["Zbus"]
    Yf = ppc["internal"]["Yf"]
    Yt = ppc["internal"]["Yf"]
    baseI = ppc["internal"]["baseI"]
    V = (current * baseI) * Zbus
    fb = np.real(ppc["branch"][:,0]).astype(int)
    tb = np.real(ppc["branch"][:,1]).astype(int)
    i_all_f = abs(np.conj(Yf.dot(V)))
    i_all_t = abs(np.conj(Yt.dot(V)))
    if case == "max":
        current_from = np.max(i_all_f, axis=1) / baseI[fb]
        current_to = np.max(i_all_t, axis=1) / baseI[tb]
    elif case == "min":
        i_all_f[i_all_f < 1e-10] = np.inf
        i_all_t[i_all_t < 1e-10] = np.inf
        current_from = np.min(i_all_f, axis=1) / baseI[fb]
        current_to = np.max(i_all_t, axis=1) / baseI[tb]
    return np.c_[current_from, current_to]