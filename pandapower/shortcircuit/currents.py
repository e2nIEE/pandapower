# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from pandapower.idx_bus import BASE_KV
import pandas as pd

from pandapower.shortcircuit.idx_brch import IKSS_F, IKSS_T, IP_F, IP_T, ITH_F, ITH_T
from pandapower.shortcircuit.idx_bus import C_MIN, C_MAX, KAPPA, R_EQUIV, IKSS1, IP, ITH, X_EQUIV, IKSS2, IKCV, M
from pandapower.auxiliary import _sum_by_group

def _calc_ikss(net, ppc):
    fault = net._options["fault"]
    case = net._options["case"]
    c = ppc["bus"][:, C_MIN] if case == "min" else ppc["bus"][:, C_MAX]
    ppc["internal"]["baseI"] = ppc["bus"][:, BASE_KV] * np.sqrt(3) / ppc["baseMVA"]
    z_equiv = abs(ppc["bus"][:, R_EQUIV] + ppc["bus"][:, X_EQUIV] *1j)
    if fault == "3ph":
        ppc["bus"][:, IKSS1] = c / z_equiv  / ppc["bus"][:, BASE_KV] / np.sqrt(3) * ppc["baseMVA"]
    elif fault == "2ph":
        ppc["bus"][:, IKSS1] = c / z_equiv / ppc["bus"][:, BASE_KV] / 2 * ppc["baseMVA"]
    _current_source_current(net, ppc)

def _current_source_current(net, ppc):
    ppc["bus"][:, IKCV] = 0
    ppc["bus"][:, IKSS2] = 0
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    if not "motor" in net.sgen.type.values:
        sgen = net.sgen[net._is_elements["sgen"]]
    else:
        sgen = net.sgen[(net._is_elements["sgen"]) & (net.sgen.type != "motor")]
    if len(sgen) == 0:
        return
    if any(pd.isnull(sgen.sn_kva)):
        raise UserWarning("sn_kva needs to be specified for all sgens in net.sgen.sn_kva")
    baseI = ppc["internal"]["baseI"]
    sgen_buses = sgen.bus.values
    sgen_buses_ppc = bus_lookup[sgen_buses]
    Zbus = ppc["internal"]["Zbus"]
    i_sgen_pu = sgen.sn_kva.values / net.sn_kva * sgen.k.values
    buses, ikcv_pu, _ = _sum_by_group(sgen_buses_ppc, i_sgen_pu, i_sgen_pu)
    ppc["bus"][buses, IKCV] = ikcv_pu
    ppc["bus"][:, IKSS2] = abs(1 / np.diag(Zbus) * np.dot(Zbus, ppc["bus"][:, IKCV] *-1j) / baseI)
    ppc["bus"][buses, IKCV] /= baseI[buses]

def _calc_ip(net, ppc):
    ip = np.sqrt(2) * (ppc["bus"][:, KAPPA] * ppc["bus"][:, IKSS1] + ppc["bus"][:, IKSS2])
    ppc["bus"][:, IP] = ip

def _calc_ith(net, ppc):
    tk_s = net["_options"]["tk_s"]
    kappa = ppc["bus"][:, KAPPA]
    f = 50
    n = 1
    m = (np.exp(4 * f * tk_s * np.log(kappa - 1)) - 1) / (2 * f * tk_s * np.log(kappa - 1))
    m[np.where(kappa > 1.99)] = 0
    ppc["bus"][:, M] = m
    ith = (ppc["bus"][:, IKSS1] + ppc["bus"][:, IKSS2])  * np.sqrt(m + n)
    ppc["bus"][:, ITH] = ith

def _calc_branch_currents(net, ppc):
    case = net._options["case"]
    Zbus = ppc["internal"]["Zbus"]
    Yf = ppc["internal"]["Yf"]
    Yt = ppc["internal"]["Yf"]
    baseI = ppc["internal"]["baseI"]
    n = ppc["bus"].shape[0]
    fb = np.real(ppc["branch"][:, 0]).astype(int)
    tb = np.real(ppc["branch"][:, 1]).astype(int)
    minmax = np.nanmin if case == "min" else np.nanmax
    #calculate voltage source branch current
    V_ikss = (ppc["bus"][:, IKSS1] * baseI) * Zbus
    ikss1_all_f = np.conj(Yf.dot(V_ikss))
    ikss1_all_t = np.conj(Yt.dot(V_ikss))
    ikss1_all_f[abs(ikss1_all_f) < 1e-10] = np.nan
    ikss1_all_t[abs(ikss1_all_t) < 1e-10] = np.nan

    #add current source branch current if there is one
    current_sources = any(ppc["bus"][:, IKCV]) > 0
    if current_sources:
        current = np.tile(-ppc["bus"][:, IKCV], (n,1))
        np.fill_diagonal(current, current.diagonal() + ppc["bus"][:, IKSS2])
        V = np.dot((current * baseI), Zbus).T
        fb = np.real(ppc["branch"][:,0]).astype(int)
        tb = np.real(ppc["branch"][:,1]).astype(int)
        ikss2_all_f = np.conj(Yf.dot(V))
        ikss2_all_t = np.conj(Yt.dot(V))
        ikss_all_f = abs(ikss1_all_f + ikss2_all_f)
        ikss_all_t = abs(ikss1_all_t + ikss2_all_t)
    else:
        ikss_all_f = abs(ikss1_all_f)
        ikss_all_t = abs(ikss1_all_t)

    ppc["branch"][:, IKSS_F] = minmax(ikss_all_f, axis=1) / baseI[fb]
    ppc["branch"][:, IKSS_T] = minmax(ikss_all_t, axis=1) / baseI[tb]

    if net._options["ip"]:
        kappa = ppc["bus"][:, KAPPA]
        if current_sources:
            ip_all_f = np.sqrt(2) * (ikss1_all_f * kappa + ikss2_all_f)
            ip_all_t = np.sqrt(2) * (ikss1_all_t * kappa + ikss2_all_t)
        else:
            ip_all_f = np.sqrt(2) * ikss1_all_f * kappa
            ip_all_t = np.sqrt(2) * ikss1_all_t * kappa

        ppc["branch"][:, IP_F] = minmax(abs(ip_all_f), axis=1) / baseI[fb]
        ppc["branch"][:, IP_T] = minmax(abs(ip_all_t), axis=1) / baseI[tb]

    if net._options["ith"]:
        n = 1
        m = ppc["bus"][:, M]
        ith_all_f = ikss_all_f * np.sqrt(m + n)
        ith_all_t = ikss_all_t * np.sqrt(m + n)
        ppc["branch"][:, ITH_F] = minmax(ith_all_f, axis=1) / baseI[fb]
        ppc["branch"][:, ITH_T] = minmax(ith_all_t, axis=1) / baseI[fb]



