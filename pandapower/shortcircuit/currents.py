# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
from pandapower.pypower.idx_bus import BASE_KV, VM, VA
import pandas as pd

from pandapower.shortcircuit.idx_brch import IKSS_F, IKSS_T, IP_F, IP_T, ITH_F, ITH_T
from pandapower.shortcircuit.idx_bus import C_MIN, C_MAX, KAPPA, R_EQUIV, IKSS1, IP, ITH, X_EQUIV, IKSS2, IKCV, M
from pandapower.auxiliary import _sum_by_group


def _calc_ikss(net, ppc):
    fault = net._options["fault"]
    case = net._options["case"]
    c = ppc["bus"][:, C_MIN] if case == "min" else ppc["bus"][:, C_MAX]
    ppc["internal"]["baseI"] = ppc["bus"][:, BASE_KV] * np.sqrt(3) / ppc["baseMVA"]
    z_equiv = abs(ppc["bus"][:, R_EQUIV] + ppc["bus"][:, X_EQUIV] * 1j)
    if fault == "3ph":
        ppc["bus"][:, IKSS1] = c / z_equiv / ppc["bus"][:, BASE_KV] / np.sqrt(3) * ppc["baseMVA"]
    elif fault == "2ph":
        ppc["bus"][:, IKSS1] = c / z_equiv / ppc["bus"][:, BASE_KV] / 2 * ppc["baseMVA"]
    _current_source_current(net, ppc)


def _calc_ikss_1ph(net, ppc, ppc_0):
    fault = net._options["fault"]
    case = net._options["case"]
    c = ppc["bus"][:, C_MIN] if case == "min" else ppc["bus"][:, C_MAX]
    ppc["internal"]["baseI"] = ppc["bus"][:, BASE_KV] * np.sqrt(3) / ppc["baseMVA"]
    ppc_0["internal"]["baseI"] = ppc_0["bus"][:, BASE_KV] * np.sqrt(3) / ppc_0["baseMVA"]
    z_equiv = abs((ppc["bus"][:, R_EQUIV] + ppc["bus"][:, X_EQUIV] * 1j) * 2 +
                  (ppc_0["bus"][:, R_EQUIV] + ppc_0["bus"][:, X_EQUIV] * 1j))
    if fault == "1ph":
        ppc_0["bus"][:, IKSS1] = c / z_equiv / ppc_0["bus"][:, BASE_KV] * np.sqrt(3) * ppc_0["baseMVA"]
        _current_source_current(net, ppc)


def _current_source_current(net, ppc):
    ppc["bus"][:, IKCV] = 0
    ppc["bus"][:, IKSS2] = 0
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    if not False in net.sgen.current_source.values:
        sgen = net.sgen[net._is_elements["sgen"]]
    else:
        sgen = net.sgen[net._is_elements["sgen"] & net.sgen.current_source]
    if len(sgen) == 0:
        return
    if any(pd.isnull(sgen.sn_mva)):
        raise ValueError("sn_mva needs to be specified for all sgens in net.sgen.sn_mva")
    baseI = ppc["internal"]["baseI"]
    sgen_buses = sgen.bus.values
    sgen_buses_ppc = bus_lookup[sgen_buses]
    Zbus = ppc["internal"]["Zbus"]
    if not "k" in sgen:
        raise ValueError("Nominal to short-circuit current has to specified in net.sgen.k")        
    i_sgen_pu = sgen.sn_mva.values / net.sn_mva * sgen.k.values
    buses, ikcv_pu, _ = _sum_by_group(sgen_buses_ppc, i_sgen_pu, i_sgen_pu)
    ppc["bus"][buses, IKCV] = ikcv_pu
    ppc["bus"][:, IKSS2] = abs(1 / np.diag(Zbus) * np.dot(Zbus, ppc["bus"][:, IKCV] * -1j) / baseI)
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
    ith = (ppc["bus"][:, IKSS1] + ppc["bus"][:, IKSS2]) * np.sqrt(m + n)
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
    # calculate voltage source branch current
    V_ikss = (ppc["bus"][:, IKSS1] * baseI) * Zbus
    ikss1_all_f = np.conj(Yf.dot(V_ikss))
    ikss1_all_t = np.conj(Yt.dot(V_ikss))
    ikss1_all_f[abs(ikss1_all_f) < 1e-10] = np.nan
    ikss1_all_t[abs(ikss1_all_t) < 1e-10] = np.nan

    # add current source branch current if there is one
    current_sources = any(ppc["bus"][:, IKCV]) > 0
    if current_sources:
        current = np.tile(-ppc["bus"][:, IKCV], (n, 1))
        np.fill_diagonal(current, current.diagonal() + ppc["bus"][:, IKSS2])
        V = np.dot((current * baseI), Zbus).T
        fb = np.real(ppc["branch"][:, 0]).astype(int)
        tb = np.real(ppc["branch"][:, 1]).astype(int)
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
        
def _calc_single_bus_sc(net, ppc, bus):
    case = net._options["case"]
    bus_idx = net._pd2ppc_lookups["bus"][bus]
    Zbus = ppc["internal"]["Zbus"]
#    Yf = ppc["internal"]["Yf"]
#    Yt = ppc["internal"]["Yf"]
    baseI = ppc["internal"]["baseI"]
#    fb = np.real(ppc["branch"][:, 0]).astype(int)
#    tb = np.real(ppc["branch"][:, 1]).astype(int)
    c = ppc["bus"][:, C_MIN] if case == "min" else ppc["bus"][:, C_MAX]

    # calculate voltage source branch current
    V_ikss = (ppc["bus"][:, IKSS1] * baseI) * Zbus
    V = V_ikss[:, bus_idx]
#    ikss_all_f = np.conj(Yf.dot(V_ikss))
#    ikss_all_t = np.conj(Yt.dot(V_ikss))

    # add current source branch current if there is one
#    ppc["branch"][:, IKSS_F] = abs(ikss_all_f[:, bus_idx] / baseI[fb])
#    ppc["branch"][:, IKSS_T] = abs(ikss_all_t[:, bus_idx] / baseI[tb])

    calc_branch_results(net, ppc, V)
    
from pandapower.pypower.pfsoln import pfsoln as pfsoln_pypower
from pandapower.pf.ppci_variables import _get_pf_variables_from_ppci

def calc_branch_results(net, ppci, V):
    Ybus = ppci["internal"]["Ybus"]
    Yf = ppci["internal"]["Yf"]
    Yt = ppci["internal"]["Yt"]
    baseMVA, bus, gen, branch, ref, _, pq, _, _, V0, ref_gens = _get_pf_variables_from_ppci(ppci)
    bus, gen, branch = pfsoln_pypower(baseMVA, bus, gen, branch, Ybus, Yf, Yt, V, ref, ref_gens)
    ppci["bus"], ppci["gen"], ppci["branch"] = bus, gen, branch