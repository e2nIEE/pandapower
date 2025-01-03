# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np

from pandapower.pypower.idx_bus import PD, QD
from pandapower.pf.ppci_variables import _get_pf_variables_from_ppci
from pandapower.pf.pfsoln_numba import pfsoln
from pandapower.results import _copy_results_ppci_to_ppc, _extract_results_se, init_results
from pandapower.auxiliary import get_values


def _calc_power_flow(ppci, V):
    # store results for all elements
    # calculate branch results (in ppc_i)
    baseMVA, bus, gen, branch, svc, tcsc, ssc, vsc, ref, pv, pq, *_, ref_gens = _get_pf_variables_from_ppci(ppci)
    Ybus, Yf, Yt = ppci['internal']['Ybus'], ppci['internal']['Yf'], ppci['internal']['Yt']
    ppci['bus'], ppci['gen'], ppci['branch'] = \
        pfsoln(baseMVA, bus, gen, branch, svc, tcsc, ssc, vsc, Ybus, Yf, Yt, V, ref, ref_gens)

    # calculate bus power injections
    Sbus = np.multiply(V, np.conj(Ybus * V)) * baseMVA
    ppci["bus"][:, PD] = -Sbus.real  # saved in MW, injection -> demand
    ppci["bus"][:, QD] = -Sbus.imag  # saved in Mvar, injection -> demand
    return ppci


def _extract_result_ppci_to_pp(net, ppc, ppci):
    # convert to pandapower indices
    ppc = _copy_results_ppci_to_ppc(ppci, ppc, mode="se")

    # inits empty result tables
    init_results(net, mode="se")

    # writes res_bus.vm_pu / va_degree and branch res
    _extract_results_se(net, ppc)

    # additionally, write bus power demand results (these are not written in _extract_results)
    mapping_table = net["_pd2ppc_lookups"]["bus"]
    net.res_bus_est.index = net.bus.index
    net.res_bus_est.p_mw = get_values(ppc["bus"][:, 2], net.bus.index.values,
                                      mapping_table)
    net.res_bus_est.q_mvar = get_values(ppc["bus"][:, 3], net.bus.index.values,
                                        mapping_table)
    # overwrite power values for buses that were merged because they would not have the same power inj
    # as the bus they were merged to
    merged_bus = net["_pd2ppc_lookups"]["merged_bus"]
    merged_bus_idx = np.where(merged_bus == True)[0]
    net.res_bus_est.loc[merged_bus_idx, 'p_mw'] = 0
    net.res_bus_est.loc[merged_bus_idx, "q_mvar"] = 0
    # add shunt power because the injection at the node computed via Ybus is only the extra injection on top of the shunt
    for element in ["shunt", "ward", "xward"]:
        if ~net[element].empty:
            for i in range(net[element].shape[0]):
                bus = net[element].bus.iloc[i]
                if element == "shunt":
                    Sn = complex(net[element].p_mw.iloc[i],net[element].q_mvar.iloc[i])*net[element].step.iloc[i]
                    Ysh = Sn / (net[element].vn_kv.iloc[i]**2)
                else:
                    Sn = complex(net[element].pz_mw.iloc[i],net[element].qz_mvar.iloc[i])
                    Ysh = Sn / (net.bus.loc[bus,"vn_kv"]**2)
                V = net["res_bus_est"].loc[bus,"vm_pu"]*net["bus"].loc[bus,"vn_kv"]
                Sinj = Ysh*(V**2)
                net["res_bus_est"].loc[bus,"p_mw"] += Sinj.real
                net["res_bus_est"].loc[bus,"q_mvar"] += Sinj.imag
                if element == "shunt":
                    element_res_est = "res_" + element + "_est"
                    net[element_res_est].loc[net[element].loc[:,"bus"]==bus,"p_mw"] = Sinj.real
                    net[element_res_est].loc[net[element].loc[:,"bus"]==bus,"q_mvar"] = Sinj.imag
                    net[element_res_est].loc[net[element].loc[:,"bus"]==bus,"vm_pu"] = net["res_bus_est"].loc[bus,"vm_pu"]
    return net


def eppci2pp(net, ppc, eppci):
    # calculate the branch power flow and bus power injection based on the estimated voltage vector
    eppci = _calc_power_flow(eppci, eppci.V)

    # extract the result from ppci to ppc and pandpower network
    net = _extract_result_ppci_to_pp(net, ppc, eppci)
    return net
