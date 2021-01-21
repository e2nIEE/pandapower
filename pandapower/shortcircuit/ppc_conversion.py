# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import warnings
from copy import deepcopy
import numpy as np

from pandapower.pd2ppc import _pd2ppc, _ppc2ppci
from pandapower.auxiliary import _add_auxiliary_elements, _clean_up, _sum_by_group
from pandapower.build_branch import _trafo_df_from_trafo3w, get_trafo_values

from pandapower.pypower.idx_bus import BASE_KV, GS, BS, bus_cols
from pandapower.pypower.idx_brch import branch_cols, F_BUS, T_BUS, BR_X, BR_R
from pandapower.shortcircuit.idx_bus import  C_MAX, C_MIN, K_G, K_SG, V_G,\
    PS_TRAFO_IX, bus_cols_sc
from pandapower.shortcircuit.idx_brch import K_T, K_ST, branch_cols_sc


def _is_bus_station_gen_bus(net, ppc, bus):
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    return ~np.isnan(ppc["bus"][bus_lookup[bus], K_SG])

def _init_ppc(net):
    _add_auxiliary_elements(net)
    ppc, _ = _pd2ppc(net)
    _calc_k_and_add_ppc(net, ppc)
    ppci = _ppc2ppci(ppc, net)
    return ppc, ppci

def _create_k_updated_ppci(net, ppci_orig, bus):
    ppci = deepcopy(ppci_orig)
    
    n_ppci_bus = ppci["bus"].shape[0]
    non_ps_gen_bus = np.arange(n_ppci_bus)[np.isnan(ppci["bus"][:, K_SG])]
    ps_gen_bus = np.arange(n_ppci_bus)[~np.isnan(ppci["bus"][:, K_SG])]

    ps_gen_bus_mask = ~np.isnan(ppci["bus"][:, K_SG])
    ps_trafo_mask = ~np.isnan(ppci["branch"][:, K_ST])
    if np.any(ps_gen_bus_mask):
        ppci["bus"][np.ix_(ps_gen_bus_mask, [GS, BS])] /=\
            ppci["bus"][np.ix_(ps_gen_bus_mask, [K_SG])]
        ppci["branch"][np.ix_(ps_trafo_mask, [BR_X, BR_R])] *=\
            ppci["branch"][np.ix_(ps_trafo_mask, [K_ST])]

    gen_bus_mask = np.isnan(ppci["bus"][:, K_SG]) & (~np.isnan(ppci["bus"][:, K_G]))
    if np.any(gen_bus_mask):
        ppci["bus"][np.ix_(gen_bus_mask, [GS, BS])] /=\
            ppci["bus"][np.ix_(gen_bus_mask, [K_G])]

    trafo_mask = np.isnan(ppci["branch"][:, K_ST]) & (~np.isnan(ppci["branch"][:, K_T]))
    if np.any(trafo_mask):
        ppci["branch"][np.ix_(ps_trafo_mask, [BR_X, BR_R])] *=\
            ppci["branch"][np.ix_(ps_trafo_mask, [K_T])]
    
    bus_ppci = {}
    if ps_gen_bus.size > 0:
        for bus in ps_gen_bus:
            ppci_gen = deepcopy(ppci)
            assert np.isfinite(ppci_gen["bus"][bus, K_SG])
            # Correct ps gen bus
            ppci_gen["bus"][bus, [GS, BS]] /=\
                (ppci_gen["bus"][bus, K_G] / ppci_gen["bus"][bus, K_SG])

            # Correct ps transfomer
            trafo_ix = ppci_gen["bus"][bus, PS_TRAFO_IX].astype(int)
            ppci_gen["branch"][trafo_ix, [BR_X, BR_R]] *= \
                (ppci_gen["branch"][trafo_ix, K_T] / ppci_gen["branch"][trafo_ix, K_ST])

            bus_ppci.update({bus: ppci_gen})
    return non_ps_gen_bus, ppci, bus_ppci


def _init_append_array(ppc):
    bus_append = np.full((ppc["bus"].shape[0], bus_cols_sc),
                            np.nan, dtype=ppc["branch"].dtype)
    branch_append = np.full((ppc["branch"].shape[0], branch_cols_sc),
                            np.nan, dtype=ppc["branch"].dtype)

    # Check append or update
    if ppc["bus"].shape[1] == bus_cols:
        ppc["bus"] = np.hstack((ppc["bus"], bus_append))
    else:
        ppc["bus"][:, bus_cols: bus_cols + bus_cols_sc] = bus_append

    if ppc["branch"].shape[1] == branch_cols:
        ppc["branch"] = np.hstack((ppc["branch"], branch_append))
    else:
        ppc["branch"][:, branch_cols: branch_cols + branch_cols_sc] = branch_append


def _calc_k_and_add_ppc(net, ppc):
    _add_kt(net, ppc)
    _add_gen_sc_z_kg_ks(net, ppc)
    return ppc


def _add_kt(net, ppc):
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    branch = ppc["branch"]

    for trafo_type in ("trafo", "trafo3w"):
        if not net[trafo_type].empty:
            f, t = net["_pd2ppc_lookups"]["branch"]["trafo"]
            trafo_df = net["trafo"] if trafo_type == "trafo" else _trafo_df_from_trafo3w(net)
            
            cmax = ppc["bus"][bus_lookup[trafo_df.lv_bus.values], C_MAX]
            zt = trafo_df.vk_percent.values / 100 / trafo_df.sn_mva.values
            rt = trafo_df.vkr_percent.values / 100 / trafo_df.sn_mva.values
            xt = np.sqrt(zt ** 2 - rt ** 2)
            kt = 0.95 * cmax / (1 + .6 * xt * trafo_df.sn_mva.values)
            branch[f:t, K_T] = kt


def _add_gen_sc_z_kg_ks(net, ppc):
    gen = net["gen"][net._is_elements["gen"]]
    if len(gen) == 0:
        return
    gen_buses = gen.bus.values
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    gen_buses_ppc = bus_lookup[gen_buses]
    vn_gen = gen.vn_kv.values
    sn_gen = gen.sn_mva.values

    rdss_ohm = gen.rdss_ohm.values
    xdss_pu = gen.xdss_pu.values
    pg_percent = gen.pg_percent.values
    vn_net = ppc["bus"][gen_buses_ppc, BASE_KV]
    sin_phi_gen = np.sin(np.arccos(gen.cos_phi.values))  

    gen_baseZ = (vn_net ** 2 / ppc["baseMVA"])
    gens_without_r = np.isnan(rdss_ohm)
    if gens_without_r.any():
        #  use the estimations from the IEC standard for generators without defined rdss_pu
        lv_gens = (vn_gen <= 1.) & gens_without_r
        hv_gens = (vn_gen > 1.) & gens_without_r
        large_hv_gens = (sn_gen >= 100) & hv_gens
        small_hv_gens = (sn_gen < 100) & hv_gens
        rdss_ohm[lv_gens] = 0.15 * xdss_pu[lv_gens] * vn_gen ** 2 / sn_gen
        rdss_ohm[large_hv_gens] = 0.05 * xdss_pu[large_hv_gens] * vn_gen ** 2 / sn_gen
        rdss_ohm[small_hv_gens] = 0.07 * xdss_pu[small_hv_gens] * vn_gen ** 2 / sn_gen


    r_gen, x_gen = rdss_ohm, xdss_pu * vn_gen ** 2 / sn_gen
    z_gen = (r_gen + x_gen * 1j)
    z_gen_pu = z_gen / gen_baseZ
    y_gen_pu = 1 / z_gen_pu

    buses, gs, bs = _sum_by_group(gen_buses_ppc, y_gen_pu.real, y_gen_pu.imag)
    ppc["bus"][buses, GS] = gs
    ppc["bus"][buses, BS] = bs

    cmax = ppc["bus"][gen_buses_ppc, C_MAX]
    kg = (1/(1+pg_percent/100)) * cmax / (1 + (xdss_pu * sin_phi_gen))
    ppc["bus"][buses, K_G] = kg
    ppc["bus"][buses, V_G] = vn_gen

    # Calculate K_S on power station configuration
    if np.any(~np.isnan(net.gen.power_station_trafo.values)):
        ps_gen_ix = net.gen.loc[~np.isnan(net.gen.power_station_trafo.values),:].index.values
        ps_trafo_ix = net.gen.loc[ps_gen_ix, "power_station_trafo"].values.astype(int)
        ps_trafo_with_tap_mask = (~np.isnan(net.trafo.loc[ps_trafo_ix, "tap_pos"]))
        ps_gen_buses_ppc = bus_lookup[net.gen.loc[ps_gen_ix, "bus"]]
        f, t = net["_pd2ppc_lookups"]["branch"]["trafo"]

        ps_cmax = ppc["bus"][ps_gen_buses_ppc, C_MAX]
        v_trafo_hv, v_trafo_lv =\
            net.trafo.loc[ps_trafo_ix, "vn_hv_kv"].values, net.trafo.loc[ps_trafo_ix, "vn_lv_kv"].values
        v_q = net.bus.loc[net.trafo.loc[ps_trafo_ix, "hv_bus"], "vn_kv"].values
        v_g = vn_gen[ps_gen_ix]
        x_g = xdss_pu[ps_gen_ix]
        x_t = net.trafo.loc[ps_trafo_ix, "vk_percent"].values / 100
        
        if np.any(ps_trafo_with_tap_mask):
            ks = (v_q**2/v_g**2) * (v_trafo_lv**2/v_trafo_hv**2) *\
                ps_cmax / (1 + np.abs(x_g - x_t) * sin_phi_gen[ps_gen_ix])
            ppc["bus"][ps_gen_buses_ppc[ps_trafo_with_tap_mask], K_SG] = ks[ps_trafo_with_tap_mask]
            ppc["branch"][np.arange(f, t)[ps_trafo_ix][ps_trafo_with_tap_mask], K_ST] = ks[ps_trafo_with_tap_mask]
        else:
            kso = (v_q/v_g/(1+pg_percent/100)) * (v_trafo_lv/v_trafo_hv) *\
                ps_cmax / (1 + x_g * sin_phi_gen[ps_gen_ix])
            ppc["bus"][ps_gen_buses_ppc[~ps_trafo_with_tap_mask], K_SG] = kso[~ps_trafo_with_tap_mask]
            ppc["branch"][np.arange(f, t)[ps_trafo_ix][~ps_trafo_with_tap_mask], K_ST] = kso[~ps_trafo_with_tap_mask]

        ppc["bus"][ps_gen_buses_ppc, PS_TRAFO_IX] = np.arange(f, t)[ps_trafo_ix]
