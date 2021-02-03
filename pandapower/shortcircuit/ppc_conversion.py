# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import warnings
from copy import deepcopy
import numpy as np

from pandapower.pd2ppc import _pd2ppc, _ppc2ppci
from pandapower.auxiliary import _add_auxiliary_elements, _clean_up, _sum_by_group
from pandapower.build_branch import _trafo_df_from_trafo3w, get_trafo_values, _transformer_correction_factor

from pandapower.pypower.idx_bus import BASE_KV, GS, BS, bus_cols
from pandapower.pypower.idx_brch import branch_cols, F_BUS, T_BUS, BR_X, BR_R
from pandapower.shortcircuit.idx_bus import  C_MAX, C_MIN, K_G, K_SG, V_G,\
    PS_TRAFO_IX, GS_P, BS_P, bus_cols_sc
from pandapower.shortcircuit.idx_brch import K_T, K_ST, branch_cols_sc


def _check_sc_data_integrity(net):
    if not net.gen.empty:
        for col in ("power_station_trafo", "pg_percent"):
            if col not in net.gen.columns:
                net.gen[col] = np.nan

    if not net.trafo.empty:
        for col in ("pt_percent",):
            if col not in net.trafo.columns:
                net.trafo[col] = np.nan

def _init_ppc(net):
    _check_sc_data_integrity(net)
    _add_auxiliary_elements(net)
    ppc, _ = _pd2ppc(net)
    _init_append_array(ppc)
    _calc_k_and_add_ppc(net, ppc)
    ppci = _ppc2ppci(ppc, net)
    return ppc, ppci

def _create_k_updated_ppci(net, ppci_orig, bus):
    ppci = deepcopy(ppci_orig)
    
    is_bus = bus[np.in1d(bus, net._is_elements["bus_is_idx"])]
    all_needed_ppci_bus = np.unique(net._pd2ppc_lookups["bus"][is_bus])

    non_ps_gen_bus = all_needed_ppci_bus[np.isnan(ppci["bus"][all_needed_ppci_bus, K_SG])]
    ps_gen_bus = all_needed_ppci_bus[~np.isnan(ppci["bus"][all_needed_ppci_bus, K_SG])]

    ps_gen_bus_mask = ~np.isnan(ppci["bus"][:, K_SG])
    ps_trafo_mask = ~np.isnan(ppci["branch"][:, K_ST])
    if np.any(ps_gen_bus_mask):
        ppci["bus"][np.ix_(ps_gen_bus_mask, [GS, BS, GS_P, BS_P])] /=\
            ppci["bus"][np.ix_(ps_gen_bus_mask, [K_SG])]
        ppci["branch"][np.ix_(ps_trafo_mask, [BR_X, BR_R])] *=\
            ppci["branch"][np.ix_(ps_trafo_mask, [K_ST])] / ppci["branch"][np.ix_(ps_trafo_mask, [K_T])]

    gen_bus_mask = np.isnan(ppci["bus"][:, K_SG]) & (~np.isnan(ppci["bus"][:, K_G]))
    if np.any(gen_bus_mask):
        ppci["bus"][np.ix_(gen_bus_mask, [GS, BS, GS_P, BS_P])] /=\
            ppci["bus"][np.ix_(gen_bus_mask, [K_G])]

    # trafo_mask = np.isnan(ppci["branch"][:, K_ST]) & (~np.isnan(ppci["branch"][:, K_T]))
    # if np.any(trafo_mask):
    #     ppci["branch"][np.ix_(trafo_mask, [BR_X, BR_R])] *=\
    #         ppci["branch"][np.ix_(trafo_mask, [K_T])]
    
    bus_ppci = {}
    if ps_gen_bus.size > 0:
        for bus in ps_gen_bus:
            ppci_gen = deepcopy(ppci)
            assert np.isfinite(ppci_gen["bus"][bus, K_SG])
            # Correct ps gen bus
            ppci_gen["bus"][bus, [GS, BS, GS_P, BS_P]] /=\
                (ppci_gen["bus"][bus, K_G] / ppci_gen["bus"][bus, K_SG])

            # Correct ps transfomer
            trafo_ix = ppci_gen["bus"][bus, PS_TRAFO_IX].astype(int)
            # ppci_gen["branch"][trafo_ix, [BR_X, BR_R]] *= \
            #     (ppci_gen["branch"][trafo_ix, K_T] / ppci_gen["branch"][trafo_ix, K_ST])
            # TODO: This need to be checked!
            ppci_gen["branch"][trafo_ix, [BR_X, BR_R]] /= \
                ( ppci_gen["branch"][trafo_ix, K_ST])

            bus_ppci.update({bus: ppci_gen})
    return non_ps_gen_bus, ppci, bus_ppci


def _init_append_array(ppc):
    # Check append or update
    ppc["bus"][:, [K_G, K_SG, V_G, PS_TRAFO_IX, GS_P, BS_P,]] = np.nan
    ppc["branch"][:, [K_T, K_ST]] = np.nan

def _calc_k_and_add_ppc(net, ppc):
    _add_kt(net, ppc)
    _add_gen_sc_z_kg_ks(net, ppc)
    return ppc


def _add_kt(net, ppc):
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    branch = ppc["branch"]
    # Trafo3w is corrected in pd2ppc only need to correct "trafo"
    if not net.trafo.empty:
        f, t = net["_pd2ppc_lookups"]["branch"]["trafo"]
        trafo_df = net["trafo"]
        cmax = ppc["bus"][bus_lookup[get_trafo_values(trafo_df, "lv_bus")], C_MAX]
        kt = _transformer_correction_factor(trafo_df.vk_percent, trafo_df.vkr_percent,
                                            trafo_df.sn_mva, cmax)
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
    # gens_without_r = np.isnan(rdss_ohm)
    # if gens_without_r.any():
    #     #  use the estimations from the IEC standard for generators without defined rdss_pu
    #     lv_gens = (vn_gen <= 1.) & gens_without_r
    #     hv_gens = (vn_gen > 1.) & gens_without_r
    #     large_hv_gens = (sn_gen >= 100) & hv_gens
    #     small_hv_gens = (sn_gen < 100) & hv_gens
    #     rdss_ohm[lv_gens] = 0.15 * xdss_pu[lv_gens] * vn_gen ** 2 / sn_gen
    #     rdss_ohm[large_hv_gens] = 0.05 * xdss_pu[large_hv_gens] * vn_gen ** 2 / sn_gen
    #     rdss_ohm[small_hv_gens] = 0.07 * xdss_pu[small_hv_gens] * vn_gen ** 2 / sn_gen

    r_gen, x_gen = rdss_ohm, xdss_pu * vn_gen ** 2 / sn_gen
    z_gen = (r_gen + x_gen * 1j)
    z_gen_pu = z_gen / gen_baseZ
    y_gen_pu = 1 / z_gen_pu

    buses, gs, bs = _sum_by_group(gen_buses_ppc, y_gen_pu.real, y_gen_pu.imag)
    ppc["bus"][buses, GS] = gs
    ppc["bus"][buses, BS] = bs

    # Calculate K_G
    cmax = ppc["bus"][gen_buses_ppc, C_MAX]
    kg = (1/(1+pg_percent/100)) * cmax / (1 + (xdss_pu * sin_phi_gen))
    ppc["bus"][gen_buses_ppc, K_G] = kg
    ppc["bus"][gen_buses_ppc, V_G] = vn_gen

    # Calculate G,B (r,x) on generator for peak current calculation
    r_gen_p = np.full_like(r_gen, fill_value=np.nan)
    lv_gens, hv_gens = (vn_gen <= 1.), (vn_gen > 1.)
    small_hv_gens = (sn_gen < 100) & hv_gens
    large_hv_gens = (sn_gen >= 100) & hv_gens
    if np.any(lv_gens):
        r_gen_p[lv_gens] = 0.15 * x_gen[lv_gens]
    if np.any(small_hv_gens):
        r_gen_p[small_hv_gens] = 0.07 * x_gen[small_hv_gens]
    if np.any(large_hv_gens):
        r_gen_p[large_hv_gens] = 0.05 * x_gen[large_hv_gens]
    z_gen_p = (r_gen_p + x_gen * 1j)
    z_gen_p_pu = z_gen_p / gen_baseZ
    y_gen_p_pu = 1 / z_gen_p_pu

    _, ppc["bus"][buses, GS_P], ppc["bus"][buses, BS_P] =\
        _sum_by_group(gen_buses_ppc, y_gen_p_pu.real, y_gen_p_pu.imag)


    # Calculate K_S on power station configuration
    if np.any(~np.isnan(net.gen.power_station_trafo.values)):
        # TODO: Check this
        ps_gen_ix = net.gen.loc[~np.isnan(net.gen.power_station_trafo.values),:].index.values
        ps_trafo_ix = net.gen.loc[ps_gen_ix, "power_station_trafo"].values.astype(int)
        ps_trafo_with_tap_mask = (~np.isnan(net.trafo.loc[ps_trafo_ix, "pt_percent"]))
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

        if np.any(~ps_trafo_with_tap_mask):
            kso = (v_q/v_g/(1 + pg_percent[ps_gen_ix] / 100)) * (v_trafo_lv/v_trafo_hv) *\
                ps_cmax / (1 + x_g * sin_phi_gen[ps_gen_ix])
            ppc["bus"][ps_gen_buses_ppc[~ps_trafo_with_tap_mask], K_SG] = kso[~ps_trafo_with_tap_mask]
            ppc["branch"][np.arange(f, t)[ps_trafo_ix][~ps_trafo_with_tap_mask], K_ST] = kso[~ps_trafo_with_tap_mask]

        ppc["bus"][ps_gen_buses_ppc, PS_TRAFO_IX] = np.arange(f, t)[ps_trafo_ix]
