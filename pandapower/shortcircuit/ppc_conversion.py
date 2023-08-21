# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from copy import deepcopy
import numpy as np
import pandas as pd

from pandapower.pd2ppc import _pd2ppc, _ppc2ppci
from pandapower.auxiliary import _add_auxiliary_elements, _sum_by_group
from pandapower.build_branch import get_trafo_values, _transformer_correction_factor
from pandapower.pypower.idx_bus import GS, BS, BASE_KV
from pandapower.pypower.idx_brch import BR_X, BR_R, T_BUS, F_BUS
from pandapower.pypower.idx_bus_sc import C_MAX, K_G, K_SG, V_G, \
    PS_TRAFO_IX, GS_P, BS_P, KAPPA, GS_GEN, BS_GEN
from pandapower.pypower.idx_brch_sc import K_T, K_ST

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def _get_is_ppci_bus(net, bus):
    is_bus = bus[np.in1d(bus, net._is_elements_final["bus_is_idx"])]
    ppci_bus = np.unique(net._pd2ppc_lookups["bus"][is_bus])
    return ppci_bus


def _init_ppc(net):
    _check_sc_data_integrity(net)
    _add_auxiliary_elements(net)
    ppc, _ = _pd2ppc(net)

    # Init the required columns to nan
    ppc["bus"][:, [K_G, K_SG, V_G, PS_TRAFO_IX, GS_P, BS_P, GS_GEN, BS_GEN]] = np.nan
    ppc["branch"][:, [K_T, K_ST]] = np.nan

    # Add parameter K into ppc
    _add_kt(net, ppc)
    _add_gen_sc_z_kg_ks(net, ppc)
    _add_sgen_sc_z(net, ppc)
    _add_ward_sc_z(net, ppc)

    ppci = _ppc2ppci(ppc, net)

    return ppc, ppci

def _check_sc_data_integrity(net):
    if not net.gen.empty:
        for col in ("power_station_trafo", "pg_percent"):
            if col not in net.gen.columns:
                net.gen[col] = np.nan

    if not net.trafo.empty:
        for col in ("power_station_unit", "pt_percent",):
            if col not in net.trafo.columns:
                net.trafo[col] = np.nan

        for col in ("oltc", ):
            if col not in net.trafo.columns:
                net.trafo[col] = False


def _add_kt(net, ppc):
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    branch = ppc["branch"]
    # "trafo/trafo3w" are already corrected in pd2ppc, write parameter kt for trafo
    if not net.trafo.empty:
        f, t = net["_pd2ppc_lookups"]["branch"]["trafo"]
        trafo_df = net["trafo"]
        cmax = ppc["bus"][bus_lookup[get_trafo_values(trafo_df, "lv_bus")], C_MAX]
        kt = _transformer_correction_factor(trafo_df, trafo_df.vk_percent, trafo_df.vkr_percent, trafo_df.sn_mva, cmax)
        branch[f:t, K_T] = kt


def _add_ward_sc_z(net, ppc):
    for element in ("ward", "xward"):
        ward = net[element][net._is_elements_final[element]]
        if len(ward) == 0:
            continue

        ward_buses = ward.bus.values
        bus_lookup = net["_pd2ppc_lookups"]["bus"]
        ward_buses_ppc = bus_lookup[ward_buses]

        y_ward_pu = (ward["pz_mw"].values + ward["qz_mvar"].values * 1j)
        # how to calculate r and x in Ohm:
        # z_ward_pu = 1/y_ward_pu
        # vn_net = net.bus.loc[ward_buses, "vn_kv"].values
        # z_base_ohm = (vn_net ** 2)# / base_sn_mva)
        # z_ward_ohm = z_ward_pu * z_base_ohm

        buses, gs, bs = _sum_by_group(ward_buses_ppc, y_ward_pu.real, y_ward_pu.imag)
        ppc["bus"][buses, GS] += gs
        ppc["bus"][buses, BS] += bs


def _add_sgen_sc_z(net, ppc):
    # implement wind power station unit
    # implement doubly fed asynchronous generator
    # todo: what is this "_is_elements_final" thing?
    # sgen_wd = net.sgen.loc[net._is_elements_final["sgen"] & (net.sgen.generator_type=="async_doubly_fed")]
    if len(net.sgen) == 0 or "generator_type" not in net.sgen.columns:
        return

    sgen_wd = net.sgen.loc[net.sgen.in_service & (net.sgen.generator_type=="async_doubly_fed")]
    if len(sgen_wd) > 0:
        sgen_buses = sgen_wd.bus.values
        sgen_buses_ppc = net["_pd2ppc_lookups"]["bus"][sgen_buses]

        vn_net = net.bus.loc[sgen_buses, "vn_kv"].values
        base_z_ohm = vn_net ** 2 # / net.sn_mva  # by logic it must by divided by sn_mva, but why is it wrong?

        z_wd_ohm = np.sqrt(2) * sgen_wd.kappa * net.bus.loc[sgen_buses, "vn_kv"].values / (np.sqrt(3) * sgen_wd.max_ik_ka)
        z_wd_complex_ohm = (sgen_wd.rx + 1j) * z_wd_ohm / (np.sqrt(1+sgen_wd.rx**2))
        z_wd_complex_pu = z_wd_complex_ohm / base_z_ohm
        y_wd_pu = 1 / z_wd_complex_pu.values

        bus_idx, sgen_gs, sgen_bs = _sum_by_group(sgen_buses_ppc, y_wd_pu.real, y_wd_pu.imag)
        ppc['bus'][bus_idx, GS] = sgen_gs
        ppc['bus'][bus_idx, BS] = sgen_bs

        # add kappa for peak current
        ppc['bus'][sgen_buses_ppc, KAPPA] = sgen_wd.kappa.values

    sgen_g = net.sgen.loc[net.sgen.in_service & (net.sgen.generator_type=="async")]
    if len(sgen_g) > 0:
        sgen_buses = sgen_g.bus.values
        sgen_buses_ppc = net["_pd2ppc_lookups"]["bus"][sgen_buses]

        vn_net = net.bus.loc[sgen_buses, "vn_kv"].values
        base_z_ohm = vn_net ** 2 # / net.sn_mva  # by logic it must by divided by sn_mva, but why is it wrong?

        i_rg_ka = sgen_g.sn_mva / (vn_net * np.sqrt(3))
        z_g_ohm = 1 / sgen_g.lrc_pu * np.square(vn_net) / sgen_g.sn_mva
        z_g_complex_ohm = (sgen_g.rx + 1j) * z_g_ohm / (np.sqrt(1 + np.square(sgen_g.rx)))
        z_g_complex_pu = z_g_complex_ohm / base_z_ohm
        y_g_pu = 1 / z_g_complex_pu.values

        bus_idx, sgen_gs, sgen_bs = _sum_by_group(sgen_buses_ppc, y_g_pu.real, y_g_pu.imag)
        ppc['bus'][bus_idx, GS] = sgen_gs
        ppc['bus'][bus_idx, BS] = sgen_bs


def _add_gen_sc_z_kg_ks(net, ppc):
    gen = net["gen"][net._is_elements_final["gen"]]
    if len(gen) == 0:
        return
    gen_buses = gen.bus.values
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    gen_buses_ppc = bus_lookup[gen_buses]

    vn_gen = gen.vn_kv.values
    sn_gen = gen.sn_mva.values
    rdss_ohm = gen.rdss_ohm.values
    xdss_pu = gen.xdss_pu.values

    # Set to zero to avoid nan
    pg_percent = np.nan_to_num(gen.pg_percent.values)
    vn_net = net.bus.loc[gen_buses, "vn_kv"].values
    # Avoid warning by slight zero crossing caused
    sin_phi_gen = np.sqrt(np.clip(1 - gen.cos_phi.values**2, 0, None))

    # todo: division by net.sn_mva is correct but why does it cause tests to fail????
    gen_base_z_ohm = vn_net ** 2# / net.sn_mva
    r_gen, x_gen = rdss_ohm, xdss_pu * vn_gen ** 2 / sn_gen
    z_gen = (r_gen + x_gen * 1j)
    z_gen_pu = z_gen / gen_base_z_ohm
    y_gen_pu = 1 / z_gen_pu

    # this includes the Z_G from equation 21 by adding g and b to the ppc bus table
    buses, gs, bs = _sum_by_group(gen_buses_ppc, y_gen_pu.real, y_gen_pu.imag)
    ppc["bus"][buses, GS] += gs
    ppc["bus"][buses, BS] += bs

    # we need to keep track of the GS abd BS values that only come from generators
    ppc["bus"][buses, GS_GEN] = gs
    ppc["bus"][buses, BS_GEN] = bs

    # Calculate K_G
    cmax = ppc["bus"][gen_buses_ppc, C_MAX]
    # if the terminal voltage of the generator is permanently different from the nominal voltage of the generator, it may be
    # reqired to have a correction for U_{rG} as below (compare to equation 18) when calculating maximum SC current:
    kg = vn_net/(vn_gen * (1+pg_percent/100)) * cmax / (1 + xdss_pu * sin_phi_gen)
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
    z_gen_p_pu = z_gen_p / gen_base_z_ohm
    y_gen_p_pu = 1 / z_gen_p_pu

    # GS_P/GS_B will only be updated here
    _, gen_gs_p, gen_bs_p = _sum_by_group(gen_buses_ppc, y_gen_p_pu.real, y_gen_p_pu.imag)
    ppc["bus"][buses, GS_P] = gen_gs_p
    ppc["bus"][buses, BS_P] = gen_bs_p

    # Calculate K_S on power station configuration
    if np.any(~gen.power_station_trafo.isnull().values):
        f, _ = net["_pd2ppc_lookups"]["branch"]["trafo"]

        # If power station units defined with index in gen, no topological search needed
        ps_gen_mask = ~gen.power_station_trafo.isnull().values
        ps_trafo_ix = gen.loc[ps_gen_mask, "power_station_trafo"].values.astype(np.int64)
        ps_trafo = net.trafo.loc[ps_trafo_ix, :]
        _ps_trafo_real_ix =\
            pd.Series(index=net.trafo.index.values,
                      data=np.arange(net.trafo.shape[0])).loc[ps_trafo_ix].values
        ps_trafo_ppc_ix = f + _ps_trafo_real_ix
        ps_trafo_oltc_mask = ps_trafo["oltc"].values.astype(bool)
        ps_gen_buses_ppc = bus_lookup[gen.loc[ps_gen_mask, "bus"]]
        ps_cmax = ppc["bus"][ps_gen_buses_ppc, C_MAX]

        v_trafo_hv, v_trafo_lv = ps_trafo["vn_hv_kv"].values, ps_trafo["vn_lv_kv"].values
        v_q = net.bus.loc[ps_trafo["hv_bus"].values, "vn_kv"].values
        sn_trafo_mva = ps_trafo["sn_mva"].values
        Z_t = ps_trafo["vk_percent"].values / 100 * np.square(v_trafo_hv) / sn_trafo_mva
        R_t = ps_trafo["vkr_percent"].values / 100 * np.square(v_trafo_hv) / sn_trafo_mva
        X_t = np.sqrt(np.square(Z_t) - np.square(R_t))
        x_t = X_t / (np.square(v_trafo_hv) / sn_trafo_mva)
        p_t = ps_trafo["pt_percent"].values / 100

        if np.any(np.isnan(p_t)):
            # TODO: Check if tap is always on HV side
            p_t[np.isnan(p_t)] =\
                 -(ps_trafo["tap_step_percent"].values *
                   (ps_trafo["tap_max"].values - ps_trafo["tap_neutral"].values))[np.isnan(p_t)] / 100
            p_t[np.isnan(p_t)] = 0.0

        v_g = vn_gen[ps_gen_mask]
        x_g = xdss_pu[ps_gen_mask]
        p_g = pg_percent[ps_gen_mask] / 100


        if np.any(ps_trafo_oltc_mask):
            # x_g here is x''_d -> ks is correct
            # todo: if the voltage U_G is permanently higher than U_rG, then U_Gmax = U_rG*(1+p_G)
            ks = (v_q**2/v_g**2) * (v_trafo_lv**2/v_trafo_hv**2) *\
                ps_cmax / (1 + np.abs(x_g - x_t) * sin_phi_gen[ps_gen_mask])
            ppc["bus"][ps_gen_buses_ppc[ps_trafo_oltc_mask], K_SG] = ks[ps_trafo_oltc_mask]
            ppc["branch"][ps_trafo_ppc_ix[ps_trafo_oltc_mask], K_ST] = ks[ps_trafo_oltc_mask]

            # kg for sc calculation inside power station units
            kg = ps_cmax / (1 + x_g * sin_phi_gen[ps_gen_mask])
            ppc["bus"][ps_gen_buses_ppc[ps_trafo_oltc_mask], K_G] = kg[ps_trafo_oltc_mask]

        if np.any(~ps_trafo_oltc_mask):
            # (1+-p_t) in the standard
            # (1-p_t) is used if the highest partial short-circuit current of the power station unit at the high-voltage side of the unit transformer is searched for
            # if the unit transformer has no off-load taps or if no such taps are permanently used -> 1-p_t = 1
            kso = (v_q / (v_g * (1 + p_g))) * (v_trafo_lv / v_trafo_hv) * (1 - p_t) * \
                  ps_cmax / (1 + x_g * sin_phi_gen[ps_gen_mask])

            ppc["bus"][ps_gen_buses_ppc[~ps_trafo_oltc_mask], K_SG] = kso[~ps_trafo_oltc_mask]
            ppc["branch"][ps_trafo_ppc_ix[~ps_trafo_oltc_mask], K_ST] = kso[~ps_trafo_oltc_mask]

            # kg for sc calculation inside power station units
            kg = 1 / (1+p_g) * ps_cmax / (1 + x_g * sin_phi_gen[ps_gen_mask])
            ppc["bus"][ps_gen_buses_ppc[~ps_trafo_oltc_mask], K_G] = kg[~ps_trafo_oltc_mask]


def _create_k_updated_ppci(net, ppci_orig, ppci_bus, zero_sequence=False):
    ppci = deepcopy(ppci_orig)
    base_z_ohm = ppci['bus'][:, BASE_KV] ** 2 / net.sn_mva

    non_ps_gen_bus = ppci_bus[np.isnan(ppci["bus"][ppci_bus, K_SG])]
    ps_gen_bus = ppci_bus[~np.isnan(ppci["bus"][ppci_bus, K_SG])]

    ps_gen_bus_mask = ~np.isnan(ppci["bus"][:, K_SG])
    ps_trafo_mask = ~np.isnan(ppci["branch"][:, K_ST])

    if np.any(ps_trafo_mask):
        ps_trafo_ppci_ix = np.argwhere(ps_trafo_mask)
        ps_trafo_ppci_lv_bus = ppci["branch"][ps_trafo_mask, T_BUS].real.astype(np.int64)
        ps_trafo_ppci_hv_bus = ppci["branch"][ps_trafo_mask, F_BUS].real.astype(np.int64)
        ppci["bus"][np.ix_(ps_trafo_ppci_lv_bus, [PS_TRAFO_IX])] = ps_trafo_ppci_ix
        # if zero_sequence:
        #     ppci["bus"][np.ix_(ps_trafo_ppci_hv_bus, [BS])] += 1/(3 * 22 / (110 ** 2))

    if np.any(ps_gen_bus_mask):
        ppci["bus"][np.ix_(ps_gen_bus_mask, [GS_P, BS_P])] /= ppci["bus"][np.ix_(ps_gen_bus_mask, [K_SG])]
        ppci["bus"][np.ix_(ps_gen_bus_mask, [GS, BS])] += (1 / ppci["bus"][np.ix_(ps_gen_bus_mask, [K_SG])] - 1) * \
                                                          ppci["bus"][np.ix_(ps_gen_bus_mask, [GS_GEN, BS_GEN])]
        # Then, the R and X are multiplied by K_S (named K_ST here)
        ppci["branch"][np.ix_(ps_trafo_mask, [BR_X, BR_R])] *= ppci["branch"][np.ix_(ps_trafo_mask, [K_ST])]

    gen_bus_mask = np.isnan(ppci["bus"][:, K_SG]) & (~np.isnan(ppci["bus"][:, K_G]))
    if np.any(gen_bus_mask):
        ppci["bus"][np.ix_(gen_bus_mask, [GS_P, BS_P])] /= ppci["bus"][np.ix_(gen_bus_mask, [K_G])]
        ppci["bus"][np.ix_(gen_bus_mask, [GS, BS])] += (1 / ppci["bus"][np.ix_(gen_bus_mask, [K_G])] - 1) * \
                                                       ppci["bus"][np.ix_(gen_bus_mask, [GS_GEN, BS_GEN])]

    bus_ppci = {}
    if ps_gen_bus.size > 0:
        for bus in ps_gen_bus:
            ppci_gen = deepcopy(ppci)
            if not np.isfinite(ppci_gen["bus"][bus, K_SG]):
                raise UserWarning("Parameter error of K SG")
            # Correct ps gen bus
            ppci_gen["bus"][bus, [GS_P, BS_P]] /= (ppci_gen["bus"][bus, K_G] / ppci_gen["bus"][bus, K_SG])
            ppci_gen["bus"][bus, [GS, BS]] += (1 / (ppci_gen["bus"][bus, K_G]) - 1 / (ppci_gen["bus"][bus, K_SG])) * \
                                              ppci_gen["bus"][bus, [GS_GEN, BS_GEN]]

            # Correct ps transfomer
            trafo_ix = ppci_gen["bus"][bus, PS_TRAFO_IX].astype(np.int64)
            # Calculating SC inside power system unit
            ppci_gen["branch"][trafo_ix, [BR_X, BR_R]] /= ppci_gen["branch"][trafo_ix, K_ST]

            bus_ppci.update({bus: ppci_gen})

    return non_ps_gen_bus, ppci, bus_ppci

# TODO Roman: correction factor for 1ph cases

