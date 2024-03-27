# -*- coding: utf-8 -*-

"""
Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
and Energy System Technology (IEE), Kassel. All rights reserved.

"""

import math
import numpy as np
from itertools import product

import pandapower.auxiliary as aux
from pandapower.build_bus import _build_bus_ppc, _build_svc_ppc, _build_ssc_ppc
from pandapower.build_gen import _build_gen_ppc
# from pandapower.pd2ppc import _ppc2ppci, _init_ppc
from pandapower.pypower.idx_brch import BR_B, BR_R, BR_X, F_BUS, T_BUS, branch_cols, BR_STATUS, SHIFT, TAP, BR_R_ASYM, \
    BR_X_ASYM
from pandapower.pypower.idx_bus import BASE_KV, BS, GS, BUS_TYPE
from pandapower.pypower.idx_brch_sc import branch_cols_sc
from pandapower.pypower.idx_bus_sc import C_MAX, C_MIN
from pandapower.build_branch import _calc_tap_from_dataframe, _transformer_correction_factor, \
    _calc_nominal_ratio_from_dataframe, \
    get_trafo_values, _trafo_df_from_trafo3w, _calc_branch_values_from_trafo_df, _calc_switch_parameter, \
    _calc_impedance_parameters_from_dataframe, _build_tcsc_ppc
from pandapower.build_branch import _switch_branches, _branches_with_oos_buses, _initialize_branch_lookup, _end_temperature_correction_factor
from pandapower.pd2ppc import _ppc2ppci, _init_ppc


def _pd2ppc_zero(net, k_st, sequence=0):
    """
    Builds the ppc data structure for zero impedance system. Includes the impedance values of
    lines and transformers, but no load or generation data.

    For short-circuit calculation, the short-circuit impedance of external grids is also considered.
    """
    # select elements in service (time consuming, so we do it once)
    net["_is_elements"] = aux._select_is_elements_numba(net, sequence=sequence)

    ppc = _init_ppc(net, sequence)

    _build_bus_ppc(net, ppc)
    _build_gen_ppc(net, ppc)
    _build_svc_ppc(net, ppc, "sc")   # needed for shape reasons
    _build_tcsc_ppc(net, ppc, "sc")  # needed for shape reasons
    _build_ssc_ppc(net, ppc, "sc")  # needed for shape reasons
    _add_gen_sc_impedance_zero(net, ppc)
    _add_ext_grid_sc_impedance_zero(net, ppc)
    _build_branch_ppc_zero(net, ppc, k_st)

    # adds auxilary buses for open switches at branches
    _switch_branches(net, ppc)

    # add auxilary buses for out of service buses at in service lines.
    # Also sets lines out of service if they are connected to two out of service buses
    _branches_with_oos_buses(net, ppc)
    if hasattr(net, "_isolated_buses"):
        ppc["bus"][net._isolated_buses, BUS_TYPE] = 4.

    # generates "internal" ppci format (for powerflow calc) from "external" ppc format and updates the bus lookup
    # Note: Also reorders buses and gens in ppc
    ppci = _ppc2ppci(ppc, net)
    # net._ppc0 = ppc    <--Obsolete. now covered in _init_ppc
    return ppc, ppci


def _build_branch_ppc_zero(net, ppc, k_st=None):
    """
    Takes the empty ppc network and fills it with the zero imepdance branch values. The branch
    datatype will be np.complex 128 afterwards.

    .. note:: The order of branches in the ppc is:
            1. Lines
            2. Transformers

    **INPUT**:
        **net** -The pandapower format network

        **ppc** - The PYPOWER format network to fill in values

    """
    length = _initialize_branch_lookup(net)
    lookup = net._pd2ppc_lookups["branch"]
    mode = net._options["mode"]
    ppc["branch"] = np.zeros(shape=(length, branch_cols), dtype=np.complex128)
    if mode == "sc":
        branch_sc = np.empty(shape=(length, branch_cols_sc), dtype=float)
        branch_sc.fill(np.nan)
        ppc["branch"] = np.hstack((ppc["branch"], branch_sc))
    ppc["branch"][:, :13] = np.array([0, 0, 0, 0, 0, 250, 250, 250, 1, 0, 1, -360, 360])

    _add_line_sc_impedance_zero(net, ppc)
    _add_impedance_sc_impedance_zero(net, ppc)
    _add_trafo_sc_impedance_zero(net, ppc, k_st=k_st)
    if "switch" in lookup:
        _calc_switch_parameter(net, ppc)
    if mode == "sc":
        _add_trafo3w_sc_impedance_zero(net, ppc)
    else:
        if "trafo3w" in lookup:
            raise NotImplementedError("Three winding transformers are not implemented for unbalanced calculations")


def _add_trafo_sc_impedance_zero(net, ppc, trafo_df=None, k_st=None):
    if trafo_df is None:
        trafo_df = net["trafo"]
    if k_st is None:
        k_st = np.ones(len(ppc['branch']))
    if "xn_ohm" not in trafo_df.columns:
        trafo_df["xn_ohm"] = 0.
    branch_lookup = net["_pd2ppc_lookups"]["branch"]
    if "trafo" not in branch_lookup:
        return
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    mode = net["_options"]["mode"]
    trafo_model = net["_options"]["trafo_model"]
    f, t = branch_lookup["trafo"]
    trafo_df["_ppc_idx"] = range(f, t)
    trafo_df['k_st'] = k_st[trafo_df["_ppc_idx"].values].real
    bus_lookup = net["_pd2ppc_lookups"]["bus"]

    hv_bus = get_trafo_values(trafo_df, "hv_bus").astype(np.int64)
    lv_bus = get_trafo_values(trafo_df, "lv_bus").astype(np.int64)
    in_service = get_trafo_values(trafo_df, "in_service").astype(np.int64)
    ppc["branch"][f:t, F_BUS] = bus_lookup[hv_bus]
    ppc["branch"][f:t, T_BUS] = bus_lookup[lv_bus]
    buses_all, gs_all, bs_all = np.array([], dtype=np.int64), np.array([]), np.array([])
    BIG_NUMBER = 1e20 * ppc["baseMVA"]
    if mode == "sc":
        # Should be considered as connected for all in_service branches
        ppc["branch"][f:t, BR_X] = BIG_NUMBER
        ppc["branch"][f:t, BR_R] = BIG_NUMBER
        ppc["branch"][f:t, BR_B] = 0
        ppc["branch"][f:t, BR_STATUS] = in_service
    else:
        ppc["branch"][f:t, BR_STATUS] = 0

    if "vector_group" not in trafo_df:
        raise ValueError("Vector Group of transformer needs to be specified for zero "
                         "sequence modelling \n Try : net.trafo[\"vector_group\"] = 'Dyn'")

    for vector_group, trafos in trafo_df.groupby("vector_group"):
        # TODO Roman: check this/expand this
        ppc_idx = trafos["_ppc_idx"].values.astype(np.int64)

        if vector_group.lower() in ["yy", "yd", "dy", "dd"]:
            continue

        vk_percent = trafos["vk_percent"].values.astype(float)
        vkr_percent = trafos["vkr_percent"].values.astype(float)
        sn_trafo_mva = trafos["sn_mva"].values.astype(float)
        # Just put pos seq parameter if zero seq parameter is zero
        if "vk0_percent" not in trafos:
            raise ValueError("Short circuit voltage of transformer Vk0 needs to be specified for zero "
                             "sequence modelling \n Try : net.trafo[\"vk0_percent\"] = net.trafo[\"vk_percent\"]")
        vk0_percent = trafos["vk0_percent"].values.astype(float) if \
            trafos["vk0_percent"].values.astype(float).all() != 0. else \
            trafos["vk_percent"].values.astype(float)
        # Just put pos seq parameter if zero seq parameter is zero
        if "vkr0_percent" not in trafos:
            raise ValueError("Real part of short circuit voltage Vk0(Real) needs to be specified for transformer "
                             "modelling \n Try : net.trafo[\"vkr0_percent\"] = net.trafo[\"vkr_percent\"]")
        vkr0_percent = trafos["vkr0_percent"].values.astype(float) if \
            trafos["vkr0_percent"].values.astype(float).all() != 0. else \
            trafos["vkr_percent"].values.astype(float)
        lv_buses = trafos["lv_bus"].values.astype(np.int64)
        hv_buses = trafos["hv_bus"].values.astype(np.int64)
        lv_buses_ppc = bus_lookup[lv_buses]
        hv_buses_ppc = bus_lookup[hv_buses]
        if "mag0_percent" not in trafos:
            # For Shell Type transformers vk0 = vk * 1
            #        and mag0_percent = 10 ... 100  Zm0/ Zsc0
            # --pg 50 DigSilent Power Factory Transformer manual
            raise ValueError("Magnetizing impedance to vk0 ratio needs to be specified for transformer "
                             "modelling  \n Try : net.trafo[\"mag0_percent\"] = 100")
        mag0_ratio = trafos.mag0_percent.values.astype(float)
        if "mag0_rx" not in trafos:
            raise ValueError("Magnetizing impedance R/X ratio needs to be specified for transformer "
                             "modelling \n Try : net.trafo[\"mag0_rx\"] = 0 ")
        mag0_rx = trafos["mag0_rx"].values.astype(float)
        if "si0_hv_partial" not in trafos:
            raise ValueError("Zero sequence short circuit impedance partition towards HV side needs to be specified "
                             "for transformer modelling \n Try : net.trafo[\"si0_hv_partial\"] = 0.9 ")
        si0_hv_partial = trafos.si0_hv_partial.values.astype(float)
        parallel = trafos.parallel.values.astype(float)
        if "power_station_unit" in trafos.columns:
            power_station_unit = trafos.power_station_unit.fillna(False).astype(bool)
        else:
            power_station_unit = np.zeros(len(trafos), dtype=bool)
        in_service = trafos["in_service"].astype(np.int64)

        ppc["branch"][ppc_idx, F_BUS] = hv_buses_ppc
        ppc["branch"][ppc_idx, T_BUS] = lv_buses_ppc

        vn_trafo_hv, vn_trafo_lv, shift = _calc_tap_from_dataframe(net, trafos)
        vn_bus_lv = ppc["bus"][lv_buses_ppc, BASE_KV]
        vn_bus_hv = ppc["bus"][hv_buses_ppc, BASE_KV]
        ratio = _calc_nominal_ratio_from_dataframe(ppc, trafos, vn_trafo_hv,
                                                   vn_trafo_lv, bus_lookup)
        ppc["branch"][ppc_idx, TAP] = ratio
        ppc["branch"][ppc_idx, SHIFT] = shift

        # zero seq. transformer impedance
        tap_lv = np.square(vn_trafo_lv / vn_bus_lv) * net.sn_mva
        tap_hv = np.square(vn_trafo_hv / vn_bus_hv) * net.sn_mva
        if mode == 'pf_3ph':
            if vector_group.lower() not in ["ynyn", "dyn", "yzn"]:
                raise NotImplementedError("Calculation of 3-phase power flow is only implemented for the transformer "
                                          "vector groups 'YNyn', 'Dyn', 'Yzn'")
            # =============================================================================
            #     Changing base from transformer base to Network base to get Zpu(Net)
            #     Zbase = (kV).squared/S_mva
            #     Zpu(Net)={Zpu(trafo) * Zb(trafo)} / {Zb(Net)}
            #        Note:
            #             Network base voltage is Line-Neutral voltage in each phase
            #             Line-Neutral voltage= Line-Line Voltage(vn_bus_lv) divided by sq.root(3)
            # =============================================================================
            tap_lv = np.square(vn_trafo_lv / vn_bus_lv) * (3 * net.sn_mva)
            tap_hv = np.square(vn_trafo_hv / vn_bus_hv) * (3 * net.sn_mva)

        tap_corr = tap_hv if vector_group.lower() in ("ynd", "yny") else tap_lv
        z_sc = vk0_percent / 100. / sn_trafo_mva * tap_corr
        r_sc = vkr0_percent / 100. / sn_trafo_mva * tap_corr
        z_sc = z_sc.astype(float)
        r_sc = r_sc.astype(float)
        x_sc = np.sign(z_sc) * np.sqrt(z_sc ** 2 - r_sc ** 2)
        # TODO: This equation needs to be checked!
        # z0_k = (r_sc + x_sc * 1j) / parallel  * max(1, ratio) **2
        # z0_k = (r_sc + x_sc * 1j) / parallel * vn_trafo_hv / vn_bus_hv
        # z0_k = (r_sc + x_sc * 1j) / parallel * tap_hv
        z0_k = (r_sc + x_sc * 1j) / parallel
        z_n_ohm = trafos["xn_ohm"].fillna(0).values
        k_st_tr = trafos["k_st"].fillna(1).values

        if mode == "sc":  # or trafo_model == "pi":
            cmax = net._ppc["bus"][lv_buses_ppc, C_MAX]
            kt = _transformer_correction_factor(trafos, vk_percent, vkr_percent, sn_trafo_mva, cmax)
            z0_k *= kt

            # different formula must be applied for power station unit transformers:
            # z_0THV is for power station block unit transformer -> page 20 of IEC60909-4:2021 (example 4.4.2):
            # todo: check if sn_mva must be included here?
            vkx0_percent = np.sqrt(np.square(vk0_percent) - np.square(vkr0_percent))
            z_0THV = (vkr0_percent / 100 + 1j * vkx0_percent / 100) * (np.square(vn_trafo_hv) / sn_trafo_mva) / parallel
            z0_k_psu = (z_0THV * k_st_tr + 3j * z_n_ohm) / ((vn_bus_hv ** 2) / net.sn_mva)
            z0_k = np.where(power_station_unit, z0_k_psu, z0_k)

        y0_k = 1 / z0_k  # adding admittance for "pi" model
        # y0_k = 1 / (z0_k * k_st_tr + 3j * z_n_ohm)  # adding admittance for "pi" model

        # =============================================================================
        #       Transformer magnetising impedance for zero sequence
        # =============================================================================
        z_m = z_sc * mag0_ratio
        x_m = z_m / np.sqrt(mag0_rx ** 2 + 1)
        r_m = x_m * mag0_rx
        r0_trafo_mag = r_m / parallel
        x0_trafo_mag = x_m / parallel
        z0_mag = r0_trafo_mag + x0_trafo_mag * 1j
        # =============================================================================
        #         Star - Delta conversion ( T model to Pi Model)
        #      ----------- |__zc=ZAB__|-----------------
        #            _|                   _|
        #     za=ZAN|_|                  |_| zb=ZBN
        #            |                    |
        # =============================================================================
        z1 = si0_hv_partial * z0_k
        z2 = (1 - si0_hv_partial) * z0_k
        z3 = z0_mag
        z_temp = z1 * z2 + z2 * z3 + z1 * z3
        za = z_temp / z2
#        za = z_temp / (z2+z3)
        zb = z_temp / z1
#        zb = z_temp / (z1+z3)
        zc = z_temp / z3  # ZAB  Transfer impedance
#        zc = z_temp / (z1+z2)  # ZAB  Transfer impedance
        YAB = 1 / zc.astype(complex)
        YAN = 1 / za.astype(complex)
        YBN = 1 / zb.astype(complex)

#        YAB_AN = (zc + za) /(zc * za).astype(complex)  # Series conn YAB and YAN
#        YAB_BN = (zc + zb) / (zc * zb).astype(complex)  # Series conn YAB and YBN

        YAB_AN = 1 / (zc + za).astype(complex)  # Series conn YAB and YAN
        YAB_BN = 1 / (zc + zb).astype(complex)  # Series conn YAB and YBN

        # y0_k = 1 / z0_k #adding admittance for "pi" model
        if vector_group.lower() == "dyn":
            buses_all = np.hstack([buses_all, lv_buses_ppc])
            if trafo_model == "pi":
                y = y0_k * ppc["baseMVA"]  # pi model
            else:
                y = (YAB + YBN).astype(complex) * ppc["baseMVA"]  # T model
            gs_all = np.hstack([gs_all, y.real * in_service])
            bs_all = np.hstack([bs_all, y.imag * in_service])

        elif vector_group.lower() == "ynd":
            buses_all = np.hstack([buses_all, hv_buses_ppc])
            if trafo_model == "pi":
                y = y0_k * ppc["baseMVA"]  # pi model
                # y = 1/0.99598 * 1 / (1/(y0_k * ppc["baseMVA"]) + 1/0.99598 * (1j * 3 * 22 /( (110 ** 2) / 1))) # pi
                # y = 1/0.99598 * 1 / (1/(y0_k * ppc["baseMVA"]) + 1/0.99598 * (1j * 3 * 22 /( (110 ** 2) / 1))) # pi

                # z0_k_k = z0_k * 0.99598 + 1j * 3 * 22 /( (110 ** 2) / 1)
                # print(z0_k_k)
                # y = 1 / z0_k_k # pi model
            else:
                y = (YAB_BN + YAN).astype(complex) * ppc["baseMVA"]  # T model
            gs_all = np.hstack([gs_all, y.real * in_service])
            bs_all = np.hstack([bs_all, y.imag * in_service])

        elif vector_group.lower() == "yyn":
            buses_all = np.hstack([buses_all, lv_buses_ppc])
            if trafo_model == "pi":
                y = 1/(z0_mag+z0_k).astype(complex) * ppc["baseMVA"]  # pi model
            else:
                # y = (YAB_AN + YBN).astype(complex)  # T model
                y = (YAB + YAB_BN + YBN).astype(complex) * ppc["baseMVA"]  # T model

            gs_all = np.hstack([gs_all, y.real * in_service])
            bs_all = np.hstack([bs_all, y.imag * in_service])

        elif vector_group.lower() == "ynyn":
            ppc["branch"][ppc_idx, BR_STATUS] = in_service
            # zc = ZAB
            ppc["branch"][ppc_idx, BR_R] = zc.real
            ppc["branch"][ppc_idx, BR_X] = zc.imag

            buses_all = np.hstack([buses_all, hv_buses_ppc])
            gs_all = np.hstack([gs_all, YAN.real * in_service * ppc["baseMVA"] * tap_lv / tap_hv])
            bs_all = np.hstack([bs_all, YAN.imag * in_service * ppc["baseMVA"] * tap_lv / tap_hv])

            buses_all = np.hstack([buses_all, lv_buses_ppc])
            gs_all = np.hstack([gs_all, YBN.real * in_service * ppc["baseMVA"]])
            bs_all = np.hstack([bs_all, YBN.imag * in_service * ppc["baseMVA"]])

        elif vector_group.lower() == "yny":
            buses_all = np.hstack([buses_all, hv_buses_ppc])
            if trafo_model == "pi":
                y = 1/(z0_mag+z0_k).astype(complex) * ppc["baseMVA"]  # pi model
            else:
                y = (YAB_BN + YAN).astype(complex) * ppc["baseMVA"]  # T model
            gs_all = np.hstack([gs_all, y.real * in_service])
            bs_all = np.hstack([bs_all, y.imag * in_service])

        elif vector_group.lower() == "yzn":
            buses_all = np.hstack([buses_all, lv_buses_ppc])
            #            y = 1/(z0_mag+z0_k).astype(complex)* int(ppc["baseMVA"])#T model
            #            y= (za+zb+zc)/((za+zc)*zb).astype(complex)* int(ppc["baseMVA"])#pi model
            y = (YAB_AN + YBN).astype(complex) * ppc["baseMVA"] ** 2  # T model # why sn_mva squared here?
            gs_all = np.hstack([gs_all, (1.1547) * y.real * in_service])  # what's the 1.1547 value?
            bs_all = np.hstack([bs_all, (1.1547) * y.imag * in_service])

        elif vector_group[-1].isdigit():
            raise ValueError("Unknown transformer vector group %s - "
                             "please specify vector group without "
                             "phase shift number. Phase shift can be "
                             "specified in net.trafo.shift_degree" % vector_group)
        else:
            raise ValueError("Transformer vector group %s is unknown "
                             "/ not implemented for three phase load flow" % vector_group)

    buses, gs, bs = aux._sum_by_group(buses_all, gs_all, bs_all)
    ppc["bus"][buses, GS] += gs
    ppc["bus"][buses, BS] += bs
    del net.trafo["_ppc_idx"]


def _add_gen_sc_impedance_zero(net, ppc):
    mode = net["_options"]["mode"]
    if mode == 'pf_3ph':
        return

    eg = net["gen"][net._is_elements["gen"]]
    if len(eg) == 0:
        return
    eg_buses = eg.bus.values
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    eg_buses_ppc = bus_lookup[eg_buses]

    y0_gen = 1 / (1e3 + 1e3*1j)
    # buses, gs, bs = aux._sum_by_group(eg_buses_ppc, y0_gen.real, y0_gen.imag)
    ppc["bus"][eg_buses_ppc, GS] += y0_gen.real
    ppc["bus"][eg_buses_ppc, BS] += y0_gen.imag


def _add_ext_grid_sc_impedance_zero(net, ppc):
    mode = net["_options"]["mode"]

    if mode == "sc":
        case = net._options["case"]
    else:
        case = "max"
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    eg = net["ext_grid"][net._is_elements["ext_grid"]]
    if len(eg) == 0:
        return
    eg_buses = eg.bus.values
    eg_buses_ppc = bus_lookup[eg_buses]

    if mode == "sc":
        c = ppc["bus"][eg_buses_ppc, C_MAX] if case == "max" else ppc["bus"][eg_buses_ppc, C_MIN]
    else:
        c = 1.1
    if not "s_sc_%s_mva" % case in eg:
        raise ValueError("short circuit apparent power s_sc_%s_mva needs to be specified for " % case +
                         "external grid")
    s_sc = eg["s_sc_%s_mva" % case].values
    if not "rx_%s" % case in eg:
        raise ValueError("short circuit R/X rate rx_%s needs to be specified for external grid" %
                         case)
    rx = eg["rx_%s" % case].values

    z_grid = c / s_sc
    if mode == 'pf_3ph':
        z_grid = c / (s_sc/3)
    x_grid = z_grid / np.sqrt(rx ** 2 + 1)
    r_grid = rx * x_grid
    eg["r"] = r_grid
    eg["x"] = x_grid

    # ext_grid zero sequence impedance
    if case == "max":
        x0_grid = net.ext_grid["x0x_%s" % case].values * x_grid
        r0_grid = net.ext_grid["r0x0_%s" % case].values * x0_grid
    elif case == "min":
        x0_grid = net.ext_grid["x0x_%s" % case].values * x_grid
        r0_grid = net.ext_grid["r0x0_%s" % case].values * x0_grid
    y0_grid = 1 / (r0_grid + x0_grid*1j)

    buses, gs, bs = aux._sum_by_group(eg_buses_ppc, y0_grid.real, y0_grid.imag)
    ppc["bus"][buses, GS] = gs
    ppc["bus"][buses, BS] = bs


def _add_line_sc_impedance_zero(net, ppc):
    branch_lookup = net["_pd2ppc_lookups"]["branch"]
    mode = net["_options"]["mode"]
    if "line" not in branch_lookup:
        return
    line = net["line"]
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    length = line["length_km"].values
    parallel = line["parallel"].values

    fb = bus_lookup[line["from_bus"].values]
    tb = bus_lookup[line["to_bus"].values]
    baseR = np.square(ppc["bus"][fb, BASE_KV]) / net.sn_mva
    if mode == 'pf_3ph':
        baseR = np.square(ppc["bus"][fb, BASE_KV]) / (3*net.sn_mva)
    f, t = branch_lookup["line"]
    # line zero sequence impedance
    ppc["branch"][f:t, F_BUS] = fb
    ppc["branch"][f:t, T_BUS] = tb
    ppc["branch"][f:t, BR_R] = line["r0_ohm_per_km"].values * length / baseR / parallel
    if mode == "sc":
        # temperature correction
        if net["_options"]["case"] == "min":
            ppc["branch"][f:t, BR_R] *= _end_temperature_correction_factor(net, short_circuit=True)
    ppc["branch"][f:t, BR_X] = line["x0_ohm_per_km"].values * length / baseR / parallel
    ppc["branch"][f:t, BR_B] = (2 * net["f_hz"] * math.pi * line["c0_nf_per_km"].values *
                                1e-9 * baseR * length * parallel)
    ppc["branch"][f:t, BR_STATUS] = line["in_service"].astype(np.int64)


def _add_impedance_sc_impedance_zero(net, ppc):
    branch_lookup = net["_pd2ppc_lookups"]["branch"]
    if "impedance" not in branch_lookup:
        return
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    branch = ppc["branch"]

    f, t = branch_lookup["impedance"]

    # impedance zero sequence impedance
    rij, xij, r_asym, x_asym = _calc_impedance_parameters_from_dataframe(net, zero_sequence=True)
    branch[f:t, BR_R] = rij
    branch[f:t, BR_X] = xij
    branch[f:t, BR_R_ASYM] = r_asym
    branch[f:t, BR_X_ASYM] = x_asym
    branch[f:t, F_BUS] = bus_lookup[net.impedance["from_bus"].values]
    branch[f:t, T_BUS] = bus_lookup[net.impedance["to_bus"].values]
    branch[f:t, BR_STATUS] = net["impedance"]["in_service"].values.astype(np.int64)


def _add_trafo3w_sc_impedance_zero(net, ppc):
    # TODO Roman: check this/expand this
    branch_lookup = net["_pd2ppc_lookups"]["branch"]
    if "trafo3w" not in branch_lookup:
        return
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    branch = ppc["branch"]
    f, t = net["_pd2ppc_lookups"]["branch"]["trafo3w"]
    trafo_df = _trafo_df_from_trafo3w(net, sequence=0)
    hv_bus = get_trafo_values(trafo_df, "hv_bus").astype(np.int64)
    lv_bus = get_trafo_values(trafo_df, "lv_bus").astype(np.int64)
    in_service = get_trafo_values(trafo_df, "in_service").astype(np.int64)
    branch[f:t, F_BUS] = bus_lookup[hv_bus]
    branch[f:t, T_BUS] = bus_lookup[lv_bus]

    r, x, _, ratio, shift = _calc_branch_values_from_trafo_df(net, ppc, trafo_df, sequence=0)

    # Y0y0d5,  YN0y0d5,  Y0yn0d5,  YN0yn0d5, Y0y0y0, Y0d5d5,
    # YN0d5d5,  Y0d5y0,  Y0y0d11  und  D0d0d0

    #
    # already implemented:
    #   YNyd
    #   YYnd
    #   YNynd
    #   YNdd
    # not relevant (for 1ph):
    #   Yyy
    #   Ydd
    #   Ydy
    #   Yyd
    #   Ddd

    BIG_NUMBER = 1e20 * ppc["baseMVA"]

    n_t3 = net.trafo3w.shape[0]
    for t3_ix in np.arange(n_t3):
        t3 = net.trafo3w.iloc[t3_ix, :]

        if t3.vector_group.lower() in set(map(lambda vg: "".join(vg), product("dy", repeat=3))):
            x[[t3_ix, t3_ix+n_t3, t3_ix+n_t3*2]] = BIG_NUMBER
            r[[t3_ix, t3_ix+n_t3, t3_ix+n_t3*2]] = BIG_NUMBER
        elif t3.vector_group.lower() == "ynyd":
            # Correction for YNyd
            # z3->y3
            ys = ppc["baseMVA"] / ((x[t3_ix+n_t3*2] * 1j + r[t3_ix+n_t3*2]) * ratio[t3_ix+n_t3*2] ** 2)
            aux_bus = bus_lookup[lv_bus[t3_ix]]
            ppc["bus"][aux_bus, BS] += ys.imag
            ppc["bus"][aux_bus, GS] += ys.real

            # Set y2/y3 to almost 0 to avoid isolated bus
            x[[t3_ix+n_t3, t3_ix+n_t3*2]] = BIG_NUMBER
            r[[t3_ix+n_t3, t3_ix+n_t3*2]] = BIG_NUMBER
        elif t3.vector_group.lower() == "yndy":
            # Correction for YNyd
            # z3->y3
            ys = ppc["baseMVA"] / ((x[t3_ix + n_t3] * 1j + r[t3_ix + n_t3]) * ratio[t3_ix + n_t3] ** 2)
            aux_bus = bus_lookup[lv_bus[t3_ix]]
            ppc["bus"][aux_bus, BS] += ys.imag
            ppc["bus"][aux_bus, GS] += ys.real

            # Set y2/y3 to almost 0 to avoid isolated bus
            x[[t3_ix+n_t3, t3_ix+n_t3*2]] = BIG_NUMBER
            r[[t3_ix+n_t3, t3_ix+n_t3*2]] = BIG_NUMBER
        elif t3.vector_group.lower() == "yynd":
            # z3->y3
            ys = ppc["baseMVA"] / ((x[t3_ix+n_t3*2] * 1j + r[t3_ix+n_t3*2]) * ratio[t3_ix+n_t3*2] ** 2)
            aux_bus = bus_lookup[lv_bus[t3_ix]]
            ppc["bus"][aux_bus, BS] += ys.imag
            ppc["bus"][aux_bus, GS] += ys.real

            # Set y1/y3 to almost 0 to avoid isolated bus
            x[[t3_ix, t3_ix+n_t3*2]] = BIG_NUMBER
            r[[t3_ix, t3_ix+n_t3*2]] = BIG_NUMBER
        elif t3.vector_group.lower() == "ydyn":
            # z3->y3
            ys = ppc["baseMVA"] / ((x[t3_ix+n_t3] * 1j + r[t3_ix+n_t3]) * ratio[t3_ix+n_t3] ** 2)
            aux_bus = bus_lookup[lv_bus[t3_ix]]
            ppc["bus"][aux_bus, BS] += ys.imag
            ppc["bus"][aux_bus, GS] += ys.real

            # Set y1/y3 to almost 0 to avoid isolated bus
            x[[t3_ix, t3_ix+n_t3]] = BIG_NUMBER
            r[[t3_ix, t3_ix+n_t3]] = BIG_NUMBER
        elif t3.vector_group.lower() == "ynynd":
            # z3->y3
            ys = ppc["baseMVA"] / ((x[t3_ix+n_t3*2] * 1j + r[t3_ix+n_t3*2]) * ratio[t3_ix+n_t3*2] ** 2)
            aux_bus = bus_lookup[lv_bus[t3_ix]]
            ppc["bus"][aux_bus, BS] += ys.imag
            ppc["bus"][aux_bus, GS] += ys.real

            # Set y3 to almost 0 to avoid isolated bus
            x[t3_ix+n_t3*2] = BIG_NUMBER
            r[t3_ix+n_t3*2] = BIG_NUMBER
        elif t3.vector_group.lower() == "yndyn":
            # z3->y3
            ys = ppc["baseMVA"] / ((x[t3_ix + n_t3] * 1j + r[t3_ix + n_t3]) * ratio[t3_ix + n_t3] ** 2)
            aux_bus = bus_lookup[lv_bus[t3_ix]]
            ppc["bus"][aux_bus, BS] += ys.imag
            ppc["bus"][aux_bus, GS] += ys.real

            # Set y3 to almost 0 to avoid isolated bus
            x[t3_ix + n_t3] = BIG_NUMBER
            r[t3_ix + n_t3] = BIG_NUMBER
        elif t3.vector_group.lower() == "yndd":
            # Correction for YNdd
            # z3->y3
            # we need a shunt impedance for both "delta" windings -> ys1, ys2
            ys1 = ppc["baseMVA"] / ((x[t3_ix + n_t3] * 1j + r[t3_ix + n_t3]) * ratio[t3_ix + n_t3] ** 2)
            ys2 = ppc["baseMVA"] / ((x[t3_ix + n_t3 * 2] * 1j + r[t3_ix + n_t3 * 2]) * ratio[t3_ix + n_t3 * 2] ** 2)
            aux_bus = bus_lookup[lv_bus[t3_ix]]
            ppc["bus"][aux_bus, BS] += ys1.imag + ys2.imag
            ppc["bus"][aux_bus, GS] += ys1.real + ys2.real

            # Set y2/y3 to almost 0 to avoid isolated bus
            x[[t3_ix + n_t3, t3_ix + n_t3 * 2]] = BIG_NUMBER
            r[[t3_ix + n_t3, t3_ix + n_t3 * 2]] = BIG_NUMBER
        elif t3.vector_group.lower() == "ynyy":
            # Correction for YNyy
            x[[t3_ix, t3_ix + n_t3, t3_ix + n_t3 * 2]] = BIG_NUMBER
            r[[t3_ix, t3_ix + n_t3, t3_ix + n_t3 * 2]] = BIG_NUMBER
        else:
            raise UserWarning(f"{t3.vector_group} not supported yet for trafo3w!")

    branch[f:t, BR_R] = r
    branch[f:t, BR_X] = x
    branch[f:t, BR_B] = 0
    branch[f:t, TAP] = ratio
    branch[f:t, SHIFT] = shift
    branch[f:t, BR_STATUS] = in_service
