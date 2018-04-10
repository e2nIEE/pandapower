# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import math
import numpy as np
import copy
import pandapower.auxiliary as aux
from pandapower.pd2ppc import _init_ppc
from pandapower.build_bus import _build_bus_ppc
from pandapower.build_gen import _init_ppc_gen, _build_gen_ppc
from pandapower.pd2ppc import _pd2ppc, _ppc2ppci
from pandapower.idx_brch import BR_B, BR_R, BR_X, F_BUS, T_BUS, branch_cols, BR_STATUS, SHIFT, TAP
from pandapower.idx_bus import BASE_KV, BS, GS
from pandapower.build_branch import _calc_tap_from_dataframe, _transformer_correction_factor, _calc_nominal_ratio_from_dataframe
from pandapower.shortcircuit.idx_bus import C_MAX
from pandapower.build_branch import _switch_branches, _branches_with_oos_buses, _initialize_branch_lookup
#

def _pd2ppc_zero(net):
    """
    Converter Flow:
        1. Create an empty pypower datatructure
        2. Calculate loads and write the bus matrix
        3. Build the gen (Infeeder)- Matrix
        4. Calculate the line parameter and the transformer parameter,
           and fill it in the branch matrix.
           Order: 1st: Line values, 2nd: Trafo values
        5. if opf: make opf objective (gencost)
        6. convert internal ppci format for pypower powerflow / opf without out of service elements and rearanged buses

    INPUT:
        **net** - The pandapower format network

    OUTPUT:
        **ppc** - The simple matpower format network. Which consists of:
                  ppc = {
                        "baseMVA": 1., *float*
                        "version": 2,  *int*
                        "bus": np.array([], dtype=float),
                        "branch": np.array([], dtype=np.complex128),
                        "gen": np.array([], dtype=float),
                        "gencost" =  np.array([], dtype=float), only for OPF
                        "internal": {
                              "Ybus": np.array([], dtype=np.complex128)
                              , "Yf": np.array([], dtype=np.complex128)
                              , "Yt": np.array([], dtype=np.complex128)
                              , "branch_is": np.array([], dtype=bool)
                              , "gen_is": np.array([], dtype=bool)
                              }
        **ppci** - The "internal" pypower format network for PF calculations
    """
    # select elements in service (time consuming, so we do it once)
    net["_is_elements"] = aux._select_is_elements_numba(net)

    # get options
    mode = net["_options"]["mode"]

    ppc = _init_ppc(net)
    # init empty ppci
    ppci = copy.deepcopy(ppc)
    _build_bus_ppc(net, ppc)
    _build_gen_ppc(net, ppc)
    if mode == "sc":
        _add_ext_grid_sc_impedance_zero(net, ppc)
    _build_branch_ppc_zero(net, ppc)

    # adds auxilary buses for open switches at branches
    _switch_branches(net, ppc)

    # add auxilary buses for out of service buses at in service lines.
    # Also sets lines out of service if they are connected to two out of service buses
    _branches_with_oos_buses(net, ppc)

    # generates "internal" ppci format (for powerflow calc) from "external" ppc format and updates the bus lookup
    # Note: Also reorders buses and gens in ppc
    ppci = _ppc2ppci(ppc, ppci, net)
    net._ppc0 = ppc
    return ppc, ppci

def _build_branch_ppc_zero(net, ppc):
    """
    Takes the empty ppc network and fills it with the branch values. The branch
    datatype will be np.complex 128 afterwards.

    .. note:: The order of branches in the ppc is:
            1. Lines
            2. Transformers
            3. 3W Transformers (each 3W Transformer takes up three branches)
            4. Impedances
            5. Internal branch for extended ward

    **INPUT**:
        **net** -The pandapower format network

        **ppc** - The PYPOWER format network to fill in values

    """
    length = _initialize_branch_lookup(net)
    lookup = net._pd2ppc_lookups["branch"]
    mode = net._options["mode"]
    ppc["branch"] = np.zeros(shape=(length, branch_cols), dtype=np.complex128)
    if mode == "sc":
        from pandapower.shortcircuit.idx_brch import branch_cols_sc
        branch_sc = np.empty(shape=(length, branch_cols_sc), dtype=float)
        branch_sc.fill(np.nan)
        ppc["branch"] = np.hstack((ppc["branch"], branch_sc ))
    ppc["branch"][:, :13] = np.array([0, 0, 0, 0, 0, 250, 250, 250, 1, 0, 1, -360, 360])
    _add_line_sc_impedance_zero(net, ppc)
    _add_trafo_sc_impedance_zero(net, ppc)
    if "trafo3w" in lookup:
        raise NotImplemented("Three winding transformers are not implemented for unbalanced calculations")


def _add_trafo_sc_impedance_zero(net, ppc, trafo_df=None):
    if trafo_df is None:
        trafo_df = net["trafo"]
    branch_lookup = net["_pd2ppc_lookups"]["branch"]
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    mode = net["_options"]["mode"]
    f, t = branch_lookup["trafo"]
    trafo_df["_ppc_idx"] = range(f,t)
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    buses_all, gs_all, bs_all = np.array([], dtype=int), np.array([]), np.array([])
    for vector_group, trafos in trafo_df.groupby("vector_group"):
        ppc_idx = trafos["_ppc_idx"]
        ppc["branch"][ppc_idx, BR_STATUS] = 0

        if vector_group in ["Yy", "Yd", "Dy", "Dd"]:
            continue

        vsc_percent = trafos["vsc_percent"].values.astype(float)
        vscr_percent = trafos["vscr_percent"].values.astype(float)
        sn_kva = trafos["sn_kva"].values.astype(float)
        vsc0_percent = trafos["vsc0_percent"].values.astype(float)
        vscr0_percent = trafos["vscr0_percent"].values.astype(float)
        lv_buses = trafos["lv_bus"].values.astype(int)
        hv_buses = trafos["hv_bus"].values.astype(int)
        lv_buses_ppc = bus_lookup[lv_buses]
        hv_buses_ppc = bus_lookup[hv_buses]
        mag0_percent = trafos.mag0_percent.values.astype(float)
        mag0_rx = trafos["mag0_rx"].values.astype(float)
        si0_hv_partial = trafos.si0_hv_partial.values.astype(float)
        parallel = trafos.parallel.values.astype(float)
        in_service = trafos["in_service"].astype(int)

        ppc["branch"][ppc_idx, F_BUS] = hv_buses_ppc
        ppc["branch"][ppc_idx, T_BUS] = lv_buses_ppc

        vn_trafo_hv, vn_trafo_lv, shift = _calc_tap_from_dataframe(net, trafos)
        vn_lv = ppc["bus"][lv_buses_ppc, BASE_KV]
        ratio = _calc_nominal_ratio_from_dataframe(ppc, trafos, vn_trafo_hv, vn_trafo_lv,
                                                   bus_lookup)
        ppc["branch"][ppc_idx, TAP] = ratio
        ppc["branch"][ppc_idx, SHIFT] = shift

        # zero seq. transformer impedance
        tap_lv = np.square(vn_trafo_lv / vn_lv) * net.sn_kva  # adjust for low voltage side voltage converter
        z_sc = vsc0_percent / 100. / sn_kva * tap_lv
        r_sc = vscr0_percent / 100. / sn_kva * tap_lv
        z_sc = z_sc.astype(float)
        r_sc = r_sc.astype(float)
        x_sc = np.sign(z_sc) * np.sqrt(z_sc**2 - r_sc**2)
        z0_k = (r_sc + x_sc * 1j) / parallel
        if mode == "sc":
            cmax = net._ppc["bus"][lv_buses_ppc, C_MAX]
            kt = _transformer_correction_factor(vsc_percent, vscr_percent, sn_kva, cmax)
            z0_k *= kt
        y0_k = 1 / z0_k
        # zero sequence transformer magnetising impedance 
        z_m = vsc0_percent * mag0_percent / 100. / sn_kva * tap_lv
        x_m = z_m / np.sqrt(mag0_rx**2 + 1)
        r_m = x_m * mag0_rx
        r0_trafo_mag = r_m / parallel
        x0_trafo_mag = x_m / parallel
        z0_mag = r0_trafo_mag + x0_trafo_mag * 1j

        if vector_group == "Dyn":
            buses_all = np.hstack([buses_all, lv_buses_ppc])
            gs_all = np.hstack([gs_all, y0_k.real*in_service])
            bs_all = np.hstack([bs_all, y0_k.imag*in_service])
             
        elif vector_group == "YNd":
            buses_all = np.hstack([buses_all, hv_buses_ppc])
            gs_all = np.hstack([gs_all, y0_k.real*in_service])
            bs_all = np.hstack([bs_all, y0_k.imag*in_service])

        elif vector_group == "Yyn":
            buses_all = np.hstack([buses_all, lv_buses_ppc])
            y = 1/(z0_mag+z0_k).astype(complex)
            gs_all = np.hstack([gs_all, y.real*in_service])
            bs_all = np.hstack([bs_all, y.imag*in_service])

        elif vector_group == "YNyn":
            ppc["branch"][ppc_idx, BR_STATUS] = in_service
            # convert the t model to pi model
            z1 = si0_hv_partial * z0_k
            z2 = (1 - si0_hv_partial) * z0_k
            z3 = z0_mag

            z_temp = z1*z2 + z2*z3 + z1*z3
            za = z_temp / z2
            zb = z_temp / z1
            zc = z_temp / z3

            ppc["branch"][ppc_idx, BR_R] = zc.real
            ppc["branch"][ppc_idx, BR_X] = zc.imag
            ppc["branch"][ppc_idx, BR_B] = (2/za).imag - (2/za).real*1j
            # add a shunt element parallel to zb if the leakage impedance distribution is unequal
            #TODO: this only necessary if si0_hv_partial!=0.5 --> test
            zs = (za * zb)/(za - zb)
            ys = 1 / zs.astype(complex)
            buses_all = np.hstack([buses_all, lv_buses_ppc])
            gs_all = np.hstack([gs_all, ys.real*in_service])
            bs_all = np.hstack([bs_all, ys.imag*in_service])
        elif vector_group=="YNy":
            buses_all = np.hstack([buses_all, hv_buses_ppc])
            y = 1/(z0_mag+z0_k).astype(complex)
            gs_all = np.hstack([gs_all, y.real*in_service])
            bs_all = np.hstack([bs_all, y.imag*in_service])
        elif vector_group[-1].isdigit():
            raise ValueError("Unknown transformer vector group %s - please specify vector group without phase shift number. Phase shift can be specified in net.trafo.shift_degree"%vector_group)
        else:
            raise ValueError("Transformer vector group %s is unknown / not implemented"%vector_group)

    buses, gs, bs = aux._sum_by_group(buses_all, gs_all, bs_all)
    ppc["bus"][buses, GS] += gs
    ppc["bus"][buses, BS] += bs

def _add_ext_grid_sc_impedance_zero(net, ppc):
    from pandapower.shortcircuit.idx_bus import C_MAX, C_MIN
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    case = net._options["case"]
    eg = net["ext_grid"][net._is_elements["ext_grid"]]
    if len(eg) == 0:
        return
    eg_buses = eg.bus.values
    eg_buses_ppc = bus_lookup[eg_buses]

    c = ppc["bus"][eg_buses_ppc, C_MAX] if case == "max" else ppc["bus"][eg_buses_ppc, C_MIN]
    if not "s_sc_%s_mva" % case in eg:
        raise ValueError("short circuit apparent power s_sc_%s_mva needs to be specified for "% case +
                         "external grid" )
    s_sc = eg["s_sc_%s_mva" % case].values
    if not "rx_%s" % case in eg:
        raise ValueError("short circuit R/X rate rx_%s needs to be specified for external grid" %
                         case)
    rx = eg["rx_%s" % case].values

    z_grid = c / s_sc
    x_grid = z_grid / np.sqrt(rx ** 2 + 1)
    r_grid = rx * x_grid
    eg["r"] = r_grid
    eg["x"] = x_grid

    # ext_grid zero sequence impedance 
    if case == "max":
        x0_grid = net.ext_grid["x0x_%s" % case] * x_grid
        r0_grid = net.ext_grid["r0x0_%s" % case] * x0_grid
    elif case == "min":
        x0_grid = net.ext_grid["x0x_%s" % case] * x_grid
        r0_grid = net.ext_grid["r0x0_%s" % case] * x0_grid       
    y0_grid = 1 / (r0_grid + x0_grid*1j)
    buses, gs, bs = aux._sum_by_group(eg_buses_ppc, y0_grid.real, y0_grid.imag)
    ppc["bus"][buses, GS] = gs
    ppc["bus"][buses, BS] = bs


def _add_line_sc_impedance_zero(net, ppc):
    line = net["line"]
    branch_lookup = net["_pd2ppc_lookups"]["branch"]
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    length = line["length_km"].values
    parallel = line["parallel"].values
    fb = bus_lookup[line["from_bus"].values]
    baseR = np.square(ppc["bus"][fb, BASE_KV]) / net.sn_kva * 1e3

    if "line" in branch_lookup:
        f, t = branch_lookup["line"]
        # line zero sequence impedance
        ppc["branch"][f:t, F_BUS] = bus_lookup[line["from_bus"].values]
        ppc["branch"][f:t, T_BUS] = bus_lookup[line["to_bus"].values]
        ppc["branch"][f:t, BR_R] = line["r0_ohm_per_km"].values * length / baseR / parallel
        ppc["branch"][f:t, BR_X] = line["x0_ohm_per_km"].values * length / baseR / parallel
        ppc["branch"][f:t, BR_B] = (2 * net["f_hz"] * math.pi * line["c0_nf_per_km"].values * 1e-9 * baseR * length * parallel)
        ppc["branch"][f:t, BR_STATUS] = line["in_service"].astype(int)