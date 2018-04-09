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
    net._ppc0 = ppci
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
    _add_trafo_sc_impedance_zero_new(net, ppc)
    if "trafo3w" in lookup:
        raise NotImplemented("Three winding transformers are not implemented for unbalanced calculations")
    
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

def _add_trafo_sc_impedance_zero_new(net, ppc, trafo_df=None):
    if trafo_df is None:
        trafo_df = net["trafo"]
    branch_lookup = net["_pd2ppc_lookups"]["branch"]
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    mode = net["_options"]["mode"]
    f, t = branch_lookup["trafo"]
    ppc["branch"][f:t, F_BUS] = bus_lookup[trafo_df["hv_bus"].values]
    ppc["branch"][f:t, T_BUS] = bus_lookup[trafo_df["lv_bus"].values]
    vn_trafo_hv, vn_trafo_lv, shift = _calc_tap_from_dataframe(net, trafo_df)
    ratio = _calc_nominal_ratio_from_dataframe(ppc, trafo_df, vn_trafo_hv, vn_trafo_lv,
                                               bus_lookup)
    ppc["branch"][f:t, TAP] = ratio
    ppc["branch"][f:t, SHIFT] = shift
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
#    for idx, trafo_idx in trafo_df.iterrows():
    for idx in trafo_df.index:
        ppc["branch"][f, BR_STATUS] = 0
        trafo_idx = trafo_df[trafo_df.index == idx]
#        bus_lookup = net["_pd2ppc_lookups"]["bus"]
#        vn_lv = get_values(ppc["bus"][:, BASE_KV], trafo_idx["lv_bus"].values, bus_lookup)
        vn_trafo_hv, vn_trafo_lv, shift = _calc_tap_from_dataframe(net, trafo_idx)
        trafo_buses_lv = trafo_idx["lv_bus"].values
        trafo_buses_hv = trafo_idx["hv_bus"].values
        trafo_type = trafo_idx.loc[idx, "vector_group"]
        vn_lv = ppc["bus"][trafo_buses_lv, BASE_KV]
# kt: transformer correction factor 
        if mode == "sc":
            cmax = net._ppc["bus"][bus_lookup[int(trafo_idx["lv_bus"])], C_MAX]
            kt = _transformer_correction_factor(trafo_idx["vsc_percent"], trafo_idx["vscr_percent"], trafo_idx["sn_kva"], cmax)
        else:
            kt = 1.
# zero seq. transformer impedance without kt
        r0_trafo, x0_trafo = _calc_trafo_sc_impedance_zero(net, trafo_idx, vn_lv, vn_trafo_lv, net.sn_kva)
# zero sequence transformer magnetising impedance 
        tap_lv = np.square(vn_trafo_lv / vn_lv) * net.sn_kva
        z_m = (trafo_idx.loc[idx,"vsc0_percent"] * trafo_idx.loc[idx,"mag0_percent"]) / 100. / trafo_idx.loc[idx,"sn_kva"] * tap_lv
        x_m = z_m / np.sqrt(trafo_idx.loc[idx,"mag0_rx"]**2 + 1)
        r_m = x_m * trafo_idx.loc[idx,"mag0_rx"]
        parallel = trafo_idx.loc[idx,"parallel"]
        r0_trafo_mag = r_m / parallel
        x0_trafo_mag = x_m / parallel
        
        if "Dyn" in trafo_type:
#            r0_dyn, x0_dyn = _calc_trafo_sc_impedance_zero(net, trafo_idx, vn_lv, vn_trafo_lv, net.sn_kva)
            r0_dyn = kt * r0_trafo
            x0_dyn = kt * x0_trafo          
            y0_dyn = 1 / (r0_dyn + x0_dyn*1j)
            trafo_buses_ppc = bus_lookup[trafo_buses_lv]
            buses, gs, bs = aux._sum_by_group(trafo_buses_ppc, y0_dyn.real, y0_dyn.imag)
            ppc["bus"][buses, GS] += gs
            ppc["bus"][buses, BS] += bs
             
        elif "YNd" in trafo_type:
#            r0_ynd, x0_ynd = _calc_trafo_sc_impedance_zero(net, trafo_idx, vn_lv, vn_trafo_lv, net.sn_kva)
            r0_ynd = r0_trafo * kt
            x0_ynd = x0_trafo * kt
            y0_ynd = 1/ (r0_ynd + x0_ynd*1j)
            trafo_buses_ppc = bus_lookup[trafo_buses_hv]
            buses, gs, bs = aux._sum_by_group(trafo_buses_ppc, y0_ynd.real, y0_ynd.imag)
            ppc["bus"][buses, GS] += gs
            ppc["bus"][buses, BS] += bs

        elif "Yyn" in trafo_type:
#            r0_trafo, x0_trafo = _calc_trafo_sc_impedance_zero(net, trafo_idx, vn_lv, vn_trafo_lv, net.sn_kva)
#            r0_trafo_mag, x0_trafo_mag = _calc_trafo_mag_impedance_zero(net, trafo_idx, vn_lv, vn_trafo_lv, net.sn_kva)
            r0_yyn = r0_trafo*kt + r0_trafo_mag
            x0_yyn = x0_trafo*kt + x0_trafo_mag
            y0_yyn = 1 / (r0_yyn + x0_yyn*1j)
            trafo_buses_ppc = bus_lookup[trafo_buses_lv]
            buses, gs, bs = aux._sum_by_group(trafo_buses_ppc, y0_yyn.real, y0_yyn.imag)
            ppc["bus"][buses, GS] += gs
            ppc["bus"][buses, BS] += bs

        elif "YNy" in trafo_type and "YNyn" not in trafo_type:
#            r0_trafo, x0_trafo = _calc_trafo_sc_impedance_zero(net, trafo_idx, vn_lv, vn_trafo_lv, net.sn_kva)
#            r0_trafo_mag, x0_trafo_mag = _calc_trafo_mag_impedance_zero(net, trafo_idx, vn_lv, vn_trafo_lv, net.sn_kva)
            r0_yny = r0_trafo*kt + r0_trafo_mag
            x0_yny = x0_trafo*kt + x0_trafo_mag
            y0_yny = 1 / (r0_yny + x0_yny*1j)
            trafo_buses_ppc = bus_lookup[trafo_buses_hv]
            buses, gs, bs = aux._sum_by_group(trafo_buses_ppc, y0_yny.real, y0_yny.imag)
            ppc["bus"][buses, GS] += gs
            ppc["bus"][buses, BS] += bs

        elif "YNyn" in trafo_type:
            ppc["branch"][f, BR_STATUS] = 1
#            r0_ynyn, x0_ynyn = _calc_trafo_sc_impedance_zero(net, trafo_idx, vn_lv, vn_trafo_lv, net.sn_kva)
            r0_ynyn = kt * r0_trafo
            x0_ynyn = kt * x0_trafo
            tap_lv = np.square(vn_trafo_lv / vn_lv) * net.sn_kva
            z_m = (trafo_idx.loc[idx,"vsc0_percent"] * trafo_idx.loc[idx,"mag0_percent"]) / 100. / trafo_idx.loc[idx,"sn_kva"] * tap_lv
            x_m = z_m / np.sqrt(trafo_idx.loc[idx,"mag0_rx"]**2 + 1)
            r_m = x_m * trafo_idx.loc[idx,"mag0_rx"]
            parallel = trafo_idx.loc[idx,"parallel"]
            r0_trafo_mag = r_m / parallel
            x0_trafo_mag = x_m / parallel
# convert the t model to pi model

            z1 = trafo_idx.loc[idx,"si0_hv_partial"] * (r0_ynyn + x0_ynyn*1j)
            z2 = (1 - trafo_idx.loc[idx,"si0_hv_partial"]) * (r0_ynyn + x0_ynyn*1j)
            z3 = r0_trafo_mag + x0_trafo_mag*1j

            z_temp = z1*z2 + z2*z3 + z1*z3
            za = z_temp / z2
            zb = z_temp / z1
            zc = z_temp / z3

            ppc["branch"][f, BR_R] = zc.real
            ppc["branch"][f, BR_X] = zc.imag
            ppc["branch"][f, BR_B] = (2/za).imag - (2/za).real*1j
            if trafo_idx.loc[idx, "si0_hv_partial"] != 0.5:
# add a shunt element parallel to zb if the leakage impedance distribution is unequal                
                zs = (za * zb)/(za - zb)
                ys = 1 / zs
                t += 1
                
                trafo_buses_ppc = bus_lookup[trafo_buses_lv]
                buses, gs, bs = aux._sum_by_group(trafo_buses_ppc, ys.real, ys.imag)
                ppc["bus"][buses, GS] += gs
                ppc["bus"][buses, BS] += bs

        elif "Yy" in trafo_type and "Yyn" not in trafo_type:
            pass

        elif "Dy" in trafo_type and "Dyn" not in trafo_type:
            pass          

        elif "Yd" in trafo_type:
            pass

        elif "Dd" in trafo_type:
            pass
        f += 1

def _calc_trafo_sc_impedance_zero(net, trafo_df, vn_lv, vn_trafo_lv, sn_kva):
    """
        calculates the zero sequence trafo impedance with correction factor
    """
    tap_lv = np.square(vn_trafo_lv / vn_lv) * sn_kva  # adjust for low voltage side voltage converter
    sn_trafo_kva = trafo_df.sn_kva.values
    parallel = trafo_df["parallel"].values
#    mode = net["_options"]["mode"]
    z_sc = trafo_df["vsc0_percent"].values / 100. / sn_trafo_kva * tap_lv
    r_sc = trafo_df["vscr0_percent"].values / 100. / sn_trafo_kva * tap_lv
    z_sc = z_sc.astype(float)
    r_sc = r_sc.astype(float)
#    z_sc = np.float(z_sc)
#    r_sc = np.float(r_sc)
    x_sc = np.sign(z_sc) * np.sqrt(z_sc**2 - r_sc**2)
    return r_sc / parallel, x_sc / parallel

#    if mode == "sc":
#        if trafo_df.equals(net.trafo):
#            from pandapower.shortcircuit.idx_bus import C_MAX
#            bus_lookup = net._pd2ppc_lookups["bus"]
#            cmax = net._ppc["bus"][bus_lookup[net.trafo.lv_bus.values], C_MAX]
#            print("cmax=",cmax)
#            kt = _transformer_correction_factor(trafo_df["vsc_percent"], trafo_df["vscr_percent"], trafo_df["sn_kva"], cmax)
#            print("kt=",kt)
#            r_sc *= kt
#            x_sc *= kt
#    return r_sc / parallel, x_sc / parallel

#def _calc_trafo_mag_impedance_zero(net, trafo_df, vn_lv, vn_trafo_lv, sn_kva):
#    """
#        calculates the zero sequence trafo magnetising impedance
#    """
#    tap_lv = np.square(vn_trafo_lv / vn_lv) * sn_kva  # adjust for low voltage side voltage converter
#    sn_trafo_kva = trafo_df.sn_kva.values
#    parallel = trafo_df["parallel"].values
#    z_m = (trafo_df["vsc0_percent"].values * trafo_df["mag0_percent"].values) / 100. / sn_trafo_kva * tap_lv
#    z_m = np.float(z_m)
#    x_m = z_m / np.sqrt(trafo_df["mag0_rx"].astype(float)**2 + 1)
#    r_m = x_m * trafo_df["mag0_rx"]
#    
#    return r_m / parallel, x_m / parallel