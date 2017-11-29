# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 11:45:31 2017

@author: Lu
"""

import math
import numpy as np

from pandapower.pd2ppc import _pd2ppc, _ppc2ppci
from pandapower.idx_bus import BASE_KV, BS, GS
from pandapower.auxiliary import _sum_by_group, get_values
from pandapower.idx_brch import BR_B, BR_R, BR_X
from pandapower.build_branch import _calc_tap_from_dataframe, _transformer_correction_factor
from pandapower.build_branch import _calc_r_x_y_from_dataframe, _end_temperature_correction_factor



def _pd2ppc_zero(net):
    """
        Input: 
        **net** - the pandapower format network
        Output:
        **ppc_0** - the matpower format zero sequence network
        **ppci_0** - the "internal" pypower format zero sequence network
    """

    ppc_0, ppci_0 = _pd2ppc(net)
    _add_ext_grid_sc_impedance_zero(net, ppc_0)
    _add_trafo_sc_impedance_zero(net, ppc_0)
    _calc_line_parameter_zero(net, ppc_0)
    ppci_0 = _ppc2ppci(ppc_0, ppci_0, net)
    return ppc_0, ppci_0



def _add_ext_grid_sc_impedance_zero(net, ppc):
    """
        fills the zero sequence impedance of external grid in ppc
    """
    from pandapower.shortcircuit.idx_bus import C_MAX, C_MIN
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    case = net._options["case"]
    eg = net["ext_grid"][net._is_elements["ext_grid"]]
    if len(eg) == 0:
        return
    eg_buses = eg.bus.values
    eg_buses_ppc  = bus_lookup[eg_buses]

    c = ppc["bus"][eg_buses_ppc, C_MAX] if case == "max" else ppc["bus"][eg_buses_ppc, C_MIN]
    if not "s_sc_%s_mva"%case in eg:
        raise ValueError("short circuit apparent power s_sc_%s_mva needs to be specified for external grid"%case)
    s_sc = eg["s_sc_%s_mva"%case].values
    if not "rx_%s"%case in eg:
        raise ValueError("short circuit R/X rate rx_%s needs to be specified for external grid"%case)
    rx = eg["rx_%s"%case].values
    # ext_grid impedance positive sequence
    z_grid = c / s_sc
    x_grid = z_grid / np.sqrt(rx**2 + 1)
    r_grid = rx * x_grid
    # ext_grid impedance zero sequence
    if case == "max":
        x0_grid = net.ext_grid["x0x_max"] * x_grid                     #x0x: ratio of the ext_grid reactance between zero seq and positiv seq
        r0_grid = net.ext_grid["r0x0_max"] * x0_grid                   #r0x0: ratio of the ext_grid resistance and reactance at zero sequence
        y0_grid = 1 / (r0_grid + x0_grid * 1j)
    elif case == "min":
        x0_grid = net.ext_grid["x0x_min"] * x_grid                     #x0x: ratio of the ext_grid reactance between zero seq and positiv seq
        r0_grid = net.ext_grid["r0x0_min"] * x0_grid                   #r0x0: ratio of the ext_grid resistance and reactance at zero sequence
        y0_grid = 1 / (r0_grid + x0_grid * 1j)
    buses, gs, bs = _sum_by_group(eg_buses_ppc, y0_grid.real, y0_grid.imag)
    ppc["bus"][buses, GS] = gs
    ppc["bus"][buses, BS] = bs
    
def _add_trafo_sc_impedance_zero(net, ppc, trafo_df=None):
    """
        add trafos'values of different vector groups in ppc_0 dataframe
    """
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    trafo_buses_lv = net["trafo"]["lv_bus"].values
    trafo_buses_hv = net["trafo"]["hv_bus"].values
    trafo_type = str(net["trafo"]["vector_group"])
    lookup = net._pd2ppc_lookups["branch"]

    if trafo_df is None:
        trafo_df = net["trafo"]
        
    vn_lv = get_values(ppc["bus"][:, BASE_KV], trafo_df["lv_bus"].values, bus_lookup)
    vn_trafo_hv, vn_trafo_lv, shift = _calc_tap_from_dataframe(net, trafo_df)
    r0_trafo, x0_trafo = _calc_trafo_impedance_zero(net, trafo_df, vn_lv, vn_trafo_lv, net.sn_kva)
    r0_trafo_mag, x0_trafo_mag = _calc_trafo_mag_impedance_zero(net, trafo_df, vn_lv, vn_trafo_lv, net.sn_kva)
    y0_trafo = 1 / (r0_trafo + x0_trafo * 1j)

    
    # delete trafo in ppc["branch"]
    if "trafo" in lookup:
        if "YNy" not in trafo_type:
            f, t = lookup["trafo"]
            ppc["branch"] = np.delete(ppc["branch"], [f,t], 0)
        elif "YNyn" in trafo_type:
            # convert the t model to pi model  
            z1 = trafo_df["si_hv_partial"].values * (r0_trafo + x0_trafo * 1j)
            z2 = (1 - trafo_df["si_hv_partial"]).values * (r0_trafo + x0_trafo * 1j)
            z3 = r0_trafo_mag + x0_trafo_mag * 1j
            
            z_temp = z1*z2 + z2*z3 + z1*z3
            za = z_temp / z2
            zb = z_temp / z1
            zc = z_temp / z3
            
            if trafo_df["si_hv_partial"].values == 0.5:
                f, t = lookup["trafo"]
                ppc["branch"][f:t, BR_R] = zc.real
                ppc["branch"][f:t, BR_X] = zc.imag
                ppc["branch"][f:t, BR_B] = (2/za).imag - (2/za).real*1j
            else:             
                # add a shunt element parallel to zb if the leakage impedance distribution is unequal                
                zs = (za * zb)/(za- zb)
                ys = 1 / zs
                      
                f, t = lookup["trafo"]
                ppc["branch"][f:t, BR_R] = zc.real
                ppc["branch"][f:t, BR_X] = zc.imag
                ppc["branch"][f:t, BR_B] = (2/za).imag - (2/za).real*1j
                
                trafo_buses_ppc = bus_lookup[trafo_buses_lv]
                buses, gs, bs = _sum_by_group(trafo_buses_ppc, ys.real, ys.imag)
                ppc["bus"][buses, GS] += gs
                ppc["bus"][buses, BS] += bs
        else:
            f, t = lookup["trafo"]
            ppc["branch"] = np.delete(ppc["branch"], [f,t], 0)
            trafo_buses_ppc = bus_lookup[trafo_buses_hv]
            r0_yny = r0_trafo_mag + r0_trafo
            x0_yny = x0_trafo_mag + x0_trafo
            y0_yny = 1 / (r0_yny + x0_yny * 1j)
            buses, gs, bs = _sum_by_group(trafo_buses_ppc, y0_yny.real, y0_yny.imag)
            ppc["bus"][buses, GS] += gs
            ppc["bus"][buses, BS] += bs                     

    # add trafo values in ppc["bus"] according to vector groups
    if "Dyn" in trafo_type:
        trafo_buses_ppc = bus_lookup[trafo_buses_lv]
        buses, gs, bs = _sum_by_group(trafo_buses_ppc, y0_trafo.real, y0_trafo.imag)
        ppc["bus"][buses, GS] = gs
        ppc["bus"][buses, BS] = bs
    elif "Yyn" in trafo_type:
        trafo_buses_ppc = bus_lookup[trafo_buses_lv]
        r0_yyn = r0_trafo_mag + r0_trafo
        x0_yyn = x0_trafo_mag + x0_trafo
        y0_yyn = 1 / (r0_yyn + x0_yyn * 1j)
        buses, gs, bs = _sum_by_group(trafo_buses_ppc, y0_yyn.real, y0_yyn.imag)
        ppc["bus"][buses, GS] = gs
        ppc["bus"][buses, BS] = bs
    elif "YNd" in trafo_type:
        trafo_buses_ppc = bus_lookup[trafo_buses_hv]
        buses, gs, bs = _sum_by_group(trafo_buses_ppc, y0_trafo.real, y0_trafo.imag)
        ppc["bus"][buses, GS] += gs
        ppc["bus"][buses, BS] += bs

    elif "Yy" == trafo_type:
        pass
    elif "Yd" == trafo_type:
        pass
    elif "Dd" == trafo_type:
        pass
    elif "Dy" == trafo_type:
        pass


        
        

def _calc_trafo_impedance_zero(net, trafo_df, vn_lv, vn_trafo_lv, sn_kva):
    """
        calculates the zero sequence trafo impedance with correction factor
    """
    tap_lv = np.square(vn_trafo_lv / vn_lv) * sn_kva  # adjust for low voltage side voltage converter
    sn_trafo_kva = trafo_df.sn_kva.values
    parallel = trafo_df["parallel"].values
    mode = net["_options"]["mode"]
    z_sc = trafo_df["vsc0_percent"].values / 100. / sn_trafo_kva * tap_lv
    r_sc = trafo_df["vscr0_percent"].values / 100. / sn_trafo_kva * tap_lv
    x_sc = np.sign(z_sc) * np.sqrt(z_sc ** 2 - r_sc ** 2)
#    print (r_sc, x_sc)
    if mode == "sc":
        if trafo_df.equals(net.trafo):
            from pandapower.shortcircuit.idx_bus import C_MAX
            bus_lookup = net._pd2ppc_lookups["bus"]
            cmax = net._ppc["bus"][bus_lookup[net.trafo.lv_bus.values], C_MAX]
            kt = _transformer_correction_factor(trafo_df.vsc0_percent, trafo_df.vscr0_percent,
                                                trafo_df.sn_kva, cmax)
            r_sc *= kt
            x_sc *= kt
    return r_sc / parallel, x_sc / parallel


def _calc_trafo_mag_impedance_zero(net, trafo_df, vn_lv, vn_trafo_lv, sn_kva):
    """
        calculates the zero sequence trafo magnetising impedance
    """
    tap_lv = np.square(vn_trafo_lv / vn_lv) * sn_kva  # adjust for low voltage side voltage converter
#    print (tap_lv)
    sn_trafo_kva = trafo_df.sn_kva.values
    parallel = trafo_df["parallel"].values
    mode = net["_options"]["mode"]
    z_m = (trafo_df["vsc0_percent"] * trafo_df["mag_percent"]).values / 100. / sn_trafo_kva * tap_lv
    x_m = z_m / np.sqrt(trafo_df["mag_r0x0"]**2 + 1)
    r_m = x_m * trafo_df["mag_r0x0"]

    return r_m / parallel, x_m / parallel



def _calc_line_parameter_zero(net, ppc):
    """
        calculates the line parameters in zero sequence network.
        fills the values in ppc.
    """
    line = net["line"]
    lookup = net._pd2ppc_lookups["branch"]
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    length = line["length_km"].values
    parallel = line["parallel"].values
    fb = bus_lookup[line["from_bus"].values]
    baseR = np.square(ppc["bus"][fb, BASE_KV]) / net.sn_kva * 1e3
    k_t = _end_temperature_correction_factor(net)
    if "line" in lookup:
        f, t = lookup["line"]
        ppc["branch"][f:t, BR_B] = (2 * net.f_hz * math.pi * line["c0_nf_per_km"].values * 1e-9 * baseR *
                     length * parallel)
        if net["_options"]["case"] == "max": 
            ppc["branch"][f:t, BR_R] = line["r0_ohm_per_km"].values * length / baseR / parallel
        elif net["_options"]["case"] == "min":
            ppc["branch"][f:t, BR_R] = (line["r0_ohm_per_km"] * k_t).values * length / baseR / parallel
        ppc["branch"][f:t, BR_X] = line["x0_ohm_per_km"].values * length / baseR / parallel

            
        








