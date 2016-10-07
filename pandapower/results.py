from __future__ import absolute_import
# -*- coding: utf-8 -*-

__author__ = 'tdess, lthurner, scheidler'


import numpy as np

from pypower.idx_brch import F_BUS, T_BUS, PF, QF, PT, QT
from pypower.idx_bus import BASE_KV, VM, VA, PD, QD
from pypower.idx_gen import PG, QG

from pandapower.auxiliary import get_indices
from pandapower.build_bus import _sum_by_group

def _extract_results(net, mpc, gen_is, eg_is, bus_lookup, trafo_loading, return_voltage_angles):
    _set_busses_out_of_service(mpc)
    bus_pq = _get_p_q_results(net, mpc, bus_lookup)
    _get_shunt_results(net, mpc, bus_lookup, bus_pq)
    _get_branch_results(net, mpc, bus_lookup, bus_pq, trafo_loading)
    _get_gen_results(net, mpc, gen_is, eg_is, bus_lookup, bus_pq, return_voltage_angles)
    _get_bus_results(net, mpc, bus_lookup, bus_pq, return_voltage_angles)


def _set_busses_out_of_service(mpc):   
    disco = np.where(mpc["bus"][:, 1] == 4)[0]
    mpc["bus"][disco, VM] = np.nan
    mpc["bus"][disco, VA] = np.nan
    mpc["bus"][disco, PD] = 0
    mpc["bus"][disco, QD] = 0

    
def _get_bus_results(net, mpc, bus_lookup, bus_pq, return_voltage_angles, res_bus_table = "res_bus"):   
    net[res_bus_table]["p_kw"] = bus_pq[:, 0]
    net[res_bus_table]["q_kvar"] = bus_pq[:, 1]
    
    ppi = net["bus"].index.values
    bus_idx = get_indices(ppi, bus_lookup)
    net[res_bus_table]["vm_pu"] = mpc["bus"][bus_idx][:, VM]
    net[res_bus_table].index = net["bus"].index
    if return_voltage_angles:
        net[res_bus_table]["va_degree"] = mpc["bus"][bus_idx][:, VA]

def _get_branch_results(net, mpc, bus_lookup, pq_busses, trafo_loading, res_line_table="res_line", 
                        res_trafo_table="res_trafo", res_trafo3w_table="res_trafo3w", 
                        res_impedance_table="res_impedance", res_xward_table="res_xward" ):
    """
    Extract the bus results and writes it in the Dataframe net.res_line and net.res_trafo.

    INPUT:

        **results** - the result of runpf loadflow calculation

        **p** - the dict to dump the "res_line" and "res_trafo" Dataframe

    """
    line_end = len(net["line"])
    trafo_end =  line_end + len(net["trafo"])
    trafo3w_end = trafo_end + len(net["trafo3w"]) * 3
    impedance_end = trafo3w_end + len(net["impedance"])
    xward_end = impedance_end + len(net["xward"])
    
    i_ft, s_ft = _get_branch_flows(net, mpc)
    if line_end > 0:
        _get_line_results(net, mpc, i_ft, 0, line_end, res_line_table=res_line_table )
        net[res_line_table].index = net["line"].index
    
    if trafo_end > line_end:
        _get_trafo_results(net, mpc, trafo_loading, s_ft, i_ft, line_end, trafo_end, 
                           res_trafo_table=res_trafo_table)
        net[res_trafo_table].index = net["trafo"].index
    
    if trafo3w_end > trafo_end:
        _get_trafo3w_results(net, mpc, trafo_loading, s_ft, i_ft, trafo_end, trafo3w_end, 
                             res_trafo3w_table=res_trafo3w_table)
        net[res_trafo3w_table].index = net["trafo3w"].index
        
    if impedance_end > trafo3w_end:
        _get_impedance_results(net, mpc, i_ft, trafo3w_end, impedance_end, 
                               res_impedance_table=res_impedance_table)
        net[res_impedance_table].index = net["impedance"].index
 
    if xward_end > impedance_end:
        _get_xward_branch_results(net, mpc, bus_lookup, pq_busses, impedance_end, xward_end, 
                                  res_xward_table=res_xward_table)
        

def _get_gen_results(net, mpc, gen_is, eg_is, bus_lookup, pq_bus, return_voltage_angles,
                     res_ext_grid_table="res_ext_grid", res_gen_table="res_gen"):
    eg_end = len(eg_is)
    gen_end = eg_end + len(gen_is)
    
    p = -mpc["gen"][:eg_end, PG] * 1e3
    q = -mpc["gen"][:eg_end, QG] * 1e3
    net[res_ext_grid_table]["p_kw"] = p
    net[res_ext_grid_table]["q_kvar"] = q
    net[res_ext_grid_table].index = eg_is.index
    b = eg_is["bus"].values

    if gen_end > eg_end:
        gidx = gen_is.bus.values
        b = np.hstack([b, gidx])

        p_gen = -mpc["gen"][eg_end:gen_end, PG] * 1e3
        q_gen = -mpc["gen"][eg_end:gen_end, QG] * 1e3
        p = np.hstack([p, p_gen])
        q = np.hstack([q, q_gen])
        
        gidx_mpc = get_indices(gidx, bus_lookup)
        net[res_gen_table]["p_kw"] = p_gen
        net[res_gen_table]["q_kvar"] = q_gen              
        net[res_gen_table]["vm_pu"] = mpc["bus"][gidx_mpc][:, VM]
        if return_voltage_angles:
            net[res_gen_table]["va_degree"] = mpc["bus"][gidx_mpc][:, VA]
        net[res_gen_table].index = gen_is.index

    b_sum, p_sum, q_sum = _sum_by_group(b, p, q)
    b = get_indices(b_sum, bus_lookup, fused_indices=False)
    pq_bus[b, 0] += p_sum
    pq_bus[b, 1] += q_sum

def _get_xward_branch_results(net, mpc, bus_lookup, pq_busses, f, t, res_xward_table="res_xward"):
    p_branch_xward = mpc["branch"][f:t, PF].real * 1e3
    q_branch_xward = mpc["branch"][f:t, QF].real * 1e3
    net[res_xward_table]["p_kw"] += p_branch_xward
    net[res_xward_table]["q_kvar"] += q_branch_xward
    b_pp, p, q = _sum_by_group(net["xward"]["bus"].values, p_branch_xward, q_branch_xward)
    b_mpc = get_indices(b_pp, bus_lookup, fused_indices=False)
    
    pq_busses[b_mpc, 0] += p
    pq_busses[b_mpc, 1] += q
    

def _get_branch_flows(net, mpc):
    br_idx = mpc["branch"][:, (F_BUS, T_BUS)].real.astype(int)
    u_ft = mpc["bus"][br_idx, 7] * mpc["bus"][br_idx, BASE_KV]
    s_ft = (np.sqrt(mpc["branch"][:, (PF, PT)].real**2 +
            mpc["branch"][:, (QF, QT)].real**2) * 1e3)
    i_ft = s_ft * 1e-3 / u_ft / np.sqrt(3)
    return i_ft, s_ft

def _get_line_results(net, mpc, i_ft, f, t, res_line_table="res_line"):
    pf_kw = mpc["branch"][f:t, PF].real * 1e3
    qf_kvar =  mpc["branch"][f:t, QF].real * 1e3
    net[res_line_table]["p_from_kw"] = pf_kw
    net[res_line_table]["q_from_kvar"] = qf_kvar
    
    pt_kw = mpc["branch"][f:t, PT].real * 1e3
    qt_kvar = mpc["branch"][f:t, QT].real * 1e3
    net[res_line_table]["p_to_kw"] = pt_kw
    net[res_line_table]["q_to_kvar"] = qt_kvar
    
    net[res_line_table]["pl_kw"] = pf_kw + pt_kw
    net[res_line_table]["ql_kvar"] = qf_kvar + qt_kvar
    
    i_ka = np.max(i_ft[f:t], axis=1)
    i_max = net["line"]["imax_ka"].values * net["line"]["df"].values * \
                                                                net["line"]["parallel"].values
    
    net[res_line_table]["i_ka"] = i_ka
    net[res_line_table]["loading_percent"] = i_ka / i_max * 100

def _get_trafo_results(net, mpc, trafo_loading, s_ft, i_ft, f, t, res_trafo_table="res_trafo"):
    phv_kw = mpc["branch"][f:t, PF].real * 1e3
    plv_kw = mpc["branch"][f:t, PT].real * 1e3
    net[res_trafo_table]["p_hv_kw"] = phv_kw
    net[res_trafo_table]["p_lv_kw"] = plv_kw
    net[res_trafo_table]["pl_kw"] = phv_kw + plv_kw
    
    qhv_kvar = mpc["branch"][f:t, QF].real * 1e3
    qlv_kvar = mpc["branch"][f:t, QT].real * 1e3
    net[res_trafo_table]["q_hv_kvar"] = qhv_kvar
    net[res_trafo_table]["q_lv_kvar"] = qlv_kvar
    net[res_trafo_table]["ql_kvar"] = qhv_kvar + qlv_kvar
    
    net[res_trafo_table]["i_hv_ka"] = i_ft[:,0][f:t]
    net[res_trafo_table]["i_lv_ka"] = i_ft[:,1][f:t]
    if trafo_loading == "current":
        lds_trafo = i_ft[f:t] * net["trafo"][["vn_hv_kv", "vn_lv_kv"]].values * 1000. * np.sqrt(3) \
                 / net["trafo"]["sn_kva"].values[:,np.newaxis] * 100.
        ld_trafo = np.max(lds_trafo, axis=1)
    elif trafo_loading == "power":
        ld_trafo = np.max(s_ft[f:t] / net["trafo"]["sn_kva"].values[:,np.newaxis] * 100., axis=1)
    else:
        raise ValueError("Unknown transformer loading parameter %s - choose 'current' or 'power'"%trafo_loading)
    net[res_trafo_table]["loading_percent"] = ld_trafo

def _get_trafo3w_results(net, mpc, trafo_loading, s_ft, i_ft, f, t, res_trafo3w_table="res_trafo3w"):
    hv = int(f + (t - f) / 3)
    mv = int(f + 2*(t - f) / 3)
    lv = t
    
    phv_kw = mpc["branch"][f:hv, PF].real * 1e3 
    pmv_kw = mpc["branch"][hv:mv, PT].real * 1e3
    plv_kw = mpc["branch"][mv:lv, PT].real * 1e3
    
    net[res_trafo3w_table]["p_hv_kw"] = phv_kw
    net[res_trafo3w_table]["p_mv_kw"] = pmv_kw
    net[res_trafo3w_table]["p_lv_kw"] = plv_kw
    net[res_trafo3w_table]["pl_kw"] = phv_kw + pmv_kw + plv_kw

    qhv_kvar = mpc["branch"][f:hv, QF].real * 1e3
    qmv_kvar = mpc["branch"][hv:mv, QT].real * 1e3 
    qlv_kvar = mpc["branch"][mv:lv, QT].real * 1e3
    net[res_trafo3w_table]["q_hv_kvar"] = qhv_kvar
    net[res_trafo3w_table]["q_mv_kvar"] = qmv_kvar 
    net[res_trafo3w_table]["q_lv_kvar"] = qlv_kvar
    net[res_trafo3w_table]["ql_kvar"] = qhv_kvar + qmv_kvar + qlv_kvar
    
    i_h = i_ft[:,0][f:hv]
    i_m = i_ft[:,1][hv:mv]
    i_l = i_ft[:,1][mv:lv]
    net[res_trafo3w_table]["i_hv_ka"] = i_h
    net[res_trafo3w_table]["i_mv_ka"] = i_m
    net[res_trafo3w_table]["i_lv_ka"] = i_l
    
    t3 = net["trafo3w"]
    if trafo_loading == "current":
        ld_h = i_h * t3["vn_hv_kv"].values * 1000. * np.sqrt(3) / t3["sn_hv_kva"].values * 100
        ld_m = i_m * t3["vn_mv_kv"].values * 1000. * np.sqrt(3) / t3["sn_mv_kva"].values * 100
        ld_l = i_l * t3["vn_lv_kv"].values * 1000. * np.sqrt(3) / t3["sn_lv_kva"].values * 100
        ld_trafo = np.max(np.vstack([ld_h, ld_m, ld_l]), axis=0)
    elif trafo_loading == "power":
        ld_h = s_ft[:,0][f:hv] / t3["sn_hv_kva"] * 100.
        ld_m = s_ft[:,1][hv:mv] / t3["sn_mv_kva"] * 100.
        ld_l = s_ft[:,1][mv:lv] / t3["sn_lv_kva"] * 100.
        ld_trafo = np.max(np.vstack([ld_h, ld_m, ld_l]), axis=0)
    else:
        raise ValueError("Unknown transformer loading parameter %s - choose 'current' or 'power'"%trafo_loading)
    net[res_trafo3w_table]["loading_percent"] = ld_trafo

def _get_impedance_results(net, mpc, i_ft, f, t, res_impedance_table="res_impedance"):
    pf_kw = mpc["branch"][f:t, (PF)].real * 1e3
    pt_kw = mpc["branch"][f:t, (PT)].real * 1e3    
    net[res_impedance_table]["p_from_kw"] = pf_kw
    net[res_impedance_table]["p_to_kw"] = pt_kw
    net[res_impedance_table]["pl_kw"] = pf_kw + pt_kw

    qf_kvar = mpc["branch"][f:t, (QF)].real * 1e3
    qt_kvar = mpc["branch"][f:t, (QT)].real * 1e3
    net[res_impedance_table]["q_from_kvar"] = qf_kvar 
    net[res_impedance_table]["q_to_kvar"] = qt_kvar
    net[res_impedance_table]["ql_kvar"] = qf_kvar + qt_kvar
    
    net[res_impedance_table]["i_from_ka"] = i_ft[f:t][:,0]
    net[res_impedance_table]["i_to_ka"] = i_ft[f:t][:,1]

def _get_p_q_results(net, mpc, bus_lookup, res_load_table="res_load", res_sgen_table="res_sgen",
                     res_ward_table="res_ward", res_xward_table="res_xward"):
    bus_pq = np.zeros(shape=(len(net["bus"].index), 2), dtype=np.float)
    b, p, q = np.array([]), np.array([]), np.array([])

    l = net["load"]
    if len(l) > 0:
        load_is = l["in_service"].values
        scaling = l["scaling"].values
        pl = l["p_kw"].values * scaling * load_is
        ql = l["q_kvar"].values * scaling * load_is
        net[res_load_table]["p_kw"] = pl
        net[res_load_table]["q_kvar"] = ql
        b = np.hstack([b, l["bus"].values])
        p = np.hstack([p, pl])
        q = np.hstack([q, ql])
        net[res_load_table].index = net["load"].index
    
    sg = net["sgen"]
    if len(sg) > 0:
        sgen_is = sg["in_service"].values
        scaling = sg["scaling"].values
        psg = sg["p_kw"].values * scaling * sgen_is
        qsg = sg["q_kvar"].values * scaling * sgen_is
        net[res_sgen_table]["p_kw"] = psg
        net[res_sgen_table]["q_kvar"] = qsg
        b = np.hstack([b, sg["bus"].values])
        p = np.hstack([p, psg])
        q = np.hstack([q, qsg])
        net[res_sgen_table].index = net["sgen"].index
        
    w = net["ward"]
    if len(w) > 0:
        ward_is = w["in_service"].values
        pw = w["ps_kw"].values * ward_is
        qw = w["qs_kvar"].values * ward_is
        net[res_ward_table]["p_kw"] = pw
        net[res_ward_table]["q_kvar"] = qw
        b = np.hstack([b, w["bus"].values])
        p = np.hstack([p, pw])
        q = np.hstack([q, qw])
    
    xw = net["xward"]
    if len(xw) > 0:
        xward_is = xw["in_service"].values
        pxw = xw["ps_kw"].values * xward_is
        qxw = xw["qs_kvar"].values * xward_is 
        net[res_xward_table]["p_kw"] = pxw
        net[res_xward_table]["q_kvar"] = qxw       
        b = np.hstack([b, xw["bus"].values])
        p = np.hstack([p, pxw])
        q = np.hstack([q, qxw])
        
    b_pp, vp, vq = _sum_by_group(b.astype(int), p, q)
    b_mpc = get_indices(b_pp, bus_lookup, fused_indices=False)
    bus_pq[b_mpc, 0] = vp
    bus_pq[b_mpc, 1] = vq
    return bus_pq

def _get_shunt_results(net, mpc, bus_lookup, bus_pq, res_shunt_table="res_shunt", 
                       res_ward_table="res_ward", res_xward_table="res_xward"):
    b, p, q = np.array([]), np.array([]), np.array([])
    s = net["shunt"]
    if len(s) > 0:    
        sidx = get_indices(s["bus"], bus_lookup)
        shunt_is = s["in_service"].values
        u_shunt = mpc["bus"][sidx, VM]
        net[res_shunt_table]["vm_pu"] = u_shunt
        p_shunt = u_shunt**2 * net["shunt"]["p_kw"].values * shunt_is
        q_shunt = u_shunt**2 * net["shunt"]["q_kvar"].values * shunt_is
        net[res_shunt_table]["p_kw"] = p_shunt
        net[res_shunt_table]["q_kvar"] = q_shunt
        b = np.hstack([b, s["bus"].values])
        p = np.hstack([p, p_shunt])
        q = np.hstack([q, q_shunt])
        net[res_shunt_table].index = net["shunt"].index

          
    w = net["ward"]
    if len(w) > 0:
        widx = get_indices(w["bus"], bus_lookup)
        ward_is = w["in_service"].values
        u_ward = mpc["bus"][widx, VM]
        net[res_ward_table]["vm_pu"] = u_ward
        p_ward = u_ward**2 * net["ward"]["pz_kw"].values * ward_is
        q_ward = u_ward**2 * net["ward"]["qz_kvar"].values * ward_is
        net[res_ward_table]["p_kw"] += p_ward
        net[res_ward_table]["q_kvar"] += q_ward
        b = np.hstack([b, w["bus"].values])
        p = np.hstack([p, p_ward])
        q = np.hstack([q, q_ward])
        net[res_ward_table].index = net["ward"].index

    xw = net["xward"]
    if len(xw) > 0:
        widx = get_indices(xw["bus"], bus_lookup)
        xward_is = xw["in_service"].values
        u_xward = mpc["bus"][widx, VM]
        net[res_xward_table]["vm_pu"] = u_xward
        p_xward = u_xward**2 * net["xward"]["pz_kw"].values * xward_is
        q_xward = u_xward**2 * net["xward"]["qz_kvar"].values * xward_is
        net[res_xward_table]["p_kw"] += p_xward
        net[res_xward_table]["q_kvar"] += q_xward
        b = np.hstack([b, xw["bus"].values])
        p = np.hstack([p, p_xward])
        q = np.hstack([q, q_xward])
        net[res_xward_table].index = net["xward"].index

    b_pp, vp, vq = _sum_by_group(b.astype(int), p, q)
    b_mpc = get_indices(b_pp, bus_lookup, fused_indices=False)

    bus_pq[b_mpc, 0] += vp
    bus_pq[b_mpc, 1] += vq
