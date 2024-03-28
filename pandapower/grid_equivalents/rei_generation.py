from pandas.api.types import is_bool_dtype, is_numeric_dtype #, is_integer_dtype
import pandapower as pp
from pandapower.grid_equivalents.auxiliary import calc_zpbn_parameters, \
    drop_internal_branch_elements, \
    build_ppc_and_Ybus, drop_measurements_and_controllers, \
    drop_and_edit_cost_functions, _runpp_except_voltage_angles, \
        replace_motor_by_load
from pandapower.grid_equivalents.toolbox import get_connected_switch_buses_groups
from copy import deepcopy
import pandas as pd
import numpy as np
import operator
import time
import uuid
import re
from functools import reduce
try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def _calculate_equivalent_Ybus(net_zpbn, bus_lookups, eq_type,
                               show_computing_time=False, **kwargs):
    """
    The function orders the admittance matrix of the original network into
    new format firstly, which is convenient for rei equivalent calculation.d
    Then it calculates the equivalent admittance matrix of the given network

    i: internal   b: boundary    e: external    g: ground    t: total

    Ymat_trans =

    [ Ybus_ii, Ybus_ib,     0   ]    [Ybus_ii, Ybus_ib,      0   ,    0   ,    0      ]
    [ Ybus_bi, Ybus_bb, Ybus_be ] =             _________________   _________________
    [   0   ,  Ybus_eb, Ybus_ee ]    [Ybus_bi, |Ybus_bb ,   0   |, |   0   , Ybus_be| ]
                                               |                |  |                |
                                     [   0   , |   0    ,Ybus_tt|, |Ybus_tg,    0   | ]
                                               |________________|  |________________|
                                                _________________   _________________
                                     [   0   , |   0    ,Ybus_gt|, |Ybus_gg, Ybus_ge| ]
                                     [   0   , |Ybus_eb ,    0  |, |Ybus_eg, Ybus_ee| ]
                                               |________________|  |________________|

    INPUT:
        **net_zpbn** -  zero power balance network (pandapower network)

        **bus_lookups** (dict) -  bus lookups

        **eq_type** (str) -  the equavalten type

    OPTIONAL:
        **check_validity** (bool, False) - TODO

    OUTPUT:
        **Ybus** - equivalent admittance matrix of the external network

    """
    t_start = time.perf_counter()
    # --- initialization
    Ybus_origin = net_zpbn._ppc["internal"]["Ybus"].todense()
    Ybus_sorted = net_zpbn._ppc["internal"]["Ybus"].todense()
    bus_lookup_ppc = bus_lookups["bus_lookup_ppc"]
    nb_dict = {}
    for key in bus_lookup_ppc.keys():
        if key != "b_area_buses_no_switch":
            nb_dict["nb_"+key.split("_")[0]] = len(bus_lookup_ppc[key])
    Ybus_buses = list(bus_lookup_ppc.values())
    Ybus_new_sequence = reduce(operator.concat, Ybus_buses)

    # --- transform Ybus_origin to Ybus_new according to the Ybus_new_sequence
    for i in range(len(Ybus_new_sequence)):
        for j in range(len(Ybus_new_sequence)):
            # --- if xward, put very large admittance at the diagonals (PV-bus) of Ybus
            if eq_type == "xward" and i >= nb_dict["nb_i"]+nb_dict["nb_b"] and \
                    i == j and Ybus_new_sequence[i] in net_zpbn._ppc["gen"][:, 0]:
                Ybus_sorted[i, j] = 1e8
            else:
                Ybus_sorted[i, j] = Ybus_origin[Ybus_new_sequence[i], Ybus_new_sequence[j]]

    # --- calculate calculate equivalent Ybus and equivalent Ybus without_internals
    Ybus_bb = Ybus_sorted[nb_dict["nb_i"]:(nb_dict["nb_i"] + nb_dict["nb_b"] + nb_dict["nb_t"]),
                          nb_dict["nb_i"]:(nb_dict["nb_i"] + nb_dict["nb_b"] + nb_dict["nb_t"])]
    Ybus_ee = Ybus_sorted[-(nb_dict["nb_e"] + nb_dict["nb_g"]):,
                          -(nb_dict["nb_e"] + nb_dict["nb_g"]):]
    Ybus_eb = Ybus_sorted[-(nb_dict["nb_e"] + nb_dict["nb_g"]):,
                          nb_dict["nb_i"]:(nb_dict["nb_i"] + nb_dict["nb_b"] + nb_dict["nb_t"])]
    Ybus_be = Ybus_eb.T

    try:
        inverse_Ybus_ee = np.linalg.inv(Ybus_ee)
    except np.linalg.LinAlgError as err:
        if 'Singular matrix' in str(err):
            logger.debug("Ymat_ee is a singular martix, now try to compute the \
                         pseudo-inverse of the matrix.")
            inverse_Ybus_ee = np.linalg.pinv(Ybus_ee)
    Ybus_eq_boundary = Ybus_bb - (Ybus_be * inverse_Ybus_ee * Ybus_eb)
    Ybus_eq = np.copy(Ybus_sorted[0: nb_dict["nb_i"] + nb_dict["nb_b"] + nb_dict["nb_t"],
                                  0: nb_dict["nb_i"] + nb_dict["nb_b"] + nb_dict["nb_t"]])
    Ybus_eq[-(nb_dict["nb_b"] + nb_dict["nb_t"]):, -(nb_dict["nb_b"] +
                                                     nb_dict["nb_t"]):] = Ybus_eq_boundary

    t_end = time.perf_counter()
    if show_computing_time:
        logger.info("\"calculate_equivalent_Ybus\" finished in %s seconds:" % round((
            t_end-t_start), 2))
    return Ybus_eq


def adapt_impedance_params(Z, sign=1, adaption=1e-15):
    """
    In some extreme cases, the created admittance matrix of the
    zpbn network is singular. The routine is unsolvalbe with it.
    In response, an impedance adaption is created and added.
    """
    rft_pu = Z.real + sign*adaption
    xft_pu = Z.imag + sign*adaption
    return rft_pu, xft_pu


def _create_net_zpbn(net, boundary_buses, all_internal_buses, all_external_buses,
                     load_separate=False, sgen_separate=True, gen_separate=True,
                     show_computing_time=False, calc_volt_angles=True,
                     runpp_fct=_runpp_except_voltage_angles, **kwargs):
    """
    The function builds the zero power balance network with
    calculated impedance and voltage

    INPUT:
        **net** - pandapower network

        **boundary_buses** (list) - boundary buses

        **all_internal_buses** - all the internal buses

        **all_external_buses** - all the external buses

    OPTIONAL:
        **load_separate** (bool, False) - flag if all the loads
            are reserved integrally

        **sgen_separate** (bool, True) - flag if all the DER are
            reserved separately

        **gen_separate** (bool, True) - flag if all the gens are
            reserved separately

        **tolerance_mva** (float, 1e-3) - loadflow termination
            condition referring to P / Q mismatch of node power
            in MVA. The loalflow hier is to get the admittance
            matrix of the zpbn network

    OUTPUT:
        **net_zpbn** - zero power balance networks
    """

    net_internal, net_external = _get_internal_and_external_nets(
            net, boundary_buses, all_internal_buses, all_external_buses,
            show_computing_time, calc_volt_angles=calc_volt_angles, runpp_fct=runpp_fct, **kwargs)
    net_zpbn = net_external
    # --- remove buses without power flow results in net_eq
    pp.drop_buses(net_zpbn, net_zpbn.res_bus.index[net_zpbn.res_bus.vm_pu.isnull()])

    Z, S, v, limits = calc_zpbn_parameters(net_zpbn, boundary_buses, all_external_buses)
    # --- remove the original load, sgen and gen in exteranl area,
    #     and creat new buses and impedance
    t_buses, g_buses = [], []
    sn_mva = net_zpbn.sn_mva
    for elm, separate in [("load", load_separate), ("sgen", sgen_separate), ("gen", gen_separate), ("ext_grid", False)]:
        # in Z columns only gen, load and sgens are considered, so we can leave out ext_grid
        net_zpbn[elm].drop(net_zpbn[elm].index[net_zpbn[elm].bus.isin(all_external_buses)], inplace=True)
        if elm == "ext_grid":
            continue

        if not np.isnan(Z[elm+"_ground"].values).all():
            if separate:
                Z = Z.drop([elm+"_integrated_total"], axis=1)

                # add buses
                idxs = Z.index[~np.isnan(Z[elm+"_ground"].values)]
                vn_kvs = net_zpbn.bus.vn_kv[Z.ext_bus.loc[idxs]]
                new_g_buses = pp.create_buses(net_zpbn, len(idxs), vn_kvs, name=[
                    "%s_separate-ground %s" % (elm, str(Z.ext_bus.loc[i])) for i in idxs])
                new_t_buses = pp.create_buses(net_zpbn, len(idxs), vn_kvs, name=[
                    "%s_separate-total %s" % (elm, str(Z.ext_bus.loc[i])) for i in idxs],
                    max_vm_pu = limits.max_vm_pu.loc[idxs], min_vm_pu=limits.min_vm_pu.loc[idxs])

                # add impedances
                rft_pu_g, xft_pu_g = adapt_impedance_params(Z[elm+"_ground"].loc[idxs].values)
                max_idx = net_zpbn.impedance.index.max() if net_zpbn.impedance.shape[0] else 0
                new_imps_g = pd.DataFrame({
                    "from_bus": Z.ext_bus.loc[idxs].astype(np.int64).values, "to_bus": new_g_buses,
                    "rft_pu": rft_pu_g, "xft_pu": xft_pu_g,
                    "rtf_pu": rft_pu_g, "xtf_pu": xft_pu_g},
                    index=range(max_idx+1, max_idx+1+len(new_g_buses)))
                new_imps_g["name"] = "eq_impedance_ext_to_ground"
                new_imps_g["sn_mva"] = sn_mva
                new_imps_g["in_service"] = True

                rft_pu_t, xft_pu_t = adapt_impedance_params(Z[elm+"_separate_total"].loc[
                    idxs].values)
                new_imps_t = pd.DataFrame({
                    "from_bus": new_g_buses, "to_bus": new_t_buses,
                    "rft_pu": rft_pu_t, "xft_pu": xft_pu_t,
                    "rtf_pu": rft_pu_t, "xtf_pu": xft_pu_t},
                    index=range(new_imps_g.index.max()+1,
                                new_imps_g.index.max()+1+len(new_g_buses)))
                new_imps_t["name"] = "eq_impedance_ground_to_total"
                new_imps_t["sn_mva"] = sn_mva
                new_imps_t["in_service"] = True

                net_zpbn["impedance"] = pd.concat([net_zpbn["impedance"], new_imps_g, new_imps_t])
                g_buses += list(new_g_buses)
                t_buses += list(new_t_buses)
            else:
                Z = Z.drop([elm+"_separate_total"], axis=1)
                vn_kv = net_zpbn.bus.vn_kv[all_external_buses].values[0]
                new_g_bus = pp.create_bus(net_zpbn, vn_kv, name=elm+"_integrated-ground ")
                i_all_integrated = []
                for i in Z.index[~np.isnan(Z[elm+"_ground"].values)]:
                    rft_pu, xft_pu = adapt_impedance_params(Z[elm+"_ground"][i])
                    pp.create_impedance(net_zpbn, Z.ext_bus[i], new_g_bus, rft_pu, xft_pu,
                                        sn_mva,name="eq_impedance_ext_to_ground")
                    i_all_integrated.append(i)
                # in case of integrated, the tightest vm limits are assumed
                ext_buses = Z.ext_bus[~np.isnan(Z[elm+"_ground"])].values
                ext_buses_name = "/".join([str(eb) for eb in ext_buses])
                new_t_bus = pp.create_bus(
                    net_zpbn, vn_kv, name=elm+"_integrated-total "+ext_buses_name,
                    max_vm_pu=limits.max_vm_pu.loc[i_all_integrated].min(),
                    min_vm_pu=limits.min_vm_pu.loc[i_all_integrated].max())
                rft_pu, xft_pu = adapt_impedance_params(Z[elm+"_integrated_total"][0])
                pp.create_impedance(net_zpbn, new_g_bus, new_t_bus, rft_pu, xft_pu,
                                    sn_mva, name="eq_impedance_ground_to_total")
                g_buses += [new_g_bus.tolist()]
                t_buses += [new_t_bus.tolist()]
        else:
            Z.drop([elm+"_ground", elm+"_separate_total", elm+"_integrated_total"], axis=1,
                   inplace=True)

    # --- create load, sgen and gen
    elm_old = None
    max_load_idx = max(-1, net.load.index[~net.load.bus.isin(all_external_buses)].max() - len(net_zpbn.load))
    max_sgen_idx = max(-1, net.sgen.index[~net.sgen.bus.isin(all_external_buses)].max() - len(net_zpbn.sgen))
    max_gen_idx = max(-1, net.gen.index[~net.gen.bus.isin(all_external_buses)].max() - len(net_zpbn.gen))
    for i in t_buses:
        busstr = net_zpbn.bus.name[i].split(" ")[1]
        bus = int(busstr.split("/")[0])
        key = net_zpbn.bus.name[i].split("-")[0]
        elm = net_zpbn.bus.name[i].split("_")[0]
        idx = S.index[S.ext_bus == bus].values[0]
        P = S[key][idx].real * sn_mva
        Q = S[key][idx].imag * sn_mva
        Sn = S["sn_"+key][idx].real
        if elm == "load":
            elm_idx = pp.create_load(net_zpbn, i, -float(P), -float(Q), name=key+"_rei_"+busstr,
                                     sn_mva=Sn, index=max_load_idx+len(net_zpbn.load)+1)
        elif elm == "sgen":
            elm_idx = pp.create_sgen(net_zpbn, i, float(P), float(Q), name=key+"_rei_"+busstr,
                                     sn_mva=Sn, index=max_sgen_idx+len(net_zpbn.sgen)+1)
        elif elm == "gen":
            vm_pu = v[key+"_vm_total"][v.ext_bus == int(re.findall('\d+', busstr)[0])].values.real
            elm_idx = pp.create_gen(net_zpbn, i, float(P), float(vm_pu), name=key+"_rei_"+busstr,
                                    sn_mva=Sn, index=max_gen_idx+len(net_zpbn.gen)+1)

    # ---- match other columns
        elm_org = net[elm]
        if elm_old is None or elm_old != elm:
            other_cols = set(elm_org.columns) - \
                         {"name", "bus", "p_mw", "q_mvar", "sn_mva", "in_service", "scaling"}
            other_cols_bool = set(
                net[elm][list(other_cols)].columns[net[elm][list(other_cols)].apply(is_bool_dtype)])
            other_cols -= other_cols_bool
            other_cols_number = set(
                net[elm][list(other_cols)].columns[
                    net[elm][list(other_cols)].apply(is_numeric_dtype)])
            other_cols -= other_cols_number
            other_cols_str = set()
            other_cols_none = set()
            other_cols_mixed = set()
            for c in other_cols.copy():
                value_types = net[elm][c][elm_org.bus.isin(all_external_buses)].apply(type).unique()
                if len(value_types) > 1:
                    other_cols_mixed |= {c}
                elif value_types[0] in (float, int):
                    other_cols_number |= {c}
                elif value_types[0] == bool:
                    other_cols_bool |= {c}
                elif value_types[0] == str:
                    other_cols_str |= {c}
                else: # value_types[0] is None:
                    other_cols_none |= {c}
                other_cols -= {c}
            assert len(other_cols) == 0
        if "integrated" in key:
            if "voltLvl" in other_cols_number:
                net_zpbn[elm].loc[elm_idx, "voltLvl"] = \
                    net_zpbn.bus.voltLvl[boundary_buses].max()
                other_cols_number -= {"voltLvl"}
            net_zpbn[elm].loc[elm_idx, list(other_cols_number)] = \
                elm_org[list(other_cols_number)][elm_org.bus.isin(all_external_buses)].sum(axis=0)
            net_zpbn[elm].loc[elm_idx, list(other_cols_bool)] = elm_org[list(other_cols_bool)][
                elm_org.bus.isin(all_external_buses)].values.sum(axis=0) > 0
            all_str_values = list(zip(*elm_org[list(other_cols_str)]\
                                      [elm_org.bus.isin(all_external_buses)].values[::-1]))
            for asv, colid in zip(all_str_values, other_cols_str):
                if len(set(asv)) == 1:
                    net_zpbn[elm].loc[elm_idx, colid] = asv[0]
                else:
                    net_zpbn[elm].loc[elm_idx, colid] = "//".join(asv)
            net_zpbn[elm].loc[elm_idx, list(other_cols_none)] = None
            for ocm in other_cols_mixed:
                net_zpbn[elm][ocm] = net_zpbn[elm][ocm].astype("object")
            net_zpbn[elm].loc[elm_idx, list(other_cols_mixed)] = "mixed data type"
        else:
            if elm == "gen" and bus in net.ext_grid.bus.values and \
                    net.ext_grid.in_service[net.ext_grid.bus == bus].values[0]:
                net_zpbn[elm].name[elm_idx] = str(net.ext_grid.name[
                    net.ext_grid.bus == bus].values[0]) + "-" + net_zpbn[elm].name[elm_idx]
                ext_grid_cols = list(set(elm_org.columns) & set(net.ext_grid.columns) - \
                    {"name", "bus", "p_mw", "sn_mva", "in_service", "scaling"})
                net_zpbn[elm].loc[elm_idx, ext_grid_cols] = net.ext_grid[ext_grid_cols][
                    net.ext_grid.bus == bus].values[0]
            else:
                names = elm_org.name[elm_org.bus == bus].values
                names = [str(n) for n in names]
                net_zpbn[elm].name[elm_idx] = "//".join(names) + "-" + net_zpbn[elm].name[elm_idx]
                if len(names) > 1:
                    net_zpbn[elm].loc[elm_idx, list(other_cols_number)] = \
                        elm_org[list(other_cols_number)][elm_org.bus == bus].sum(axis=0)
                    if "voltLvl" in other_cols_number:
                        net_zpbn[elm].loc[elm_idx, "voltLvl"] = \
                            net_zpbn.bus.voltLvl[boundary_buses].max()
                    net_zpbn[elm].loc[elm_idx, list(other_cols_bool)] = \
                        elm_org[list(other_cols_bool)][elm_org.bus == bus].values.sum(axis=0) > 0

                    all_str_values = list(zip(*elm_org[list(other_cols_str)]\
                                              [elm_org.bus == bus].values[::-1]))
                    for asv, colid in zip(all_str_values, other_cols_str):
                        if len(set(asv)) == 1:
                            net_zpbn[elm].loc[elm_idx, colid] = asv[0]
                        else:
                            net_zpbn[elm].loc[elm_idx, colid] = "//".join(asv)
                    net_zpbn[elm].loc[elm_idx, list(other_cols_none)] = None
                    for ocm in other_cols_mixed:
                        net_zpbn[elm][ocm] = net_zpbn[elm][ocm].astype("object")
                    net_zpbn[elm].loc[elm_idx, list(other_cols_mixed)] = "mixed data type"
                else:
                    net_zpbn[elm].loc[elm_idx, list(other_cols_bool | other_cols_number |
                                                    other_cols_str | other_cols_none)] = \
                        elm_org[list(other_cols_bool | other_cols_number |
                                     other_cols_str | other_cols_none)][
                            elm_org.bus == bus].values[0]
                    net_zpbn[elm].loc[elm_idx, list(other_cols)] = elm_org[list(other_cols)][
                        elm_org.bus == bus].values[0]
        elm_old = net_zpbn.bus.name[i].split("_")[0]

    # --- match poly_cost to new created elements
    for cost_elm in ["poly_cost", "pwl_cost"]:
        if len(net[cost_elm]):
            df = net_zpbn[cost_elm].copy()
            df.et[(df.et == "ext_grid") &
                  (~df.bus.isin(boundary_buses))] = "gen"
            df.et[(df.et.isin(["storage", "dcline"]) &
                             (~df.bus.isin(boundary_buses)))] = "load"

            logger.debug("During the equivalencing, also in polt_cost, " +
                         "storages and dclines are treated as loads, and" +
                         "ext_grids are treated as gens ")

            for elm in ["load", "gen", "sgen"]:
                for idx in net_zpbn[elm].index:
                    if net_zpbn[elm].bus[idx] in boundary_buses:
                        continue
                    else:
                        pc_idx = df.index[df.et == elm]
                        if net_zpbn[elm].name.str.contains("integrated").any() and len(pc_idx):
                            logger.debug("Attention! After equivalencing, " + elm + "s are modeled as " +
                                         "an aggregated " + elm + ". The " + cost_elm + " data of the first " +
                                         "original " + elm + " is used as the " + cost_elm + " data of the " +
                                         "aggregated " + elm + ". It is NOT correct at present.")
                            df.element[pc_idx[0]] = net_zpbn[elm].index[net_zpbn[elm].name.str.contains(
                                "integrated", na=False)][0]
                            df = df.drop(pc_idx[1:])
                        elif len(pc_idx):
                            related_bus = int(str(net_zpbn[elm].name[idx]).split("_")[-1])
                            pc_idx = df.index[(df.bus == related_bus) &
                                              (df.et == elm)]
                            if len(pc_idx) > 1:
                                logger.debug("Attention! There are at least two " + elm + "s connected to a " +
                                             "common bus. The " + elm + "s with commen bus are modeled as an " +
                                             "aggreated " + elm + " during the equivalencing. " +
                                             "The " + cost_elm + " data of the first " + elm + " is used as the " +
                                             cost_elm + " data of the aggregated " + elm + ". " +
                                             "It is NOT correct at present.")
                                pc_idx = df.index[(df.bus == related_bus) &
                                                  (df.et == elm)]
                                df.element[pc_idx[0]] = idx
                                df = df.drop(pc_idx[1:])
                            elif len(pc_idx) == 1:
                                df.element[pc_idx[0]] = idx
            net_zpbn[cost_elm] = df

    drop_and_edit_cost_functions(net_zpbn, [], False, True, False)
    # pp.runpp(net_zpbn)
    runpp_fct(net_zpbn, calculate_voltage_angles=calc_volt_angles,
              tolerance_mva=1e-3, max_iteration=100, **kwargs)
    return net_zpbn, net_internal, net_external


def _create_bus_lookups(net_zpbn, boundary_buses, all_internal_buses,
                        all_external_buses,
                        boundary_buses_inclusive_bswitch,
                        show_computing_time=False):
    """
    The function creates a bus lookup table according to the given zpbn network
    and the bus groups
    """
    t_start = time.perf_counter()
    build_ppc_and_Ybus(net_zpbn)
    ppc_branch = net_zpbn._ppc["branch"]
    pd2ppc_bus_lookup = net_zpbn._pd2ppc_lookups["bus"]

    # --- create bus lookup
    bus_lookup_pd = {"i_area_buses": [],
                     "b_area_buses": boundary_buses,
                     "t_area_buses": net_zpbn.bus.index[net_zpbn.bus.name.str.contains(
                        "-total", na=False)].tolist(),
                     "g_area_buses": net_zpbn.bus.index[net_zpbn.bus.name.str.contains(
                        "-ground", na=False)].tolist(),
                     "e_area_buses": all_external_buses,
                     "b_area_buses_no_switch": []}

    # --- create ppc bus lookup
    bus_lookup_ppc = bus_lookup_pd.copy()
    bus_lookup_ppc_origin = bus_lookup_pd.copy()
    for key in ["b", "i", "e", "g", "t"]:
        bus_lookup_ppc[key+"_area_buses"] = pd2ppc_bus_lookup[list(bus_lookup_pd[
            key+"_area_buses"])].tolist()
        origin_sequence = bus_lookup_ppc[key+"_area_buses"].copy()
        # remove repeated "neg." (e.g. bus-bus-switch) ppc buses
        bus_lookup_ppc[key+"_area_buses"] = sorted(set(bus_lookup_ppc[
            key+"_area_buses"]), key=origin_sequence.index)
        if key == "b":
            # bus_lookup_pd["b_area_buses_no_switch"] = bus_lookup_pd[key+"_area_buses"].copy()
            # for i in range(len(bus_lookup_ppc[key+"_area_buses"])):
            #     if bus_lookup_ppc[key+"_area_buses"][i] != \
            #         pd2ppc_bus_lookup[bus_lookup_pd["b_area_buses_no_switch"][i]]:
            #             del bus_lookup_pd["b_area_buses_no_switch"][i]

            bus_lookup_pd["b_area_buses_no_switch"] = bus_lookup_pd[key+"_area_buses"].copy()
            val_ppc = []
            for val in bus_lookup_pd[key+"_area_buses"].copy():
                if pd2ppc_bus_lookup[val] in val_ppc:
                    bus_lookup_pd["b_area_buses_no_switch"].remove(val)
                else:
                    val_ppc.append(pd2ppc_bus_lookup[val])

        # remove ppc buses appearing in b_area as well as in i_area or e_area
        if key != "b":
            common_ppc_busese = set(bus_lookup_ppc["b_area_buses"]) & set(bus_lookup_ppc[
                key+"_area_buses"])
            if common_ppc_busese:
                bus_lookup_ppc[key+"_area_buses"].remove(list(common_ppc_busese))
        if key == "b" and len(bus_lookup_ppc["b_area_buses"]) != len(bus_lookup_pd["b_area_buses"]):
            logger.info("some boundary buses are connetected via switches")

    # --- identify "pos." (eg. bus-line-switch) ppc_buses
    Ybus_size = net_zpbn._ppc["internal"]["Ybus"]._shape[0]
    all_ppc_buses_lists = list(bus_lookup_ppc.values())
    all_ppc_buses = reduce(operator.concat, all_ppc_buses_lists)
    pos_aux_buses_ppc = set(np.arange(Ybus_size))-set(all_ppc_buses)

    # --- identify the "pos." (eg. bus-line-switch) ppc buses belongs zu which bus group
    if pos_aux_buses_ppc:
        for br_idx in range(ppc_branch.shape[0]):
            f_bus_ppc = ppc_branch[br_idx, 0].real
            t_bus_ppc = ppc_branch[br_idx, 1].real
            if f_bus_ppc in pos_aux_buses_ppc:
                if t_bus_ppc in bus_lookup_ppc["e_area_buses"]:
                    bus_lookup_ppc["e_area_buses"].append(int(f_bus_ppc))
                    pos_aux_buses_ppc.remove(f_bus_ppc)
                elif t_bus_ppc in bus_lookup_ppc["i_area_buses"]:
                    bus_lookup_ppc["i_area_buses"].append(f_bus_ppc)
                    pos_aux_buses_ppc.remove(f_bus_ppc)
            elif t_bus_ppc in pos_aux_buses_ppc:
                if f_bus_ppc in bus_lookup_ppc["e_area_buses"]:
                    bus_lookup_ppc["e_area_buses"].append(int(t_bus_ppc))
                    pos_aux_buses_ppc.remove(t_bus_ppc)
                elif f_bus_ppc in bus_lookup_ppc["i_area_buses"]:
                    bus_lookup_ppc["i_area_buses"].append(int(t_bus_ppc))
                    pos_aux_buses_ppc.remove(t_bus_ppc)
        bus_lookup_ppc["e_area_buses"] += pos_aux_buses_ppc

    bus_lookups = ({"bus_lookup_pd": bus_lookup_pd,
                    "bus_lookup_ppc": bus_lookup_ppc,
                    "bus_lookup_ppc_origin": bus_lookup_ppc_origin,
                    "pos_aux_bus_ppc": pos_aux_buses_ppc,
                    "boundary_buses_inclusive_bswitch":
                        boundary_buses_inclusive_bswitch,
                    "origin_all_internal_buses": all_internal_buses})
    t_end = time.perf_counter()
    if show_computing_time:
        logger.info("\"create_bus_lookup\" finished in %s seconds:" % round((t_end-t_start), 2))
    return bus_lookups


def _get_internal_and_external_nets(net, boundary_buses, all_internal_buses,
                                    all_external_buses, show_computing_time=False,
                                    calc_volt_angles=True,
                                    runpp_fct=_runpp_except_voltage_angles, **kwargs):
    "This function identifies the internal area and the external area"
    t_start = time.perf_counter()
    if not all_internal_buses:
        net_internal = None
    else:
        net_internal = deepcopy(net)
        drop_measurements_and_controllers(net_internal, all_external_buses, True)
        drop_and_edit_cost_functions(net_internal,
                                     all_external_buses+boundary_buses,
                                     True, True)
        pp.drop_buses(net_internal, all_external_buses)

    net_external = deepcopy(net)
    if "group" in net_external:
        net_external.group = net_external.group.drop(net_external.group.index)
    drop_and_edit_cost_functions(net_external, all_internal_buses,
                                 True, True)
    drop_measurements_and_controllers(net_external, net_external.bus.index.tolist())
    pp.drop_buses(net_external, all_internal_buses)
    replace_motor_by_load(net_external, all_external_buses)
#    add_ext_grids_to_boundaries(net_external, boundary_buses, runpp_fct=runpp_fct, **kwargs)
#    runpp_fct(net_external, calculate_voltage_angles=calc_volt_angles, **kwargs)
    _integrate_power_elements_connected_with_switch_buses(net, net_external,
                                                          all_external_buses) # for sgens, gens, and loads
    runpp_fct(net_external, calculate_voltage_angles=calc_volt_angles, **kwargs)
    t_end = time.perf_counter()
    if show_computing_time:
        logger.info("\"get_int_and_ext_nets\" " +
                    "finished in %s seconds:" % round((t_end-t_start), 2))

    return net_internal, net_external


def _calclate_equivalent_element_params(net_zpbn, Ybus_eq, bus_lookups,
                                        show_computing_time=False,
                                        max_allowed_impedance=1e8, **kwargs):
    """ This function calculates the equivalent parameters

    INPUT:
      **Ybus_eq** (array) - equivalent admittance matrix of the external area

      **bus_lookup** (dict) - bus lookup table

    OUTPUT:
      **shunt_params** - parameters of the equivalent shunts

      **impedance_params** - parameters of the equivalent impedances

    """
    t_start = time.perf_counter()
    # --- calculate impedance paramter
    bt_buses_ppc = list(bus_lookups["bus_lookup_ppc"]["b_area_buses"]) + \
        list(bus_lookups["bus_lookup_ppc"]["t_area_buses"])
    bt_buses_pd = list(bus_lookups["bus_lookup_pd"]["b_area_buses_no_switch"]) + \
        list(bus_lookups["bus_lookup_pd"]["t_area_buses"])
    nb_bt_buses_ppc = len(bt_buses_ppc)

    shunt_params = pd.DataFrame(columns=["bus_pd", "bus_ppc", "parameter"])
    shunt_params["bus_ppc"] = bt_buses_ppc
    shunt_params["bus_pd"] = bt_buses_pd
    shunt_params["parameter"] = Ybus_eq.sum(axis=1)[-nb_bt_buses_ppc:]
    shunt_params["local_voltage"] = net_zpbn.res_bus.vm_pu[bt_buses_pd].values

    # --- calculate impedance paramter
    params = Ybus_eq[-nb_bt_buses_ppc:, -nb_bt_buses_ppc:]
    nl = (nb_bt_buses_ppc) * (nb_bt_buses_ppc - 1) // 2
    tri_upper = np.triu(params, k=1)
    non_zero = np.abs(tri_upper) > 1/max_allowed_impedance
    rows = (np.arange(params.shape[0]).reshape(-1, 1) * np.ones(params.shape)).astype(np.int64)[non_zero]
    cols = (np.arange(params.shape[1]) * np.ones(params.shape)).astype(np.int64)[non_zero]

    impedance_params = pd.DataFrame(columns=["from_bus", "to_bus", "rft_pu",
                                             "xft_pu", "rtf_pu", "xtf_pu"], index=range(len(rows)))
    impedance_params["from_bus"] = np.array(bt_buses_pd)[rows]
    impedance_params["to_bus"] = np.array(bt_buses_pd)[cols]
    impedance_params["rft_pu"] = (-1 / params[rows, cols]).real
    impedance_params["xft_pu"] = (-1 / params[rows, cols]).imag

    non_zero_cr = np.abs(params[cols, rows]) > 1/max_allowed_impedance
    impedance_params["rtf_pu"] = 1e5
    impedance_params["xtf_pu"] = 1e5
    impedance_params.loc[non_zero_cr, "rtf_pu"] = (-1 / params[cols, rows]).real[non_zero_cr]
    impedance_params.loc[non_zero_cr, "xtf_pu"] = (-1 / params[cols, rows]).imag[non_zero_cr]

    t_end = time.perf_counter()
    if show_computing_time:
        logger.info("\"calclate_equivalent_element_params\" finished in %s seconds:" %
                    round((t_end-t_start), 2))
    return shunt_params, impedance_params


def _replace_ext_area_by_impedances_and_shunts(
        net_eq, bus_lookups, impedance_params, shunt_params, net_internal,
        return_internal, show_computing_time=False, calc_volt_angles=True, imp_threshold=1e-8,
        runpp_fct=_runpp_except_voltage_angles, **kwargs):
    """
    This function implements the parameters of the equivalent shunts and equivalent impedance
    """
    # --- drop all external elements
    eg_buses_pd = bus_lookups["bus_lookup_pd"]["e_area_buses"] + \
        bus_lookups["bus_lookup_pd"]["g_area_buses"]
    pp.drop_buses(net_eq, eg_buses_pd)

    try:
        runpp_fct(net_eq, calculate_voltage_angles=calc_volt_angles,
                                     tolerance_mva=1e-6, max_iteration=100, **kwargs)
    except:
        logger.error("The power flow did not converge.")

    # --- drop all branch elements except switches between boundary buses
    drop_internal_branch_elements(net_eq, bus_lookups["boundary_buses_inclusive_bswitch"])
    # --- drop shunt elements attached to boundary buses
    traget_shunt_idx = net_eq.shunt.index[net_eq.shunt.bus.isin(bus_lookups[
        "boundary_buses_inclusive_bswitch"])]
    net_eq.shunt = net_eq.shunt.drop(traget_shunt_idx)

    # --- create impedance
    not_very_low_imp = (impedance_params.rft_pu.abs() > imp_threshold) | (
        impedance_params.xft_pu.abs() > imp_threshold) | (
        impedance_params.rtf_pu.abs() > imp_threshold) | (
        impedance_params.xtf_pu.abs() > imp_threshold) | (
        impedance_params.from_bus.isin(set(net_eq.gen.bus)|set(net_eq.ext_grid.bus)) &
        impedance_params.to_bus.isin(set(net_eq.gen.bus)|set(net_eq.ext_grid.bus)))
    new_imps = impedance_params[["from_bus", "to_bus", "rft_pu", "xft_pu", "rtf_pu",
                                 "xtf_pu"]].loc[not_very_low_imp]
    max_idx = net_eq.impedance.index.max() if net_eq.impedance.shape[0] else 0
    new_imps.index = range(max_idx+1, max_idx+1+sum(not_very_low_imp))
    new_imps["name"] = "eq_impedance"
    new_imps["sn_mva"] = net_eq.sn_mva
    new_imps["in_service"] = True
    net_eq["impedance"] = pd.concat([net_eq["impedance"], new_imps])

    # --- create switches instead of very low impedances
    new_sws = impedance_params[["from_bus", "to_bus"]].loc[~not_very_low_imp].astype(np.int64)
    new_sws = new_sws.rename(columns={"from_bus": "bus", "to_bus": "element"})
    max_idx = net_eq.switch.index.max() if net_eq.switch.shape[0] else 0
    new_sws.index = range(max_idx+1, max_idx+1+sum(~not_very_low_imp))
    new_sws["et"] = "b"
    new_sws["name"] = "eq_switch"
    new_sws["closed"] = True
    new_sws["z_ohm"] = 0
    net_eq["switch"] = pd.concat([net_eq["switch"], new_sws])
    # If some buses are connected through switches, their shunts are connected in parallel
    # to same bus. The shunt parameters needs to be adapted. TODO
    if not not_very_low_imp.all():
        fb = impedance_params.from_bus[~not_very_low_imp].values.tolist()
        tb = impedance_params.to_bus[~not_very_low_imp].values.tolist()
        # fb_values = shunt_params.parameter[shunt_params.bus_pd.isin(fb)].values
        # tb_values = shunt_params.parameter[shunt_params.bus_pd.isin(tb)].values
        # adapted_params = fb_values * tb_values / (tb_values + fb_values)
        # shunt_params.parameter[shunt_params.bus_pd.isin(tb)] = adapted_params
        shunt_params.drop(shunt_params.index[shunt_params.bus_pd.isin(fb)], inplace=True)
        shunt_params.drop(shunt_params.index[shunt_params.bus_pd.isin(tb)], inplace=True)

    # --- create shunts
    max_idx = net_eq.shunt.index.max() if net_eq.shunt.shape[0] else 0
    shunt_buses = shunt_params.bus_pd.values.astype(np.int64)
    new_shunts = pd.DataFrame({"bus": shunt_buses,
                               "q_mvar": -shunt_params.parameter.values.imag * net_eq.sn_mva,
                               "p_mw": shunt_params.parameter.values.real * net_eq.sn_mva
                               }, index=range(max_idx+1, max_idx+1+shunt_params.shape[0]))
    new_shunts["name"] = "eq_shunt"
    isin_sh = new_shunts.bus.isin(net_eq.bus.index)
    new_shunts.loc[isin_sh, "vn_kv"] = net_eq.bus.vn_kv.loc[new_shunts.bus.loc[isin_sh]].values
    new_shunts["step"] = 1
    new_shunts["max_step"] = 1
    new_shunts["in_service"] = True
    net_eq["shunt"] = pd.concat([net_eq["shunt"], new_shunts])
    if n_disconnected_new_eq_shunts := sum(~isin_sh):
        msg = f"{n_disconnected_new_eq_shunts=}, missing buses: {new_shunts.bus.loc[~isin_sh]}"
        raise ValueError(msg)

    runpp_fct(net_eq, calculate_voltage_angles=calc_volt_angles,
              tolerance_mva=1e-6, max_iteration=100, **kwargs)


def _integrate_power_elements_connected_with_switch_buses(net, net_external, all_external_buses):
    all_buses, bus_dict = get_connected_switch_buses_groups(net_external,
                                                            all_external_buses)
    for elm in ["sgen", "load", "gen"]:
        for bd in bus_dict:
            if elm != "gen":
                connected_elms = net[elm].index[(net[elm].bus.isin(bd)) &
                                                (net[elm].in_service==True) &
                                                ~((net[elm].p_mw==0) & (net[elm].q_mvar==0))]
            else:
                connected_elms = net[elm].index[(net[elm].bus.isin(bd)) &
                                                (net[elm].in_service==True)]
            if len(connected_elms) <= 1:
                continue
            else:  # There ars some "external" elements connected with bus-bus switches.
                   # They will be aggregated.
                elm1 = connected_elms[0]
                net[elm].bus[connected_elms] = net[elm].bus[elm1]
                net_external[elm].bus[connected_elms] = net_external[elm].bus[elm1]
