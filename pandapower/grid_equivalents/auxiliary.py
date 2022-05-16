import numpy as np
import pandas as pd
import pandapower as pp
from pandapower.pd2ppc import _pd2ppc
from pandapower.pf.ppci_variables import _get_pf_variables_from_ppci
from pandapower.pf.run_newton_raphson_pf import _get_numba_functions, _get_Y_bus
from pandapower.run import _passed_runpp_parameters
import uuid
from pandapower.auxiliary import _init_runpp_options

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def add_ext_grids_to_boundaries(net, boundary_buses, adapt_va_degree=False,
                                calc_volt_angles=True, allow_net_change_for_convergence=False):
    """
    adds ext_grids for the given network. If the bus results are
    available, ext_grids are created according to the given bus results;
    otherwise, ext_grids are created with vm_pu=1 and va_degreee=0
    """
    buses_to_add_ext_grids = set(boundary_buses) - set(net.ext_grid.bus[net.ext_grid.in_service]) \
                             - set(net.gen.bus[net.gen.in_service & net.gen.slack])
    res_buses = set(
        net.res_bus.index[~net.res_bus[["vm_pu", "va_degree"]].isnull().any(axis=1)])
    btaegwr = list(buses_to_add_ext_grids & res_buses)
    add_eg = []
    vms = pd.Series(np.ones(len(buses_to_add_ext_grids)),
                    index=buses_to_add_ext_grids)
    vas = pd.Series(np.zeros(len(buses_to_add_ext_grids)),
                    index=buses_to_add_ext_grids)
    vms.loc[btaegwr] = net.res_bus.vm_pu.loc[btaegwr]
    vas.loc[btaegwr] = net.res_bus.va_degree.loc[btaegwr]

    for ext_bus, vm, va in zip(buses_to_add_ext_grids, vms, vas):
        add_eg += [pp.create_ext_grid(net, ext_bus,
                                      vm, va, name="assist_ext_grid")]
        new_bus = pp.create_bus(net, net.bus.vn_kv[ext_bus], name="assist_bus")
        pp.create_impedance(net, ext_bus, new_bus, 0.0001, 0.0001, net.sn_mva,
                            name="assist_impedance")

    # works fine if there is only one slack in net:
    if adapt_va_degree and net.gen.slack.any() and net.ext_grid.shape[0]:
        slack_buses = net.gen.bus.loc[net.gen.slack]
        net.gen.slack = False
        try:
            pp.runpp(net, calculate_voltage_angles=calc_volt_angles,
                     max_iteration=100)
        except pp.LoadflowNotConverged as e:
            if allow_net_change_for_convergence:

                # --- various fix trials

                # --- trail 1 -> massive change of data (switch sign of impedances)
                imp_neg = net.impedance.index[(net.impedance.xft_pu < 0)]
                imp_neg = net.impedance[["xft_pu"]].loc[imp_neg].sort_values("xft_pu").index
                for no, idx in enumerate(imp_neg):
                    net.impedance.loc[idx, ["rft_pu", "rtf_pu", "xft_pu", "xtf_pu"]] *= -1
                    try:
                        pp.runpp(net, calculate_voltage_angles=True, max_iteration=100)
                        logger.warning("The sign of these impedances were changed to enable a power"
                                    f" flow: {imp_neg[:no]}")
                        break
                    except pp.LoadflowNotConverged as e:
                        pass

                if not net.converged:
                    net.impedance.loc[imp_neg, ["rft_pu", "rtf_pu", "xft_pu", "xtf_pu"]] *= -1

                    # --- trail 2 -> increase impedance values to avoid close to zero values
                    changes = False
                    for col in ["xtf_pu", "xft_pu"]:
                        is2small = net.impedance[col].abs() < 5e-6
                        changes |= is2small.any()
                        sign = np.sign(net.impedance[col].values[is2small])
                        net.impedance[col].loc[is2small] = sign * 5e-6
                    if changes:
                        try:
                            pp.runpp(net, calculate_voltage_angles=calc_volt_angles,
                                    max_iteration=100)
                            logger.warning("Reactances of these impedances has been increased to "
                                        f"enable a power flow: {is2small}")
                        except pp.LoadflowNotConverged as e:
                            diag = pp.diagnostic(net)
                            print(net)
                            print(diag.keys())
                            raise pp.LoadflowNotConverged(e)
                    else:
                        diag = pp.diagnostic(net)
                        print(net)
                        print(diag.keys())
                        raise pp.LoadflowNotConverged(e)
            else:
                raise pp.LoadflowNotConverged(e)


        va = net.res_bus.va_degree.loc[slack_buses]
        va_ave = va.sum() / va.shape[0]
        net.ext_grid.va_degree.loc[add_eg] -= va_ave
        pp.runpp(net, calculate_voltage_angles=calc_volt_angles,
                 max_iteration=100)


def drop_internal_branch_elements(net, internal_buses, branch_elements=None):
    """
    This function drops all branch elements which have 'internal_buses' connected at all sides of
    the branch element (e.g. for lines at 'from_bus' and 'to_bus').
    """
    bebd = pp.branch_element_bus_dict()
    if branch_elements is not None:
        bebd = {elm: bus_types for elm,
                bus_types in bebd.items() if elm in branch_elements}
    for elm, bus_types in bebd.items():
        n_elms = net[elm].shape[0]
        if n_elms:
            should_be_dropped = np.array([True]*n_elms)
            for bus_type in bus_types:
                should_be_dropped &= net[elm][bus_type].isin(internal_buses)
            idx_to_drop = net[elm].index[should_be_dropped]
            if elm == "line":
                pp.drop_lines(net, idx_to_drop)
            elif "trafo" in elm:
                pp.drop_trafos(net, idx_to_drop, table=elm)
            else:
                net[elm].drop(idx_to_drop, inplace=True)


def calc_zpbn_parameters(net, boundary_buses, all_external_buses, slack_as="gen", 
                         existing_shift_degree=False):
    """
    The function calculats the parameters for zero power balance network

    OUTPUT:
        **Z** - impedance between essential buses and new buses

        **v** - voltage at new buses
    """
#    pp.runpp(net, calculate_voltage_angles=True)
    be_buses = boundary_buses + all_external_buses
    if ((net.trafo.hv_bus.isin(be_buses)) & (net.trafo.shift_degree!=0)).any() \
        or ((net.trafo3w.hv_bus.isin(be_buses)) & \
             ((net.trafo3w.shift_mv_degree!=0) | (net.trafo3w.shift_lv_degree!=0))).any():
            existing_shift_degree = True
            logger.info("Transformers with non-zero shift-degree are existed," + 
                        " they could cause small inaccuracy.")
    # creata dataframe to collect the current injections of the external area
    nb_ext_buses = len(all_external_buses)
    S = pd.DataFrame(np.zeros((nb_ext_buses, 15)), dtype=complex)
    S.columns = ["ext_bus", "v_m", "v_cpx", "gen_integrated", "gen_separate",
                 "load_integrated", "load_separate", "sgen_integrated",
                 "sgen_separate", "sn_load_separate", "sn_load_integrated",
                 "sn_sgen_separate", "sn_sgen_integrated", "sn_gen_separate",
                 "sn_gen_integrated"]

    k, ind = 0, 0
    if slack_as == "gen":
        elements = set([("load", "res_load", "load_separate", "sn_load_separate", -1),
                        ("sgen", "res_sgen", "sgen_separate", "sn_sgen_separate", 1),
                        ("gen", "res_gen", "gen_separate", "sn_gen_separate", 1),
                        ("ext_grid", "res_ext_grid", "gen_separate", "sn_gen_separate", 1)])

    elif slack_as == "load":
        elements = set([("load", "res_load", "load_separate", "sn_load_separate", -1),
                        ("sgen", "res_sgen", "sgen_separate", "sn_sgen_separate", 1),
                        ("gen", "res_gen", "gen_separate", "sn_gen_separate", 1),
                        ("ext_grid", "res_ext_grid", "load_separate", "sn_load_separate", 1)])

    for i in all_external_buses:
        for ele, res_ele, power, sn, sign in elements:
            if i in net[ele].bus.values and net[ele].in_service[net[ele].bus == i].values.any():
                ind = list(net[ele].index[net[ele].bus == i].values)
                # act. values --> ref. values:
                S[power][k] += sum(net[res_ele].p_mw[ind].values * sign) / net.sn_mva + \
                    1j * sum(net[res_ele].q_mvar[ind].values *
                             sign) / net.sn_mva
                S[sn][k] = sum(net[ele].sn_mva[ind].values) + \
                    1j * 0 if ele != "ext_grid" else 1e6 + 1j * 0
                S[power.replace('_separate', '_integrated')][0] += S[power][k]
                S[sn.replace('_separate', '_integrated')][0] += S[sn][k]
        S.ext_bus[k] = all_external_buses[k]
        S.v_m[k] = net.res_bus.vm_pu[i]
        S.v_cpx[k] = S.v_m[k] * \
            np.exp(1j * net.res_bus.va_degree[i] * np.pi / 180)
        k = k + 1

    # create dataframe to calculate the impedance of the ZPBN-network
    Y = pd.DataFrame(np.zeros((nb_ext_buses, 10)), dtype=complex)
    Y.columns = ["ext_bus", "load_ground", "load_integrated_total", "load_separate_total",
                  "gen_ground", "gen_integrated_total", "gen_separate_total",
                  "sgen_ground", "sgen_integrated_total", "sgen_separate_total"]
    v = pd.DataFrame(dtype=complex)
    Y.ext_bus, v["ext_bus"] = all_external_buses, all_external_buses

    for elm in ["load", "gen", "sgen"]:
        if existing_shift_degree:
            Y[elm+"_ground"] = (S[elm+"_separate"].values / S.v_cpx.values).conjugate() / \
                S.v_cpx.values
        else:
            Y[elm+"_ground"] = S[elm+"_separate"].values.conjugate() / \
                np.square(S.v_m)
        I_elm_integrated_total = sum((S[elm+"_separate"].values /
                                      S.v_cpx.values).conjugate())
        if I_elm_integrated_total == 0:
            Y[elm+"_integrated_total"] = float("nan")
        else:
            vm_elm_integrated_total = S[elm+"_integrated"][0] / \
                                        I_elm_integrated_total.conjugate()
            if existing_shift_degree:
                Y[elm+"_integrated_total"] = (-S[elm+"_integrated"][0] / \
                                              vm_elm_integrated_total).conjugate() / \
                                              vm_elm_integrated_total
            else:
                Y[elm+"_integrated_total"] = -S[elm+"_integrated"][0].conjugate() / \
                    np.square(abs(vm_elm_integrated_total))
        Y[elm+"_separate_total"] = -Y[elm+"_ground"]
        if elm == "gen" and any(S.gen_separate):
            v["gen_integrated_vm_total"] = abs(vm_elm_integrated_total)
            v["gen_separate_vm_total"] = S.v_m

    Z = -1 / Y
    Z.ext_bus = all_external_buses

    # determine original external bus limits
    limits = pd.DataFrame([], index=range(nb_ext_buses), columns=[
        "min_vm_pu", "max_vm_pu", "ext_bus"])
    for limit in ["min_vm_pu", "max_vm_pu"]:
        if limit in net.bus.columns:
            limits[limit] = net.bus[limit].loc[all_external_buses].values
    limits["ext_bus"] = all_external_buses

    return Z, S, v, limits


def check_validity_of_Ybus_eq(net_zpbn, Ybus_eq, bus_lookups):
    """
    This Funktion proves the validity of the equivalent Ybus. If teh eqv. Ybus (Ybus_eq)
    is calculated correctly, the new_power and the origial power flow results should be equal
    """
    logger.debug("validiting the calculated Ybus_eq")

    ibt_buses = []
    for key in ["i", "b", "t"]:
        ibt_buses += bus_lookups["bus_lookup_ppc"][key+"_area_buses"]
    df = pd.DataFrame(columns=["bus_ppc", "bus_pd", "ext_grid_index", "power"])
    df.bus_ppc = ibt_buses

    for idx in df.index:
        df.bus_pd[idx] = list(
            net_zpbn._pd2ppc_lookups["bus"]).index(df.bus_ppc[idx])
        if df.bus_pd[idx] in net_zpbn.ext_grid.bus.values:
            df.ext_grid_index[idx] = net_zpbn.ext_grid.index[net_zpbn.ext_grid.bus == df.bus_pd[
                idx]][0]

    v_m = net_zpbn._ppc["bus"][df.bus_ppc.values, 7]
    delta = net_zpbn._ppc["bus"][df.bus_ppc.values, 8] * np.pi / 180
    v_cpx = v_m * np.exp(1j * delta)
    df.power = np.multiply(np.mat(v_cpx).T, np.conj(Ybus_eq * np.mat(v_cpx).T))
    df.dropna(axis=0, how="any", inplace=True)

    return df


def _ensure_unique_boundary_bus_names(net, boundary_buses):
    """ This function possibly changes the bus names of the boundaries buses to ensure
        that the names are unique.
    """
    idx_dupl_null = net.bus.index[net.bus.name.duplicated(
        keep=False) | net.bus.name.isnull()]
    idx_add_names = set(boundary_buses) & set(idx_dupl_null)
    if len(idx_add_names):
        net.bus.name.loc[idx_add_names] = ["Boundary bus " + str(uuid.uuid1()) for _ in
                                           idx_add_names]


def drop_assist_elms_by_creating_ext_net(net, elms=None):
    """
    This function drops the assist elements by creating external nets.
    """
    if elms is None:
        elms = ["ext_grid", "bus", "impedance"]
    for elm in elms:
        target_elm_idx = net[elm].index[net[elm].name.astype(str).str.contains(
            "assist_"+elm, na=False, regex=False)]
        net[elm].drop(target_elm_idx, inplace=True)
        if net["res_"+elm].shape[0]:
            res_target_elm_idx = net["res_" +
                                     elm].index.intersection(target_elm_idx)
            net["res_"+elm].drop(res_target_elm_idx, inplace=True)


def build_ppc_and_Ybus(net):
    """ This function build ppc and gets the Ybus of given network without
    runing power flow calculation
    """
    loc = locals()
    loc['kwargs'] = {}
    _init_runpp_options(net,
                        algorithm='nr',
                        calculate_voltage_angles="auto",
                        init="auto",
                        max_iteration="auto",
                        tolerance_mva=1e-8,
                        trafo_model="t",
                        trafo_loading="current",
                        enforce_q_lims=False,
                        check_connectivity=True,
                        voltage_depend_loads=True,
                        consider_line_temperature=False,
                        passed_parameters=_passed_runpp_parameters(loc))

    ppc, ppci = _pd2ppc(net)
    net["_ppc"] = ppc
    makeYbus, pfsoln = _get_numba_functions(ppci, net["_options"])
    baseMVA, bus, gen, branch, _, _, _, _, _, V0, ref_gens = _get_pf_variables_from_ppci(
        ppci)
    _, Ybus, _, _ = _get_Y_bus(
        ppci, net["_options"], makeYbus, baseMVA, bus, branch)

    net._ppc["internal"]["Ybus"] = Ybus


def drop_measurements_and_controller(net, buses):
    """This function drops the measurements of the given buses.
    Also, the related controller parameter will be removed. """
    # --- dropping measurements
    if len(net.measurement):
        elms = set(net.measurement.element_type.values)
        for elm in elms:
            if elm == "bus":
                elm_idx = buses
            elif elm == "line":
                elm_idx = net.line.index[(net.line.from_bus.isin(buses)) &
                                         (net.line.from_bus.isin(buses))]
            elif elm == "trafo":
                elm_idx = net.trafo.index[(net.trafo.hv_bus.isin(buses)) &
                                          (net.trafo.lv_bus.isin(buses))]
            elif elm == "trafo3w":
                elm_idx = net.trafo3w.index[(net.trafo3w.hv_bus.isin(buses)) &
                                            (net.trafo3w.mv_bus.isin(buses)) &
                                            (net.trafo3w.lv_bus.isin(buses))]
            target_idx = net.measurement.index[(net.measurement.element_type == elm) &
                                               (net.measurement.element.isin(elm_idx))]
            net.measurement.drop(target_idx, inplace=True)

    # --- dropping controller
    """
    only for test at present, only consider sgen.
    """
    if len(net.controller):
        if len(net.sgen) != len(set(net.sgen.name.values)):
            raise ValueError("if controllers are used, please give a name for every "
                             "element, and make sure the name is unique.")
        # only the sgen controllers are considered
        idx_pool = net.sgen.index[net.sgen.bus.isin(buses)].tolist()
        target_idx = []
        for idx in net.controller.index:
            try:  # problem caused by contoller
                sgen_idx = net.controller.object[idx].gid[0]
            except TypeError:
                sgen_idx = net.controller.object[idx].gid
            except AttributeError:
                sgen_idx = net.controller.object[idx].element_index[0]

            if sgen_idx in idx_pool:
                target_idx.append(idx)
        net.controller.drop(target_idx, inplace=True)


def match_controller_and_new_elements(net):
    """This function makes the original controllers and the
    new created sgen to match"""
    """
    only for test at present. only consider sgen.
    """
    if len(net.controller):
        count = 0
        if "origin_all_internal_buses" in net.bus_lookups and \
                "boundary_buses_inclusive_bswitch" in net.bus_lookups:
            internal_buses = net.bus_lookups["origin_all_internal_buses"] + \
                net.bus_lookups["boundary_buses_inclusive_bswitch"]
        else:
            internal_buses = []
        for idx in net.controller.index.tolist():
            # net.controller.object[idx].net = net
            try:
                bus = net.controller.object[idx].bus
            except AttributeError:
                bus = net.controller.object[idx].element_buses[0]
            else:
                pass
            # --- remove repeated controller at the boundary buses
            if bus in net.bus_lookups["boundary_buses_inclusive_bswitch"]:
                count += 1
                if count == 2:
                    net.controller.drop(idx, inplace=True)
                    continue

            if bus in internal_buses:
                try:
                    name = net.controller.object[idx].name
                except KeyError:
                    name = "found_no_element"
            else:
                name = "_rei_"+str(bus)

            new_idx = net.sgen.index[net.sgen.name.str.strip(
            ).str[-len(name):] == name].values
            if len(new_idx):
                assert len(new_idx) == 1
                new_bus = net.sgen.bus[new_idx[0]]
                net.controller.object[idx].gid = new_idx[0]
                net.controller.object[idx].element_index = [new_idx[0]]
                net.controller.object[idx].bus = new_bus
                net.controller.object[idx].element_buses = np.array(
                    [new_bus], dtype="int64")
            else:
                net.controller.drop(idx, inplace=True)

    """
    TODO: After nets merging, the net information in controller is not updated.
    """


def ensure_origin_id(net, no_start=0, elms=None):
    """
    Ensures completely filled column 'origin_id' in every pp element.
    """
    if elms is None:
        elms = pp.pp_elements()

    for elm in elms:
        if "origin_id" not in net[elm].columns:
            net[elm]["origin_id"] = pd.Series([None]*net[elm].shape[0], dtype=object)
        idxs = net[elm].index[net[elm].origin_id.isnull()]
        net[elm].origin_id.loc[idxs] = ["%s_%i_%s" % (elm, idx, str(uuid.uuid4())) for idx in idxs]


def drop_and_edit_cost_functions(net, buses, drop_cost, add_origin_id,
                                 check_unique_elms_name=True):
    """
    This function drops the ploy_cost/pwl_cost data
    related to the given buses.
    """
    for cost_elm in ["poly_cost", "pwl_cost"]:
        if len(net[cost_elm]):
            cost_backup = net[cost_elm].copy()
            # drop poly_cost and pwl_cost
            if drop_cost:
                net[cost_elm]["bus"] = None
                for elm in set(net[cost_elm].et.values):
                    idx = net[cost_elm].element.index[(net[cost_elm].et == elm) &
                                                      (net[cost_elm].element.isin(net[elm].index))]
                    net[cost_elm]["bus"].loc[idx] = net[elm].bus.loc[net[cost_elm].element.loc[
                        idx]].values
                to_drop = net[cost_elm].index[net[cost_elm].bus.isin(buses) |
                                              net[cost_elm].bus.isnull()]
                net[cost_elm].drop(to_drop, inplace=True)

            # add origin_id to cost df and corresponding elms
            if add_origin_id:
                ensure_origin_id(net, elms=set(net[cost_elm].et.values))
                if "et_origin_id" not in net[cost_elm].columns:
                    net[cost_elm]["et_origin_id"] = None
                    net[cost_elm]["origin_idx"] = None
                    net[cost_elm]["origin_seq"] = None
                for elm in set(net[cost_elm].et.values):
                    idx = net[cost_elm].index[net[cost_elm].et == elm]
                    net[cost_elm]["et_origin_id"].loc[idx] = net[elm].origin_id.loc[net[
                        cost_elm].element.loc[idx]].values
                    net[cost_elm]["origin_idx"].loc[idx] = idx
                    net[cost_elm]["origin_seq"].loc[idx] = [cost_backup.index.tolist().index(t) for t in idx]


def match_cost_functions_and_eq_net(net, boundary_buses, eq_type):
    """
    This function makes the element indices in poly_cost/pwl_cost and the
    new element indecies after merging to match.
    """
    for cost_elm in ["poly_cost", "pwl_cost"]:
        if len(net[cost_elm]):
            if "ward" not in eq_type:
                net[cost_elm].sort_values(by=["origin_seq"], inplace=True)
                net[cost_elm].index = net[cost_elm]["origin_idx"].values
                for pc in net[cost_elm].itertuples():
                    new_idx = net[pc.et].index[
                        net[pc.et].origin_id == pc.et_origin_id].values
                    net[cost_elm].element[pc.Index] = new_idx[0]
            net[cost_elm].drop(columns=["bus", "et_origin_id", "origin_idx", "origin_seq"], inplace=True)


def check_network(net):
    """
    checks the given network. If the network does not meet conditions,
    the program will report an error.
    """
    pass
    # --- condition 1: shift_degree of transformers must be 0.
    # if not np.allclose(net.trafo.shift_degree.values, 0) & \
    #         np.allclose(net.trafo3w.shift_mv_degree.values, 0) & \
    #         np.allclose(net.trafo3w.shift_lv_degree.values, 0):
    #     net["phase_shifter_actived"] = True
    # else:
    #     net["phase_shifter_actived"] = False
        # raise ValueError("the parameter 'shift_degree' of some transformers is not zero. "
        #                   "Currently, the get_equivalent function can not reduce "
        #                   "a network with non-zero shift_degree accurately.")


def get_boundary_vp(net_eq, bus_lookups):
    v_boundary = net_eq.ext_grid
    p_boundary = net_eq.res_ext_grid.values + net_eq.res_bus[["p_mw", "q_mvar"]].loc[bus_lookups["bus_lookup_pd"]["b_area_buses"]].values
    p_boundary = pd.DataFrame(p_boundary, index=bus_lookups["bus_lookup_pd"]["b_area_buses"],
                              columns=["p_mw", "q_mvar"])
    return v_boundary, p_boundary


def adaptation_phase_shifter(net, v_boundary, p_boundary):
    target_buses = list(v_boundary.bus.values)
    phase_errors = v_boundary.va_degree.values - \
        net.res_bus.va_degree[target_buses].values
    vm_errors = v_boundary.vm_pu.values - \
        net.res_bus.vm_pu[target_buses].values
    # p_errors = p_boundary.p_mw.values - \
    #     net.res_bus.p_mw[target_buses].values
    # q_errors = p_boundary.q_mvar.values - \
    #     net.res_bus.q_mvar[target_buses].values
    # print(q_errors)
    for idx, lb in enumerate(target_buses):
        if abs(vm_errors[idx] > 1e-6) and abs(vm_errors[idx]) > 1e-6:
            hb = pp.create_bus(net, net.bus.vn_kv[lb]*(1-vm_errors[idx]),
                               name="phase_shifter_adapter_"+str(lb))
            elm_dict = pp.get_connected_elements_dict(net, lb)
            for e, e_list in elm_dict.items():
                for i in e_list:
                    name = net[e].name[i]
                    if "eq_" not in name and "_integrated_" not in name and \
                        "_separate_" not in name:
                        if e in ["impedance", "line"]:
                            if net[e].from_bus[i] == lb:
                                net[e].from_bus[i] = hb
                            else:
                                net[e].to_bus[i] = hb
                        elif e == "trafo":
                            if net[e].hv_bus[i] == lb:
                                net[e].hv_bus[i] = hb
                            else:
                                net[e].lv_bus[i] = hb
                        elif e == "trafo3w":
                            if net[e].hv_bus[i] == lb:
                                net[e].hv_bus[i] == hb
                            elif net[e].mv_bus[i] == lb:
                                net[e].mv_bus[i] == hb
                            else:
                                net[e].lv_bus[i] == lb
                        elif e in ["bus", "load", "sgen", "gen", "shunt", "ward", "xward"]:
                            pass
                        else:
                            net[e].bus[i] = hb
            pp.create_transformer_from_parameters(net, hb, lb, 1e5,
                                                  net.bus.vn_kv[hb]*(1-vm_errors[idx]),
                                                  net.bus.vn_kv[lb],
                                                  vkr_percent=0, vk_percent=100,
                                                  pfe_kw=.0, i0_percent=.0,
                                                  # shift_degree=-phase_errors[idx],
                                                  tap_step_degree=-phase_errors[idx],
                                                  # tap_phase_shifter=True,
                                                  name="phase_shifter_adapter_"+str(lb))
        # pp.create_load(net, lb, -p_errors[idx], -q_errors[idx],
        #                name="phase_shifter_adapter_"+str(lb))
    print("debug")
    # pp.runpp(net, calculate_voltage_angles=True)
    return net

