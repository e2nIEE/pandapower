import os
from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd

import pandapower.toolbox
from pandapower.pd2ppc import _pd2ppc
from pandapower.pf.ppci_variables import _get_pf_variables_from_ppci
from pandapower.pf.run_newton_raphson_pf import _get_numba_functions, _get_Y_bus
from pandapower.run import _passed_runpp_parameters
from pandapower.auxiliary import _init_runpp_options, _add_dcline_gens
import pandapower as pp
import uuid

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)
home = str(Path.home())
desktop = os.path.join(home, "Desktop")


def _runpp_except_voltage_angles(net, **kwargs):
    if "calculate_voltage_angles" not in kwargs or not kwargs["calculate_voltage_angles"]:
        pp.runpp(net, **kwargs)
    else:
        try:
            pp.runpp(net, **kwargs)
        except pp.LoadflowNotConverged:
            kwargs1 = deepcopy(kwargs)
            kwargs1["calculate_voltage_angles"] = False
            pp.runpp(net, **kwargs1)
            logger.warning("In grid equivalent generation, the power flow did converge only without"
                           " calculate_voltage_angles.")
    return net


def add_ext_grids_to_boundaries(net, boundary_buses, adapt_va_degree=False,
                                runpp_fct=_runpp_except_voltage_angles,
                                calc_volt_angles=True, allow_net_change_for_convergence=False,
                                **kwargs):
    """
    adds ext_grids for the given network. If the bus results are
    available, ext_grids are created according to the given bus results;
    otherwise, ext_grids are created with vm_pu=1 and va_degreee=0
    """
    orig_slack_gens = net.gen.index[net.gen.slack]
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
    vms.loc[pd.Index(net.gen.bus.loc[net.gen.in_service]).intersection(vms.index)] = \
        net.gen.vm_pu.loc[net.gen.in_service & net.gen.bus.isin(vms.index) &
                          ~net.gen.bus.duplicated()].values  # avoid
        # different vm_pu setpoints at same buses
    vas.loc[btaegwr] = net.res_bus.va_degree.loc[btaegwr]

    for ext_bus, vm, va in zip(buses_to_add_ext_grids, vms, vas):
        add_eg += [pp.create_ext_grid(net, ext_bus,
                                      vm, va, name="assist_ext_grid")]
        new_bus = pp.create_bus(net, net.bus.vn_kv[ext_bus], name="assist_bus")
        pp.create_impedance(net, ext_bus, new_bus, 1e6, 1e6, net.sn_mva,
                            name="assist_impedance")

    # works fine if there is only one slack in net:
    if adapt_va_degree and net.gen.slack.any() and net.ext_grid.shape[0]:
        slack_buses = net.gen.bus.loc[net.gen.slack]
        net.gen.slack = False
        try:
            runpp_fct(net, calculate_voltage_angles=calc_volt_angles,
                      max_iteration=100, **kwargs)
        except pp.LoadflowNotConverged as e:
            if allow_net_change_for_convergence:

                # --- various fix trials

                # --- trail 1 -> massive change of data (switch sign of impedances)
                imp_neg = net.impedance.index[(net.impedance.xft_pu < 0)]
                imp_neg = net.impedance[["xft_pu"]].loc[imp_neg].sort_values("xft_pu").index
                for no, idx in enumerate(imp_neg):
                    net.impedance.loc[idx, ["rft_pu", "rtf_pu", "xft_pu", "xtf_pu"]] *= -1
                    try:
                        runpp_fct(net, calculate_voltage_angles=True, max_iteration=100, **kwargs)
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
                            runpp_fct(net, calculate_voltage_angles=calc_volt_angles,
                                    max_iteration=100, **kwargs)
                            logger.warning("Reactances of these impedances has been increased to "
                                        f"enable a power flow: {is2small}")
                        except pp.LoadflowNotConverged as e:
                            diag = pp.diagnostic(net)
                            print(net)
                            print(diag.keys())
                            pp.to_json(net, os.path.join(desktop, "diverged_net.json"))
                            raise pp.LoadflowNotConverged(e)
                    else:
                        diag = pp.diagnostic(net)
                        print(net)
                        print(diag.keys())
                        pp.to_json(net, os.path.join(desktop, "diverged_net.json"))
                        raise pp.LoadflowNotConverged(e)
            else:
                raise pp.LoadflowNotConverged(e)


        va = net.res_bus.va_degree.loc[slack_buses]
        va_ave = va.sum() / va.shape[0]
        net.ext_grid.va_degree.loc[add_eg] -= va_ave
        runpp_fct(net, calculate_voltage_angles=calc_volt_angles,
                 max_iteration=100, **kwargs)
    return orig_slack_gens


def drop_internal_branch_elements(net, internal_buses, branch_elements=None):
    """
    This function drops all branch elements which have 'internal_buses' connected at all sides of
    the branch element (e.g. for lines at 'from_bus' and 'to_bus').
    """
    bebd = pandapower.toolbox.branch_element_bus_dict()
    if branch_elements is not None:
        bebd = {elm: bus_types for elm,
                bus_types in bebd.items() if elm in branch_elements}
    for elm, bus_types in bebd.items():
        n_elms = net[elm].shape[0]
        if n_elms:
            should_be_dropped = np.ones((n_elms, ), dtype=bool)
            for bus_type in bus_types:
                should_be_dropped &= net[elm][bus_type].isin(internal_buses)
            idx_to_drop = net[elm].index[should_be_dropped]
            if elm == "line":
                pp.drop_lines(net, idx_to_drop)
            elif "trafo" in elm:
                pp.drop_trafos(net, idx_to_drop, table=elm)
            else:
                net[elm] = net[elm].drop(idx_to_drop)


def calc_zpbn_parameters(net, boundary_buses, all_external_buses, slack_as="gen",
                         existing_shift_degree=False):
    """
    The function calculats the parameters for zero power balance network

    OUTPUT:
        **Z** - impedance between essential buses and new buses

        **v** - voltage at new buses
    """
#    runpp_fct(net, calculate_voltage_angles=True)
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
                S[power.replace('_separate', '_integrated')] += S[power][k]
                S[sn.replace('_separate', '_integrated')] += S[sn][k]
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


def _ensure_unique_boundary_bus_names(net, boundary_buses):
    """ This function ad a unique name to each bounary bus. The original
        boundary bus names are retained.
    """
    assert "name_equivalent" not in net.bus.columns.tolist()
    net.bus["name_equivalent"] = "uuid"
    net.bus.loc[boundary_buses, "name_equivalent"] = ["Boundary bus " + str(uuid.uuid1()) for _ in
                                                      boundary_buses]


def drop_assist_elms_by_creating_ext_net(net, elms=None):
    """
    This function drops the assist elements by creating external nets.
    """
    if elms is None:
        elms = ["ext_grid", "bus", "impedance"]
    for elm in elms:
        target_elm_idx = net[elm].index[net[elm].name.astype(str).str.contains(
            "assist_"+elm, na=False, regex=False)]
        net[elm] = net[elm].drop(target_elm_idx)
        if net["res_"+elm].shape[0]:
            res_target_elm_idx = net["res_" +
                                     elm].index.intersection(target_elm_idx)
            net["res_"+elm] = net["res_"+elm].drop(res_target_elm_idx)

    if "name_equivalent" in net.bus.columns.tolist():
        net.bus = net.bus.drop(columns=["name_equivalent"])


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
    baseMVA, bus, gen, branch, *_, V0, ref_gens = _get_pf_variables_from_ppci(ppci)
    _, Ybus, _, _ = _get_Y_bus(ppci, net["_options"], makeYbus, baseMVA, bus, branch)

    net._ppc["internal"]["Ybus"] = Ybus


def drop_measurements_and_controllers(net, buses, skip_controller=False):
    """This function drops the measurements of the given buses.
    Also, the related controller parameters will be removed. """
    # --- dropping measurements
    if len(net.measurement):
        pp.drop_measurements_at_elements(net, "bus", idx=buses)
        lines = net.line.index[(net.line.from_bus.isin(buses)) &
                               (net.line.from_bus.isin(buses))]
        pp.drop_measurements_at_elements(net, "line", idx=lines)
        trafos = net.trafo3w.index[(net.trafo3w.hv_bus.isin(buses)) &
                                    (net.trafo3w.mv_bus.isin(buses)) &
                                    (net.trafo3w.lv_bus.isin(buses))]
        pp.drop_measurements_at_elements(net, "trafo", idx=trafos)
        trafo3ws = net.trafo3w.index[(net.trafo3w.hv_bus.isin(buses)) &
                                    (net.trafo3w.mv_bus.isin(buses)) &
                                    (net.trafo3w.lv_bus.isin(buses))]
        pp.drop_measurements_at_elements(net, "trafo3w", idx=trafo3ws)

    # --- dropping controller
    pp.drop_controllers_at_buses(net, buses)


def match_controller_and_new_elements(net, net_org):
    """
    This function makes the original controllers and the
    new created sgen to match

    test at present: controllers in the external area are removed.
    """
    if len(net.controller):
        tobe_removed = []
        if "origin_all_internal_buses" in net.bus_lookups and \
                "boundary_buses_inclusive_bswitch" in net.bus_lookups:
            internal_buses = net.bus_lookups["origin_all_internal_buses"] + \
                net.bus_lookups["boundary_buses_inclusive_bswitch"]
        else:
            internal_buses = []
        for idx in net.controller.index.tolist():
            et = net.controller.object[idx].__dict__.get("element")
            # var = net.controller.object[idx].__dict__.get("variable")
            elm_idxs = net.controller.object[idx].__dict__.get("element_index")
            if et is None or elm_idxs is None:
                continue
            org_elm_buses = list(net_org[et].bus[elm_idxs].values)

            new_elm_idxs = net[et].index[net[et].bus.isin(org_elm_buses)].tolist()
            if len(new_elm_idxs) == 0:
                tobe_removed.append(idx)
            else:
                profile_name = [org_elm_buses.index(a) for a in net[et].bus[new_elm_idxs].values]

                net.controller.object[idx].__dict__["element_index"] = new_elm_idxs
                net.controller.object[idx].__dict__["matching_params"]["element_index"] = new_elm_idxs
                net.controller.object[idx].__dict__["profile_name"] = profile_name
        net.controller = net.controller.drop(tobe_removed)
    # TODO: match the controllers in the external area

def ensure_origin_id(net, no_start=0, elms=None):
    """
    Ensures completely filled column 'origin_id' in every pp element.
    """
    if elms is None:
        elms = pandapower.toolbox.pp_elements()

    for elm in elms:
        if "origin_id" not in net[elm].columns:
            net[elm]["origin_id"] = pd.Series([None]*net[elm].shape[0], dtype=object)
        idxs = net[elm].index[net[elm].origin_id.isnull()]
        net[elm].loc[idxs, "origin_id"] = ["%s_%i_%s" % (elm, idx, str(uuid.uuid4())) for idx in idxs]


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
                    net[cost_elm].loc[idx, "bus"] = net[elm].bus.loc[net[cost_elm].element.loc[
                        idx]].values
                to_drop = net[cost_elm].index[net[cost_elm].bus.isin(buses) |
                                              net[cost_elm].bus.isnull()]
                net[cost_elm] = net[cost_elm].drop(to_drop)

            # add origin_id to cost df and corresponding elms
            if add_origin_id:
                ensure_origin_id(net, elms=set(net[cost_elm].et.values))
                if "et_origin_id" not in net[cost_elm].columns:
                    net[cost_elm]["et_origin_id"] = None
                    net[cost_elm]["origin_idx"] = None
                    net[cost_elm]["origin_seq"] = None
                for elm in set(net[cost_elm].et.values):
                    idx = net[cost_elm].index[net[cost_elm].et == elm]
                    net[cost_elm].loc[idx, "et_origin_id"] = net[elm].origin_id.loc[net[
                        cost_elm].element.loc[idx]].values
                    net[cost_elm].loc[idx, "origin_idx"] = idx
                    net[cost_elm].loc[idx, "origin_seq"] = [cost_backup.index.tolist().index(t) for t in idx]


def match_cost_functions_and_eq_net(net, boundary_buses, eq_type):
    """
    This function makes the element indices in poly_cost/pwl_cost and the
    new element indecies after merging to match.
    """
    for cost_elm in ["poly_cost", "pwl_cost"]:
        if len(net[cost_elm]):
            if "ward" not in eq_type:
                net[cost_elm] = net[cost_elm].sort_values(by=["origin_seq"])
                net[cost_elm].index = net[cost_elm]["origin_idx"].values
                for pc in net[cost_elm].itertuples():
                    new_idx = net[pc.et].index[
                        net[pc.et].origin_id == pc.et_origin_id].values
                    net[cost_elm].element[pc.Index] = new_idx[0]
            net[cost_elm] = net[cost_elm].drop(columns=["bus", "et_origin_id", "origin_idx", "origin_seq"])


def _check_network(net):
    """
    This function will perfoms some checks and modifications on the given grid model.
    """
    # --- check invative elements
    if net.res_bus.vm_pu.isnull().any():
        logger.info("There are some inactive buses. It is suggested to remove "
                    "them using 'pandapower.drop_inactive_elements()' "
                    "before starting the grid equivalent calculation.")

    # --- check dclines
    if "dcline" in net and len(net.dcline.query("in_service")) > 0:
        _add_dcline_gens(net)
        dcline_index = net.dcline.index.values
        net.dcline.loc[dcline_index, 'in_service'] = False
        logger.info(f"replaced dcline {dcline_index} by gen elements")

    # --- check controller names
    if len(net.controller):
       for i in net.controller.index:
           et = net.controller.object[i].__dict__.get("element")
           if et is not None and len(net[et]) != len(set(net[et].name.values)):
               raise ValueError("if controllers are used, please give a name for every "
                                 "element ("+et+"), and make sure the name is unique.")


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
        if abs(vm_errors[idx]) > 1e-6 and abs(vm_errors[idx]) > 1e-6:
            hb = pp.create_bus(net, net.bus.vn_kv[lb]*(1-vm_errors[idx]),
                               name="phase_shifter_adapter_"+str(lb))
            elm_dict = pp.get_connected_elements_dict(net, lb)
            for e, e_list in elm_dict.items():
                for i in e_list:
                    name = str(net[e].name[i])
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
    # runpp_fct(net, calculate_voltage_angles=True)
    return net


def replace_motor_by_load(net, all_external_buses):
    """
    replace the 'external' motors by loads. The name is modified.
    e.g., "equivalent_MotorName_3" ("equivalent"+"orignial name"+"original index")
    """
    motors = net.motor.index[net.motor.bus.isin(all_external_buses)]
    for mi, m in net.motor.loc[motors].iterrows():
        p_mech = m.pn_mech_mw / (m.efficiency_percent / 100)
        p_mw = p_mech * m.loading_percent / 100 * m.scaling
        s = p_mw / m.cos_phi
        q_mvar = np.sqrt(s**2 - p_mw**2)
        li = pp.create_load(net, m.bus, p_mw, q_mvar, sn_mva=s, scalling=m.scaling,
                            in_service=m.in_service, name="equivalent_"+str(m["name"])+"_"+str(mi))
        p = p_mw if not np.isnan(net.res_bus.vm_pu[m.bus]) and m.in_service else 0.0
        q = q_mvar if not np.isnan(net.res_bus.vm_pu[m.bus]) and m.in_service else 0.0
        net.res_load.loc[li] = p, q
    net.motor = net.motor.drop(motors)
    net.res_motor = net.res_motor.drop(motors)


if __name__ == "__main__":
    pass
