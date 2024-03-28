# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pandapower as pp
from pandapower.toolbox import replace_zero_branches_with_switches
from pandapower import diagnostic

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

def _get_pf_results(net, is_unbalanced=False):
    if not net["pf_converged"]:
        raise UserWarning("Load flow didn't converge in PowerFactory, no validation possible!")

    pf_results = None
    if is_unbalanced:
        pf_results = _get_pf_results_unbalanced(net)
    else:
        pf_results = _get_pf_results_balanced(net)

    return pf_results


def _get_pf_results_balanced(net):
    pf_switch_status = net.res_switch.pf_closed & \
               net.res_switch.pf_in_service if len(net.switch) > 0 and \
                                               'res_switch' in net.keys() else pd.Series(dtype=np.float64)
    pf_bus_vm = net.res_bus.pf_vm_pu.replace(0, np.nan)
    pf_bus_va = net.res_bus.pf_va_degree
    pf_ext_grid_p = net.res_ext_grid.get("pf_p", pd.Series([], dtype=np.float64))
    pf_ext_grid_q = net.res_ext_grid.get("pf_q", pd.Series([], dtype=np.float64))
    pf_gen_p = net.res_gen.get("pf_p", pd.Series([], dtype=np.float64))
    pf_gen_q = net.res_gen.get("pf_q", pd.Series([], dtype=np.float64))
    pf_ward_p = net.res_ward.get("pf_p", pd.Series([], dtype=np.float64))
    pf_ward_q = net.res_ward.get("pf_q", pd.Series([], dtype=np.float64))
    pf_xward_p = net.res_xward.get("pf_p", pd.Series([], dtype=np.float64))
    pf_xward_q = net.res_xward.get("pf_q", pd.Series([], dtype=np.float64))
    pf_sgen_p = net.res_sgen.get("pf_p", pd.Series([], dtype=np.float64))
    pf_sgen_q = net.res_sgen.get("pf_q", pd.Series([], dtype=np.float64))
    pf_load_p = net.res_load.get("pf_p", pd.Series([], dtype=np.float64))
    pf_load_q = net.res_load.get("pf_q", pd.Series([], dtype=np.float64))
    pf_line_loading = net.res_line.get("pf_loading", pd.Series([], dtype=np.float64))
    pf_trafo_loading = net.res_trafo.get("pf_loading", pd.Series([], dtype=np.float64))
    pf_trafo3w_loading = net.res_trafo3w.get("pf_loading", pd.Series([], dtype=np.float64))

    pf_results = {
        "pf_bus_vm": pf_bus_vm, "pf_bus_va": pf_bus_va, "pf_ext_grid_p": pf_ext_grid_p,
        "pf_ext_grid_q": pf_ext_grid_q, "pf_gen_p": pf_gen_p, "pf_gen_q": pf_gen_q,
        "pf_ward_p": pf_ward_p, "pf_ward_q": pf_ward_q, "pf_xward_p": pf_xward_p,
        "pf_xward_q": pf_xward_q, "pf_sgen_p": pf_sgen_p, "pf_sgen_q": pf_sgen_q,
        "pf_load_p": pf_load_p, "pf_load_q": pf_load_q, "pf_line_loading": pf_line_loading,
        "pf_trafo_loading": pf_trafo_loading, "pf_trafo3w_loading": pf_trafo3w_loading,
        'pf_switch_status': pf_switch_status
    }
    return pf_results


def _get_pf_results_unbalanced(net):
    pf_switch_status = net.res_switch.pf_closed & \
               net.res_switch.pf_in_service if len(net.switch) > 0 and \
                                               'res_switch' in net.keys() else pd.Series([], dtype=bool)
    # unbalanced get results
    pf_bus_vm_a = net.res_bus_3ph.pf_vm_a_pu.replace(0, np.nan)
    pf_bus_vm_b = net.res_bus_3ph.pf_vm_b_pu.replace(0, np.nan)
    pf_bus_vm_c = net.res_bus_3ph.pf_vm_c_pu.replace(0, np.nan)
    pf_bus_va_a = net.res_bus_3ph.pf_va_a_degree.replace(0, np.nan)
    pf_bus_va_b = net.res_bus_3ph.pf_va_b_degree.replace(0, np.nan)
    pf_bus_va_c = net.res_bus_3ph.pf_va_c_degree.replace(0, np.nan)

    pf_ext_grid_p_a = net.res_ext_grid_3ph.pf_p_a
    pf_ext_grid_p_b = net.res_ext_grid_3ph.pf_p_b
    pf_ext_grid_p_c = net.res_ext_grid_3ph.pf_p_c
    pf_ext_grid_q_a = net.res_ext_grid_3ph.pf_q_a
    pf_ext_grid_q_b = net.res_ext_grid_3ph.pf_q_b
    pf_ext_grid_q_c = net.res_ext_grid_3ph.pf_q_c

    pf_load_p_a = net.res_asymmetric_load_3ph.pf_p_a if len(net.asymmetric_load) > 0 else pd.Series([], dtype=np.float64)
    pf_load_p_b = net.res_asymmetric_load_3ph.pf_p_b if len(net.asymmetric_load) > 0 else pd.Series([], dtype=np.float64)
    pf_load_p_c = net.res_asymmetric_load_3ph.pf_p_c if len(net.asymmetric_load) > 0 else pd.Series([], dtype=np.float64)
    pf_load_q_a = net.res_asymmetric_load_3ph.pf_q_a if len(net.asymmetric_load) > 0 else pd.Series([], dtype=np.float64)
    pf_load_q_b = net.res_asymmetric_load_3ph.pf_q_b if len(net.asymmetric_load) > 0 else pd.Series([], dtype=np.float64)
    pf_load_q_c = net.res_asymmetric_load_3ph.pf_q_c if len(net.asymmetric_load) > 0 else pd.Series([], dtype=np.float64)

    pf_sgen_p_a = net.res_asymmetric_sgen_3ph.pf_p_a if len(net.asymmetric_sgen) > 0 else pd.Series([], dtype=np.float64)
    pf_sgen_p_b = net.res_asymmetric_sgen_3ph.pf_p_b if len(net.asymmetric_sgen) > 0 else pd.Series([], dtype=np.float64)
    pf_sgen_p_c = net.res_asymmetric_sgen_3ph.pf_p_c if len(net.asymmetric_sgen) > 0 else pd.Series([], dtype=np.float64)
    pf_sgen_q_a = net.res_asymmetric_sgen_3ph.pf_q_a if len(net.asymmetric_sgen) > 0 else pd.Series([], dtype=np.float64)
    pf_sgen_q_b = net.res_asymmetric_sgen_3ph.pf_q_b if len(net.asymmetric_sgen) > 0 else pd.Series([], dtype=np.float64)
    pf_sgen_q_c = net.res_asymmetric_sgen_3ph.pf_q_c if len(net.asymmetric_sgen) > 0 else pd.Series([], dtype=np.float64)

    pf_line_i_a_from_ka = net.res_line_3ph.pf_i_a_from_ka if len(net.line) > 0 else pd.Series([], dtype=np.float64)
    pf_line_i_a_to_ka = net.res_line_3ph.pf_i_a_to_ka if len(net.line) > 0 else pd.Series([], dtype=np.float64)
    pf_line_i_b_from_ka = net.res_line_3ph.pf_i_b_from_ka if len(net.line) > 0 else pd.Series([], dtype=np.float64)
    pf_line_i_b_to_ka = net.res_line_3ph.pf_i_b_to_ka if len(net.line) > 0 else pd.Series([], dtype=np.float64)
    pf_line_i_c_from_ka = net.res_line_3ph.pf_i_c_from_ka if len(net.line) > 0 else pd.Series([], dtype=np.float64)
    pf_line_i_c_to_ka = net.res_line_3ph.pf_i_c_to_ka if len(net.line) > 0 else pd.Series([], dtype=np.float64)
    pf_line_i_n_from_ka = net.res_line_3ph.pf_i_n_from_ka if len(net.line) > 0 else pd.Series([], dtype=np.float64)
    pf_line_i_n_to_ka = net.res_line_3ph.pf_i_n_to_ka if len(net.line) > 0 else pd.Series([], dtype=np.float64)
    pf_line_3ph_loading = net.res_line_3ph.pf_loading_percent if len(net.line) > 0 else pd.Series([], dtype=np.float64)

    pf_trafo_i_a_hv_ka = net.res_trafo_3ph.pf_i_a_hv_ka if len(net.trafo) > 0 else pd.Series([], dtype=np.float64)
    pf_trafo_i_a_lv_ka = net.res_trafo_3ph.pf_i_a_lv_ka if len(net.trafo) > 0 else pd.Series([], dtype=np.float64)
    pf_trafo_i_b_hv_ka = net.res_trafo_3ph.pf_i_b_hv_ka if len(net.trafo) > 0 else pd.Series([], dtype=np.float64)
    pf_trafo_i_b_lv_ka = net.res_trafo_3ph.pf_i_b_lv_ka if len(net.trafo) > 0 else pd.Series([], dtype=np.float64)
    pf_trafo_i_c_hv_ka = net.res_trafo_3ph.pf_i_c_hv_ka if len(net.trafo) > 0 else pd.Series([], dtype=np.float64)
    pf_trafo_i_c_lv_ka = net.res_trafo_3ph.pf_i_c_lv_ka if len(net.trafo) > 0 else pd.Series([], dtype=np.float64)
#    pf_trafo_i_n_hv_ka = net.res_trafo_3ph.pf_i_n_hv_ka if len(net.trafo) > 0 else pd.Series([], dtype=np.float64)
#    pf_trafo_i_n_lv_ka = net.res_trafo_3ph.pf_i_n_lv_ka if len(net.trafo) > 0 else pd.Series([], dtype=np.float64)
    pf_trafo_3ph_loading = net.res_trafo_3ph.pf_loading_percent if len(net.trafo) > 0 else pd.Series([], dtype=np.float64)


    pf_results = {
        'pf_switch_status': pf_switch_status,"pf_bus_vm_a": pf_bus_vm_a, "pf_bus_vm_b": pf_bus_vm_b,
        "pf_bus_vm_c": pf_bus_vm_c, "pf_bus_va_a": pf_bus_va_a, "pf_bus_va_b": pf_bus_va_b,
        "pf_bus_va_c": pf_bus_va_c,
        "pf_ext_grid_p_a": pf_ext_grid_p_a, "pf_ext_grid_p_b": pf_ext_grid_p_b,
        "pf_ext_grid_p_c": pf_ext_grid_p_c, "pf_ext_grid_q_a": pf_ext_grid_q_a,
        "pf_ext_grid_q_b": pf_ext_grid_q_b, "pf_ext_grid_q_c": pf_ext_grid_q_c,
        "pf_load_p_a": pf_load_p_a,  "pf_load_p_b": pf_load_p_b,  "pf_load_p_c": pf_load_p_c,
        "pf_load_q_a": pf_load_q_a, "pf_load_q_b": pf_load_q_b, "pf_load_q_c": pf_load_q_c,
        "pf_sgen_p_a": pf_sgen_p_a,  "pf_sgen_p_b": pf_sgen_p_b,  "pf_sgen_p_c": pf_sgen_p_c,
        "pf_sgen_q_a": pf_sgen_q_a, "pf_sgen_q_b": pf_sgen_q_b, "pf_sgen_q_c": pf_sgen_q_c,
        "pf_i_a_from_ka" : pf_line_i_a_from_ka, "pf_i_a_to_ka" : pf_line_i_a_to_ka,
        "pf_i_b_from_ka" : pf_line_i_b_from_ka, "pf_i_b_to_ka" : pf_line_i_b_to_ka,
        "pf_i_c_from_ka" : pf_line_i_c_from_ka, "pf_i_c_to_ka" : pf_line_i_c_to_ka,
        "pf_i_n_from_ka" : pf_line_i_n_from_ka, "pf_i_n_to_ka" : pf_line_i_n_to_ka,
        "pf_line_3ph_loading": pf_line_3ph_loading,
        "pf_i_a_hv_ka" : pf_trafo_i_a_hv_ka, "pf_i_a_lv_ka" : pf_trafo_i_a_lv_ka,
        "pf_i_b_hv_ka" : pf_trafo_i_b_hv_ka, "pf_i_b_lv_ka" : pf_trafo_i_b_lv_ka,
        "pf_i_c_hv_ka" : pf_trafo_i_c_hv_ka, "pf_i_c_lv_ka" : pf_trafo_i_c_lv_ka,
#        "pf_i_n_hv_ka" : pf_trafo_i_n_hv_ka, "pf_i_n_lv_ka" : pf_trafo_i_n_lv_ka,
        "pf_trafo_3ph_loading": pf_trafo_3ph_loading,

    }
    return pf_results


def _set_pf_results(net, pf_results, is_unbalanced=False):
    logger.debug("copying powerfactory results to pandapower")

    if is_unbalanced:
        _set_pf_results_unbalanced(net, pf_results)
    else:
        _set_pf_results_balanced(net, pf_results)


def _set_pf_results_balanced(net, pf_results):
    if 'res_switch' in net.keys():
        net.res_switch['pf_closed'] = pf_results['pf_switch_status']

    net.res_bus["pf_vm_pu"] = pf_results["pf_bus_vm"]
    net.res_bus["pf_va_degree"] = pf_results["pf_bus_va"]
    net.res_ext_grid["pf_p"] = pf_results["pf_ext_grid_p"]
    net.res_ext_grid["pf_q"] = pf_results["pf_ext_grid_q"]
    net.res_gen["pf_p"] = pf_results["pf_gen_p"]
    net.res_gen["pf_q"] = pf_results["pf_gen_q"]
    net.res_ward["pf_p"] = pf_results["pf_ward_p"]
    net.res_ward["pf_q"] = pf_results["pf_ward_q"]
    net.res_xward["pf_p"] = pf_results["pf_xward_p"]
    net.res_xward["pf_q"] = pf_results["pf_xward_q"]
    net.res_sgen["pf_p"] = pf_results["pf_sgen_p"]
    net.res_sgen["pf_q"] = pf_results["pf_sgen_q"]
    net.res_load["pf_p"] = pf_results["pf_load_p"]
    net.res_load["pf_q"] = pf_results["pf_load_q"]
    net.res_line["pf_loading"] = pf_results["pf_line_loading"]
    net.res_trafo["pf_loading"] = pf_results["pf_trafo_loading"]
    net.res_trafo3w["pf_loading"] = pf_results["pf_trafo3w_loading"]


def _set_pf_results_unbalanced(net, pf_results):
    if 'res_switch' in net.keys():
        net.res_switch['pf_closed'] = pf_results['pf_switch_status']

    #unbalanced set results
    net.res_bus_3ph["pf_vm_a_pu"] = pf_results["pf_bus_vm_a"]
    net.res_bus_3ph["pf_vm_b_pu"] = pf_results["pf_bus_vm_b"]
    net.res_bus_3ph["pf_vm_c_pu"] = pf_results["pf_bus_vm_c"]
    net.res_bus_3ph["pf_va_a_degree"] = pf_results["pf_bus_va_a"]
    net.res_bus_3ph["pf_va_b_degree"] = pf_results["pf_bus_va_b"]
    net.res_bus_3ph["pf_va_c_degree"] = pf_results["pf_bus_va_c"]

    net.res_ext_grid_3ph["pf_p_a"] = pf_results["pf_ext_grid_p_a"]
    net.res_ext_grid_3ph["pf_p_b"] = pf_results["pf_ext_grid_p_b"]
    net.res_ext_grid_3ph["pf_p_c"] = pf_results["pf_ext_grid_p_c"]
    net.res_ext_grid_3ph["pf_q_a"] = pf_results["pf_ext_grid_q_a"]
    net.res_ext_grid_3ph["pf_q_b"] = pf_results["pf_ext_grid_q_b"]
    net.res_ext_grid_3ph["pf_q_c"] = pf_results["pf_ext_grid_q_c"]

    net.res_asymmetric_load_3ph["pf_p_a"] = pf_results["pf_load_p_a"]
    net.res_asymmetric_load_3ph["pf_p_b"] = pf_results["pf_load_p_b"]
    net.res_asymmetric_load_3ph["pf_p_c"] = pf_results["pf_load_p_c"]
    net.res_asymmetric_load_3ph["pf_q_a"] = pf_results["pf_load_q_a"]
    net.res_asymmetric_load_3ph["pf_q_b"] = pf_results["pf_load_q_b"]
    net.res_asymmetric_load_3ph["pf_q_c"] = pf_results["pf_load_q_c"]

    net.res_asymmetric_sgen_3ph["pf_p_a"] = pf_results["pf_sgen_p_a"]
    net.res_asymmetric_sgen_3ph["pf_p_b"] = pf_results["pf_sgen_p_b"]
    net.res_asymmetric_sgen_3ph["pf_p_c"] = pf_results["pf_sgen_p_c"]
    net.res_asymmetric_sgen_3ph["pf_q_a"] = pf_results["pf_sgen_q_a"]
    net.res_asymmetric_sgen_3ph["pf_q_b"] = pf_results["pf_sgen_q_b"]
    net.res_asymmetric_sgen_3ph["pf_q_c"] = pf_results["pf_sgen_q_c"]

    net.res_line_3ph["pf_i_a_from_ka"] = pf_results["pf_i_a_from_ka"]
    net.res_line_3ph["pf_i_a_to_ka"] = pf_results["pf_i_a_to_ka"]
    net.res_line_3ph["pf_i_b_from_ka"] = pf_results["pf_i_b_from_ka"]
    net.res_line_3ph["pf_i_b_to_ka"] = pf_results["pf_i_b_to_ka"]
    net.res_line_3ph["pf_i_c_from_ka"] = pf_results["pf_i_c_from_ka"]
    net.res_line_3ph["pf_i_c_to_ka"] = pf_results["pf_i_c_to_ka"]
    net.res_line_3ph["pf_i_n_from_ka"] = pf_results["pf_i_n_from_ka"]
    net.res_line_3ph["pf_i_n_to_ka"] = pf_results["pf_i_n_to_ka"]
    net.res_line_3ph["pf_loading_percent"] = pf_results["pf_line_3ph_loading"]

    net.res_trafo_3ph["pf_i_a_hv_ka"] = pf_results["pf_i_a_hv_ka"]
    net.res_trafo_3ph["pf_i_a_lv_ka"] = pf_results["pf_i_a_lv_ka"]
    net.res_trafo_3ph["pf_i_b_hv_ka"] = pf_results["pf_i_b_hv_ka"]
    net.res_trafo_3ph["pf_i_b_lv_ka"] = pf_results["pf_i_b_lv_ka"]
    net.res_trafo_3ph["pf_i_c_hv_ka"] = pf_results["pf_i_c_hv_ka"]
    net.res_trafo_3ph["pf_i_c_lv_ka"] = pf_results["pf_i_c_lv_ka"]
#    net.res_trafo_3ph["pf_i_n_hv_ka"] = pf_results["pf_i_n_hv_ka"]
#    net.res_trafo_3ph["pf_i_n_lv_ka"] = pf_results["pf_i_n_lv_ka"]
    net.res_trafo_3ph["pf_loading_percent"] = pf_results["pf_trafo_3ph_loading"]


def validate_pf_conversion(net, is_unbalanced=False, **kwargs):
    """
    Trys to run a Loadflow with the converted pandapower network. If the loadflow converges, \
    PowerFactory and pandapower loadflow results will be compared. Note that a pf validation can \
    only be done if there are pf results available.

    INPUT:

        - **net** (PandapowerNetwork) - converted pandapower network

    OUTPUT:
        - **all_diffs** (list) - returns a list with the difference in all validated values.
                                 If all values are zero -> results are equal

    """
    logger.debug('starting verification')
    replace_zero_branches_with_switches(net)
    pf_results = _get_pf_results(net, is_unbalanced=is_unbalanced)

    run_control = "controller" in net.keys() and len(net.controller) > 0
    for arg in 'trafo_model check_connectivity'.split():
        if arg in kwargs:
            kwargs.pop(arg)
    if is_unbalanced:
        logger.info("running pandapower 3ph loadflow")
        pp.runpp_3ph(net, trafo_model="t", check_connectivity=True, run_control=run_control, **kwargs)
    else:
        logger.info("running pandapower loadflow")
        pp.runpp(net, trafo_model="t", check_connectivity=True, run_control=run_control, **kwargs)

    all_diffs = dict()
    logger.info('pandapower net converged: %s' % net.converged)
    _set_pf_results(net, pf_results, is_unbalanced=is_unbalanced)

    net.bus.name = net.bus.name.fillna("")
    only_in_pandapower = np.union1d(net.bus[net.bus.name.str.endswith("_aux")].index,
                                    net.bus[net.bus.type == "ls"].index)
    in_both = np.setdiff1d(net.bus.index, only_in_pandapower)

    pf_closed = pf_results['pf_switch_status']
    wrong_switches = net.res_switch.loc[
            pf_closed != net.switch.loc[pf_closed.index, 'closed']
    ].index.values if 'res_switch' in net.keys() else []
    if len(net.switch) > 0:
        logger.info('%d switches are wrong: %s' % (len(wrong_switches), wrong_switches))

    if len(net.trafo3w[net.trafo3w.in_service]) > 0:
        trafo3w_idx = net.trafo3w.query('in_service').index
        trafo3w_diff = net.res_trafo3w.loc[trafo3w_idx].pf_loading - net.res_trafo3w.loc[
            trafo3w_idx].loading_percent
        trafo3w_id = abs(trafo3w_diff).idxmax()
        logger.info("Maximum trafo3w loading difference between pandapower and powerfactory: %.1f "
                    "percent at trafo3w %d (%s)" % (
                        max(abs(trafo3w_diff)), trafo3w_id, net.trafo3w.at[trafo3w_id, 'name']))
        all_diffs["trafo3w_diff"] = trafo3w_diff

    if len(net.sgen[net.sgen.in_service]) > 0:
        logger.debug('verifying sgen')
        sgen_p_diff = net.res_sgen.pf_p.replace(np.nan, 0) - net.res_sgen.p_mw
        sgen_q_diff = net.res_sgen.pf_q.replace(np.nan, 0) - net.res_sgen.q_mvar
        sgen_p_diff_is = net.res_sgen.pf_p.replace(np.nan, 0) * net.sgen.loc[
            net.res_sgen.index, 'in_service'] - net.res_sgen.p_mw
        sgen_q_diff_is = net.res_sgen.pf_q.replace(np.nan, 0) * net.sgen.loc[
            net.res_sgen.index, 'in_service'] - net.res_sgen.q_mvar
        logger.info("Maximum sgen active power difference between pandapower and powerfactory: "
                    "%.1f MW, in service only: %.1f MW" % (max(abs(sgen_p_diff)),
                                                           max(abs(sgen_p_diff_is))))
        logger.info("Maximum sgen reactive power difference between pandapower and powerfactory: "
                    "%.1f Mvar, in service only: %.1f Mvar" % (max(abs(sgen_q_diff)),
                                                               max(abs(sgen_q_diff_is))))
        all_diffs["sgen_p_diff_is"] = sgen_p_diff_is
        all_diffs["sgen_q_diff_is"] = sgen_q_diff_is

    if len(net.gen[net.gen.in_service]) > 0:
        logger.debug('verifying gen')
        gen_p_diff = net.res_gen.pf_p.replace(np.nan, 0) - net.res_gen.p_mw
        gen_q_diff = net.res_gen.pf_q.replace(np.nan, 0) - net.res_gen.q_mvar
        gen_p_diff_is = net.res_gen.pf_p.replace(np.nan, 0) * net.gen.loc[
            net.res_gen.index, 'in_service'] - net.res_gen.p_mw
        gen_q_diff_is = net.res_gen.pf_q.replace(np.nan, 0) * net.gen.loc[
            net.res_gen.index, 'in_service'] - net.res_gen.q_mvar
        logger.info("Maximum gen active power difference between pandapower and powerfactory: "
                    "%.1f MW, in service only: %.1f MW" % (max(abs(gen_p_diff)),
                                                           max(abs(gen_p_diff_is))))
        logger.info("Maximum gen reactive power difference between pandapower and powerfactory: "
                    "%.1f Mvar, in service only: %.1f Mvar" % (max(abs(gen_q_diff)),
                                                               max(abs(gen_q_diff_is))))
        all_diffs["gen_p_diff_is"] = gen_p_diff_is
        all_diffs["gen_q_diff_is"] = gen_q_diff_is

    if len(net.ward[net.ward.in_service]) > 0:
        logger.debug('verifying ward')
        ward_p_diff = net.res_ward.pf_p.replace(np.nan, 0) - net.res_ward.p_mw
        ward_q_diff = net.res_ward.pf_q.replace(np.nan, 0) - net.res_ward.q_mvar
        ward_p_diff_is = net.res_ward.pf_p.replace(np.nan, 0) * net.ward.loc[
            net.res_ward.index, 'in_service'] - net.res_ward.p_mw
        ward_q_diff_is = net.res_ward.pf_q.replace(np.nan, 0) * net.ward.loc[
            net.res_ward.index, 'in_service'] - net.res_ward.q_mvar
        logger.info("Maximum ward active power difference between pandapower and powerfactory: "
                    "%.1f MW, in service only: %.1f MW" % (max(abs(ward_p_diff)),
                                                           max(abs(ward_p_diff_is))))
        logger.info("Maximum ward reactive power difference between pandapower and powerfactory: "
                    "%.1f Mvar, in service only: %.1f Mvar" % (max(abs(ward_q_diff)),
                                                               max(abs(ward_q_diff_is))))
        all_diffs["ward_p_diff_is"] = ward_p_diff_is
        all_diffs["ward_q_diff_is"] = ward_q_diff_is

    if is_unbalanced:
        _validate_pf_conversion_unbalanced(net, in_both, all_diffs)
    else:
        _validate_pf_conversion_balanced(net, in_both, all_diffs)

    return all_diffs


def _validate_pf_conversion_balanced(net, in_both, all_diffs):
    logger.debug('res_bus:\n%s' % net.res_bus)
    logger.debug('res_line:\n%s' % net.res_line)
    logger.debug('res_load:\n%s' % net.res_load)

    pfu = net.res_bus.pf_vm_pu.loc[in_both].replace(0, np.nan)
    pfa = net.res_bus.pf_va_degree.loc[in_both].replace(0, np.nan)
    ppu = net.res_bus.vm_pu.loc[in_both]
    ppa = net.res_bus.va_degree.loc[in_both]

    diff_vm = pfu - ppu
    diff_va = pfa - ppa

    pp_nans = diff_vm[(pd.notnull(pfu) & pd.isnull(ppu))]
    logger.info("%s buses are unsupplied in pandapower but supplied in powerfactory" % len(pp_nans))

    pf_nans = diff_vm[(pd.isnull(pfu) & pd.notnull(ppu))]
    logger.info("%s buses are unsupplied in powerfactory but supplied in pandapower" % len(pf_nans))

    diff_vm = diff_vm[(pd.notnull(pfu) & pd.notnull(ppu))]
    bus_id = abs(diff_vm).idxmax().astype('int64')
    logger.info("Maximum voltage magnitude difference between pandapower and powerfactory: "
                "%f pu at bus %d (%s)" % (max(abs(diff_vm)), bus_id, net.bus.at[bus_id, 'name']))

    diff_va = diff_va[(pd.notnull(pfa) & pd.notnull(ppa))]
    bus_id = abs(diff_va).idxmax().astype(np.int64)
    logger.info("Maximum voltage angle difference between pandapower and powerfactory: "
                "%.2f degrees at bus %d (%s)" % (
                    max(abs(diff_va)), bus_id, net.bus.at[bus_id, 'name']))

    all_diffs["diff_vm"] = diff_vm
    all_diffs["diff_va"] = diff_va

    if len(net.line[net.line.in_service]) > 0:
        section_loadings = pd.concat([net.line[["name", "line_idx"]], net.res_line[
            ["loading_percent", "pf_loading"]]], axis=1)
        line_loadings = section_loadings.groupby("line_idx").max()
        line_diff = line_loadings.loading_percent - line_loadings.pf_loading
        if sum(np.isnan(line_diff.values)):
            logger.info("Some line loading values are NaN.")
            line_diff = line_diff.dropna()
        if len(line_diff):
            line_id = int(abs(line_diff).idxmax())
            logger.info("Maximum line loading difference between pandapower and powerfactory: %.1f "
                        "percent at line %d (%s)" % (
                            max(abs(line_diff)), line_id, net.line.at[line_id, 'name']))
        all_diffs["line_diff"] = line_diff

    if len(net.trafo[net.trafo.in_service]) > 0:
        trafo_idx = net.trafo.query('in_service').index
        trafo_diff = net.res_trafo.loc[trafo_idx].pf_loading - net.res_trafo.loc[
            trafo_idx].loading_percent
        trafo_id = abs(trafo_diff).idxmax().astype('int64')
        logger.info("Maximum trafo loading difference between pandapower and powerfactory: %.1f "
                    "percent at trafo %d (%s)" % (
                        max(abs(trafo_diff)), trafo_id, net.trafo.at[trafo_id, 'name']))
        all_diffs["trafo_diff"] = trafo_diff

    if len(net.load[net.load.in_service]) > 0:
        logger.debug('verifying load')
        load_p_diff = net.res_load.pf_p.replace(np.nan, 0) - net.res_load.p_mw
        load_q_diff = net.res_load.pf_q.replace(np.nan, 0) - net.res_load.q_mvar
        load_p_diff_is = net.res_load.pf_p.replace(np.nan, 0) * net.load.loc[
            net.res_load.index, 'in_service'] - net.res_load.p_mw
        load_q_diff_is = net.res_load.pf_q.replace(np.nan, 0) * net.load.loc[
            net.res_load.index, 'in_service'] - net.res_load.q_mvar
        logger.info("Maximum load active power difference between pandapower and powerfactory: "
                    "%.1f MW, in service only: %.1f MW" % (max(abs(load_p_diff)),
                                                           max(abs(load_p_diff_is))))
        logger.info("Maximum load reactive power difference between pandapower and powerfactory: "
                    "%.1f Mvar, in service only: %.1f Mvar" % (max(abs(load_q_diff)),
                                                               max(abs(load_q_diff_is))))
        all_diffs["load_p_diff_is"] = load_p_diff_is
        all_diffs["load_q_diff_is"] = load_q_diff_is

    logger.debug('verifying ext_grid')
    eg_oos = net.ext_grid[~net.ext_grid.in_service].index
    ext_grid_p_diff = net.res_ext_grid.pf_p.replace(np.nan, 0).drop(eg_oos) - net.res_ext_grid.p_mw
    ext_grid_q_diff = net.res_ext_grid.pf_q.replace(np.nan, 0).drop(
        eg_oos) - net.res_ext_grid.q_mvar
    logger.info("Maximum ext_grid active power difference between pandapower and powerfactory: "
                "%.1f MW" % max(abs(ext_grid_p_diff)))
    logger.info("Maximum ext_grid reactive power difference between pandapower and powerfactory: "
                "%.1f Mvar" % max(abs(ext_grid_q_diff)))
    all_diffs["ext_grid_p_diff"] = ext_grid_p_diff
    all_diffs["ext_grid_q_diff"] = ext_grid_q_diff

    return all_diffs


def _validate_pf_conversion_unbalanced(net, in_both, all_diffs):
    logger.debug('res_bus_3ph:\n%s' % net.res_bus_3ph)
    logger.debug('res_line_3ph:\n%s' % net.res_line_3ph)
    logger.debug('res_asymmetric_load_3ph:\n%s' % net.res_asymmetric_load_3ph)

    for phase in ["a", "b", "c"]:
        pfu = net.res_bus_3ph["pf_vm_%s_pu" % phase].loc[in_both].replace(0, np.nan)
        ppu = net.res_bus_3ph["vm_%s_pu" % phase].loc[in_both]
        diff_vm = pfu - ppu

        pfa = net.res_bus_3ph["pf_va_%s_degree" % phase].loc[in_both].replace(0, np.nan)
        ppa = net.res_bus_3ph["va_%s_degree" % phase].loc[in_both]
        diff_va = pfa - ppa

        all_diffs["diff_vm_%s" % phase] = diff_vm
        all_diffs["diff_va_%s" % phase] = diff_va

        if phase == "a":
            pp_nans = diff_vm[(pd.notnull(pfu) & pd.isnull(ppu))]
            logger.info("%s buses are unsupplied in pandapower but supplied in powerfactory" % len(pp_nans))
            pf_nans = diff_vm[(pd.isnull(pfu) & pd.notnull(ppu))]
            logger.info("%s buses are unsupplied in powerfactory but supplied in pandapower" % len(pf_nans))

        diff_vm = diff_vm[(pd.notnull(pfu) & pd.notnull(ppu))]
        bus_id = abs(diff_vm).idxmax().astype('int64')
        logger.info("Maximum voltage magnitude difference of phase '%s' between pandapower and powerfactory: "
                    "%f pu at bus %d (%s)" % (phase, max(abs(diff_vm)), bus_id, net.bus.at[bus_id, 'name']))

        diff_va = diff_va[(pd.notnull(pfa) & pd.notnull(ppa))]
        bus_id = abs(diff_va).idxmax().astype(np.int64)
        logger.info("Maximum voltage angle difference of phase '%s' between pandapower and powerfactory: "
                    "%.2f degrees at bus %d (%s)" % (phase, max(abs(diff_va)), bus_id, net.bus.at[bus_id, 'name']))

    if len(net.line[net.line.in_service]) > 0:
        section_loadings = pd.concat([net.line[["name", "line_idx"]], net.res_line_3ph[
            ["loading_percent", "pf_loading_percent"]]], axis=1)
        line_loadings = section_loadings.groupby("line_idx").max()
        line_loading_diff = line_loadings.loading_percent - line_loadings.pf_loading_percent
        if sum(np.isnan(line_loading_diff.values)):
            logger.info("Some line loading values are NaN.")
            line_loading_diff = line_loading_diff.dropna()
        if len(line_loading_diff):
            line_id = int(abs(line_loading_diff).idxmax())
            logger.info("Maximum line loading difference between pandapower and powerfactory: %.1f "
                        "percent at line %d (%s)" % (
                            max(abs(line_loading_diff)), line_id, net.line.at[line_id, 'name']))
        all_diffs["line_loading_diff"] = line_loading_diff

        for phase in ["a", "b", "c", "n"]:
            for direction in ["from", "to"]:
                section_currents = pd.concat([net.line[["name", "line_idx"]], net.res_line_3ph[
                ["i_%s_%s_ka" % (phase, direction), "pf_i_%s_%s_ka" % (phase, direction)]]], axis=1)
                line_currents = section_currents.groupby("line_idx").max()
                line_currents_diff = line_currents["i_%s_%s_ka" % (phase, direction)] - \
                                     line_currents["pf_i_%s_%s_ka" % (phase, direction)]
                if sum(np.isnan(line_currents_diff.values)):
                    logger.info("Some line current values are NaN for phase %s, direction %s." % (phase, direction))
                    line_currents_diff = line_currents_diff.dropna()
                if len(line_currents_diff):
                    line_id = int(abs(line_currents_diff).idxmax())
                    logger.info("Maximum line current difference between pandapower and powerfactory: %.4f "
                                "current at phase %s, direction %s of line %d (%s)" % (
                                max(abs(line_currents_diff)), phase, direction, line_id, net.line.at[line_id, 'name']))
                all_diffs["line_currents_%s_%s_diff" % (phase, direction)] = line_currents_diff

    if len(net.trafo[net.trafo.in_service]) > 0:
        trafo_idx = net.trafo.query('in_service').index
        trafo_loading_diff = net.res_trafo_3ph.loc[trafo_idx].loading_percent - net.res_trafo_3ph.loc[trafo_idx].pf_loading_percent
        if sum(np.isnan(trafo_loading_diff.values)):
            logger.info("Some trafo loading values are NaN.")
            trafo_loading_diff = trafo_loading_diff.dropna()
        if len(trafo_loading_diff):
            trafo_id = int(abs(trafo_loading_diff).idxmax())
            logger.info("Maximum trafo loading difference between pandapower and powerfactory: %.1f "
                        "percent at trafo %d (%s)" % (
                            max(abs(trafo_loading_diff)), trafo_id, net.trafo.at[trafo_id, 'name']))
        all_diffs["trafo_loading_diff"] = trafo_loading_diff

        for phase in ["a", "b", "c"]:#, "n"]:
            for side in ["hv", "lv"]:
                trafo_idx = net.trafo.query('in_service').index
                trafo_currents_diff = net.res_trafo_3ph.loc[trafo_idx]["i_%s_%s_ka" % (phase, side)] - \
                net.res_trafo_3ph.loc[trafo_idx]["pf_i_%s_%s_ka" % (phase, side)]
                if sum(np.isnan(trafo_currents_diff.values)):
                    logger.info("Some trafo current values are NaN for phase %s, side %s." % (phase, side))
                    trafo_currents_diff = trafo_currents_diff.dropna()
                if len(trafo_currents_diff):
                    trafo_id = int(abs(trafo_currents_diff).idxmax())
                    logger.info("Maximum trafo current difference between pandapower and powerfactory: %.4f "
                                "current at phase %s, side %s of line %d (%s)" % (
                                max(abs(trafo_currents_diff)), phase, side, trafo_id, net.trafo.at[trafo_id, 'name']))
                all_diffs["trafo_currents_%s_%s_diff" % (phase, side)] = trafo_currents_diff

    if len(net.asymmetric_load[net.asymmetric_load.in_service]) > 0:
        logger.debug('verifying asymmetric load')

        for phase in ["a", "b", "c"]:
            asymmetric_load_p_diff = net.res_asymmetric_load_3ph["pf_p_%s" % phase].replace(np.nan, 0) - \
                net.res_asymmetric_load_3ph["p_%s_mw" % phase]
            asymmetric_load_q_diff = net.res_asymmetric_load_3ph["pf_q_%s" % phase].replace(np.nan, 0) - \
                net.res_asymmetric_load_3ph["q_%s_mvar" % phase]
            asymmetric_load_p_diff_is = net.res_asymmetric_load_3ph["pf_p_%s" % phase].replace(np.nan, 0) * net.asymmetric_load.loc[
                net.res_asymmetric_load_3ph["pf_p_%s" % phase].index, 'in_service'] - net.res_asymmetric_load_3ph["p_%s_mw" % phase]
            asymmetric_load_q_diff_is = net.res_asymmetric_load_3ph["pf_q_%s" % phase].replace(np.nan, 0) * net.asymmetric_load.loc[
                net.res_asymmetric_load_3ph["pf_q_%s" % phase].index, 'in_service'] - net.res_asymmetric_load_3ph["q_%s_mvar" % phase]

            logger.info("Maximum asymmetric load active power difference between pandapower and powerfactory: "
                        "%.1f MW, in service only: %.1f MW" % (max(abs(asymmetric_load_p_diff)),
                                                               max(abs(asymmetric_load_p_diff_is))))
            logger.info("Maximum asymmetric load reactive power difference between pandapower and powerfactory: "
                        "%.1f Mvar, in service only: %.1f Mvar" % (max(abs(asymmetric_load_q_diff)),
                                                                   max(abs(asymmetric_load_q_diff_is))))
            all_diffs["asymmetric_load_p_diff_is"] = asymmetric_load_p_diff_is
            all_diffs["asymmetric_load_q_diff_is"] = asymmetric_load_q_diff_is

        if len(net.ext_grid[net.ext_grid.in_service]) > 0:

            logger.debug('verifying ext_grid')

            for phase in ["a", "b", "c"]:
                ext_grid_p_diff = net.res_ext_grid_3ph["pf_p_%s" % phase].replace(np.nan, 0) - \
                    net.res_ext_grid_3ph["p_%s_mw" % phase]
                ext_grid_q_diff = net.res_ext_grid_3ph["pf_q_%s" % phase].replace(np.nan, 0) - \
                    net.res_ext_grid_3ph["q_%s_mvar" % phase]
                ext_grid_p_diff_is = net.res_ext_grid_3ph["pf_p_%s" % phase].replace(np.nan, 0) * net.ext_grid.loc[
                    net.res_ext_grid_3ph["pf_p_%s" % phase].index, 'in_service'] - net.res_ext_grid_3ph["p_%s_mw" % phase]
                ext_grid_q_diff_is = net.res_ext_grid_3ph["pf_q_%s" % phase].replace(np.nan, 0) * net.ext_grid.loc[
                    net.res_ext_grid_3ph["pf_q_%s" % phase].index, 'in_service'] - net.res_ext_grid_3ph["q_%s_mvar" % phase]

                logger.info("Maximum ext_grid active power difference between pandapower and powerfactory: "
                            "%.1f MW, in service only: %.1f MW" % (max(abs(ext_grid_p_diff)), max(abs(ext_grid_p_diff_is))))
                logger.info("Maximum ext_grid reactive power difference between pandapower and powerfactory: "
                            "%.1f Mvar, in service only: %.1f Mvar" % (max(abs(ext_grid_q_diff)), max(abs(ext_grid_q_diff_is))))
                all_diffs["ext_grid_p_diff"] = ext_grid_p_diff_is
                all_diffs["ext_grid_q_diff"] = ext_grid_q_diff_is

    if len(net.asymmetric_sgen[net.asymmetric_sgen.in_service]) > 0:
        logger.debug('verifying asymmetric sgen')

        for phase in ["a", "b", "c"]:
            asymmetric_sgen_p_diff = net.res_asymmetric_sgen_3ph["pf_p_%s" % phase].replace(np.nan, 0) - \
                net.res_asymmetric_sgen_3ph["p_%s_mw" % phase]
            asymmetric_sgen_q_diff = net.res_asymmetric_sgen_3ph["pf_q_%s" % phase].replace(np.nan, 0) - \
                net.res_asymmetric_sgen_3ph["q_%s_mvar" % phase]
            asymmetric_sgen_p_diff_is = net.res_asymmetric_sgen_3ph["pf_p_%s" % phase].replace(np.nan, 0) * net.asymmetric_sgen.loc[
                net.res_asymmetric_sgen_3ph["pf_p_%s" % phase].index, 'in_service'] - net.res_asymmetric_sgen_3ph["p_%s_mw" % phase]
            asymmetric_sgen_q_diff_is = net.res_asymmetric_sgen_3ph["pf_q_%s" % phase].replace(np.nan, 0) * net.asymmetric_sgen.loc[
                net.res_asymmetric_sgen_3ph["pf_q_%s" % phase].index, 'in_service'] - net.res_asymmetric_sgen_3ph["q_%s_mvar" % phase]

            logger.info("Maximum asymmetric sgen active power difference between pandapower and powerfactory: "
                        "%.1f MW, in service only: %.1f MW" % (max(abs(asymmetric_sgen_p_diff)),
                                                               max(abs(asymmetric_sgen_p_diff_is))))
            logger.info("Maximum asymmetric sgen reactive power difference between pandapower and powerfactory: "
                        "%.1f Mvar, in service only: %.1f Mvar" % (max(abs(asymmetric_sgen_q_diff)),
                                                                   max(abs(asymmetric_sgen_q_diff_is))))
            all_diffs["asymmetric_sgen_p_diff_is"] = asymmetric_sgen_p_diff_is
            all_diffs["asymmetric_sgen_q_diff_is"] = asymmetric_sgen_q_diff_is

    return all_diffs

