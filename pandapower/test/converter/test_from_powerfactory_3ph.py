# -*- coding: utf-8 -*-

import os

import pandapower as pp
import pytest
import numpy as np

try:
    from pandaplan.core import pplog as logging
except ImportError:
    import logging

import json

logger = logging.getLogger(__name__)
testfiles_path = os.path.join(pp.pp_dir, 'test', 'converter', 'testfiles')

try:
    import powerfactory as pf

    PF_INSTALLED = True
except ImportError:
    PF_INSTALLED = False
    logger.info('could not import powerfactory, unbalanced load flow comparsion between pf and pp not possible')

from pandapower.converter.powerfactory.pf_export_functions import run_load_flow as run_powerfactory_load_flow
from pandapower.converter.powerfactory.validate import _get_pf_results_unbalanced, _set_pf_results_unbalanced

import pandas as pd


# @pytest.mark.skipif(not PF_INSTALLED, reason='powerfactory must be installed')
# def test_trafo_asym():
#     if not PF_INSTALLED:
#         return
#     # Only works with PowerFactory installed and configured
#     # and Yyn needs to be corrected, so there will be xfail atm
#     prj_name= "\\e2n049.IntUser\\Transformer 2 bus_trafo_11.03.20.IntPrj"
#     app = initialize_powerfactory(prj_name)
#     net = pp.from_json(os.path.join(this_file_path, "test_trafo_3ph.json"))
#     for loadtype in ["delta", "wye", "bal_wye"]:
#         for vector_group in [ "Dyn", "YNyn","Yzn"]:
#             print(loadtype, vector_group)
#             analyse_3ph_loadtypes_and_vectorgroups(app, net, loadtype, vector_group)

# @pytest.mark.skipif(not PF_INSTALLED, reason='powerfactory must be installed')
# def test_line_asym():
#     if not PF_INSTALLED:
#         return
#     # Only works with PowerFactory installed and configured atm
#     prj_name= "\\e2n049.IntUser\\2 bus_line_load_25.02.20.IntPrj"
#     app = initialize_powerfactory(prj_name)
#     net = pp.from_json(os.path.join(this_file_path, "test_line_3ph.json"))
#     pp.runpp_3ph(net)
#     get_pf_results(app, net)
#     compare_3ph_pp_pf_results(net)
#     return net

# @pytest.mark.skipif(not PF_INSTALLED, reason='powerfactory must be installed')
# def test_net_asym():
#     if not PF_INSTALLED:
#         return
#     # Only works with PowerFactory installed and configured
#     # and Yyn needs to be corrected, so there will be xfail atm
#     prj_name= "\\e2n049.IntUser\\3 bus_trafo_line_load_22.06.20.IntPrj"
#     app = initialize_powerfactory(prj_name)
#     net = pp.from_json(os.path.join(this_file_path, "test_net_3ph.json"))
#     for loadtype in ["delta", "wye", "bal_wye"]:
#         for vector_group in [ "Dyn", "YNyn","Yzn"]:
#             print(loadtype, vector_group)
#             analyse_3ph_loadtypes_and_vectorgroups(app, net, loadtype, vector_group)


def test_trafo_asym():
    net = pp.from_json(os.path.join(testfiles_path, "test_trafo_3ph.json"))
    for loadtype in ["delta", "wye"]:  # , "bal_wye"]:
        for vector_group in ["Dyn", "YNyn", "Yzn"]:
            logger.debug(loadtype, vector_group)
            net.asymmetric_load.type = loadtype
            net.trafo.vector_group = vector_group
            pp.runpp_3ph(net)
            read_pf_results_from_file_to_net(os.path.join(testfiles_path, "pf_combinations_results_trafo.json"), net,
                                             vector_group + "_" + loadtype)
            compare_3ph_pp_pf_results(net)
    return net


def test_line_asym():
    net = pp.from_json(os.path.join(testfiles_path, "test_line_3ph.json"))
    pp.runpp_3ph(net)
    read_pf_results_from_file_to_net(os.path.join(testfiles_path, "pf_results_line.json"), net)
    compare_3ph_pp_pf_results(net)
    return net


def test_net_asym():
    net = pp.from_json(os.path.join(testfiles_path, "test_net_3ph_test_2020_10_19.json"))
    for loadtype in ["wye", "delta"]:
        for vector_group in ["Dyn", "YNyn", "Yzn"]:
            logger.debug(loadtype, vector_group)
            net.asymmetric_load.type = loadtype
            net.trafo.vector_group = vector_group
            pp.runpp_3ph(net)
            read_pf_results_from_file_to_net(
                os.path.join(testfiles_path, "pf_combinations_results_net_bus_trafo_line_load_19.10.20.json"), net,
                vector_group + "_" + loadtype)
            compare_3ph_pp_pf_results(net)
    return net


def initialize_powerfactory(prj_name):
    app = pf.GetApplication()
    app.ActivateProject(prj_name)
    com_ldf = app.GetFromStudyCase('ComLdf')
    com_ldf.iopt_net = 1
    com_ldf.i_power = 1
    return app


def activate_study_case(app, study_case_name):
    study_cases = app.GetProjectFolder("study")
    logger.debug(study_cases)
    study_case = study_cases.GetContents(study_case_name)[0]
    study_case.Activate()


def analyse_3ph_loadtypes_and_vectorgroups(app, net, loadtype, vector_group):
    if not loadtype.startswith("bal_"):
        net.asymmetric_load.type = loadtype
        net.trafo.vector_group = vector_group
        pp.runpp_3ph(net)
        study_case_name = "Study Case_%s_%s" % (vector_group, loadtype)
        activate_study_case(app, study_case_name)
        get_pf_results(app, net)
        compare_3ph_pp_pf_results(net)


def compare_3ph_pp_pf_results(net):
    vm_tol = 1e-3
    va_tol = 1e-1

    i_line_tol = 1e-2
    i_trafo_tol = 1e-1
    #    trafo_loading_tol = 5e-1

    p_load_tol = 1e-2
    q_load_tol = 1e-1

    for phase in ["a", "b", "c"]:
        # bus
        diff_vm_pu = abs(net.res_bus_3ph["vm_%s_pu" % phase] - net.res_bus_3ph["pf_vm_%s_pu" % phase])
        logger.debug("vm_pu difference (phase %s):\n%s\n" % (phase, diff_vm_pu))
        assert all(diff_vm_pu < vm_tol)
        diff_va_degree = abs(net.res_bus_3ph["va_%s_degree" % phase] - net.res_bus_3ph["pf_va_%s_degree" % phase])
        logger.debug("va_degree difference (phase %s):\n%s\n" % (phase, diff_va_degree))
        assert all(diff_va_degree < va_tol)

        # trafo
        diff_i_hv_ka = abs(net.res_trafo_3ph["i_%s_hv_ka" % phase] - net.res_trafo_3ph["pf_i_%s_hv_ka" % phase])
        logger.debug("i_hv_ka difference:\n%s\n" % diff_i_hv_ka)
        assert all(diff_i_hv_ka < i_trafo_tol)
        diff_i_lv_ka = abs(net.res_trafo_3ph["i_%s_lv_ka" % phase] - net.res_trafo_3ph["pf_i_%s_lv_ka" % phase])
        logger.debug("i_lv_ka difference:\n%s\n" % diff_i_lv_ka)
        assert all(diff_i_lv_ka < i_trafo_tol)

        # load
        diff_p_mw = abs(net.res_asymmetric_load_3ph["p_%s_mw" % phase] - net.res_asymmetric_load_3ph["pf_p_%s" % phase])
        logger.debug("p_mw difference:\n%s\n" % diff_p_mw)
        assert all(diff_p_mw < p_load_tol)
        diff_q_mvar = abs(
            net.res_asymmetric_load_3ph["q_%s_mvar" % phase] - net.res_asymmetric_load_3ph["pf_q_%s" % phase])
        logger.debug("q_mvar difference:\n%s\n" % diff_q_mvar)
        assert all(diff_q_mvar < q_load_tol)

    # line
    for phase in ["a", "b", "c", "n"]:
        diff_i_from_ka = abs(net.res_line_3ph["i_%s_from_ka" % phase] - net.res_line_3ph["pf_i_%s_from_ka" % phase])
        logger.debug("i_from_ka difference:\n%s\n" % diff_i_from_ka)
        assert all(
            abs(net.res_line_3ph["i_%s_from_ka" % phase] - net.res_line_3ph["pf_i_%s_from_ka" % phase]) < i_line_tol)
        diff_i_to_ka = abs(net.res_line_3ph["i_%s_to_ka" % phase] - net.res_line_3ph["pf_i_%s_to_ka" % phase])
        logger.debug("i_to_ka difference:\n%s\n" % diff_i_to_ka)
        assert all(abs(net.res_line_3ph["i_%s_to_ka" % phase] - net.res_line_3ph["pf_i_%s_to_ka" % phase]) < i_line_tol)


def get_pf_results(app, net):
    run_powerfactory_load_flow(app)

    pf_types = {"bus": ".ElmTerm", "ext_grid": ".ElmXnet", "line": ".ElmLne", "trafo": ".ElmTr2",
                "asymmetric_load": ".ElmLod"}

    pf_result_variables = {
        "bus": {
            "pf_vm_a_pu": "m:u:A",
            "pf_va_a_degree": "m:phiu:A",
            "pf_vm_b_pu": "m:u:B",
            "pf_va_b_degree": "m:phiu:B",
            "pf_vm_c_pu": "m:u:C",
            "pf_va_c_degree": "m:phiu:C",
        },

        "ext_grid": {
            "pf_p_a": "m:P:bus1:A",
            "pf_p_b": "m:P:bus1:B",
            "pf_p_c": "m:P:bus1:C",
            "pf_q_a": "m:Q:bus1:A",
            "pf_q_b": "m:Q:bus1:B",
            "pf_q_c": "m:Q:bus1:C",
        },

        "line": {
            "pf_i_a_from_ka": "m:I:bus1:A",
            "pf_i_a_to_ka": "m:I:bus2:A",
            "pf_i_b_from_ka": "m:I:bus1:B",
            "pf_i_b_to_ka": "m:I:bus2:B",
            "pf_i_c_from_ka": "m:I:bus1:C",
            "pf_i_c_to_ka": "m:I:bus2:C",
            "pf_i_n_from_ka": "m:I0x3:bus1",
            "pf_i_n_to_ka": "m:I0x3:bus2",
            #          "pf_loading_percent'": "m:u:C",
        },
        "trafo": {
            "pf_i_a_hv_ka": "m:I:bushv:A",
            "pf_i_a_lv_ka": "m:I:buslv:A",
            "pf_i_b_hv_ka": "m:I:bushv:B",
            "pf_i_b_lv_ka": "m:I:buslv:B",
            "pf_i_c_hv_ka": "m:I:bushv:C",
            "pf_i_c_lv_ka": "m:I:buslv:C",
            #          "pf_loading_percent": "c:loading",
        },
        "asymmetric_load": {
            "pf_p_a": "m:P:bus1:A",
            "pf_p_b": "m:P:bus1:B",
            "pf_p_c": "m:P:bus1:C",
            "pf_q_a": "m:Q:bus1:A",
            "pf_q_b": "m:Q:bus1:B",
            "pf_q_c": "m:Q:bus1:C",
        }
    }

    for element_type, variables in pf_result_variables.items():
        for pp_pf_variable in variables.keys():
            net["res_" + element_type + "_3ph"][pp_pf_variable] = np.nan
        for idx, element in net[element_type].iterrows():
            pp_name = element["name"]
            pf_element = app.GetCalcRelevantObjects(pp_name + pf_types[element_type])[0]
            for pp_attr, pf_attr in variables.items():
                if pf_element.HasResults(0):
                    logger.debug(pp_attr, pf_element.GetAttribute(pf_attr))
                    net["res_" + element_type + "_3ph"][pp_attr].loc[idx] = pf_element.GetAttribute(pf_attr)


def write_pf_results_to_file(app, net, filename, combinations):
    if combinations:
        pf_results = dict()
        for loadtype in ["delta", "wye"]:
            for vector_group in ["Dyn", "YNyn", "Yzn"]:
                study_case_name = "Study Case_%s_%s" % (vector_group, loadtype)
                activate_study_case(app, study_case_name)
                get_pf_results(app, net)
                pf_res = _get_pf_results_unbalanced(net)
                pf_results["%s_%s" % (vector_group, loadtype)] = pf_res

        for comb_key, comb_item in pf_results.items():
            for key, item in comb_item.items():
                pf_results[comb_key][key] = item.to_dict()
    else:
        get_pf_results(app, net)
        pf_results = _get_pf_results_unbalanced(net)
        for key, item in pf_results.items():
            pf_results[key] = item.to_dict()

    import json
    with open(filename, "w") as f:
        json.dump(pf_results, f)


def read_pf_results_from_file_to_net(filename, net, combination=None):
    with open(filename, "r") as f:
        results = json.load(f)

    if combination is not None:
        results = results[combination]

    results_to_set = {}
    # need to avoid modifying the dist in the loop over itself
    for key, result_dict in results.items():
        result = pd.Series(result_dict, dtype=np.float64)
        result.index = result.index.astype(np.int64)
        results_to_set[key] = result
    _set_pf_results_unbalanced(net, results_to_set)


if __name__ == "__main__":
    pytest.main(['-xs', __file__])
