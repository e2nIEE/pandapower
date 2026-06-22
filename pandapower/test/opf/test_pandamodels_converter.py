# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from copy import deepcopy
import json
import os

import numpy as np
import pytest

import pandapower.test as test
from pandapower.converter.pandamodels import convert_pp_to_pm
from pandapower.converter.pandamodels.from_pm import read_pm_results_to_net
from pandapower.create import create_empty_network, create_bus, create_ext_grid, create_line_from_parameters, \
    create_load, create_poly_cost, create_sgen
from pandapower.pd2ppc import _pd2ppc
from pandapower.run import runopp
from pandapower.test.opf.test_basic import simple_opf_test_net, net_3w_trafo_opf

try:
    from juliacall import JuliaError as UnsupportedPythonError # type: ignore
except ImportError:
    UnsupportedPythonError = Exception

try:
    from juliacall import Main # type: ignore
    julia_installed = True
except (ImportError, RuntimeError, UnsupportedPythonError) as e:
    julia_installed = False


def test_pp_to_pm_conversion(net_3w_trafo_opf):
    # tests if the conversion to power models works
    net = deepcopy(net_3w_trafo_opf)
    convert_pp_to_pm(net)
    convert_pp_to_pm(net, opf_flow_lim="I")


def test_pm_to_pp_conversion(simple_opf_test_net):
    # this tests checks if the runopp results are the same as the ones from powermodels.
    # Results are read from a result file containing the simple_opf_test_net

    net = simple_opf_test_net
    create_poly_cost(net, 0, "gen", cp1_eur_per_mw=100)

    # get pandapower opf results
    runopp(net, delta=1e-13)
    va_degree = deepcopy(net.res_bus.va_degree)
    vm_pu = deepcopy(net.res_bus.vm_pu)

    # get previously calculated power models results
    pm_res_file = os.path.join(os.path.abspath(os.path.dirname(test.__file__)),
                               "test_files", "pm_example_res.json")

    with open(pm_res_file, "r") as fp:
        result_pm = json.load(fp)
    net._options["correct_pm_network_data"] = True
    ppc, ppci = _pd2ppc(net)
    read_pm_results_to_net(net, ppc, ppci, result_pm)
    assert np.allclose(net.res_bus.vm_pu, vm_pu, atol=1e-4)
    assert np.allclose(net.res_bus.va_degree, va_degree, atol=1e-2, rtol=1e-2)


def test_obj_factors(net_3w_trafo_opf):
    net = deepcopy(net_3w_trafo_opf)
    net["obj_factors"] = [0.9, 0.1]
    pm = convert_pp_to_pm(net)
    assert pm["user_defined_params"]["obj_factors"]["fac_1"] == 0.9
    assert pm["user_defined_params"]["obj_factors"]["fac_2"] == 0.1
    assert pm["user_defined_params"]["gen_and_controllable_sgen"]["2"] == 2
    assert pm["user_defined_params"]["gen_and_controllable_sgen"]["3"] == 3


def test_controllable_load_polynomial_cost_signs_for_powermodels():
    net = create_empty_network()
    create_bus(net, max_vm_pu=1.05, min_vm_pu=0.95, vn_kv=10.)
    create_bus(net, max_vm_pu=1.05, min_vm_pu=0.95, vn_kv=.4)
    create_ext_grid(net, 0, controllable=False)
    create_line_from_parameters(net, 0, 1, 1, name="line2", r_ohm_per_km=0.1,
                                c_nf_per_km=0.0, max_i_ka=1.0, x_ohm_per_km=0.1,
                                max_loading_percent=100)
    load = create_load(net, 1, p_mw=0.5, q_mvar=0.1, controllable=True,
                       min_p_mw=0.1, max_p_mw=1.0, min_q_mvar=-0.5, max_q_mvar=0.5)
    sgen = create_sgen(net, 1, p_mw=0.5, q_mvar=0.1, controllable=True,
                       min_p_mw=0.1, max_p_mw=1.0, min_q_mvar=-0.5, max_q_mvar=0.5)

    create_poly_cost(net, load, "load", cp0_eur=3, cp1_eur_per_mw=1, cp2_eur_per_mw2=2)
    create_poly_cost(net, sgen, "sgen", cp0_eur=30, cp1_eur_per_mw=10, cp2_eur_per_mw2=20)

    pm = convert_pp_to_pm(net)
    load_gen = net._pd2ppc_lookups["load_controllable"][load] + 1
    sgen_gen = net._pd2ppc_lookups["sgen_controllable"][sgen] + 1

    assert pm["gen"][str(load_gen)]["cost"] == [2, -1, 3]
    assert pm["gen"][str(sgen_gen)]["cost"] == [20, 10, 30]


if __name__ == '__main__':
    pytest.main([__file__, "-xs"])
