# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import copy
import json
import os
import numpy as np
import pytest
import pandapower.control
import pandapower.timeseries
import pandapower as pp
from pandapower.converter.pandamodels.from_pm import read_pm_results_to_net
from pandapower.converter.pandamodels.to_pm import init_ne_line
from pandapower.pd2ppc import _pd2ppc
from pandapower.test.consistency_checks import consistency_checks
from pandapower.test.helper_functions import add_grid_connection, create_test_line
from pandapower.converter import convert_pp_to_pm
from pandapower.test.opf.test_basic import simple_opf_test_net, net_3w_trafo_opf

try:
    from julia.core import UnsupportedPythonError
except ImportError:
    UnsupportedPythonError = Exception
try:
    from julia import Main

    julia_installed = True
except (ImportError, RuntimeError, UnsupportedPythonError) as e:
    julia_installed = False


def test_pp_to_pm_conversion(net_3w_trafo_opf):
    # tests if the conversion to power models works
    net = net_3w_trafo_opf
    pm_S = convert_pp_to_pm(net)
    pm_I = convert_pp_to_pm(net, opf_flow_lim="I")


def test_pm_to_pp_conversion(simple_opf_test_net):
    # this tests checks if the runopp results are the same as the ones from powermodels.
    # Results are read from a result file containing the simple_opf_test_net

    net = simple_opf_test_net
    pp.create_poly_cost(net, 0, "gen", cp1_eur_per_mw=100)

    # get pandapower opf results
    pp.runopp(net, delta=1e-13)
    va_degree = copy.deepcopy(net.res_bus.va_degree)
    vm_pu = copy.deepcopy(net.res_bus.vm_pu)

    # get previously calculated power models results
    pm_res_file = os.path.join(os.path.abspath(os.path.dirname(pp.test.__file__)),
                               "test_files", "pm_example_res.json")

    with open(pm_res_file, "r") as fp:
        result_pm = json.load(fp)
    net._options["correct_pm_network_data"] = True
    ppc, ppci = _pd2ppc(net)
    read_pm_results_to_net(net, ppc, ppci, result_pm)
    assert np.allclose(net.res_bus.vm_pu, vm_pu, atol=1e-4)
    assert np.allclose(net.res_bus.va_degree, va_degree, atol=1e-2, rtol=1e-2)


def test_obj_factors(net_3w_trafo_opf):
    net = net_3w_trafo_opf
    net["obj_factors"] = [0.9, 0.1]
    pm = convert_pp_to_pm(net)
    assert pm["user_defined_params"]["obj_factors"]["fac_1"] == 0.9
    assert pm["user_defined_params"]["obj_factors"]["fac_2"] == 0.1
    assert pm["user_defined_params"]["gen_and_controllable_sgen"]["2"] == 2
    assert pm["user_defined_params"]["gen_and_controllable_sgen"]["3"] == 3


if __name__ == '__main__':
    if 1:
        pytest.main(['-x', __file__])
    else:
        test_obj_factors()

    pass
