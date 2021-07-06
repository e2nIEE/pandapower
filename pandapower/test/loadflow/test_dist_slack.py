# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import copy
import os

import numpy as np
import pandas as pd
import pytest

import pandapower as pp
from pandapower.auxiliary import _check_connectivity, _add_ppc_options, lightsim2grid_available
from pandapower.networks import create_cigre_network_mv, four_loads_with_branches_out, \
    example_simple, simple_four_bus_system, example_multivoltage
from pandapower.pd2ppc import _pd2ppc
from pandapower.pf.create_jacobian import _create_J_without_numba
from pandapower.pf.run_newton_raphson_pf import _get_pf_variables_from_ppci
from pandapower.powerflow import LoadflowNotConverged
from pandapower.test.consistency_checks import runpp_with_consistency_checks
from pandapower.test.loadflow.result_test_network_generator import add_test_xward, add_test_trafo3w, \
    add_test_line, add_test_oos_bus_with_is_element, result_test_network_generator, add_test_trafo
from pandapower.test.toolbox import add_grid_connection, create_test_line, assert_net_equal
from pandapower.toolbox import nets_equal


def test_small_example():
    tol_mw = 1e-6
    net = pp.create_empty_network()
    pp.create_buses(net, 3, 20)

    pp.create_gen(net, 0, p_mw=100, vm_pu=1, slack=True)
    # pp.create_ext_grid(net, 0)

    pp.create_load(net, 1, p_mw=100, q_mvar=100)

    pp.create_line_from_parameters(net, 0, 1, length_km=1, r_ohm_per_km=0.01, x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1)
    pp.create_line_from_parameters(net, 1, 2, length_km=1, r_ohm_per_km=0.01, x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1)
    pp.create_line_from_parameters(net, 2, 0, length_km=1, r_ohm_per_km=0.01, x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1)

    # net.ext_grid["contribution_factor"] = 0
    # if no dspf is given for ext_grid, 1 is assumed, because normally
    # ext_grids are responsible to take the slack power
    net.gen["contribution_factor"] = 1

    pp.runpp(net, distributed_slack=True, numba=False)


if __name__ == "__main__":
    pytest.main([__file__])

