# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import numpy as np
import pytest
from pandapower.optimal_powerflow import OPFNotConverged


import pandapower as pp

try:
    import pplog as logging
except:
    import logging

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


def test_cost_piecewise_linear_gen():
    """ Testing a very simple network for the resulting cost value
    constraints with OPF """
    # boundaries:
    vm_max = 1.05
    vm_min = 0.95

    # create net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_gen(net, 1, p_kw=-100, controllable=True, max_p_kw=-5, min_p_kw=-150, max_q_kvar=50,
                  min_q_kvar=-50)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, p_kw=20, controllable=False)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)

    pp.create_piecewise_linear_cost(net, 0, "gen", np.array([[-150, 100], [-75, 50], [0, 0]]))
    # run OPF
    pp.runopp(net, verbose=False)

    assert net["OPF_converged"]
    assert net.res_cost - net.res_gen.p_kw.values / 1.5 < 1e-3


def test_cost_piecewise_linear_eg():
    """ Testing a very simple network for the resulting cost value
    constraints with OPF """

    # boundaries:
    vm_max = 1.05
    vm_min = 0.95

    # create net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10)
    pp.create_ext_grid(net, 0, max_p_kw=0, min_p_kw=-50)
    pp.create_gen(net, 1, p_kw=-10, max_p_kw=0, min_p_kw=-50, controllable=True)
    # pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, p_kw=20, controllable=False)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)

    pp.create_piecewise_linear_cost(net, 0, "ext_grid", np.array([[-50, -500], [0, 0]]))
    # run OPF
    pp.runopp(net, verbose=False)

    assert net["OPF_converged"]
    assert net.res_cost - - net.res_ext_grid.p_kw.values * 10 < 1e-3
    # check and assert result


def test_get_costs():
    """ Testing a very simple network for the resulting cost value
    constraints with OPF """

    # boundaries:
    vm_max = 1.05
    vm_min = 0.95

    # create net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_gen(net, 1, p_kw=-100, controllable=True, max_p_kw=-5, min_p_kw=-150, max_q_kvar=50,
                  min_q_kvar=-50)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, p_kw=20, controllable=False)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)

    pp.create_piecewise_linear_cost(net, 0, "gen", np.array([[-150, 300], [0, 0]]))
    # run OPF
    pp.runopp(net, verbose=False)

    assert net["OPF_converged"]
    assert net.res_cost == 2 * net.res_gen.p_kw.values

    # check and assert result


def test_cost_piecewise_linear_sgen():
    """ Testing a very simple network for the resulting cost value
    constraints with OPF """

    # boundaries:
    vm_max = 1.05
    vm_min = 0.95

    # create net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_sgen(net, 1, p_kw=-100, controllable=True, max_p_kw=-5, min_p_kw=-150, max_q_kvar=50,
                   min_q_kvar=-50)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, p_kw=20, controllable=False)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)

    pp.create_piecewise_linear_cost(net, 0, "sgen", np.array([[-150, 100], [-75, 50], [0, 0]]))
    # run OPF
    pp.runopp(net, verbose=False)

    assert net["OPF_converged"]
    assert net.res_cost - net.res_sgen.p_kw.values / 1.5 < 1e-3


def test_cost_piecewise_linear_load():
    """ Testing a very simple network for the resulting cost value
    constraints with OPF """

    # boundaries:
    vm_max = 1.05
    vm_min = 0.95

    # create net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_load(net, 1, p_kw=100, controllable=True, max_p_kw=150, min_p_kw=50, max_q_kvar=0,
                   min_q_kvar=0)
    pp.create_ext_grid(net, 0)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)

    pp.create_piecewise_linear_cost(net, 0, "load", np.array([[0, 0], [75, 50], [150, 100]]))
    # run OPF
    pp.runopp(net, verbose=False)

    assert net["OPF_converged"]
    assert abs(net.res_cost - net.res_load.p_kw.values / 1.5) < 1e-3

def test_cost_piecewise_linear_sgen_uneven_slopes():
    """ Testing a very simple network for the resulting cost value
    constraints with OPF """

    # boundaries:
    vm_max = 1.05
    vm_min = 0.95

    # create net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_sgen(net, 1, p_kw=-100, controllable=True, max_p_kw=-5, min_p_kw=-150, max_q_kvar=50,
                   min_q_kvar=-50)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, p_kw=20, controllable=False)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)

    pp.create_piecewise_linear_cost(net, 0, "sgen", np.array([[-150, 200], [-75, 50], [0, 0]]))
    # run OPF
    pp.runopp(net, verbose=False)

    assert net["OPF_converged"]
    assert net.res_cost - net.res_sgen.p_kw.values / 1.5 < 1e-3


def test_cost_piecewise_linear_load_uneven_slopes():
    """ Testing a very simple network for the resulting cost value
    constraints with OPF """

    # boundaries:
    vm_max = 1.05
    vm_min = 0.95

    # create net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_load(net, 1, p_kw=100, controllable=True, max_p_kw=150, min_p_kw=50, max_q_kvar=0,
                   min_q_kvar=0)
    pp.create_ext_grid(net, 0)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)


    pp.create_piecewise_linear_cost(net, 0, "load", np.array([[0, 0], [75, 51], [150, 101]]))
    # run OPF
    with pytest.raises(OPFNotConverged):
        pp.runopp(net, verbose=False)
        assert net["OPF_converged"]
        assert abs(net.res_cost - net.res_load.p_kw.values / 1.5) < 1e-3

if __name__ == "__main__":
    pytest.main(["test_costs_pwl.py", "-xs"])
