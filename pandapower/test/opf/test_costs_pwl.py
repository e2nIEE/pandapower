# -*- coding: utf-8 -*-

# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pytest

import pandapower as pp

try:
    import pplog as logging
except ImportError:
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
    pp.create_gen(net, 1, p_kw=100, controllable=True, min_p_kw=5, max_p_kw=150, max_q_kvar=50,
                  min_q_kvar=-50)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, p_kw=20, controllable=False)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)

    pp.create_pwl_cost(net, 0, "gen", [(0, 75, 1.5), (75, 150, 1.5)])

    pp.runopp(net, verbose=False)

    assert net["OPF_converged"]
    assert np.isclose(net.res_cost, net.res_gen.p_kw.values * 1.5, atol=1e-3)


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
    pp.create_ext_grid(net, 0, min_p_kw=0, max_p_kw=50)
    pp.create_gen(net, 1, p_kw=10, min_p_kw=0, max_p_kw=50, controllable=True)
    # pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, p_kw=20, controllable=False)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)

    pp.create_pwl_cost(net, 0, "ext_grid", [(0, 50, -10)])
#    pp.create_piecewise_linear_cost(net, 0, "ext_grid", np.array([[0, 0], [50, -500]]))
    # run OPF
    pp.runopp(net, verbose=False)

    assert net["OPF_converged"]
    assert np.isclose(net.res_cost, -10*net.res_ext_grid.p_kw.values)


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
    pp.create_gen(net, 1, p_kw=100, controllable=True, min_p_kw=5, max_p_kw=150, max_q_kvar=50,
                  min_q_kvar=-50)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, p_kw=20, controllable=False)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)

    pp.create_pwl_cost(net, 0, "gen", [(0, 150, 2)])
#    pp.create_piecewise_linear_cost(net, 0, "gen", np.array([[0, 0], [150, 300]]))
    # run OPF
    pp.runopp(net, verbose=False)

    assert net["OPF_converged"]
    assert net.res_gen.p_kw.values[0] - net.gen.min_p_kw.values[0] < 1e-2
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

    pp.create_pwl_cost(net, 0, "sgen", [(-150, -75, 1.5), (-75, 0, -1.5)])
#    pp.create_piecewise_linear_cost(net, 0, "sgen", np.array([[-150, -100], [-75, -50], [0, 0]]))
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

    pp.create_pwl_cost(net, 0, "load", [(0, 75, 1.5), (75, 150, 1.5)])

    pp.runopp(net, verbose=False)

    assert net["OPF_converged"]
    assert abs(net.res_cost - net.res_load.p_kw.values * 1.5) < 1e-3

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
    pp.create_sgen(net, 1, p_kw=100, controllable=True, min_p_kw=5, max_p_kw=150, max_q_kvar=50,
                   min_q_kvar=-50)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, p_kw=20, controllable=False)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)

    pp.create_pwl_cost(net, 0, "sgen", [(0, 75, 1.5), (75, 150, 1.5)])
#    pp.create_pwl_cost(net, 0, "sgen", np.array([[0, 0], [75, 50], [150, 200]]))
    # run OPF
    pp.runopp(net, verbose=False)

    assert net["OPF_converged"]
    assert net.res_cost - net.res_sgen.p_kw.values * 1.5 < 1e-3


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
    pp.create_load(net, 1, p_kw=50)
    pp.create_ext_grid(net, 0)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)


    pp.create_pwl_cost(net, 0, "ext_grid", [(0, 75, 1), (75, 150, 2)])

    pp.runopp(net, verbose=False)
    assert net["OPF_converged"]
    assert np.isclose(net.res_cost, net.res_ext_grid.p_kw.values[0])

    net.load.p_kw = 100
    pp.runopp(net, verbose=False)
    assert np.isclose(net.res_cost, (75 + 2*(net.res_ext_grid.p_kw.values[0] - 75)))

def test_cost_piecewise_linear_sgen_very_unsteady_slopes():
    """ Testing a very simple network for the resulting cost value
    constraints with OPF """

    # boundaries:
    vm_max = 1.5
    vm_min = 0.5

    # create net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_sgen(net, 1, p_kw=1000, controllable=True, min_p_kw=0, max_p_kw=1500,
                   max_q_kvar=50, min_q_kvar=-50)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, p_kw=20, controllable=False)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)

#    pp.create_pwl_cost(net, 0, "sgen", np.array([[0,2], [750,1 ], [1500, 2]]))
    pp.create_pwl_cost(net, 0, "sgen", [(0, 750, -1), (750, 1500, 2)])
    # run OPF
    pp.runopp(net, verbose=False)

    assert net["OPF_converged"]
    assert np.isclose(net.res_sgen.p_kw.values[0], 750)
    assert np.isclose(net.res_sgen.p_kw.values[0], -net.res_cost)


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
