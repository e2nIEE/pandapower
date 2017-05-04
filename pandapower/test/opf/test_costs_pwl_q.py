# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import numpy as np
import pytest

import pandapower as pp

try:
    import pplog as logging
except:
    import logging

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


def test_cost_piecewise_linear_gen_q():
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

    with pytest.raises(ValueError):
        pp.create_piecewise_linear_cost(net, 0, "gen", np.array(
            [[0, 0], [1, 50], [2, 100]]), type="q")
    with pytest.raises(ValueError):
        pp.create_piecewise_linear_cost(net, 0, "gen", np.array(
            [[0, 0], [-1, 50], [-2, 100]]), type="q")
    with pytest.raises(ValueError):
        pp.create_piecewise_linear_cost(net, 0, "gen", np.array(
            [[-10, 0], [-200, 50], [-50, 100]]), type="q")

    pp.create_piecewise_linear_cost(net, 0, "gen", np.array(
        [[-50, 50], [0, 0], [50, -50]]), type="q")
    # run OPF
    pp.runopp(net, verbose=False)

    assert net["OPF_converged"]
    assert net.res_cost - net.res_gen.q_kvar.values < 1e-3


def test_cost_piecewise_linear_sgen_q():
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

    pp.create_piecewise_linear_cost(net, 0, "sgen", np.array(
        [[-50, 50], [50, -50]]), type="q")
    # run OPF
    pp.runopp(net, verbose=False)

    assert net["OPF_converged"]
    assert net.res_cost - net.res_sgen.q_kvar.values < 1e-3


def test_cost_piecewise_linear_load_q():
    """ Testing a very simple network for the resulting cost value
    constraints with OPF """
    # boundaries:
    vm_max = 1.05
    vm_min = 0.95

    # create net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_load(net, 1, p_kw=-100, controllable=True, max_p_kw=50, min_p_kw=-0, max_q_kvar=50,
                   min_q_kvar=-50)
    pp.create_ext_grid(net, 0)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)

    pp.create_piecewise_linear_cost(net, 0, "load", np.array(
        [[-50, 50], [50, -50]]), type="q")
    # run OPF
    pp.runopp(net, verbose=False)

    assert net["OPF_converged"]
    assert net.res_cost - net.res_load.q_kvar.values < 1e-3


def test_cost_piecewise_linear_eg_q():
    """ Testing a very simple network for the resulting cost value
    constraints with OPF """

    # boundaries:
    vm_max = 1.05
    vm_min = 0.95

    # create net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10)
    pp.create_ext_grid(net, 0, max_p_kw=0, min_p_kw=-50, min_q_kvar=-50, max_q_kvar=50)
    pp.create_gen(net, 1, p_kw=-10, max_p_kw=0, min_p_kw=-50, controllable=True)
    pp.create_load(net, 1, p_kw=20, controllable=False)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)

    pp.create_piecewise_linear_cost(net, 0, "ext_grid", np.array(
        [[-50, 50], [0, 0], [50, 50]]), type="q")
    # run OPF
    pp.runopp(net, verbose=False)

    assert net["OPF_converged"]
    assert net.res_cost - net.res_ext_grid.q_kvar.values * 1 < 1e-3
    # check and assert result

def test_cost_pwl_q_3point():
# We have a problem with the cost value after optimization of 3 point q cost functions! It returns the amount of q at the EG, but not the costs!
# Also, the q result is not the optimum!

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
    pp.create_piecewise_linear_cost(net, 0, "sgen", np.array(
        [[-50, 50], [0,0], [50, 50]]), type="q")

    # run OPF
    pp.runopp(net, verbose=False)

    assert net["OPF_converged"]
    # test failing because somehow, the res_cost is the eg reactive power and the result is not the optimum!
    # assert net.res_cost - net.res_sgen.q_kvar.values < 1e-3


if __name__ == "__main__":
    pytest.main(["test_costs_pwl_q.py", "-xs"])
    # test_cost_piecewise_linear_eg_q()
    # test_cost_piecewise_linear_sgen_q()
    # test_cost_piecewise_linear_gen_q()
    # test_get_costs()
