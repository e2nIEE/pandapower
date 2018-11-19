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


def test_cost_mixed():
    """ Testing a very simple network for the resulting cost value
    constraints with OPF """
    # boundaries:
    vm_max = 1.05
    vm_min = 0.95

    # create net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_gen(net, 1, p_kw=-100, controllable=True, min_p_kw=5, max_p_kw=150, max_q_kvar=50,
                  min_q_kvar=-50)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, p_kw=20, controllable=False, max_q_kvar=50, max_p_kw=100, min_p_kw=50,
                   min_q_kvar=-50)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)

    # testing some combinations
#    pp.create_polynomial_cost(net, 0, "gen", np.array([0, 1, 0]))
    pp.create_poly_cost(net, 0, "gen", cp1_eur_per_kw=1)
    pp.runopp(net, verbose=False)
    assert net["OPF_converged"]
    assert net.res_cost == net.res_gen.p_kw.values[0]

    net.poly_cost.cp1_eur_per_kw.at[0] = 0
    net.poly_cost.cp2_eur_per_kw2.at[0] = 1
    pp.runopp(net, verbose=False)
    assert net["OPF_converged"]
    assert np.isclose(net.res_cost, net.res_gen.p_kw.values**2)

    net.poly_cost.cp0_eur.at[0] = 1
    pp.runopp(net, verbose=False)
    assert net["OPF_converged"]
    assert np.isclose(net.res_cost, net.res_gen.p_kw.values**2 + 1)

    net.load.controllable.at[0] = True
    pp.runopp(net, verbose=False)
    assert np.isclose(net.res_cost, net.res_gen.p_kw.values ** 2 + 1)

#    pp.create_piecewise_linear_cost(net, 0, "load", np.array([[0, 0], [100, -100]]), type="p")

#    net.poly_cost.cp1_eur_per_kw.at[0] = 1
#    net.poly_cost.cp2_eur_per_kw2.at[0] = 0
#    net.load.controllable.at[0] = True
#    pp.create_pwl_cost(net, 0, "load", [(0, 100, -1)], power_type="p")
#    pp.runopp(net, verbose=False)
#    assert np.isclose(net.res_cost, net.res_gen.p_kw.values[0] - net.res_load.p_kw.values[0] )
#    assert net.res_cost - net.res_gen.p_kw.values ** 2 - 1 - net.res_load.p_kw.values < 1e-5


def test_mixed_p_q_pol():
    vm_max = 1.05
    vm_min = 0.95

    # create net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_gen(net, 1, p_kw=-100, controllable=True, min_p_kw=5, max_p_kw=150, max_q_kvar=50,
                  min_q_kvar=-50)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, p_kw=20, controllable=False, max_q_kvar=50, max_p_kw=100, min_p_kw=50,
                   min_q_kvar=-50)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)

    # testing some combinations
#    pp.create_polynomial_cost(net, 0, "gen", cp1_eur_per_kw 1, 0]))
#    pp.create_polynomial_cost(net, 0, "gen", cp1_eur_per_kw 1, 0]), type ="q")
    pp.runopp(net, verbose=False)
    assert net["OPF_converged"]
    assert net.res_cost == net.res_gen.p_kw.values + net.res_gen.q_kvar.values


def test_mixed_p_q_pwl():
    vm_max = 1.05
    vm_min = 0.95

    # create net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_gen(net, 1, p_kw=-100, controllable=True, min_p_kw=5, max_p_kw=150, max_q_kvar=50,
                  min_q_kvar=-50)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, p_kw=20, controllable=False, max_q_kvar=50, max_p_kw=100, min_p_kw=50,
                   min_q_kvar=-50)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)

    # testing some combinations
    pp.create_piecewise_linear_cost(net, 0, "gen", np.array([[-150, -150],[150, 150]]))
    pp.create_piecewise_linear_cost(net, 0, "gen", np.array([[-150, -150],[150, 150]]), type ="q")
    pp.runopp(net, verbose=False)
    assert net["OPF_converged"]
    assert net.res_cost == net.res_gen.p_kw.values + net.res_gen.q_kvar.values

if __name__ == "__main__":
    pytest.main([])

