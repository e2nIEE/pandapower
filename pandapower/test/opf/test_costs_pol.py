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


def test_cost_pol_gen():
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

    pp.create_polynomial_cost(net, 0, "gen", np.array([0, 1, 0]))
    # run OPF
    pp.runopp(net, verbose=False)

    assert net["OPF_converged"]
    assert net.res_cost == - net.res_gen.p_kw.values

    net.polynomial_cost.c.at[0] = np.array([[1, 0, 0]])
    # run OPF
    pp.runopp(net, verbose=False)

    assert net["OPF_converged"]
    assert abs(net.res_cost - net.res_gen.p_kw.values**2) < 1e-5


def test_cost_pol_all_elements():
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
    pp.create_sgen(net, 1, p_kw=-100, controllable=True, max_p_kw=-5, min_p_kw=-150, max_q_kvar=50,
                   min_q_kvar=-50)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, p_kw=20, controllable=False)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)

    pp.create_polynomial_cost(net, 0, "gen", np.array([0, 1, 0]))
    pp.create_polynomial_cost(net, 0, "sgen", np.array([0, 1, 0]))
    # run OPF
    pp.runopp(net, verbose=False)

    assert net["OPF_converged"]
    assert abs(net.res_cost + (net.res_gen.p_kw.values + net.res_sgen.p_kw.values)) < 1e-2

    net.polynomial_cost.c.at[0] = np.array([[1, 0, 0]])
    # run OPF
    pp.runopp(net, verbose=False)

    assert net["OPF_converged"]
    assert abs(net.res_cost - net.res_gen.p_kw.values**2 + net.res_sgen.p_kw.values) < 1e-5

def test_cost_pol_q():
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

    pp.create_polynomial_cost(net, 0, "sgen", np.array([0, -1, 0]), type="q")
    # run OPF
    pp.runopp(net, verbose=False)

    assert net["OPF_converged"]
    assert abs(net.res_cost + (net.res_sgen.q_kvar.values)) < 1e-2

    net.polynomial_cost.c.at[0] = np.array([[1, 0, 0]])
    # run OPF
    pp.runopp(net, verbose=False)

    assert net["OPF_converged"]
    assert abs(net.res_cost - net.res_sgen.q_kvar.values**2) < 1e-5



if __name__ == "__main__":
       pytest.main(["test_costs_pol.py", "-xs"])