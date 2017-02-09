# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

import pandapower as pp
import pytest
import numpy as np
try:
    import pplog as logging
except:
    import logging

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")

def test_cost_initial():
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
                  min_q_kvar=-50, cost_per_kw=100)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, p_kw=20, controllable = False)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, imax_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100*690)
    # run OPF
    pp.runopp(net, cost_function="linear",verbose=False)
    assert net["OPF_converged"]

    # check and assert result
    logger.debug("test_costs")
    logger.debug("res_cost:\n%s" % net.res_cost)
    assert net.res_cost == 500.0003004141078

def test_cost_piecewise_linear():
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
                  min_q_kvar=-50, cost_per_kw=100)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, p_kw=20, controllable = False)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, imax_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100*690)

    pp.create_piecewise_linear_cost(net, 0, "gen", np.array([[0,0],[1,50],[2,100]]))

    logger.debug("net.piecewise_linear_cost: %s" % net.piecewise_linear_cost)

    # run OPF
    pp.runopp(net, cost_function="piecewise_linear",verbose=False)

    assert net["OPF_converged"]

    # check and assert result
    logger.debug("test_costs")
    logger.debug("res_cost:\n%s" % net.res_cost)
#    assert net.res_cost == 500.0003004141078



if __name__ == "__main__":
    # pytest.main(["test_costs.py", "-xs"])
    test_cost_piecewise_linear()