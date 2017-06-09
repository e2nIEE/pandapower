# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import pytest
from numpy import array

import pandapower as pp
from pandapower.test.toolbox import add_grid_connection
from pandapower.toolbox import convert_format

try:
    import pplog as logging
except:
    import logging

logger = logging.getLogger(__name__)


def test_convert_format():
    """ Testing a very simple network without transformer for voltage
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
    net.gen["cost_per_kw"] = 100
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, p_kw=20, controllable=False)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)
    # run OPF
    convert_format(net)
    pp.runopp(net, verbose=False)
    assert net["OPF_converged"]

    # check and assert result
    logger.debug("test_simplest_voltage")
    logger.debug("res_gen:\n%s" % net.res_gen)
    logger.debug("res_ext_grid:\n%s" % net.res_ext_grid)
    logger.debug("res_bus.vm_pu: \n%s" % net.res_bus.vm_pu)
    assert max(net.res_bus.vm_pu) < vm_max
    assert min(net.res_bus.vm_pu) > vm_min


def test_simplest_voltage():
    """ Testing a very simple network without transformer for voltage
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
                                   max_loading_percent=100)
    pp.create_polynomial_cost(net, 0, "gen", array([-100, 0]))
    # run OPF
    pp.runopp(net, verbose=False)
    assert net["OPF_converged"]

    # check and assert result
    logger.debug("test_simplest_voltage")
    logger.debug("res_gen:\n%s" % net.res_gen)
    logger.debug("res_ext_grid:\n%s" % net.res_ext_grid)
    logger.debug("res_bus.vm_pu: \n%s" % net.res_bus.vm_pu)
    assert max(net.res_bus.vm_pu) < vm_max
    assert min(net.res_bus.vm_pu) > vm_min

    pp.runopp(net, verbose=False, check_connectivity=True)
    assert net["OPF_converged"]

    # check and assert result
    logger.debug("test_simplest_voltage")
    logger.debug("res_gen:\n%s" % net.res_gen)
    logger.debug("res_ext_grid:\n%s" % net.res_ext_grid)
    logger.debug("res_bus.vm_pu: \n%s" % net.res_bus.vm_pu)
    assert max(net.res_bus.vm_pu) < vm_max
    assert min(net.res_bus.vm_pu) > vm_min


def test_eg_voltage():
    """ Testing a very simple network without transformer for voltage
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
    pp.create_ext_grid(net, 0, vm_pu=1.01)
    pp.create_load(net, 1, p_kw=20, controllable=False)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100)
    # run OPF
    pp.runopp(net, verbose=False)
    assert net["OPF_converged"]

    # check and assert result
    logger.debug("test_simplest_voltage")
    logger.debug("res_gen:\n%s" % net.res_gen)
    logger.debug("res_ext_grid:\n%s" % net.res_ext_grid)
    logger.debug("res_bus.vm_pu: \n%s" % net.res_bus.vm_pu)
    assert net.res_bus.vm_pu.at[0] == net.ext_grid.vm_pu.values


def test_simplest_dispatch():
    """ Testing a very simple network without transformer for voltage
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
    pp.create_polynomial_cost(net, 0, "gen", array([-100, 0]))
    pp.create_ext_grid(net, 0)
    pp.create_polynomial_cost(net, 0, "ext_grid", array([-101, 0]))
    pp.create_load(net, 1, p_kw=20, controllable=False)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)
    # run OPF
    pp.runopp(net, cost_function="linear", verbose=False)
    assert net["OPF_converged"]

    # check and assert result
    logger.debug("test_simplest_voltage")
    logger.debug("res_gen:\n%s" % net.res_gen)
    logger.debug("res_est_grid:\n%s" % net.res_ext_grid)
    logger.debug("res_bus.vm_pu: \n%s" % net.res_bus.vm_pu)
    assert max(net.res_bus.vm_pu) < vm_max
    assert min(net.res_bus.vm_pu) > vm_min


def test_opf_gen_voltage():
    """ Testing a  simple network with transformer for voltage
    constraints with OPF using a generator """

    # boundaries:
    vm_max = 1.05
    vm_min = 0.95

    # ceate net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_transformer_from_parameters(net, 0, 1, vsc_percent=3.75,
                                          tp_max=2, vn_lv_kv=0.4,
                                          shift_degree=150, tp_mid=0,
                                          vn_hv_kv=10.0, vscr_percent=2.8125,
                                          tp_pos=0, tp_side="hv", tp_min=-2,
                                          tp_st_percent=2.5, i0_percent=0.68751,
                                          sn_kva=16.0, pfe_kw=0.11, name=None,
                                          in_service=True, index=None, max_loading_percent=200)
    pp.create_gen(net, 3, p_kw=-10, controllable=True, max_p_kw=0, min_p_kw=-25, max_q_kvar=500,
                  min_q_kvar=-500)
    pp.create_polynomial_cost(net, 0, "gen", array([-10, 0]))
    pp.create_ext_grid(net, 0)
    pp.create_line_from_parameters(net, 1, 2, 1, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100000)
    pp.create_line_from_parameters(net, 2, 3, 1, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100000)

    # run OPF
    pp.runopp(net, verbose=False)
    assert net["OPF_converged"]

    # check and assert result
    logger.debug("test_opf_gen_voltage")
    logger.debug("res_gen:\n%s" % net.res_gen)
    logger.debug("res_bus.vm_pu: \n%s" % net.res_bus.vm_pu)
    assert max(net.res_bus.vm_pu) < vm_max
    assert min(net.res_bus.vm_pu) > vm_min


def test_opf_sgen_voltage():
    """ Testing a  simple network with transformer for voltage
    constraints with OPF using a static generator """

    # boundaries
    vm_max = 1.04
    vm_min = 0.96

    # create net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_transformer_from_parameters(net, 0, 1, vsc_percent=3.75,
                                          tp_max=2, vn_lv_kv=0.4,
                                          shift_degree=150, tp_mid=0,
                                          vn_hv_kv=10.0, vscr_percent=2.8125,
                                          tp_pos=0, tp_side="hv", tp_min=-2,
                                          tp_st_percent=2.5, i0_percent=0.68751,
                                          sn_kva=16.0, pfe_kw=0.11, name=None,
                                          in_service=True, index=None, max_loading_percent=1000000)
    pp.create_sgen(net, 3, p_kw=-10, controllable=True, max_p_kw=-5, min_p_kw=-15, max_q_kvar=25,
                   min_q_kvar=-25)
    pp.create_polynomial_cost(net, 0, "sgen", array([-100, 0]))
    pp.create_ext_grid(net, 0)
    pp.create_line_from_parameters(net, 1, 2, 1, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=1000000)
    pp.create_line_from_parameters(net, 2, 3, 1, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=1000000)

    # run OPF
    pp.runopp(net, verbose=False)
    assert net["OPF_converged"]

    # assert and check result
    logger.debug("test_opf_sgen_voltage")
    logger.debug("res_sgen:\n%s" % net.res_sgen)
    logger.debug("res_bus.vm_pu: \n%s" % net.res_bus.vm_pu)
    assert max(net.res_bus.vm_pu) < vm_max
    assert min(net.res_bus.vm_pu) > vm_min


def test_opf_gen_loading():
    """ Testing a  simple network with transformer for loading
    constraints with OPF using a generator """

    # wide open voltage boundaries to make sure they don't interfere with loading constraints
    vm_max = 1.5
    vm_min = 0.5
    max_line_loading = 11

    # create net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_transformer_from_parameters(net, 0, 1, vsc_percent=3.75,
                                          tp_max=2, vn_lv_kv=0.4,
                                          shift_degree=150, tp_mid=0,
                                          vn_hv_kv=10.0, vscr_percent=2.8125,
                                          tp_pos=0, tp_side="hv", tp_min=-2,
                                          tp_st_percent=2.5, i0_percent=0.68751,
                                          sn_kva=16.0, pfe_kw=0.11, name=None,
                                          in_service=True, index=None, max_loading_percent=145)
    pp.create_gen(net, 3, p_kw=-10, controllable=True, max_p_kw=-5, min_p_kw=-15, max_q_kvar=50,
                  min_q_kvar=-50)
    pp.create_polynomial_cost(net, 0, "gen", array([10, 0]))
    pp.create_ext_grid(net, 0)
    pp.create_polynomial_cost(net, 0, "ext_grid", array([-.1, 0]))
    pp.create_line_from_parameters(net, 1, 2, 1, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=max_line_loading)
    pp.create_line_from_parameters(net, 2, 3, 1, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=max_line_loading)

    # run OPF

    pp.runopp(net, verbose=False, OPF_VIOLATION=1e-1, OUT_LIM_LINE=2,
              PDIPM_GRADTOL=1e-10, PDIPM_COMPTOL=1e-10, PDIPM_COSTTOL=1e-10)
    assert net["OPF_converged"]

    # assert and check result
    logger.debug("test_opf_gen_loading")
    logger.debug("res_gen:\n%s" % net.res_gen)
    logger.debug("res_line.loading_percent:\n%s" % net.res_line.loading_percent)
    assert max(net.res_line.loading_percent) < max_line_loading
    logger.debug("res_trafo.loading_percent:\n%s" % net.res_trafo.loading_percent)
    assert max(net.res_trafo.loading_percent) < 145
    assert max(net.res_bus.vm_pu) < vm_max
    assert min(net.res_bus.vm_pu) > vm_min


def test_opf_sgen_loading():
    """ Testing a  simple network with transformer for loading
    constraints with OPF using a generator """

    # boundaries
    vm_max = 1.5
    vm_min = 0.5
    max_trafo_loading = 800
    max_line_loading = 13

    # create net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_transformer_from_parameters(net, 0, 1, vsc_percent=3.75, tp_max=2, vn_lv_kv=0.4,
                                          shift_degree=150, tp_mid=0, vn_hv_kv=10.0,
                                          vscr_percent=2.8125, tp_pos=0, tp_side="hv", tp_min=-2,
                                          tp_st_percent=2.5, i0_percent=0.68751, sn_kva=16.0,
                                          pfe_kw=0.11, name=None, in_service=True, index=None,
                                          max_loading_percent=max_trafo_loading)
    pp.create_sgen(net, 3, p_kw=-10, controllable=True, max_p_kw=-5, min_p_kw=-15, max_q_kvar=25,
                   min_q_kvar=-25)
    pp.create_polynomial_cost(net, 0, "sgen", array([10, 0]))
    pp.create_ext_grid(net, 0)
    pp.create_polynomial_cost(net, 0, "ext_grid", array([-.1, 0]))
    pp.create_line_from_parameters(net, 1, 2, 1, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=max_line_loading)
    pp.create_line_from_parameters(net, 2, 3, 1, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=max_line_loading)

    # run OPF
    pp.runopp(net, verbose=False)
    assert net["OPF_converged"]

    # assert and check result
    logger.debug("test_opf_sgen_loading")
    logger.debug("res_sgen:\n%s" % net.res_sgen)
    logger.debug("res_line.loading_percent:\n%s" % net.res_line.loading_percent)
    assert max(net.res_line.loading_percent) - max_line_loading < 1e-2
    logger.debug("res_trafo.loading_percent:\n%s" % net.res_trafo.loading_percent)
    assert max(net.res_trafo.loading_percent) < max_trafo_loading
    assert max(net.res_bus.vm_pu) < vm_max
    assert min(net.res_bus.vm_pu) > vm_min
    # check connectivity check
    pp.runopp(net, verbose=False, check_connectivity=True)

def test_unconstrained_line():
    """ Testing a very simple network without transformer for voltage
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
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876)
    pp.create_polynomial_cost(net, 0, "gen", array([-1, 0]))
    # run OPF
    pp.runopp(net, verbose=False)
    assert net["OPF_converged"]

    # check and assert result
    logger.debug("test_simplest_voltage")
    logger.debug("res_gen:\n%s" % net.res_gen)
    logger.debug("res_ext_grid:\n%s" % net.res_ext_grid)
    logger.debug("res_bus.vm_pu: \n%s" % net.res_bus.vm_pu)
    assert max(net.res_bus.vm_pu) < vm_max
    assert min(net.res_bus.vm_pu) > vm_min


def test_trafo3w_loading():
    net = pp.create_empty_network()
    b1, b2, l1 = add_grid_connection(net, vn_kv=110.)
    b3 = pp.create_bus(net, vn_kv=20.)
    b4 = pp.create_bus(net, vn_kv=10.)
    tidx = pp.create_transformer3w(
        net, b2, b3, b4, std_type='63/25/38 MVA 110/20/10 kV', max_loading_percent=120)
    pp.create_load(net, b3, 5e3, controllable=False)
    id = pp.create_load(net, b4, 5e3, controllable=True, max_p_kw=5e4, min_p_kw=0)
    pp.create_polynomial_cost(net, id, "load", array([-1, 0]))
    #pp.create_xward(net, b4, 1000, 1000, 1000, 1000, 0.1, 0.1, 1.0)
    net.trafo3w.shift_lv_degree.at[tidx] = 120
    net.trafo3w.shift_mv_degree.at[tidx] = 80

    # pp.runopp(net, calculate_voltage_angles = True)  >> Doesn't converge
    pp.runopp(net, calculate_voltage_angles=False)
    assert abs(net.res_trafo3w.loading_percent.values - 120) < 1e-3


def test_dcopf():
    # create net
    net = pp.create_empty_network()
    pp.create_bus(net, vn_kv=10.)
    pp.create_bus(net, vn_kv=.4)
    pp.create_gen(net, 1, p_kw=-100, controllable=True, max_p_kw=-5, min_p_kw=-150, max_q_kvar=50,
                  min_q_kvar=-50)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, p_kw=20, controllable=False)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100)
    pp.create_polynomial_cost(net, 0, "gen", array([-100, 0]))
    # run OPF
    pp.rundcopp(net, verbose=False)
    assert net["OPF_converged"]

    # check and assert result
    logger.debug("test_simplest_voltage")
    logger.debug("res_gen:\n%s" % net.res_gen)
    logger.debug("res_ext_grid:\n%s" % net.res_ext_grid)
    logger.debug("res_bus.vm_pu: \n%s" % net.res_bus.vm_pu)
    assert abs(100 * net.res_gen.p_kw.values - net.res_cost) < 1e-3

if __name__ == "__main__":
    # pytest.main(["-s"])
    # pytest.main(["test_basic.py", "-xs"])
    # test_simplest_dispatch()
    # test_trafo3w_loading()
    # test_trafo3w_loading()
    test_opf_sgen_loading()