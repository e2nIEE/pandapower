# -*- coding: utf-8 -*-

# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pytest
import numpy as np

import pandapower as pp
from pandapower.test.toolbox import add_grid_connection
from pandapower.toolbox import convert_format

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

@pytest.fixture
def simple_opf_test_net():
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
    return net

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
    for init in ["pf", "flat"]:
        pp.runopp(net, verbose=False, init=init)
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
    pp.create_polynomial_cost(net, 0, "gen", np.array([-100, 0]))
    # run OPF
    for init in ["pf", "flat"]:
        pp.runopp(net, verbose=False, init=init)
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
    for init in ["pf", "flat"]:
        pp.runopp(net, verbose=False, init=init)
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
    pp.create_polynomial_cost(net, 0, "gen", np.array([-100, 0]))
    pp.create_ext_grid(net, 0)
    pp.create_polynomial_cost(net, 0, "ext_grid", np.array([-101, 0]))
    pp.create_load(net, 1, p_kw=20, controllable=False)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)
    # run OPF
    for init in ["pf", "flat"]:
        pp.runopp(net, cost_function="linear", verbose=False, init=init)
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
    pp.create_polynomial_cost(net, 0, "gen", np.array([-10, 0]))
    pp.create_ext_grid(net, 0)
    pp.create_line_from_parameters(net, 1, 2, 1, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100000)
    pp.create_line_from_parameters(net, 2, 3, 1, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100000)

    # run OPF
    for init in ["pf", "flat"]:
        pp.runopp(net, verbose=False, init=init)
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
    pp.create_polynomial_cost(net, 0, "sgen", np.array([-100, 0]))
    pp.create_ext_grid(net, 0)
    pp.create_line_from_parameters(net, 1, 2, 1, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=1000000)
    pp.create_line_from_parameters(net, 2, 3, 1, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=1000000)

    # run OPF
    for init in ["pf", "flat"]:
        pp.runopp(net, verbose=False, init=init)
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
    pp.create_polynomial_cost(net, 0, "gen", np.array([10, 0]))
    pp.create_ext_grid(net, 0)
    pp.create_polynomial_cost(net, 0, "ext_grid", np.array([-.1, 0]))
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
    pp.create_polynomial_cost(net, 0, "sgen", np.array([10, 0]))
    pp.create_ext_grid(net, 0)
    pp.create_polynomial_cost(net, 0, "ext_grid", np.array([-.1, 0]))
    pp.create_line_from_parameters(net, 1, 2, 1, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=max_line_loading)
    pp.create_line_from_parameters(net, 2, 3, 1, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=max_line_loading)

    # run OPF
    for init in ["pf", "flat"]:
        pp.runopp(net, verbose=False, init=init)
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
    pp.create_polynomial_cost(net, 0, "gen", np.array([-1, 0]))
    # run OPF
    for init in ["pf", "flat"]:
        pp.runopp(net, verbose=False, init=init)
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
    id = pp.create_load(net, b4, 5e3, controllable=True, max_p_kw=5e4, min_p_kw=0, min_q_kvar=-1e9, max_q_kvar= 1e9)
    pp.create_polynomial_cost(net, id, "load", np.array([-1, 0]))
    #pp.create_xward(net, b4, 1000, 1000, 1000, 1000, 0.1, 0.1, 1.0)
    net.trafo3w.shift_lv_degree.at[tidx] = 120
    net.trafo3w.shift_mv_degree.at[tidx] = 80

    # pp.runopp(net, calculate_voltage_angles = True)  >> Doesn't converge
    for init in ["pf", "flat"]:
        pp.runopp(net, calculate_voltage_angles=False, verbose=False, init=init)
        assert net["OPF_converged"]
    assert abs(net.res_trafo3w.loading_percent.values - 120) < 1e-3


def test_dcopf_poly(simple_opf_test_net):
    net = simple_opf_test_net
    pp.create_polynomial_cost(net, 0, "gen", np.array([-100, 0]))
    # run OPF
    pp.rundcopp(net, verbose=False)

    # check and assert result
    logger.debug("test_simplest_voltage")
    logger.debug("res_gen:\n%s" % net.res_gen)
    logger.debug("res_ext_grid:\n%s" % net.res_ext_grid)
    logger.debug("res_bus.vm_pu: \n%s" % net.res_bus.vm_pu)
    assert abs(100 * net.res_gen.p_kw.values - net.res_cost) < 1e-3


def test_opf_poly(simple_opf_test_net):
    net = simple_opf_test_net
    pp.create_polynomial_cost(net, 0, "gen", np.array([-100, 0]))
    # run OPF
    for init in ["pf", "flat"]:
        pp.runopp(net, verbose=False, init=init)
        assert net["OPF_converged"]

    # check and assert result
    logger.debug("test_simplest_voltage")
    logger.debug("res_gen:\n%s" % net.res_gen)
    logger.debug("res_ext_grid:\n%s" % net.res_ext_grid)
    logger.debug("res_bus.vm_pu: \n%s" % net.res_bus.vm_pu)
    assert abs(100 * net.res_gen.p_kw.values - net.res_cost) < 1e-3


def test_opf_pwl(simple_opf_test_net):
    # create net
    net = simple_opf_test_net
    # pp.create_polynomial_cost(net, 0, "gen", np.array([-100, 0]))
    pp.create_piecewise_linear_cost(net, 0, "gen", np.array([[-200, 20000], [-100, 10000], [0, 0]]))
    # run OPF
    for init in ["pf", "flat"]:
        pp.runopp(net, verbose=False, init=init)
        assert net["OPF_converged"]

    # check and assert result
    logger.debug("test_simplest_voltage")
    logger.debug("res_gen:\n%s" % net.res_gen)
    logger.debug("res_ext_grid:\n%s" % net.res_ext_grid)
    logger.debug("res_bus.vm_pu: \n%s" % net.res_bus.vm_pu)
    assert abs(100 * net.res_gen.p_kw.values - net.res_cost) < 1e-3



def test_dcopf_pwl(simple_opf_test_net):
    # create net
    net = simple_opf_test_net
    # pp.create_polynomial_cost(net, 0, "gen", np.array([-100, 0]))
    pp.create_piecewise_linear_cost(net, 0, "gen", np.array([[-200, 20000], [-100, 10000], [0, 0]]))
    # run OPF
    pp.rundcopp(net, verbose=False)
    assert net["OPF_converged"]

    # check and assert result
    logger.debug("test_simplest_voltage")
    logger.debug("res_gen:\n%s" % net.res_gen)
    logger.debug("res_ext_grid:\n%s" % net.res_ext_grid)
    logger.debug("res_bus.vm_pu: \n%s" % net.res_bus.vm_pu)
    assert abs(100 * net.res_gen.p_kw.values - net.res_cost) < 1e-3

def test_opf_varying_max_line_loading():
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



    pp.create_sgen(net, 3, p_kw=-100, controllable=True, max_p_kw=-5, min_p_kw=-150, max_q_kvar=25,
                   min_q_kvar=-25)
    pp.create_sgen(net, 2, p_kw=-100, controllable=True, max_p_kw=-5, min_p_kw=-150, max_q_kvar=25,
                   min_q_kvar=-25)
    pp.create_polynomial_cost(net, 0, "sgen", np.array([-10, 0]))
    pp.create_polynomial_cost(net, 1, "sgen", np.array([-10, 0]))
    pp.create_ext_grid(net, 0)
    pp.create_polynomial_cost(net, 0, "ext_grid", np.array([-.1, 0]))
    pp.create_line_from_parameters(net, 1, 2, 1, name="line1", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.200, x_ohm_per_km=0.1159876,
                                   max_loading_percent=20)
    pp.create_line_from_parameters(net, 1, 3, 1, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.100, x_ohm_per_km=0.1159876,
                                   max_loading_percent=10)

    # run OPF
    pp.runopp(net, verbose=False, init="flat")
    assert net["OPF_converged"]

    assert sum(net["_ppc"]["branch"][:, 5] - np.array([ 0.02771281+0.j,  0.00692820+0.j,  0.12800000+0.j])) < 1e-8


    # assert and check result
    logger.debug("test_opf_sgen_loading")
    logger.debug("res_sgen:\n%s" % net.res_sgen)
    logger.debug("res_line.loading_percent:\n%s" % net.res_line.loading_percent)
    assert net.res_line.loading_percent.at[0] - 20 < 1e-2
    logger.debug("res_line.loading_percent:\n%s" % net.res_line.loading_percent)
    assert net.res_line.loading_percent.at[1] - 10 < 1e-2


def test_storage_opf():
    """ Testing a simple network with storage to ensure the correct behaviour
    of the storage OPF-Functions """
    
    # boundaries
    vm_max = 1.1
    vm_min = 0.9
    max_line_loading_percent = 100
    
    # create network
    net = pp.create_empty_network()
    
    b1 = pp.create_bus(net, vn_kv=0.4, max_vm_pu=vm_max, min_vm_pu=vm_min)
    b2 = pp.create_bus(net, vn_kv=0.4, max_vm_pu=vm_max, min_vm_pu=vm_min)
    
    pp.create_line(net, b1, b2, length_km=5, std_type="NAYY 4x50 SE", 
                   max_loading_percent=max_line_loading_percent)
    
    # test elements static
    pp.create_ext_grid(net, b2)
    pp.create_load(net, b1, p_kw=7.5, controllable=False)
    pp.create_sgen(net, b1, p_kw=-25, controllable=True, max_p_kw=-10, min_p_kw=-25,
                   max_q_kvar=25, min_q_kvar=-25)
    
    # test elements 
    pp.create_storage(net, b1, p_kw=-25, max_e_kwh=50, controllable=True, max_p_kw=0,
                      min_p_kw=-25, max_q_kvar=25, min_q_kvar=-25)
    pp.create_sgen(net, b1, p_kw=-25, controllable=True, max_p_kw=0, min_p_kw=-25,
                   max_q_kvar=25, min_q_kvar=-25)
    pp.create_load(net, b1, p_kw=25, controllable=True, max_p_kw=25, min_p_kw=0,
                   max_q_kvar=25, min_q_kvar=-25)

    # costs
    pp.create_polynomial_cost(net, 0, "ext_grid", np.array([0, 3, 0]))
    pp.create_polynomial_cost(net, 0, "sgen", np.array([0, 2, 0]))
    pp.create_polynomial_cost(net, 0, "storage", np.array([0, 1, 0]))
    pp.create_polynomial_cost(net, 1, "sgen", np.array([0, 1, 0]))
    pp.create_polynomial_cost(net, 1, "load", np.array([0, -3, 0]))

    # test storage generator behaviour
    net["storage"].in_service.iloc[0] = True
    net["storage"].p_kw.iloc[0] = -25
    net["sgen"].in_service.iloc[1] = False    
    net["load"].in_service.iloc[1] = False
    
    pp.runopp(net, verbose=False)
    assert net["OPF_converged"]
    
    res_stor_p_kw = net["res_storage"].p_kw.iloc[0]
    res_stor_q_kvar = net["res_storage"].q_kvar.iloc[0]
    res_cost_stor = net["res_cost"]

    net["storage"].in_service.iloc[0] = False
    net["storage"].p_kw.iloc[0] = -25
    net["sgen"].in_service.iloc[1] = True
    net["load"].in_service.iloc[1] = False    
    
    pp.runopp(net, verbose=False)
    assert net["OPF_converged"]
    
    res_sgen_p_kw = net["res_sgen"].p_kw.iloc[1]
    res_sgen_q_kvar = net["res_sgen"].q_kvar.iloc[1]
    res_cost_sgen = net["res_cost"]
    
    # assert storage generator behaviour
    assert np.isclose(res_stor_p_kw, res_sgen_p_kw)
    assert np.isclose(res_stor_q_kvar, res_sgen_q_kvar)
    assert np.isclose(res_cost_stor, res_cost_sgen)
    
    # test storage load behaviour
    net["storage"].in_service.iloc[0] = True
    net["storage"].p_kw.iloc[0] = 25
    net["storage"].max_p_kw.iloc[0] = 25
    net["storage"].min_p_kw.iloc[0] = 0
    net["storage"].max_q_kvar.iloc[0] = 25
    net["storage"].min_q_kvar.iloc[0] = -25
    # gencost for storages: positive costs in pandapower per definition
    # --> storage gencosts are similar to sgen gencosts (make_objective.py, l.128ff. and l.185ff.)
    net["polynomial_cost"].c.iloc[2] = - net["polynomial_cost"].c.iloc[4]
    net["sgen"].in_service.iloc[1] = False
    net["load"].in_service.iloc[1] = False 
    
    pp.runopp(net, verbose=False)
    assert net["OPF_converged"]
    
    res_stor_p_kw = net["res_storage"].p_kw.iloc[0]
    res_stor_q_kvar = net["res_storage"].q_kvar.iloc[0]
    res_cost_stor = net["res_cost"]
    
    net["storage"].in_service.iloc[0] = False
    net["storage"].p_kw.iloc[0] = 25
    net["sgen"].in_service.iloc[1] = False
    net["load"].in_service.iloc[1] = True 
    
    pp.runopp(net, verbose=False)
    assert net["OPF_converged"]
    
    res_load_p_kw = net["res_load"].p_kw.iloc[1]
    res_load_q_kvar = net["res_load"].q_kvar.iloc[1]
    res_cost_load = net["res_cost"]
    
    # assert storage load behaviour
    assert np.isclose(res_stor_p_kw, res_load_p_kw)
    assert np.isclose(res_stor_q_kvar, res_load_q_kvar)
    assert np.isclose(res_cost_stor, res_cost_load)


def test_opf_no_controllables_vs_pf():
    """ Comparing the calculation results of PF and OPF in a simple network with non-controllable
     elements """

    # boundaries
    vm_max = 1.3
    vm_min = 0.9
    max_line_loading_percent = 100

    # create network
    net = pp.create_empty_network()

    b1 = pp.create_bus(net, vn_kv=0.4, max_vm_pu=vm_max, min_vm_pu=vm_min)
    b2 = pp.create_bus(net, vn_kv=0.4, max_vm_pu=vm_max, min_vm_pu=vm_min)

    pp.create_line(net, b1, b2, length_km=5, std_type="NAYY 4x50 SE",
                   max_loading_percent=max_line_loading_percent)

    # test elements static
    pp.create_ext_grid(net, b2)
    pp.create_load(net, b1, p_kw=7.5, controllable=False)
    pp.create_sgen(net, b1, p_kw=-25, controllable=False, max_p_kw=-10, min_p_kw=-25,
                   max_q_kvar=25, min_q_kvar=-25)

    # testing cost assignment (for non-controllable elements - see Gitlab Issue #27)
    pp.create_polynomial_cost(net, 0, "ext_grid", np.array([0, 3, 0]))
    pp.create_polynomial_cost(net, 0, "load", np.array([0, -3, 0]))
    pp.create_polynomial_cost(net, 0, "sgen", np.array([0, 2, 0]))

    # do calculations
    pp.runopp(net, verbose=False)
    assert net["OPF_converged"]

    res_opf_line_loading = net.res_line.loading_percent
    res_opf_bus_voltages = net.res_bus.vm_pu

    pp.runpp(net, verbose=False)
    assert net["converged"]

    res_pf_line_loading = net.res_line.loading_percent
    res_pf_bus_voltages = net.res_bus.vm_pu

    # assert calculation behaviour
    assert np.isclose(res_opf_line_loading, res_pf_line_loading).all()
    assert np.isclose(res_opf_bus_voltages, res_pf_bus_voltages).all()


if __name__ == "__main__":
    #pytest.main(['-s', __file__])
    #test_storage_opf()
    test_opf_no_controllables_vs_pf()
    #test_opf_varying_max_line_loading()
     # pytest.main(["test_basic.py", "-s"])
    # test_simplest_dispatch()
    # test_trafo3w_loading()
    # test_trafo3w_loading()
    # test_dcopf_pwl()