# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import copy

import numpy as np
import pytest

from pandapower.auxiliary import _check_connectivity, _add_ppc_options, LoadflowNotConverged
from pandapower.create import (create_empty_network, create_bus, create_transformer, create_transformer3w, create_load,
                               create_xward, create_switch, create_ext_grid, create_line_from_parameters, create_bus_dc,
                               create_vsc, create_line_dc_from_parameters)
from pandapower.networks.power_system_test_cases import case4gs, case118
from pandapower.pd2ppc import _pd2ppc
from pandapower.run import rundcpp, runpp
from pandapower.test.consistency_checks import rundcpp_with_consistency_checks
from pandapower.test.helper_functions import add_grid_connection, create_test_line, assert_net_equal
from pandapower.test.loadflow.result_test_network_generator import result_test_network_generator_dcpp


def test_rundcpp_init():
    net = create_empty_network()
    _, b2, _ = add_grid_connection(net)
    b3 = create_bus(net, vn_kv=0.4)
    create_transformer(net, hv_bus=b2, lv_bus=b3, std_type="0.25 MVA 20/0.4 kV")
    rundcpp(net)


def test_rundcpp_init_auxiliary_buses():
    net = create_empty_network()
    _, b2, _ = add_grid_connection(net, vn_kv=110.)
    b3 = create_bus(net, vn_kv=20.)
    b4 = create_bus(net, vn_kv=10.)
    tidx = create_transformer3w(net, b2, b3, b4, std_type='63/25/38 MVA 110/20/10 kV')
    create_load(net, b3, p_mw=5)
    create_load(net, b4, p_mw=5)
    create_xward(net, b4, 1, 1, 1, 1, 0.1, 0.1, 1.0)
    net.trafo3w.at[tidx, "shift_lv_degree"] = 120
    net.trafo3w.at[tidx, "shift_mv_degree"] = 80
    rundcpp(net)
    va = net.res_bus.va_degree.at[b2]
    rundcpp(net)
    assert np.allclose(va - net.trafo3w.shift_mv_degree.at[tidx], net.res_bus.va_degree.at[b3],
                       atol=2)
    assert np.allclose(va - net.trafo3w.shift_lv_degree.at[tidx], net.res_bus.va_degree.at[b4],
                       atol=2)


# ToDo: Bugs because of float datatypes -> Check Travis on linux machines...
def test_result_iter():
    for net in result_test_network_generator_dcpp():
        try:
            rundcpp_with_consistency_checks(net)
        except (AssertionError):
            raise UserWarning("Consistency Error after adding %s" % net.last_added_case)
        except(LoadflowNotConverged):
            raise UserWarning("Power flow did not converge after adding %s" % net.last_added_case)


def test_two_open_switches():
    net = create_empty_network()
    b1, b2, _ = add_grid_connection(net)
    b3 = create_bus(net, vn_kv=20.)
    l2 = create_test_line(net, b2, b3)
    create_test_line(net, b3, b1)
    create_switch(net, b2, l2, et="l", closed=False)
    create_switch(net, b3, l2, et="l", closed=False)
    rundcpp(net)
    assert np.isnan(net.res_line.i_ka.at[l2]) or net.res_line.i_ka.at[l2] == 0


def test_test_sn_mva():
    test_net_gen1 = result_test_network_generator_dcpp(sn_mva=1)
    test_net_gen2 = result_test_network_generator_dcpp(sn_mva=2)
    for net1, net2 in zip(test_net_gen1, test_net_gen2):
        rundcpp(net1)
        rundcpp(net2)
        try:
            assert_net_equal(net1, net2, exclude_elms=["sn_mva"])
        except:
            raise UserWarning("Result difference due to sn_mva after adding %s" % net1.last_added_case)


def test_single_bus_network():
    net = create_empty_network()
    b = create_bus(net, vn_kv=20.)
    create_ext_grid(net, b)

    runpp(net)
    assert net.converged

    rundcpp(net)
    assert net.converged


def test_missing_gen():
    net = case4gs()
    rundcpp(net)
    res_gen = copy.deepcopy(net.res_gen.values)
    net.pop("res_gen")
    rundcpp(net)
    assert np.allclose(net.res_gen.values, res_gen, equal_nan=True)


def test_res_bus_vm():
    net = case4gs()
    # run power flow to have bus vm_pu values
    runpp(net)
    # now run DC pf and check that the vm_pu values are reset to 1
    rundcpp(net)
    assert np.allclose(net.res_bus.loc[net.line.from_bus.values, "vm_pu"], net.res_line.vm_from_pu, equal_nan=True)
    assert np.allclose(net.res_bus.loc[net.line.from_bus.values, "va_degree"], net.res_line.va_from_degree,
                       equal_nan=True)
    assert np.allclose(net.res_bus.loc[net.line.to_bus.values, "vm_pu"], net.res_line.vm_to_pu, equal_nan=True)
    assert np.allclose(net.res_bus.loc[net.line.to_bus.values, "va_degree"], net.res_line.va_to_degree, equal_nan=True)


def test_dc_after_ac():
    net = case118()

    # after a dc powerflow, q_mvar is nan, which is correct
    rundcpp(net)
    assert not np.isfinite(net.res_load["q_mvar"]).any()

    # then I run an AC powerflow, q_mvar is finite for all, which is again correct
    runpp(net)
    assert np.isfinite(net.res_load["q_mvar"]).all()
    res_load = 1. * net.res_load["q_mvar"]

    # I run a second DC powerflow after the AC one, results from AC are kept
    rundcpp(net)
    assert not np.isfinite(net.res_load["q_mvar"]).any()


def test_dc_vsc():
    net = create_empty_network()
    b1 = create_bus(net, name="B1", vn_kv=380)
    b2 = create_bus(net, name="B2", vn_kv=380)
    b3 = create_bus(net, name="B3", vn_kv=380)
    b4 = create_bus(net, name="B4", vn_kv=380)

    create_ext_grid(net, bus=b1, vm_pu=1, va_degree=0)
    create_load(net, bus=b4, p_mw=100, q_mvar=0)

    create_line_from_parameters(net, name="L1", from_bus=b1, to_bus=b2, length_km=30, r_ohm_per_km=0.049,
                                x_ohm_per_km=0.136, g_us_per_km=0, c_nf_per_km=142, max_i_ka=1.5)
    create_line_from_parameters(net, name="L2", from_bus=b3, to_bus=b4, length_km=30, r_ohm_per_km=0.049,
                                x_ohm_per_km=0.136, g_us_per_km=0, c_nf_per_km=142, max_i_ka=1.5)

    # DC part
    dc_b1 = create_bus_dc(net, 380., 'DC_A')
    dc_b2 = create_bus_dc(net, 380., 'DC_B')
    create_vsc(net, b2, dc_b1, 0, 15, r_dc_ohm=0.5, control_mode_ac='vm_pu', control_value_ac=1,
               control_mode_dc="vm_pu", control_value_dc=1.)
    create_vsc(net, b3, dc_b2, 0, 15, r_dc_ohm=0.5, control_mode_ac='slack', control_value_ac=1,
               control_mode_dc="vm_pu", control_value_dc=1.)
    create_line_dc_from_parameters(net, dc_b1, dc_b2, length_km=100, r_ohm_per_km=0.0212, max_i_ka=0.963)

    rundcpp(net)
    assert True  # todo


def test_dc_vsc_p():
    net = create_empty_network()
    b1 = create_bus(net, name="B1", vn_kv=380)
    b2 = create_bus(net, name="B2", vn_kv=380)
    b3 = create_bus(net, name="B3", vn_kv=380)
    b4 = create_bus(net, name="B4", vn_kv=380)

    create_ext_grid(net, bus=b1, vm_pu=1, va_degree=0)
    create_load(net, bus=b4, p_mw=100, q_mvar=0)

    create_line_from_parameters(net, name="L1", from_bus=b1, to_bus=b2, length_km=30, r_ohm_per_km=0.049,
                                x_ohm_per_km=0.136, g_us_per_km=0, c_nf_per_km=142, max_i_ka=1.5)
    create_line_from_parameters(net, name="L2", from_bus=b3, to_bus=b4, length_km=30, r_ohm_per_km=0.049,
                                x_ohm_per_km=0.136, g_us_per_km=0, c_nf_per_km=142, max_i_ka=1.5)

    # DC part
    dc_b1 = create_bus_dc(net, 380., 'DC_A')
    dc_b2 = create_bus_dc(net, 380., 'DC_B')
    dc_b3 = create_bus_dc(net, 380., 'DC_C')
    dc_b4 = create_bus_dc(net, 380., 'DC_D')
    create_vsc(net, b2, dc_b1, 0, 15, r_dc_ohm=0.5, control_mode_ac='vm_pu', control_value_ac=1,
               control_mode_dc="vm_pu", control_value_dc=1.)
    create_vsc(net, b2, dc_b3, 0, 15, r_dc_ohm=0.5, control_mode_ac='vm_pu', control_value_ac=1,
               control_mode_dc="p_mw", control_value_dc=50)
    create_vsc(net, b3, dc_b2, 0, 15, r_dc_ohm=0.5, control_mode_ac='slack', control_value_ac=1,
               control_mode_dc="vm_pu", control_value_dc=1.)
    create_vsc(net, b3, dc_b4, 0, 15, r_dc_ohm=0.5, control_mode_ac='slack', control_value_ac=1,
               control_mode_dc="vm_pu", control_value_dc=1.)
    create_line_dc_from_parameters(net, dc_b1, dc_b2, length_km=100, r_ohm_per_km=0.0212, max_i_ka=0.963)
    create_line_dc_from_parameters(net, dc_b3, dc_b4, length_km=100, r_ohm_per_km=0.0212, max_i_ka=0.963)

    rundcpp(net)
    assert True  # todo


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
