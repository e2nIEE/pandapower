import copy

import numpy as np
import pytest

from pandapower.create import create_buses, create_bus, create_empty_network, create_line_from_parameters, \
    create_load, create_ext_grid, create_ssc

from pandapower.run import runpp
from pandapower.test.consistency_checks import runpp_with_consistency_checks
from pandapower.test.loadflow.test_facts import copy_with_impedance, facts_case_study_grid, compare_ssc_impedance_gen


def test_ssc_minimal():
    net = create_empty_network()
    create_bus(net, 110)
    create_ext_grid(net, 0)
    create_ssc(net, 0, 0, 5, 1)
    runpp_with_consistency_checks(net)

    net_ref = copy_with_impedance(net)
    runpp(net_ref)

    ### compare (ssc) to bus 1(net)

    assert np.isclose(net.res_bus.at[0, "vm_pu"], net.ssc.set_vm_pu.at[0], rtol=0, atol=1e-6)
    assert np.isclose(np.abs(net._ppc["internal"]["V"][-1]), net.res_ssc.vm_internal_pu.at[0], rtol=0, atol=1e-6)

    assert np.isclose(net.res_ssc.vm_pu[0], net.res_bus.vm_pu.at[0], rtol=0, atol=1e-6)
    assert np.isclose(net.res_ssc.va_degree[0], net.res_bus.va_degree.at[0], rtol=0, atol=1e-6)

    compare_ssc_impedance_gen(net, net_ref)

    assert np.isclose(net.res_bus.q_mvar[0], net_ref.res_bus.q_mvar.at[0], rtol=0, atol=1e-6)
    assert np.isclose(net.res_ssc.q_mvar[0], net_ref.res_impedance.q_from_mvar.at[0], rtol=0, atol=1e-6)


def test_ssc_controllable():
    net = create_empty_network()
    create_buses(net, 3, 110)
    create_ext_grid(net, 0)
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 0, 2, 30, 0.0487, 0.13823, 160, 0.664)

    z_base = np.square(110) / net.sn_mva
    x = 5
    # both not controllable
    net1 = copy.deepcopy(net)
    create_ssc(net1, 1, 0, x, 1, controllable=False)
    create_ssc(net1, 2, 0, x, 1)
    runpp_with_consistency_checks(net1)
    assert np.isclose(net1.res_ssc.vm_internal_pu.at[0], 1, rtol=0, atol=1e-6)
    assert np.isclose(net1.res_ssc.vm_pu.at[1], 1, rtol=0, atol=1e-6)

    net2 = copy.deepcopy(net)
    create_ssc(net2, 1, 0, x, 1, controllable=False, vm_internal_pu=1.02, va_internal_degree=150)
    runpp_with_consistency_checks(net2)
    assert np.isclose(net2.res_ssc.vm_internal_pu, 1.02, rtol=0, atol=1e-6)


def test_ssc_case_study():
    net = facts_case_study_grid()

    create_ssc(net, bus=6, r_ohm=0, x_ohm=5, set_vm_pu=1, controllable=True)
    # create_svc(net, 6, 1, -10, 1., 90,controllable=True)
    # net.res_ssc.q_mvar = -9.139709

    runpp_with_consistency_checks(net)


def test_2_sscs():
    net = create_empty_network()
    create_buses(net, 3, 110)
    create_ext_grid(net, 0)
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 0, 2, 30, 0.0487, 0.13823, 160, 0.664)

    z_base = np.square(110) / net.sn_mva
    x = 5
    # both not controllable
    net1 = copy.deepcopy(net)
    create_ssc(net1, 1, 0, x, 1, controllable=True)
    create_ssc(net1, 2, 0, x, 1, controllable=True)
    runpp_with_consistency_checks(net1)

    net2 = copy_with_impedance(net1)
    runpp(net2)
    compare_ssc_impedance_gen(net1, net2)

    # first controllable
    net1 = copy.deepcopy(net)
    create_ssc(net1, 1, 0, x, 1, in_service=False, controllable=False)
    create_ssc(net1, 2, 0, x, 1, in_service=False, controllable=False)
    runpp(net1)
    net2 = copy_with_impedance(net1)
    runpp(net2)
    compare_ssc_impedance_gen(net1, net2)

    return
    # todo:
    # # second controllable
    # net1 = copy.deepcopy(net)
    # create_ssc(net1, 1, 0, 121/z_base, 1, controllable=False)
    # create_ssc(net1, 2, 0, 121/z_base, 1, controllable=True)
    #
    # runpp(net1)
    # net2 = copy.deepcopy(net)
    # create_load(net2, [1, 2], 100, 25)
    # runpp(net2)
    # assert_frame_equal(net1.res_bus, net2.res_bus)
    #
    # # both controllable
    # net1 = copy.deepcopy(net)
    # create_ssc(net1, 1, 0, 121/z_base, 1, controllable=True)
    # create_ssc(net1, 2, 0, 121/z_base, 1, controllable=True)
    # runpp(net1)
    # net2 = copy.deepcopy(net)
    # create_load(net2, [1, 2], 100, 25)
    # runpp(net2)
    # assert_frame_equal(net1.res_bus, net2.res_bus)


def test_ssc_simple():
    net = create_empty_network()
    create_buses(net, 2, 110)
    create_ext_grid(net, 0)
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_load(net, 1, 100, 25)
    create_ssc(net, 1, 0, 5, 1)
    runpp_with_consistency_checks(net)

    net_ref = copy_with_impedance(net)
    runpp(net_ref)

    ### compare (ssc) to bus 1(net)

    assert np.isclose(net.res_bus.at[1, "vm_pu"], net.ssc.set_vm_pu.at[0], rtol=0, atol=1e-6)
    assert np.isclose(np.abs(net._ppc["internal"]["V"][-1]), net.res_ssc.vm_internal_pu.at[0], rtol=0, atol=1e-6)

    assert np.isclose(net.res_ssc.vm_pu[0], net.res_bus.vm_pu.at[1], rtol=0, atol=1e-6)
    assert np.isclose(net.res_ssc.va_degree[0], net.res_bus.va_degree.at[1], rtol=0, atol=1e-6)

    compare_ssc_impedance_gen(net, net_ref)

    assert np.isclose(net.res_bus.q_mvar[0], net_ref.res_bus.q_mvar.at[0], rtol=0, atol=1e-6)
    assert np.isclose(net.res_ssc.q_mvar[0], net.res_bus.q_mvar.at[1] - net.load.q_mvar.at[0], rtol=0, atol=1e-6)
    assert np.isclose(net.res_ssc.q_mvar[0], net_ref.res_impedance.q_from_mvar.at[0], rtol=0, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
