import copy

import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from pandapower.create import (
    create_buses, create_svc, create_line_from_parameters, create_load, create_shunt, create_ext_grid, create_loads
)
from pandapower.network import pandapowerNet
from pandapower.networks.power_system_test_cases import case9
from pandapower.run import runpp
from pandapower.test.consistency_checks import runpp_with_consistency_checks


@pytest.mark.parametrize("vm_set_pu", [0.96, 1., 1.04])
def test_svc(vm_set_pu):
    net = case9()
    net3 = copy.deepcopy(net)

    create_svc(net, bus=3, x_l_ohm=1, x_cvar_ohm=-10, set_vm_pu=vm_set_pu, thyristor_firing_angle_degree=90)
    net2 = copy.deepcopy(net)
    runpp_with_consistency_checks(net)
    assert 90 <= net.res_svc.at[0, "thyristor_firing_angle_degree"] <= 180
    assert np.isclose(net.res_bus.at[3, 'vm_pu'], net.svc.at[0, 'set_vm_pu'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_svc.q_mvar.at[0],
                      np.square(net.res_svc.vm_pu.at[0] * net.bus.at[net.svc.bus.at[0], 'vn_kv']) /
                      net.res_svc.x_ohm.at[0])

    # try writing the values to a load and see if the effect is the same:
    lidx = create_load(net3, 3, 0, 0)
    net3.load.loc[lidx, "q_mvar"] = net.res_svc.q_mvar.at[0]
    runpp(net3)
    assert np.isclose(net3.res_bus.at[3, 'vm_pu'], net.svc.at[0, 'set_vm_pu'], rtol=0, atol=1e-6)
    assert np.isclose(net3.res_bus.at[3, 'va_degree'], net.res_svc.at[0, 'va_degree'], rtol=0, atol=1e-6)
    assert np.isclose(net3.res_bus.at[3, 'q_mvar'], net.res_bus.at[3, 'q_mvar'], rtol=0, atol=1e-6)

    net3.load.at[3, "in_service"] = False
    baseZ = np.square(net.bus.at[3, 'vn_kv']) / net.sn_mva
    x_pu = net.res_svc.x_ohm.at[0] / baseZ
    create_shunt(net3, 3, 0)
    # net3.shunt.at[0, "q_mvar"] = x_pu * net.sn_mva
    net3.shunt.at[0, "q_mvar"] = net.res_svc.q_mvar.at[0]
    net3.shunt.at[0, "vn_kv"] = net.bus.at[3, 'vn_kv'] * vm_set_pu
    runpp(net3)
    assert np.isclose(net3.res_bus.at[3, 'vm_pu'], net.svc.at[0, 'set_vm_pu'], rtol=0, atol=1e-6)
    assert np.isclose(net3.res_bus.at[3, 'va_degree'], net.res_svc.at[0, 'va_degree'], rtol=0, atol=1e-6)
    assert np.isclose(net3.res_bus.at[3, 'q_mvar'], net.res_bus.at[3, 'q_mvar'], rtol=0, atol=1e-6)

    net2.svc.at[0, "thyristor_firing_angle_degree"] = net.res_svc.thyristor_firing_angle_degree.at[0]
    net2.svc.at[0, "controllable"] = False
    runpp(net2)
    assert np.isclose(net2.res_bus.at[3, 'vm_pu'], net.svc.at[0, 'set_vm_pu'], rtol=0, atol=1e-6)
    assert np.isclose(net2.res_bus.at[3, 'q_mvar'], net.res_bus.at[3, 'q_mvar'], rtol=0, atol=1e-6)
    assert np.isclose(net2.res_svc.at[0, 'vm_pu'], net.res_svc.at[0, 'vm_pu'], rtol=0, atol=1e-6)
    assert np.isclose(net2.res_svc.at[0, 'va_degree'], net.res_svc.at[0, 'va_degree'], rtol=0, atol=1e-6)
    assert np.isclose(net2.res_svc.at[0, 'q_mvar'], net.res_svc.at[0, 'q_mvar'], rtol=0, atol=1e-6)
    assert np.isclose(net2.res_svc.at[0, 'x_ohm'], net.res_svc.at[0, 'x_ohm'], rtol=0, atol=1e-6)

    # create_svc(net, 6, 1, -10, vm_set_pu, net.res_svc.thyristor_firing_angle_degree.at[0], controllable=False)
    # create_svc(net, 7, 1, -10, vm_set_pu, 90, controllable=True)
    # runpp_with_consistency_checks(net)


@pytest.mark.parametrize("vm_set_pu", [0.96, 1., 1.04])
def test_2_svcs(vm_set_pu):
    net = pandapowerNet(name="test_2_svcs")
    create_buses(net, 3, 110)
    create_ext_grid(net, 0)
    create_line_from_parameters(net, 0, 1, 20, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 0, 2, 20, 0.0487, 0.13823, 160, 0.664)

    # both not controllable
    net1 = copy.deepcopy(net)
    create_svc(net1, bus=1, x_l_ohm=1, x_cvar_ohm=-10, set_vm_pu=vm_set_pu, thyristor_firing_angle_degree=145,
               controllable=False)
    create_svc(net1, bus=2, x_l_ohm=1, x_cvar_ohm=-10, set_vm_pu=vm_set_pu, thyristor_firing_angle_degree=145,
               controllable=False)
    runpp(net1)
    net2 = copy.deepcopy(net)
    create_loads(net2, [1, 2], 0, net1.res_svc.q_mvar.values)
    runpp(net2)
    assert_frame_equal(net1.res_bus, net2.res_bus)

    # first controllable
    net1 = copy.deepcopy(net)
    create_svc(net1, bus=1, x_l_ohm=1, x_cvar_ohm=-10, set_vm_pu=vm_set_pu, thyristor_firing_angle_degree=145)
    create_svc(net1, bus=2, x_l_ohm=1, x_cvar_ohm=-10, set_vm_pu=vm_set_pu, thyristor_firing_angle_degree=145,
               controllable=False)
    runpp(net1)
    net2 = copy.deepcopy(net)
    create_loads(net2, [1, 2], 0, net1.res_svc.q_mvar.values)
    runpp(net2)
    assert_frame_equal(net1.res_bus, net2.res_bus)

    # second controllable
    net1 = copy.deepcopy(net)
    create_svc(net1, bus=1, x_l_ohm=1, x_cvar_ohm=-10, set_vm_pu=vm_set_pu, thyristor_firing_angle_degree=145,
               controllable=False)
    create_svc(net1, bus=2, x_l_ohm=1, x_cvar_ohm=-10, set_vm_pu=vm_set_pu, thyristor_firing_angle_degree=145)
    runpp(net1)
    net2 = copy.deepcopy(net)
    create_loads(net2, [1, 2], 0, net1.res_svc.q_mvar.values)
    runpp(net2)
    assert_frame_equal(net1.res_bus, net2.res_bus)

    # both controllable
    net1 = copy.deepcopy(net)
    create_svc(net1, bus=1, x_l_ohm=1, x_cvar_ohm=-10, set_vm_pu=vm_set_pu, thyristor_firing_angle_degree=145)
    create_svc(net1, bus=2, x_l_ohm=1, x_cvar_ohm=-10, set_vm_pu=vm_set_pu, thyristor_firing_angle_degree=145)
    runpp(net1)
    net2 = copy.deepcopy(net)
    create_loads(net2, [1, 2], 0, net1.res_svc.q_mvar.values)
    runpp(net2)
    assert_frame_equal(net1.res_bus, net2.res_bus)

    # connected at ext_grid_bus - does not work
    # create_svc(net1, bus=0, x_l_ohm=1, x_cvar_ohm=-10, set_vm_pu=1, thyristor_firing_angle_degree=145, controllable=False)
    # runpp(net1)


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
