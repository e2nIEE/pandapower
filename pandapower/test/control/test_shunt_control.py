import pytest
import pandapower as pp
from pandapower.control.controller.shunt_control import DiscreteShuntController
from pandapower.control.controller.station_control import BinarySearchControl

def simple_test_net_shunt_control():
    net = pp.create_empty_network()
    b = pp.create_buses(net, 2, 110)
    pp.create_ext_grid(net, b[0])
    pp.create_line_from_parameters(net, from_bus=b[0], to_bus=b[1], length_km=50, r_ohm_per_km=0.1021,
                                   x_ohm_per_km=0.1570796,
                                   max_i_ka=0.461, c_nf_per_km=130)
    pp.create_shunt(net, bus=b[1], q_mvar=-50, p_mw=0, step=1, max_step=5)
    return net

def test_continuous_shunt_control(tol=1e-6):
    net = simple_test_net_shunt_control()
    BinarySearchControl(net, ctrl_in_service=True, output_element='shunt', output_variable='step',
                        output_element_index=[0], output_element_in_service=[True], output_values_distribution=[1],
                        input_element='res_bus', input_variable='vm_pu', input_element_index=[1], set_point=1.08, voltage_ctrl=True,
                        tol=tol)
    pp.runpp(net, run_control=False)
    assert (abs(net.res_bus.loc[1, "vm_pu"] - 1.041789) < tol)
    pp.runpp(net, run_control=True)
    assert (abs(net.res_bus.loc[1, "vm_pu"] - 1.08) < tol)
    assert (abs(net.shunt.loc[0, "step"] - 2.075275) < tol)

def test_discrete_shunt_control(tol=1e-6):
    net = simple_test_net_shunt_control()
    DiscreteShuntController(net, shunt_index=0, bus_index=1, vm_set_pu=1.08, tol=1e-2)
    pp.runpp(net, run_control=False)
    assert (abs(net.res_bus.loc[1, "vm_pu"] - 1.041789) < tol)
    pp.runpp(net, run_control=True)
    assert (abs(net.res_bus.loc[1, "vm_pu"] - 1.077258) < tol)
    assert net.shunt.loc[0, "step"] == 2

if __name__ == '__main__':
    pytest.main(['-s', __file__])
