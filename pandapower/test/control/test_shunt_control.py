import pandas as pd
import pytest
from pandapower.control.controller.shunt_control import DiscreteShuntController
from pandapower.control.controller.station_control import BinarySearchControl
from pandapower.create import create_empty_network, create_buses, create_ext_grid, create_line_from_parameters, \
    create_shunt
from pandapower.run import runpp


def simple_test_net_shunt_control():
    net = create_empty_network()
    b = create_buses(net, 2, 110)
    create_ext_grid(net, b[0])
    create_line_from_parameters(net, from_bus=b[0], to_bus=b[1], length_km=50, r_ohm_per_km=0.1021,
                                   x_ohm_per_km=0.1570796,
                                   max_i_ka=0.461, c_nf_per_km=130)
    create_shunt(net, bus=b[1], q_mvar=-50, p_mw=0, step=1, max_step=5)
    return net

def test_continuous_shunt_control(tol=1e-6):
    net = simple_test_net_shunt_control()
    BinarySearchControl(net, ctrl_in_service=True, output_element='shunt', output_variable='step',
                        output_element_index=[0], output_element_in_service=[True], output_values_distribution=[1],
                        input_element='res_bus', input_variable='vm_pu', input_element_index=[1], set_point=1.08, voltage_ctrl=True,
                        tol=tol)
    runpp(net, run_control=False)
    assert (abs(net.res_bus.loc[1, "vm_pu"] - 1.041789) < tol)
    runpp(net, run_control=True)
    assert (abs(net.res_bus.loc[1, "vm_pu"] - 1.08) < tol)
    assert (abs(net.shunt.loc[0, "step"] - 2.075275) < tol)

def test_discrete_shunt_control(tol=1e-6):
    net = simple_test_net_shunt_control()
    DiscreteShuntController(net, shunt_index=0, bus_index=1, vm_set_pu=1.08, tol=1e-2)
    runpp(net, run_control=False)
    assert (abs(net.res_bus.loc[1, "vm_pu"] - 1.041789) < tol)
    runpp(net, run_control=True)
    assert (abs(net.res_bus.loc[1, "vm_pu"] - 1.077258) < tol)
    assert net.shunt.loc[0, "step"] == 2

def test_discrete_shunt_control_with_step_dependency_table(tol=1e-6):
    net = simple_test_net_shunt_control()
    net["shunt_characteristic_table"] = pd.DataFrame(
        {'id_characteristic': [0, 0, 0, 0, 0], 'step': [1, 2, 3, 4, 5], 'q_mvar': [-25, -50, -75, -100, -125],
         'p_mw': [0, 0, 0, 0, 0]})
    net.shunt.step_dependency_table.at[0] = True
    net.shunt.id_characteristic_table.at[0] = 0
    net.shunt.step.at[0] = 2

    DiscreteShuntController(net, shunt_index=0, bus_index=1, vm_set_pu=1.08, tol=1e-2)
    runpp(net, run_control=False)

    assert (abs(net.res_bus.loc[1, "vm_pu"] - 1.041789) < tol)

    runpp(net, run_control=True)
    assert (abs(net.res_bus.loc[1, "vm_pu"] - 1.077258) < tol)
    assert net.shunt.loc[0, "step"] == 4


if __name__ == '__main__':
    pytest.main(['-s', __file__])
