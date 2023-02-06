import pytest
import pandapower as pp
import pandapower.grid_equivalents
import pandapower.networks
from pandapower.grid_equivalents.auxiliary import drop_measurements_and_controller, \
    check_network, get_boundary_vp, adaptation_phase_shifter


def test_drop_internal_branch_elements():
    net = pp.networks.example_simple()
    pp.grid_equivalents.drop_internal_branch_elements(net, net.bus.index)
    assert not net.line.shape[0]
    assert not net.trafo.shape[0]

    net = pp.networks.example_simple()
    n_trafo = net.trafo.shape[0]
    pp.grid_equivalents.drop_internal_branch_elements(net, net.bus.index, branch_elements=["line"])
    assert not net.line.shape[0]
    assert net.trafo.shape[0] == n_trafo

    net = pp.networks.example_simple()
    n_trafo = net.trafo.shape[0]
    pp.grid_equivalents.drop_internal_branch_elements(net, [2, 3, 4, 5])
    assert set(net.line.index) == {0, 2, 3}
    assert set(net.trafo.index) == set()

    net = pp.networks.example_simple()
    n_trafo = net.trafo.shape[0]
    pp.grid_equivalents.drop_internal_branch_elements(net, [4, 5, 6])
    assert set(net.line.index) == {0}
    assert set(net.trafo.index) == {0}


def test_trafo_phase_shifter():
    net = pp.networks.create_cigre_network_mv(with_der="pv_wind")
    net.trafo.shift_degree[0] = 150
    pp.runpp(net)
    net_eq = pp.grid_equivalents.get_equivalent(net, "rei", [4, 8], [0], 
                                                retain_original_internal_indices=True)
    v, p = get_boundary_vp(net_eq, net_eq.bus_lookups)
    net.res_bus.vm_pu = net.res_bus.vm_pu.values + 1e-3
    net.res_bus.va_degree = net.res_bus.va_degree.values + 1e-3
    adaptation_phase_shifter(net, v, p)
    assert len(net.trafo) == 3


def test_drop_measurement():
    net = pp.networks.case9()
    pp.runpp(net)
    pp.create_bus(net, net.bus.vn_kv.values[0])
    pp.create_bus(net, net.bus.vn_kv.values[0])
    pp.create_bus(net, net.bus.vn_kv.values[0])
    buses = [1, 2, 5, 6, 7, 9, 10, 11]
    pp.create_transformer(net, 1, 9, "0.4 MVA 10/0.4 kV")
    pp.create_transformer3w (net, 2, 10, 11, "63/25/38 MVA 110/20/10 kV")
    net.measurement.loc[0] = ["mb", "v", "bus", 0, 1.0, 0.01, None]
    net.measurement.loc[1] = ["mb", "v", "bus", 5, 1.0, 0.01, None]
    net.measurement.loc[2] = ["mb", "i", "line", 0, 0.9, 0.01, "to"]
    net.measurement.loc[3] = ["mb", "i", "line", 3, 1.3, 0.01, "from"]
    net.measurement.loc[4] = ["mb", "p", "trafo", 0, 89.3, 0.01, "hv"]
    net.measurement.loc[5] = ["mb", "i", "trafo3w", 0, 23.56, 0.01, "mv"]
    assert len(net.measurement) == 6
    drop_measurements_and_controller(net, buses)
    assert len(net.measurement) == 2


def test_check_network():
    net = pp.networks.case9()
    net.bus.in_service[5] = False
    pp.runpp(net)
    check_network(net)
    
    net.bus.in_service[5] = True
    pp.runpp(net)
    pp.create_bus(net, net.bus.vn_kv.values[0])
    pp.create_bus(net, net.bus.vn_kv.values[0])
    pp.create_dcline(net, from_bus=4, to_bus=9, p_mw=1e4, loss_percent=1.2, loss_mw=25, \
                     vm_from_pu=1.01, vm_to_pu=1.02)
    pp.create_dcline(net, from_bus=8, to_bus=10, p_mw=1e4, loss_percent=1.2, loss_mw=25, \
                     vm_from_pu=1.01, vm_to_pu=1.02)
    check_network(net)
    assert len(net.gen) == 6


def test_check_validity():
    net = pp.networks.case9()
    pp.replace_gen_by_sgen(net)
    pp.runpp(net)
    pp.grid_equivalents.get_equivalent(net, "rei", [8, 4], [0], 
                                       retain_original_internal_indices=True,
                                       check_validity=True)


if __name__ == "__main__":
    if 0:
        pytest.main(['-x', __file__])
    else:
        # test_drop_internal_branch_elements()
        test_drop_measurement()
        # test_check_validity()
        # test_trafo_phase_shifter()
        # test_check_validity()
    pass