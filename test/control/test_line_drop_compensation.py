# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import copy

import numpy as np
import pandas as pd
import pytest

from pandapower.control import (
    ContinuousTapControl,
    ContinuousTapControlLDC,
    DiscreteTapControl,
    DiscreteTapControlLDC,
)
from pandapower.control.controller.trafo.line_drop_compensation import LineDropCompensation
from pandapower.create import (
    create_buses,
    create_ext_grid,
    create_impedance,
    create_lines,
    create_load,
    create_switch,
    create_transformer,
    create_transformer3w,
    create_transformers,
)
from pandapower.file_io import from_json_string, to_json
from pandapower.network import pandapowerNet
from pandapower.networks import simple_four_bus_system
from pandapower.run import runpp, set_user_pf_options


def _radial_feeder(tap_pos=0, tap_min=-9, tap_max=9, p_mw=0.03, q_mvar=0.01, n_lines=3):
    """
    Simple radial LV feeder: ext_grid -> trafo -> three lines -> load at the far end.
    """
    net = pandapowerNet(name="radial_feeder_ldc")
    create_buses(net, 1, 20)
    create_buses(net, n_lines + 1, 0.4)
    create_ext_grid(net, 0)
    create_transformer(net, 0, 1, "0.4 MVA 20/0.4 kV")
    create_lines(net, list(range(1, n_lines + 1)), list(range(2, n_lines + 2)), 0.3, "NAYY 4x50 SE")
    create_load(net, n_lines + 1, p_mw, q_mvar)
    net.trafo[["tap_side", "tap_neutral", "tap_min", "tap_max", "tap_step_percent", "tap_pos"]] = [
        "lv",
        0,
        tap_min,
        tap_max,
        2.5,
        tap_pos,
    ]
    set_user_pf_options(net, init="dc", calculate_voltage_angles=True)
    runpp(net)
    return net


def _vectorized_feeder(n_trafos=3, tap_pos=0, tap_min=-5, tap_max=5):
    """
    Several identical transformers at the same HV bus, each supplying one LV feeder with a load at the far end.
    """
    net = pandapowerNet(name="vectorized_ldc")
    create_buses(net, 1, 20)
    create_buses(net, n_trafos, 0.4)
    create_ext_grid(net, 0)
    remote_buses = []
    for lv_bus in range(1, n_trafos + 1):
        create_transformer(net, 0, lv_bus, "0.25 MVA 20/0.4 kV")
        remote_bus = create_buses(net, 1, 0.4)[0]
        create_lines(net, lv_bus, remote_bus, 0.3, "NAYY 4x50 SE")
        create_load(net, remote_bus, 0.05, 0.02)
        remote_buses.append(remote_bus)
    net.trafo[["tap_side", "tap_neutral", "tap_min", "tap_max", "tap_step_percent", "tap_pos"]] = [
        "lv",
        0,
        tap_min,
        tap_max,
        2.5,
        tap_pos,
    ]
    set_user_pf_options(net, init="dc", calculate_voltage_angles=True)
    runpp(net)
    return net, remote_buses


def test_line_drop_compensation_voltage_drop():
    """
    The compensated voltage has to match the actual voltage at the remote bus in a radial feeder.
    """
    net = _radial_feeder()
    c = ContinuousTapControlLDC(net, 0, vm_set_pu=1.0, calculate_impedance=True, remote_bus=4)

    # the fictitious line consists of the three lines between the transformer and the remote bus
    r_expected = net.line.r_ohm_per_km.sum() * 0.3
    x_expected = net.line.x_ohm_per_km.sum() * 0.3
    assert np.isclose(c.r_ohm, r_expected)
    assert np.isclose(c.x_ohm, x_expected)

    # without control, the compensated voltage is close to the voltage at the remote bus
    assert np.isclose(c._get_vm_pu(net), net.res_bus.vm_pu.at[4], atol=2e-3)


def test_continuous_tap_control_ldc():
    """
    The continuous controller with LDC has to regulate the compensated (remote) voltage to the set point.
    """
    net = _radial_feeder()
    vm_remote_without_control = net.res_bus.vm_pu.at[4]
    c = ContinuousTapControlLDC(net, 0, vm_set_pu=1.0, calculate_impedance=True, remote_bus=4)

    runpp(net, run_control=True)
    runpp(net)

    assert np.isclose(c._get_vm_pu(net), 1.0, atol=1e-3)
    assert abs(net.res_bus.vm_pu.at[4] - 1.0) < abs(vm_remote_without_control - 1.0)
    assert net.trafo.tap_pos.at[0] > 0


def test_continuous_tap_control_ldc_manual_impedance_matches_topology():
    """
    Calculating the impedance from the topology has to lead to the same result as providing it manually.
    """
    net_topology = _radial_feeder()
    net_manual = _radial_feeder()
    ContinuousTapControlLDC(net_topology, 0, vm_set_pu=1.0, calculate_impedance=True, remote_bus=4)
    r_ohm = net_topology.line.r_ohm_per_km.sum() * 0.3
    x_ohm = net_topology.line.x_ohm_per_km.sum() * 0.3
    ContinuousTapControlLDC(net_manual, 0, vm_set_pu=1.0, r_ohm=r_ohm, x_ohm=x_ohm)

    runpp(net_topology, run_control=True)
    runpp(net_manual, run_control=True)

    assert np.isclose(net_topology.trafo.tap_pos.at[0], net_manual.trafo.tap_pos.at[0])
    assert np.isclose(net_topology.res_bus.vm_pu.at[4], net_manual.res_bus.vm_pu.at[4])


def test_continuous_tap_control_ldc_partial_compensation():
    """
    Only a part of the feeder can be compensated by reducing the fictitious line impedance.
    """
    net = _radial_feeder()
    ContinuousTapControlLDC(net, 0, vm_set_pu=1.0, r_ohm=0.0, x_ohm=0.0)
    runpp(net, run_control=True)
    runpp(net)
    # without compensation, the local voltage at the transformer bus is regulated
    assert np.isclose(net.res_bus.vm_pu.at[1], 1.0, atol=1e-3)
    assert net.res_bus.vm_pu.at[4] < net.res_bus.vm_pu.at[1]

    net = _radial_feeder()
    ContinuousTapControlLDC(net, 0, vm_set_pu=1.0, r_ohm=net.line.r_ohm_per_km.sum() * 0.15)
    runpp(net, run_control=True)
    runpp(net)
    # the voltage at the transformer bus is above the set point, but lower than with full compensation
    assert 1.0 < net.res_bus.vm_pu.at[1]


def test_discrete_tap_control_ldc():
    """
    The discrete controller with LDC has to bring the compensated (remote) voltage into the voltage band.
    """
    net = _radial_feeder()
    c = DiscreteTapControlLDC(net, 0, vm_lower_pu=0.99, vm_upper_pu=1.01, calculate_impedance=True, remote_bus=4)
    runpp(net, run_control=True)

    assert 0.99 <= c._get_vm_pu(net) <= 1.01
    assert net.trafo.tap_pos.at[0] > 0

    # the remote bus is much closer to 1 pu than with the local discrete tap control
    vm_remote_ldc = net.res_bus.vm_pu.at[4]

    net_local = _radial_feeder()
    DiscreteTapControl(net_local, 0, vm_lower_pu=0.99, vm_upper_pu=1.01)
    runpp(net_local, run_control=True)
    vm_remote_local = net_local.res_bus.vm_pu.at[4]

    assert abs(vm_remote_ldc - 1.0) < abs(vm_remote_local - 1.0)


def test_line_drop_compensation_calculate_impedance_with_switch():
    """
    Parallel lines have to be considered and open line switches have to be respected.
    """
    net = _radial_feeder()
    # add a parallel identical line, which should halve the impedance of the segment between bus 2 and bus 3
    create_lines(net, 2, 3, 0.3, "NAYY 4x50 SE")
    c = ContinuousTapControlLDC(net, 0, vm_set_pu=1.0, calculate_impedance=True, remote_bus=4)
    r_segment = net.line.r_ohm_per_km.at[1] * 0.3
    r_parallel = 2 * r_segment + r_segment / 2
    assert np.isclose(c.r_ohm, r_parallel)

    # an open switch on the parallel line restores the original impedance
    create_switch(net, 2, net.line.index[-1], et="l", closed=False)
    c.update_impedance(net)
    assert np.isclose(c.r_ohm, 3 * r_segment)

    # the impedance is also recalculated automatically whenever the controller is initialized
    net.switch.at[net.switch.index[-1], "closed"] = True
    c.initialize_control(net)
    assert np.isclose(c.r_ohm, r_parallel)


def test_line_drop_compensation_errors():
    # no impedance and no topological calculation
    with pytest.raises(ValueError):
        ContinuousTapControlLDC(simple_four_bus_system(), 0, vm_set_pu=1.0)

    # topological calculation without remote bus
    net = _radial_feeder()
    with pytest.raises(ValueError):
        ContinuousTapControlLDC(net, 0, vm_set_pu=1.0, calculate_impedance=True)

    # remote bus beyond a transformer is not allowed
    net = simple_four_bus_system()
    with pytest.raises(ValueError, match="no path without transformers"):
        ContinuousTapControlLDC(net, 0, vm_set_pu=1.0, calculate_impedance=True, remote_bus=0)

    # inconsistent number of impedance values for a vectorized controller
    net = _radial_feeder()
    with pytest.raises(ValueError, match="r_ohm has 3 values"):
        ContinuousTapControlLDC(net, [0, 0], vm_set_pu=1.0, r_ohm=[0.1, 0.2, 0.3], x_ohm=0.05)


def test_line_drop_compensation_out_of_service_trafo():
    """
    A controller of an out-of-service transformer has to be inactive without raising errors.
    """
    net = _radial_feeder()
    net.trafo.at[0, "in_service"] = False
    c = ContinuousTapControlLDC(net, 0, vm_set_pu=1.0, calculate_impedance=True, remote_bus=4)
    assert c.nothing_to_do(net)
    runpp(net)
    assert c.is_converged(net)


def test_line_drop_compensation_vectorized():
    """
    Vectorized (multiple transformers) controllers have to work as well.
    """
    net, remote_buses = _vectorized_feeder()

    c = ContinuousTapControlLDC(
        net, net.trafo.index.values, vm_set_pu=1.0, calculate_impedance=True, remote_bus=remote_buses
    )
    runpp(net, run_control=True)

    assert np.allclose(c._get_vm_pu(net), 1.0, atol=1e-3)
    assert np.all(net.trafo.tap_pos.values > 0)
    # each transformer has its own fictitious line to its own remote bus
    assert len(np.atleast_1d(c.r_ohm)) == len(net.trafo)


def test_line_drop_compensation_vectorized_single_remote_bus_and_manual_impedance():
    """
    A single remote bus and a single impedance value have to be broadcast to all controlled transformers.
    """
    net, _remote_buses = _vectorized_feeder(n_trafos=2)
    c = ContinuousTapControlLDC(net, net.trafo.index.values, vm_set_pu=1.0, r_ohm=0.1, x_ohm=0.02)

    assert np.allclose(np.atleast_1d(c.r_ohm), 0.1)
    runpp(net, run_control=True)
    assert np.allclose(c._get_vm_pu(net), 1.0, atol=1e-3)

    # two parallel transformers at the same buses share the same remote bus
    net = pandapowerNet(name="parallel_transformers_ldc")
    create_buses(net, 1, 20)
    create_buses(net, 3, 0.4)
    create_ext_grid(net, 0)
    create_transformers(net, [0, 0], [1, 1], "0.4 MVA 20/0.4 kV")
    create_lines(net, [1, 2], [2, 3], 0.3, "NAYY 4x50 SE")
    create_load(net, 3, 0.05, 0.02)
    net.trafo[["tap_side", "tap_neutral", "tap_min", "tap_max", "tap_step_percent", "tap_pos"]] = [
        "lv",
        0,
        -5,
        5,
        2.5,
        0,
    ]
    set_user_pf_options(net, init="dc", calculate_voltage_angles=True)
    runpp(net)

    c = ContinuousTapControlLDC(net, net.trafo.index.values, vm_set_pu=1.0, calculate_impedance=True, remote_bus=3)
    assert np.allclose(np.atleast_1d(c.r_ohm), net.line.r_ohm_per_km.sum() * 0.3)
    runpp(net, run_control=True)
    assert np.allclose(c._get_vm_pu(net), 1.0, atol=1e-3)


def test_line_drop_compensation_json_roundtrip():
    net = _radial_feeder()
    ContinuousTapControlLDC(net, 0, vm_set_pu=1.0, calculate_impedance=True, remote_bus=4)
    net2 = from_json_string(to_json(net))
    c = net2.controller.object.at[0]
    assert isinstance(c, LineDropCompensation)
    assert isinstance(c, ContinuousTapControlLDC)
    assert c.remote_bus == 4
    runpp(net2, run_control=True)
    runpp(net)
    runpp(net, run_control=True)
    assert np.isclose(net2.trafo.tap_pos.at[0], net.trafo.tap_pos.at[0])


def test_line_drop_compensation_hv_side():
    """
    Line drop compensation also works on the HV side of the transformer. There, the power flows from the remote bus
    into the transformer, so the compensated voltage has to be higher than the measured voltage at the transformer bus.
    """
    net = pandapowerNet(name="radial_feeder_ldc_hv")
    create_buses(net, 2, 110)
    create_buses(net, 1, 20)
    create_ext_grid(net, 0)
    create_lines(net, 0, 1, 20.0, "243-AL1/39-ST1A 110.0")
    create_transformer(net, 1, 2, "25 MVA 110/20 kV")
    create_load(net, 2, 15.0, 5.0)
    net.trafo[["tap_side", "tap_neutral", "tap_min", "tap_max", "tap_step_percent", "tap_pos"]] = [
        "hv",
        0,
        -9,
        9,
        1.5,
        0,
    ]
    set_user_pf_options(net, init="dc", calculate_voltage_angles=True)
    runpp(net)

    c = ContinuousTapControlLDC(net, 0, vm_set_pu=1.0, side="hv", calculate_impedance=True, remote_bus=0)
    assert np.isclose(c.r_ohm, net.line.r_ohm_per_km.at[0] * 20.0)
    assert np.isclose(c.x_ohm, net.line.x_ohm_per_km.at[0] * 20.0)

    # the compensated voltage corresponds to the (higher) voltage at the remote bus
    assert c._get_vm_pu(net) > net.res_bus.vm_pu.at[1]
    assert np.isclose(c._get_vm_pu(net), net.res_bus.vm_pu.at[0], atol=2e-3)


def test_line_drop_compensation_trafo3w():
    """
    Line drop compensation also works for three winding transformers.
    """
    net = pandapowerNet(name="ldc_trafo3w")
    hv_bus = create_buses(net, 1, 110)[0]
    mv_buses = create_buses(net, 3, 20)
    lv_bus = create_buses(net, 1, 10)[0]
    create_ext_grid(net, hv_bus)
    create_transformer3w(net, hv_bus, mv_buses[0], lv_bus, "63/25/38 MVA 110/20/10 kV")
    create_lines(net, mv_buses[:2], mv_buses[1:], 5.0, "NA2XS2Y 1x185 RM/25 12/20 kV")
    create_load(net, mv_buses[2], 2.0, 0.5)
    net.trafo3w[["tap_side", "tap_neutral", "tap_min", "tap_max", "tap_step_percent", "tap_pos"]] = [
        "hv",
        0,
        -9,
        9,
        1.5,
        0,
    ]
    set_user_pf_options(net, init="dc", calculate_voltage_angles=True)
    runpp(net)

    c = ContinuousTapControlLDC(
        net, 0, vm_set_pu=1.0, side="mv", element="trafo3w", calculate_impedance=True, remote_bus=mv_buses[2]
    )
    assert np.isclose(c.r_ohm, net.line.r_ohm_per_km.sum() * 5.0)
    assert np.isclose(c.x_ohm, net.line.x_ohm_per_km.sum() * 5.0)
    assert np.isclose(c._get_vm_pu(net), net.res_bus.vm_pu.at[mv_buses[2]], atol=2e-3)

    runpp(net, run_control=True)
    assert np.isclose(c._get_vm_pu(net), 1.0, atol=1e-3)
    assert np.isclose(net.res_bus.vm_pu.at[mv_buses[2]], 1.0, atol=2e-3)


def test_line_drop_compensation_calculate_impedance_with_impedance_element_and_bus_switch():
    """
    Per unit impedance elements have to be converted to ohm, and closed bus-bus switches are ideal connections.
    """
    net = pandapowerNet(name="ldc_impedance_element")
    create_buses(net, 1, 20)
    lv_buses = create_buses(net, 3, 0.4)
    create_ext_grid(net, 0)
    create_transformer(net, 0, lv_buses[0], "0.4 MVA 20/0.4 kV")
    create_lines(net, lv_buses[0], lv_buses[1], 0.3, "NAYY 4x50 SE")
    create_impedance(net, lv_buses[1], lv_buses[2], rft_pu=0.5, xft_pu=0.1, sn_mva=1.0)
    create_load(net, lv_buses[2], 0.03, 0.01)
    net.trafo[["tap_side", "tap_neutral", "tap_min", "tap_max", "tap_step_percent", "tap_pos"]] = [
        "lv",
        0,
        -9,
        9,
        2.5,
        0,
    ]
    runpp(net)

    c = ContinuousTapControlLDC(net, 0, vm_set_pu=1.0, calculate_impedance=True, remote_bus=lv_buses[2])
    z_base_ohm = net.bus.vn_kv.at[lv_buses[1]] ** 2 / 1.0
    assert np.isclose(c.r_ohm, net.line.r_ohm_per_km.at[0] * 0.3 + 0.5 * z_base_ohm)
    assert np.isclose(c.x_ohm, net.line.x_ohm_per_km.at[0] * 0.3 + 0.1 * z_base_ohm)
    assert np.isclose(c._get_vm_pu(net), net.res_bus.vm_pu.at[lv_buses[2]], atol=2e-3)

    # a closed bus-bus switch in parallel to the impedance element is an ideal short circuit
    create_switch(net, lv_buses[1], lv_buses[2], et="b", closed=True)
    c.update_impedance(net)
    assert np.isclose(c.r_ohm, net.line.r_ohm_per_km.at[0] * 0.3)
    assert np.isclose(c.x_ohm, net.line.x_ohm_per_km.at[0] * 0.3)

    # opening the switch restores the impedance of the impedance element
    net.switch.at[0, "closed"] = False
    c.update_impedance(net)
    assert np.isclose(c.r_ohm, net.line.r_ohm_per_km.at[0] * 0.3 + 0.5 * z_base_ohm)


def test_line_drop_compensation_unreachable_remote_bus(caplog):
    """
    If the remote bus is not reachable, the controller raises an error on creation. If the impedance was calculated
    before, it is kept and a warning is logged.
    """
    net = _radial_feeder()
    # bus 5 is on a separate feeder that is only connected via an open switch
    create_buses(net, 1, 0.4)
    create_lines(net, 1, 5, 0.2, "NAYY 4x50 SE")
    create_switch(net, 1, net.line.index[-1], et="l", closed=False)

    with pytest.raises(ValueError, match="no path without transformers"):
        ContinuousTapControlLDC(net, 0, vm_set_pu=1.0, calculate_impedance=True, remote_bus=5)

    c = ContinuousTapControlLDC(net, 0, vm_set_pu=1.0, calculate_impedance=True, remote_bus=4)
    r_ohm, x_ohm = c.r_ohm, c.x_ohm
    c.remote_bus = 5
    c.update_impedance(net)
    # the previously calculated impedance is kept
    assert c.r_ohm == r_ohm
    assert c.x_ohm == x_ohm
    assert "could not be recalculated" in caplog.text


def test_line_drop_compensation_manual_impedance_is_kept():
    """
    Without calculate_impedance, the given impedance is not overwritten by the automatic recalculation.
    """
    net = _radial_feeder()
    c = ContinuousTapControlLDC(net, 0, vm_set_pu=1.0, r_ohm=0.1, x_ohm=0.02, remote_bus=4)
    c.initialize_control(net)
    c.update_impedance(net)
    assert c.r_ohm == 0.1
    assert c.x_ohm == 0.02

    # only one of the two values is required, the other one defaults to zero
    c = ContinuousTapControlLDC(net, 0, vm_set_pu=1.0, x_ohm=0.05)
    assert c.r_ohm == 0.0
    assert c.x_ohm == 0.05


def test_line_drop_compensation_usable_before_first_control_step():
    """
    The transformer bus has to be known directly after creation, also if the impedance is given manually, so that the
    compensated voltage can be evaluated before the first control step.
    """
    net = _radial_feeder()
    r_ohm = net.line.r_ohm_per_km.sum() * 0.3
    x_ohm = net.line.x_ohm_per_km.sum() * 0.3
    c = ContinuousTapControlLDC(net, 0, vm_set_pu=1.0, r_ohm=r_ohm, x_ohm=x_ohm, remote_bus=4)
    assert c.trafobus == net.trafo.lv_bus.at[0]
    assert np.isclose(c._get_vm_pu(net), net.res_bus.vm_pu.at[4], atol=2e-3)

    # the same holds for the discrete controller
    d = DiscreteTapControlLDC(net, 0, vm_lower_pu=0.99, vm_upper_pu=1.01, r_ohm=r_ohm, x_ohm=x_ohm, remote_bus=4)
    assert d.trafobus == net.trafo.lv_bus.at[0]
    assert np.isclose(d._get_vm_pu(net), net.res_bus.vm_pu.at[4], atol=2e-3)


def test_line_drop_compensation_equality_and_hash():
    net = _radial_feeder()
    c1 = ContinuousTapControlLDC(net, 0, vm_set_pu=1.0, calculate_impedance=True, remote_bus=4)
    c2 = DiscreteTapControlLDC(net, 0, vm_lower_pu=0.99, vm_upper_pu=1.01, r_ohm=0.3, x_ohm=0.05)
    # controllers with an __eq__ implementation also have to remain hashable
    assert isinstance(hash(c1), int)
    assert isinstance(hash(c2), int)

    net2 = copy.deepcopy(net)
    assert c1 == net2.controller.object.at[0]
    assert c2 == net2.controller.object.at[1]
    net2.controller.object.at[1].r_ohm = 0.9
    assert c2 != net2.controller.object.at[1]

    # different number of impedance values
    net3, _remote_buses = _vectorized_feeder(n_trafos=2)
    c3 = ContinuousTapControlLDC(net3, net3.trafo.index.values, vm_set_pu=1.0, r_ohm=0.1, x_ohm=0.02)
    net4 = copy.deepcopy(net3)
    net4.controller.object.at[0].r_ohm = np.array([0.1, 0.2])
    assert c3 != net4.controller.object.at[0]


def test_line_drop_compensation_time_series_recycling(caplog):
    """
    The topological impedance calculation must not destroy the internal state of the net that is needed for the
    result recycling of a time series simulation.
    """
    from pandapower.control import ConstControl
    from pandapower.timeseries import DFData, OutputWriter, run_timeseries

    net = _radial_feeder()
    load = net.load.index[0]
    ds = DFData(pd.DataFrame({"load": [0.2, 0.5, 1.0, 1.3, 0.8]}))
    ConstControl(net, "load", "scaling", load, profile_name="load", data_source=ds)
    ContinuousTapControlLDC(net, 0, vm_set_pu=1.0, calculate_impedance=True, remote_bus=4)
    # the internal ppc state of the net is needed for the recycle of the load flow
    assert "internal" in net["_ppc"]
    ow = OutputWriter(net, [(0, 5)], output_path=None)
    ow.log_variable("res_bus", "vm_pu")
    ow.log_variable("trafo", "tap_pos")

    with caplog.at_level("WARNING"):
        run_timeseries(net, time_steps=range(5))

    assert "internal variables are missing" not in caplog.text
    # the compensated voltage is regulated to the set point in every time step, i.e. the recycling is consistent
    assert np.allclose(ow.output["res_bus.vm_pu"].iloc[:, 4].values, 1.0, atol=3e-3)
    # the tap position is adapted to the load (recycling must not freeze the results)
    assert np.unique(ow.output["trafo.tap_pos"].values).size > 1


def test_continuous_tap_control_unchanged_by_ldc_hook():
    """
    The refactoring of the voltage measurement must not change the behavior of the local controller.
    """
    net = _radial_feeder()
    ContinuousTapControl(net, 0, vm_set_pu=1.0)
    runpp(net, run_control=True)
    runpp(net)
    assert np.isclose(net.res_bus.vm_pu.at[1], 1.0, atol=1e-3)

    net2 = _radial_feeder()
    DiscreteTapControl(net2, 0, vm_lower_pu=0.99, vm_upper_pu=1.01)
    runpp(net2, run_control=True)
    assert 0.99 <= net2.res_bus.vm_pu.at[1] <= 1.01


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
