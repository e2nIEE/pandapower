# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pytest
import numpy as np

from pandapower.create import (
    create_bus, create_line_from_parameters,
    create_load, create_gen, create_measurement
)
from pandapower.network import pandapowerNet
from pandapower.network_schema.tools.validation.network_validation import validate_network


def test_create_measurement():
    # Test basic measurement creation with required parameters
    net = pandapowerNet(name="test_create_measurement")
    b1 = create_bus(net, 110.0)
    create_load(net, b1, p_mw=1.0, q_mvar=0.5)

    # Create power measurement on load
    midx = create_measurement(
        net,
        meas_type="p",
        element_type="load",
        value=1.0,
        std_dev=0.05,
        element=0,
    )

    assert len(net.measurement) == 1
    assert net.measurement.at[midx, "measurement_type"] == "p"
    assert net.measurement.at[midx, "element_type"] == "load"
    assert net.measurement.at[midx, "element"] == 0
    assert np.isclose(net.measurement.at[midx, "value"], 1.0)
    assert np.isclose(net.measurement.at[midx, "std_dev"], 0.05)

    validate_network(net)


def test_create_measurement_bus_voltage():
    # Test voltage measurement on bus
    net = pandapowerNet(name="test_create_measurement_bus_voltage")
    b1 = create_bus(net, 110.0)

    midx = create_measurement(
        net,
        meas_type="v",
        element_type="bus",
        value=1.02,
        std_dev=0.01,
        element=b1,
    )

    assert len(net.measurement) == 1
    assert net.measurement.at[midx, "measurement_type"] == "v"
    assert net.measurement.at[midx, "element_type"] == "bus"
    assert np.isclose(net.measurement.at[midx, "value"], 1.02)

    validate_network(net)


def test_create_measurement_line_with_side():
    # Test line measurement with side parameter
    net = pandapowerNet(name="test_create_measurement_line_with_side")
    b1 = create_bus(net, 110.0)
    b2 = create_bus(net, 110.0)
    create_line_from_parameters(
        net, b1, b2, length_km=10.0, r_ohm_per_km=0.1,
        x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1.0
    )

    # Create measurement on "from" side of line
    midx = create_measurement(
        net,
        meas_type="p",
        element_type="line",
        value=5.0,
        std_dev=0.1,
        element=0,
        side="from",
        name="line_measurement",
    )

    assert len(net.measurement) == 1
    assert net.measurement.at[midx, "element_type"] == "line"
    assert net.measurement.at[midx, "element"] == 0
    assert net.measurement.at[midx, "side"] == "from"
    assert net.measurement.at[midx, "name"] == "line_measurement"

    validate_network(net)


def test_create_measurement_with_optional_params():
    # Test measurement with all optional parameters
    net = pandapowerNet(name="test_create_measurement_with_optional_params")
    b1 = create_bus(net, 110.0)
    create_gen(net, b1, p_mw=50.0)

    midx = create_measurement(
        net,
        meas_type="p",
        element_type="gen",
        value=50.0,
        std_dev=1.0,
        element=0,
        name="gen_power_measurement",
        check_existing=False,
        index=10,
        test_kwargs="dummy_string",
    )

    assert midx == 10
    assert net.measurement.at[midx, "name"] == "gen_power_measurement"
    assert net.measurement.test_kwargs.at[midx] == "dummy_string"

    validate_network(net)


def test_create_measurement_missing_side_for_line():
    # Test that missing side for line raises error
    net = pandapowerNet(name="test_create_measurement_missing_side_for_line")
    b1 = create_bus(net, 110.0)
    b2 = create_bus(net, 110.0)
    create_line_from_parameters(
        net, b1, b2, length_km=10.0, r_ohm_per_km=0.1,
        x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1.0
    )

    with pytest.raises(UserWarning, match="requires parameter 'side'"):
        create_measurement(
            net,
            meas_type="p",
            element_type="line",
            value=5.0,
            std_dev=0.1,
            element=0,
        )


def test_create_measurement_current_on_bus():
    # Test that current measurement on bus raises error
    net = pandapowerNet(name="test_create_measurement_current_on_bus")
    b1 = create_bus(net, 110.0)

    with pytest.raises(UserWarning, match="Line current measurements cannot be placed at buses"):
        create_measurement(
            net,
            meas_type="i",
            element_type="bus",
            value=0.5,
            std_dev=0.01,
            element=b1,
        )


def test_create_measurement_voltage_on_non_bus():
    # Test that voltage measurement on non-bus raises error
    net = pandapowerNet(name="test_create_measurement_voltage_on_non_bus")
    b1 = create_bus(net, 110.0)
    b2 = create_bus(net, 110.0)
    l1 = create_line_from_parameters(
        net, b1, b2, length_km=10.0, r_ohm_per_km=0.1,
        x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1.0
    )

    with pytest.raises(UserWarning, match="Voltage measurements can only be placed at a bus"):
        create_measurement(
            net,
            meas_type="v",
            element_type="line",
            side="from",
            value=1.0,
            std_dev=0.01,
            element=l1,
        )


def test_create_measurement_nonexistent_element():
    # Test that measurement on non-existent element raises error
    net = pandapowerNet(name="test_create_measurement_nonexistent_element")
    create_bus(net, 110.0)

    with pytest.raises(UserWarning, match="load with index=5 does not exist"):
        create_measurement(
            net,
            meas_type="p",
            element_type="load",
            value=1.0,
            std_dev=0.1,
            element=5,  # Load doesn't exist
        )


def test_create_measurement_check_existing():
    # Test check_existing behavior
    net = pandapowerNet(name="test_create_measurement_check_existing")
    b1 = create_bus(net, 110.0)
    create_load(net, b1, p_mw=1.0, q_mvar=0.5)

    # Create first measurement
    midx1 = create_measurement(
        net,
        meas_type="p",
        element_type="load",
        value=1.0,
        std_dev=0.05,
        element=0,
    )

    # Create second measurement with check_existing=True (should overwrite)
    midx2 = create_measurement(
        net,
        meas_type="p",
        element_type="load",
        value=2.0,
        std_dev=0.1,
        element=0,
        check_existing=True,
    )

    # Should have same index (overwritten)
    assert midx1 == midx2
    assert len(net.measurement) == 1
    assert np.isclose(net.measurement.at[midx1, "value"], 2.0)
