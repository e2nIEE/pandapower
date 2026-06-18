# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pytest
import numpy as np

from pandapower.network_structure import get_default_value
from pandapower.create import create_bus, create_ext_grid
from pandapower.network import pandapowerNet
from pandapower.network_schema.tools.validation.network_validation import validate_network


def test_create_ext_grid():
    """Test basic external grid creation with required parameters."""
    net = pandapowerNet(name="test_create_ext_grid")
    b1 = create_bus(net, 110.0)

    # Create external grid with only required parameter (bus)
    idx = create_ext_grid(net, bus=b1)

    assert len(net.ext_grid) == 1
    assert net.ext_grid.at[idx, "bus"] == b1
    assert np.isclose(net.ext_grid.at[idx, "vm_pu"], 1.0)  # default value
    assert np.isclose(net.ext_grid.at[idx, "va_degree"], 0.0)  # default value
    assert net.ext_grid.at[idx, "in_service"]  # default value
    assert np.isclose(net.ext_grid.at[idx, "slack_weight"], 1.0)  # default value

    validate_network(net)


def test_create_ext_grid_with_optional_params():
    """Test external grid creation with optional parameters."""
    net = pandapowerNet(name="test_create_ext_grid_with_optional_params")
    b1 = create_bus(net, 110.0)

    idx = create_ext_grid(
        net,
        bus=b1,
        vm_pu=1.03,
        va_degree=5.0,
        name="external_grid_1",
        in_service=True,
        s_sc_max_mva=1000.0,
        s_sc_min_mva=500.0,
        rx_max=0.1,
        rx_min=0.05,
        max_p_mw=200.0,
        min_p_mw=0.0,
        max_q_mvar=100.0,
        min_q_mvar=-100.0,
        index=5,
        r0x0_max=0.1,
        x0x_max=1.0,
        controllable=True,
        slack_weight=2.0,
        test_kwargs="dummy_string",
    )

    assert idx == 5
    assert len(net.ext_grid) == 1
    assert net.ext_grid.at[idx, "bus"] == b1
    assert np.isclose(net.ext_grid.at[idx, "vm_pu"], 1.03)
    assert np.isclose(net.ext_grid.at[idx, "va_degree"], 5.0)
    assert net.ext_grid.at[idx, "name"] == "external_grid_1"
    assert net.ext_grid.at[idx, "in_service"]
    assert np.isclose(net.ext_grid.at[idx, "s_sc_max_mva"], 1000.0)
    assert np.isclose(net.ext_grid.at[idx, "s_sc_min_mva"], 500.0)
    assert np.isclose(net.ext_grid.at[idx, "rx_max"], 0.1)
    assert np.isclose(net.ext_grid.at[idx, "rx_min"], 0.05)
    assert np.isclose(net.ext_grid.at[idx, "max_p_mw"], 200.0)
    assert np.isclose(net.ext_grid.at[idx, "min_p_mw"], 0.0)
    assert np.isclose(net.ext_grid.at[idx, "max_q_mvar"], 100.0)
    assert np.isclose(net.ext_grid.at[idx, "min_q_mvar"], -100.0)
    assert np.isclose(net.ext_grid.at[idx, "r0x0_max"], 0.1)
    assert np.isclose(net.ext_grid.at[idx, "x0x_max"], 1.0)
    assert net.ext_grid.at[idx, "controllable"]
    assert np.isclose(net.ext_grid.at[idx, "slack_weight"], 2.0)
    assert net.ext_grid.test_kwargs.at[idx] == "dummy_string"

    validate_network(net)


def test_create_ext_grid_out_of_service():
    """Test external grid creation with in_service=False."""
    net = pandapowerNet(name="test_create_ext_grid_out_of_service")
    b1 = create_bus(net, 110.0)

    idx = create_ext_grid(net, bus=b1, in_service=False)

    assert len(net.ext_grid) == 1
    assert not net.ext_grid.at[idx, "in_service"]

    validate_network(net)


def test_create_ext_grid_with_custom_index():
    """Test external grid creation with custom index."""
    net = pandapowerNet(name="test_create_ext_grid_with_custom_index")
    b1 = create_bus(net, 110.0)

    idx = create_ext_grid(net, bus=b1, index=10)

    assert idx == 10
    assert len(net.ext_grid) == 1

    validate_network(net)


def test_create_ext_grid_nonexistent_bus():
    """Test that creating an external grid with non-existent bus raises an error."""
    net = pandapowerNet(name="test_create_ext_grid_nonexistent_bus")

    with pytest.raises(Exception):
        create_ext_grid(net, bus=0)  # Bus doesn't exist


def test_create_ext_grid_multiple():
    """Test creating multiple external grids."""
    net = pandapowerNet(name="test_create_ext_grid_multiple")
    b1 = create_bus(net, 110.0)
    b2 = create_bus(net, 110.0)

    idx1 = create_ext_grid(net, bus=b1, name="slack_1")
    idx2 = create_ext_grid(net, bus=b2, name="slack_2")

    assert len(net.ext_grid) == 2
    assert net.ext_grid.at[idx1, "name"] == "slack_1"
    assert net.ext_grid.at[idx2, "name"] == "slack_2"

    validate_network(net)


def test_create_ext_grid_opf_limits():
    """Test external grid with OPF limit parameters."""
    net = pandapowerNet(name="test_create_ext_grid_opf_limits")
    b1 = create_bus(net, 110.0)

    idx = create_ext_grid(
        net,
        bus=b1,
        max_p_mw=100.0,
        min_p_mw=10.0,
        max_q_mvar=50.0,
        min_q_mvar=-50.0,
    )

    assert np.isclose(net.ext_grid.at[idx, "max_p_mw"], 100.0)
    assert np.isclose(net.ext_grid.at[idx, "min_p_mw"], 10.0)
    assert np.isclose(net.ext_grid.at[idx, "max_q_mvar"], 50.0)
    assert np.isclose(net.ext_grid.at[idx, "min_q_mvar"], -50.0)

    validate_network(net)


def test_create_ext_grid_short_circuit_params():
    """Test external grid with short circuit parameters."""
    net = pandapowerNet(name="test_create_ext_grid_short_circuit_params")
    b1 = create_bus(net, 110.0)

    idx = create_ext_grid(
        net,
        bus=b1,
        s_sc_max_mva=500.0,
        s_sc_min_mva=200.0,
        rx_max=0.15,
        rx_min=0.08,
        r0x0_max=0.2,
        x0x_max=1.5,
    )

    assert np.isclose(net.ext_grid.at[idx, "s_sc_max_mva"], 500.0)
    assert np.isclose(net.ext_grid.at[idx, "s_sc_min_mva"], 200.0)
    assert np.isclose(net.ext_grid.at[idx, "rx_max"], 0.15)
    assert np.isclose(net.ext_grid.at[idx, "rx_min"], 0.08)
    assert np.isclose(net.ext_grid.at[idx, "r0x0_max"], 0.2)
    assert np.isclose(net.ext_grid.at[idx, "x0x_max"], 1.5)

    validate_network(net)


def test_create_ext_grid_controllable():
    """Test external grid controllable parameter."""
    net = pandapowerNet(name="test_create_ext_grid_controllable")
    b1 = create_bus(net, 110.0)

    # Create first ext_grid without controllable
    idx1 = create_ext_grid(net, bus=b1)
    assert 'controllable' in net.ext_grid.columns

    # Create second ext_grid with controllable=True
    idx2 = create_ext_grid(net, bus=b1, controllable=True)

    assert not net.ext_grid.at[idx1, 'controllable']
    assert net.ext_grid.at[idx2, 'controllable']

    validate_network(net)


def test_create_ext_grid_slack_weight():
    """Test external grid slack_weight parameter."""
    net = pandapowerNet(name="test_create_ext_grid_slack_weight")
    b1 = create_bus(net, 110.0)

    idx = create_ext_grid(net, bus=b1, slack_weight=3.5)

    assert np.isclose(net.ext_grid.at[idx, "slack_weight"], 3.5)

    validate_network(net)


def test_create_ext_grid_default_values():
    """Test that default values are set correctly."""
    net = pandapowerNet(name="test_create_ext_grid_default_values")
    b1 = create_bus(net, 110.0)

    idx = create_ext_grid(net, bus=b1)

    # Check all default values
    assert net.ext_grid.at[idx, "vm_pu"] == get_default_value("ext_grid", "vm_pu")
    assert net.ext_grid.at[idx, "va_degree"] == get_default_value("ext_grid", "va_degree")
    assert net.ext_grid.at[idx, "in_service"] == get_default_value("ext_grid", "in_service")
    assert net.ext_grid.at[idx, "slack_weight"] == get_default_value("ext_grid", "slack_weight")

    # Check that optional columns are not created
    assert (
        "max_p_mw", "min_p_mw", "max_q_mvar", "min_q_mvar", "s_sc_max_mva", "s_sc_min_mva", "rx_max", "rx_min"
    ) not in net.ext_grid.columns

    validate_network(net)