# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pytest
import numpy as np

from pandapower.create import create_bus_dc, create_source_dc
from pandapower.network import pandapowerNet
from pandapower.network_schema.tools.validation.network_validation import validate_network


def test_create_source_dc():
    net = pandapowerNet(name="test_create_source_dc")
    b1 = create_bus_dc(net, 100)

    # Test basic source_dc creation
    idx = create_source_dc(net, bus_dc=b1, vm_pu=1.0)
    assert net.source_dc.at[idx, "bus_dc"] == b1
    assert np.isclose(net.source_dc.at[idx, "vm_pu"], 1.0)

    # Test source_dc with all optional parameters
    idx2 = create_source_dc(
        net,
        bus_dc=b1,
        vm_pu=1.05,
        name="test_source",
        in_service=False,
        type="voltage_source",
    )
    assert net.source_dc.at[idx2, "bus_dc"] == b1
    assert np.isclose(net.source_dc.at[idx2, "vm_pu"], 1.05)
    assert net.source_dc.at[idx2, "name"] == "test_source"
    assert net.source_dc.at[idx2, "in_service"] == False
    assert net.source_dc.at[idx2, "type"] == "voltage_source"

    # Test default values (vm_pu defaults to 1.0, in_service defaults to True)
    idx3 = create_source_dc(net, bus_dc=b1)
    assert np.isclose(net.source_dc.at[idx3, "vm_pu"], 1.0)
    assert net.source_dc.at[idx3, "in_service"] == True

    # Test with custom index
    idx4 = create_source_dc(net, bus_dc=b1, vm_pu=0.95, index=42)
    assert idx4 == 42
    assert net.source_dc.at[42, "bus_dc"] == b1
    assert np.isclose(net.source_dc.at[42, "vm_pu"], 0.95)

    validate_network(net)


def test_create_source_dc_invalid_bus():
    """Test that creating source_dc with non-existent bus raises error"""
    net = pandapowerNet(name="test_create_source_dc_invalid_bus")
    create_bus_dc(net, 100)  # creates bus with index 0

    # Test with non-existent bus_dc
    with pytest.raises(UserWarning, match=r"bus_dc .* does not exist"):
        create_source_dc(net, bus_dc=5, vm_pu=1.0)


def test_create_source_dc_with_kwargs():
    """Test that additional kwargs are passed correctly to the table"""
    net = pandapowerNet(name="test_create_source_dc_with_kwargs")
    b1 = create_bus_dc(net, 100)

    # Test with additional kwargs
    idx = create_source_dc(
        net,
        bus_dc=b1,
        vm_pu=1.02,
        name="custom_source",
        type="converter",
        test_custom_param="custom_value",
    )
    assert net.source_dc.at[idx, "name"] == "custom_source"
    assert net.source_dc.at[idx, "type"] == "converter"
    assert net.source_dc.at[idx, "test_custom_param"] == "custom_value"

    validate_network(net)


def test_create_source_dc_multiple():
    """Test creating multiple source_dc elements"""
    net = pandapowerNet(name="test_create_source_dc_multiple")
    b1 = create_bus_dc(net, 100)
    b2 = create_bus_dc(net, 110)

    # Create multiple sources
    idx1 = create_source_dc(net, bus_dc=b1, vm_pu=1.0, name="source_1")
    idx2 = create_source_dc(net, bus_dc=b2, vm_pu=0.98, name="source_2")
    idx3 = create_source_dc(net, bus_dc=b1, vm_pu=1.02, name="source_3")

    assert len(net.source_dc) == 3
    assert net.source_dc.at[idx1, "bus_dc"] == b1
    assert net.source_dc.at[idx2, "bus_dc"] == b2
    assert net.source_dc.at[idx3, "bus_dc"] == b1
    assert net.source_dc.at[idx1, "name"] == "source_1"
    assert net.source_dc.at[idx2, "name"] == "source_2"
    assert net.source_dc.at[idx3, "name"] == "source_3"

    validate_network(net)
