# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pytest

from pandapower.create import (
    create_buses, create_ext_grid, create_gen,
    create_load, create_line, create_group, create_group_from_dict
)
from pandapower.network import pandapowerNet


def test_create_group():
    """Test basic group creation with create_group function."""
    # Create a simple network
    net = pandapowerNet(name="test_create_group")

    # Create some elements to group
    buses = create_buses(net, 4, vn_kv=0.4)
    create_ext_grid(net, bus=buses[0], name="ext_grid")
    create_gen(net, bus=buses[1], p_mw=10, name="gen1")
    create_gen(net, bus=buses[2], p_mw=20, name="gen2")
    create_load(net, bus=buses[3], p_mw=5, q_mvar=2, name="load1")
    create_line(net, from_bus=buses[0], to_bus=buses[1], length_km=1, std_type="NAYY 4x50 SE")

    # Test basic group creation
    idx = create_group(
        net,
        element_types=["bus", "gen"],
        element_indices=[[buses[0], buses[1]], [0, 1]],
        name="test_group"
    )

    # Verify the group was created correctly
    assert idx == 0
    assert len(net.group) == 2
    assert all(net.group.at[idx, "name"] == "test_group")
    assert all(net.group.at[idx, "element_type"] == ["bus", "gen"])
    assert net.group.at[idx, "element_index"].iloc[0] == [0, 1]
    assert net.group.at[idx, "element_index"].iloc[1] == [0, 1]


def test_create_group_with_custom_index():
    """Test group creation with a custom index."""
    net = pandapowerNet(name="test_create_group_with_custom_index")

    buses = create_buses(net, 3, vn_kv=0.4)
    create_gen(net, bus=buses[0], p_mw=10)
    create_gen(net, bus=buses[1], p_mw=20)

    # Create first group
    idx1 = create_group(net, "gen", [[0]], name="gen_group_1")
    assert idx1 == 0

    # Create second group with custom index
    idx2 = create_group(
        net, "gen", [[1]], name="gen_group_2", index=5
    )
    assert idx2 == 5
    assert len(net.group) == 2


def test_create_group_with_reference_column():
    """Test group creation using reference_column (by name instead of index)."""
    net = pandapowerNet(name="test_create_group_with_reference_column")

    buses = create_buses(net, 3, vn_kv=0.4)
    create_ext_grid(net, bus=buses[0], name="grid")
    create_gen(net, bus=buses[1], p_mw=10, name="wind_gen")
    create_gen(net, bus=buses[2], p_mw=20, name="solar_gen")

    # Create group using names as reference
    idx = create_group(
        net,
        element_types=["gen"],
        element_indices=[["wind_gen", "solar_gen"]],
        name="renewable_gens",
        reference_columns="name"
    )

    assert net.group.at[idx, "reference_column"] == "name"
    # The element_index should now contain names, not indices
    assert "wind_gen" in net.group.at[idx, "element_index"][0]
    assert "solar_gen" in net.group.at[idx, "element_index"][1]


def test_create_group_nonexistent_element_raises():
    """Test that creating a group with non-existent elements raises UserWarning."""
    net = pandapowerNet(name="test_create_group_nonexistent_element_raises")

    buses = create_buses(net, 2, vn_kv=0.4)
    create_gen(net, bus=buses[0], p_mw=10)

    # Attempting to create group with non-existent element should raise UserWarning
    with pytest.raises(UserWarning):
        create_group(
            net,
            element_types=["gen", "sgen"],
            element_indices=[[0], [0, 99]],  # sgen index 99 doesn't exist
            name="invalid_group"
        )


def test_create_group_from_dict():
    """Test group creation using the dictionary-based wrapper."""
    net = pandapowerNet(name="test_create_group_from_dict")

    buses = create_buses(net, 4, vn_kv=0.4)
    create_ext_grid(net, bus=buses[0])
    create_gen(net, bus=buses[1], p_mw=10)
    create_gen(net, bus=buses[2], p_mw=20)
    create_load(net, bus=buses[3], p_mw=5, q_mvar=2)

    # Create group from dictionary
    elements_dict = {
        "bus": [buses[0], buses[1], buses[2]],
        "gen": [0, 1]
    }

    idx = create_group_from_dict(
        net,
        elements_dict=elements_dict,
        name="dict_group"
    )

    # Verify the group was created
    assert all(net.group.at[idx, "name"] == "dict_group")
    assert all(net.group.at[idx, "element_type"] == ["bus", "gen"])


def test_create_group_from_dict_with_reference_column():
    """Test create_group_from_dict with reference_column."""
    net = pandapowerNet(name="test_create_group_from_dict_with_reference_column")

    buses = create_buses(net, 3, vn_kv=0.4)
    create_gen(net, bus=buses[0], p_mw=10, name="gen_a")
    create_gen(net, bus=buses[1], p_mw=20, name="gen_b")
    create_gen(net, bus=buses[2], p_mw=30, name="gen_c")

    # Create group from dict using names
    elements_dict = {
        "gen": ["gen_a", "gen_b"]
    }

    idx = create_group_from_dict(
        net,
        elements_dict=elements_dict,
        name="gens_by_name",
        reference_column="name"
    )

    assert net.group.at[idx, "reference_column"] == "name"
    assert "gen_a" in net.group.at[idx, "element_index"][0]
    assert "gen_c" not in net.group.at[idx, "element_index"][0]


def test_create_group_multiple_element_types():
    """Test creating a group with multiple different element types."""
    net = pandapowerNet(name="test_create_group_multiple_element_types")

    buses = create_buses(net, 3, vn_kv=0.4)
    create_ext_grid(net, bus=buses[0])
    create_gen(net, bus=buses[1], p_mw=10)
    create_load(net, bus=buses[2], p_mw=5, q_mvar=2)
    create_line(net, from_bus=buses[0], to_bus=buses[1], length_km=1, std_type="NAYY 4x50 SE")

    idx = create_group(
        net,
        element_types=["bus", "gen", "load", "line"],
        element_indices=[[buses[0], buses[1], buses[2]], [0], [0], [0]],
        name="mixed_elements"
    )

    assert len(net.group.at[idx, "element_type"]) == 4
    assert all(net.group.at[idx, "element_type"] == ["bus", "gen", "load", "line"])
