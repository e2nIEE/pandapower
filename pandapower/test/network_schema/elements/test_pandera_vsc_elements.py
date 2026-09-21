# test_pandera_vsc_elements.py

import itertools
import numpy as np
import pandas as pd
import pandera as pa
import pytest

from pandapower.create import create_bus, create_bus_dc, create_vsc
from pandapower.network import pandapowerNet
from pandapower.network_schema.tools.validation.network_validation import validate_network
from pandapower.test.network_schema.elements.helper import (
    strings,
    bools,
    not_strings_list,
    not_floats_list,
    not_boolean_list,
    positiv_ints_plus_zero,
    negativ_ints,
    negativ_floats,
    not_ints_list,
    positiv_floats_plus_zero,
    all_allowed_floats,
)

# Allowed/invalid categorical values

allowed_control_mode_ac = ["vm_pu", "q_mvar", "slack"]
invalid_control_mode_ac = [s for s in strings if s not in allowed_control_mode_ac]

allowed_control_mode_dc = ["vm_pu", "p_mw"]
invalid_control_mode_dc = [s for s in strings if s not in allowed_control_mode_dc]


def _create_net_with_buses():
    """Helper to create a network with AC and DC buses."""
    net = pandapowerNet(name="_create_net_with_buses")
    create_bus(net, vn_kv=110.0)  # index 0
    create_bus(net, vn_kv=20.0)  # index 1
    create_bus_dc(net, vm_pu=1.0, vn_kv=110.0)  # index 0
    create_bus_dc(net, vm_pu=1.0, vn_kv=110.0)  # index 1
    return net


def _create_valid_vsc(net, bus=0, bus_dc=0, name="VSC-1"):
    """Helper to create a valid VSC element."""
    create_vsc(
        net,
        bus=bus,
        bus_dc=bus_dc,
        r_ohm=0.0,
        x_ohm=0.0,
        r_dc_ohm=0.0,
        pl_dc_mw=0.0,
        control_mode_ac="vm_pu",
        control_value_ac=1.0,
        control_mode_dc="vm_pu",
        control_value_dc=1.0,
        controllable=True,
        in_service=True,
        name=name,
    )


class TestVscRequiredFields:
    """Tests for required VSC fields"""

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], positiv_ints_plus_zero),
                itertools.product(["bus_dc"], positiv_ints_plus_zero),
                itertools.product(["r_ohm"], positiv_floats_plus_zero),
                itertools.product(["x_ohm"], positiv_floats_plus_zero),
                itertools.product(["r_dc_ohm"], all_allowed_floats),
                itertools.product(["pl_dc_mw"], all_allowed_floats),
                itertools.product(["control_mode_ac"], allowed_control_mode_ac),
                itertools.product(["control_value_ac"], all_allowed_floats),
                itertools.product(["control_mode_dc"], allowed_control_mode_dc),
                itertools.product(["control_value_dc"], all_allowed_floats),
                itertools.product(["controllable"], bools),
                itertools.product(["in_service"], bools),
            )
        ),
    )
    def test_valid_required_values(self, parameter, valid_value):
        """Test: valid required values are accepted"""
        net = pandapowerNet(name="test_valid_required_values")
        # AC buses
        create_bus(net, vn_kv=110.0)  # index 0
        create_bus(net, vn_kv=20.0)  # index 1
        create_bus(net, vn_kv=0.4, index=42)
        # DC buses
        create_bus_dc(net, vm_pu=1.0, vn_kv=110.0)  # index 0
        create_bus_dc(net, vm_pu=1.0, vn_kv=110.0)  # index 1
        create_bus_dc(net, vm_pu=1.0, index=42, vn_kv=110.0)

        _create_valid_vsc(net)

        net.vsc[parameter] = valid_value
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], [float(np.nan), pd.NA, None, *negativ_ints, *not_ints_list]),
                itertools.product(["bus_dc"], [float(np.nan), pd.NA, None, *negativ_ints, *not_ints_list]),
                itertools.product(["r_ohm"], [float(np.nan), pd.NA, None, *negativ_floats, *not_floats_list]),
                itertools.product(["x_ohm"], [float(np.nan), pd.NA, None, *negativ_floats, *not_floats_list]),
                itertools.product(["r_dc_ohm"], [float(np.nan), pd.NA, None, *not_floats_list]),
                itertools.product(["pl_dc_mw"], [float(np.nan), pd.NA, None, *not_floats_list]),
                itertools.product(["control_mode_ac"], [float(np.nan), pd.NA, None, *invalid_control_mode_ac, *not_strings_list]),
                itertools.product(["control_value_ac"], [float(np.nan), pd.NA, None, *not_floats_list]),
                itertools.product(["control_mode_dc"], [float(np.nan), pd.NA, None, *invalid_control_mode_dc, *not_strings_list]),
                itertools.product(["control_value_dc"], [float(np.nan), pd.NA, None, *not_floats_list]),
                itertools.product(["controllable"], [float(np.nan), pd.NA, None, *not_boolean_list]),
                itertools.product(["in_service"], [float(np.nan), pd.NA, None, *not_boolean_list]),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Test: invalid required values are rejected"""
        net = pandapowerNet(name="test_invalid_required_values")
        # AC buses
        create_bus(net, vn_kv=110.0)  # index 0
        create_bus(net, vn_kv=20.0)  # index 1
        # DC buses
        create_bus_dc(net, vm_pu=1.0, vn_kv=110.0)  # index 0
        create_bus_dc(net, vm_pu=1.0, vn_kv=110.0)  # index 1

        _create_valid_vsc(net)

        net.vsc[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestVscOptionalFields:
    """Tests for optional VSC fields"""

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(itertools.product(["name"], [pd.NA, *strings])),
    )
    def test_valid_optional_values(self, parameter, valid_value):
        """Test: valid optional values are accepted"""
        net = _create_net_with_buses()

        _create_valid_vsc(net)

        net.vsc[parameter] = pd.Series([valid_value], dtype=pd.StringDtype())
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(itertools.product(["name"], [float(np.nan), *not_strings_list])),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        """Test: Invalid optional values are rejected"""
        net = _create_net_with_buses()

        _create_valid_vsc(net)

        net.vsc[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_mixed_null_and_valid_values_in_rows(self):
        """Test: Multiple rows with mixed NA and valid values"""
        net = pandapowerNet(name="test_mixed_null_and_valid_values_in_rows")
        # AC buses
        create_bus(net, vn_kv=110.0)  # 0
        create_bus(net, vn_kv=20.0)  # 1
        create_bus(net, vn_kv=10.0)  # 2
        # DC buses
        create_bus_dc(net, vm_pu=1.0, vn_kv=110.0)  # 0
        create_bus_dc(net, vm_pu=1.0, vn_kv=110.0)  # 1
        create_bus_dc(net, vm_pu=1.0, vn_kv=110.0)  # 2

        # Row 1: all optional fields filled
        create_vsc(
            net,
            bus=0,
            bus_dc=0,
            r_ohm=0.1,
            x_ohm=0.2,
            r_dc_ohm=0.05,
            pl_dc_mw=0.3,
            control_mode_ac="vm_pu",
            control_value_ac=1.0,
            control_mode_dc="vm_pu",
            control_value_dc=1.0,
            controllable=True,
            in_service=True,
            name="VSC A",
        )

        # Row 2: name is null
        create_vsc(
            net,
            bus=1,
            bus_dc=1,
            r_ohm=0.0,
            x_ohm=0.0,
            r_dc_ohm=0.0,
            pl_dc_mw=0.0,
            control_mode_ac="q_mvar",
            control_value_ac=5.0,
            control_mode_dc="p_mw",
            control_value_dc=10.0,
            controllable=False,
            in_service=False,
            name=None,
        )

        # Row 3: name filled again
        create_vsc(
            net,
            bus=2,
            bus_dc=2,
            r_ohm=0.05,
            x_ohm=0.1,
            r_dc_ohm=0.02,
            pl_dc_mw=0.1,
            control_mode_ac="slack",
            control_value_ac=0.0,
            control_mode_dc="vm_pu",
            control_value_dc=1.02,
            controllable=True,
            in_service=True,
            name="VSC C",
        )

        # Set nullable columns with mixed values
        net.vsc["name"] = pd.Series(["VSC A", pd.NA, "VSC C"], dtype=pd.StringDtype())

        validate_network(net)


class TestVscForeignKey:
    """Tests for foreign key constraints"""

    def test_invalid_bus_index(self):
        """Test: bus FK must reference an existing bus index"""
        net = _create_net_with_buses()

        _create_valid_vsc(net)

        net.vsc["bus"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_invalid_bus_dc_index(self):
        """Test: bus_dc FK must reference an existing bus_dc index"""
        net = _create_net_with_buses()

        _create_valid_vsc(net)

        net.vsc["bus_dc"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_valid_bus_index_non_sequential(self):
        """Test: bus FK works with non-sequential bus indices"""
        net = pandapowerNet(name="test_invalid_bus_dc_index")
        create_bus(net, vn_kv=110.0, index=10)
        create_bus(net, vn_kv=20.0, index=42)
        create_bus_dc(net, vm_pu=1.0, vn_kv=110.0, index=5)
        create_bus_dc(net, vm_pu=1.0, vn_kv=110.0, index=99)

        create_vsc(
            net,
            bus=10,
            bus_dc=5,
            r_ohm=0.0,
            x_ohm=0.0,
            r_dc_ohm=0.0,
            pl_dc_mw=0.0,
            control_mode_ac="vm_pu",
            control_value_ac=1.0,
            control_mode_dc="vm_pu",
            control_value_dc=1.0,
            controllable=True,
            in_service=True,
        )

        validate_network(net)


class TestVscResults:
    """Tests for vsc results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_vsc_power_results(self):
        """Test: Power results are consistent"""
        pass

    @pytest.mark.skip(reason="Not yet implemented")
    def test_vsc_voltages(self):
        """Test: AC/DC voltages are within valid ranges"""
        pass
