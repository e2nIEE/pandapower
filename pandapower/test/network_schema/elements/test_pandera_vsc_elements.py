# test_pandera_vsc_elements.py

import itertools
import numpy as np
import pandas as pd
import pandera as pa
import pytest

from pandapower import create_vsc
from pandapower.create import create_empty_network, create_bus, create_bus_dc
from pandapower.network_schema.tools.validation.network_validation import validate_network
from pandapower.test.network_schema.elements.helper import (
    strings,
    bools,
    not_strings_list,
    not_floats_list,
    not_boolean_list,
    positiv_ints_plus_zero,
    negativ_ints,
    not_ints_list,
    positiv_floats_plus_zero,
    all_allowed_floats,
)

# Allowed/invalid categorical values

allowed_control_mode_ac = ["vm_pu", "q_mvar", "slack"]
invalid_control_mode_ac = [s for s in strings if s not in allowed_control_mode_ac]

allowed_control_mode_dc = ["vm_pu", "p_mw"]
invalid_control_mode_dc = [s for s in strings if s not in allowed_control_mode_dc]


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
        net = create_empty_network()
        # AC buses
        create_bus(net, vn_kv=110.0)      # index 0
        create_bus(net, vn_kv=20.0)       # index 1
        create_bus(net, vn_kv=0.4, index=42)
        # DC buses
        create_bus_dc(net, vm_pu=1.0, vn_kv=110.0)     # index 0
        create_bus_dc(net, vm_pu=1.0, vn_kv=110.0)     # index 1
        create_bus_dc(net, vm_pu=1.0, index=42, vn_kv=110.0)

        create_vsc(
            net,
            bus=0,
            bus_dc=0,
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
            name="VSC-1",
        )

        net.vsc[parameter] = valid_value
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], [*negativ_ints, *not_ints_list]),
                itertools.product(["bus_dc"], [*negativ_ints, *not_ints_list]),
                itertools.product(["r_ohm"], [*not_floats_list, -0.1]),
                itertools.product(["x_ohm"], [*not_floats_list, -0.1]),
                itertools.product(["r_dc_ohm"], not_floats_list),
                itertools.product(["pl_dc_mw"], not_floats_list),
                itertools.product(["control_mode_ac"], [*invalid_control_mode_ac, *not_strings_list]),
                itertools.product(["control_value_ac"], not_floats_list),
                itertools.product(["control_mode_dc"], [*invalid_control_mode_dc, *not_strings_list]),
                itertools.product(["control_value_dc"], not_floats_list),
                itertools.product(["controllable"], not_boolean_list),
                itertools.product(["in_service"], not_boolean_list),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Test: invalid required values are rejected"""
        net = create_empty_network()
        # AC buses
        create_bus(net, vn_kv=110.0)      # index 0
        create_bus(net, vn_kv=20.0)       # index 1
        # DC buses
        create_bus_dc(net, vm_pu=1.0, vn_kv=110.0)     # index 0
        create_bus_dc(net, vm_pu=1.0, vn_kv=110.0)     # index 1

        create_vsc(
            net,
            bus=0,
            bus_dc=0,
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

        net.vsc[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestVscOptionalFields:
    """Tests for optional VSC fields"""

    def test_all_optional_fields_valid(self):
        """Test: VSC with optional 'name' set is valid"""
        net = create_empty_network()
        create_bus(net, vn_kv=110.0)      # AC
        create_bus_dc(net, vm_pu=1.0, vn_kv=110.0)     # DC

        create_vsc(
            net,
            bus=0,
            bus_dc=0,
            r_ohm=0.1,
            x_ohm=0.2,
            r_dc_ohm=0.05,
            pl_dc_mw=0.3,
            control_mode_ac="q_mvar",
            control_value_ac=10.0,
            control_mode_dc="p_mw",
            control_value_dc=5.0,
            controllable=False,
            in_service=True,
            name="Alpha",
        )
        net.vsc["name"] = net.vsc["name"].astype("string")
        validate_network(net)

    def test_optional_fields_with_nulls(self):
        """Test: VSC with optional 'name' including nulls is valid"""
        net = create_empty_network()
        # AC/DC buses
        create_bus(net, vn_kv=20.0)       # 0
        create_bus(net, vn_kv=10.0)       # 1
        create_bus_dc(net, vm_pu=1.0, vn_kv=110.0)     # 0
        create_bus_dc(net, vm_pu=1.0, vn_kv=110.0)     # 1

        create_vsc(
            net,
            bus=0,
            bus_dc=0,
            r_ohm=0.0,
            x_ohm=0.0,
            r_dc_ohm=0.0,
            pl_dc_mw=0.0,
            control_mode_ac="vm_pu",
            control_value_ac=1.0,
            control_mode_dc="vm_pu",
            control_value_dc=1.0,
            controllable=True,
            in_service=False,
            name="hello",
        )
        create_vsc(
            net,
            bus=1,
            bus_dc=1,
            r_ohm=0.0,
            x_ohm=0.0,
            r_dc_ohm=0.0,
            pl_dc_mw=0.0,
            control_mode_ac="slack",
            control_value_ac=0.0,
            control_mode_dc="p_mw",
            control_value_dc=0.0,
            controllable=False,
            in_service=True,
            name=None,
        )

        net.vsc["name"] = pd.Series(["V1", pd.NA], dtype=pd.StringDtype())
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(itertools.product(["name"], [pd.NA, *strings])),
    )
    def test_valid_optional_values(self, parameter, valid_value):
        """Test: valid optional values are accepted"""
        net = create_empty_network()
        create_bus(net, vn_kv=110.0)
        create_bus_dc(net, vm_pu=1.0, vn_kv=110.0)

        create_vsc(
            net,
            bus=0,
            bus_dc=0,
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
        net.vsc[parameter] = pd.Series([valid_value], dtype=pd.StringDtype())
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(itertools.product(["name"], not_strings_list)),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        """Test: Invalid optional values are rejected"""
        net = create_empty_network()
        create_bus(net, vn_kv=110.0)
        create_bus_dc(net, vm_pu=1.0, vn_kv=110.0)

        create_vsc(
            net,
            bus=0,
            bus_dc=0,
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
        net.vsc[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestVscForeignKey:
    """Tests for foreign key constraints"""

    def test_invalid_bus_index(self):
        net = create_empty_network()
        create_bus(net, vn_kv=110.0)
        create_bus_dc(net, vm_pu=1.0, vn_kv=110.0)

        create_vsc(
            net,
            bus=0,
            bus_dc=0,
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

        net.vsc["bus"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_invalid_bus_dc_index(self):
        net = create_empty_network()
        create_bus(net, vn_kv=110.0)
        create_bus_dc(net, vm_pu=1.0, vn_kv=110.0)

        create_vsc(
            net,
            bus=0,
            bus_dc=0,
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

        net.vsc["bus_dc"] = 9999
        with pytest.raises(pa.errors.SchemaError):
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