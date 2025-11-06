import itertools
import numpy as np
import pandas as pd
import pandera as pa
import pytest

from pandapower.create import create_empty_network, create_bus
from pandapower.network_schema.tools.validation.network_validation import validate_network
from pandapower.network_schema.tools.helper import get_dtypes
from pandapower.network_schema.bus import bus_schema

# Boolean types
bools = [True, False, np.bool_(True), np.bool_(False)]
# Numeric types
floats = [0, 0.0, 1.0, float("inf"), float("-inf"), np.float32(1.0), np.float64(1.0)]
ints = [0, 1, -1, 42, np.int8(1), np.int16(1), np.int32(1), np.int64(1), np.uint8(1)]
# String types
strings = [
    "True",
    "False",
    "true",
    "false",
    "1",
    "0",
    "yes",
    "no",
    "not a number",
    "",
    " ",
]
others = [
    # None and NaN variants
    None,
    pd.NaT,
    np.nan,
    # Collections
    {},
    {"value": True},
    # Objects
    object(),
    type,
    lambda x: x,
    # Complex numbers
    complex(1, 0),
    1 + 0j,
]

not_boolean_list = [*others, *strings, *ints, *floats]
not_floats_list = [*others, *strings, *ints, *bools]
not_strings_list = [*others, *strings, *ints, *bools]


class TestBusRequiredFields:
    """Tests for required bus fields"""

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["name"], [pd.NA, *strings]),
                itertools.product(["vn_kv"], floats),
                itertools.product(["in_service"], bools),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Test: Invalid required values are rejected"""
        net = create_empty_network()
        create_bus(net, 0.4)
        net.bus[parameter] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["name"], [float(np.nan), -1.5, *not_strings_list]),
                itertools.product(["vn_kv"], [float(np.nan), -1.5, pd.NA, *not_floats_list]),
                itertools.product(["in_service"], [float(np.nan), -1.5, pd.NA, *not_boolean_list]),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Test: Invalid required values are rejected"""
        net = create_empty_network()
        create_bus(net, 0.4)
        net.bus[parameter] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestBusOptionalFields:
    """Tests for optional bus fields"""

    def test_empty_network_validation(self):
        """Test: Bus with every optional fields is valid"""
        net = create_empty_network()
        # create_bus(net, 0.4, zone='everywhere', max_vm_pu=1.1, min_vm_pu=0.9, geodata=(0, 0))
        validate_network(net)

    def test_bus_with_optional_fields(self):
        """Test: Buses with some optional fields is valid"""
        net = create_empty_network()
        create_bus(net, 0.4, zone="nowhere")
        # create_bus(net, 0.4, max_vm_pu=1)
        # create_bus(net, 0.4, min_vm_pu=0.9)
        create_bus(net, 0.4, geodata=(1, 2))
        validate_network(net)

    def test_valid_type_values(self):
        """Test: Valid 'type' values are accepted"""
        net = create_empty_network()
        create_bus(net, 0.4)
        create_bus(net, 0.4)

        net.bus["type"].at[0] = "x"
        net.bus["type"].at[1] = pd.NA

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["min_vm_pu", "max_vm_pu"], [-1.5, *not_floats_list]),
                itertools.product(["type", "zone", "geo"], [float(np.nan), -1.5, *not_strings_list]),
            )
        ),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        """Test: Invalid optional values are rejected"""
        net = create_empty_network()
        create_bus(net, 0.4)
        net.bus[parameter] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestBusResults:
    """Tests for bus results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_bus_voltage_results(self):
        """Test: Voltage results are within valid range"""
        pass

    @pytest.mark.skip(reason="Not yet implemented")
    def test_bus_power_results(self):
        """Test: Power results are consistent"""
        pass
