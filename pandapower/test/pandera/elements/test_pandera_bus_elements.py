import itertools
import numpy as np
import pandas as pd
import pandera as pa
import pytest

from pandapower.create import create_empty_network, create_bus
from pandapower.network_schema.tools.validation.network_validation import validate_network
from pandapower.network_schema.tools.helper import get_dtypes
from pandapower.network_schema.bus import bus_schema
from pandapower.test.pandera.elements.helper import (
    strings,
    bools,
    not_strings_list,
    not_floats_list,
    not_allowed_floats,
    not_boolean_list,
    negativ_floats,
    positiv_floats,
)


class TestBusRequiredFields:
    """Tests for required bus fields"""

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["name"], [pd.NA, *strings]),
                itertools.product(["vn_kv"], positiv_floats),
                itertools.product(["in_service"], bools),
            )
        ),
    )
    def test_valid_required_values(self, parameter, valid_value):
        """Test: Invalid required values are rejected"""
        net = create_empty_network()
        kwargs = {parameter: valid_value}
        vn_kv = kwargs.pop("vn_kv", 0.4)
        create_bus(net, vn_kv, **kwargs)

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["name"], [float(np.nan), *not_strings_list]),
                itertools.product(["vn_kv"], [float(np.nan), pd.NA, *not_floats_list, *negativ_floats]),
                itertools.product(["in_service"], [float(np.nan), pd.NA, *not_boolean_list]),
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

    def test_bus_with_optional_fields(self):
        """Test: Bus with every optional fields is valid"""
        net = create_empty_network()
        create_bus(net, 0.4, zone="everywhere", max_vm_pu=1.1, min_vm_pu=0.9, geodata=(0, 0), type="x")
        validate_network(net)

    def test_buses_with_optional_fields_including_nullvalues(self):
        """Test: Buses with some optional fields is valid"""
        net = create_empty_network()
        create_bus(net, 0.4, zone="nowhere")
        create_bus(net, 0.4, max_vm_pu=1)  # TODO: create method has 0.0 as default
        create_bus(net, 0.4, min_vm_pu=0.9)  # TODO: create method has 0.0 as default
        create_bus(net, 0.4, geodata=(1, 2))
        create_bus(net, 0.4, type="x")

        net.bus["min_vm_pu"].at[0] = float(np.nan)
        net.bus["min_vm_pu"].at[1] = float(np.nan)
        net.bus["min_vm_pu"].at[3] = float(np.nan)
        net.bus["min_vm_pu"].at[4] = float(np.nan)
        net.bus["max_vm_pu"].at[0] = float(np.nan)
        net.bus["max_vm_pu"].at[3] = float(np.nan)

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
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["min_vm_pu", "max_vm_pu"], [float(np.nan), np.nan, *positiv_floats]),
                itertools.product(["type", "zone", "geo"], [pd.NA, *strings]),
            )
        ),
    )
    def test_valid_optional_values(self, parameter, valid_value):
        """Test: valid optional values are accepted"""
        net = create_empty_network()
        create_bus(net, 0.4, **{parameter: valid_value})

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["min_vm_pu", "max_vm_pu"], [*not_floats_list, *not_allowed_floats]),
                itertools.product(["type", "zone", "geo"], [np.nan, float(np.nan), *not_strings_list]),
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
