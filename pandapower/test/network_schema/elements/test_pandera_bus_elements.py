# test_pandera_bus_elements.py

import itertools
import numpy as np
import pandas as pd
import pandera as pa
import pytest

from pandapower.create import create_bus
from pandapower.create._utils import add_tag_group_to_df
from pandapower.network import pandapowerNet
from pandapower.network_schema.tools.validation.network_validation import validate_network
from pandapower.test.network_schema.elements.helper import (
    strings,
    bools,
    not_strings_list,
    not_floats_list,
    not_allowed_floats,
    not_boolean_list,
    negativ_floats,
    negativ_floats_plus_zero,
    positiv_floats,
    positiv_floats_plus_zero,
    zero_float,
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
        net = pandapowerNet(name="test_valid_required_values")
        kwargs = {parameter: valid_value}
        vn_kv = kwargs.pop("vn_kv", 0.4)
        create_bus(net, vn_kv, **kwargs)

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["name"], [np.nan, *not_strings_list]),
                itertools.product(["vn_kv"], [np.nan, pd.NA, *not_floats_list, *negativ_floats, *zero_float]),
                itertools.product(["in_service"], [np.nan, pd.NA, *not_boolean_list]),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Test: Invalid required values are rejected"""
        net = pandapowerNet(name="test_invalid_required_values")
        create_bus(net, 0.4)
        net.bus[parameter] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestBusOptionalFields:
    """Tests for optional bus fields"""

    def test_bus_with_optional_fields(self):
        """Test: Bus with every optional fields is valid"""
        net = pandapowerNet(name="test_bus_with_optional_fields")
        create_bus(net, vn_kv=0.4, zone="everywhere", max_vm_pu=1.1, min_vm_pu=0.9, geodata=(0, 0), type="b")
        validate_network(net)

    def test_buses_with_optional_fields_including_nullvalues(self):
        """Test: Buses with some optional fields is valid"""
        net = pandapowerNet(name="test_buses_with_optional_fields_including_nullvalues")
        create_bus(net, 0.4, zone="nowhere")
        create_bus(net, 0.4, max_vm_pu=1.1, min_vm_pu=0.9)
        create_bus(net, 0.4, geodata=(1, 2))
        create_bus(net, 0.4, type="x")

        validate_network(net)

    def test_valid_type_values(self):
        """Test: Valid 'type' values are accepted"""
        net = pandapowerNet(name="test_valid_type_values")
        create_bus(net, 0.4)
        create_bus(net, 0.4)

        net.bus["type"].at[0] = "x"
        net.bus["type"].at[1] = pd.NA

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["min_vm_pu", "max_vm_pu"], [np.nan, *positiv_floats]),
                itertools.product(["type", "zone", "geo"], [pd.NA, *strings]),
            )
        ),
    )
    def test_valid_optional_values(self, parameter, valid_value):
        """Test: valid optional values are accepted"""
        net = pandapowerNet(name="test_valid_optional_values")
        create_bus(net, 0.4, **{parameter: valid_value})

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["min_vm_pu"], [*negativ_floats, *not_floats_list, *not_allowed_floats]),
                itertools.product(["max_vm_pu"], [*negativ_floats_plus_zero, *not_floats_list, *not_allowed_floats]),
                itertools.product(["type", "zone", "geo"], [np.nan, float(np.nan), *not_strings_list]),
            )
        ),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        """Test: Invalid optional values are rejected"""
        net = pandapowerNet(name="test_invalid_optional_values")
        create_bus(net, 0.4)

        # for OPF columns, add group dependency so only target parameter triggers failure
        #  otherwise the "min < max" check will fail.
        if parameter in ["min_vm_pu", "max_vm_pu"]:
            add_tag_group_to_df(net, "bus", "opf")

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