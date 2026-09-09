import itertools

import pandas as pd
import pandera as pa
import pytest

from pandapower import create_measurement
from pandapower.create import create_bus
from pandapower.network import pandapowerNet
from pandapower.network_schema.tools.validation.network_validation import validate_network
from pandapower.test.network_schema.elements.helper import (
    all_allowed_floats,
    all_allowed_ints,
    negativ_ints,
    not_floats_list,
    not_ints_list,
    not_strings_list,
    positiv_ints_plus_zero,
    strings,
)

valid_measurement_types = ["p", "q", "i", "v"]
valid_element_types = ["bus", "line", "trafo", "trafo3w", "load", "gen", "sgen", "shunt", "ward", "xward", "ext_grid"]
invalid_measurement_types = ["power", "voltage", "current", "", " "]
invalid_element_types = ["branch", "generator", "line3w", "", " "]


class TestMeasurementRequiredFields:
    """Tests for required measurement fields"""

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["name"], [pd.NA, *strings]),
                itertools.product(["measurement_type"], valid_measurement_types),
                itertools.product(["element_type"], valid_element_types),
                itertools.product(["value"], all_allowed_floats),
                itertools.product(["std_dev"], all_allowed_floats),
                itertools.product(["element"], all_allowed_ints),
                itertools.product(["side"], [pd.NA, *strings]),
            )
        ),
    )
    def test_valid_required_values(self, parameter, valid_value):
        """Test: valid required values are accepted"""
        net = pandapowerNet(name="test_valid_required_values")
        b0 = create_bus(net, 0.4)  # index 0
        create_bus(net, 0.4)  # index 1
        create_bus(net, 0.4, index=42)

        net.measurement = pd.DataFrame(
            {
                "name": pd.Series(["m1"], dtype=pd.StringDtype()),
                "measurement_type": ["p"],
                "element_type": ["bus"],
                "value": [10.0],
                "std_dev": [0.1],
                "element": [b0],
                "side": pd.Series(["hv"], dtype=pd.StringDtype()),
            }
        )
        if parameter in {"name", "side"}:
            net.measurement[parameter] = pd.Series([valid_value], dtype=pd.StringDtype())
        else:
            net.measurement[parameter] = valid_value

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["name"], not_strings_list),
                itertools.product(["measurement_type"], [*invalid_measurement_types, *not_strings_list]),
                itertools.product(["element_type"], [*invalid_element_types, *not_strings_list]),
                itertools.product(["value"], not_floats_list),
                itertools.product(["std_dev"], not_floats_list),
                itertools.product(["element"], not_ints_list),
                itertools.product(["side"], not_strings_list),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Test: invalid required values are rejected"""
        net = pandapowerNet(name="test_invalid_required_values")
        create_bus(net, 0.4)  # index 0
        create_bus(net, 0.4)  # index 1

        create_measurement(
            net=net,
            meas_type="p",
            element_type="bus",
            value=10.0,
            std_dev=0.1,
            element=0,
            check_existing=True,
            side="hv",
        )

        net.measurement[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestMeasurementOptionalFields:
    """Tests for optional measurement fields"""

    def test_measurement_without_bus_column_is_valid(self):
        """Test: 'bus' column is optional and may be absent"""
        net = pandapowerNet(name="test_measurement_without_bus_column_is_valid")
        create_bus(net, 0.4)

        net.measurement = pd.DataFrame(
            {
                "name": pd.Series(["m1"], dtype="string"),
                "measurement_type": ["q"],
                "element_type": ["gen"],
                "value": [5.0],
                "std_dev": [0.2],
                "element": [0],
                "side": pd.Series(["from"], dtype="string")
            }
        )
        validate_network(net)


class TestMeasurementResults:
    """Tests for measurement results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_measurement_usage(self):
        """Test: measurement values are consumed correctly by state estimation"""
        raise NotImplementedError("Test not yet implemented")
