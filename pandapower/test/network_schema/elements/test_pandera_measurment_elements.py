
import itertools
import pandas as pd
import pandera as pa
import pytest

from pandapower.create import create_empty_network, create_bus
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
    all_allowed_floats,
    all_allowed_ints,
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
                itertools.product(["name"], strings),
                itertools.product(["measurement_type"], valid_measurement_types),
                itertools.product(["element_type"], valid_element_types),
                itertools.product(["value"], all_allowed_floats),
                itertools.product(["std_dev"], all_allowed_floats),
                itertools.product(["element"], all_allowed_ints),
                itertools.product(["check_existing"], bools),
                itertools.product(["side"], strings),
            )
        ),
    )
    def test_valid_required_values(self, parameter, valid_value):
        """Test: valid required values are accepted"""
        net = create_empty_network()
        create_bus(net, 0.4)          # index 0
        create_bus(net, 0.4)          # index 1
        create_bus(net, 0.4, index=42)

        net.measurement = pd.DataFrame(
            {
                "name": pd.Series(["m1"], dtype="string"),
                "measurement_type": ["p"],
                "element_type": ["bus"],
                "value": [10.0],
                "std_dev": [0.1],
                "bus": [0],
                "element": [0],
                "check_existing": [True],
                "side": ["hv"],
            }
        )
        if parameter in {"name"}:
            net.measurement[parameter] = pd.Series([valid_value], dtype="string")
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
                itertools.product(["check_existing"], not_boolean_list),
                itertools.product(["side"], not_strings_list),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Test: invalid required values are rejected"""
        net = create_empty_network()
        create_bus(net, 0.4)          # index 0
        create_bus(net, 0.4)          # index 1

        net.measurement = pd.DataFrame(
            {
                "name": pd.Series(["m1"], dtype="string"),
                "measurement_type": ["p"],
                "element_type": ["bus"],
                "value": [10.0],
                "std_dev": [0.1],
                "bus": [0],
                "element": [0],
                "check_existing": [True],
                "side": ["hv"],
            }
        )

        net.measurement[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestMeasurementOptionalFields:
    """Tests for optional measurement fields"""

    def test_measurement_without_bus_column_is_valid(self):
        """Test: 'bus' column is optional and may be absent"""
        net = create_empty_network()
        create_bus(net, 0.4)

        net.measurement = pd.DataFrame(
            {
                "name": pd.Series(["m1"], dtype="string"),
                "measurement_type": ["q"],
                "element_type": ["gen"],
                "value": [5.0],
                "std_dev": [0.2],
                "element": [0],
                "check_existing": [False],
                "side": ["from"],
            }
        )
        validate_network(net)

    @pytest.mark.parametrize(
        "valid_bus",
        positiv_ints_plus_zero,
    )
    def test_optional_bus_valid_values(self, valid_bus):
        """Test: optional 'bus' column accepts valid values and FK passes if index exists"""
        net = create_empty_network()
        create_bus(net, 0.4)          # 0
        create_bus(net, 0.4)          # 1
        create_bus(net, 0.4, index=42)

        net.measurement = pd.DataFrame(
            {
                "name": pd.Series(["m2"], dtype="string"),
                "measurement_type": ["i"],
                "element_type": ["line"],
                "value": [3.3],
                "std_dev": [0.05],
                "bus": [0],
                "element": [0],
                "check_existing": [True],
                "side": ["to"],
            }
        )
        net.measurement["bus"] = valid_bus
        validate_network(net)

    @pytest.mark.parametrize(
        "invalid_bus",
        [*negativ_ints, *not_ints_list],
    )
    def test_optional_bus_invalid_values(self, invalid_bus):
        """Test: optional 'bus' column rejects invalid values"""
        net = create_empty_network()
        create_bus(net, 0.4)

        net.measurement = pd.DataFrame(
            {
                "name": pd.Series(["m3"], dtype="string"),
                "measurement_type": ["v"],
                "element_type": ["trafo"],
                "value": [1.01],
                "std_dev": [0.01],
                "bus": [0],
                "element": [0],
                "check_existing": [True],
                "side": ["hv"],
            }
        )
        net.measurement["bus"] = invalid_bus
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestMeasurementForeignKey:
    """Tests for foreign key constraints (bus FK)"""

    def test_invalid_bus_index(self):
        """Test: bus must reference an existing bus index if present"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        net.measurement = pd.DataFrame(
            {
                "name": pd.Series(["m4"], dtype="string"),
                "measurement_type": ["p"],
                "element_type": ["bus"],
                "value": [2.0],
                "std_dev": [0.1],
                "bus": [b0],
                "element": [b0],
                "check_existing": [True],
                "side": ["hv"],
            }
        )

        net.measurement["bus"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestMeasurementResults:
    """Tests for measurement results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_measurement_usage(self):
        """Test: measurement values are consumed correctly by state estimation"""
        pass