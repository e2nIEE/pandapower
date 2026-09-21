import itertools

import pandas as pd
import pandera as pa
import pytest

from pandapower import create_measurement
from pandapower.create import create_bus, create_line, create_transformer, create_transformer3w
from pandapower.network import pandapowerNet
from pandapower.network_schema.tools.validation.network_validation import validate_network
from pandapower.test.network_schema.elements.helper import (
    all_allowed_floats,
    all_allowed_ints,
    not_floats_list,
    not_ints_list,
    not_strings_list,
    strings,
)

valid_measurement_types = ["p", "q", "i", "v"]
valid_element_types = ["bus", "line", "trafo", "trafo3w", "load", "gen", "sgen", "shunt", "ward", "xward", "ext_grid"]
valid_sides = ["from", "to", "hv", "mv", "lv"]
invalid_measurement_types = ["power", "voltage", "current", "", " "]
invalid_element_types = ["branch", "generator", "line3w", "", " "]

measurement_sides_element_type_lookup = {
    "from": "line",
    "to": "line",
    "hv": "trafo3w",
    "mv": "trafo3w",
    "lv": "trafo3w",
}
measurement_element_type_side_lookup = {
    "line": "from",
    "trafo": "hv",
    "trafo3w": "hv",
}


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
                itertools.product(["side"], valid_sides),
            )
        ),
    )
    def test_valid_required_values(self, parameter, valid_value):
        """Test: valid required values are accepted"""
        net = pandapowerNet(name="test_valid_required_values")
        b0 = create_bus(net, 0.4)  # index 0
        create_bus(net, 0.4)  # index 1
        create_bus(net, 0.4, index=42)

        # TODO: add line/trafo/trafo3w to net for enabling foreign_key check.

        type = "bus"
        side = pd.NA
        if parameter == "side":
            type = measurement_sides_element_type_lookup[valid_value]
        if parameter == "element_type" and valid_value in ["line", "trafo", "trafo3w"]:
            side = measurement_element_type_side_lookup[valid_value]
        net.measurement = pd.DataFrame(
            {
                "name": pd.Series(["m1"], dtype=pd.StringDtype()),
                "measurement_type": ["p"],
                "element_type": [type],
                "value": [10.0],
                "std_dev": [0.1],
                "element": [b0],
                "side": pd.Series([side], dtype=pd.StringDtype()),
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

        net.measurement = pd.DataFrame(
            {
                "name": pd.Series(["m1"], dtype="string"),
                "measurement_type": ["p"],
                "element_type": ["bus"],
                "value": [10.0],
                "std_dev": [0.1],
                "bus": [0],
                "element": [0],
                "side": ["hv"],
            }
        )
        net.measurement[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    @pytest.mark.parametrize(
        "element_type, side",
        itertools.chain(
            itertools.product(["line"], ["from", "to"]),
            itertools.product(["trafo"], ["hv", "lv"]),
            itertools.product(["trafo3w"], ["hv", "mv", "lv"]),
        ),
    )
    def test_valid_et_side_combinations(self, element_type, side):
        net = pandapowerNet("test_valid_et_side_combinations")
        b0 = create_bus(net, 0.4)  # index 0
        b1 = create_bus(net, 0.4)  # index 1
        b2 = create_bus(net, 0.8)  # index 2
        b3 = create_bus(net, 1.2)  # index 3

        l0 = create_line(net, b0, b1, 1, "NAYY 4x50 SE")
        t0 = create_transformer(net, b2, b1, "160 MVA 380/110 kV")
        t3w0 = create_transformer3w(net, b3, b2, b1, "63/25/38 MVA 110/20/10 kV")

        match element_type:
            case "line":
                element = l0
            case "trafo":
                element = t0
            case "trafo3w":
                element = t3w0
            case _:
                raise AttributeError("element not in net")

        create_measurement(
            net=net,
            meas_type="p",
            element_type=element_type,
            value=10.0,
            std_dev=0.1,
            element=element,
            side=side,
            check_existing=True,
        )
        validate_network(net)

    @pytest.mark.parametrize(
        "element_type, side",
        itertools.chain(
            itertools.product(["line"], ["hv", "mv", "lv"]),
            itertools.product(["trafo"], ["from", "to", "mv"]),
            itertools.product(["trafo3w"], ["from", "to"]),
        ),
    )
    def test_invalid_et_side_combiantions(self, element_type, side):
        net = pandapowerNet("test_valid_et_side_combinations")
        b0 = create_bus(net, 0.4)  # index 0
        b1 = create_bus(net, 0.4)  # index 1
        b2 = create_bus(net, 0.8)  # index 2
        b3 = create_bus(net, 1.2)  # index 3

        l0 = create_line(net, b0, b1, 1, "NAYY 4x50 SE")
        t0 = create_transformer(net, b2, b1, "160 MVA 380/110 kV")
        t3w0 = create_transformer3w(net, b3, b2, b1, "63/25/38 MVA 110/20/10 kV")

        match element_type:
            case "line":
                element = l0
            case "trafo":
                element = t0
            case "trafo3w":
                element = t3w0
            case _:
                raise AttributeError("element not in net")

        m0 = create_measurement(
            net=net,
            meas_type="p",
            element_type=element_type,
            value=10.0,
            std_dev=0.1,
            element=element,
            side=measurement_element_type_side_lookup[element_type],
            check_existing=True,
        )
        net.measurement.at[m0, "side"] = side
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

    @pytest.mark.parametrize(
        "valid_bus",
        positiv_ints_plus_zero,
    )
    def test_optional_bus_valid_values(self, valid_bus):
        """Test: optional 'bus' column accepts valid values and FK passes if index exists"""
        net = pandapowerNet(name="test_optional_bus_valid_values")
        create_bus(net, 0.4)  # 0
        create_bus(net, 0.4)  # 1
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
                "side": pd.Series(["to"], dtype="string"),
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
        net = pandapowerNet(name="test_optional_bus_invalid_values")
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
                "side": ["hv"],
            }
        )
        net.measurement["bus"] = invalid_bus
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_all_optional_fields_valid(self):
        """Test: measurement with all optional fields set is valid"""
        net = pandapowerNet(name="test_all_optional_fields_valid")
        b0 = create_bus(net, 0.4)

        net.measurement = pd.DataFrame({
            "name": pd.Series(["measurement1"], dtype=pd.StringDtype()),
            "measurement_type": ["p"],
            "element_type": ["bus"],
            "value": [10.0],
            "std_dev": [0.1],
            "bus": [b0],
            "element": [b0],
            "side": pd.Series(["hv"], dtype=pd.StringDtype()),
            "origin_id": pd.Series(["origin_1"], dtype=pd.StringDtype()),
            "origin_class": pd.Series(["class_a"], dtype=pd.StringDtype()),
            "source": pd.Series(["scada"], dtype=pd.StringDtype()),
            "analog_id": pd.Series(["analog_1"], dtype=pd.StringDtype()),
            "terminal_id": pd.Series(["term_1"], dtype=pd.StringDtype()),
            "description": pd.Series(["test measurement"], dtype=pd.StringDtype()),
        })
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        [
            ("origin_id", pd.NA),
            ("origin_class", pd.NA),
            ("source", pd.NA),
            ("analog_id", pd.NA),
            ("terminal_id", pd.NA),
            ("description", pd.NA),
        ],
    )
    def test_cim_fields_valid_values(self, parameter, valid_value):
        """Test: CIM fields accept valid values"""
        net = pandapowerNet(name="test_cim_fields_valid_values")
        b0 = create_bus(net, 0.4)

        net.measurement = pd.DataFrame({
            "name": pd.Series(["m1"], dtype=pd.StringDtype()),
            "measurement_type": ["p"],
            "element_type": ["bus"],
            "value": [10.0],
            "std_dev": [0.1],
            "bus": [b0],
            "element": [b0],
            "side": pd.Series(["hv"], dtype=pd.StringDtype()),
        })

        net.measurement[parameter] = pd.Series([valid_value], dtype=pd.StringDtype())
        validate_network(net)


class TestMeasurementForeignKey:
    """Tests for foreign key constraints (bus FK)"""

    def test_invalid_bus_index(self):
        """Test: bus must reference an existing bus index if present"""
        net = pandapowerNet(name="test_invalid_bus_index")
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
                "side": ["hv"],
            }
        )

        net.measurement["bus"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


    def test_valid_bus_index_non_sequential(self):
        """Test: bus FK works with non-sequential bus indices"""
        net = pandapowerNet(name="test_valid_bus_index_non_sequential")
        create_bus(net, 0.4, index=10)
        create_bus(net, 0.4, index=42)
        create_bus(net, 0.4, index=100)

        net.measurement = pd.DataFrame(
            {
                "name": pd.Series(["m1", "m2", "m3"], dtype="string"),
                "measurement_type": ["p", "q", "v"],
                "element_type": ["bus", "bus", "bus"],
                "value": [10.0, 5.0, 1.01],
                "std_dev": [0.1, 0.2, 0.01],
                "bus": [10, 42, 100],
                "element": [10, 42, 100],
                "side": pd.Series([pd.NA, pd.NA, pd.NA], dtype="string"),
            }
        )

        validate_network(net)


class TestMeasurementResults:
    """Tests for measurement results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_measurement_usage(self):
        """Test: measurement values are consumed correctly by state estimation"""
        raise NotImplementedError("Test not yet implemented")
