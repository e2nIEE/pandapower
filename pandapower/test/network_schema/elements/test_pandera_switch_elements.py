# test_pandera_switch_elements.py

import itertools
import numpy as np
import pandas as pd
import pandera as pa
import pytest

from pandapower.create import create_bus, create_switch
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
    not_ints_list,
    positiv_floats,
    negativ_floats_plus_zero,
    all_allowed_floats,
)


class TestSwitchRequiredFields:
    """Tests for required switch fields"""

    @pytest.mark.parametrize(
        "parameter, valid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], positiv_ints_plus_zero),
                itertools.product(["element"], positiv_ints_plus_zero),
                itertools.product(["et"], ["b", "l", "t", "t3"]),
                itertools.product(["closed"], bools),
                itertools.product(["in_ka"], [float(np.nan), *positiv_floats]),
                itertools.product(["z_ohm"], [float(np.nan), *all_allowed_floats]),
            )
        ),
    )
    def test_valid_required_values(self, parameter, valid_value):
        """Valid required values are accepted"""
        net = pandapowerNet(name="test_valid_required_values")
        create_bus(net, 0.4)  # 0
        create_bus(net, 0.4)  # 1
        create_bus(net, 0.4, index=42)

        # Base: bus-bus switch between bus 0 and bus 1
        create_switch(net, bus=0, element=1, et="b", type="CB", closed=True, in_ka=1.0, z_ohm=1.0)

        # Assign the parameter
        if parameter == "et":
            net.switch[parameter] = pd.Series([valid_value], dtype="string")
        else:
            net.switch[parameter] = valid_value

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter, invalid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], [float(np.nan), pd.NA, *negativ_ints, *not_ints_list]),
                itertools.product(["element"], [float(np.nan), pd.NA, *negativ_ints, *not_ints_list]),
                itertools.product(["et"], [float(np.nan), pd.NA, *strings, *not_strings_list]),  # anything not in {"b","l","t","t3"}
                itertools.product(["closed"], [float(np.nan), pd.NA, *not_boolean_list]),
                itertools.product(["in_ka"], [pd.NA, *negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["z_ohm"], [pd.NA, *not_floats_list]),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Invalid required values are rejected"""
        net = pandapowerNet(name="test_invalid_required_values")
        create_bus(net, 0.4)  # 0
        create_bus(net, 0.4)  # 1

        create_switch(net, bus=0, element=1, et="b", type="CB", closed=True, in_ka=1.0, z_ohm=1.0)

        net.switch[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    @pytest.mark.parametrize("parameter", ["bus", "element", "et", "closed"])
    def test_required_fields_nan_invalid(self, parameter):
        """NaN in required columns is invalid"""
        net = pandapowerNet(name="test_required_fields_nan_invalid")
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)

        create_switch(net, bus=b0, element=b1, et="b", closed=True)
        net.switch[parameter] = float(np.nan)
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestSwitchOptionalFields:
    """Tests for optional switch fields"""

    def test_all_optional_fields_valid(self):
        """All optional fields set"""
        net = pandapowerNet(name="test_all_optional_fields_valid")
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)

        create_switch(net, bus=b0, element=b1, et="b", type="CB", closed=False, in_ka=20.0, z_ohm=0.01)
        # Optional fields
        net.switch["name"] = pd.Series(["SW-A"], dtype="string")
        net.switch["type"] = pd.Series(["CB"], dtype="string")

        validate_network(net)

    def test_optional_fields_with_nulls(self):
        """Optional fields including nulls are valid"""
        net = pandapowerNet(name="test_optional_fields_with_nulls")
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)

        # Row 1: name/type set, in_ka/z_ohm null
        create_switch(net, bus=b0, element=b1, et="b", type="CB", closed=True)
        # Row 2: all optionals null
        create_switch(net, bus=b0, element=b1, et="b", closed=False)

        net.switch["name"] = pd.Series(["S1", pd.NA], dtype="string")
        net.switch["type"] = pd.Series(["CB", pd.NA], dtype="string")
        net.switch["in_ka"] = [float(np.nan), float(np.nan)]
        net.switch["z_ohm"] = [float(np.nan), float(np.nan)]

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
               itertools.product(["name"], [pd.NA, *strings]),
                itertools.product(["type"], [pd.NA, *strings]),
                itertools.product(["origin_id"], [pd.NA, *strings]),
                itertools.product(["origin_class"], [pd.NA, *strings]),
                itertools.product(["description"], [pd.NA, *strings]),
                itertools.product(["terminal_bus"], [pd.NA, *strings]),
                itertools.product(["terminal_element"], [pd.NA, *strings]),
            )
        ),
    )
    def test_valid_optional_values(self, parameter, valid_value):
        """Valid optional values are accepted"""
        net = pandapowerNet(name="test_valid_optional_values")
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)

        create_switch(net, bus=b0, element=b1, et="b", closed=True)

        net.switch[parameter] = pd.Series([valid_value], dtype="string")
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["name"], not_strings_list),
                itertools.product(["type"], not_strings_list),
                itertools.product(["origin_id"], not_strings_list),
                itertools.product(["origin_class"], not_strings_list),
                itertools.product(["description"], not_strings_list),
                itertools.product(["terminal_bus"], not_strings_list),
                itertools.product(["terminal_element"], not_strings_list),
            )
        ),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        """Invalid optional values are rejected"""
        net = pandapowerNet(name="test_invalid_optional_values")
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)

        create_switch(net, bus=b0, element=b1, et="b", closed=True, in_ka=1.0, z_ohm=0.01)
        net.switch[parameter] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestSwitchForeignKey:
    """Tests for foreign key constraints"""

    def test_invalid_bus_index(self):
        """bus must reference an existing bus index"""
        net = pandapowerNet(name="test_invalid_bus_index")
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)

        create_switch(net, bus=b0, element=b1, et="b", closed=True)

        net.switch["bus"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_valid_bus_index_non_sequential(self):
        """Test: bus FK works with non-sequential bus indices"""
        net = pandapowerNet(name="test_valid_bus_index_non_sequential")
        create_bus(net, 0.4, index=10)
        create_bus(net, 0.4, index=42)
        create_bus(net, 0.4, index=100)

        # Bus-bus switches with non-sequential indices
        create_switch(net, bus=10, element=42, et="b", closed=True)
        create_switch(net, bus=42, element=100, et="b", closed=False)
        create_switch(net, bus=100, element=10, et="b", closed=True)

        validate_network(net)


class TestSwitchResults:
    """Tests for switch results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_switch_result_totals(self):
        """Aggregated power and current results are consistent"""
        pass
