# test_pandera_switch_elements.py

import itertools
import numpy as np
import pandas as pd
import pandera as pa
import pytest

from pandapower.create import create_empty_network, create_bus, create_switch
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
    positiv_floats_plus_zero,
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
                itertools.product(["et"], [["b"], ["l"], ["t"], ["t3"]]),
                itertools.product(["closed"], bools),
                itertools.product(["in_ka"], positiv_floats),
                itertools.product(["z_ohm"], all_allowed_floats),
            )
        ),
    )
    def test_valid_required_values(self, parameter, valid_value):
        """Valid required values are accepted"""
        net = create_empty_network()
        create_bus(net, 0.4)  # 0
        create_bus(net, 0.4)  # 1
        create_bus(net, 0.4, index=42)

        # Base: bus-bus switch between bus 0 and bus 1
        create_switch(net, bus=0, element=1, et="b", type="CB", closed=True, in_ka=1.0, z_ohm=1.0)

        # Assign the parameter
        if parameter == "et":
            net.switch[parameter] = pd.Series(valid_value, dtype="string")
        else:
            net.switch[parameter] = valid_value

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter, invalid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], [*negativ_ints, *not_ints_list]),
                itertools.product(["element"], [*negativ_ints, *not_ints_list]),
                itertools.product(["et"], [*strings, *not_strings_list]),  # anything not in {"b","l","t","t3"}
                itertools.product(["closed"], not_boolean_list),
                itertools.product(["in_ka"], [*negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["z_ohm"], not_floats_list),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Invalid required values are rejected"""
        net = create_empty_network()
        create_bus(net, 0.4)  # 0
        create_bus(net, 0.4)  # 1

        create_switch(net, bus=0, element=1, et="b", type="CB", closed=True)

        net.switch[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    @pytest.mark.parametrize("parameter", ["bus", "element", "et", "closed"])
    def test_required_fields_nan_invalid(self, parameter):
        """NaN in required columns is invalid"""
        net = create_empty_network()
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
        net = create_empty_network()
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)

        create_switch(net, bus=b0, element=b1, et="b", type="CB", closed=False)
        # Optional fields
        net.switch["name"] = pd.Series(["SW-A"], dtype="string")
        net.switch["type"] = pd.Series(["CB"], dtype="string")
        net.switch["in_ka"] = 20.0  # gt(0)
        net.switch["z_ohm"] = 0.01  # any float

        validate_network(net)

    def test_optional_fields_with_nulls(self):
        """Optional fields including nulls are valid"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)

        # Row 1: name/type set, in_ka/z_ohm null
        create_switch(net, bus=b0, element=b1, et="b", type="CB", closed=True)
        net.switch["name"] = pd.Series(["S1"], dtype="string")
        net.switch["type"] = pd.Series(["CB"], dtype="string")
        net.switch["in_ka"] = [float(np.nan)]  # nullable
        net.switch["z_ohm"] = [float(np.nan)]  # nullable

        # Row 2: all optionals null
        create_switch(net, bus=b0, element=b1, et="b", closed=False)
        net.switch["name"].iat[1] = pd.NA
        net.switch["type"].iat[1] = pd.NA

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter, valid_value",
        list(
            itertools.chain(
                itertools.product(["name"], [pd.NA, *strings]),
                itertools.product(["type"], [pd.NA, *strings]),
            )
        ),
    )
    def test_valid_optional_values(self, parameter, valid_value):
        """Valid optional values are accepted"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)

        create_switch(net, bus=b0, element=b1, et="b", closed=True)

        if parameter in {"name", "type"}:
            net.switch[parameter] = pd.Series([valid_value], dtype="string")
        else:
            net.switch[parameter] = valid_value

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter, invalid_value",
        list(
            itertools.chain(
                itertools.product(["name"], not_strings_list),
                itertools.product(["type"], not_strings_list),
            )
        ),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        """Invalid optional values are rejected"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)

        create_switch(net, bus=b0, element=b1, et="b", closed=True)
        net.switch[parameter] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestSwitchForeignKey:
    """Tests for foreign key constraints"""

    def test_invalid_bus_index(self):
        """bus must reference an existing bus index"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)

        create_switch(net, bus=b0, element=b1, et="b", closed=True)

        net.switch["bus"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestSwitchResults:
    """Tests for switch results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_switch_result_totals(self):
        """Aggregated power and current results are consistent"""
        pass
