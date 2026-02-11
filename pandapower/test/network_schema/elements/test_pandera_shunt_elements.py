# test_pandera_shunt_elements.py

import itertools
import pandas as pd
import pandera as pa
import pytest
import numpy as np

from pandapower.create import create_empty_network, create_bus, create_shunt
from pandapower.network_schema.tools.validation.network_validation import validate_network

from pandapower.test.network_schema.elements.helper import (
    strings,
    bools,
    not_strings_list,
    not_floats_list,
    not_boolean_list,
    positiv_ints_plus_zero,
    positiv_floats,
    negativ_floats,
    positiv_floats_plus_zero,
    negativ_floats_plus_zero,
    positiv_ints,
    negativ_ints_plus_zero,
    negativ_ints,
    not_ints_list,
    all_allowed_floats,
)


class TestShuntRequiredFields:
    """Tests for required shunt fields"""

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], positiv_ints_plus_zero),
                itertools.product(["p_mw"], positiv_floats_plus_zero),
                itertools.product(["q_mvar"], all_allowed_floats),
                itertools.product(["vn_kv"], positiv_floats),
                itertools.product(["step"], positiv_ints),
                itertools.product(["in_service"], bools),
                itertools.product(["id_characteristic_table"], [pd.NA, *positiv_ints_plus_zero]),
            )
        ),
    )
    def test_valid_required_values(self, parameter, valid_value):
        """Test: valid required values are accepted"""
        net = create_empty_network()
        create_bus(net, 0.4)  # index 0
        create_bus(net, 0.4)  # index 1
        create_bus(net, 0.4, index=42)

        create_shunt(
            net, bus=0, q_mvar=0.0, p_mw=0.0, in_service=True, vn_kv=0.4, step=1, id_characteristic_table=0, max_step=42
        )

        if parameter == "id_characteristic_table":
            net.shunt[parameter] = pd.Series([valid_value], dtype="Int64")
        else:
            net.shunt[parameter] = valid_value

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], [*negativ_ints, *not_ints_list]),
                itertools.product(["p_mw"], [*negativ_floats, *not_floats_list]),
                itertools.product(["q_mvar"], not_floats_list),
                itertools.product(["vn_kv"], [*negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["step"], [*negativ_ints_plus_zero, *not_ints_list]),
                itertools.product(["in_service"], not_boolean_list),
                itertools.product(["id_characteristic_table"], not_ints_list),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Test: invalid required values are rejected"""
        net = create_empty_network()
        create_bus(net, 0.4)  # 0
        create_bus(net, 0.4)  # 1

        create_shunt(net, bus=0, q_mvar=0.0, p_mw=0.0, in_service=True)
        net.shunt["p_mw"] = 0.0
        net.shunt["q_mvar"] = 0.0
        net.shunt["vn_kv"] = 0.4
        net.shunt["step"] = 1
        net.shunt["in_service"] = True
        net.shunt["id_characteristic_table"] = pd.Series([0], dtype="Int64")

        if parameter == "id_characteristic_table":
            # If invalid_value is an int, keep Int64 dtype to trigger 'ge(0)' check;
            # otherwise assign as-is to trigger dtype mismatch.
            if isinstance(invalid_value, (int, np.integer)) and not isinstance(invalid_value, (bool, np.bool_)):
                net.shunt[parameter] = pd.Series([invalid_value], dtype="Int64")
            else:
                net.shunt[parameter] = invalid_value
        else:
            net.shunt[parameter] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestShuntOptionalFields:
    """Tests for optional shunt fields"""

    def test_all_optional_fields_valid(self):
        """Test: shunt with all optional fields is valid"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)
        create_bus(net, 0.4)

        create_shunt(net, bus=b0, q_mvar=0.2, p_mw=0.0, in_service=True)
        # Required fields
        net.shunt["vn_kv"] = 0.4
        net.shunt["step"] = 2
        net.shunt["id_characteristic_table"] = pd.Series([0], dtype="Int64")
        # Optional fields
        net.shunt["name"] = pd.Series(["Shunt A"], dtype=pd.StringDtype())
        net.shunt["max_step"] = pd.Series([3], dtype="Int64")
        net.shunt["step_dependency_table"] = pd.Series([True], dtype=pd.BooleanDtype())

        # Check passes if step <= max_step (2 <= 3)
        validate_network(net)

    def test_optional_fields_with_nulls(self):
        """Test: optional fields including nulls are valid"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        # Row 1: name only
        create_shunt(net, bus=b0, q_mvar=0.0, p_mw=0.0, in_service=True)
        net.shunt["vn_kv"].iat[0] = 0.4
        net.shunt["step"].iat[0] = 1
        net.shunt["id_characteristic_table"] = pd.Series([0], dtype="Int64")
        net.shunt["name"] = pd.Series(["alpha"], dtype=pd.StringDtype())

        # Row 2: max_step with NA name and NA step_dependency_table
        create_shunt(net, bus=b0, q_mvar=0.1, p_mw=0.0, in_service=False)
        net.shunt["vn_kv"].iat[1] = 0.4
        net.shunt["step"].iat[1] = 2
        net.shunt["id_characteristic_table"].iat[1] = pd.NA
        net.shunt["name"].iat[1] = pd.NA
        # max_step present; ensure step <= max_step
        max_step_series = net.shunt.get("max_step", pd.Series([pd.NA] * len(net.shunt), dtype="Int64"))
        if len(max_step_series) < len(net.shunt):
            # Extend to match length
            max_step_series = pd.concat([max_step_series, pd.Series([pd.NA], dtype="Int64")], ignore_index=True)
        max_step_series.iat[1] = 3
        net.shunt["max_step"] = max_step_series
        net.shunt["step_dependency_table"] = pd.Series([pd.NA, pd.NA], dtype=pd.BooleanDtype())

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["name"], [pd.NA, *strings]),
                itertools.product(["max_step"], [1, 2, 5]),
                itertools.product(["step_dependency_table"], [True, False]),
            )
        ),
    )
    def test_valid_optional_values(self, parameter, valid_value):
        """Test: valid optional values are accepted"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        create_shunt(net, bus=b0, q_mvar=0.0, p_mw=0.0, in_service=True)
        # Required fields
        net.shunt["vn_kv"] = 0.4
        net.shunt["step"] = 1
        net.shunt["id_characteristic_table"] = pd.Series([0], dtype="Int64")

        if parameter == "name":
            net.shunt[parameter] = pd.Series([valid_value], dtype=pd.StringDtype())
        elif parameter == "max_step":
            net.shunt[parameter] = pd.Series([valid_value], dtype="Int64")
            # Ensure step <= max_step
            net.shunt["step"] = min(int(net.shunt["step"].iat[0]), int(valid_value))
        elif parameter == "step_dependency_table":
            net.shunt[parameter] = pd.Series([valid_value], dtype=pd.BooleanDtype())
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["name"], not_strings_list),
                itertools.product(["max_step"], [*negativ_ints_plus_zero, *not_ints_list]),
                itertools.product(["step_dependency_table"], not_boolean_list),
            )
        ),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        """Test: invalid optional values are rejected"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        create_shunt(net, bus=b0, q_mvar=0.0, p_mw=0.0, in_service=True)
        net.shunt["vn_kv"] = 0.4
        net.shunt["step"] = 1
        net.shunt["id_characteristic_table"] = pd.Series([0], dtype="Int64")

        if parameter == "name":
            net.shunt[parameter] = invalid_value
        elif parameter == "max_step":
            net.shunt[parameter] = invalid_value
        elif parameter == "step_dependency_table":
            net.shunt[parameter] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_step_less_equal_max_check_passes(self):
        """Test: 'step' <= 'max_step' passes"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        create_shunt(net, bus=b0, q_mvar=0.0, p_mw=0.0, in_service=True)
        net.shunt["vn_kv"] = 0.4
        net.shunt["step"] = 2
        net.shunt["max_step"] = pd.Series([3], dtype="Int64")
        net.shunt["id_characteristic_table"] = pd.Series([0], dtype="Int64")
        validate_network(net)

    def test_step_greater_than_max_fails(self):
        """Test: 'step' > 'max_step' fails"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        create_shunt(net, bus=b0, q_mvar=0.0, p_mw=0.0, in_service=True)
        net.shunt["vn_kv"] = 0.4
        net.shunt["step"] = 5
        net.shunt["max_step"] = pd.Series([3], dtype="Int64")
        net.shunt["id_characteristic_table"] = pd.Series([0], dtype="Int64")

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestShuntForeignKey:
    """Tests for foreign key constraints"""

    def test_invalid_bus_index(self):
        """Test: bus must reference an existing bus index"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        create_shunt(net, bus=b0, q_mvar=0.0, p_mw=0.0, in_service=True)
        net.shunt["vn_kv"] = 0.4
        net.shunt["step"] = 1
        net.shunt["id_characteristic_table"] = pd.Series([0], dtype="Int64")

        net.shunt["bus"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestShuntResults:
    """Tests for shunt results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_shunt_result_totals(self):
        """Test: aggregated p_mw / q_mvar results are consistent"""
        pass
