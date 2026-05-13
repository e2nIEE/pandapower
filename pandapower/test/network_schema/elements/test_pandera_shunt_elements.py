# test_pandera_shunt_elements.py

import itertools
import numpy as np
import pandas as pd
import pandera as pa
import pytest

from pandapower.create import create_bus, create_shunt
from pandapower.network import pandapowerNet
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
                itertools.product(["step"], positiv_floats),
                itertools.product(["in_service"], bools),
                itertools.product(["id_characteristic_table"], [pd.NA, *positiv_ints_plus_zero]), # TODO switch to false, when step_dependancy_table is gone
            )
        ),
    )
    def test_valid_required_values(self, parameter, valid_value):
        """Test: valid required values are accepted"""
        net = pandapowerNet(name="test_valid_required_values")
        create_bus(net, 0.4)  # index 0
        create_bus(net, 0.4)  # index 1
        create_bus(net, 0.4, index=42)

        create_shunt(
            net, bus=0, q_mvar=0.0, p_mw=0.0, in_service=True, vn_kv=0.4, step=1, id_characteristic_table=0, max_step=42
        )

        # Ensure step is valid for the step parameter test
        if parameter == "step":
            # step must be >= 1
            if valid_value < 1:
                valid_value = 1.0
                net.shunt[parameter] = valid_value
        elif parameter == "id_characteristic_table":
            net.shunt[parameter] = pd.Series([valid_value], dtype="Int64")
        else:
            net.shunt[parameter] = valid_value

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], [float(np.nan), pd.NA, *negativ_ints, *not_ints_list]),
                itertools.product(["p_mw"], [float(np.nan), pd.NA, *negativ_floats, *not_floats_list]),
                itertools.product(["q_mvar"], [float(np.nan), pd.NA, *not_floats_list]),
                itertools.product(["vn_kv"], [float(np.nan), pd.NA, *negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["step"], [float(np.nan), pd.NA, *negativ_floats_plus_zero, 0.5, *not_floats_list]),  # step must be >= 1
                itertools.product(["in_service"], [float(np.nan), pd.NA, *not_boolean_list]),
                itertools.product(["id_characteristic_table"], [float(np.nan), *not_ints_list]), # TODO switch to false, when step_dependancy_table is gone
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Test: invalid required values are rejected"""
        net = pandapowerNet(name="test_invalid_required_values")
        b0 = create_bus(net, 0.4)  # 0
        create_bus(net, 0.4)  # 1
        create_shunt(
            net,
            bus=b0,
            q_mvar=0.0,
            p_mw=0.0,
            in_service=True,
            vn_kv=0.4,
            step=1.0,
            id_characteristic_table=0,
        )

        net.shunt[parameter] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestShuntOptionalFields:
    """Tests for optional shunt fields"""

    def test_all_optional_fields_valid(self):
        """Test: shunt with all optional fields is valid"""
        net = pandapowerNet(name="test_all_optional_fields_valid")
        b0 = create_bus(net, 0.4)

        create_shunt(
            net,
            bus=b0,
            q_mvar=0.0,
            p_mw=0.0,
            in_service=True,
            vn_kv=0.4,
            step=2.0,
            id_characteristic_table=0,
        )

        # Optional fields
        net.shunt["name"] = pd.Series(["Shunt A"], dtype=pd.StringDtype())
        net.shunt["max_step"] = pd.Series([3], dtype="Int64")
        net.shunt["step_dependency_table"] = pd.Series([True], dtype=pd.BooleanDtype())
        net.shunt["id_characteristic_table"] = pd.Series([0], dtype="Int64")

        # CIM columns
        net.shunt["origin_id"] = pd.Series(["cim_id_1"], dtype=pd.StringDtype())
        net.shunt["origin_class"] = pd.Series(["ShuntCompensator"], dtype=pd.StringDtype())
        net.shunt["terminal"] = pd.Series(["term_1"], dtype=pd.StringDtype())
        net.shunt["description"] = pd.Series(["Test shunt"], dtype=pd.StringDtype())
        net.shunt["sVCControlMode"] = pd.Series(["voltage"], dtype=pd.StringDtype())

        # Check passes if step <= max_step (2 <= 3)
        validate_network(net)

    def test_optional_fields_with_nulls(self):
        """Test: optional fields including nulls are valid"""
        net = pandapowerNet(name="test_optional_fields_with_nulls")
        b0 = create_bus(net, 0.4)
        # Row 1: name only
        create_shunt(
            net,
            bus=b0,
            q_mvar=0.0,
            p_mw=0.0,
            in_service=True,
            vn_kv=0.4,
            step=1.0,
            id_characteristic_table=0,
        )

        # Row 2: max_step with NA name
        create_shunt(net, bus=b0, q_mvar=0.1, p_mw=0.0, in_service=False, vn_kv=0.4, step=2.0)

        # Row 3: step_dependency_table only
        create_shunt(net, bus=b0, q_mvar=-0.2, p_mw=0.1, in_service=True, vn_kv=0.4, step=1.0)

        # Set nullable columns with mixed values
        net.shunt["name"] = pd.Series(["alpha", pd.NA, pd.NA], dtype=pd.StringDtype())
        net.shunt["max_step"] = pd.Series([pd.NA, 3, pd.NA], dtype="Int64")
        net.shunt["step_dependency_table"] = pd.Series([pd.NA, pd.NA, True], dtype=pd.BooleanDtype())
        net.shunt["id_characteristic_table"] = pd.Series([0, pd.NA, pd.NA], dtype="Int64")

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["name"], [pd.NA, *strings]),
                itertools.product(["max_step"], [pd.NA, 1, 2, 5]),
                itertools.product(["step_dependency_table"], [pd.NA, True, False]),
                itertools.product(["id_characteristic_table"], [pd.NA, *positiv_ints_plus_zero]),
                itertools.product(["origin_id"], [pd.NA, *strings]),
                itertools.product(["origin_class"], [pd.NA, *strings]),
                itertools.product(["terminal"], [pd.NA, *strings]),
                itertools.product(["description"], [pd.NA, *strings]),
                itertools.product(["sVCControlMode"], [pd.NA, *strings]),
            )
        ),
    )
    def test_valid_optional_values(self, parameter, valid_value):
        """Test: valid optional values are accepted"""
        net = pandapowerNet(name="test_valid_optional_values")
        b0 = create_bus(net, 0.4)
        create_shunt(
            net,
            bus=b0,
            q_mvar=0.0,
            p_mw=0.0,
            in_service=True,
            vn_kv=0.4,
            step=1.0,
            id_characteristic_table=0,
        )

        # Handle dtype preservation for nullable columns
        if parameter in ["name", "origin_id", "origin_class", "terminal", "description", "sVCControlMode"]:
            net.shunt[parameter] = pd.Series([valid_value], dtype=pd.StringDtype())
        elif parameter == "max_step":
            net.shunt[parameter] = pd.Series([valid_value], dtype="Int64")
            # Ensure step <= max_step if max_step is not NA
            if valid_value is not pd.NA and valid_value is not None:
                net.shunt["step"] = min(float(net.shunt["step"].iat[0]), float(valid_value))
        elif parameter == "step_dependency_table":
            net.shunt[parameter] = pd.Series([valid_value], dtype=pd.BooleanDtype())
        elif parameter == "id_characteristic_table":
            net.shunt[parameter] = pd.Series([valid_value], dtype="Int64")
        else:
            net.shunt[parameter] = valid_value

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["name"], not_strings_list),
                itertools.product(["max_step"], [*negativ_ints_plus_zero, *not_ints_list]),
                itertools.product(["step_dependency_table"], not_boolean_list),
                itertools.product(["id_characteristic_table"], [*negativ_ints, *not_ints_list]),
                itertools.product(["origin_id"], not_strings_list),
                itertools.product(["origin_class"], not_strings_list),
                itertools.product(["terminal"], not_strings_list),
                itertools.product(["description"], not_strings_list),
                itertools.product(["sVCControlMode"], not_strings_list),
            )
        ),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        """Test: invalid optional values are rejected"""
        net = pandapowerNet(name="test_invalid_optional_values")
        b0 = create_bus(net, 0.4)
        create_shunt(
            net,
            bus=b0,
            q_mvar=0.0,
            p_mw=0.0,
            in_service=True,
            vn_kv=0.4,
            step=1.0,
        )

        # For id_characteristic_table with invalid int values, use proper dtype
        if parameter == "id_characteristic_table":
            if isinstance(invalid_value, (int, np.integer)) and not isinstance(invalid_value, (bool, np.bool_)):
                net.shunt[parameter] = pd.Series([invalid_value], dtype="Int64")
            else:
                net.shunt[parameter] = invalid_value
        else:
            net.shunt[parameter] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_step_less_equal_max_check_passes(self):
        """Test: 'step' <= 'max_step' passes"""
        net = pandapowerNet(name="test_step_less_equal_max_check_passes")
        b0 = create_bus(net, 0.4)
        create_shunt(
            net,
            bus=b0,
            q_mvar=0.0,
            p_mw=0.0,
            in_service=True,
            vn_kv=0.4,
            step=2.0,
        )
        net.shunt["max_step"] = pd.Series([3], dtype="Int64")

        validate_network(net)

    def test_step_equal_max_check_passes(self):
        """Test: 'step' == 'max_step' passes"""
        net = pandapowerNet(name="test_step_equal_max_check_passes")
        b0 = create_bus(net, 0.4)
        create_shunt(
            net,
            bus=b0,
            q_mvar=0.0,
            p_mw=0.0,
            in_service=True,
            vn_kv=0.4,
            step=3,
        )
        net.shunt["max_step"] = pd.Series([3], dtype="Int64")

        validate_network(net)

    def test_step_greater_than_max_fails(self):
        """Test: 'step' > 'max_step' fails"""
        net = pandapowerNet(name="test_step_greater_than_max_fails")
        b0 = create_bus(net, 0.4)
        create_shunt(
            net,
            bus=b0,
            q_mvar=0.0,
            p_mw=0.0,
            in_service=True,
            vn_kv=0.4,
            step=5.0,
            id_characteristic_table=0,
        )
        net.shunt["max_step"] = pd.Series([3], dtype="Int64")

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_max_step_na_step_any_valid(self):
        """Test: When max_step is NA, any valid step value passes"""
        net = pandapowerNet(name="test_max_step_na_step_any_valid")
        b0 = create_bus(net, 0.4)
        create_shunt(
            net,
            bus=b0,
            q_mvar=0.0,
            p_mw=0.0,
            in_service=True,
            vn_kv=0.4,
            step=100.0,
        )
        net.shunt["max_step"] = pd.Series([pd.NA], dtype="Int64")

        validate_network(net)


    def test_mixed_null_and_valid_values_in_rows(self):
        """Test: Multiple rows with mixed NA and valid values"""
        net = pandapowerNet(name="test_mixed_null_and_valid_values_in_rows")
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)

        # Row 1: all optional fields filled
        create_shunt(
            net,
            bus=b0,
            q_mvar=0.1,
            p_mw=0.0,
            in_service=True,
            vn_kv=0.4,
            step=2.0,
            name="Shunt A",
        )

        # Row 2: minimal fields
        create_shunt(
            net,
            bus=b1,
            q_mvar=-0.2,
            p_mw=0.05,
            in_service=False,
            vn_kv=0.4,
            step=1.0,
        )

        # Row 3: some optional fields
        create_shunt(
            net,
            bus=b0,
            q_mvar=0.3,
            p_mw=0.0,
            in_service=True,
            vn_kv=0.4,
            step=3.0,
        )

        # Set nullable columns with mixed values
        net.shunt["name"] = pd.Series(["Shunt A", pd.NA, "Shunt C"], dtype=pd.StringDtype())
        net.shunt["max_step"] = pd.Series([3, pd.NA, 5], dtype="Int64")
        net.shunt["step_dependency_table"] = pd.Series([True, pd.NA, pd.NA], dtype=pd.BooleanDtype())
        net.shunt["id_characteristic_table"] = pd.Series([0, pd.NA, 1], dtype="Int64")

        # CIM columns with mixed values
        net.shunt["origin_id"] = pd.Series(["cim_1", pd.NA, pd.NA], dtype=pd.StringDtype())
        net.shunt["origin_class"] = pd.Series([pd.NA, pd.NA, "ShuntCompensator"], dtype=pd.StringDtype())
        net.shunt["terminal"] = pd.Series([pd.NA, pd.NA, pd.NA], dtype=pd.StringDtype())
        net.shunt["description"] = pd.Series([pd.NA, "Desc 2", pd.NA], dtype=pd.StringDtype())

        validate_network(net)

    def test_cim_columns_all_na_valid(self):
        """Test: All CIM-related columns can be NA"""
        net = pandapowerNet(name="test_cim_columns_all_na_valid")
        b0 = create_bus(net, 0.4)

        create_shunt(
            net,
            bus=b0,
            q_mvar=0.0,
            p_mw=0.0,
            in_service=True,
            vn_kv=0.4,
            step=1.0,
        )

        # CIM columns from schema metadata
        cim_string_columns = ["name", "origin_id", "origin_class", "terminal", "description", "sVCControlMode"]

        for col in cim_string_columns:
            net.shunt[col] = pd.Series([pd.NA], dtype=pd.StringDtype())

        validate_network(net)


class TestShuntColumnConditionWithNulls:
    """Tests for column condition checks with null values"""

    def test_step_max_step_check_with_na_max_step(self):
        """Test: step <= max_step check passes when max_step is NA"""
        net = pandapowerNet(name="test_step_max_step_check_with_na_max_step")
        b0 = create_bus(net, 0.4)

        create_shunt(
            net,
            bus=b0,
            q_mvar=0.0,
            p_mw=0.0,
            in_service=True,
            vn_kv=0.4,
            step=10.0,  # Large value
        )

        # max_step is NA - check should pass
        net.shunt["max_step"] = pd.Series([pd.NA], dtype="Int64")

        validate_network(net)

    def test_step_max_step_check_row_wise_consistency(self):
        """Test: step <= max_step check works row-wise with mixed NA"""
        net = pandapowerNet(name="test_step_max_step_check_row_wise_consistency")
        b0 = create_bus(net, 0.4)

        # Row 1: step=2, max_step=3 (valid)
        create_shunt(net, bus=b0, q_mvar=0.1, p_mw=0.0, in_service=True, vn_kv=0.4, step=2.0)
        # Row 2: step=5, max_step=NA (valid because NA)
        create_shunt(net, bus=b0, q_mvar=0.2, p_mw=0.0, in_service=True, vn_kv=0.4, step=5.0)
        # Row 3: step=1, max_step=1 (valid, equal)
        create_shunt(net, bus=b0, q_mvar=0.3, p_mw=0.0, in_service=True, vn_kv=0.4, step=1.0)

        net.shunt["max_step"] = pd.Series([3, pd.NA, 1], dtype="Int64")

        validate_network(net)

    def test_step_max_step_check_row_wise_fails_when_exceeded(self):
        """Test: step <= max_step check fails when step > max_step in any row"""
        net = pandapowerNet(name="test_step_max_step_check_row_wise_fails_when_exceeded")
        b0 = create_bus(net, 0.4)

        # Row 1: step=2, max_step=3 (valid)
        create_shunt(net, bus=b0, q_mvar=0.1, p_mw=0.0, in_service=True, vn_kv=0.4, step=2.0)
        # Row 2: step=5, max_step=3 (invalid - step > max_step)
        create_shunt(net, bus=b0, q_mvar=0.2, p_mw=0.0, in_service=True, vn_kv=0.4, step=5.0)

        net.shunt["max_step"] = pd.Series([3, 3], dtype="Int64")

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestShuntForeignKey:
    """Tests for foreign key constraints"""

    def test_invalid_bus_index(self):
        """Test: bus must reference an existing bus index"""
        net = pandapowerNet(name="test_invalid_bus_index")
        b0 = create_bus(net, 0.4)
        create_shunt(
            net,
            bus=b0,
            q_mvar=0.0,
            p_mw=0.0,
            in_service=True,
            vn_kv=0.4,
            step=1,
        )

        net.shunt["bus"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_valid_bus_index_non_sequential(self):
        """Test: bus FK works with non-sequential bus indices"""
        net = pandapowerNet(name="test_valid_bus_index_non_sequential")
        create_bus(net, 0.4, index=10)
        create_bus(net, 0.4, index=42)
        create_bus(net, 0.4, index=100)

        create_shunt(net, bus=10, q_mvar=0.1, p_mw=0.0, in_service=True, vn_kv=0.4, step=10, max_step=100)
        create_shunt(net, bus=42, q_mvar=0.2, p_mw=0.0, in_service=True, vn_kv=0.4, step=20, max_step=100)
        create_shunt(net, bus=100, q_mvar=-0.1, p_mw=0.05, in_service=False, vn_kv=0.4, step=10, max_step=100)

        validate_network(net)


class TestShuntResults:
    """Tests for shunt results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_shunt_result_totals(self):
        """Test: aggregated p_mw / q_mvar results are consistent"""
        pass