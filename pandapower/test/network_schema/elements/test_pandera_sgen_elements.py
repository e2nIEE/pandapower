# test_pandera_sgen_elements.py

import itertools
import numpy as np
import pandas as pd
import pandera as pa
import pytest

from pandapower.create import create_bus, create_sgen
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
    positiv_floats_plus_zero,
    negativ_floats_plus_zero,
    negativ_floats,
    not_ints_list,
    negativ_ints,
    all_allowed_floats,
    all_allowed_ints,
)


class TestSgenRequiredFields:
    """Tests for required sgen fields"""

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], positiv_ints_plus_zero),
                itertools.product(["p_mw"], all_allowed_floats),
                itertools.product(["q_mvar"], all_allowed_floats),
                itertools.product(["scaling"], positiv_floats_plus_zero),
                itertools.product(["in_service"], bools),
            )
        ),
    )
    def test_valid_required_values(self, parameter, valid_value):
        net = pandapowerNet(name="test_valid_required_values")
        create_bus(net, 0.4)  # 0
        create_bus(net, 0.4)  # 1
        create_bus(net, 0.4, index=42)

        create_sgen(net, bus=0, p_mw=1.0, q_mvar=0.0, scaling=1.0, in_service=True)
        net.sgen[parameter] = valid_value
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], [float(np.nan), pd.NA, *negativ_ints, *not_ints_list]),
                itertools.product(["p_mw"], [float(np.nan), pd.NA, *not_floats_list]),
                itertools.product(["q_mvar"], [float(np.nan), pd.NA, *not_floats_list]),
                itertools.product(["scaling"], [float(np.nan), pd.NA, *negativ_floats, *not_floats_list]),
                itertools.product(["in_service"], [float(np.nan), pd.NA, *not_boolean_list]),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        net = pandapowerNet(name="test_invalid_required_values")
        create_bus(net, 0.4)
        create_bus(net, 0.4)

        create_sgen(net, bus=0, p_mw=1.0, q_mvar=0.0, scaling=1.0, in_service=True)
        net.sgen[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestSgenOptionalFields:
    """Tests for optional sgen fields and group dependencies (opf, qcc)"""

    def test_all_optional_fields_valid(self):
        net = pandapowerNet(name="test_all_optional_fields_valid")
        b0 = create_bus(net, 0.4)

        # Create base sgen
        create_sgen(net, bus=b0, p_mw=1.0, q_mvar=0.2, scaling=1.0, in_service=True)

        # String/boolean optionals
        net.sgen["name"] = pd.Series(["SGen A"], dtype="string")
        net.sgen["type"] = pd.Series(["PV"], dtype="string")
        net.sgen["controllable"] = pd.Series([True], dtype=bool)

        # OPF group (complete)
        net.sgen["max_p_mw"] = 2.0
        net.sgen["min_p_mw"] = -2.0
        net.sgen["max_q_mvar"] = 1.0
        net.sgen["min_q_mvar"] = -1.0

        # QCC group (complete)
        net.sgen["id_q_capability_characteristic"] = pd.Series([0], dtype="Int64")
        net.sgen["curve_style"] = pd.Series(["straightLineYValues"], dtype="string")
        net.sgen["reactive_capability_curve"] = pd.Series([True], dtype="boolean")

        # SC-related optionals (not enforced by group dependency in schema)
        net.sgen["sn_mva"] = 1.0
        net.sgen["k"] = 0.0
        net.sgen["rx"] = 0.0
        net.sgen["current_source"] = pd.Series([False], dtype="boolean")
        net.sgen["generator_type"] = pd.Series(["async"], dtype="string")
        net.sgen["lrc_pu"] = 5.0
        net.sgen["max_ik_ka"] = 10.0
        net.sgen["kappa"] = 1.8

        validate_network(net)

    def test_optional_fields_with_nulls(self):
        """Optional fields incl. nulls; groups satisfied when present"""
        net = pandapowerNet(name="test_optional_fields_with_nulls")
        b0 = create_bus(net, 0.4)

        # Row 1: OPF group complete
        create_sgen(
            net,
            bus=b0,
            p_mw=0.8,
            q_mvar=0.1,
            scaling=1.0,
            in_service=True,
            max_p_mw=1.5,
            min_p_mw=-1.0,
            max_q_mvar=0.8,
            min_q_mvar=-0.6,
            name="alpha",
        )
        # Row 2: QCC group complete
        create_sgen(
            net,
            bus=b0,
            p_mw=1.2,
            q_mvar=0.0,
            scaling=0.9,
            in_service=True,
            id_q_capability_characteristic=1,
            curve_style="constantYValue",
            reactive_capability_curve=True,
            type="wye",
        )

        # Row 3: other optionals without triggering groups
        create_sgen(net, bus=b0, p_mw=2.0, q_mvar=-0.2, scaling=1.1, in_service=False, sn_mva=2.0)
        net.sgen["name"] = pd.Series(["alpha", pd.NA, "gamma"], dtype="string")

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                # Non-group nullable columns - can include pd.NA / float(np.nan)
                itertools.product(["name"], [pd.NA, *strings]),
                itertools.product(["type"], [pd.NA, *strings]),
                itertools.product(["sn_mva"], [float(np.nan), *positiv_floats]),
                itertools.product(["controllable"], bools),
                itertools.product(["k"], [float(np.nan), *positiv_floats_plus_zero]),
                itertools.product(["rx"], [float(np.nan), *positiv_floats_plus_zero]),
                itertools.product(["current_source"], [pd.NA, *bools]),
                itertools.product(["generator_type"], [pd.NA, "current_source", "async", "async_doubly_fed"]),
                itertools.product(["lrc_pu"], [float(np.nan), *all_allowed_floats]),
                itertools.product(["max_ik_ka"], [float(np.nan), *all_allowed_floats]),
                itertools.product(["kappa"], [float(np.nan), *all_allowed_floats]),
                # OPF group columns - test NON-NULL values only
                itertools.product(["max_p_mw"], all_allowed_floats),
                itertools.product(["min_p_mw"], all_allowed_floats),
                itertools.product(["max_q_mvar"], all_allowed_floats),
                itertools.product(["min_q_mvar"], all_allowed_floats),
                # QCC group columns - test NON-NULL values only
                itertools.product(["id_q_capability_characteristic"], all_allowed_ints),
                itertools.product(["curve_style"], ["straightLineYValues", "constantYValue"]),
                itertools.product(["reactive_capability_curve"], bools),
            )
        ),
    )
    def test_valid_optional_values(self, parameter, valid_value):
        net = pandapowerNet(name="test_valid_optional_values")
        b0 = create_bus(net, 0.4)

        create_sgen(net, bus=b0, p_mw=1.0, q_mvar=0.0, scaling=1.0, in_service=True)

        # Satisfy OPF + QCC groups so target column won't fail on dependency
        net.sgen["max_p_mw"] = 2.0
        net.sgen["min_p_mw"] = -2.0
        net.sgen["max_q_mvar"] = 1.0
        net.sgen["min_q_mvar"] = -1.0
        net.sgen["id_q_capability_characteristic"] = pd.Series([0], dtype="Int64")
        net.sgen["curve_style"] = pd.Series(["straightLineYValues"], dtype="string")
        net.sgen["reactive_capability_curve"] = pd.Series([True], dtype="boolean")

        # Handle nullable types properly
        if parameter in {"name", "type", "curve_style", "generator_type"}:
            net.sgen[parameter] = pd.Series([valid_value], dtype="string")
        elif parameter in {"current_source", "reactive_capability_curve"}:
            net.sgen[parameter] = pd.Series([valid_value], dtype=pd.BooleanDtype())
        elif parameter == "id_q_capability_characteristic":
            net.sgen[parameter] = pd.Series([valid_value], dtype="Int64")
        else:
            net.sgen[parameter] = valid_value

        validate_network(net)

    def test_opf_group_partial_missing_invalid(self):
        net = pandapowerNet(name="test_opf_group_partial_missing_invalid")
        b0 = create_bus(net, 0.4)
        create_sgen(net, bus=b0, p_mw=1.0, q_mvar=0.0, scaling=1.0, in_service=True)

        # Set only one OPF column
        net.sgen["max_p_mw"] = 100.0
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net, "opf")

    @pytest.mark.xfail  # TODO add back when reactive_capability_curve is removed
    def test_qcc_group_partial_missing_invalid(self):
        # Only id_q_capability_characteristic
        net = pandapowerNet(name="test_qcc_group_partial_missing_invalid0")
        b0 = create_bus(net, 0.4)
        create_sgen(net, bus=b0, p_mw=1.0, q_mvar=0.0, scaling=1.0, in_service=True)
        net.sgen["id_q_capability_characteristic"] = pd.Series([0], dtype="Int64")
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net, "qcc")

        # Only curve_style
        net = pandapowerNet(name="test_qcc_group_partial_missing_invalid1")
        b0 = create_bus(net, 0.4)
        create_sgen(net, bus=b0, p_mw=1.0, q_mvar=0.0, scaling=1.0, in_service=True)
        create_sgen(net, bus=b0, p_mw=1.0, q_mvar=0.0, scaling=1.0, in_service=True)
        net.sgen["curve_style"] = pd.Series([pd.NA, "straightLineYValues"], dtype="string")
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net, "qcc")

        # Only reactive_capability_curve
        net = pandapowerNet(name="test_qcc_group_partial_missing_invalid2")
        b0 = create_bus(net, 0.4)
        create_sgen(net, bus=b0, p_mw=1.0, q_mvar=0.0, scaling=1.0, in_service=True)
        create_sgen(net, bus=b0, p_mw=1.0, q_mvar=0.0, scaling=1.0, in_service=True)
        create_sgen(net, bus=b0, p_mw=1.0, q_mvar=0.0, scaling=1.0, in_service=True)
        create_sgen(net, bus=b0, p_mw=1.0, q_mvar=0.0, scaling=1.0, in_service=True)
        net.sgen["reactive_capability_curve"] = pd.Series([pd.NA, pd.NA, True], dtype="boolean")
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net, "qcc")

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["name"], not_strings_list),
                itertools.product(["type"], not_strings_list),
                itertools.product(["sn_mva"], [*negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["max_p_mw"], not_floats_list),
                itertools.product(["min_p_mw"], not_floats_list),
                itertools.product(["max_q_mvar"], not_floats_list),
                itertools.product(["min_q_mvar"], not_floats_list),
                itertools.product(["controllable"], not_boolean_list),
                itertools.product(["k"], [*negativ_floats, *not_floats_list]),
                itertools.product(["rx"], [*negativ_floats, *not_floats_list]),
                itertools.product(["current_source"], not_boolean_list),
                itertools.product(["generator_type"], not_strings_list),
                itertools.product(["lrc_pu"], not_floats_list),
                itertools.product(["max_ik_ka"], not_floats_list),
                itertools.product(["kappa"], not_floats_list),
                itertools.product(["id_q_capability_characteristic"], not_ints_list),
                itertools.product(["curve_style"], not_strings_list),
                itertools.product(["reactive_capability_curve"], not_boolean_list),
            )
        ),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        net = pandapowerNet(name="test_invalid_optional_values")
        b0 = create_bus(net, 0.4)

        create_sgen(net, bus=b0, p_mw=1.0, q_mvar=0.0, scaling=1.0, in_service=True)

        # Provide complete OPF + QCC groups
        net.sgen["max_p_mw"] = 2.0
        net.sgen["min_p_mw"] = -2.0
        net.sgen["max_q_mvar"] = 1.0
        net.sgen["min_q_mvar"] = -1.0
        net.sgen["id_q_capability_characteristic"] = pd.Series([0], dtype="Int64")
        net.sgen["curve_style"] = pd.Series(["straightLineYValues"], dtype="string")
        net.sgen["reactive_capability_curve"] = pd.Series([True], dtype="boolean")

        net.sgen[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestSgenDependencyGroupNullValues:
    """Tests for nullable dependency group columns"""

    def test_opf_group_all_nan_valid(self):
        """Test: OPF group columns can all be NaN together (group not triggered)"""
        net = pandapowerNet(name="test_opf_group_all_nan_valid")
        b0 = create_bus(net, 0.4)

        create_sgen(net, bus=b0, p_mw=1.0, q_mvar=0.0, scaling=1.0, in_service=True)

        # Set all OPF columns to NaN
        net.sgen["max_p_mw"] = float(np.nan)
        net.sgen["min_p_mw"] = float(np.nan)
        net.sgen["max_q_mvar"] = float(np.nan)
        net.sgen["min_q_mvar"] = float(np.nan)

        validate_network(net)

    def test_qcc_group_all_na_valid(self):
        """Test: QCC group columns can all be NA/NaN together (group not triggered)"""
        net = pandapowerNet(name="test_qcc_group_all_na_valid")
        b0 = create_bus(net, 0.4)

        create_sgen(net, bus=b0, p_mw=1.0, q_mvar=0.0, scaling=1.0, in_service=True)

        # Set all QCC columns to NA/NaN with correct dtypes
        net.sgen["id_q_capability_characteristic"] = pd.Series([pd.NA], dtype="Int64")
        net.sgen["curve_style"] = pd.Series([pd.NA], dtype=pd.StringDtype())
        net.sgen["reactive_capability_curve"] = pd.Series([pd.NA], dtype=pd.BooleanDtype())

        validate_network(net)

    def test_mixed_null_and_valid_values_in_rows(self):
        """Test: Multiple rows with mixed NA and valid values"""
        net = pandapowerNet(name="test_mixed_null_and_valid_values_in_rows")
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)

        # Row 1: OPF group complete
        create_sgen(
            net,
            bus=b0,
            p_mw=1.0,
            q_mvar=0.1,
            scaling=1.0,
            in_service=True,
            name="SGen A",
            sn_mva=1.0,
            type="PV",
            max_p_mw=2.0,
            min_p_mw=-1.0,
            max_q_mvar=1.0,
            min_q_mvar=-0.5,
        )

        # Row 2: all optional fields NA/NaN (OPF group all NaN)
        create_sgen(
            net,
            bus=b1,
            p_mw=2.0,
            q_mvar=0.2,
            scaling=0.8,
            in_service=False,
        )

        # Row 3: QCC group complete, OPF group all NaN
        create_sgen(
            net,
            bus=b0,
            p_mw=0.5,
            q_mvar=0.05,
            scaling=1.2,
            in_service=True,
        )

        # Set nullable columns with mixed values
        net.sgen["name"] = pd.Series(["SGen A", pd.NA, pd.NA], dtype=pd.StringDtype())
        net.sgen["type"] = pd.Series(["PV", pd.NA, "WP"], dtype=pd.StringDtype())
        net.sgen["sn_mva"] = [1.0, float(np.nan), float(np.nan)]

        # OPF columns - Row 1 has values, Row 2 and 3 have NaN (group consistency per row)
        net.sgen["max_p_mw"] = [2.0, float(np.nan), float(np.nan)]
        net.sgen["min_p_mw"] = [-1.0, float(np.nan), float(np.nan)]
        net.sgen["max_q_mvar"] = [1.0, float(np.nan), float(np.nan)]
        net.sgen["min_q_mvar"] = [-0.5, float(np.nan), float(np.nan)]

        # QCC columns - Row 3 has values, others have NA/NaN (group consistency per row)
        net.sgen["id_q_capability_characteristic"] = pd.Series([pd.NA, pd.NA, 0], dtype="Int64")
        net.sgen["curve_style"] = pd.Series([pd.NA, pd.NA, "straightLineYValues"], dtype=pd.StringDtype())
        net.sgen["reactive_capability_curve"] = pd.Series([pd.NA, pd.NA, True], dtype=pd.BooleanDtype())

        # SC-related columns with mixed values (not in group dependency)
        net.sgen["k"] = [0.5, float(np.nan), float(np.nan)]
        net.sgen["rx"] = [float(np.nan), 0.1, float(np.nan)]
        net.sgen["current_source"] = pd.Series([True, pd.NA, pd.NA], dtype=pd.BooleanDtype())

        # CIM columns with mixed values
        net.sgen["origin_id"] = pd.Series(["cim_1", pd.NA, pd.NA], dtype=pd.StringDtype())
        net.sgen["origin_class"] = pd.Series([pd.NA, pd.NA, "GeneratingUnit"], dtype=pd.StringDtype())

        validate_network(net)

    def test_cim_columns_all_na_valid(self):
        """Test: All CIM-related columns can be NA"""
        net = pandapowerNet(name="test_cim_columns_all_na_valid")
        b0 = create_bus(net, 0.4)

        create_sgen(net, bus=b0, p_mw=1.0, q_mvar=0.0, scaling=1.0, in_service=True)

        # CIM columns from schema metadata
        cim_string_columns = ["name", "origin_id", "origin_class", "terminal", "description", "type"]

        for col in cim_string_columns:
            net.sgen[col] = pd.Series([pd.NA], dtype=pd.StringDtype())

        validate_network(net)

    def test_opf_group_row_consistency_valid(self):
        """Test: OPF group - each row must have all values or all NaN"""
        net = pandapowerNet(name="test_opf_group_row_consistency_valid")
        b0 = create_bus(net, 0.4)

        # Row 1: all OPF values present
        create_sgen(net, bus=b0, p_mw=1.0, q_mvar=0.0, scaling=1.0, in_service=True)
        # Row 2: all OPF values NaN
        create_sgen(net, bus=b0, p_mw=2.0, q_mvar=0.1, scaling=1.0, in_service=True)

        net.sgen["max_p_mw"] = [2.0, float(np.nan)]
        net.sgen["min_p_mw"] = [-2.0, float(np.nan)]
        net.sgen["max_q_mvar"] = [1.0, float(np.nan)]
        net.sgen["min_q_mvar"] = [-1.0, float(np.nan)]

        validate_network(net)

    def test_opf_group_row_partial_invalid(self):
        """Test: OPF group - partial values in a row should fail"""
        net = pandapowerNet(name="test_opf_group_row_partial_invalid")
        b0 = create_bus(net, 0.4)

        create_sgen(net, bus=b0, p_mw=1.0, q_mvar=0.0, scaling=1.0, in_service=True)
        create_sgen(net, bus=b0, p_mw=2.0, q_mvar=0.1, scaling=1.0, in_service=True)

        # Row 1: all OPF values present
        # Row 2: partial OPF values
        net.sgen["max_p_mw"] = [2.0, 3.0]
        net.sgen["min_p_mw"] = [-2.0, float(np.nan)]
        net.sgen["max_q_mvar"] = [1.0, float(np.nan)]
        net.sgen["min_q_mvar"] = [-1.0, float(np.nan)]

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_qcc_group_row_consistency_valid(self):
        """Test: QCC group - each row must have all values or all NA"""
        net = pandapowerNet(name="test_qcc_group_row_consistency_valid")
        b0 = create_bus(net, 0.4)

        # Row 1: all QCC values present
        create_sgen(net, bus=b0, p_mw=1.0, q_mvar=0.0, scaling=1.0, in_service=True)
        # Row 2: all QCC values NA
        create_sgen(net, bus=b0, p_mw=2.0, q_mvar=0.1, scaling=1.0, in_service=True)

        net.sgen["id_q_capability_characteristic"] = pd.Series([0, pd.NA], dtype="Int64")
        net.sgen["curve_style"] = pd.Series(["straightLineYValues", pd.NA], dtype=pd.StringDtype())
        net.sgen["reactive_capability_curve"] = pd.Series([True, pd.NA], dtype=pd.BooleanDtype())

        validate_network(net)

    @pytest.mark.xfail  # TODO add back when reactive_capability_curve is removed
    def test_qcc_group_row_partial_invalid(self):
        """Test: QCC group - partial values in a row should fail"""
        net = pandapowerNet(name="test_qcc_group_row_partial_invalid")
        b0 = create_bus(net, 0.4)

        create_sgen(net, bus=b0, p_mw=1.0, q_mvar=0.0, scaling=1.0, in_service=True)
        create_sgen(net, bus=b0, p_mw=2.0, q_mvar=0.1, scaling=1.0, in_service=True)

        # Row 1: all QCC values present
        # Row 2: partial QCC values (only id set) - should fail
        net.sgen["id_q_capability_characteristic"] = pd.Series([0, 1], dtype="Int64")
        net.sgen["curve_style"] = pd.Series(["straightLineYValues", pd.NA], dtype=pd.StringDtype())
        net.sgen["reactive_capability_curve"] = pd.Series([True, pd.NA], dtype=pd.BooleanDtype())

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestSgenDependencyGroupNullValues:
    """Tests for nullable dependency group columns"""

    def test_opf_group_all_nan_valid(self):
        """Test: OPF group columns can all be NaN together (group not triggered)"""
        net = pandapowerNet(name="test_opf_group_all_nan_valid")
        b0 = create_bus(net, 0.4)

        create_sgen(net, bus=b0, p_mw=1.0, q_mvar=0.0, scaling=1.0, in_service=True)

        # Set all OPF columns to NaN
        net.sgen["max_p_mw"] = float(np.nan)
        net.sgen["min_p_mw"] = float(np.nan)
        net.sgen["max_q_mvar"] = float(np.nan)
        net.sgen["min_q_mvar"] = float(np.nan)

        validate_network(net)

    def test_qcc_group_all_na_valid(self):
        """Test: QCC group columns can all be NA/NaN together (group not triggered)"""
        net = pandapowerNet(name="test_qcc_group_all_na_valid")
        b0 = create_bus(net, 0.4)

        create_sgen(net, bus=b0, p_mw=1.0, q_mvar=0.0, scaling=1.0, in_service=True)

        # Set all QCC columns to NA/NaN with correct dtypes
        net.sgen["id_q_capability_characteristic"] = pd.Series([pd.NA], dtype="Int64")
        net.sgen["curve_style"] = pd.Series([pd.NA], dtype=pd.StringDtype())
        net.sgen["reactive_capability_curve"] = pd.Series([pd.NA], dtype=pd.BooleanDtype())

        validate_network(net)

    def test_mixed_null_and_valid_values_in_rows(self):
        """Test: Multiple rows with mixed NA and valid values"""
        net = pandapowerNet(name="test_mixed_null_and_valid_values_in_rows")
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)

        # Row 1: OPF group complete
        create_sgen(
            net,
            bus=b0,
            p_mw=1.0,
            q_mvar=0.1,
            scaling=1.0,
            in_service=True,
            name="SGen A",
            sn_mva=1.0,
            type="PV",
            max_p_mw=2.0,
            min_p_mw=-1.0,
            max_q_mvar=1.0,
            min_q_mvar=-0.5,
        )

        # Row 2: all optional fields NA/NaN (OPF group all NaN)
        create_sgen(
            net,
            bus=b1,
            p_mw=2.0,
            q_mvar=0.2,
            scaling=0.8,
            in_service=False,
        )

        # Row 3: QCC group complete, OPF group all NaN
        create_sgen(
            net,
            bus=b0,
            p_mw=0.5,
            q_mvar=0.05,
            scaling=1.2,
            in_service=True,
        )

        # Set nullable columns with mixed values
        net.sgen["name"] = pd.Series(["SGen A", pd.NA, pd.NA], dtype=pd.StringDtype())
        net.sgen["type"] = pd.Series(["PV", pd.NA, "WP"], dtype=pd.StringDtype())
        net.sgen["sn_mva"] = [1.0, float(np.nan), float(np.nan)]

        # OPF columns - Row 1 has values, Row 2 and 3 have NaN (group consistency per row)
        net.sgen["max_p_mw"] = [2.0, float(np.nan), float(np.nan)]
        net.sgen["min_p_mw"] = [-1.0, float(np.nan), float(np.nan)]
        net.sgen["max_q_mvar"] = [1.0, float(np.nan), float(np.nan)]
        net.sgen["min_q_mvar"] = [-0.5, float(np.nan), float(np.nan)]

        # QCC columns - Row 3 has values, others have NA/NaN (group consistency per row)
        net.sgen["id_q_capability_characteristic"] = pd.Series([pd.NA, pd.NA, 0], dtype="Int64")
        net.sgen["curve_style"] = pd.Series([pd.NA, pd.NA, "straightLineYValues"], dtype=pd.StringDtype())
        net.sgen["reactive_capability_curve"] = pd.Series([pd.NA, pd.NA, True], dtype=pd.BooleanDtype())

        # SC-related columns with mixed values (not in group dependency)
        net.sgen["k"] = [0.5, float(np.nan), float(np.nan)]
        net.sgen["rx"] = [float(np.nan), 0.1, float(np.nan)]
        net.sgen["current_source"] = pd.Series([True, pd.NA, pd.NA], dtype=pd.BooleanDtype())

        # CIM columns with mixed values
        net.sgen["origin_id"] = pd.Series(["cim_1", pd.NA, pd.NA], dtype=pd.StringDtype())
        net.sgen["origin_class"] = pd.Series([pd.NA, pd.NA, "GeneratingUnit"], dtype=pd.StringDtype())

        validate_network(net)

    def test_cim_columns_all_na_valid(self):
        """Test: All CIM-related columns can be NA"""
        net = pandapowerNet(name="test_cim_columns_all_na_valid")
        b0 = create_bus(net, 0.4)

        create_sgen(net, bus=b0, p_mw=1.0, q_mvar=0.0, scaling=1.0, in_service=True)

        # CIM columns from schema metadata
        cim_string_columns = ["name", "origin_id", "origin_class", "terminal", "description", "type"]

        for col in cim_string_columns:
            net.sgen[col] = pd.Series([pd.NA], dtype=pd.StringDtype())

        validate_network(net)

    def test_opf_group_row_consistency_valid(self):
        """Test: OPF group - each row must have all values or all NaN"""
        net = pandapowerNet(name="test_opf_group_row_consistency_valid")
        b0 = create_bus(net, 0.4)

        # Row 1: all OPF values present
        create_sgen(net, bus=b0, p_mw=1.0, q_mvar=0.0, scaling=1.0, in_service=True)
        # Row 2: all OPF values NaN
        create_sgen(net, bus=b0, p_mw=2.0, q_mvar=0.1, scaling=1.0, in_service=True)

        net.sgen["max_p_mw"] = [2.0, float(np.nan)]
        net.sgen["min_p_mw"] = [-2.0, float(np.nan)]
        net.sgen["max_q_mvar"] = [1.0, float(np.nan)]
        net.sgen["min_q_mvar"] = [-1.0, float(np.nan)]

        validate_network(net)

    def test_opf_group_row_partial_invalid(self):
        """Test: OPF group - partial values in a row should fail"""
        net = pandapowerNet(name="test_opf_group_row_partial_invalid")
        b0 = create_bus(net, 0.4)

        create_sgen(net, bus=b0, p_mw=1.0, q_mvar=0.0, scaling=1.0, in_service=True)
        create_sgen(net, bus=b0, p_mw=2.0, q_mvar=0.1, scaling=1.0, in_service=True)

        # Row 1: all OPF values present
        # Row 2: partial OPF values
        net.sgen["max_p_mw"] = [2.0, 3.0]
        net.sgen["min_p_mw"] = [-2.0, float(np.nan)]
        net.sgen["max_q_mvar"] = [1.0, float(np.nan)]
        net.sgen["min_q_mvar"] = [-1.0, float(np.nan)]

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_qcc_group_row_consistency_valid(self):
        """Test: QCC group - each row must have all values or all NA"""
        net = pandapowerNet(name="test_qcc_group_row_consistency_valid")
        b0 = create_bus(net, 0.4)

        # Row 1: all QCC values present
        create_sgen(net, bus=b0, p_mw=1.0, q_mvar=0.0, scaling=1.0, in_service=True)
        # Row 2: all QCC values NA
        create_sgen(net, bus=b0, p_mw=2.0, q_mvar=0.1, scaling=1.0, in_service=True)

        net.sgen["id_q_capability_characteristic"] = pd.Series([0, pd.NA], dtype="Int64")
        net.sgen["curve_style"] = pd.Series(["straightLineYValues", pd.NA], dtype=pd.StringDtype())
        net.sgen["reactive_capability_curve"] = pd.Series([True, pd.NA], dtype=pd.BooleanDtype())

        validate_network(net)

    @pytest.mark.xfail  # TODO add back when reactive_capability_curve is removed
    def test_qcc_group_row_partial_invalid(self):
        """Test: QCC group - partial values in a row should fail"""
        net = pandapowerNet(name="test_qcc_group_row_partial_invalid")
        b0 = create_bus(net, 0.4)

        create_sgen(net, bus=b0, p_mw=1.0, q_mvar=0.0, scaling=1.0, in_service=True)
        create_sgen(net, bus=b0, p_mw=2.0, q_mvar=0.1, scaling=1.0, in_service=True)

        # Row 1: all QCC values present
        # Row 2: partial QCC values (only id set) - should fail
        net.sgen["id_q_capability_characteristic"] = pd.Series([0, 1], dtype="Int64")
        net.sgen["curve_style"] = pd.Series(["straightLineYValues", pd.NA], dtype=pd.StringDtype())
        net.sgen["reactive_capability_curve"] = pd.Series([True, pd.NA], dtype=pd.BooleanDtype())

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestSgenForeignKey:
    """Tests for foreign key constraints"""

    def test_invalid_bus_index(self):
        net = pandapowerNet(name="")
        b0 = create_bus(net, 0.4)
        create_sgen(net, bus=b0, p_mw=1.0, q_mvar=0.0, scaling=1.0, in_service=True)

        net.sgen["bus"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_valid_bus_index_non_sequential(self):
        """Test: bus FK works with non-sequential bus indices"""
        net = pandapowerNet(name="test_valid_bus_index_non_sequential")
        create_bus(net, 0.4, index=10)
        create_bus(net, 0.4, index=42)
        create_bus(net, 0.4, index=100)

        create_sgen(net, bus=10, p_mw=1.0, q_mvar=0.1, scaling=1.0, in_service=True)
        create_sgen(net, bus=42, p_mw=2.0, q_mvar=0.2, scaling=0.9, in_service=True)
        create_sgen(net, bus=100, p_mw=0.5, q_mvar=0.05, scaling=1.1, in_service=False)

        validate_network(net)

    def test_valid_bus_index_non_sequential(self):
        """Test: bus FK works with non-sequential bus indices"""
        net = create_empty_network()
        create_bus(net, 0.4, index=10)
        create_bus(net, 0.4, index=42)
        create_bus(net, 0.4, index=100)

        create_sgen(net, bus=10, p_mw=1.0, q_mvar=0.1, scaling=1.0, in_service=True)
        create_sgen(net, bus=42, p_mw=2.0, q_mvar=0.2, scaling=0.9, in_service=True)
        create_sgen(net, bus=100, p_mw=0.5, q_mvar=0.05, scaling=1.1, in_service=False)

        validate_network(net)


class TestSgenResults:
    """Tests for sgen results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_sgen_result_totals(self):
        """Aggregated p_mw / q_mvar results are consistent"""
        pass
