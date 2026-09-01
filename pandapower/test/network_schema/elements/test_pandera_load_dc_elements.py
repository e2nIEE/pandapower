# test_pandera_load_dc_elements.py

import itertools
import numpy as np
import pandas as pd
import pandera as pa
import pytest

from pandapower.create import create_bus_dc, create_load_dc
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
    all_allowed_floats,
    negativ_floats,
    positiv_floats_plus_zero,
)


class TestLoadDcRequiredFields:
    """Tests for required load_dc fields"""

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["bus_dc"], positiv_ints_plus_zero),
                itertools.product(["p_dc_mw"], all_allowed_floats),
                itertools.product(["scaling"], positiv_floats_plus_zero),
                itertools.product(["in_service"], bools),
            )
        ),
    )
    def test_valid_required_values(self, parameter, valid_value):
        """Test: valid required values are accepted"""
        net = pandapowerNet(name="test_valid_required_values")
        create_bus_dc(net, 0.4)  # index 0
        create_bus_dc(net, 0.4)  # index 1
        create_bus_dc(net, 0.4, index=42)

        create_load_dc(net, bus_dc=0, p_dc_mw=1.0, scaling=1.0, in_service=True)
        net.load_dc[parameter] = valid_value
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter, invalid_value",
        list(
            itertools.chain(
                itertools.product(["bus_dc"], [float(np.nan), pd.NA, *negativ_ints, *not_ints_list]),
                itertools.product(["p_dc_mw"], [float(np.nan), pd.NA, *not_floats_list]),
                itertools.product(["scaling"], [float(np.nan), pd.NA, *negativ_floats, *not_floats_list]),
                itertools.product(["in_service"], [float(np.nan), pd.NA, *not_boolean_list]),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Test: invalid required values are rejected"""
        net = pandapowerNet(name="test_invalid_required_values")
        create_bus_dc(net, 0.4)  # index 0
        create_bus_dc(net, 0.4)  # index 1

        create_load_dc(net, bus_dc=0, p_dc_mw=1.0, scaling=1.0, in_service=True)
        net.load_dc[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestLoadDcOptionalFields:
    """Tests for optional load_dc fields"""

    def test_all_optional_fields_valid(self):
        """Test: load_dc with all optional fields is valid"""
        net = pandapowerNet(name="test_all_optional_fields_valid")
        b0 = create_bus_dc(net, 0.4)

        create_load_dc(
            net,
            bus_dc=b0,
            p_dc_mw=1.2,
            scaling=1.0,
            in_service=True,
            name="LoadDC A",
            type="consumer",
            zone="area-1",
            controllable=True,
        )

        validate_network(net)

    def test_optional_fields_with_nulls(self):
        """Test: optional fields including nulls are valid"""
        net = pandapowerNet(name="test_optional_fields_with_nulls")
        b0 = create_bus_dc(net, 0.4)

        # Row 1: strings None, controllable NA
        create_load_dc(net, bus_dc=b0, p_dc_mw=0.5, scaling=1.0, in_service=True, name=None)
        # Row 2: set strings and controllable True
        create_load_dc(net, bus_dc=b0, p_dc_mw=1.0, scaling=0.8, in_service=True, type="prosumer", zone="Z2")
        # Row 3: controllable False
        create_load_dc(net, bus_dc=b0, p_dc_mw=2.0, scaling=1.1, in_service=False)

        # Cast to required extension dtypes and set nulls
        net.load_dc["name"] = pd.Series([pd.NA, "L2", "L3"], dtype="string")
        net.load_dc["type"] = pd.Series([pd.NA, "prosumer", pd.NA], dtype="string")
        net.load_dc["zone"] = pd.Series(["Z1", "Z2", pd.NA], dtype="string")

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                # Nullable string columns - include pd.NA directly in chain
                itertools.product(["name"], [pd.NA, *strings]),
                itertools.product(["type"], [pd.NA, *strings]),
                itertools.product(["zone"], [pd.NA, *strings]),
                # Boolean column (required=False but not nullable)
                itertools.product(["controllable"], bools),
            )
        ),
    )
    def test_valid_optional_values(self, parameter, valid_value):
        """Test: valid optional values are accepted"""
        net = pandapowerNet(name="test_valid_optional_values")
        b0 = create_bus_dc(net, 0.4)
        create_load_dc(net, bus_dc=b0, p_dc_mw=1.0, scaling=1.0, in_service=True)

        if parameter in {"name", "type", "zone"}:
            net.load_dc[parameter] = pd.Series([valid_value], dtype="string")
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter, invalid_value",
        list(
            itertools.chain(
                itertools.product(["name"], not_strings_list),
                itertools.product(["type"], not_strings_list),
                itertools.product(["zone"], not_strings_list),
                itertools.product(["controllable"], not_boolean_list),
            )
        ),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        """Test: invalid optional values are rejected"""
        net = pandapowerNet(name="test_invalid_optional_values")
        b0 = create_bus_dc(net, 0.4)
        create_load_dc(
            net,
            bus_dc=b0,
            p_dc_mw=1.0,
            scaling=1.0,
            in_service=True,
            # set valid defaults so only target param fails
            name="ok",
            type="ok",
            zone="ok",
            controllable=True,
        )

        net.load_dc[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestLoadDcNullableColumns:
    """Tests for nullable columns - testing NA/NaN acceptance"""

    def test_all_nullable_string_columns_na_valid(self):
        """Test: All nullable string columns can be NA"""
        net = pandapowerNet(name="test_all_nullable_string_columns_na_valid")
        b0 = create_bus_dc(net, 0.4)

        create_load_dc(net, bus_dc=b0, p_dc_mw=1.0, scaling=1.0, in_service=True)

        # Set all nullable string columns to NA
        net.load_dc["name"] = pd.Series([pd.NA], dtype=pd.StringDtype())
        net.load_dc["type"] = pd.Series([pd.NA], dtype=pd.StringDtype())
        net.load_dc["zone"] = pd.Series([pd.NA], dtype=pd.StringDtype())

        validate_network(net)

    @pytest.mark.parametrize(
        "column_name",
        ["name", "type", "zone"],
    )
    def test_individual_nullable_string_column_na_valid(self, column_name):
        """Test: Each nullable string column accepts NA individually"""
        net = pandapowerNet(name="test_individual_nullable_string_column_na_valid")
        b0 = create_bus_dc(net, 0.4)

        create_load_dc(net, bus_dc=b0, p_dc_mw=1.0, scaling=1.0, in_service=True)

        net.load_dc[column_name] = pd.Series([pd.NA], dtype=pd.StringDtype())
        validate_network(net)

    def test_mixed_null_and_valid_values_in_rows(self):
        """Test: Multiple rows with mixed NA and valid values"""
        net = pandapowerNet(name="test_mixed_null_and_valid_values_in_rows")
        b0 = create_bus_dc(net, 0.4)
        b1 = create_bus_dc(net, 0.4)
        b2 = create_bus_dc(net, 0.4)

        # Row 1: all optional string fields filled
        create_load_dc(
            net,
            bus_dc=b0,
            p_dc_mw=1.0,
            scaling=1.0,
            in_service=True,
            name="Load A",
            type="consumer",
            zone="zone-1",
            controllable=True,
        )
        # Row 2: only some fields filled
        create_load_dc(
            net,
            bus_dc=b1,
            p_dc_mw=2.0,
            scaling=0.8,
            in_service=False,
            name="Load B",
            controllable=False,
        )
        # Row 3: minimal - only required fields
        create_load_dc(
            net,
            bus_dc=b2,
            p_dc_mw=0.5,
            scaling=1.2,
            in_service=True,
        )

        # Set nullable columns with mixed values
        net.load_dc["name"] = pd.Series(["Load A", "Load B", pd.NA], dtype=pd.StringDtype())
        net.load_dc["type"] = pd.Series(["consumer", pd.NA, pd.NA], dtype=pd.StringDtype())
        net.load_dc["zone"] = pd.Series(["zone-1", pd.NA, pd.NA], dtype=pd.StringDtype())

        validate_network(net)


class TestLoadDcForeignKey:
    """Tests for foreign key constraints"""

    def test_invalid_bus_index(self):
        """Test: bus_dc must reference an existing bus_dc index"""
        net = pandapowerNet(name="test_invalid_bus_index")
        b0 = create_bus_dc(net, 0.4)
        create_load_dc(net, bus_dc=b0, p_dc_mw=1.0, scaling=1.0, in_service=True)

        net.load_dc["bus_dc"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_valid_bus_dc_index_non_sequential(self):
        """Test: bus_dc FK works with non-sequential bus_dc indices"""
        net = pandapowerNet(name="test_valid_bus_dc_index_non_sequential")
        create_bus_dc(net, 0.4, index=10)
        create_bus_dc(net, 0.4, index=42)
        create_bus_dc(net, 0.4, index=100)

        create_load_dc(net, bus_dc=10, p_dc_mw=1.0, scaling=1.0, in_service=True)
        create_load_dc(net, bus_dc=42, p_dc_mw=2.0, scaling=0.9, in_service=True)
        create_load_dc(net, bus_dc=100, p_dc_mw=0.5, scaling=1.1, in_service=False)

        validate_network(net)


class TestLoadDcResults:
    """Tests for load_dc results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_load_dc_result_totals(self):
        """Test: aggregated p_dc_mw results are consistent"""
        pass
