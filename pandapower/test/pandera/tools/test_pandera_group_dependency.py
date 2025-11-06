import pandas as pd
import pandera.pandas as pa
import pytest
import numpy as np

from pandapower.network_schema.tools.validation.group_dependency import (
    create_column_group_dependency_validation_func,
    create_column_dependency_checks_from_metadata,
)


class TestCreateColumnGroupDependencyValidationFunc:
    """Tests for create_column_group_dependency_validation_func function."""

    def test_all_columns_present_returns_true(self):
        """Test that validator returns True when all specified columns are present."""
        validator = create_column_group_dependency_validation_func(["lat", "lon", "altitude"])
        df = pd.DataFrame({"lat": [1], "lon": [2], "altitude": [3], "other": [4]})

        assert validator(df) is True

    def test_no_columns_present_returns_true(self):
        """Test that validator returns True when none of the specified columns are present."""
        validator = create_column_group_dependency_validation_func(["lat", "lon", "altitude"])
        df = pd.DataFrame({"name": ["John"], "age": [25]})

        assert validator(df) is True

    def test_partial_columns_present_returns_false(self):
        """Test that validator returns False when only some specified columns are present."""
        validator = create_column_group_dependency_validation_func(["lat", "lon", "altitude"])
        df = pd.DataFrame({"lat": [1], "lon": [2], "name": ["John"]})

        assert validator(df) is False

    def test_single_column_present_returns_false(self):
        """Test that validator returns False when only one of multiple specified columns is present."""
        validator = create_column_group_dependency_validation_func(["lat", "lon", "altitude"])
        df = pd.DataFrame({"lat": [1], "name": ["John"]})

        assert validator(df) is False

    def test_empty_column_list(self):
        """Test validator behavior with empty column list."""
        validator = create_column_group_dependency_validation_func([])
        df = pd.DataFrame({"name": ["John"], "age": [25]})

        assert validator(df) is True

    def test_single_column_dependency(self):
        """Test validator with single column dependency."""
        validator = create_column_group_dependency_validation_func(["lat"])

        # Column present
        df_present = pd.DataFrame({"lat": [1], "other": [2]})
        assert validator(df_present) is True

        # Column absent (this is ok since a single column dependency does not exist)
        df_absent = pd.DataFrame({"other": [2]})
        assert validator(df_absent) is True

    def test_duplicate_columns_in_input(self):
        """Test that duplicate column names in input are handled correctly."""
        validator = create_column_group_dependency_validation_func(["lat", "lon", "lat"])

        # All unique columns present
        df = pd.DataFrame({"lat": [1], "lon": [2], "other": [3]})
        assert validator(df) is True

        # Only one column present
        df_partial = pd.DataFrame({"lat": [1], "other": [3]})
        assert validator(df_partial) is False

    def test_empty_dataframe(self):
        """Test validator with empty DataFrame."""
        validator = create_column_group_dependency_validation_func(["lat", "lon"])
        df = pd.DataFrame()

        assert validator(df) is True

    def test_case_sensitive_column_names(self):
        """Test that column name matching is case sensitive."""
        validator = create_column_group_dependency_validation_func(["Lat", "Lon"])
        df = pd.DataFrame({"lat": [1], "lon": [2]})  # lowercase

        assert validator(df) is True  # None of the specified columns present

    def test_partial_column_existence(self):
        """Test that validation fails when only some columns from the group are present."""
        validator = create_column_group_dependency_validation_func(["Lat", "Lon"])

        # Only 'Lat' present, 'Lon' missing
        df_partial_1 = pd.DataFrame({"Lat": [1], "other_col": [2]})

        assert validator(df_partial_1) is False  # Partial presence should fail

        # Only 'Lon' present, 'Lat' missing
        df_partial_2 = pd.DataFrame({"Lon": [1], "other_col": [2]})

        assert validator(df_partial_2) is False  # Partial presence should fail


class TestCreateColumnDependencyChecksFromMetadata:
    """Tests for create_column_dependency_checks_from_metadata function."""

    def test_single_dependency_group(self):
        """Test creating checks for a single dependency group."""
        names = ["required_group"]
        schema_columns = {
            "col1": pa.Column(metadata={"required_group": True}),
            "col2": pa.Column(metadata={"required_group": True}),
            "col3": pa.Column(metadata={"other_group": True}),
        }

        checks = create_column_dependency_checks_from_metadata(names, schema_columns)

        assert len(checks) == 1  # Only col1 and col2 have required_group
        assert all(isinstance(check, pa.Check) for check in checks)

    def test_multiple_dependency_groups(self):
        """Test creating checks for multiple dependency groups."""
        names = ["required_group", "optional_group"]
        schema_columns = {
            "col1": pa.Column(metadata={"required_group": True}),
            "col2": pa.Column(metadata={"required_group": True}),
            "col3": pa.Column(metadata={"optional_group": True}),
            "col4": pa.Column(metadata={"other_group": True}),
        }

        checks = create_column_dependency_checks_from_metadata(names, schema_columns)

        assert len(checks) == 2

    def test_no_matching_dependencies(self):
        """Test when no columns have the specified dependencies."""
        names = ["nonexistent_group"]
        schema_columns = {
            "col1": pa.Column(metadata={"other_group": True}),
            "col2": pa.Column(metadata={"another_group": True}),
        }

        checks = create_column_dependency_checks_from_metadata(names, schema_columns)

        assert len(checks) == 0

    def test_empty_names_list(self):
        """Test with empty dependency names list."""
        names = []
        schema_columns = {
            "col1": pa.Column(metadata={"required_group": True}),
            "col2": pa.Column(metadata={"optional_group": True}),
        }

        checks = create_column_dependency_checks_from_metadata(names, schema_columns)

        assert len(checks) == 0

    def test_empty_schema_columns(self):
        """Test with empty schema columns dictionary."""
        names = ["required_group"]
        schema_columns = {}

        checks = create_column_dependency_checks_from_metadata(names, schema_columns)

        assert len(checks) == 0

    def test_columns_without_metadata(self):
        """Test columns that have no metadata attribute."""
        names = ["required_group"]
        schema_columns = {
            "col1": pa.Column(metadata=None),
            "col2": pa.Column(metadata={"required_group": True}),
            "col3": pa.Column(),  # pa.Column without metadata attribute
        }

        checks = create_column_dependency_checks_from_metadata(names, schema_columns)

        assert len(checks) == 1  # Only col2 has valid metadata

    def test_dependency_with_false_value(self):
        """Test that dependencies with False values are not included."""
        names = ["required_group"]
        schema_columns = {
            "col1": pa.Column(metadata={"required_group": True}),
            "col2": pa.Column(metadata={"required_group_false": False}),
            "col3": pa.Column(metadata={"required_group": True}),
        }

        checks = create_column_dependency_checks_from_metadata(names, schema_columns)

        assert len(checks) == 1  # Only col1 and col3

    def test_check_error_messages(self):
        """Test that generated checks have correct error messages."""
        names = ["test_group"]
        schema_columns = {"col1": pa.Column(metadata={"test_group": True})}

        checks = create_column_dependency_checks_from_metadata(names, schema_columns)

        assert len(checks) == 1
        expected_error = "test_group columns have dependency violations. Please ensure columns ['col1'] are present in the dataframe."
        assert checks[0].error == expected_error

    def test_mixed_metadata_types(self):
        """Test handling of different metadata value types."""
        names = ["test_group"]
        schema_columns = {
            "col1": pa.Column(metadata={"test_group": True}),
            "col2": pa.Column(metadata={"test_group": 1}),  # Truthy value
            "col3": pa.Column(metadata={"test_group_false": 0}),  # Falsy value
            "col4": pa.Column(metadata={"test_group": "yes"}),  # Truthy string
            "col5": pa.Column(metadata={"test_group_false": ""}),  # Falsy string
        }

        checks = create_column_dependency_checks_from_metadata(names, schema_columns)

        assert len(checks) == 1  # col1, col2, col4 have truthy values


class TestIntegration:
    """Integration tests combining both functions."""

    def test_full_workflow_valid_dependencies(self):
        """Test complete workflow with valid column dependencies."""
        # Setup schema with dependencies
        names = ["location_group"]
        schema_columns = {
            "lat": pa.Column(nullable=True, metadata={"location_group": True}),
            "lon": pa.Column(nullable=True, metadata={"location_group": True}),
            "name": pa.Column(nullable=True, metadata={"other_group": True}),
        }

        # Create checks
        checks = create_column_dependency_checks_from_metadata(names, schema_columns)

        # Create schema
        full_workflow_valid_schema = pa.DataFrameSchema(schema_columns, strict=False, checks=checks)

        # Test with valid DataFrame (all location columns present)
        df_valid = pd.DataFrame({"lat": [1.0], "lon": [2.0], "name": [pd.NA], "other": ["value"]})

        # All checks should pass
        full_workflow_valid_schema.validate(df_valid)

    def test_full_workflow_invalid_dependencies(self):
        """Test complete workflow with invalid column dependencies."""
        # Setup schema with dependencies
        names = ["location_group"]
        schema_columns = {
            "lat": pa.Column(metadata={"location_group": True}),
            "lon": pa.Column(metadata={"location_group": True}),
        }

        # Create checks
        checks = create_column_dependency_checks_from_metadata(names, schema_columns)

        # Create schema
        full_workflow_invalid_schema = pa.DataFrameSchema(schema_columns, strict=False, checks=checks)

        # Test with invalid DataFrame (only one location column present)
        df_invalid = pd.DataFrame({"lat": [1.0], "name": ["Test"]})

        with pytest.raises(pa.errors.SchemaError):
            full_workflow_invalid_schema.validate(df_invalid)

    def test_full_workflow_invalid_entry_dependencies(self):
        """Test complete workflow with invalid column dependencies, because of nullable entry."""
        # Setup schema with dependencies
        names = ["location_group"]
        schema_columns = {
            "lat": pa.Column(metadata={"location_group": True}),
            "lon": pa.Column(metadata={"location_group": True}),
        }

        # Create checks
        checks = create_column_dependency_checks_from_metadata(names, schema_columns)

        # Create schema
        full_workflow_invalid_schema = pa.DataFrameSchema(schema_columns, strict=False, checks=checks)

        # Test with invalid DataFrame (not assigned entry)
        df_invalid_na = pd.DataFrame({"lat": [pd.NA], "name": ["Test"]})

        with pytest.raises(pa.errors.SchemaError):
            full_workflow_invalid_schema.validate(df_invalid_na)

        # Test with invalid DataFrame (None entry)
        df_invalid_none = pd.DataFrame({"lat": None, "name": ["Test"]})

        with pytest.raises(pa.errors.SchemaError):
            full_workflow_invalid_schema.validate(df_invalid_none)

        # Test with invalid DataFrame (nan entry)
        df_invalid_nan = pd.DataFrame({"lat": float(np.nan), "name": ["Test"]})

        with pytest.raises(pa.errors.SchemaError):
            full_workflow_invalid_schema.validate(df_invalid_nan)

        # Test with invalid DataFrame (multi row null entry)
        df_invalid_row = pd.DataFrame(
            {"lat": [1.0, np.nan, 1.0, 1.0, 1.0], "name": ["Test", "Test", pd.NA, "Test", "Test"]}
        )

        with pytest.raises(pa.errors.SchemaError):
            full_workflow_invalid_schema.validate(df_invalid_row)
