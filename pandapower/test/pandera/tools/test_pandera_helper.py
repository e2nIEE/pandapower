import pytest
import pandera as pa
import pandas as pd
import numpy as np
from pandapower.network_schema.tools.helper import get_dtypes


class TestGetDtypes:
    """Test suite for the get_dtypes function."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Schema with mixed required/optional columns
        self.mixed_schema = pa.DataFrameSchema(
            {
                "name": pa.Column(str, required=True),
                "age": pa.Column(int, required=True),
                "score": pa.Column(float, required=False),
                "active": pa.Column(bool, required=False),
                "category": pa.Column(str, required=True),
            }
        )

        # Schema with pandas dtypes
        self.pandas_dtypes_schema = pa.DataFrameSchema(
            {
                "string_col": pa.Column(pd.StringDtype(), required=True),
                "int_col": pa.Column(pd.Int64Dtype(), required=True),
                "float_col": pa.Column(pd.Float64Dtype(), required=False),
                "bool_col": pa.Column(pd.BooleanDtype(), required=False),
                "categorical_col": pa.Column(pd.CategoricalDtype(["A", "B", "C"]), required=True),
            }
        )

        # Schema with pandas datetime dtypes
        self.datetime_schema = pa.DataFrameSchema(
            {
                "datetime_col": pa.Column(pd.DatetimeTZDtype(tz="UTC"), required=True),
                "date_col": pa.Column("datetime64[ns]", required=False),
                "period_col": pa.Column(pd.PeriodDtype(freq="D"), required=True),
                "timedelta_col": pa.Column("timedelta64[ns]", required=False),
            }
        )

        # Schema with numpy dtypes
        self.numpy_dtypes_schema = pa.DataFrameSchema(
            {
                "int8_col": pa.Column(np.int8, required=True),
                "int16_col": pa.Column(np.int16, required=False),
                "int32_col": pa.Column(np.int32, required=True),
                "int64_col": pa.Column(np.int64, required=False),
                "float32_col": pa.Column(np.float32, required=True),
                "float64_col": pa.Column(np.float64, required=False),
                "complex128_col": pa.Column(np.complex128, required=True),
            }
        )

        # Schema with mixed pandas and numpy dtypes
        self.mixed_dtypes_schema = pa.DataFrameSchema(
            {
                "pandas_string": pa.Column(pd.StringDtype(), required=True),
                "numpy_int": pa.Column(np.int64, required=True),
                "pandas_float": pa.Column(pd.Float32Dtype(), required=False),
                "python_bool": pa.Column(bool, required=False),
                "pandas_categorical": pa.Column(pd.CategoricalDtype(), required=True),
            }
        )

        # Empty schema
        self.empty_schema = pa.DataFrameSchema({})

    def test_get_dtypes_required_only_default(self):
        """Test get_dtypes with default required_only=True."""
        result = get_dtypes(self.mixed_schema)
        expected = {"name": str, "age": int, "category": str}
        assert result == expected

    def test_get_dtypes_required_only_true(self):
        """Test get_dtypes with explicit required_only=True."""
        result = get_dtypes(self.mixed_schema, required_only=True)
        expected = {"name": str, "age": int, "category": str}
        assert result == expected

    def test_get_dtypes_required_only_false(self):
        """Test get_dtypes with required_only=False."""
        result = get_dtypes(self.mixed_schema, required_only=False)
        expected = {"name": str, "age": int, "score": float, "active": bool, "category": str}
        assert result == expected

    def test_get_dtypes_pandas_dtypes_required_only(self):
        """Test get_dtypes with pandas dtypes and required_only=True."""
        result = get_dtypes(self.pandas_dtypes_schema, required_only=True)

        assert len(result) == 3
        assert "string_col" in result
        assert "int_col" in result
        assert "categorical_col" in result

        assert isinstance(result["string_col"], pd.StringDtype)
        assert isinstance(result["int_col"], pd.Int64Dtype)
        assert isinstance(result["categorical_col"], pd.CategoricalDtype)

    def test_get_dtypes_pandas_dtypes_all_columns(self):
        """Test get_dtypes with pandas dtypes and required_only=False."""
        result = get_dtypes(self.pandas_dtypes_schema, required_only=False)

        assert len(result) == 5
        assert isinstance(result["string_col"], pd.StringDtype)
        assert isinstance(result["int_col"], pd.Int64Dtype)
        assert isinstance(result["float_col"], pd.Float64Dtype)
        assert isinstance(result["bool_col"], pd.BooleanDtype)
        assert isinstance(result["categorical_col"], pd.CategoricalDtype)

    def test_get_dtypes_datetime_dtypes_required_only(self):
        """Test get_dtypes with datetime-related pandas dtypes."""
        result = get_dtypes(self.datetime_schema, required_only=True)

        assert len(result) == 2
        assert "datetime_col" in result
        assert "period_col" in result

        assert isinstance(result["datetime_col"], pd.DatetimeTZDtype)
        assert isinstance(result["period_col"], pd.PeriodDtype)

    def test_get_dtypes_datetime_dtypes_all_columns(self):
        """Test get_dtypes with datetime dtypes and required_only=False."""
        result = get_dtypes(self.datetime_schema, required_only=False)

        # Check that all columns are present
        assert "datetime_col" in result
        assert "date_col" in result
        assert "period_col" in result
        assert "timedelta_col" in result

        assert isinstance(result["datetime_col"], pd.DatetimeTZDtype)
        assert isinstance(result["period_col"], pd.PeriodDtype)

    def test_get_dtypes_numpy_dtypes_required_only(self):
        """Test get_dtypes with numpy dtypes and required_only=True."""
        result = get_dtypes(self.numpy_dtypes_schema, required_only=True)
        expected = {
            "int8_col": np.int8,
            "int32_col": np.int32,
            "float32_col": np.float32,
            "complex128_col": np.complex128,
        }
        assert result == expected

    def test_get_dtypes_numpy_dtypes_all_columns(self):
        """Test get_dtypes with numpy dtypes and required_only=False."""
        result = get_dtypes(self.numpy_dtypes_schema, required_only=False)
        expected = {
            "int8_col": np.int8,
            "int16_col": np.int16,
            "int32_col": np.int32,
            "int64_col": np.int64,
            "float32_col": np.float32,
            "float64_col": np.float64,
            "complex128_col": np.complex128,
        }
        assert result == expected

    def test_get_dtypes_mixed_dtypes_required_only(self):
        """Test get_dtypes with mixed pandas and numpy dtypes."""
        result = get_dtypes(self.mixed_dtypes_schema, required_only=True)

        assert len(result) == 3
        assert "pandas_string" in result
        assert "numpy_int" in result
        assert "pandas_categorical" in result

        assert isinstance(result["pandas_string"], pd.StringDtype)
        assert result["numpy_int"] == np.int64
        assert isinstance(result["pandas_categorical"], pd.CategoricalDtype)

    def test_get_dtypes_mixed_dtypes_all_columns(self):
        """Test get_dtypes with mixed dtypes and required_only=False."""
        result = get_dtypes(self.mixed_dtypes_schema, required_only=False)

        assert len(result) == 5
        assert isinstance(result["pandas_string"], pd.StringDtype)
        assert result["numpy_int"] == np.int64
        assert isinstance(result["pandas_float"], pd.Float32Dtype)
        assert result["python_bool"] == bool
        assert isinstance(result["pandas_categorical"], pd.CategoricalDtype)

    def test_get_dtypes_all_required_schema(self):
        """Test get_dtypes with schema containing only required columns."""
        all_required_schema = pa.DataFrameSchema(
            {"id": pa.Column(int, required=True), "title": pa.Column(str, required=True)}
        )

        result_required_only = get_dtypes(all_required_schema, required_only=True)
        result_all = get_dtypes(all_required_schema, required_only=False)

        expected = {"id": int, "title": str}

        assert result_required_only == expected
        assert result_all == expected

    def test_get_dtypes_all_optional_schema_required_only(self):
        """Test get_dtypes with schema containing only optional columns and required_only=True."""
        all_optional_schema = pa.DataFrameSchema(
            {"notes": pa.Column(str, required=False), "rating": pa.Column(float, required=False)}
        )

        result = get_dtypes(all_optional_schema, required_only=True)
        assert result == {}

    def test_get_dtypes_all_optional_schema_all_columns(self):
        """Test get_dtypes with schema containing only optional columns and required_only=False."""
        all_optional_schema = pa.DataFrameSchema(
            {"notes": pa.Column(str, required=False), "rating": pa.Column(float, required=False)}
        )

        result = get_dtypes(all_optional_schema, required_only=False)
        expected = {"notes": str, "rating": float}
        assert result == expected

    def test_get_dtypes_empty_schema(self):
        """Test get_dtypes with empty schema."""
        result_required_only = get_dtypes(self.empty_schema, required_only=True)
        result_all = get_dtypes(self.empty_schema, required_only=False)

        assert result_required_only == {}
        assert result_all == {}

    def test_get_dtypes_categorical_with_categories(self):
        """Test get_dtypes with categorical dtype that has specific categories."""
        schema = pa.DataFrameSchema(
            {
                "status": pa.Column(pd.CategoricalDtype(["active", "inactive", "pending"]), required=True),
                "priority": pa.Column(pd.CategoricalDtype(["low", "medium", "high"], ordered=True), required=False),
            }
        )

        result_required = get_dtypes(schema, required_only=True)
        result_all = get_dtypes(schema, required_only=False)

        assert len(result_required) == 1
        assert "status" in result_required
        assert isinstance(result_required["status"], pd.CategoricalDtype)

        assert len(result_all) == 2
        assert isinstance(result_all["status"], pd.CategoricalDtype)
        assert isinstance(result_all["priority"], pd.CategoricalDtype)

    def test_get_dtypes_interval_dtype(self):
        """Test get_dtypes with pandas IntervalDtype."""
        schema = pa.DataFrameSchema(
            {
                "intervals": pa.Column(pd.IntervalDtype(subtype="int64"), required=True),
                "float_intervals": pa.Column(pd.IntervalDtype(subtype="float64"), required=False),
            }
        )

        result_required = get_dtypes(schema, required_only=True)
        result_all = get_dtypes(schema, required_only=False)

        assert len(result_required) == 1
        assert isinstance(result_required["intervals"], pd.IntervalDtype)

        assert len(result_all) == 2
        assert isinstance(result_all["intervals"], pd.IntervalDtype)
        assert isinstance(result_all["float_intervals"], pd.IntervalDtype)

    def test_get_dtypes_sparse_dtype(self):
        """Test get_dtypes with pandas SparseDtype."""
        schema = pa.DataFrameSchema(
            {
                "sparse_int": pa.Column(pd.SparseDtype(dtype=np.int64, fill_value=0), required=True),
                "sparse_float": pa.Column(pd.SparseDtype(dtype=np.float64), required=False),
            }
        )

        result_required = get_dtypes(schema, required_only=True)
        result_all = get_dtypes(schema, required_only=False)

        assert len(result_required) == 1
        assert isinstance(result_required["sparse_int"], pd.SparseDtype)

        assert len(result_all) == 2
        assert isinstance(result_all["sparse_int"], pd.SparseDtype)
        assert isinstance(result_all["sparse_float"], pd.SparseDtype)

    def test_get_dtypes_return_type(self):
        """Test that get_dtypes returns a dictionary."""
        result = get_dtypes(self.mixed_schema)
        assert isinstance(result, dict)

    def test_get_dtypes_keys_are_strings(self):
        """Test that all keys in the returned dictionary are strings."""
        result = get_dtypes(self.mixed_schema, required_only=False)
        for key in result.keys():
            assert isinstance(key, str)

    def test_get_dtypes_comprehensive_mixed_schema(self):
        """Comprehensive test with a schema containing various dtype combinations."""
        comprehensive_schema = pa.DataFrameSchema(
            {
                # Python built-ins
                "python_str": pa.Column(str, required=True),
                "python_int": pa.Column(int, required=False),
                "python_float": pa.Column(float, required=True),
                "python_bool": pa.Column(bool, required=False),
                # Pandas extension dtypes
                "pandas_string": pa.Column(pd.StringDtype(), required=True),
                "pandas_int": pa.Column(pd.Int32Dtype(), required=False),
                "pandas_float": pa.Column(pd.Float32Dtype(), required=True),
                "pandas_bool": pa.Column(pd.BooleanDtype(), required=False),
                # Numpy dtypes
                "numpy_int": pa.Column(np.int64, required=True),
                "numpy_float": pa.Column(np.float32, required=False),
                # Specialized pandas dtypes
                "categorical": pa.Column(pd.CategoricalDtype(["A", "B"]), required=True),
                "datetime_tz": pa.Column(pd.DatetimeTZDtype(tz="UTC"), required=False),
            }
        )

        result_required = get_dtypes(comprehensive_schema, required_only=True)
        result_all = get_dtypes(comprehensive_schema, required_only=False)

        # Test required columns
        assert len(result_required) == 6
        assert result_required["python_str"] == str
        assert result_required["python_float"] == float
        assert isinstance(result_required["pandas_string"], pd.StringDtype)
        assert isinstance(result_required["pandas_float"], pd.Float32Dtype)
        assert result_required["numpy_int"] == np.int64
        assert isinstance(result_required["categorical"], pd.CategoricalDtype)

        # Test all columns
        assert len(result_all) == 12
        assert result_all["python_str"] == str
        assert result_all["python_int"] == int
        assert result_all["python_float"] == float
        assert result_all["python_bool"] == bool
        assert isinstance(result_all["pandas_string"], pd.StringDtype)
        assert isinstance(result_all["pandas_int"], pd.Int32Dtype)
        assert isinstance(result_all["pandas_float"], pd.Float32Dtype)
        assert isinstance(result_all["pandas_bool"], pd.BooleanDtype)
        assert result_all["numpy_int"] == np.int64
        assert result_all["numpy_float"] == np.float32
        assert isinstance(result_all["categorical"], pd.CategoricalDtype)
        assert isinstance(result_all["datetime_tz"], pd.DatetimeTZDtype)


# Parametrized tests with different pandas dtypes


@pytest.mark.parametrize(
    "dtype_instance,expected_type_class",
    [
        (pd.StringDtype(), pd.StringDtype),
        (pd.Int64Dtype(), pd.Int64Dtype),
        (pd.Float64Dtype(), pd.Float64Dtype),
        (pd.BooleanDtype(), pd.BooleanDtype),
        (pd.CategoricalDtype(), pd.CategoricalDtype),
        (pd.DatetimeTZDtype(tz="UTC"), pd.DatetimeTZDtype),
        (pd.PeriodDtype(freq="D"), pd.PeriodDtype),
        (pd.IntervalDtype(), pd.IntervalDtype),
        (pd.SparseDtype(dtype=np.int64), pd.SparseDtype),
    ],
)
def test_get_dtypes_parametrized_pandas_dtypes(dtype_instance, expected_type_class):
    """Parametrized tests for various pandas dtypes."""
    schema = pa.DataFrameSchema({"test_col": pa.Column(dtype_instance, required=True)})

    result = get_dtypes(schema, required_only=True)
    assert len(result) == 1
    assert "test_col" in result
    assert isinstance(result["test_col"], expected_type_class)


@pytest.mark.parametrize(
    "numpy_dtype",
    [
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
        np.bool_,
    ],
)
def test_get_dtypes_parametrized_numpy_dtypes(numpy_dtype):
    """Parametrized tests for various numpy dtypes."""
    schema = pa.DataFrameSchema({"test_col": pa.Column(numpy_dtype, required=True)})

    result = get_dtypes(schema, required_only=True)
    assert result == {"test_col": numpy_dtype}


@pytest.mark.parametrize(
    "required_only,expected_length",
    [
        (True, 3),  # only required columns
        (False, 5),  # all columns
    ],
)
def test_get_dtypes_parametrized_mixed_schema(required_only, expected_length):
    """Parametrized tests for get_dtypes with mixed required/optional columns."""
    schema = pa.DataFrameSchema(
        {
            "name": pa.Column(str, required=True),
            "age": pa.Column(int, required=True),
            "score": pa.Column(float, required=False),
            "active": pa.Column(bool, required=False),
            "category": pa.Column(str, required=True),
        }
    )

    result = get_dtypes(schema, required_only=required_only)
    assert len(result) == expected_length


def test_get_dtypes_invalid_input():
    """Test get_dtypes with invalid input."""
    with pytest.raises(AttributeError):
        get_dtypes(None)

    with pytest.raises(AttributeError):
        get_dtypes("not_a_schema")


def test_get_dtypes_edge_cases():
    """Test edge cases and boundary conditions."""
    # Schema with object dtype
    object_schema = pa.DataFrameSchema({"object_col": pa.Column(object, required=True)})

    result = get_dtypes(object_schema)
    assert result == {"object_col": object}

    # Schema with mixed Python and pandas nullable dtypes
    mixed_nullable_schema = pa.DataFrameSchema(
        {
            "regular_int": pa.Column(int, required=True),
            "nullable_int": pa.Column(pd.Int64Dtype(), required=True),
            "regular_float": pa.Column(float, required=False),
            "nullable_float": pa.Column(pd.Float64Dtype(), required=False),
        }
    )

    result = get_dtypes(mixed_nullable_schema, required_only=False)

    assert len(result) == 4
    assert result["regular_int"] == int
    assert isinstance(result["nullable_int"], pd.Int64Dtype)
    assert result["regular_float"] == float
    assert isinstance(result["nullable_float"], pd.Float64Dtype)
