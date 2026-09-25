"""
Functions for creating and validating bus index cross-references.
"""
import pandas as pd
import pandera.pandas as pa
from copy import deepcopy
from pandapower import pandapowerNet


def _create_index_validation_check(reference_df: pd.DataFrame, column_name: str, element_name: str) -> pa.Check:
    """
    Creates a pandera check function that validates if values exist in the index of a reference dataframe.

    This function generates a validation check that ensures all values in a pandas Series
    are present in the index of the provided reference dataframe. It's commonly used
    to validate foreign key relationships, such as ensuring bus IDs in a transformer
    table exist in the bus table.

    Parameters:
        reference_df:
            The reference dataframe whose index will be used for validation.
        column_name: The name of the column being validated. Used for generating specific error messages and determining
            the reference type (e.g., 'dc' columns reference 'bus_dc' table).
        element_name: The name of the element/table being validated. Used in error messages for better debugging.

    Returns:
        A pandera Check object that can be used in schema validation. The check will raise a ValueError if any values
        are not found in the reference index.

    Raises:
        ValueError: Raised by the returned check function when validation fails, including details about failing values
            and their indices.

    Examples:
        >>> bus_df = pd.DataFrame(index=[1, 2, 3])
        >>> check = _create_index_validation_check(bus_df, 'hv_bus', 'trafo')
        >>> # This check can now be used to validate that trafo hv_bus values exist in bus_df
    """
    reference_index = set(reference_df.index) | {pd.NA}

    return pa.Check(
        lambda s: s.isin(reference_index),
        name=f"foreign key for {element_name}.{column_name}",
        determined_by_unique=True,
    )

def _create_multi_column_reference_schema(
    reference_df: pd.DataFrame,
    columns_to_validate: list[str],
    element_name: str,
) -> pa.DataFrameSchema:
    """
    Creates a pandera schema that validates multiple columns against the index of a reference dataframe.

    This function generates a comprehensive validation schema for dataframes where multiple
    columns need to reference the same reference table. Each specified column will be
    validated to ensure all its values exist in the reference dataframe's index.

    Parameters:
        reference_df: The reference dataframe whose index serves as the valid value set
            for all columns being validated.
        columns_to_validate: List of column names that should be validated against the reference
            dataframe's index. All columns will be treated as int64 dtype.
        element_name: Name of the element/table being validated. Used for schema naming and error message generation.

    Returns:
        A pandera DataFrameSchema object with validation rules for all specified columns. The schema is non-strict,
        allowing additional columns not specified in columns_to_validate.

    Notes:
        - All validated columns are configured as int64 dtype and non-nullable
        - The schema uses strict=False, allowing additional columns beyond those specified

        - Each column gets its own index validation check via :func:`_create_index_validation_check`

    Example:
        >>> bus_df = pd.DataFrame(index=[1, 2, 3])
        >>> schema = _create_multi_column_reference_schema(
        ...     bus_df, ['from_bus', 'to_bus'], 'line'
        ... )
        >>> # This schema can validate that both from_bus and to_bus columns
        >>> # contain only values that exist in bus_df.index
    """
    schema_columns = {}
    for col_name in columns_to_validate:
        schema_columns[col_name] = pa.Column(
            dtype="int64",
            checks=[_create_index_validation_check(reference_df, col_name, element_name)],
            nullable=False,
        )

    return pa.DataFrameSchema(columns=schema_columns, name=element_name, strict=False)


def build_foreign_key_index_checks(schema:pa.DataFrameSchema, net:pandapowerNet) -> None:
    """
        Creates a deepcopy of schema with foreign key validation checks added based on metadata.

        Parses the 'foreign_key' metadata from columns (format: "table")
        and adds index validation checks against the corresponding network element.

        Parameters:
            schema: The DataFrameSchema to analyze for foreign key columns
            net: The pandapower network object containing reference tables

        Returns:
            A deepcopy of the schema with foreign key checks added to appropriate columns
        """
    for col_name, column in schema.columns.items():
        if column.metadata and "foreign_key" in column.metadata:
            ref_table_name = column.metadata["foreign_key"]
            reference_df = net[ref_table_name]
            check = _create_index_validation_check(reference_df, col_name, schema.name or ref_table_name)
            existing_checks = column.checks or []
            column.checks = existing_checks + [check]
