import pandas as pd
import pandera.pandas as pa

from pandapower import pandapowerNet


def _create_index_validation_check(reference_df: pd.DataFrame, column_name: str, element_name: str) -> pa.Check:
    """
    Creates a pandera check function that validates if values exist in the index of a reference dataframe.

    This function generates a validation check that ensures all values in a pandas Series
    are present in the index of the provided reference dataframe. It's commonly used
    to validate foreign key relationships, such as ensuring bus IDs in a transformer
    table exist in the bus table.

    Parameters
    ----------
    reference_df : pd.DataFrame
        The reference dataframe whose index will be used for validation.
    column_name : str
        The name of the column being validated. Used for generating specific
        error messages and determining the reference type (e.g., 'dc' columns
        reference 'bus_dc' table).
    element_name : str
        The name of the element/table being validated. Used in error messages
        for better debugging.

    Returns
    -------
    pa.Check
        A pandera Check object that can be used in schema validation.
        The check will raise a ValueError if any values are not found
        in the reference index.

    Raises
    ------
    ValueError
        Raised by the returned check function when validation fails,
        including details about failing values and their indices.

    Examples
    --------
    >>> bus_df = pd.DataFrame(index=[1, 2, 3])
    >>> check = _create_index_validation_check(bus_df, 'hv_bus', 'trafo')
    >>> # This check can now be used to validate that trafo hv_bus values exist in bus_df
    """
    reference_index = set(reference_df.index)

    def check_values_in_reference_index(series: pd.Series) -> bool:
        if "dc" in column_name:
            ref_name = "bus_dc"
        else:
            ref_name = "bus"
        mask = series.isin(reference_index)
        if not mask.all():
            failing_values = series[~mask].unique()
            failing_indices = series.index[~mask].values.tolist()
            raise ValueError(
                f"The following values for net.{element_name}.{column_name} at index {failing_indices} are not in the index of the {ref_name}-dataframe: {failing_values.tolist()}"
            )
        return True

    return pa.Check(
        check_values_in_reference_index,
        name=f"{check_values_in_reference_index.__name__}_{column_name}",
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

    Parameters
    ----------
    reference_df : pd.DataFrame
        The reference dataframe whose index serves as the valid value set
        for all columns being validated.
    columns_to_validate : list[str]
        List of column names that should be validated against the reference
        dataframe's index. All columns will be treated as int64 dtype.
    element_name : str
        Name of the element/table being validated. Used for schema naming
        and error message generation.

    Returns
    -------
    pa.DataFrameSchema
        A pandera DataFrameSchema object with validation rules for all
        specified columns. The schema is non-strict, allowing additional
        columns not specified in columns_to_validate.

    Notes
    -----
    - All validated columns are configured as int64 dtype and non-nullable
    - The schema uses strict=False, allowing additional columns beyond those specified

    - Each column gets its own index validation check via _create_index_validation_check

    Examples
    --------
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


def _bus_index_validation(element: str, net: pandapowerNet):
    """
    Validates that all bus references in a network element exist in the corresponding bus tables.

    This function ensures that every bus index referenced in an element's bus-related columns
    actually exists in the network's bus or bus_dc tables. It handles both AC buses (bus table)
    and DC buses (bus_dc table) separately.

    Parameters
    ----------
    element : str
        Name of the network element to validate (e.g., 'line', 'load', 'gen', etc.).
        Must not be 'bus' or 'bus_dc' as these are the reference tables themselves.
    net : pandapowerNet
        The pandapower network object containing all network elements and bus tables.

    Raises
    ------
    ValidationError
        If any bus reference in the element doesn't exist in the corresponding bus table.

    Notes
    -----
    - Automatically identifies all columns containing 'bus' in their name
    - Separates DC bus columns (containing 'dc') from regular AC bus columns

    - For 'switch' elements, also validates the 'element' column
    - Uses multi-column reference schema validation to ensure referential integrity

    Examples
    --------
    >>> _bus_index_validation('line', net)  # Validates from_bus, to_bus columns
    >>> _bus_index_validation('load', net)  # Validates bus column
    """
    if element not in ["bus", "bus_dc"]:
        bus_columns = [col for col in net[element].columns if "bus" in col.lower()]
        if element == "switch":
            bus_columns.append("element")
        dc_items = [item for item in bus_columns if "dc" in item]
        non_dc_items = [item for item in bus_columns if "dc" not in item]
        for df, lst in [(net.bus, non_dc_items), (net.bus_dc, dc_items)]:
            _create_multi_column_reference_schema(
                reference_df=df, columns_to_validate=lst, element_name=element
            ).validate(net[element])
