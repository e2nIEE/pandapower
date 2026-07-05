import pandera as pa


def get_dtypes(schema: pa.DataFrameSchema, required_only: bool = True) -> dict:
    """
    Extract column data types from a Pandera DataFrame schema.

    This function parses a Pandera schema and returns a dictionary mapping
    column names to their corresponding data types. Optionally filters to
    include only required columns.

    Args:
        schema (pa.DataFrameSchema): The Pandera DataFrame schema to extract
            data types from.
        required_only (bool, optional): If True, only includes required columns
            in the result. If False, includes all columns regardless of their
            required status. Defaults to True.

    Returns:
        dict: A dictionary where keys are column names (str) and values are
            the corresponding data type objects (type).

    Example:
        >>> import pandera as pa
        >>> schema = pa.DataFrameSchema({
        ...     "name": pa.Column(str, required=True),
        ...     "age": pa.Column(int, required=True),
        ...     "score": pa.Column(float, required=False)
        ... })
        >>> get_dtypes(schema)
        {'name': <class 'str'>, 'age': <class 'int'>}
        >>> get_dtypes(schema, required_only=False)
        {'name': <class 'str'>, 'age': <class 'int'>, 'score': <class 'float'>}
    """
    return {
        name: col.dtype.type
        for name, col in schema.columns.items()
        if not required_only or schema.columns[name].required
    }


def create_docu_csv_from_schema(schema: pa.DataFrameSchema):
    import pandas as pd
    import json

    schema_json = schema.to_json()
    schema_dict = json.loads(schema_json)

    columns_info = []

    def get_checks(checks: list):
        if checks is None:
            return ""

        for check in checks:
            if check["options"]["check_name"] == "greater_than":
                return f">{check['value']}"
            elif check["options"]["check_name"] == "greater_than_or_equal_to":
                return f">={check['value']}"
            elif check["options"]["check_name"] == "less_than":
                return f"<{check['value']}"
            elif check["options"]["check_name"] == "less_than_or_equal_to":
                return f"<={check['value']}"
            elif check["options"]["check_name"] == "in_range":
                return f"[{check['min_value']}, {check['max_value']}]"
            elif check["options"]["check_name"] == "isin":
                return f"{check['value']}"
        return ""

    def _get_metadata(schema: pa.DataFrameSchema, name: str, kind: str):
        try:
            return schema.columns[name].metadata[kind]
        except:
            return False

    for col_name, col_details in schema_dict["columns"].items():
        columns_info.append(
            {
                "Parameter": col_name,
                "Datatype": col_details.get("dtype", ""),
                "Value Range": get_checks(col_details.get("checks")),
                "nullable": col_details.get("nullable", True),
                "required": col_details.get("required", False),
                "optimal power flow": _get_metadata(schema, col_name, "opf"),
                "short circuit": _get_metadata(schema, col_name, "sc"),
                "3ph": _get_metadata(schema, col_name, "3ph"),
                "Explanation": col_details.get("description", ""),
            }
        )

    # Create CSV with column metadata
    df = pd.DataFrame(columns_info)

    pd.set_option("display.max_columns", None)
    print(df)
    df.to_csv("column_schema.csv", index=False)
