import importlib.util
import logging
import os
from pathlib import Path

import pandera as pa

logger = logging.getLogger()


def get_element_schema(element_name: str) -> pa.DataFrameSchema | None:
    """
    get the pandera DataFrameSchema for an element by its name.

    Parameters:
        element_name: the name of the element

    Returns:
         the pandera DataFrameSchema for the element or None if it can't be found.
    """

    def _dynamic_import(element, schema_path):
        # Dynamic import for schemata from files
        spec = importlib.util.spec_from_file_location(element, schema_path)
        if spec is None:
            logger.warning(f"Schema for {element} not found, no spec")
            return None
        schema_module = importlib.util.module_from_spec(spec)
        loader = spec.loader
        if loader is None:
            logger.warning(f"Schema for {element} not found, no loader")
            return None
        loader.exec_module(schema_module)
        return schema_module

    schema_path = Path(Path(__file__).parents[1], f"{element_name}.py")
    if not os.path.exists(schema_path):
        return None
    schema_module = _dynamic_import(element_name, schema_path)
    if schema_module is None:
        return None
    return getattr(schema_module, f"{element_name}_schema", None)


def get_dtypes(schema: pa.DataFrameSchema, required_only: bool = True, metadata: list = []) -> dict:
    """
    Extract column data types from a Pandera DataFrame schema.

    This function parses a Pandera schema and returns a dictionary mapping
    column names to their corresponding data types. Optionally filters to
    include only required columns or columns with specific metadata.

    Args:
        schema (pa.DataFrameSchema): The Pandera DataFrame schema to extract
            data types from.
        required_only (bool, optional): If True, only includes required columns
            in the result. If False, includes all columns regardless of their
            required status. Defaults to True.
        metadata (list, optional): List of metadata keys to check. Columns will
            be included if they have metadata containing any of these keys with
            truthy values. Defaults to empty list.

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

    Note:
        When metadata parameter is provided, columns with matching metadata keys
        having truthy values will be included regardless of required_only setting.
    """
    return {
        name: col.dtype.type
        for name, col in schema.columns.items()
        if not required_only
        or schema.columns[name].required
        or schema.columns[name].metadata is not None
        and any(item in schema.columns[name].metadata for item in metadata)
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
