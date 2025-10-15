import os
import importlib.util
from pathlib import Path
import logging

import pandas as pd
import pandera.pandas as pa

from pandapower import pandapowerNet

logger = logging.getLogger()


def get_dtypes(schema: pa.DataFrameSchema, required_only: bool = True):
    if required_only:
        return {name: col.dtype.type for name, col in schema.columns.items() if schema.columns[name].required}
    else:
        return {name: col.dtype.type for name, col in schema.columns.items() if schema.columns[name]}


def validate_element_by_schema(schema: pa.DataFrameSchema, element: pd.DataFrame):
    assert schema.validate(element) is not None


def validate_column_group_dependency(columns):
    return lambda df: (
        # Either none of the columns exists or all of the columns exist
        not any(col in df.columns for col in columns) or all(col in df.columns for col in columns)
    )


def validate_dataframes_for_network(net: pandapowerNet):
    """
    Validate dataframes using schemas from schema folder

    Args:
        net: pandapower Network
    """

    for element in net:
        schema_path = Path(Path(__file__).parent, f"{element}.py")

        if not os.path.exists(schema_path):
            continue

        # Dynamic import
        spec = importlib.util.spec_from_file_location(element, schema_path)
        if spec is None:
            logger.warning(f"Schema for {element} not found, no spec")
            continue
        schema_module = importlib.util.module_from_spec(spec)
        loader = spec.loader
        if loader is None:
            logger.warning(f"Schema for {element} not found, no loader")
            continue
        loader.exec_module(schema_module)

        schema = getattr(schema_module, f"{element}_schema")

        try:
            schema.validate(net[element])
        except Exception as e:
            raise pa.errors.SchemaError(data=e, message=f"Validation failed for {element}", schema=schema)


def create_docu_csv_from_schema(schema):
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

    def get_metadata(schema: pa.DataFrameSchema, name: str, kind: str):
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
                "optimal power flow": get_metadata(schema, col_name, "opf"),
                "short circuit": get_metadata(schema, col_name, "sc"),
                "3ph": get_metadata(schema, col_name, "3ph"),
                "Explanation": col_details.get("description", ""),
            }
        )

    # Create CSV with column metadata
    df = pd.DataFrame(columns_info)

    pd.set_option("display.max_columns", None)
    print(df)
    df.to_csv("column_schema.csv", index=False)
