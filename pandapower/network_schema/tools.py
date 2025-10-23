import os
import importlib.util
from pathlib import Path
import logging

import numpy as np
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


def _create_index_validation_check(reference_df: pd.DataFrame, column_name: str = ""):
    """
    Erstellt eine Pandera-Check-Funktion, die validiert, ob Werte im Index
    des Referenz-DataFrames existieren.
    """
    reference_index = set(reference_df.index)

    def check_values_in_reference_index(series: pd.Series) -> np.bool_:
        return series.isin(reference_index).all()

    check_name = "values_must_exist_in_reference_index"
    if column_name:
        check_name += f"_{column_name}"

    return pa.Check(
        check_values_in_reference_index,
        name=check_name,
        error="Alle Werte müssen im Index des Referenz-DataFrames existieren",
    )


def _create_multi_column_reference_schema(
    reference_df: pd.DataFrame, columns_to_validate: list[str], schema_name: str = "MultiColumnReferenceSchema"
) -> pa.DataFrameSchema:
    """
    Erstellt ein Pandera-Schema, das mehrere Spalten gegen den Index
    eines Referenz-DataFrames validiert.

    Parameters:
    -----------
    reference_df : pd.DataFrame
        Das Referenz-DataFrame, dessen Index für die Validierung verwendet wird
    columns_to_validate : List[str]
        Liste der Spaltennamen, die validiert werden sollen
    schema_name : str
        Name des Schemas

    Returns:
    --------
    DataFrameSchema
        Das erstellte Pandera-Schema
    """

    schema_columns = {}

    # Erstelle Validierung für die spezifizierten Spalten
    for col_name in columns_to_validate:
        schema_columns[col_name] = pa.Column(
            dtype="int64", checks=[_create_index_validation_check(reference_df, col_name)], nullable=False
        )

    return pa.DataFrameSchema(columns=schema_columns, name=schema_name, strict=False)


def _dynamic_import(element, schema_path):
    # Dynamic import
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

        schema = getattr(_dynamic_import(element, schema_path), f"{element}_schema")

        if schema is None:
            continue

        try:
            schema.validate(net[element])
        except Exception as e:
            raise pa.errors.SchemaError(data=e, message=f"Validation failed for {element}", schema=schema)

        try:
            _bus_index_validation(element, net)
        except Exception as e:
            raise e  # TODO: better exception


def _bus_index_validation(element, net: pandapowerNet):
    # bus index validation
    # makes sure every bus/bus_dc set in an element exists in the bus/bus_dc index
    if element not in ["bus", "bus_dc"]:
        bus_columns = [col for col in net[element].columns if "bus" in col.lower()]
        if element == "switch":
            bus_columns.append("element")
        dc_items = [item for item in bus_columns if "dc" in item]
        non_dc_items = [item for item in bus_columns if "dc" not in item]
        if non_dc_items:
            _create_multi_column_reference_schema(reference_df=net.bus, columns_to_validate=non_dc_items).validate(
                net[element]
            )
        if dc_items:
            _create_multi_column_reference_schema(reference_df=net.bus_dc, columns_to_validate=dc_items).validate(
                net[element]
            )


def get_metadata_columns_from_schema_dict(schema, name):
    """Extract column names that have opf=True in their metadata."""
    return [
        col_name for col_name, col_schema in schema.items() if col_schema.metadata and col_schema.metadata.get(name)
    ]


def create_checks_from_metadata(names, schema_columns):
    checks = []
    for name in names:
        for col in get_metadata_columns_from_schema_dict(schema_columns, name):
            checks.append(
                pa.Check(
                    validate_column_group_dependency(col),
                    error=f"{name} columns have dependency violations. Please ensure {col} columns are present in the dataframe.",
                )
            )
    return checks


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
