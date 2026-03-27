"""
Sphinx extension for generating the element csv's for pandapower use.
"""
import os
import shutil

import json
import pandas as pd
from pandera.api.pandas.container import DataFrameSchema

file_path = os.path.dirname(os.path.realpath(__file__))

check_name_to_symbol: dict[str, str] = {
    "greater_than": ">",
    "greater_than_or_equal_to": ">=",
    "less_than": "<",
    "less_than_or_equal_to": "<=",
    "isin": ""
}

dtype_to_str: dict[str, str] = {
    "float64": "float",
    "string[python]": "string",
    "str": "string",
    "int64": "int",
    "Int64": "int",
    "bool": "bool",
    "boolean": "bool",
    "": "",
}

bool_to_checkmark: dict[bool, str] = {
    True: '✓',
    False: pd.NA
}

def get_dtype(dtype: str):
    if dtype in dtype_to_str:
        return dtype_to_str[dtype]
    print(f"Unknown dtype {dtype}")
    return ""

def create_docu_csv_from_schema(schema: DataFrameSchema, path: str, filename: str | None = None):
    schema_json = schema.to_json()
    schema_dict = json.loads(schema_json)

    columns_info = []

    def get_checks(checks: list):
        if checks is None:
            return pd.NA

        for check in checks:
            if check["options"]["check_name"] == "in_range":
                return f"[{check['min_value']}, {check['max_value']}]"
            return f"{check_name_to_symbol[check["options"]["check_name"]]}{check['value']}"
        return pd.NA

    def _get_metadata(name: str, kind: str):
        metadata = schema.columns[name].metadata
        return metadata is not None and kind in metadata

    for col_name, col_details in schema_dict["columns"].items():
        columns_info.append(
            {
                "Parameter": col_name,
                "Datatype": get_dtype(col_details.get("dtype", "")),
                "Value Range": get_checks(col_details.get("checks")),
                "Explanation": col_details.get("description", ""),
                "nullable": bool_to_checkmark[col_details.get("nullable", True)],
                "required": bool_to_checkmark[col_details.get("required", False)],
                "optimal power flow": bool_to_checkmark[_get_metadata(col_name, "opf")],
                "short circuit": bool_to_checkmark[_get_metadata(col_name, "sc")],
                "3ph": bool_to_checkmark[_get_metadata(col_name, "3ph")],
            }
        )

    # Create CSV with column metadata
    df = pd.DataFrame(columns_info)
    df = df.dropna(how='all', axis=1)
    df = df.fillna('')

    # pd.set_option("display.max_columns", None)
    # print(df)
    if filename is None:
        if schema.name:
            filename = f"{schema.name}.csv"
        else:
            filename = "column_schema.csv"
    df.to_csv(os.path.join(path, filename), index=False)


def gen_csv(_):
    import pandapower.network_schema as net_schema

    path = os.path.join(file_path, '..', 'elements', 'table_structures')
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)
    print("Generating element csv files")
    schemas = [getattr(net_schema, s) for s in dir(net_schema) if isinstance(getattr(net_schema, s), DataFrameSchema)]
    for schema in schemas:
        create_docu_csv_from_schema(schema, path)


def setup(app):
    app.connect('builder-inited', gen_csv)
    return {'version': '0.1'}


if __name__ == '__main__':
    gen_csv(None)
