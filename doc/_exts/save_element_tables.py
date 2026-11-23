"""
Sphinx extension for generating the element csv's for pandapower use.
"""

import shutil
from pathlib import Path

import pandas as pd
from pandera.api.pandas.container import DataFrameSchema
from sphinx.util import logging

file_path = Path(__file__).resolve().parent

logger = logging.getLogger(__name__)

check_name_to_symbol: dict[str, str] = {
    "greater_than": ">",
    "greater_than_or_equal_to": ">=",
    "less_than": "<",
    "less_than_or_equal_to": "<=",
    "equal_to": "==",
    "not_equal_to": "!=",
}

dtype_to_str: dict[str, str] = {
    "float64": "float",
    "float32": "float",
    "string": "string",
    "str": "string",
    "<U0": "string",
    "int64": "int",
    "int32": "int",
    "Int64": "int",
    "bool": "bool",
    "boolean": "bool",
}

bool_to_checkmark: dict[bool, str] = {True: "✓", False: pd.NA}


def get_dtype(dtype) -> str:
    # pandera engine dtypes expose the underlying pandas/numpy dtype as "type"
    dtype_name = str(getattr(dtype, "type", dtype))
    if dtype_name not in dtype_to_str:
        logger.warning("Unknown dtype %r in element table generation", dtype_name, once=True)
        return ""
    return dtype_to_str[dtype_name]


def format_check(check) -> str | None:
    """Format a single pandera check for the "Value Range" column.

    Returns None for checks that cannot be represented (e.g. custom checks).
    """
    stats = check.statistics or {}
    name = check.name

    if name in check_name_to_symbol:
        for key in ("value", "min_value", "max_value"):
            if key in stats:
                return f"{check_name_to_symbol[name]}{stats[key]}"
        return None
    if name == "in_range" and {"min_value", "max_value"} <= stats.keys():
        left = "[" if stats.get("include_min", True) else "("
        right = "]" if stats.get("include_max", True) else ")"
        return f"{left}{stats['min_value']}, {stats['max_value']}{right}"
    if name == "isin" and "allowed_values" in stats:
        return "{" + ", ".join(repr(value) for value in stats["allowed_values"]) + "}"
    return None


def get_checks(checks) -> str | pd.api.typing.NAType:
    formatted = [formatted for check in checks if (formatted := format_check(check)) is not None]
    return ", ".join(formatted) if formatted else pd.NA


def create_docu_csv_from_schema(schema: DataFrameSchema, path: Path, filename: str | None = None):
    columns_info = []

    for col_name, col in schema.columns.items():
        metadata = col.metadata or {}
        columns_info.append(
            {
                "Parameter": col_name,
                "Datatype": get_dtype(col.dtype),
                "Value Range": get_checks(col.checks),
                "Explanation": col.description or "",
                "nullable": bool_to_checkmark.get(bool(col.nullable), pd.NA),
                "required": bool_to_checkmark.get(bool(col.required), pd.NA),
                "optimal power flow": bool_to_checkmark.get(bool(metadata.get("opf")), pd.NA),
                "short circuit": bool_to_checkmark.get(bool(metadata.get("sc")), pd.NA),
                "3ph": bool_to_checkmark.get(bool(metadata.get("3ph")), pd.NA),
                "tdpf": bool_to_checkmark.get(bool(metadata.get("tdpf")), pd.NA),
                "doc": bool(metadata.get("doc", True)),
            }
        )

    # Create CSV with column metadata
    df = pd.DataFrame(columns_info)
    # Drop rows that should not appear in documentation and the doc column:
    df = df[df.doc]
    df = df.drop(columns=["doc"])
    # Drop empty columns
    df = df.dropna(how="all", axis=1)
    # fill NA fields with empty string
    df = df.fillna("")

    if filename is None:
        if schema.name:
            filename = f"{schema.name}.csv"
        else:
            filename = "column_schema.csv"
    df.to_csv(path / filename, index=False)


def gen_csv(_):
    import pandapower.network_schema as net_schema

    path = file_path / ".." / "elements" / "table_structures"
    if path.is_dir():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    logger.info("Generating element csv files")
    # Several schema aliases share the same name and content, only generate them once:
    schemas: dict[str, DataFrameSchema] = {}
    for name in dir(net_schema):
        schema = getattr(net_schema, name)
        if isinstance(schema, DataFrameSchema):
            schemas.setdefault(schema.name or name, schema)
    for schema in schemas.values():
        create_docu_csv_from_schema(schema, path)


def setup(app):
    app.connect("builder-inited", gen_csv)
    return {"version": "0.1"}


if __name__ == "__main__":
    gen_csv(None)
