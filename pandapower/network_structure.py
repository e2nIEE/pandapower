from typing import Any

from numpy import dtype
import pandas as pd
from pandera import DataFrameSchema

from pandapower._version import __version__, __format_version__
from pandapower.network_schema.tools.helper import get_dtypes
import pandapower.network_schema as _schema_module
from pandapower.network_schema import *  # noqa: F403


def get_table_schema() -> dict[str, DataFrameSchema]:
    # ruff: noqa: F405
    return {
        "bus": bus_schema,
        "bus_dc": bus_dc_schema,
        "load": load_schema,
        "sgen": sgen_schema,
        "motor": motor_schema,
        "asymmetric_load": asymmetric_load_schema,
        "asymmetric_sgen": asymmetric_sgen_schema,
        "storage": storage_schema,
        "gen": gen_schema,
        "switch": switch_schema,
        "shunt": shunt_schema,
        "svc": svc_schema,
        "ssc": ssc_schema,
        "vsc": vsc_schema,
        "ext_grid": ext_grid_schema,
        "line": line_schema,
        "line_dc": line_dc_schema,
        "trafo": trafo_schema,
        "trafo3w": trafo3w_schema,
        "impedance": impedance_schema,
        "tcsc": tcsc_schema,
        "dcline": dcline_schema,
        "ward": ward_schema,
        "xward": xward_schema,
        "measurement": measurement_schema,
        "source_dc": source_dc_schema,
        "load_dc": load_dc_schema,
        "vsc_stacked": vsc_stacked_schema,
        "vsc_bipolar": vsc_bipolar_schema,
    }
    # ruff: enable


def get_results_schema() -> dict[str, DataFrameSchema]:
    """
    Get a dict of all schamas that start with res_ and end with _schema listed in the init of network_schema module

    Returns:
        A dict with pandapowerNet df name as key and the schema as value
    """
    return {
        name.removesuffix("_schema"): getattr(_schema_module, name)
        for name in dir(_schema_module)
        if name.startswith("res_") and name.endswith("_schema")
    }


def get_results_structure_dict() -> dict[str, dict[str, str]]:
    """
    Builds the structure_dict for all result tables from pandera

    Returns: A dict where key is the table name and value is a dict of column names and the dtypes

    """
    # TODO: convert to pandera
    additional_tables = {
        "res_protection": {
            "switch_id": "f8",
            "prot_type": dtype(object),
            "trip_melt": "bool",
            "act_param": dtype(object),
            "act_param_val": "f8",
            "trip_melt_time_s": "f8",
        }
    }
    dtypes_dict: dict[str, Any] = {key: get_dtypes(val) for key, val in get_results_schema().items()}
    dtypes_dict.update(additional_tables)
    return dtypes_dict


def get_column_info(table: str, column: str) -> dict[str, str | bool | dict] | None:
    schema = get_table_schema().get(table, None)
    if schema is None:
        return schema
    column = schema.columns.get(column, None)
    if column is None:
        return column
    return column.__dict__

def get_default_value(table: str, column: str) -> Any:
    column_info: dict[str, Any] | None = get_column_info(table, column)
    if column_info is not None and 'metadata' in column_info and 'default' in column_info['metadata']:
        return column_info["metadata"]["default"]
    return pd.NA

def get_structure_dict(required_only: bool = True, metadata: list | None = None) -> dict:
    """
    This function returns the structure dict of the network
    """
    if metadata is None:
        metadata = []
    dtypes_dict: dict[str, Any] = {key: get_dtypes(val, required_only, metadata) for key, val in get_table_schema().items()}
    # TODO: convert to pandera
    additional_schema = {
        "pwl_cost": {
            "power_type": dtype(object),
            "element": "u4",
            "et": dtype(object),
            "points": dtype(object),
        },
        "poly_cost": {
            "element": "u4",
            "et": dtype(object),
            "cp0_eur": "f8",
            "cp1_eur_per_mw": "f8",
            "cp2_eur_per_mw2": "f8",
            "cq0_eur": "f8",
            "cq1_eur_per_mvar": "f8",
            "cq2_eur_per_mvar2": "f8",
            "redispatch_up_eur_per_mw": "f8",
            "redispatch_down_eur_per_mw": "f8",
        },
        "controller": {
            "object": dtype(object),
            "in_service": "bool",
            "order": "float64",
            "level": dtype(object),
            "initial_run": "bool",
            "recycle": dtype(object),
        },
        "group": {
            "name": dtype(object),
            "element_type": dtype(object),
            "element_index": dtype(object),
            "reference_column": pd.StringDtype(),
        }
    }
    internal_values = {
        # internal
        "_ppc": None,
        "_ppc0": None,
        "_ppc1": None,
        "_ppc2": None,
        "_is_elements": None,
        "_pd2ppc_lookups": [
            {
                "bus": None,
                "bus_dc": None,
                "ext_grid": None,
                "gen": None,
                "branch": None,
                "branch_dc": None,
            }
        ],
        "version": __version__,
        "format_version": __format_version__,
        "converged": False,
        "OPF_converged": False,
        "name": "",
        "f_hz": 50.0,
        "sn_mva": 1,
    }
    dtypes_dict.update(additional_schema)
    dtypes_dict.update(internal_values)
    return dtypes_dict


def get_std_type_structure_dict() -> dict:
    """
    This function returns the structure dict of the std_types
    """
    return {
        # structure data
        "line": get_dtypes(line_schema),
        "line_dc": get_dtypes(line_dc_schema),
        "trafo": get_dtypes(trafo_schema),
        "trafo3w": get_dtypes(trafo3w_schema),
        "fuse": {
            "fuse_type": dtype(object),
            "i_rated_a": "f8",
            "t_avg": dtype(object),
            "t_min": dtype(object),
            "t_total": dtype(object),
            "x_avg": dtype(object),
            "x_min": dtype(object),
            "x_total": dtype(object),
        },
    }
