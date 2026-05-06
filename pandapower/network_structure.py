from typing import Any

from numpy import dtype
import pandas as pd
from pandera import DataFrameSchema

from pandapower._version import __version__, __format_version__
from pandapower.network_schema.tools.helper import get_dtypes
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
        # result tables
        "_empty_res_bus": res_bus_schema,
        "_empty_res_bus_dc": res_bus_dc_schema,
        "_empty_res_ext_grid": res_ext_grid_schema,
        "_empty_res_line": res_line_schema,
        "_empty_res_line_dc": res_line_dc_schema,
        "_empty_res_trafo": res_trafo_schema,
        "_empty_res_load": res_load_schema,
        "_empty_res_load_3ph": res_load_schema,
        "_empty_res_asymmetric_load": res_asymmetric_load_schema,
        "_empty_res_asymmetric_sgen": res_asymmetric_sgen_schema,
        "_empty_res_motor": res_motor_schema,
        "_empty_res_sgen": res_sgen_schema,
        "_empty_res_sgen_3ph": res_sgen_schema,
        "_empty_res_shunt": res_shunt_schema,
        "_empty_res_svc": res_svc_schema,
        "_empty_res_ssc": res_ssc_schema,
        "_empty_res_vsc": res_vsc_schema,
        "_empty_res_switch": res_switch_schema,
        "_empty_res_impedance": res_impedance_schema,
        "_empty_res_tcsc": res_tcsc_schema,
        "_empty_res_dcline": res_dcline_schema,
        "_empty_res_source_dc": res_source_dc_schema,
        "_empty_res_load_dc": res_load_dc_schema,
        "_empty_res_ward": res_ward_schema,
        "_empty_res_xward": res_xward_schema,
        "_empty_res_trafo_3ph": res_trafo_3ph_schema,
        "_empty_res_trafo3w": res_trafo3w_schema,
        "_empty_res_bus_3ph": res_bus_3ph_schema,
        "_empty_res_ext_grid_3ph": res_ext_grid_3ph_schema,
        "_empty_res_line_3ph": res_line_3ph_schema,
        "_empty_res_asymmetric_load_3ph": res_asymmetric_load_3ph_schema,
        "_empty_res_asymmetric_sgen_3ph": res_asymmetric_sgen_3ph_schema,
        "_empty_res_storage": res_storage_schema,
        "_empty_res_storage_3ph": res_storage_3ph_schema,
        "_empty_res_gen": res_gen_schema,
        "_empty_res_vsc_stacked": res_vsc_stacked_schema,
        "_empty_res_vsc_bipolar": res_vsc_bipolar_schema
    }
    # ruff: enable


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

def get_structure_dict(required_only: bool = True, metadata: list = []) -> dict:
    """
    This function returns the structure dict of the network
    """
    dtypes_dict: dict[str, Any] = {key: get_dtypes(val, required_only, metadata) for key, val in get_table_schema().items()}
    dtypes_dict.update({
        "pwl_cost": {  # TODO: convert to pandera
            "power_type": dtype(object),
            "element": "u4",
            "et": dtype(object),
            "points": dtype(object),
        },
        "poly_cost": {  # TODO: convert to pandera
            "element": "u4",
            "et": dtype(object),
            "cp0_eur": "f8",
            "cp1_eur_per_mw": "f8",
            "cp2_eur_per_mw2": "f8",
            "cq0_eur": "f8",
            "cq1_eur_per_mvar": "f8",
            "cq2_eur_per_mvar2": "f8",
        },
        "controller": {  # TODO: convert to pandera
            "object": dtype(object),
            "in_service": "bool",
            "order": "float64",
            "level": dtype(object),
            "initial_run": "bool",
            "recycle": dtype(object),
        },
        "group": {  # TODO: convert to pandera
            "name": dtype(object),
            "element_type": dtype(object),
            "element_index": dtype(object),
            "reference_column": dtype(object),
        },
        # result tables
        "_empty_res_bus": {
            "vm_pu": "f8",
            "va_degree": "f8",
            "p_mw": "f8",
            "q_mvar": "f8"},
        "_empty_res_bus_dc": {
            "vm_pu": "f8",
            "p_mw": "f8"},
        "_empty_res_ext_grid": {
            "p_mw": "f8",
            "q_mvar": "f8"},
        "_empty_res_line": {
            "p_from_mw": "f8",
            "q_from_mvar": "f8",
            "p_to_mw": "f8",
            "q_to_mvar": "f8",
            "pl_mw": "f8",
            "ql_mvar": "f8",
            "i_from_ka": "f8",
            "i_to_ka": "f8",
            "i_ka": "f8",
            "vm_from_pu": "f8",
            "va_from_degree": "f8",
            "vm_to_pu": "f8",
            "va_to_degree": "f8",
            "loading_percent": "f8",
        },
        "_empty_res_line_dc": {
            "p_from_mw": "f8",
            "p_to_mw": "f8",
            "pl_mw": "f8",
            "i_from_ka": "f8",
            "i_to_ka": "f8",
            "i_ka": "f8",
            "vm_from_pu": "f8",
            "vm_to_pu": "f8",
            "loading_percent": "f8",
        },
        "_empty_res_trafo": {
            "p_hv_mw": "f8",
            "q_hv_mvar": "f8",
            "p_lv_mw": "f8",
            "q_lv_mvar": "f8",
            "pl_mw": "f8",
            "ql_mvar": "f8",
            "i_hv_ka": "f8",
            "i_lv_ka": "f8",
            "vm_hv_pu": "f8",
            "va_hv_degree": "f8",
            "vm_lv_pu": "f8",
            "va_lv_degree": "f8",
            "loading_percent": "f8",
        },
        "_empty_res_load": {
            "p_mw": "f8",
            "q_mvar": "f8"
        },
        "_empty_res_asymmetric_load": {
            "p_mw": "f8",
            "q_mvar": "f8"
        },
        "_empty_res_asymmetric_sgen": {
            "p_mw": "f8",
            "q_mvar": "f8"
        },
        "_empty_res_motor": {
            "p_mw": "f8",
            "q_mvar": "f8"
        },
        "_empty_res_sgen": {
            "p_mw": "f8",
            "q_mvar": "f8"
        },
        "_empty_res_shunt": {
            "p_mw": "f8",
            "q_mvar": "f8",
            "vm_pu": "f8"
        },
        "_empty_res_svc": {
            "thyristor_firing_angle_degree": "f8",
            "x_ohm": "f8",
            "q_mvar": "f8",
            "vm_pu": "f8",
            "va_degree": "f8",
        },
        "_empty_res_ssc": {
            "q_mvar": "f8",
            "vm_internal_pu": "f8",
            "va_internal_degree": "f8",
            "vm_pu": "f8",
            "va_degree": "f8",
        },
        "_empty_res_vsc": {
            "p_mw": "f8",
            "q_mvar": "f8",
            "p_dc_mw": "f8",
            "vm_internal_pu": "f8",
            "va_internal_degree": "f8",
            "vm_pu": "f8",
            "va_degree": "f8",
            "vm_internal_dc_pu": "f8",
            "vm_dc_pu": "f8",
        },
        "_empty_res_switch": {
            "i_ka": "f8",
            "loading_percent": "f8",
            "p_from_mw": "f8",
            "q_from_mvar": "f8",
            "p_to_mw": "f8",
            "q_to_mvar": "f8",
        },
        "_empty_res_impedance": {
            "p_from_mw": "f8",
            "q_from_mvar": "f8",
            "p_to_mw": "f8",
            "q_to_mvar": "f8",
            "pl_mw": "f8",
            "ql_mvar": "f8",
            "i_from_ka": "f8",
            "i_to_ka": "f8",
        },
        "_empty_res_tcsc": {
            "thyristor_firing_angle_degree": "f8",
            "x_ohm": "f8",
            "p_from_mw": "f8",
            "q_from_mvar": "f8",
            "p_to_mw": "f8",
            "q_to_mvar": "f8",
            "pl_mw": "f8",
            "ql_mvar": "f8",
            "i_ka": "f8",
            "vm_from_pu": "f8",
            "va_from_degree": "f8",
            "vm_to_pu": "f8",
            "va_to_degree": "f8",
        },
        "_empty_res_dcline": {
            "p_from_mw": "f8",
            "q_from_mvar": "f8",
            "p_to_mw": "f8",
            "q_to_mvar": "f8",
            "pl_mw": "f8",
            "vm_from_pu": "f8",
            "va_from_degree": "f8",
            "vm_to_pu": "f8",
            "va_to_degree": "f8",
        },
        "_empty_res_source_dc": {
            "p_dc_mw": "f8"
        },
        "_empty_res_load_dc": {
            "p_dc_mw": "f8"
        },
        "_empty_res_ward": {
            "p_mw": "f8",
            "q_mvar": "f8",
            "vm_pu": "f8"
        },
        "_empty_res_xward": {
            "p_mw": "f8",
            "q_mvar": "f8",
            "vm_pu": "f8",
            "va_internal_degree": "f8",
            "vm_internal_pu": "f8",
        },
        "_empty_res_trafo_3ph": {
            "p_a_hv_mw": "f8",
            "q_a_hv_mvar": "f8",
            "p_b_hv_mw": "f8",
            "q_b_hv_mvar": "f8",
            "p_c_hv_mw": "f8",
            "q_c_hv_mvar": "f8",
            "p_a_lv_mw": "f8",
            "q_a_lv_mvar": "f8",
            "p_b_lv_mw": "f8",
            "q_b_lv_mvar": "f8",
            "p_c_lv_mw": "f8",
            "q_c_lv_mvar": "f8",
            "pl_a_mw": "f8",
            "ql_a_mvar": "f8",
            "pl_b_mw": "f8",
            "ql_b_mvar": "f8",
            "pl_c_mw": "f8",
            "ql_c_mvar": "f8",
            "i_a_hv_ka": "f8",
            "i_a_lv_ka": "f8",
            "i_b_hv_ka": "f8",
            "i_b_lv_ka": "f8",
            "i_c_hv_ka": "f8",
            "i_c_lv_ka": "f8",
            "loading_a_percent": "f8",
            "loading_b_percent": "f8",
            "loading_c_percent": "f8",
            "loading_percent": "f8",
        },
        "_empty_res_trafo3w": {
            "p_hv_mw": "f8",
            "q_hv_mvar": "f8",
            "p_mv_mw": "f8",
            "q_mv_mvar": "f8",
            "p_lv_mw": "f8",
            "q_lv_mvar": "f8",
            "pl_mw": "f8",
            "ql_mvar": "f8",
            "i_hv_ka": "f8",
            "i_mv_ka": "f8",
            "i_lv_ka": "f8",
            "vm_hv_pu": "f8",
            "va_hv_degree": "f8",
            "vm_mv_pu": "f8",
            "va_mv_degree": "f8",
            "vm_lv_pu": "f8",
            "va_lv_degree": "f8",
            "va_internal_degree": "f8",
            "vm_internal_pu": "f8",
            "loading_percent": "f8",
        },
        "_empty_res_bus_3ph": {
            "vm_a_pu": "f8",
            "va_a_degree": "f8",
            "vm_b_pu": "f8",
            "va_b_degree": "f8",
            "vm_c_pu": "f8",
            "va_c_degree": "f8",
            "p_a_mw": "f8",
            "q_a_mvar": "f8",
            "p_b_mw": "f8",
            "q_b_mvar": "f8",
            "p_c_mw": "f8",
            "q_c_mvar": "f8",
        },
        "_empty_res_ext_grid_3ph": {
            "p_a_mw": "f8",
            "q_a_mvar": "f8",
            "p_b_mw": "f8",
            "q_b_mvar": "f8",
            "p_c_mw": "f8",
            "q_c_mvar": "f8",
        },
        "_empty_res_line_3ph": {
            "p_a_from_mw": "f8",
            "q_a_from_mvar": "f8",
            "p_b_from_mw": "f8",
            "q_b_from_mvar": "f8",
            "p_c_from_mw": "f8",
            "q_c_from_mvar": "f8",
            "p_a_to_mw": "f8",
            "q_a_to_mvar": "f8",
            "p_b_to_mw": "f8",
            "q_b_to_mvar": "f8",
            "p_c_to_mw": "f8",
            "q_c_to_mvar": "f8",
            "pl_a_mw": "f8",
            "ql_a_mvar": "f8",
            "pl_b_mw": "f8",
            "ql_b_mvar": "f8",
            "pl_c_mw": "f8",
            "ql_c_mvar": "f8",
            "i_a_from_ka": "f8",
            "i_a_to_ka": "f8",
            "i_b_from_ka": "f8",
            "i_b_to_ka": "f8",
            "i_c_from_ka": "f8",
            "i_c_to_ka": "f8",
            "i_a_ka": "f8",
            "i_b_ka": "f8",
            "i_c_ka": "f8",
            "i_n_from_ka": "f8",
            "i_n_to_ka": "f8",
            "i_n_ka": "f8",
            "loading_a_percent": "f8",
            "loading_b_percent": "f8",
            "loading_c_percent": "f8",
        },
        "_empty_res_load_3ph": {
            "p_mw": "f8",
            "q_mvar": "f8",
        },
        "_empty_res_asymmetric_load_3ph": {
            "p_a_mw": "f8",
            "q_a_mvar": "f8",
            "p_b_mw": "f8",
            "q_b_mvar": "f8",
            "p_c_mw": "f8",
            "q_c_mvar": "f8",
        },
        "_empty_res_asymmetric_sgen_3ph": {
            "p_a_mw": "f8",
            "q_a_mvar": "f8",
            "p_b_mw": "f8",
            "q_b_mvar": "f8",
            "p_c_mw": "f8",
            "q_c_mvar": "f8",
        },
        "_empty_res_storage": {
            "p_mw": "f8",
            "q_mvar": "f8"
        },
        "_empty_res_storage_3ph": {
            "p_a_mw": "f8",
            "p_b_mw": "f8",
            "p_c_mw": "f8",
            "q_a_mvar": "f8",
            "q_b_mvar": "f8",
            "q_c_mvar": "f8",
        },
        "_empty_res_gen": {
            "p_mw": "f8",
            "q_mvar": "f8",
            "va_degree": "f8",
            "vm_pu": "f8",
        },
        "_empty_res_protection": {
            "switch_id": "f8",
            "prot_type": dtype(object),
            "trip_melt": "bool",
            "act_param": dtype(object),
            "act_param_val": "f8",
            "trip_melt_time_s": "f8",
        },  # TODO: convert to pandera
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
    })
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
