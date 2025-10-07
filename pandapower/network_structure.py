from numpy import dtype

from pandapower._version import __version__, __format_version__
from pandapower.network_schema.tools import get_dtypes
from pandapower.network_schema.bus import (
    schema as bus_schema,
    res_schema as res_bus_schema,
    res_schema_3ph as res_bus_3ph_schema,
)
from pandapower.network_schema.bus_dc import schema as bus_dc_schema, res_schema as res_bus_dc_schema
from pandapower.network_schema.load import schema as load_schema, res_schema as res_load_schema
from pandapower.network_schema.sgen import schema as sgen_schema, res_schema as res_sgen_schema
from pandapower.network_schema.switch import schema as switch_schema, res_schema as res_switch_schema
from pandapower.network_schema.ext_grid import (
    schema as ext_grid_schema,
    res_schema as res_ext_grid_schema,
    res_schema_3ph as res_ext_grid_3ph_schema,
)
from pandapower.network_schema.line import (
    schema as line_schema,
    res_schema as res_line_schema,
    res_schema_3ph as res_line_3ph_schema,
)
from pandapower.network_schema.trafo import (
    schema as trafo_schema,
    res_schema as res_trafo_schema,
    res_schema_3ph as res_trafo_3ph_schema,
)

from pandapower.network_schema.trafo3w import schema as trafo3w_schema, res_schema as res_trafo3w_schema
from pandapower.network_schema.line_dc import schema as line_dc_schema, res_schema as res_line_dc_schema
from pandapower.network_schema.impedance import schema as impedance_schema, res_schema as res_impedance_schema
from pandapower.network_schema.tcsc import schema as tcsc_schema, res_schema as res_tcsc_schema
from pandapower.network_schema.dcline import schema as dcline_schema, res_schema as res_dcline_schema
from pandapower.network_schema.ward import schema as ward_schema, res_schema as res_ward_schema
from pandapower.network_schema.xward import schema as xward_schema, res_schema as res_xward_schema
from pandapower.network_schema.measurement import schema as measurement_schema
from pandapower.network_schema.source_dc import schema as source_dc_schema, res_schema as res_source_dc_schema
from pandapower.network_schema.load_dc import schema as load_dc_schema, res_schema as res_load_dc_schema
from pandapower.network_schema.b2b_vsc import schema as b2b_vsc_schema, res_schema as res_b2b_vsc_schema
from pandapower.network_schema.bi_vsc import schema as bi_vsc_schema, res_schema as res_bi_vsc_schema


def get_structure_dict() -> dict:
    """
    This function returns the structure dict of the network
    """
    return {
        # structure data
        "bus": get_dtypes(bus_schema),
        "bus_dc": get_dtypes(bus_dc_schema),
        "load": get_dtypes(load_schema),
        "sgen": get_dtypes(sgen_schema),
        "motor": {
            "name": dtype(str),
            "bus": "i8",
            "pn_mech_mw": "f8",
            "cos_phi": "f8",
            "cos_phi_n": "f8",
            "efficiency_percent": "f8",
            "efficiency_n_percent": "f8",
            "loading_percent ": "f8",  # marco says this parameter does not make sense
            "scaling": "f8",
            "lrc_pu": "f8",
            "rx": "f8",
            "vn_kv": "f8",
            "in_service": "bool",
        },
        "asymmetric_load": {
            "name": dtype(str),
            "bus": "i8",
            "p_a_mw": "f8",
            "p_b_mw": "f8",
            "p_c_mw": "f8",
            "q_a_mvar": "f8",
            "q_b_mvar": "f8",
            "q_c_mvar": "f8",
            "sn_mva": "f8",
            "scaling": "f8",
            "in_service": "bool",
            "type": dtype(str),
        },
        "asymmetric_sgen": {
            "name": dtype(str),
            "type": dtype(str),
            "bus": "i8",
            "p_a_mw": "f8",
            "q_a_mvar": "f8",
            "p_b_mw": "f8",
            "q_b_mvar": "f8",
            "p_c_mw": "f8",
            "q_c_mvar": "f8",
            "sn_mva": "f8",
            "scaling": "f8",
            "in_service": "bool",
            "current_source": "bool",  # missing in docu, not a create method parameter, kwargs?
        },
        "storage": {
            "name": dtype(str),
            "bus": "i8",
            "p_mw": "f8",
            "q_mvar": "f8",
            "sn_mva": "f8",
            "scaling": "f8",
            "max_e_mwh": "f8",
            "min_e_mwh": "f8",
            "max_p_mw": "f8",
            "min_p_mw": "f8",
            "soc_percent": "f8",
            "max_q_mvar": "f8",
            "min_q_mvar": "f8",
            "controllable": "bool",
            "in_service": "bool",
            "type": dtype(str),  # missing in docu
        },
        "gen": {
            "name": dtype(str),
            "type": dtype(str),
            "bus": "i8",
            "p_mw": "f8",
            "vm_pu": "f8",
            "sn_mva": "f8",
            "max_q_mvar": "f8",
            "min_q_mvar": "f8",
            "scaling": "f8",
            "max_p_mw": "f8",
            "min_p_mw": "f8",
            "vn_kv": "f8",
            "xdss_pu": "f8",
            "rdss_ohm": "f8",
            "cos_phi": "f8",
            "in_service": "bool",
            "power_station_trafo": "i8",
            "id_q_capability_characteristic": "i8",
            "curve_style": dtype(str),
            "reactive_capability_curve": "bool",
            "slack_weight": "f8",  # missing in docu
            "slack": "bool",  # missing in docu
            "controllable": "bool",  # missing in docu
        },
        "switch": get_dtypes(switch_schema),
        "shunt": {
            "name": dtype(str),
            "bus": "i8",
            "p_mw": "f8",
            "q_mvar": "f8",
            "vn_kv": "f8",
            "step": "i8",
            "max_step": "i8",
            "in_service": "bool",
            "step_dependency_table": "bool",
            "id_characteristic_table": "i8",
        },
        "svc": {
            "name": dtype(str),
            "bus": "i8",
            "x_l_ohm": "f8",
            "x_cvar_ohm": "f8",
            "set_vm_pu": "f8",
            "thyristor_firing_angle_degree": "f8",
            "controllable": "bool",
            "in_service": "bool",
            "min_angle_degree": "f8",
            "max_angle_degree": "f8",
        },
        "ssc": {
            "name": dtype(str),
            "bus": "i8",
            "r_ohm": "f8",
            "x_ohm": "f8",
            "set_vm_pu": "f8",
            "vm_internal_pu": "f8",
            "va_internal_degree": "f8",
            "controllable": "bool",
            "in_service": "bool",
        },
        "vsc": {
            "name": dtype(str),
            "bus": "i8",
            "bus_dc": "i8",
            "r_ohm": "f8",
            "x_ohm": "f8",
            "r_dc_ohm": "f8",
            "pl_dc_mw": "f8",
            "control_mode_ac": dtype(str),
            "control_value_ac": "f8",
            "control_mode_dc": dtype(str),
            "control_value_dc": "f8",
            "controllable": "bool",
            "in_service": "bool",
            "ref_bus": "u4",  # missing in docu
        },
        "ext_grid": get_dtypes(ext_grid_schema),
        "line": get_dtypes(line_schema),
        "line_dc": get_dtypes(line_dc_schema),
        "trafo": get_dtypes(trafo_schema),
        "trafo3w": get_dtypes(trafo3w_schema),
        "impedance": get_dtypes(impedance_schema),
        "tcsc": get_dtypes(tcsc_schema),
        "dcline": get_dtypes(dcline_schema),
        "ward": get_dtypes(ward_schema),
        "xward": get_dtypes(xward_schema),
        "measurement": get_dtypes(measurement_schema),
        "pwl_cost": {  # not a datastructure or element?
            "power_type": dtype(object),
            "element": "u4",
            "et": dtype(object),
            "points": dtype(object),
        },
        "poly_cost": {  # not a datastructure or element?
            "element": "u4",
            "et": dtype(object),
            "cp0_eur": "f8",
            "cp1_eur_per_mw": "f8",
            "cp2_eur_per_mw2": "f8",
            "cq0_eur": "f8",
            "cq1_eur_per_mvar": "f8",
            "cq2_eur_per_mvar2": "f8",
        },
        "controller": {  # not a datastructure or element?
            "object": dtype(object),
            "in_service": "bool",
            "order": "float64",
            "level": dtype(object),
            "initial_run": "bool",
            "recycle": dtype(object),
        },
        "group": {  # not a datastructure or element?
            "name": dtype(object),
            "element_type": dtype(object),
            "element_index": dtype(object),
            "reference_column": dtype(object),
        },
        "source_dc": get_dtypes(source_dc_schema),
        "load_dc": get_dtypes(load_dc_schema),
        "b2b_vsc": get_dtypes(b2b_vsc_schema),
        "bi_vsc": get_dtypes(bi_vsc_schema),
        # result tables
        "_empty_res_bus": get_dtypes(res_bus_schema),
        "_empty_res_bus_dc": get_dtypes(res_bus_dc_schema),
        "_empty_res_ext_grid": get_dtypes(res_ext_grid_schema),
        "_empty_res_line": get_dtypes(res_line_schema),
        "_empty_res_line_dc": get_dtypes(res_line_dc_schema),
        "_empty_res_trafo": get_dtypes(res_trafo_schema),
        "_empty_res_load": get_dtypes(res_load_schema),
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
        "_empty_res_sgen": get_dtypes(res_sgen_schema),
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
        "_empty_res_switch": get_dtypes(res_switch_schema),
        "_empty_res_impedance": get_dtypes(res_impedance_schema),
        "_empty_res_tcsc": get_dtypes(res_tcsc_schema),
        "_empty_res_dcline": get_dtypes(res_dcline_schema),
        "_empty_res_source_dc": get_dtypes(res_source_dc_schema),
        "_empty_res_load_dc": get_dtypes(res_load_dc_schema),
        "_empty_res_ward": get_dtypes(res_ward_schema),
        "_empty_res_xward": get_dtypes(res_xward_schema),
        "_empty_res_trafo_3ph": get_dtypes(res_trafo_3ph_schema),
        "_empty_res_trafo3w": get_dtypes(res_trafo3w_schema),
        "_empty_res_bus_3ph": get_dtypes(res_bus_3ph_schema),
        "_empty_res_ext_grid_3ph": get_dtypes(res_ext_grid_3ph_schema),
        "_empty_res_line_3ph": get_dtypes(res_line_3ph_schema),
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
        },
        "_empty_res_b2b_vsc": get_dtypes(res_b2b_vsc_schema),
        "_empty_res_bi_vsc": get_dtypes(res_bi_vsc_schema),
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


def get_std_type_structure_dict() -> dict:
    """
    This function returns the structure dict of the std_types
    """
    return {
        # structure data
        "line": {
            "r_ohm_per_km": "f8",
            "r0_ohm_per_km": "f8",
            "x_ohm_per_km": "f8",
            "x0_ohm_per_km": "f8",
            "c_nf_per_km": "f8",
            "c0_nf_per_km": "f8",
            "g_us_per_km": "f8",
            "g0_us_per_km": "f8",
            "max_i_ka": "f8",
            "type": dtype(object),
            "q_mm2": "f8",
            "alpha": "f8",
            "voltage_rating": dtype(object),
        },
        "line_dc": {
            "r_ohm_per_km": "f8",
            "r0_ohm_per_km": "f8",
            "g_us_per_km": "f8",
            "g0_us_per_km": "f8",
            "max_i_ka": "f8",
            "type": dtype(object),
            "q_mm2": "f8",
            "alpha": "f8",
            "voltage_rating": "f8",
        },
        "trafo": {
            "sn_mva": "f8",
            "vn_hv_kv": "f8",
            "vn_lv_kv": "f8",
            "vk_percnet": "f8",
            "vk0_percent": "f8",
            "vkr_percent": "f8",
            "vkr0_percent": "f8",
            "pfe_kw": "f8",
            "i0_percent": "f8",
            "shift_degree": "f8",
            "vector_group": dtype(object),
            "tap_side": dtype(object),
            "tap_neutral": "f8",
            "tap_min": "f8",
            "tap_max": "f8",
            "tap_step_degree": "f8",
            "tap_step_percent": "f8",
            "tap_change_type": dtype(object),
            "trafo_characteristic_table": "bool",
        },
        "trafo3w": {
            "sn_hv_mva": "f8",
            "sn_mv_mva": "f8",
            "sn_lv_mva": "f8",
            "vn_hv_kv": "f8",
            "vn_mv_kv": "f8",
            "vn_lv_kv": "f8",
            "vk_hv_percent": "f8",
            "vk_mv_percent": "f8",
            "vk_lv_percent": "f8",
            "vk0_hv_percent": "f8",
            "vk0_mv_percent": "f8",
            "vk0_lv_percent": "f8",
            "vkr_hv_percent": "f8",
            "vkr_mv_percent": "f8",
            "vkr_lv_percent": "f8",
            "vkr0_hv_percent": "f8",
            "vkr0_mv_percent": "f8",
            "vkr0_lv_percent": "f8",
            "pfe_kw": "f8",
            "i0_percent": "f8",
            "shift_mv_degree": "f8",
            "shift_lv_degree": "f8",
            "tap_side": dtype(object),
            "tap_neutral": "f8",
            "tap_min": "f8",
            "tap_max": "f8",
            "tap_step_percent": "f8",
            "tap_step_degree": "f8",
            "tap_pos": "f8",
            "tap_at_star_point": "bool",
            "tap_changer_type": dtype(object),
            "id_characteristic_table": "bool",
            "vector_group": dtype(object),
        },
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
