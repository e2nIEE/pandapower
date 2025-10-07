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

from pandapower.network_schema.motor import schema as motor_schema, res_schema as res_motor_schema
from pandapower.network_schema.asymmetric_load import schema as asymmetric_load_schema, res_schema as res_asymmetric_load_schema, res_schema_3ph as res_asymmetric_load_3ph_schema
from pandapower.network_schema.asymmetric_sgen import schema as asymmetric_sgen_schema, res_schema as res_asymmetric_sgen_schema, res_schema_3ph as res_asymmetric_sgen_3ph_schema
from pandapower.network_schema.storage import schema as storage_schema, res_schema as res_storage_schema, res_schema_3ph as res_storage_3ph_schema
from pandapower.network_schema.gen import schema as gen_schema, res_schema as res_gen_schema
from pandapower.network_schema.shunt import schema as shunt_schema, res_schema as res_shunt_schema

from pandapower.network_schema.trafo3w import schema as trafo3w
from pandapower.network_schema.line_dc import schema as line_dc
from pandapower.network_schema.impedance import schema as impedance
from pandapower.network_schema.tcsc import schema as tcsc
from pandapower.network_schema.dcline import schema as dcline
from pandapower.network_schema.ward import schema as ward
from pandapower.network_schema.xward import schema as xward
from pandapower.network_schema.measurement import schema as measurement
from pandapower.network_schema.source_dc import schema as source_dc
from pandapower.network_schema.load_dc import schema as load_dc
from pandapower.network_schema.b2b_vsc import schema as b2b_vsc

from pandapower.network_schema.svc import schema as svc_schema, res_schema as res_svc_schema
from pandapower.network_schema.vsc import schema as vsc_schema, res_schema as res_vsc_schema

from pandapower.network_schema.bi_vsc import schema as bi_vsc
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
        "motor": get_dtypes(motor_schema),
        "asymmetric_load": get_dtypes(asymmetric_load_schema),
        "asymmetric_sgen": get_dtypes(asymmetric_sgen_schema),
        "storage": get_dtypes(storage_schema),
        "gen": get_dtypes(gen_schema),
        "switch": get_dtypes(switch_schema),
        "shunt": get_dtypes(shunt_schema),
        "svc": get_dtypes(svc_schema),
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
        "vsc": get_dtypes(vsc_schema),
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
        "_empty_res_asymmetric_load": get_dtypes(res_asymmetric_load_schema),
        "_empty_res_asymmetric_sgen": get_dtypes(res_asymmetric_sgen_schema),
        "_empty_res_motor": get_dtypes(res_motor_schema),
        "_empty_res_sgen": get_dtypes(res_sgen_schema),
        "_empty_res_shunt": get_dtypes(res_shunt_schema),
        "_empty_res_svc": get_dtypes(res_svc_schema),
        "_empty_res_ssc": {
            "q_mvar": "f8",
            "vm_internal_pu": "f8",
            "va_internal_degree": "f8",
            "vm_pu": "f8",
            "va_degree": "f8",
        },
        "_empty_res_vsc": get_dtypes(res_vsc_schema),
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
        "_empty_res_asymmetric_load_3ph": get_dtypes(res_asymmetric_load_3ph_schema),
        "_empty_res_asymmetric_sgen_3ph": get_dtypes(res_asymmetric_sgen_3ph_schema),
        "_empty_res_storage": get_dtypes(res_storage_schema),
        "_empty_res_storage_3ph": get_dtypes(res_storage_3ph_schema),
        "_empty_res_gen": get_dtypes(res_gen_schema),
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
