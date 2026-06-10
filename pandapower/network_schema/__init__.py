"""
init for network_schema

ALL VARIABLES EXPOSED HERE SHOULD BE OF TYPE pandera.DataFrameSchema
"""
from pandapower.network_schema.asymmetric_load import (
    asymmetric_load_schema,
    res_asymmetric_load_schema,
    res_asymmetric_load_3ph_schema,
)
from pandapower.network_schema.asymmetric_sgen import (
    asymmetric_sgen_schema,
    res_asymmetric_sgen_schema,
    res_asymmetric_sgen_3ph_schema,
)
from pandapower.network_schema.vsc_stacked import vsc_stacked_schema, res_vsc_stacked_schema
from pandapower.network_schema.vsc_bipolar import vsc_bipolar_schema, res_vsc_bipolar_schema
from pandapower.network_schema.bus import (
    bus_schema, res_bus_schema, res_bus_3ph_schema, res_bus_sc_schema, res_bus_est_schema # res_bus_est is res_bus
)
from pandapower.network_schema.bus_dc import bus_dc_schema, res_bus_dc_schema
from pandapower.network_schema.dcline import dcline_schema, res_dcline_schema
from pandapower.network_schema.ext_grid import ext_grid_schema, res_ext_grid_schema, res_ext_grid_3ph_schema
from pandapower.network_schema.gen import gen_schema, res_gen_schema
from pandapower.network_schema.impedance import (
    impedance_schema, res_impedance_schema, res_impedance_est_schema  # res_impedance_est is res_impedance
)
from pandapower.network_schema.line import (
    line_schema, res_line_schema, res_line_3ph_schema, res_line_sc_schema,
    res_line_est_schema # res_line_est is res_line
)
from pandapower.network_schema.line_dc import line_dc_schema, res_line_dc_schema
from pandapower.network_schema.load import load_schema, res_load_schema, res_load_3ph_schema  # res_load_3ph is res_load
from pandapower.network_schema.load_dc import load_dc_schema, res_load_dc_schema
from pandapower.network_schema.measurement import measurement_schema
from pandapower.network_schema.motor import motor_schema, res_motor_schema
from pandapower.network_schema.sgen import sgen_schema, res_sgen_schema, res_sgen_3ph_schema  # res_sgen_3ph is res_sgen
from pandapower.network_schema.shunt import (
    shunt_schema, res_shunt_schema, res_shunt_est_schema  # res_shunt_est is res_shunt
)
from pandapower.network_schema.source_dc import source_dc_schema, res_source_dc_schema
from pandapower.network_schema.ssc import ssc_schema, res_ssc_schema
from pandapower.network_schema.storage import storage_schema, res_storage_schema, res_storage_3ph_schema
from pandapower.network_schema.svc import svc_schema, res_svc_schema
from pandapower.network_schema.switch import (
    switch_schema, res_switch_schema, res_switch_est_schema  # res_switch_est is res_switch
)
from pandapower.network_schema.tcsc import tcsc_schema, res_tcsc_schema
from pandapower.network_schema.trafo import (
    trafo_schema, res_trafo_schema, res_trafo_3ph_schema, res_trafo_sc_schema,
    res_trafo_est_schema  # res_trafo_est is res_trafo
)
from pandapower.network_schema.trafo3w import (
    trafo3w_schema, res_trafo3w_schema, res_trafo3w_sc_schema, res_trafo3w_est_schema  # res_trafo3w_est is res_trafo3w
)
from pandapower.network_schema.vsc import vsc_schema, res_vsc_schema
from pandapower.network_schema.ward import ward_schema, res_ward_schema
from pandapower.network_schema.xward import xward_schema, res_xward_schema
