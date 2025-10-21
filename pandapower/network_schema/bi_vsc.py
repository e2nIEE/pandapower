import pandera.pandas as pa
import pandas as pd

# TODO: where is the documentation?

bi_vsc_schema = pa.DataFrameSchema(
    {
        "name": pa.Column(pd.StringDtype, description=""),
        "bus": pa.Column(int, pa.Check.ge(0), description="", metadata={"foreign_key": "bus.index"}),
        "bus_dc_plus": pa.Column(int, pa.Check.ge(0), description="", metadata={"foreign_key": "bus_dc.index"}),
        "bus_dc_minus": pa.Column(int, pa.Check.ge(0), description="", metadata={"foreign_key": "bus_dc.index"}),
        "r_ohm": pa.Column(float, description=""),
        "x_ohm": pa.Column(float, description=""),
        "r_dc_ohm": pa.Column(float, description=""),
        "pl_dc_mw": pa.Column(float, description=""),
        "control_mode_ac": pa.Column(str, description=""),
        "control_value_ac": pa.Column(float, description=""),
        "control_mode_dc": pa.Column(str, description=""),
        "control_value_dc": pa.Column(float, description=""),
        "controllable": pa.Column(bool, description=""),
        "in_service": pa.Column(bool, description=""),
    },
    strict=False,
)

res_bi_vsc_schema = pa.DataFrameSchema(
    {
        "p_mw": pa.Column(float, description=""),
        "q_mvar": pa.Column(float, description=""),
        "p_dc_mw_p": pa.Column(float, description=""),
        "p_dc_mw_m": pa.Column(float, description=""),
        "vm_internal_pu": pa.Column(float, description=""),
        "vm_internal_degree": pa.Column(float, description=""),
        "vm_pu": pa.Column(float, description=""),
        "va_degree": pa.Column(float, description=""),
        "vm_internal_dc_pu_p": pa.Column(float, description=""),
        "vm_internal_dc_pu_m": pa.Column(float, description=""),
        "vm_dc_pu_p": pa.Column(float, description=""),
        "vm_dc_pu_m": pa.Column(float, description=""),
    },
)
