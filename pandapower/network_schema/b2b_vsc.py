import pandera.pandas as pa
import pandas as pd

b2b_vsc_schema = pa.DataFrameSchema(
    {
        "name": pa.Column(pd.StringDtype, required=False),
        "bus": pa.Column(int, pa.Check.ge(0), metadata={"foreign_key": "bus.index"}),
        "bus_dc_plus": pa.Column(int, pa.Check.ge(0), metadata={"foreign_key": "bus_dc.index"}),
        "bus_dc_minus": pa.Column(int, pa.Check.ge(0), metadata={"foreign_key": "bus_dc.index"}),
        "r_ohm": pa.Column(float),
        "x_ohm": pa.Column(float),
        "r_dc_ohm": pa.Column(float),
        "pl_dc_mw": pa.Column(float, required=False),
        "control_mode_ac": pa.Column(str, pa.Check.isin(["vm_pu", "q_mvar", "slack"])),
        "control_value_ac": pa.Column(float),
        "control_mode_dc": pa.Column(str, pa.Check.isin(["vm_pu", "p_mw"])),
        "control_value_dc": pa.Column(float),
        "controllable": pa.Column(bool),
        "in_service": pa.Column(bool),
    },
    strict=False,
)

res_b2b_vsc_schema = pa.DataFrameSchema(
    {
        "p_mw": pa.Column(float, description="total active power consumption of B2B VSC [MW]"),
        "q_mvar": pa.Column(float, description="total reactive power consumption of B2B VSC [MVAr]"),
        "p_dc_mw_p": pa.Column(float, description="voltage magnitude at vsc internal bus [pu]"),
        "p_dc_mw_m": pa.Column(float, description="voltage angle at vsc internal bus [degree]"),
        "vm_internal_pu": pa.Column(float, description="voltage magnitude at B2B VSC ac bus [pu]"),
        "vm_internal_degree": pa.Column(float, description="voltage angle at B2B VSC ac bus [degree]"),
        "vm_pu": pa.Column(float, description="active power of the plus side of the B2B VSC [MW]"),
        "va_degree": pa.Column(float, description="active power of the minus side of B2B VSC [MW]"),
        "vm_internal_dc_pu_p": pa.Column(float, description="voltage angle at the plus B2B VSC ac bus [pu]"),
        "vm_internal_dc_pu_m": pa.Column(float, description="voltage angle at the minus B2B VSC ac bus [pu]"),
        "vm_dc_pu_p": pa.Column(float, description="voltage magnitude at the plus B2B VSC ac bus [pu]"),
        "vm_dc_pu_m": pa.Column(float, description="voltage magnitude at the minus B2B VSC ac bus [pu]"),
    },
)
