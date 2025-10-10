import pandera.pandas as pa

schema = pa.DataFrameSchema(
    {  # where is the documentation? #TODO:
        "name": pa.Column(str, description=""),
        "bus": pa.Column(int, pa.Check.ge(0), description=""),
        "bus_dc_plus": pa.Column(int, pa.Check.ge(0), description=""),
        "bus_dc_minus": pa.Column(int, pa.Check.ge(0), description=""),
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

res_schema = pa.DataFrameSchema(
    {  # where is the documentation?
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
