import pandera.pandas as pa

schema = pa.DataFrameSchema(
    {
        "name": pa.Column(str),
        "bus": pa.Column(int, pa.Check.ge(0)),
        "bus_dc_plus": pa.Column(int, pa.Check.ge(0)),
        "bus_dc_minus": pa.Column(int, pa.Check.ge(0)),
        "r_ohm": pa.Column(float),
        "x_ohm": pa.Column(float),
        "r_dc_ohm": pa.Column(float),
        "pl_dc_mw": pa.Column(float),
        "control_mode_ac": pa.Column(str, pa.Check.isin(["vm_pu", "q_mvar", "slack"])),
        "control_value_ac": pa.Column(float),
        "control_mode_dc": pa.Column(str, pa.Check.isin(["vm_pu", "p_mw"])),
        "control_value_dc": pa.Column(float),
        "controllable": pa.Column(bool),
        "in_service": pa.Column(bool),
    },
    strict=False,
)
