import pandera.pandas as pa

schema = pa.DataFrameSchema(
    {
        "name": pa.Column(str),
        "bus": pa.Column(int, pa.Check.ge(0)),
        "vm_pu": pa.Column(float, pa.Check.gt(0)),
        "va_degree": pa.Column(float),
        "max_p_mw": pa.Column(float, required=False),
        "min_p_mw": pa.Column(float, required=False),
        "max_q_mvar": pa.Column(float, required=False),
        "min_q_mvar": pa.Column(float, required=False),
        "s_sc_max_mva": pa.Column(float, pa.Check.gt(0), nullable=True, required=False),
        "s_sc_min_mva": pa.Column(float, pa.Check.gt(0), nullable=True, required=False),
        "rx_max": pa.Column(float, pa.Check.between(min_value=0, max_value=1), nullable=True, required=False),
        "rx_min": pa.Column(float, pa.Check.between(min_value=0, max_value=1), nullable=True, required=False),
        "r0x0_max": pa.Column(float, pa.Check.between(min_value=0, max_value=1), required=False),
        "x0x_max": pa.Column(float, pa.Check.between(min_value=0, max_value=1), required=False),
        "slack_weight": pa.Column(float),  # missing in docu
        "in_service": pa.Column(bool),
        "controllable": pa.Column(bool, required=False),  # missing in docu
    },
    strict=False,
)
