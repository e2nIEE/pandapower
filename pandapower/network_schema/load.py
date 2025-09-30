import pandera.pandas as pa

schema = pa.DataFrameSchema(
    {
        "name": pa.Column(str),
        "bus": pa.Column(int, pa.Check.ge(0)),
        "p_mw": pa.Column(float, pa.Check.ge(0)),  # surely the docu must be wrong for ge0
        "q_mvar": pa.Column(float),
        "const_z_p_percent": pa.Column(float, pa.Check.between(min_value=0, max_value=100)),
        "const_i_p_percent": pa.Column(float, pa.Check.between(min_value=0, max_value=100)),
        "const_z_q_percent": pa.Column(float, pa.Check.between(min_value=0, max_value=100)),
        "const_i_q_percent": pa.Column(float, pa.Check.between(min_value=0, max_value=100)),
        "sn_mva": pa.Column(float, pa.Check.gt(0), nullable=True),
        "scaling": pa.Column(float, pa.Check.ge(0)),
        "in_service": pa.Column(bool),
        "type": pa.Column(str, pa.Check.isin(["wye", "delta"])),
        "controllable": pa.Column(bool, required=False),
        "zone": pa.Column(str, nullable=True, required=False),
        "max_p_mw": pa.Column(float, required=False),
        "min_p_mw": pa.Column(float, required=False),
        "max_q_mvar": pa.Column(float, required=False),
        "min_q_mvar": pa.Column(float, required=False),
    },
    strict=False,
)
