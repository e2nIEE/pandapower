import pandera.pandas as pa

schema = pa.DataFrameSchema(  # in methodcall but not parameter docu: generator_type, max_i_ka, kappa, lrc_pu
    {
        "name": pa.Column(str),
        "bus": pa.Column(int, pa.Check.ge(0)),
        "p_mw": pa.Column(float),
        "q_mvar": pa.Column(float),
        "sn_mva": pa.Column(float),
        "scaling": pa.Column(float),
        "min_p_mw": pa.Column(float, required=False),
        "max_p_mw": pa.Column(float, required=False),
        "min_q_mvar": pa.Column(float, required=False),
        "max_q_mvar": pa.Column(float, required=False),
        "controllable": pa.Column(bool, required=False),
        "k": pa.Column(float, required=False),
        "rx": pa.Column(float, required=False),
        "in_service": pa.Column(bool),
        "id_q_capability_characteristic": pa.Column(int, nullable=True),
        "curve_style": pa.Column(str, nullable=True),
        "reactive_capability_curve": pa.Column(bool),
        "type": pa.Column(str),  # missing in docu
        "current_source": pa.Column(bool),  # missing in docu
    },
    strict=False,
)
