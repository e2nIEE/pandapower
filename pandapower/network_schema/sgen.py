import pandera.pandas as pa

schema = pa.DataFrameSchema(  # in methodcall but not parameter docu: generator_type, max_i_ka, kappa, lrc_pu
    {
        "name": pa.Column(str),
        "bus": pa.Column(int, pa.Check.ge(0)),
        "p_mw": pa.Column(float, pa.Check.le(0)),  # surely the docu must be wrong for le0
        "q_mvar": pa.Column(float),
        "sn_mva": pa.Column(float, pa.Check.gt(0)),
        "scaling": pa.Column(float, pa.Check.ge(0)),
        "min_p_mw": pa.Column(float, required=False),
        "max_p_mw": pa.Column(float, required=False),
        "min_q_mvar": pa.Column(float, required=False),
        "max_q_mvar": pa.Column(float, required=False),
        "controllable": pa.Column(bool, required=False),
        "k": pa.Column(float, pa.Check.ge(0), required=False),
        "rx": pa.Column(float, pa.Check.ge(0), required=False),
        "in_service": pa.Column(bool),
        "id_q_capability_characteristic": pa.Column(int, nullable=True),
        "curve_style": pa.Column(str, pa.Check.isin(["straightLineYValues", "constantYValue"]), nullable=True),
        "reactive_capability_curve": pa.Column(bool),
        "type": pa.Column(str, pa.Check.isin(["PV", "WP", "CHP"])),
        "current_source": pa.Column(bool),  # missing in docu
    },
    strict=False,
)
