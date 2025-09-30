import pandera.pandas as pa

schema = pa.DataFrameSchema(  # in methodcall but not parameter docu: geodata, alpha, temperature_degree_celsius
    {
        "name": pa.Column(str),
        "std_type": pa.Column(str),
        "from_bus": pa.Column(int, pa.Check.ge(0)),
        "to_bus": pa.Column(int, pa.Check.ge(0)),
        "length_km": pa.Column(float, pa.Check.gt(0)),
        "r_ohm_per_km": pa.Column(float, pa.Check.ge(0)),
        "x_ohm_per_km": pa.Column(float, pa.Check.ge(0)),
        "c_nf_per_km": pa.Column(float, pa.Check.ge(0)),
        "r0_ohm_per_km": pa.Column(float, pa.Check.ge(0), required=False),
        "x0_ohm_per_km": pa.Column(float, pa.Check.ge(0), required=False),
        "c0_nf_per_km": pa.Column(float, pa.Check.ge(0), required=False),
        "g_us_per_km": pa.Column(float, pa.Check.ge(0)),
        "max_i_ka": pa.Column(float, pa.Check.gt(0)),
        "parallel": pa.Column(int, pa.Check.ge(1)),  # other elements have ge0
        "df": pa.Column(float, pa.Check.between(min_value=0, max_value=1)),
        "type": pa.Column(str, pa.Check.isin(["ol", "cs"])),
        "max_loading_percent": pa.Column(float, pa.Check.gt(0), required=False),
        "endtemp_degree": pa.Column(float, pa.Check.gt(0), required=False),  # not in create method call
        "in_service": pa.Column(bool),
        "geo": pa.Column(str),  # missing in docu
    },
    strict=False,
)
