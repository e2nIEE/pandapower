import pandera.pandas as pa

import pandapower as pp

schema = pa.DataFrameSchema(
    {
        "name": pa.Column(str),
        "std_type": pa.Column(str),
        "from_bus": pa.Column(int, pa.Check.ge(0)),
        "to_bus": pa.Column(int, pa.Check.ge(0)),
        "length_km": pa.Column(float, pa.Check.ge(0)),
        "r_ohm_per_km": pa.Column(float),
        "x_ohm_per_km": pa.Column(float),
        "c_nf_per_km": pa.Column(float),
        "r0_ohm_per_km": pa.Column(float, required=False),
        "x0_ohm_per_km": pa.Column(float, required=False),
        "c0_nf_per_km": pa.Column(float, required=False),
        "g_us_per_km": pa.Column(float),
        "max_i_ka": pa.Column(float),
        "parallel": pa.Column(int, pa.Check.ge(0)),
        "df": pa.Column(float),
        "type": pa.Column(str),
        "max_loading_percent": pa.Column(float, required=False),
        "endtemp_degree": pa.Column(float, required=False),
        "in_service": pa.Column(bool),
        "geo": pa.Column(str),
    },
    strict=False,
)


net = pp.networks.mv_oberrhein()

# This will pass validation
validated_df = schema.validate(net.line)
