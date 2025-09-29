import pandera.pandas as pa

import pandapower as pp

schema = pa.DataFrameSchema(
    {
        "name": pa.Column(str),
        "bus": pa.Column(int, pa.Check.ge(0)),
        "p_mw": pa.Column(float),
        "q_mvar": pa.Column(float),
        "const_z_p_percent": pa.Column(float, pa.Check.ge(0)),
        "const_i_p_percent": pa.Column(float, pa.Check.ge(0)),
        "const_z_q_percent": pa.Column(float, pa.Check.ge(0)),
        "const_i_q_percent": pa.Column(float, pa.Check.ge(0)),
        "sn_mva": pa.Column(float, nullable=True),
        "scaling": pa.Column(float, pa.Check.ge(0)),
        "in_service": pa.Column(bool),
        "type": pa.Column(str),
        "controllable": pa.Column(bool, required=False),
        "zone": pa.Column(str, nullable=True, required=False),
        "max_p_mw": pa.Column(float, required=False),
        "min_p_mw": pa.Column(float, required=False),
        "max_q_mvar": pa.Column(float, required=False),
        "min_q_mvar": pa.Column(float, required=False),
    },
    strict=False,
)


net = pp.networks.mv_oberrhein()

# This will pass validation
validated_df = schema.validate(net.load)
