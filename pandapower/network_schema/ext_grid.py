import pandera.pandas as pa

import pandapower as pp

schema = pa.DataFrameSchema(
    {
        "name": pa.Column(str),
        "bus": pa.Column(int, pa.Check.ge(0)),
        "vm_pu": pa.Column(float),
        "va_degree": pa.Column(float),
        "max_p_mw": pa.Column(float, required=False),
        "min_p_mw": pa.Column(float, required=False),
        "max_q_mvar": pa.Column(float, required=False),
        "min_q_mvar": pa.Column(float, required=False),
        "s_sc_max_mva": pa.Column(float, nullable=True),
        "s_sc_min_mva": pa.Column(float, nullable=True),
        "rx_max": pa.Column(float, nullable=True),
        "rx_min": pa.Column(float, nullable=True),
        "r0x0_max": pa.Column(float, required=False),
        "x0x_max": pa.Column(float, required=False),
        "slack_weight": pa.Column(float),
        "in_service": pa.Column(bool),
        "controllable": pa.Column(bool, required=False),
    },
    strict=False,
)


net = pp.networks.mv_oberrhein()

# This will pass validation
validated_df = schema.validate(net.ext_grid)
