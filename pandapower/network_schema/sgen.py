import pandera.pandas as pa

import pandapower as pp

schema = pa.DataFrameSchema(
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
        "type": pa.Column(str),
        "current_source": pa.Column(bool),
    },
    strict=False,
)


net = pp.networks.mv_oberrhein()

# This will pass validation
validated_df = schema.validate(net.sgen)
