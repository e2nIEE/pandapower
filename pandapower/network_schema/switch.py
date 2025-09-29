import pandera.pandas as pa

import pandapower as pp

schema = pa.DataFrameSchema(
    {
        "bus": pa.Column(int, pa.Check.ge(0)),
        "name": pa.Column(str),
        "element": pa.Column(int, pa.Check.ge(0)),
        "et": pa.Column(str),
        "type": pa.Column(str),
        "closed": pa.Column(bool),
        "in_ka": pa.Column(float, nullable=True),
        "z_ohm": pa.Column(float),
    },
    strict=False,
)


net = pp.networks.mv_oberrhein()

# This will pass validation
validated_df = schema.validate(net.switch)
