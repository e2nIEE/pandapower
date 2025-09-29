import pandera.pandas as pa

import pandapower as pp

schema = pa.DataFrameSchema(
    {
        "name": pa.Column(str),
        "vn_kv": pa.Column(float),
        "type": pa.Column(str),
        "zone": pa.Column(str, nullable=True),
        "max_vm_pu": pa.Column(float, required=False),
        "min_vm_pu": pa.Column(float, required=False),
        "in_service": pa.Column(bool),
        "geo": pa.Column(str),
    },
    strict=False,
)


net = pp.networks.mv_oberrhein()

# This will pass validation
validated_df = schema.validate(net.bus)
