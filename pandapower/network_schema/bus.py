import pandera.pandas as pa

schema = pa.DataFrameSchema(  # in methodcall but not parameter docu: geodata, coords
    {
        "name": pa.Column(str),
        "vn_kv": pa.Column(float),
        "type": pa.Column(str),
        "zone": pa.Column(str, nullable=True),
        "max_vm_pu": pa.Column(float, required=False),
        "min_vm_pu": pa.Column(float, required=False),
        "in_service": pa.Column(bool),
        "geo": pa.Column(
            str
        ),  # missing in docu, not a create method parameter, kwargs?
    },
    strict=False,
)
