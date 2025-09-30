import pandera.pandas as pa

schema = pa.DataFrameSchema(  # in methodcall but not parameter docu: geodata, coords
    {
        "name": pa.Column(str),
        "vn_kv": pa.Column(float, pa.Check.gt(0)),
        "type": pa.Column(str, pa.Check.isin(["n", "b", "m"])),
        "zone": pa.Column(str, nullable=True),
        "max_vm_pu": pa.Column(float, pa.Check.gt(0), required=False),
        "min_vm_pu": pa.Column(float, pa.Check.gt(0), required=False),
        "in_service": pa.Column(bool),
        "geo": pa.Column(str),  # missing in docu, not a create method parameter, kwargs?
    },
    strict=False,
)
