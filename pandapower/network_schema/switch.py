import pandera.pandas as pa

schema = pa.DataFrameSchema(
    {
        "bus": pa.Column(int, pa.Check.ge(0)),
        "name": pa.Column(str, nullable=True),
        "element": pa.Column(int, pa.Check.ge(0)),
        "et": pa.Column(str, pa.Check.isin(["b", "l", "t", "t3"])),
        "type": pa.Column(str, pa.Check.isin(["CB", "LS", "LBS", "DS"])),
        "closed": pa.Column(bool),
        "in_ka": pa.Column(float, pa.Check.gt(0), nullable=True),
        "z_ohm": pa.Column(float),  # missing in docu
    },
    strict=False,
)
