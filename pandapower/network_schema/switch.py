import pandera.pandas as pa

schema = pa.DataFrameSchema(
    {
        "bus": pa.Column(int, pa.Check.ge(0)),
        "name": pa.Column(str, nullable=True),
        "element": pa.Column(int, pa.Check.ge(0)),
        "et": pa.Column(str),
        "type": pa.Column(str),
        "closed": pa.Column(bool),
        "in_ka": pa.Column(float, nullable=True),
        "z_ohm": pa.Column(float),  # missing in docu
    },
    strict=False,
)
