import pandera.pandas as pa

schema = pa.DataFrameSchema(
    {
        "name": pa.Column(str, description="name of the extended ward equivalent"),
        "bus": pa.Column(int, pa.Check.ge(0), description="index of connected bus"),
        "ps_mw": pa.Column(float, description="constant active power demand [MW]"),
        "qs_mvar": pa.Column(float, description="constant reactive power demand [MVar]"),
        "pz_mw": pa.Column(float, description="constant impedance active power demand at 1.0 pu [MW]"),
        "qz_mvar": pa.Column(float, description="constant impedance reactive power demand at 1.0 pu [MVar]"),
        "in_service": pa.Column(bool, description="specifies if the extended ward equivalent is in service."),
    },
    strict=False,
)
