import pandera.pandas as pa

ward_schema = pa.DataFrameSchema(
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

res_ward_schema = pa.DataFrameSchema(
    {
        "p_mw": pa.Column(float, description="active power demand of the ward equivalent [MW]"),
        "q_mvar": pa.Column(float, description="reactive power demand of the ward equivalent [kVar]"),
        "vm_pu": pa.Column(float, description="voltage at the ward bus [p.u]"),
    },
)
