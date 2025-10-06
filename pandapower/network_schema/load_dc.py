import pandera.pandas as pa

schema = pa.DataFrameSchema(
    {
        "name": pa.Column(str, description="name of the load"),
        "bus_dc": pa.Column(int, pa.Check.ge(0), description="index of connected bus"),
        "p_dc_mw": pa.Column(
            float, pa.Check.ge(0), description="active power of the load [MW]"
        ),  # surely the docu must be wrong for ge0
        "scaling": pa.Column(float, pa.Check.ge(0), description="scaling factor for active and reactive power"),
        "in_service": pa.Column(bool, description="specifies if the load is in service."),
        "type": pa.Column(str, description="A string describing the type."),
        "controllable": pa.Column(
            bool,
            description="States if load is controllable or not, load will not be used as a flexibilty if it is not controllable",
        ),
    },
    strict=False,
)
