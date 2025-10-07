import pandera.pandas as pa

schema = pa.DataFrameSchema(  # docu hat sehr viele fehler...
    {
        "name": pa.Column(str, description="name of the static generator"),
        "type": pa.Column(str, description="type of source"),
        "bus_dc": pa.Column(  # not the same name in docu
            int, description="index of connected bus"
        ),
        "p_mw": pa.Column(
            float,
            pa.Check.le(0),
            description="active power of the static generator [MW]",
        ),
        "scaling": pa.Column(
            float,
            pa.Check.ge(0),
            description="scaling factor for the active and reactive power",
        ),  # not in create method
        "vm_pu": pa.Column(
            float,
            description="",
        ),  # missing in docu
        "controllable": pa.Column(
            bool,
            description="states if sgen is controllable or not, sgen will not be used as a flexibility if it is not controllable",
        ),
        "in_service": pa.Column(bool, description="specifies if the generator is in service."),
    },
    strict=False,
)

res_schema = pa.DataFrameSchema(
    {"p_dc_mw": pa.Column(float, description="resulting active power demand after scaling [MW]")},
)
