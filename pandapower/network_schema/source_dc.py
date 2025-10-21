import pandera.pandas as pa
import pandas as pd

source_dc_schema = pa.DataFrameSchema(  # TODO: docu hat sehr viele fehler...
    {
        "name": pa.Column(pd.StringDtype, required=False, description="name of the static generator"),
        "type": pa.Column(str, required=False, description="type of source"),
        "bus_dc": pa.Column(
            int, description="index of connected bus", metadata={"foreign_key": "bus_dc.index"}
        ),  # TODO: not the same name in docu
        "p_mw": pa.Column(
            float,
            pa.Check.le(0),
            description="active power of the static generator [MW]",
        ),
        "scaling": pa.Column(
            float,
            pa.Check.ge(0),
            description="scaling factor for the active and reactive power",
        ),  # TODO: not in create method
        "vm_pu": pa.Column(
            float,
            description="",
        ),  # TODO: missing in docu
        "controllable": pa.Column(
            bool,
            required=False,
            description="states if sgen is controllable or not, sgen will not be used as a flexibility if it is not controllable",
        ),
        "in_service": pa.Column(bool, description="specifies if the generator is in service."),
    },
    strict=False,
)

res_source_dc_schema = pa.DataFrameSchema(
    {"p_dc_mw": pa.Column(float, nullable=True, description="resulting active power demand after scaling [MW]")},
    strict=False,
)
