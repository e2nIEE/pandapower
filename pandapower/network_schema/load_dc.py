import pandera.pandas as pa
import pandas as pd

load_dc_schema = pa.DataFrameSchema(
    {
        "name": pa.Column(pd.StringDtype, nullable=True, required=False, description="name of the load"),
        "bus_dc": pa.Column(
            int, pa.Check.ge(0), description="index of connected bus", metadata={"foreign_key": "bus_dc.index"}
        ),
        "p_dc_mw": pa.Column(
            float, pa.Check.ge(0), description="active power of the load [MW]"
        ),  # TODO: surely the docu must be wrong for ge0
        "scaling": pa.Column(float, pa.Check.ge(0), description="scaling factor for active and reactive power"),
        "in_service": pa.Column(bool, description="specifies if the load is in service."),
        "type": pa.Column(str, description="A string describing the type."),
        "controllable": pa.Column(
            bool,
            nullable=True,
            required=False,
            description="States if load is controllable or not, load will not be used as a flexibilty if it is not controllable",
        ),
    },
    strict=False,
)

res_load_dc_schema = pa.DataFrameSchema(
    {
        "p_dc_mw": pa.Column(
            float,
            nullable=True,
            description="resulting active power demand after scaling and after considering voltage dependence [MW]",
        )
    },
    strict=False,
)
