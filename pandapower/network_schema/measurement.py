import pandas as pd
import pandera.pandas as pa

# TODO: to discuss whole concept
# TODO: whats required and whats not ?

measurement_schema = pa.DataFrameSchema(
    {
        "name": pa.Column(pd.StringDtype, description="Name of measurement"),
        "measurement_type": pa.Column(
            str, pa.Check.isin(["p", "q", "i", "v"]), description="Defines what physical quantity is measured"
        ),
        "element_type": pa.Column(
            str,
            pa.Check.isin(
                ["bus", "line", "trafo", "trafo3w", "load", "gen", "sgen", "shunt", "ward", "xward", "ext_grid"]
            ),
            description="Defines which element type is equipped with the measurement",
        ),
        "value": pa.Column(float, description="Measurement value"),
        "std_dev": pa.Column(float, description="Standard deviation (same unit as measurement)"),
        "bus": pa.Column(
            int,
            pa.Check.ge(0),
            required=False,
            description="Defines the bus at which the measurement is placed. For line or transformer measurement, it defines the side at which the measurement is placed (from_bus or to_bus). must be in net.bus.index",
            metadata={"foreign_key": "bus.index"},
        ),
        "element": pa.Column(
            int,
            description="If the element_type is “line”, “trafo”, “trafo3w”, “load”, “gen”, “sgen”, “shunt”, “ward”, “xward” or “ext_grid”, element is the index of the relevant element. For “bus” measurements, it is None (default)",
        ),
        "check_existing": pa.Column(
            bool,
            description="Checks if a measurement of the type already exists and overwrites it. If set to False, the measurement may be added twice (unsafe behaviour), but the performance increases",
        ),  # TODO: shouldn't this be called overwrite?
        "side": pa.Column(
            str,
            description="Only used for measured lines or transformers. Side defines at which end of the branch the measurement is gathered. For lines this may be “from“, “to“ to denote the side with the from_bus or to_bus. It can also be the index of the from_bus or to_bus. For transformers, it can be “hv“, “mv“ or “lv“ or the corresponding bus index, respectively.",
        ),  # TODO: check nur wenn element_type trafo oder line
    },
    strict=False,
)
