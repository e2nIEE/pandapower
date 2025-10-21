import pandera.pandas as pa

# TODO: unklar was required und was nicht

measurement_schema = pa.DataFrameSchema(
    {
        # "index": pa.Column(int, description="Defines a specific index for the new measurement (if possible)"),  # TODO:
        "name": pa.Column(str, description=""),  # TODO: missing in docu
        "measurement_type": pa.Column(
            str, pa.Check.isin(["p", "q", "i", "v"]), description="Defines what physical quantity is measured"
        ),  # TODO:  other name in docu
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
        "element": pa.Column(int, description="specifies if the bus is in service."),
        "check_existing": pa.Column(
            bool,
            description="Checks if a measurement of the type already exists and overwrites it. If set to False, the measurement may be added twice (unsafe behaviour), but the performance increases",
        ),  # TODO: only in docu
        "side": pa.Column(str, description=""),  # TODO: missing in docu
    },
    strict=False,
)
