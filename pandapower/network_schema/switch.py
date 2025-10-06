import pandera.pandas as pa

schema = pa.DataFrameSchema(
    {
        "bus": pa.Column(int, pa.Check.ge(0), description="index of connected bus"),
        "name": pa.Column(str, nullable=True, description="name of the switch"),
        "element": pa.Column(
            int,
            pa.Check.ge(0),
            description="index of the element the switch is connected to: - bus index if et = “b”, - line index if et = “l”, - trafo index if et = “t”",
        ),
        "et": pa.Column(
            str,
            pa.Check.isin(["b", "l", "t", "t3"]),
            description="element type the switch connects to: “b” - bus-bus switch “l” - bus-line switch “t” - bus-trafo “t3” - bus-trafo3w switch",
        ),
        "type": pa.Column(
            str,
            pa.Check.isin(["CB", "LS", "LBS", "DS"]),
            description="type of switch naming conventions:  “CB” - circuit breaker “LS” - load switch “LBS” - load break switch “DS” - disconnecting switch",
        ),
        "closed": pa.Column(bool, description="signals the switching state of the switch"),
        "in_ka": pa.Column(
            float,
            pa.Check.gt(0),
            nullable=True,
            description="maximum current that the switch can carry under normal operating conditions without tripping",
        ),
        "z_ohm": pa.Column(float, description=""),  # missing in docu
    },
    strict=False,
)
