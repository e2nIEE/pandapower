import pandera.pandas as pa
import pandas as pd

switch_schema = pa.DataFrameSchema(
    {
        "bus": pa.Column(
            int, pa.Check.ge(0), description="index of connected bus", metadata={"foreign_key": "bus.index"}
        ),
        "name": pa.Column(pd.StringDtype, nullable=True, required=False, description="name of the switch"),
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
            required=False,
            description="type of switch naming conventions:  “CB” - circuit breaker “LS” - load switch “LBS” - load break switch “DS” - disconnecting switch",
        ),
        "closed": pa.Column(bool, description="signals the switching state of the switch"),
        "in_ka": pa.Column(
            float,
            pa.Check.gt(0),
            nullable=True,
            description="maximum current that the switch can carry under normal operating conditions without tripping",
        ),
        "z_ohm": pa.Column(float, description=""),  # TODO: missing in docu
    },
    strict=False,
)

res_switch_schema = pa.DataFrameSchema(
    {
        "i_ka": pa.Column(float, nullable=True, description="active power from bus [MW]"),
        "loading_percent": pa.Column(float, nullable=True, description="reactive power from bus [MVAr]"),
        "p_from_mw": pa.Column(float, nullable=True, description="active power to element [MW]"),
        "q_from_mvar": pa.Column(float, nullable=True, description="reactive power to element [MVAr]"),
        "p_to_mw": pa.Column(float, nullable=True, description="current on switch [kA]"),
        "q_to_mvar": pa.Column(float, nullable=True, description="loading of switch in percent of maximum current [%]"),
    },
    strict=False,
)
