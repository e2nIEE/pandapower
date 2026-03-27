import pandas as pd
import pandera.pandas as pa

switch_schema = pa.DataFrameSchema(
    {
        "bus": pa.Column(
            int, pa.Check.ge(0), description="index of connected bus", metadata={"foreign_key": "bus.index"}
        ),
        "name": pa.Column(
            pd.StringDtype, nullable=True, required=False, description="name of the switch", metadata={"cim": True}
        ),
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
            pd.StringDtype,
            nullable=True,
            required=False,
            description="type of switch naming conventions:  “CB” - circuit breaker “LS” - load switch “LBS” - load break switch “DS” - disconnecting switch",
            metadata={"cim": True},
        ),
        "closed": pa.Column(bool, description="signals the switching state of the switch", metadata={"default": True}),
        "in_ka": pa.Column(
            float,
            pa.Check.gt(0),
            nullable=True,
            description="maximum current that the switch can carry under normal operating conditions without tripping",
        ),
        "z_ohm": pa.Column(
            float,
            nullable=True,
            description="indicates the resistance of the switch, which has effect only on bus-bus switches, if sets to 0, the buses will be fused like before, if larger than 0 a branch will be created for the switch which has also effects on the bus mapping",
            metadata={"default": 0.0},
        ),
        "origin_id": pa.Column(
            pd.StringDtype, nullable=True, required=False, description="element rdfId from CIM", metadata={"cim": True, "doc": False}
        ),
        "origin_class": pa.Column(
            pd.StringDtype,
            nullable=True,
            required=False,
            description="origin_class rdfId from CIM",
            metadata={"cim": True, "doc": False},
        ),
        "description": pa.Column(
            pd.StringDtype,
            nullable=True,
            required=False,
            description="description from converter, not relevant for calculations",
            metadata={"cim": True, "doc": False},
        ),
        "terminal_bus": pa.Column(
            pd.StringDtype,
            nullable=True,
            required=False,
            description="terminal_to from converter, not relevant for calculations",
            metadata={"cim": True, "doc": False},
        ),
        "terminal_element": pa.Column(
            pd.StringDtype,
            nullable=True,
            required=False,
            description="terminal_from from converter, not relevant for calculations",
            metadata={"cim": True, "doc": False},
        ),
    },
    name="switch",
    strict=False,
)

res_switch_schema = pa.DataFrameSchema(
    {
        "p_from_mw": pa.Column(float, nullable=True, description="active power from bus [MW]"),
        "q_from_mvar": pa.Column(float, nullable=True, description="reactive power from bus [MVAr]"),
        "p_to_mw": pa.Column(float, nullable=True, description="active power to element [MW]"),
        "q_to_mvar": pa.Column(float, nullable=True, description="reactive power to element [MVAr]"),
        "i_ka": pa.Column(float, nullable=True, description="current on switch [kA]"),
        "loading_percent": pa.Column(
            float, nullable=True, description="loading of switch in percent of maximum current [%]"
        ),
    },
    name="res_switch",
    strict=False,
)
