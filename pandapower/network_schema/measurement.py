import pandas as pd
import pandera.pandas as pa

# TODO: to discuss whole concept
# TODO: whats required and whats not ?

_side_allowed = {"line": ["from", "to"], "trafo": ["hv", "lv"], "trafo3w": ["hv", "mv", "lv"]}

def _check_elment_type_side_combination(row: pd.Series) -> bool:
    return row.element_type not in _side_allowed or row.side in _side_allowed[row.element_type]

# TODO: add cross reference (foreign_key) check for element column

measurement_schema = pa.DataFrameSchema(
    {
        "name": pa.Column(pd.StringDtype, nullable=True, description="Name of measurement"),
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
        "element": pa.Column(
            int,
            description="Element is the index of the element in `net[element_type]`.",
        ),
        "side": pa.Column(
            pd.StringDtype,
            pa.Check.isin(["from", "to", "hv", "mv", "lv"]),
            nullable=True,
            description="Used for element_type line, trafo or trafo3w. Defines where the measurement is gathered.",
        ),  # TODO: check nur wenn element_type trafo(3w) oder line
        "origin_id": pa.Column(
            pd.StringDtype,
            nullable=True,
            required=False,
            description="element rdfId from CIM",
            metadata={"cim": True, "doc": False},
        ),
        "origin_class": pa.Column(
            pd.StringDtype,
            nullable=True,
            required=False,
            description="origin_class rdfId from CIM",
            metadata={"cim": True, "doc": False},
        ),
        "source": pa.Column(
            pd.StringDtype,
            nullable=True,
            required=False,
            description="source from converter, not relevant for calculations",
            metadata={"cim": True, "doc": False},
        ),
        "analog_id": pa.Column(
            pd.StringDtype,
            nullable=True,
            required=False,
            description="analog_id from converter, not relevant for calculations",
            metadata={"cim": True, "doc": False},
        ),
        "terminal_id": pa.Column(
            pd.StringDtype,
            nullable=True,
            required=False,
            description="terminal_id from converter, not relevant for calculations",
            metadata={"cim": True, "doc": False},
        ),
        "description": pa.Column(
            pd.StringDtype,
            nullable=True,
            required=False,
            description="description from converter, not relevant for calculations",
            metadata={"cim": True, "doc": False},
        ),
    },
    pa.Check(
        _check_elment_type_side_combination,
        element_wise=True
    ),
    name="measurement",
    strict=False,
)
