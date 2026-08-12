import pandas as pd
import pandera.pandas as pa

_motor_columns = {
    "name": pa.Column(
        pd.StringDtype, nullable=True, required=False, description="name of the motor", metadata={"cim": True}
    ),
    "bus": pa.Column(int, pa.Check.ge(0), description="index of connected bus", metadata={"foreign_key": "bus.index"}),
    "pn_mech_mw": pa.Column(float, pa.Check.ge(0), description="Mechanical rated power of the motor [MW]"),
    "cos_phi": pa.Column(
        float, pa.Check.between(min_value=0, max_value=1), description="cosine phi at current operating point"
    ),
    "cos_phi_n": pa.Column(
        float,
        pa.Check.between(min_value=0, max_value=1),
        nullable=True,
        description="cosine phi at rated power of the motor for short-circuit calculation",
        metadata={"sc": True},
    ),
    "efficiency_percent": pa.Column(
        float,
        pa.Check.between(min_value=0, max_value=100),
        description="Efficiency in percent at current operating point[%]",
        metadata={"default": 100.0},
    ),
    "efficiency_n_percent": pa.Column(
        float,
        pa.Check.between(min_value=0, max_value=100),
        nullable=True,
        description="Efficiency in percent at rated power for short-circuit calculation [%]",
        metadata={"sc": True},
    ),
    "loading_percent": pa.Column(
        float,
        pa.Check.between(min_value=0, max_value=100),
        description="The mechanical loading in percentage of the rated mechanical power",
        metadata={"default": 100.0},
    ),
    "scaling": pa.Column(
        float, pa.Check.ge(0), description="scaling factor for active and reactive power", metadata={"default": 1.0}
    ),
    "lrc_pu": pa.Column(
        float,
        pa.Check.ge(0),
        nullable=True,
        description="locked rotor current in relation to the rated motor current [pu]",
        metadata={"sc": True},
    ),
    "rx": pa.Column(
        float,
        pa.Check.ge(0),
        nullable=True,
        description="R/X ratio of the motor for short-circuit calculation.",
        metadata={"sc": True},
    ),
    "vn_kv": pa.Column(
        float,
        pa.Check.ge(0),
        nullable=True,
        description="Rated voltage of the motor for short-circuit calculation",
        metadata={"sc": True},
    ),
    "in_service": pa.Column(bool, description="specifies if the motor is in service.", metadata={"default": True}),
    "origin_id": pa.Column(
        pd.StringDtype, nullable=True, required=False, description="element rdfId from CIM", metadata={"cim": True, "doc": False}
    ),
    "origin_class": pa.Column(
        pd.StringDtype, nullable=True, required=False, description="origin_class rdfId from CIM", metadata={"cim": True, "doc": False}
    ),
    "terminal": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="terminal from converter, not relevant for calculations",
        metadata={"cim": True, "doc": False},
    ),
    "description": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="description from converter, not relevant for calculations",
        metadata={"cim": True, "doc": False},
    ),
}
motor_schema = pa.DataFrameSchema(
    _motor_columns,
    name="motor",
    strict=False,
)


res_motor_schema = pa.DataFrameSchema(
    {
        "p_mw": pa.Column(float, nullable=True, description="resulting active power demand [MW]"),
        "q_mvar": pa.Column(float, nullable=True, description="resulting reactive power demand [MVar]"),
    },
    name="res_motor",
    strict=False,
)
