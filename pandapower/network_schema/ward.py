import pandas as pd
import pandera.pandas as pa

ward_schema = pa.DataFrameSchema(
    {
        "name": pa.Column(
            pd.StringDtype,
            nullable=True,
            required=False,
            description="name of the ward equivalent",
            metadata={"cim": True},
        ),
        "bus": pa.Column(
            int, pa.Check.ge(0), description="index of connected bus", metadata={"foreign_key": "bus.index"}
        ),
        "ps_mw": pa.Column(float, description="constant active power demand [MW]"),
        "qs_mvar": pa.Column(float, description="constant reactive power demand [MVar]"),
        "pz_mw": pa.Column(float, description="constant impedance active power demand at 1.0 pu [MW]"),
        "qz_mvar": pa.Column(float, description="constant impedance reactive power demand at 1.0 pu [MVar]"),
        "in_service": pa.Column(
            bool, description="specifies if the ward equivalent is in service.", metadata={"default": True}
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
    },
    name="ward",
    strict=False,
)

res_ward_schema = pa.DataFrameSchema(
    {
        "p_mw": pa.Column(float, nullable=True, description="active power demand of the ward equivalent [MW]"),
        "q_mvar": pa.Column(float, nullable=True, description="reactive power demand of the ward equivalent [MVar]"),
        "vm_pu": pa.Column(float, nullable=True, description="voltage at the ward bus [p.u]"),
    },
    name="res_ward",
    strict=False,
)
