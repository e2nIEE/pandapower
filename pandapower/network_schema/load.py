import pandas as pd
import pandera.pandas as pa

_load_columns = {
    "name": pa.Column(
        pd.StringDtype, nullable=True, required=False, description="name of the load", metadata={"cim": True}
    ),
    "bus": pa.Column(int, pa.Check.ge(0), description="index of connected bus", metadata={"foreign_key": "bus.index"}),
    "p_mw": pa.Column(float, description="active power of the load [MW], a positiv value means power consumption"),
    "q_mvar": pa.Column(
        float,
        description=" reactive power of the load [MVar], positive for inductive consumers, negative for capacitive consumers",
        metadata={"default": 0.0},
    ),
    "const_z_p_percent": pa.Column(
        float,
        pa.Check.between(min_value=0, max_value=100),
        nullable=False,
        description="percentage of p_mw that is associated to constant impedance load at rated voltage [%]",
        metadata={"zip": True, "default": 0.0},
    ),
    "const_i_p_percent": pa.Column(
        float,
        pa.Check.between(min_value=0, max_value=100),
        nullable=False,
        description="percentage of p_mw that is associated to constant current load at rated voltage [%]",
        metadata={"zip": True, "default": 0.0},
    ),
    "const_z_q_percent": pa.Column(
        float,
        pa.Check.between(min_value=0, max_value=100),
        nullable=False,
        description="percentage of q_mvar that is associated to constant impedance load at rated voltage [%]",
        metadata={"zip": True, "default": 0.0},
    ),
    "const_i_q_percent": pa.Column(
        float,
        pa.Check.between(min_value=0, max_value=100),
        nullable=False,
        description="percentage of q_mvar that is associated to constant current load at rated voltage [%]",
        metadata={"zip": True, "default": 0.0},
    ),
    "sn_mva": pa.Column(
        float,
        pa.Check.gt(0),
        nullable=True,
        required=False,
        description="rated power of the load [kVA]",
        metadata={"cim": True},
    ),
    "scaling": pa.Column(
        float, pa.Check.ge(0), description="scaling factor for active and reactive power", metadata={"default": 1.0}
    ),
    "in_service": pa.Column(bool, description="specifies if the load is in service.", metadata={"default": True}),
    "type": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="Connection Type of 3 Phase Load(Valid for three phase load flow only) Naming convention: wye, delta",
        metadata={"3ph": True, "default": "wye"},
    ),
    "controllable": pa.Column(
        bool,
        required=False,
        description="States if load is controllable or not, load will not be used as a flexibility if it is not controllable",
        metadata={"default": False},
    ),
    "zone": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="can be used to group loads, for example network groups / regions",
    ),
    "max_p_mw": pa.Column(float, nullable=True, required=False, description="Maximum active power"),
    "min_p_mw": pa.Column(float, nullable=True, required=False, description="Minimum active power"),
    "max_q_mvar": pa.Column(float, nullable=True, required=False, description="Maximum reactive power"),
    "min_q_mvar": pa.Column(float, nullable=True, required=False, description="Minimum reactive power"),
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
load_schema = pa.DataFrameSchema(
    _load_columns,
    name="load",
    strict=False,
)

res_load_schema = res_load_3ph_schema = pa.DataFrameSchema(
    {
        "p_mw": pa.Column(
            float,
            nullable=True,
            description="resulting active power demand after scaling and after considering voltage dependence [MW]",
        ),
        "q_mvar": pa.Column(
            float,
            nullable=True,
            description="resulting reactive power demand after scaling and after considering voltage dependence [MVar]",
        ),
    },
    name="res_load",
    strict=False,
)
