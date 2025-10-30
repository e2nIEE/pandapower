import pandas as pd
import pandera.pandas as pa

from network_schema.tools import create_column_dependency_checks_from_metadata

_motor_columns = {
    "name": pa.Column(pd.StringDtype, nullable=True, required=False, description="name of the motor"),
    "bus": pa.Column(int, pa.Check.ge(0), description="index of connected bus", metadata={"foreign_key": "bus.index"}),
    "pn_mech_mw": pa.Column(float, pa.Check.ge(0), description="Mechanical rated power of the motor [MW]"),
    "cos_phi": pa.Column(
        float, pa.Check.between(min_value=0, max_value=1), description="cosine phi at current operating point"
    ),
    "cos_phi_n": pa.Column(
        float,
        pa.Check.between(min_value=0, max_value=1),
        description="cosine phi at rated power of the motor for short-circuit calculation",
        metadata={"sc": True},
    ),
    "efficiency_percent": pa.Column(
        float,
        pa.Check.between(min_value=0, max_value=100),
        description="Efficiency in percent at current operating point[%]",
    ),
    "efficiency_n_percent": pa.Column(
        float,
        pa.Check.between(min_value=0, max_value=100),
        description="Efficiency in percent at rated power for short-circuit calculation [%]",
        metadata={"sc": True},
    ),
    "loading_percent": pa.Column(
        float,
        pa.Check.between(min_value=0, max_value=100),
        description="The mechanical loading in percentage of the rated mechanical power",
    ),
    "scaling": pa.Column(float, pa.Check.ge(0), description="scaling factor for active and reactive power"),
    "lrc_pu": pa.Column(
        float,
        pa.Check.ge(0),
        description="locked rotor current in relation to the rated motor current [pu]",
        metadata={"sc": True},
    ),
    "rx": pa.Column(
        float,
        pa.Check.ge(0),
        description="R/X ratio of the motor for short-circuit calculation.",
        metadata={"sc": True},
    ),
    "vn_kv": pa.Column(
        float,
        pa.Check.ge(0),
        description="Rated voltage of the motor for short-circuit calculation",
        metadata={"sc": True},
    ),
    "in_service": pa.Column(bool, description="specifies if the motor is in service."),
}
motor_schema = pa.DataFrameSchema(
    _motor_columns,
    checks=create_column_dependency_checks_from_metadata(["sc"], _motor_columns),
    strict=False,
)


res_motor_schema = pa.DataFrameSchema(
    {
        "p_mw": pa.Column(float, nullable=True, description="resulting active power demand [MW]"),
        "q_mvar": pa.Column(float, nullable=True, description="resulting reactive power demand [MVar]"),
    },
    strict=False,
)
