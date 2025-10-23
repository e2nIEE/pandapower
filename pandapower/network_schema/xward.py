import pandas as pd
import pandera.pandas as pa

xward_schema = pa.DataFrameSchema(
    {
        "name": pa.Column(
            pd.StringDtype, nullable=True, required=False, description="name of the extended ward equivalent"
        ),
        "bus": pa.Column(
            int, pa.Check.ge(0), description="index of connected bus", metadata={"foreign_key": "bus.index"}
        ),
        "ps_mw": pa.Column(float, description="constant active power demand [MW]"),
        "qs_mvar": pa.Column(float, description="constant reactive power demand [MVar]"),
        "pz_mw": pa.Column(float, description="constant impedance active power demand at 1.0 pu [MW]"),
        "qz_mvar": pa.Column(float, description="constant impedance reactive power demand at 1.0 pu [MVar]"),
        "r_ohm": pa.Column(float, pa.Check.gt(0), description="internal resistance of the voltage source [ohm]"),
        "x_ohm": pa.Column(float, pa.Check.gt(0), description="internal reactance of the voltage source [ohm]"),
        "vm_pu": pa.Column(float, pa.Check.gt(0), description="voltage source set point [p.u]"),
        "slack_weight": pa.Column(float, description=""),  # TODO: missing in docu
        "in_service": pa.Column(bool, description="specifies if the extended ward equivalent is in service."),
    },
    strict=False,
)

res_xward_schema = pa.DataFrameSchema(
    {
        "p_mw": pa.Column(float, nullable=True, description="active power demand of the ward equivalent [MW]"),
        "q_mvar": pa.Column(float, nullable=True, description="reactive power demand of the ward equivalent [MVar]"),
        "vm_pu": pa.Column(float, nullable=True, description="voltage at the ward bus [p.u]"),
        "va_internal_degree": pa.Column(float, nullable=True, description=""),  # TODO: missing in docu
        "vm_internal_pu": pa.Column(float, nullable=True, description=""),  # TODO: missing in docu
    },
    strict=False,
)
