import pandera.pandas as pa
import pandas as pd

ext_grid_schema = pa.DataFrameSchema(
    {
        "name": pa.Column(pd.StringDtype, required=False, description="name of the external grid"),
        "bus": pa.Column(
            int, pa.Check.ge(0), description="index of connected bus", metadata={"foreign_key": "bus.index"}
        ),
        "vm_pu": pa.Column(float, pa.Check.gt(0), description="voltage set point [p.u]"),
        "va_degree": pa.Column(float, description="angle set point [degree]"),
        "max_p_mw": pa.Column(float, required=False, description="Maximum active power", metadata={"opf": True}),
        "min_p_mw": pa.Column(float, required=False, description="Minimum active power", metadata={"opf": True}),
        "max_q_mvar": pa.Column(float, required=False, description="Maximum reactive power", metadata={"opf": True}),
        "min_q_mvar": pa.Column(float, required=False, description="Minimum reactive power", metadata={"opf": True}),
        "s_sc_max_mva": pa.Column(
            float,
            pa.Check.gt(0),
            nullable=True,
            required=False,
            description="maximum short circuit power provision [MVA]",
            metadata={"sc": True},
        ),
        "s_sc_min_mva": pa.Column(
            float,
            pa.Check.gt(0),
            nullable=True,
            required=False,
            description="minimum short circuit power provision [MVA]",
            metadata={"sc": True},
        ),
        "rx_max": pa.Column(
            float,
            pa.Check.ge(0),
            nullable=True,
            required=False,
            description="maxium R/X ratio of short-circuit impedance",
            metadata={"sc": True},
        ),
        "rx_min": pa.Column(
            float,
            pa.Check.ge(0),
            nullable=True,
            required=False,
            description="minimum R/X ratio of short-circuit impedance",
            metadata={"sc": True},
        ),
        "r0x0_max": pa.Column(
            float,
            pa.Check.ge(0),
            required=False,
            description="maximal R/X-ratio to calculate Zero sequence internal impedance of ext_grid",
            metadata={"sc": True, "3ph": True},
        ),
        "x0x_max": pa.Column(
            float,
            pa.Check.ge(0),
            required=False,
            description="maximal X0/X-ratio to calculate Zero sequence internal impedance of ext_grid",
            metadata={"sc": True, "3ph": True},
        ),
        "slack_weight": pa.Column(float, description=""),  # TODO: missing in docu
        "in_service": pa.Column(bool, description="specifies if the external grid is in service."),
        "controllable": pa.Column(bool, description=""),  # TODO: missing in docu
    },
    strict=False,
)

res_ext_grid_schema = pa.DataFrameSchema(
    {
        "p_mw": pa.Column(float, nullable=True, description="active power supply at the external grid [MW]"),
        "q_mvar": pa.Column(float, nullable=True, description="reactive power supply at the external grid [MVar]"),
    },
    strict=False,
)

res_ext_grid_3ph_schema = pa.DataFrameSchema(
    {
        "p_a_mw": pa.Column(
            float, nullable=True, description="active power supply at the external grid : Phase A [MW]"
        ),
        "q_a_mvar": pa.Column(
            float, nullable=True, description="reactive power supply at the external grid : Phase A [MVar]"
        ),
        "p_b_mw": pa.Column(
            float, nullable=True, description="active power supply at the external grid : Phase B [MW]"
        ),
        "q_b_mvar": pa.Column(
            float, nullable=True, description="reactive power supply at the external grid : Phase B [MVar]"
        ),
        "p_c_mw": pa.Column(
            float, nullable=True, description="active power supply at the external grid : Phase C [MW]"
        ),
        "q_c_mvar": pa.Column(
            float, nullable=True, description="reactive power supply at the external grid : Phase C [MVar]"
        ),
    },
    strict=False,
)
