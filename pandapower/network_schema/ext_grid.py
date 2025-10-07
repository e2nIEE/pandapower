import pandera.pandas as pa

schema = pa.DataFrameSchema(
    {
        "name": pa.Column(str, description="name of the external grid"),
        "bus": pa.Column(int, pa.Check.ge(0), description="index of connected bus"),
        "vm_pu": pa.Column(float, pa.Check.gt(0), description="voltage set point [p.u]"),
        "va_degree": pa.Column(float, description="angle set point [degree]"),
        "max_p_mw": pa.Column(float, required=False, description="Maximum active power"),
        "min_p_mw": pa.Column(float, required=False, description="Minimum active power"),
        "max_q_mvar": pa.Column(float, required=False, description="Maximum reactive power"),
        "min_q_mvar": pa.Column(float, required=False, description="Minimum reactive power"),
        "s_sc_max_mva": pa.Column(
            float,
            pa.Check.gt(0),
            nullable=True,
            required=False,
            description="maximum short circuit power provision [MVA]",
        ),
        "s_sc_min_mva": pa.Column(
            float,
            pa.Check.gt(0),
            nullable=True,
            required=False,
            description="minimum short circuit power provision [MVA]",
        ),
        "rx_max": pa.Column(
            float,
            pa.Check.between(min_value=0, max_value=1),
            nullable=True,
            required=False,
            description="maxium R/X ratio of short-circuit impedance",
        ),
        "rx_min": pa.Column(
            float,
            pa.Check.between(min_value=0, max_value=1),
            nullable=True,
            required=False,
            description="minimum R/X ratio of short-circuit impedance",
        ),
        "r0x0_max": pa.Column(
            float,
            pa.Check.between(min_value=0, max_value=1),
            required=False,
            description="maximal R/X-ratio to calculate Zero sequence internal impedance of ext_grid",
        ),
        "x0x_max": pa.Column(
            float,
            pa.Check.between(min_value=0, max_value=1),
            required=False,
            description="maximal X0/X-ratio to calculate Zero sequence internal impedance of ext_grid",
        ),
        "slack_weight": pa.Column(float, description=""),  # missing in docu
        "in_service": pa.Column(bool, description="specifies if the external grid is in service."),
        "controllable": pa.Column(bool, required=False, description=""),  # missing in docu
    },
    strict=False,
)

res_schema = pa.DataFrameSchema(
    {
        "p_mw": pa.Column(float, description="active power supply at the external grid [MW]"),
        "q_mvar": pa.Column(float, description="reactive power supply at the external grid [MVar]"),
    },
)
