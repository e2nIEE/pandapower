import pandera.pandas as pa
import pandas as pd

tcsc_schema = pa.DataFrameSchema(
    {
        "name": pa.Column(pd.StringDtype, nullable=True, required=False, description="name of the TCSC"),
        "from_bus": pa.Column(
            int,
            pa.Check.ge(0),
            description="index of the from bus where the TCSC is connected",
            metadata={"foreign_key": "bus.index"},
        ),
        "to_bus": pa.Column(
            int,
            pa.Check.ge(0),
            description="index of the to bus where the TCSC is connected",
            metadata={"foreign_key": "bus.index"},
        ),
        "x_l_ohm": pa.Column(float, pa.Check.ge(0), description="impedance of the reactor component of TCSC"),
        "x_cvar_ohm": pa.Column(
            float, pa.Check.le(0), description="impedance of the fixed capacitor component of TCSC"
        ),
        "set_p_to_mw": pa.Column(
            float, description="set-point for the power flowing through the TCSC element at the to bus"
        ),
        "thyristor_firing_angle_degree": pa.Column(
            float,
            pa.Check.between(min_value=90, max_value=180),
            description="the value of thyristor firing angle of TCSC",
        ),  # TODO: how does this correlate to min/max_angle_degree ?
        "controllable": pa.Column(
            bool, description="whether the element is considered as actively controlling or as a fixed series impedance"
        ),
        "in_service": pa.Column(bool, description="specifies if the TCSC is in service."),
        "min_angle_degree": pa.Column(
            float,
            pa.Check.ge(90),
            nullable=True,
            required=False,
            description="minimum value of the thyristor_firing_angle_degree",
        ),  # TODO: do values >= 180 make sense?
        "max_angle_degree": pa.Column(
            float,
            pa.Check.le(180),
            nullable=True,
            required=False,
            description="maximum value of the thyristor_firing_angle_degree",
        ),  # TODO:do values <= 90 make sense?
    },
    checks=[
        pa.Check(
            lambda df: (
                df["min_angle_degree"] <= df["max_angle_degree"]
                if all(col in df.columns for col in ["min_angle_degree", "max_angle_degree"])
                else True
            ),
            error="Column 'min_angle_degree' must be <= column 'max_angle_degree'",
        )  # TODO: makes sense, right ?
    ],
    strict=False,
)


res_tcsc_schema = pa.DataFrameSchema(
    {
        "thyristor_firing_angle_degree": pa.Column(
            float, nullable=True, description="the resulting value of thyristor firing angle of tcsc [degree]"
        ),
        "x_ohm": pa.Column(float, nullable=True, description="resulting value of the shunt impedance of tcsc [Ohm]"),
        "p_from_mw": pa.Column(float, nullable=True, description="active power consumed at the from bus of tcsc [MW]"),
        "q_from_mvar": pa.Column(
            float, nullable=True, description="reactive power consumed at the from bus of tcsc [MVAr]"
        ),
        "p_to_mw": pa.Column(float, nullable=True, description="active power consumed at the to bus of tcsc [MW]"),
        "q_to_mvar": pa.Column(
            float, nullable=True, description="reactive power consumed at the to bus of tcsc [MVAr]"
        ),
        "pl_mw": pa.Column(float, nullable=True, description="active power losses of tcsc [MW]"),
        "ql_mvar": pa.Column(float, nullable=True, description="reactive power losses of tcsc [MVAr]"),
        "i_ka": pa.Column(float, nullable=True, description=""),  # TODO: docu missing
        "vm_from_pu": pa.Column(float, nullable=True, description="voltage magnitude at the from bus [pu]"),
        "va_from_degree": pa.Column(float, nullable=True, description="voltage angle at the from bus [degree]"),
        "vm_to_pu": pa.Column(float, nullable=True, description="voltage magnitude at the to bus [pu]"),
        "va_to_degree": pa.Column(float, nullable=True, description="voltage angle at the to bus [degree]"),
    },
    strict=False,
)
