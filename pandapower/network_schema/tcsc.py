import pandera.pandas as pa

schema = pa.DataFrameSchema(
    {
        "name": pa.Column(str, description="name of the TCSC"),
        "from_bus": pa.Column(int, pa.Check.ge(0), description="index of the from bus where the TCSC is connected"),
        "to_bus": pa.Column(int, pa.Check.ge(0), description="index of the to bus where the TCSC is connected"),
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
        ),  # how does this correlate to min/max_angle_degree ?
        "controllable": pa.Column(
            bool, description="whether the element is considered as actively controlling or as a fixed series impedance"
        ),
        "in_service": pa.Column(bool, description="specifies if the TCSC is in service."),
        "min_angle_degree": pa.Column(
            float, pa.Check.ge(90), description="minimum value of the thyristor_firing_angle_degree"
        ),  # do values >= 180 make sense?
        "max_angle_degree": pa.Column(
            float, pa.Check.le(180), description="maximum value of the thyristor_firing_angle_degree"
        ),  # do values <= 90 make sense?
    },
    strict=False,
)
