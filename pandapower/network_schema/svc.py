import pandera.pandas as pa

schema = pa.DataFrameSchema(
    {
        "name": pa.Column(str, description="name of the SVC"),
        "bus": pa.Column(int, description="index of bus where the SVC is connected"),
        "x_l_ohm": pa.Column(float, pa.Check.ge(0), description="impedance of the reactor component of SVC"),
        "x_cvar_ohm": pa.Column(float, pa.Check.se(0), description="impedance of the fixed capacitor component of SVC"),
        "set_vm_pu": pa.Column(float, description="set-point for the bus voltage magnitude at the connection bus"),
        "thyristor_firing_angle_degree": pa.Column(float, pa.Check.between(min_value=90, max_value=180) ,description="the value of thyristor firing angle of SVC"),
        "controllable": pa.Column(bool, description="whether the element is considered as actively controlling or as a fixed shunt impedance"),
        "in_service": pa.Column(bool, description="specifies if the SVC is in service."),
        "min_angle_degree": pa.Column(float, pa.Check.ge(90), description="minimum value of the thyristor_firing_angle_degree"),
        "max_angle_degree": pa.Column(float, pa.Check.se(180), description="maximum value of the thyristor_firing_angle_degree"),
    },
    strict=False,
)