import pandera.pandas as pa
import pandas as pd

svc_schema = pa.DataFrameSchema(
    {
        "name": pa.Column(pd.StringDtype, required=False, description="name of the SVC"),
        "bus": pa.Column(
            int,
            pa.Check.ge(0),
            description="index of bus where the SVC is connected",
            metadata={"foreign_key": "bus.index"},
        ),
        "x_l_ohm": pa.Column(float, pa.Check.ge(0), description="impedance of the reactor component of SVC"),
        "x_cvar_ohm": pa.Column(float, pa.Check.le(0), description="impedance of the fixed capacitor component of SVC"),
        "set_vm_pu": pa.Column(float, description="set-point for the bus voltage magnitude at the connection bus"),
        "thyristor_firing_angle_degree": pa.Column(
            float,
            pa.Check.between(min_value=90, max_value=180),
            description="the value of thyristor firing angle of SVC",
        ),
        "controllable": pa.Column(
            bool, description="whether the element is considered as actively controlling or as a fixed shunt impedance"
        ),
        "in_service": pa.Column(bool, description="specifies if the SVC is in service."),
        "min_angle_degree": pa.Column(
            float, pa.Check.ge(90), required=False, description="minimum value of the thyristor_firing_angle_degree"
        ),  # TODO: do values >= 180 make sense?
        "max_angle_degree": pa.Column(
            float, pa.Check.le(180), required=False, description="maximum value of the thyristor_firing_angle_degree"
        ),  # TODO: do values <= 90 make sense?
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


res_svc_schema = pa.DataFrameSchema(
    {
        "thyristor_firing_angle_degree": pa.Column(
            float, description="the resulting value of thyristor firing angle of svc [degree]"
        ),
        "x_ohm": pa.Column(float, description="resulting value of the shunt impedance of svc [Ohm]"),
        "q_mvar": pa.Column(float, description="shunt reactive power consumption of svc [MVAr]"),
        "vm_pu": pa.Column(float, description="voltage magnitude at svc bus [pu]"),
        "va_degree": pa.Column(float, description="voltage angle at svc bus [degree]"),
    },
)
