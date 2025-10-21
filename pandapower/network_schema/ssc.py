import pandera.pandas as pa
import pandas as pd

ssc_schema = pa.DataFrameSchema(
    {
        "name": pa.Column(pd.StringDtype, required=False, description="name of the SSC"),
        "bus": pa.Column(
            int,
            pa.Check.ge(0),
            description="index of bus where the SSC is connected",
            metadata={"foreign_key": "bus.index"},
        ),
        "r_ohm": pa.Column(
            float, pa.Check.ge(0), description="resistance of the coupling transformer component of SSC"
        ),
        "x_ohm": pa.Column(float, pa.Check.le(0), description="reactance of the coupling transformer component of SSC"),
        "set_vm_pu": pa.Column(float, description="set-point for the bus voltage magnitude at the connection bus"),
        "vm_internal_pu": pa.Column(
            float, description="The voltage magnitude of the voltage source converter VSC at the SSC component."
        ),
        "va_internal_degree": pa.Column(
            float, description="The voltage angle of the voltage source converter VSC at the SSC component."
        ),
        "controllable": pa.Column(
            bool, description="whether the element is considered as actively controlling or as a fixed shunt impedance"
        ),
        "in_service": pa.Column(bool, description="specifies if the SSC is in service."),
    },
    strict=False,
)


res_ssc_schema = pa.DataFrameSchema(
    {
        "q_mvar": pa.Column(float, nullable=True, description="shunt reactive power consumption of ssc [MVAr]"),
        "vm_internal_pu": pa.Column(float, nullable=True, description="voltage magnitude at ssc internal bus [pu]"),
        "va_internal_degree": pa.Column(float, nullable=True, description="voltage angle at ssc internal bus [degree]"),
        "vm_pu": pa.Column(float, nullable=True, description="voltage magnitude at ssc bus [pu]"),
        "va_degree": pa.Column(float, nullable=True, description="voltage angle at ssc bus [degree]"),
    },
    strict=False,
)
