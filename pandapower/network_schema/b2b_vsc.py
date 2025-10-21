import pandera.pandas as pa
import pandas as pd

b2b_vsc_schema = pa.DataFrameSchema(
    {
        "name": pa.Column(pd.StringDtype, nullable=True, required=False, description="name of the B2B VSC"),
        "bus": pa.Column(
            int,
            pa.Check.ge(0),
            description="index of ac bus of the ac side of the B2B VSC",
            metadata={"foreign_key": "bus.index"},
        ),
        "bus_dc_plus": pa.Column(
            int,
            pa.Check.ge(0),
            description="index of dc bus of the plus dc side of the B2B VSC",
            metadata={"foreign_key": "bus_dc.index"},
        ),
        "bus_dc_minus": pa.Column(
            int,
            pa.Check.ge(0),
            description="index of dc bus of the minus dc side of the B2B VSC",
            metadata={"foreign_key": "bus_dc.index"},
        ),
        "r_ohm": pa.Column(float, description="resistance of the coupling transformer"),
        "x_ohm": pa.Column(float, description="reactance of the coupling transformer"),
        "r_dc_ohm": pa.Column(float, description="resistance of the internal dc resistance component of VSC"),
        "pl_dc_mw": pa.Column(
            float,
            nullable=True,
            required=False,
            description="no-load losses of the VSC on the DC side for the shunt R representing the no load losses",
        ),
        "control_mode_ac": pa.Column(
            str,
            pa.Check.isin(["vm_pu", "q_mvar", "slack"]),
            description="the control mode of the AC side of the B2B VSC",
        ),
        "control_value_ac": pa.Column(float, description="the value of the controlled parameter at the ac bus"),
        "control_mode_dc": pa.Column(
            str, pa.Check.isin(["vm_pu", "p_mw"]), description="the control mode of the dc side of the B2B VSC"
        ),
        "control_value_dc": pa.Column(float, description="the value of the controlled parameter at the dc bus"),
        "controllable": pa.Column(bool, description="whether the element is considered as actively controlling"),
        "in_service": pa.Column(bool, description="specifies if the B2B VSC is in service."),
    },
    strict=False,
)

res_b2b_vsc_schema = pa.DataFrameSchema(
    {
        "p_mw": pa.Column(float, nullable=True, description="total active power consumption of B2B VSC [MW]"),
        "q_mvar": pa.Column(float, nullable=True, description="total reactive power consumption of B2B VSC [MVAr]"),
        "p_dc_mw_p": pa.Column(float, nullable=True, description="voltage magnitude at vsc internal bus [pu]"),
        "p_dc_mw_m": pa.Column(float, nullable=True, description="voltage angle at vsc internal bus [degree]"),
        "vm_internal_pu": pa.Column(float, nullable=True, description="voltage magnitude at B2B VSC ac bus [pu]"),
        "vm_internal_degree": pa.Column(float, nullable=True, description="voltage angle at B2B VSC ac bus [degree]"),
        "vm_pu": pa.Column(float, nullable=True, description="active power of the plus side of the B2B VSC [MW]"),
        "va_degree": pa.Column(float, nullable=True, description="active power of the minus side of B2B VSC [MW]"),
        "vm_internal_dc_pu_p": pa.Column(
            float, nullable=True, description="voltage angle at the plus B2B VSC ac bus [pu]"
        ),
        "vm_internal_dc_pu_m": pa.Column(
            float, nullable=True, description="voltage angle at the minus B2B VSC ac bus [pu]"
        ),
        "vm_dc_pu_p": pa.Column(float, nullable=True, description="voltage magnitude at the plus B2B VSC ac bus [pu]"),
        "vm_dc_pu_m": pa.Column(float, nullable=True, description="voltage magnitude at the minus B2B VSC ac bus [pu]"),
    },
)
