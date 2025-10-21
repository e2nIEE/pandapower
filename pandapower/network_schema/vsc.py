import pandera.pandas as pa
import pandas as pd

vsc_schema = pa.DataFrameSchema(
    {
        "name": pa.Column(pd.StringDtype, required=False, description="name of the VSC"),
        "bus": pa.Column(
            int,
            pa.Check.ge(0),
            description="index of ac bus of the ac side of the VSC",
            metadata={"foreign_key": "bus.index"},
        ),
        "bus_dc": pa.Column(
            int,
            pa.Check.ge(0),
            description="index of dc bus of the dc side of the VSC",
            metadata={"foreign_key": "bus_dc.index"},
        ),
        "r_ohm": pa.Column(float, pa.Check.ge(0), description="resistance of the coupling transformer"),
        "x_ohm": pa.Column(float, pa.Check.ge(0), description="reactance of the coupling transformer"),
        "r_dc_ohm": pa.Column(float, description="resistance of the internal dc resistance component of VSC"),
        "pl_dc_mw": pa.Column(
            float,
            required=False,
            description="no-load losses of the VSC on the DC side for the shunt R representing the no load losses",
        ),
        "control_mode_ac": pa.Column(
            str, pa.Check.isin(["vm_pu", "q_mvar", "slack"]), description="the control mode of the AC side of the VSC"
        ),
        "control_value_ac": pa.Column(float, description="the value of the controlled parameter at the ac bus"),
        "control_mode_dc": pa.Column(
            str,
            pa.Check.isin(
                [
                    "vm_pu",
                    "p_mw",
                ]
            ),
            description="“the control mode of the dc side of the VSC",
        ),
        "control_value_dc": pa.Column(float, description="the value of the controlled parameter at the dc bus"),
        "controllable": pa.Column(bool, description="whether the element is considered as actively controlling"),
        "in_service": pa.Column(bool, description="specifies if the VSC is in service."),
        "ref_bus": pa.Column(int, pa.Check.ge(0), description=""),  # TODO: missing in docu
    },
    strict=False,
)


res_vsc_schema = pa.DataFrameSchema(
    {
        "p_mw": pa.Column(float, description="active power consumption of vsc [MW]"),
        "q_mvar": pa.Column(float, description="reactive power consumption of vsc [MVAr]"),
        "p_dc_mw": pa.Column(float, description=""),  # TODO: missing in docu
        "vm_internal_pu": pa.Column(float, description="voltage magnitude at vsc internal bus [pu]"),
        "va_internal_degree": pa.Column(float, description="voltage angle at vsc internal bus [degree]"),
        "vm_pu": pa.Column(float, description="voltage magnitude at vsc ac bus [pu]"),
        "va_degree": pa.Column(float, description="voltage angle at vsc ac bus [degree]"),
        "vm_internal_dc_pu": pa.Column(float, description=""),  # TODO: missing in docu
        "vm_dc_pu": pa.Column(float, description=""),  # TODO: missing in docu
    },
)
