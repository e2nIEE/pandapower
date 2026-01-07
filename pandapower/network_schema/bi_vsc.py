import pandas as pd
import pandera.pandas as pa

# TODO: where is the documentation?

bi_vsc_schema = pa.DataFrameSchema(
    {
        "name": pa.Column(pd.StringDtype, nullable=True, required=False, description=""),
        "bus": pa.Column(int, pa.Check.ge(0), description="", metadata={"foreign_key": "bus.index"}),
        "bus_dc_plus": pa.Column(int, pa.Check.ge(0), description="", metadata={"foreign_key": "bus_dc.index"}),
        "bus_dc_minus": pa.Column(int, pa.Check.ge(0), description="", metadata={"foreign_key": "bus_dc.index"}),
        "r_ohm": pa.Column(float, description="resistance of the coupling transformer component of VSC"),
        "x_ohm": pa.Column(float, description="reactance of the coupling transformer component of VSC"),
        "r_dc_ohm": pa.Column(float, description="resistance of the internal dc resistance component of VSC"),
        "pl_dc_mw": pa.Column(
            float,
            description="no-load losses of the VSC on the DC side for the shunt R representing the no load losses",
        ),
        "control_mode_ac": pa.Column(
            str, description="the control mode of the ac side of the VSC. it could be vm_pu, q_mvar or slack"
        ),
        "control_value_ac": pa.Column(
            float, description="the value of the controlled parameter at the ac bus in p.u. or MVAr"
        ),
        "control_mode_dc": pa.Column(
            str, description="the control mode of the dc side of the VSC. it could be vm_pu or p_mw"
        ),
        "control_value_dc": pa.Column(
            float, description="the value of the controlled parameter at the dc bus in p.u. or MW"
        ),
        "controllable": pa.Column(
            bool,
            description="whether the element is considered as actively controlling or as a fixed voltage source connected via shunt impedance",
        ),
        "in_service": pa.Column(bool, description="True for in_service or False for out of service"),
    },
    strict=False,
)

res_bi_vsc_schema = pa.DataFrameSchema(
    {
        "p_mw": pa.Column(float, nullable=True, description=""),
        "q_mvar": pa.Column(float, nullable=True, description=""),
        "p_dc_mw_p": pa.Column(float, nullable=True, description=""),
        "p_dc_mw_m": pa.Column(float, nullable=True, description=""),
        "vm_internal_pu": pa.Column(float, nullable=True, description=""),
        "vm_internal_degree": pa.Column(float, nullable=True, description=""),
        "vm_pu": pa.Column(float, nullable=True, description=""),
        "va_degree": pa.Column(float, nullable=True, description=""),
        "vm_internal_dc_pu_p": pa.Column(float, nullable=True, description=""),
        "vm_internal_dc_pu_m": pa.Column(float, nullable=True, description=""),
        "vm_dc_pu_p": pa.Column(float, nullable=True, description=""),
        "vm_dc_pu_m": pa.Column(float, nullable=True, description=""),
    },
    strict=False,
)
