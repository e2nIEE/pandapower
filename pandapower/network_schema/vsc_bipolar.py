import pandas as pd
import pandera.pandas as pa

vsc_bipolar_schema = pa.DataFrameSchema(
    {
        "name": pa.Column(pd.StringDtype, nullable=True, required=False, description="name of the bipolar VSC"),
        "bus": pa.Column(
            int, pa.Check.ge(0), description="AC connection bus of the VSC", metadata={"foreign_key": "bus.index"}
        ),
        "bus_dc_plus": pa.Column(
            int,
            pa.Check.ge(0),
            description="DC bus of the plus terminal of the VSC",
            metadata={"foreign_key": "bus_dc.index"},
        ),
        "bus_dc_minus": pa.Column(
            int,
            pa.Check.ge(0),
            description="DC bus of the minus terminal of the VSC (e.g. the neutral / DMR bus for the positive pole)",
            metadata={"foreign_key": "bus_dc.index"},
        ),
        "r_ohm": pa.Column(float, description="resistance of the coupling transformer component of VSC"),
        "x_ohm": pa.Column(float, description="reactance of the coupling transformer component of VSC"),
        "r_dc_ohm": pa.Column(float, pa.Check.gt(0), description="resistance of the internal dc resistance component of VSC"),
        "pl_dc_mw": pa.Column(
            float,
            description="no-load losses of the VSC on the DC side, modelled as conductance between the DC terminals",
        ),
        "control_mode_ac": pa.Column(
            str,
            pa.Check.isin(["vm_pu", "q_mvar", "slack"]),
            description="the control mode of the ac side of the VSC. it could be 'vm_pu', 'q_mvar' or 'slack'",
        ),
        "control_value_ac": pa.Column(
            float, description="the value of the controlled parameter at the ac bus in 'p.u.' or 'MVAr'"
        ),
        "control_mode_dc": pa.Column(
            str,
            pa.Check.isin(["vm_pu", "p_mw"]),
            description="the control mode of the dc side of the VSC. 'vm_pu' controls the voltage between plus and "
            "minus terminal, 'p_mw' the DC power flowing into the VSC",
        ),
        "control_value_dc": pa.Column(
            float,
            description="the value of the controlled parameter at the dc side: the voltage difference between plus and "
            "minus terminal in 'p.u.' or the DC power in 'MW'",
        ),
        "controllable": pa.Column(
            bool,
            description="whether the element is considered as actively controlling or as a fixed voltage source connected via shunt impedance",
        ),
        "in_service": pa.Column(bool, description="True for in_service or False for out of service"),
    },
    strict=False,
)

res_vsc_bipolar_schema = pa.DataFrameSchema(
    {
        "p_mw": pa.Column(float, nullable=True, description="active power at the AC bus [MW]"),
        "q_mvar": pa.Column(float, nullable=True, description="reactive power at the AC bus [MVAr]"),
        "p_dc_mw": pa.Column(
            float, nullable=True, description="DC power flowing from the DC grid into the VSC (p_dc_mw_p + p_dc_mw_m) [MW]"
        ),
        "p_dc_mw_p": pa.Column(float, nullable=True, description="DC power at the plus terminal bus [MW]"),
        "p_dc_mw_m": pa.Column(float, nullable=True, description="DC power at the minus terminal bus [MW]"),
        "i_dc_ka": pa.Column(
            float, nullable=True, description="DC current flowing from the plus terminal bus into the VSC [kA]"
        ),
        "vm_internal_pu": pa.Column(float, nullable=True, description="voltage magnitude of the internal AC bus [p.u.]"),
        "va_internal_degree": pa.Column(float, nullable=True, description="voltage angle of the internal AC bus [degree]"),
        "vm_pu": pa.Column(float, nullable=True, description="voltage magnitude at the AC bus [p.u.]"),
        "va_degree": pa.Column(float, nullable=True, description="voltage angle at the AC bus [degree]"),
        "vm_internal_dc_pu": pa.Column(float, nullable=True, description="voltage of the internal DC bus [p.u.]"),
        "vm_dc_pu_p": pa.Column(float, nullable=True, description="voltage at the plus terminal bus [p.u.]"),
        "vm_dc_pu_m": pa.Column(float, nullable=True, description="voltage at the minus terminal bus [p.u.]"),
    },
    strict=False,
)
