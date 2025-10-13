import pandera.pandas as pa

dcline_schema = pa.DataFrameSchema(
    {
        "name": pa.Column(str, required=False, description="name of the generator"),
        "from_bus": pa.Column(int, pa.Check.ge(0), description="Index of bus where the dc line starts"),
        "to_bus": pa.Column(int, pa.Check.ge(0), description="Index of bus where the dc line ends"),
        "p_mw": pa.Column(float, description="Active power transmitted from ‘from_bus’ to ‘to_bus’"),
        "loss_percent": pa.Column(
            float, pa.Check.gt(0), description="Relative transmission loss in percent of active power transmission"
        ),
        "loss_mw": pa.Column(float, pa.Check.gt(0), description="Total transmission loss in MW"),
        "vm_from_pu": pa.Column(float, pa.Check.gt(0), description="Voltage setpoint at from bus"),
        "vm_to_pu": pa.Column(float, pa.Check.gt(0), description="Voltage setpoint at to bus"),
        "max_p_mw": pa.Column(
            float,
            pa.Check.gt(0),
            required=False,
            description="Maximum active power transmission",
            metadata={"opf": True},
        ),  # TODO: no min_p_mw ?
        "min_q_from_mvar": pa.Column(
            float, required=False, description="Minimum reactive power at from bus", metadata={"opf": True}
        ),
        "max_q_from_mvar": pa.Column(
            float, required=False, description="Maximum reactive power at from bus", metadata={"opf": True}
        ),
        "min_q_to_mvar": pa.Column(
            float, required=False, description="Minimum reactive power at to bus", metadata={"opf": True}
        ),
        "max_q_to_mvar": pa.Column(
            float, required=False, description="Maximum reactive power at to bus", metadata={"opf": True}
        ),
        "in_service": pa.Column(bool, description="specifies if the line is in service."),
    },
    strict=False,
)

res_dcline_schema = pa.DataFrameSchema(
    {
        "p_from_mw": pa.Column(float, description="active power flow into the line at ‘from_bus’ [MW]"),
        "q_from_mvar": pa.Column(float, description="reactive power flow into the line at ‘from_bus’ [kVar]"),
        "p_to_mw": pa.Column(float, description="active power flow into the line at ‘to_bus’ [MW]"),
        "q_to_mvar": pa.Column(float, description="reactive power flow into the line at ‘to_bus’ [kVar]"),
        "pl_mw": pa.Column(float, description="active power losses of the line [MW]"),
        "vm_from_pu": pa.Column(float, description="voltage magnitude at ‘from_bus’ [p.u]"),
        "va_from_degree": pa.Column(float, description="voltage angle at ‘from_bus’ [degree]"),
        "vm_to_pu": pa.Column(float, description="voltage magnitude at ‘to_bus’ [p.u]"),
        "va_to_degree": pa.Column(float, description="voltage angle at ‘to_bus’ [degree]"),
    },
)
