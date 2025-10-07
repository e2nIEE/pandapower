import pandera.pandas as pa

schema = pa.DataFrameSchema(
    {
        "name": pa.Column(str, description="name of the generator"),
        "from_bus": pa.Column(int, pa.Check.ge(0), description="Index of bus where the dc line starts"),
        "to_bus": pa.Column(int, pa.Check.ge(0), description="Index of bus where the dc line ends"),
        "p_mw": pa.Column(float, description="Active power transmitted from ‘from_bus’ to ‘to_bus’"),
        "loss_percent": pa.Column(
            float, pa.Check.gt(0), description="Relative transmission loss in percent of active power transmission"
        ),
        "loss_mw": pa.Column(float, pa.Check.gt(0), description="Total transmission loss in MW"),
        "vm_from_pu": pa.Column(float, pa.Check.gt(0), description="Voltage setpoint at from bus"),
        "vm_to_pu": pa.Column(float, pa.Check.gt(0), description="Voltage setpoint at to bus"),
        "max_p_mw": pa.Column(float, pa.Check.gt(0), description="Maximum active power transmission"),  # no min_p_mw ?
        "min_q_from_mvar": pa.Column(float, description="Minimum reactive power at from bus"),
        "max_q_from_mvar": pa.Column(float, description="Maximum reactive power at from bus"),
        "min_q_to_mvar": pa.Column(float, description="Minimum reactive power at to bus"),
        "max_q_to_mvar": pa.Column(float, description="Maximum reactive power at to bus"),
        "in_service": pa.Column(bool, description="specifies if the line is in service."),
    },
    strict=False,
)
