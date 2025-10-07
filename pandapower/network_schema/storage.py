import pandera.pandas as pa

schema = pa.DataFrameSchema(
    {
        "name": pa.Column(str, description="Name of the storage unit"),
        "bus": pa.Column(int, description="Index of connected bus"),
        "p_mw": pa.Column(float, pa.Check.se(0), description="Momentary real power of the storage (positive for charging, negative for discharging)"),
        "q_mvar": pa.Column(float, description="Reactive power of the storage [MVar]"),
        "sn_mva": pa.Column(float, pa.Check.gt(0), description="Nominal power ot the storage [MVA]"),
        "scaling": pa.Column(float, pa.Check.ge(0), description="Scaling factor for the active and reactive power"),
        "max_e_mwh": pa.Column(float, description="The maximum energy content of the storage (maximum charge level)"),
        "min_e_mwh": pa.Column(float, description="The minimum energy content of the storage (minimum charge level)"),
        "max_p_mw": pa.Column(float, description="Maximum active power"),
        "min_p_mw": pa.Column(float, description="Minimum active power"),
        "soc_percent": pa.Column(float, pa.Check.between(min_value=0, max_value=100), description="The state of charge of the storage"),
        "max_q_mvar": pa.Column(float, description="Maximum reactive power [MVar]"),
        "min_q_mvar": pa.Column(float, description="Minimum reactive power [MVar]"),
        "controllable": pa.Column(bool, description="States if sgen is controllable or not, sgen will not be used as a flexibilty if it is not controllable"),
        "in_service": pa.Column(bool, description="Specifies if the generator is in service"),
        "type": pa.Column(str, description="")  # missing in docu
    },
    strict=False,
)



res_schema = pa.DataFrameSchema(
    {
        "p_mw": pa.Column(float, description="resulting active power after scaling [MW]"),
        "q_mvar": pa.Column(float, description="resulting reactive power after scaling [MVar]")
    },
    strict=False,
)