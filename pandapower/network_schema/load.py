import pandera.pandas as pa

load_schema = pa.DataFrameSchema(
    {
        "name": pa.Column(str, required=False, description="name of the load"),
        "bus": pa.Column(int, pa.Check.ge(0), description="index of connected bus"),
        "p_mw": pa.Column(
            float, pa.Check.ge(0), description="active power of the load [MW]"
        ),  # TODO: surely the docu must be wrong for ge0
        "q_mvar": pa.Column(float, description="reactive power of the load [MVar]"),
        "const_z_p_percent": pa.Column(
            float,
            pa.Check.between(min_value=0, max_value=100),
            description="percentage of p_mw that is associated to constant impedance load at rated voltage [%]",
        ),
        "const_i_p_percent": pa.Column(
            float,
            pa.Check.between(min_value=0, max_value=100),
            description="percentage of p_mw that is associated to constant current load at rated voltage [%]",
        ),
        "const_z_q_percent": pa.Column(
            float,
            pa.Check.between(min_value=0, max_value=100),
            description="percentage of q_mvar that is associated to constant impedance load at rated voltage [%]",
        ),
        "const_i_q_percent": pa.Column(
            float,
            pa.Check.between(min_value=0, max_value=100),
            description="percentage of q_mvar that is associated to constant current load at rated voltage [%]",
        ),
        "sn_mva": pa.Column(
            float, pa.Check.gt(0), required=False, nullable=True, description="rated power of the load [kVA]"
        ),
        "scaling": pa.Column(float, pa.Check.ge(0), description="scaling factor for active and reactive power"),
        "in_service": pa.Column(bool, description="specifies if the load is in service."),
        "type": pa.Column(
            str,
            pa.Check.isin(["wye", "delta"]),
            description="Connection Type of 3 Phase Load(Valid for three phase load flow only)",
        ),
        "controllable": pa.Column(
            bool,
            required=False,
            description="States if load is controllable or not, load will not be used as a flexibilty if it is not controllable",
        ),
        "zone": pa.Column(str, nullable=True, required=False, description=""),  # TODO: missing in docu
        "max_p_mw": pa.Column(float, required=False, description="Maximum active power"),
        "min_p_mw": pa.Column(float, required=False, description="Minimum active power"),
        "max_q_mvar": pa.Column(float, required=False, description="Maximum reactive power"),
        "min_q_mvar": pa.Column(float, required=False, description="Minimum reactive power"),
    },
    strict=False,
)

res_load_schema = pa.DataFrameSchema(
    {
        "p_mw": pa.Column(
            float,
            description="resulting active power demand after scaling and after considering voltage dependence [MW]",
        ),
        "q_mvar": pa.Column(
            float,
            description="resulting reactive power demand after scaling and after considering voltage dependence [MVar]",
        ),
    },
)
