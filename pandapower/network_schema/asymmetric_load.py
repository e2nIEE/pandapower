import pandera.pandas as pa

schema = pa.DataFrameSchema(
    {
        "name": pa.Column(str, description="name of the load"),
        "bus": pa.Column(int, description="	index of connected bus"),
        "p_a_mw": pa.Column(float, pa.Check.ge(0), description="Phase A active power of the load [MW]"),
        "p_b_mw": pa.Column(float, pa.Check.ge(0), description="Phase B active power of the load [MW]"),
        "p_c_mw": pa.Column(float, pa.Check.ge(0), description="Phase C active power of the load [MW]"),
        "q_a_mvar": pa.Column(float, description="Phase A reactive power of the load [MVar]"),
        "q_b_mvar": pa.Column(float, description="Phase B reactive power of the load [MVar]"),
        "q_c_mvar": pa.Column(float, description="Phase C reactive power of the load [MVar]"),
        "sn_mva": pa.Column(float, pa.Check.gt(0), description="rated power of the load [MVA]"),
        "scaling": pa.Column(float, pa.Check.ge(0), description="scaling factor for active and reactive power"),
        "in_service": pa.Column(bool, description="specifies if the load is in service."),
        "type": pa.Column(str, pa.Check.isin(["wye", "delta"]),description="type of load"),
        "current_source": pa.Column(bool, description="")  # missing in docu, not a create method parameter, kwargs?
    },
    strict=False,
)
