import pandera.pandas as pa

schema = pa.DataFrameSchema(
    {
        "name": pa.Column(str, description="name of the static generator"),
        "type": pa.Column(str, pa.Check.isin(["PV", "WP", "CHP"]), description="type of generator"),
        "bus": pa.Column(int, description="index of connected bus"),
        "p_a_mw": pa.Column(float, pa.Check.se(0), description="active power of the static generator : Phase A[MW]"),
        "q_a_mvar": pa.Column(float, description="reactive power of the static generator : Phase A [MVar]"),
        "p_b_mw": pa.Column(float, pa.Check.se(0), description="active power of the static generator : Phase B [MW]"),
        "q_b_mvar": pa.Column(float, description="reactive power of the static generator : Phase B [MVar]"),
        "p_c_mw": pa.Column(float, pa.Check.se(0), description="active power of the static generator : Phase C [MW]"),
        "q_c_mvar": pa.Column(float, description="reactive power of the static generator : Phase C [MVar]"),
        "sn_mva": pa.Column(float, pa.Check.gt(0), description="rated power ot the static generator [MVA]"),
        "scaling": pa.Column(float, pa.Check.ge(0), description="scaling factor for the active and reactive power"),
        "in_service": pa.Column(bool, description="specifies if the generator is in service."),
        "current_source": pa.Column(bool, description="")  # missing in docu, not a create method parameter, kwargs?
    },
    strict=False,
)
