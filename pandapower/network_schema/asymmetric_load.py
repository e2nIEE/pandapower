import pandera.pandas as pa

asymmetric_load_schema = pa.DataFrameSchema(
    {
        "name": pa.Column(str, required=False, description="name of the load"),
        "bus": pa.Column(int, description="	index of connected bus"),
        "p_a_mw": pa.Column(float, pa.Check.ge(0), description="Phase A active power of the load [MW]"),
        "p_b_mw": pa.Column(float, pa.Check.ge(0), description="Phase B active power of the load [MW]"),
        "p_c_mw": pa.Column(float, pa.Check.ge(0), description="Phase C active power of the load [MW]"),
        "q_a_mvar": pa.Column(float, description="Phase A reactive power of the load [MVar]"),
        "q_b_mvar": pa.Column(float, description="Phase B reactive power of the load [MVar]"),
        "q_c_mvar": pa.Column(float, description="Phase C reactive power of the load [MVar]"),
        "sn_mva": pa.Column(float, pa.Check.gt(0), required=False, description="rated power of the load [MVA]"),
        "scaling": pa.Column(float, pa.Check.ge(0), description="scaling factor for active and reactive power"),
        "in_service": pa.Column(bool, description="specifies if the load is in service."),
        "type": pa.Column(str, pa.Check.isin(["wye", "delta"]), description="type of load"),
    },
    strict=False,
)


res_asymmetric_load_schema = pa.DataFrameSchema(
    {
        "p_mw": pa.Column(float, description=""),  # TODO: not in docu
        "q_mvar": pa.Column(float, description=""),  # TODO: not in docu
        # "p_a_mw": pa.Column(
        #     float,
        #     description="resulting Phase A active power demand after scaling and after considering voltage dependence [MW]",
        # ),  #TODO: only in docu
        # "q_a_mvar": pa.Column(
        #     float,
        #     description="resulting Phase A reactive power demand after scaling and after considering voltage dependence [MVar]",
        # ),  #TODO: only in docu
        # "p_b_mw": pa.Column(
        #     float,
        #     description="resulting Phase B active power demand after scaling and after considering voltage dependence [MW]",
        # ),  #TODO: only in docu
        # "q_b_mvar": pa.Column(
        #     float,
        #     description="resulting Phase B reactive power demand after scaling and after considering voltage dependence [MVar]",
        # ),  #TODO: only in docu
        # "p_c_mw": pa.Column(
        #     float,
        #     description="resulting Phase C active power demand after scaling and after considering voltage dependence [MW]",
        # ),  #TODO: only in docu
        # "q_c_mvar": pa.Column(
        #     float,
        #     description="resulting Phase C reactive power demand after scaling and after considering voltage dependence [MVar]",
        # ),  #TODO: only in docu
    },
)


res_asymmetric_load_3ph_schema = pa.DataFrameSchema(
    {
        "p_a_mw": pa.Column(float, description=""),  # not in docu
        "q_a_mvar": pa.Column(float, description=""),  # not in docu
        "p_b_mw": pa.Column(float, description=""),  # not in docu
        "q_b_mvar": pa.Column(float, description=""),  # not in docu
        "p_c_mw": pa.Column(float, description=""),  # not in docu
        "q_c_mvar": pa.Column(float, description=""),  # not in docu
    },
)
