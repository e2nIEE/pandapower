import pandera.pandas as pa

asymmetric_sgen_schema = pa.DataFrameSchema(
    {
        "name": pa.Column(str, required=False, description="name of the static generator"),
        "type": pa.Column(str, pa.Check.isin(["PV", "WP", "CHP"]), required=False, description="type of generator"),
        "bus": pa.Column(int, description="index of connected bus"),
        "p_a_mw": pa.Column(float, pa.Check.le(0), description="active power of the static generator : Phase A[MW]"),
        "q_a_mvar": pa.Column(float, description="reactive power of the static generator : Phase A [MVar]"),
        "p_b_mw": pa.Column(float, pa.Check.le(0), description="active power of the static generator : Phase B [MW]"),
        "q_b_mvar": pa.Column(float, description="reactive power of the static generator : Phase B [MVar]"),
        "p_c_mw": pa.Column(float, pa.Check.le(0), description="active power of the static generator : Phase C [MW]"),
        "q_c_mvar": pa.Column(float, description="reactive power of the static generator : Phase C [MVar]"),
        "sn_mva": pa.Column(
            float, pa.Check.gt(0), required=False, description="rated power ot the static generator [MVA]"
        ),
        "scaling": pa.Column(float, pa.Check.ge(0), description="scaling factor for the active and reactive power"),
        "in_service": pa.Column(bool, description="specifies if the generator is in service."),
        "current_source": pa.Column(bool, description=""),  # TODO: missing in docu
    },
    strict=False,
)


res_asymmetric_sgen_schema = pa.DataFrameSchema(
    {
        "p_mw": pa.Column(float, description=""),  # TODO: not in docu
        "q_mvar": pa.Column(float, description=""),  # TODO: not in docu
        # "p_a_mw": pa.Column(
        #     float, description="resulting active power demand after scaling : Phase A [MW"
        # ),  #TODO: only in docu
        # "q_a_mvar": pa.Column(
        #     float, description="resulting reactive power demand after scaling : Phase A [MVar]"
        # ),  #TODO: only in docu
        # "p_b_mw": pa.Column(
        #     float, description="resulting active power demand after scaling : Phase B [MW]"
        # ),  #TODO: only in docu
        # "q_b_mvar": pa.Column(
        #     float, description="resulting reactive power demand after scaling : Phase B [MVar]"
        # ),  #TODO: only in docu
        # "p_c_mw": pa.Column(
        #     float, description="resulting active power demand after scaling : Phase C [MW]"
        # ),  #TODO: only in docu
        # "q_c_mvar": pa.Column(
        #     float, description="resulting reactive power demand after scaling : Phase C [MVar]"
        # ),  #TODO: only in docu
    },
)


res_asymmetric_sgen_3ph_schema = pa.DataFrameSchema(
    {
        "p_a_mw": pa.Column(float, description=""),  # TODO: not in docu
        "q_a_mvar": pa.Column(float, description=""),  # TODO: not in docu
        "p_b_mw": pa.Column(float, description=""),  # TODO: not in docu
        "q_b_mvar": pa.Column(float, description=""),  # TODO: not in docu
        "p_c_mw": pa.Column(float, description=""),  # TODO: not in docu
        "q_c_mvar": pa.Column(float, description=""),  # TODO: not in docu
    },
)
