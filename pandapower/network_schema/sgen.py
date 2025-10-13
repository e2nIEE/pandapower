import pandera.pandas as pa

sgen_schema = pa.DataFrameSchema(  # TODO: in methodcall but not parameter docu: generator_type, max_i_ka, kappa, lrc_pu
    {
        "name": pa.Column(str, description="name of the static generator"),
        "bus": pa.Column(int, pa.Check.ge(0), description="index of connected bus"),
        "p_mw": pa.Column(
            float, pa.Check.le(0), description="active power of the static generator [MW]"
        ),  # TODO: surely the docu must be wrong for le0
        "q_mvar": pa.Column(float, description="reactive power of the static generator [MVAr]"),
        "sn_mva": pa.Column(float, pa.Check.gt(0), description="rated power ot the static generator [MVA]"),
        "scaling": pa.Column(float, pa.Check.ge(0), description="scaling factor for the active and reactive power"),
        # "min_p_mw": pa.Column(float, required=False, description="maximum active power [MW]", metadata={"opf": True}), #TODO: only in docu
        # "max_p_mw": pa.Column(float, required=False, description="minimum active power [MW]", metadata={"opf": True}), #TODO: only in docu
        "min_q_mvar": pa.Column(
            float, required=False, description="maximum reactive power [MVAr]", metadata={"opf": True}
        ),
        "max_q_mvar": pa.Column(
            float, required=False, description="minimum reactive power [MVAr]", metadata={"opf": True}
        ),
        "controllable": pa.Column(
            bool,
            required=False,
            description="states if sgen is controllable or not, sgen will not be used as a flexibility if it is not controllable",
        ),
        # "k": pa.Column(
        #     float,
        #     pa.Check.ge(0),
        #     required=False,
        #     description="ratio of short circuit current to nominal current",
        #     metadata={"sc": True},
        # ), #TODO: only in docu
        # "rx": pa.Column(
        #     float,
        #     pa.Check.ge(0),
        #     required=False,
        #     description="R/X ratio for short circuit impedance. Only relevant if type is specified as motor so that sgen is treated as asynchronous motor",
        #     metadata={"sc": True},
        # ), #TODO: only in docu
        "in_service": pa.Column(bool, description="specifies if the generator is in service."),
        "id_q_capability_characteristic": pa.Column(
            int,
            nullable=True,
            description="references the index of the characteristic from the q_capability_characteristic",
        ),
        "curve_style": pa.Column(
            str,
            pa.Check.isin(["straightLineYValues", "constantYValue"]),
            nullable=True,
            description="the style of the static generator reactive power capability curve",
        ),
        "reactive_capability_curve": pa.Column(
            bool, description="True if static generator has dependency on q characteristic"
        ),
        "type": pa.Column(str, pa.Check.isin(["PV", "WP", "CHP"]), description="type of generator"),
        "current_source": pa.Column(bool, description=""),  # TODO: missing in docu
        "generator_type": pa.Column(
            str,
            description="can be one of current_source (full size converter), async (asynchronous generator), or async_doubly_fed (doubly fed asynchronous generator, DFIG). Represents the type of the static generator in the context of the short-circuit calculations of wind power station units. If None, other short-circuit-related parameters are not set",
        ),
        "lrc_pu": pa.Column(
            float,
            description="locked rotor current in relation to the rated generator current. Relevant if the generator_type is async.",
        ),
        "max_ik_ka": pa.Column(
            float,
            description="the highest instantaneous short-circuit value in case of a three-phase short-circuit (provided by the manufacturer). Relevant if the generator_type is async_doubly_fed.",
        ),
        "kappa": pa.Column(
            float,
            description="the factor for the calculation of the peak short-circuit current, referred to the high-voltage side (provided by the manufacturer). Relevant if the generator_type is async_doubly_fed. If the superposition method is used (use_pre_fault_voltage=True), this parameter is used to pass through the max. current limit of the machine in p.u.",
        ),
    },
    strict=False,
)

res_sgen_schema = pa.DataFrameSchema(
    {
        "p_mw": pa.Column(float, description="resulting active power demand after scaling [MW]"),
        "q_mvar": pa.Column(float, description="resulting reactive power demand after scaling [MVAr]"),
    },
)
