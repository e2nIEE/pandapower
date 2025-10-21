from importlib.metadata import metadata

import pandera.pandas as pa

storage_schema = pa.DataFrameSchema(
    {
        "name": pa.Column(str, required=False, description="Name of the storage unit"),
        "bus": pa.Column(
            int, pa.Check.ge(0), description="Index of connected bus", metadata={"foreign_key": "bus.index"}
        ),
        "p_mw": pa.Column(
            float,
            # pa.Check.le(0),
            description="Momentary real power of the storage (positive for charging, negative for discharging)",
        ),
        "q_mvar": pa.Column(float, description="Reactive power of the storage [MVar]"),
        "sn_mva": pa.Column(float, pa.Check.gt(0), nullable=True, description="Nominal power ot the storage [MVA]"),
        "scaling": pa.Column(float, pa.Check.ge(0), description="Scaling factor for the active and reactive power"),
        "max_e_mwh": pa.Column(
            float, required=False, description="The maximum energy content of the storage (maximum charge level)"
        ),
        "min_e_mwh": pa.Column(
            float, required=False, description="The minimum energy content of the storage (minimum charge level)"
        ),
        "max_p_mw": pa.Column(
            float, required=False, description="Maximum active power", metadata={"opf": True}
        ),  # TODO: only in docu
        "min_p_mw": pa.Column(
            float, required=False, description="Minimum active power", metadata={"opf": True}
        ),  # TODO: only in docu
        "soc_percent": pa.Column(
            float,
            pa.Check.between(min_value=0, max_value=100),
            required=False,
            description="The state of charge of the storage",
        ),
        "max_q_mvar": pa.Column(
            float, required=False, description="Maximum reactive power [MVar]", metadata={"opf": True}
        ),  # TODO: only in docu
        "min_q_mvar": pa.Column(
            float, required=False, description="Minimum reactive power [MVar]", metadata={"opf": True}
        ),  # TODO: only in docu
        "controllable": pa.Column(
            bool,
            required=False,
            description="States if sgen is controllable or not, sgen will not be used as a flexibilty if it is not controllable",
            metadata={"opf": True},
        ),  # TODO: only in docu
        "in_service": pa.Column(bool, description="Specifies if the generator is in service"),
        "type": pa.Column(str, required=False, description=""),  # TODO: missing in docu
    },
    strict=False,
)


res_storage_schema = pa.DataFrameSchema(
    {
        "p_mw": pa.Column(float, description="resulting active power after scaling [MW]"),
        "q_mvar": pa.Column(float, description="resulting reactive power after scaling [MVar]"),
    },
)

res_storage_3ph_schema = pa.DataFrameSchema(
    {
        "p_a_mw": pa.Column(float, description=""),  # TODO: not in docu
        "p_b_mw": pa.Column(float, description=""),  # TODO: not in docu
        "p_c_mw": pa.Column(float, description=""),  # TODO: not in docu
        "q_a_mvar": pa.Column(float, description=""),  # TODO: not in docu
        "q_b_mvar": pa.Column(float, description=""),  # TODO: not in docu
        "q_c_mvar": pa.Column(float, description=""),  # TODO: not in docu
    },
)
