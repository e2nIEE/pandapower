import pandas as pd
import pandera.pandas as pa

from pandapower.network_schema.tools.validation.group_dependency import create_column_dependency_checks_from_metadata

_sgen_columns = {
    "name": pa.Column(pd.StringDtype, nullable=True, required=False, description="name of the static generator"),
    "bus": pa.Column(int, pa.Check.ge(0), description="index of connected bus", metadata={"foreign_key": "bus.index"}),
    "p_mw": pa.Column(float, description="active power of the static generator [MW]"),
    "q_mvar": pa.Column(float, description="reactive power of the static generator [MVAr]"),
    "sn_mva": pa.Column(
        float,
        pa.Check.gt(0),
        nullable=True,
        required=False,
        description="rated power ot the static generator [MVA]",
    ),
    "scaling": pa.Column(float, pa.Check.ge(0), description="scaling factor for the active and reactive power"),
    "min_p_mw": pa.Column(
        float, nullable=True, required=False, description="maximum active power [MW]", metadata={"opf": True}
    ),
    "max_p_mw": pa.Column(
        float, nullable=True, required=False, description="minimum active power [MW]", metadata={"opf": True}
    ),
    "min_q_mvar": pa.Column(
        float, nullable=True, required=False, description="maximum reactive power [MVAr]", metadata={"opf": True}
    ),
    "max_q_mvar": pa.Column(
        float, nullable=True, required=False, description="minimum reactive power [MVAr]", metadata={"opf": True}
    ),
    "controllable": pa.Column(
        pd.BooleanDtype,
        nullable=True,
        required=False,
        description="states if sgen is controllable or not, sgen will not be used as a flexibility if it is not controllable",
    ),
    "k": pa.Column(
        float,
        pa.Check.ge(0),
        nullable=True,
        required=False,
        description="ratio of short circuit current to nominal current",
        metadata={"sc": True},
    ),
    "rx": pa.Column(
        float,
        pa.Check.ge(0),
        nullable=True,
        required=False,
        description="R/X ratio for short circuit impedance. Only relevant if type is specified as motor so that sgen is treated as asynchronous motor",
        metadata={"sc": True},
    ),
    "in_service": pa.Column(bool, description="specifies if the generator is in service."),
    "id_q_capability_characteristic": pa.Column(
        pd.Int64Dtype,
        nullable=True,
        required=False,
        description="references the index of the characteristic from the q_capability_characteristic",
        metadata={"qcc": True},
    ),
    "curve_style": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="the style of the static generator reactive power capability curve. Naming convention straightLineYValues, constantYValue",
        metadata={"qcc": True},
    ),
    "reactive_capability_curve": pa.Column(
        pd.BooleanDtype,
        required=False,
        description="True if static generator has dependency on q characteristic",
        metadata={"qcc": True},
    ),
    "type": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="type of generator naming conventions: “PV” - photovoltaic system “WP” - wind power system “CHP” - combined heating and power system",
    ),
    "current_source": pa.Column(
        pd.BooleanDtype,
        nullable=True,
        required=False,
        description=" Model this sgen as a current source during short- circuit calculations; useful in some cases, for example the simulation of full- size converters per IEC 60909-0:2016.",
        metadata={"sc": True},
    ),
    "generator_type": pa.Column(  # TODO: is this not an sgen, did someone model motor as an sgen?
        pd.StringDtype,
        nullable=True,
        required=False,
        description="can be one of current_source (full size converter), async (asynchronous generator), or async_doubly_fed (doubly fed asynchronous generator, DFIG). Represents the type of the static generator in the context of the short-circuit calculations of wind power station units. If None, other short-circuit-related parameters are not set",
        metadata={"sc": True},
    ),
    "lrc_pu": pa.Column(
        float,
        nullable=True,
        required=False,
        description="locked rotor current in relation to the rated generator current. Relevant if the generator_type is async.",
        metadata={"sc": True},
    ),
    "max_ik_ka": pa.Column(
        float,
        nullable=True,
        required=False,
        description="the highest instantaneous short-circuit value in case of a three-phase short-circuit (provided by the manufacturer). Relevant if the generator_type is async_doubly_fed.",
        metadata={"sc": True},
    ),
    "kappa": pa.Column(
        float,
        nullable=True,
        required=False,
        description="the factor for the calculation of the peak short-circuit current, referred to the high-voltage side (provided by the manufacturer). Relevant if the generator_type is async_doubly_fed. If the superposition method is used (use_pre_fault_voltage=True), this parameter is used to pass through the max. current limit of the machine in p.u.",
        metadata={"sc": True},
    ),
}
sgen_schema = pa.DataFrameSchema(
    _sgen_columns,
    strict=False,
    checks=create_column_dependency_checks_from_metadata(["opf", "sc", "qcc"], _sgen_columns),
)

res_sgen_schema = pa.DataFrameSchema(
    {
        "p_mw": pa.Column(float, nullable=True, description="resulting active power production after scaling [MW]"),
        "q_mvar": pa.Column(
            float, nullable=True, description="resulting reactive power production after scaling [MVar]"
        ),
    },
    strict=False,
)
