import pandera.pandas as pa
import pandas as pd

gen_schema = pa.DataFrameSchema(
    {
        "name": pa.Column(pd.StringDtype, required=False, description="name of the generator"),
        "type": pa.Column(
            str,
            required=False,
            description="type variable to classify generators naming conventions: “sync” - synchronous generator “async” - asynchronous generator",
        ),
        "bus": pa.Column(int, description="index of connected bus", metadata={"foreign_key": "bus.index"}),
        "p_mw": pa.Column(float, description="the real power of the generator [MW]"),
        "vm_pu": pa.Column(float, description="voltage set point of the generator [p.u.]"),
        "sn_mva": pa.Column(float, pa.Check.gt(0), required=False, description="nominal power of the generator [MVA]"),
        "max_q_mvar": pa.Column(
            float,
            required=False,
            description="maximum reactive power of the generator [MVAr]",
            metadata={"opf": True, "q_lim_enforced": True},
        ),
        "min_q_mvar": pa.Column(
            float,
            required=False,
            description="minimum reactive power of the generator [MVAr]",
            metadata={"opf": True, "q_lim_enforced": True},
        ),
        "scaling": pa.Column(
            float,
            pa.Check.ge(0),
            description="scaling factor for the active power",
        ),
        "max_p_mw": pa.Column(float, required=False, description="maximum active power", metadata={"opf": True}),
        "min_p_mw": pa.Column(float, required=False, description="minimum active power", metadata={"opf": True}),
        "vn_kv": pa.Column(float, required=False, description="rated voltage of the generator", metadata={"sc": True}),
        "xdss_pu": pa.Column(
            float,
            pa.Check.gt(0),
            required=False,
            description="subtransient generator reactance in per unit",
            metadata={"sc": True},
        ),
        "rdss_ohm": pa.Column(
            float,
            pa.Check.gt(0),
            required=False,
            description="subtransient generator resistence in ohm",
            metadata={"sc": True},
        ),
        "cos_phi": pa.Column(
            float,
            pa.Check.between(min_value=0, max_value=1),
            required=False,
            description="rated generator cosine phi",
            metadata={"sc": True},
        ),
        "in_service": pa.Column(bool, description="specifies if the generator is in service"),
        "power_station_trafo": pa.Column(
            int,
            required=False,
            description="index of the power station trafo (short-circuit relevant)",
            metadata={"sc": True},
        ),
        "id_q_capability_characteristic": pa.Column(
            pd.Int64Dtype(),
            required=False,
            description="references the index of the characteristic from the q_capability_characteristic",
        ),
        "curve_style": pa.Column(
            str,
            pa.Check.isin(["straightLineYValues", "constantYValue", ""]),
            required=False,
            description="the style of the generator reactive power capability curve",
        ),
        "reactive_capability_curve": pa.Column(
            bool, required=False, description="True if generator has dependency on q characteristic"
        ),
        "slack_weight": pa.Column(float, required=False, description="weight of the slack when using multiple slacks"),
        "slack": pa.Column(bool, description="use the gen as slack"),
        "controllable": pa.Column(bool, required=False, description="allow control for opf", metadata={"opf": True}),
        "pg_percent": pa.Column(
            float,
            required=False,
            description="Rated pg (voltage control range) of the generator for short-circuit calculation",
            metadata={"sc": True},
        ),
        "min_vm_pu": pa.Column(
            float,
            required=False,
            description="Minimum voltage magnitude. If not set, the bus voltage limit is taken - necessary for OPF.",
            metadata={"opf": True},
        ),
        "max_vm_pu": pa.Column(
            float,
            required=False,
            description="Maximum voltage magnitude. If not set, the bus voltage limit is taken - necessary for OPF",
            metadata={"opf": True},
        ),
    },
    strict=False,
)


res_gen_schema = pa.DataFrameSchema(
    {
        "p_mw": pa.Column(float, description="resulting active power demand after scaling [MW]"),
        "q_mvar": pa.Column(float, description="resulting reactive power demand after scaling [MVAr]"),
        "va_degree": pa.Column(float, description="generator voltage angle [degree]"),
        "vm_pu": pa.Column(float, description="voltage at the generator [p.u.]"),
    },
)
