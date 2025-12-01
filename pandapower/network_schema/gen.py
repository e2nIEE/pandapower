import pandas as pd
import pandera.pandas as pa

from pandapower.network_schema.tools.validation.column_condition import create_lower_equals_column_check
from pandapower.network_schema.tools.validation.group_dependency import create_column_dependency_checks_from_metadata

_gen_columns = {
    "name": pa.Column(pd.StringDtype, nullable=True, required=False, description="name of the generator"),
    "type": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="type variable to classify generators naming conventions: “sync” - synchronous generator “async” - asynchronous generator",
    ),
    "bus": pa.Column(int, pa.Check.ge(0), description="index of connected bus", metadata={"foreign_key": "bus.index"}),
    "p_mw": pa.Column(float, description="active power of the generator [MW]"),
    "vm_pu": pa.Column(float, pa.Check.gt(0), description="voltage set point of the generator [p.u.]"),
    "sn_mva": pa.Column(
        float, pa.Check.gt(0), nullable=True, required=False, description="nominal power of the generator [MVA]"
    ),
    "max_q_mvar": pa.Column(
        float,
        nullable=True,
        required=False,
        description="maximum reactive power of the generator [MVAr]",
        metadata={"opf": True, "q_lim_enforced": True},
    ),
    "min_q_mvar": pa.Column(
        float,
        nullable=True,
        required=False,
        description="minimum reactive power of the generator [MVAr]",
        metadata={"opf": True, "q_lim_enforced": True},
    ),
    "scaling": pa.Column(
        float,
        pa.Check.ge(0),
        description="scaling factor for the active power",
    ),
    "max_p_mw": pa.Column(
        float, nullable=True, required=False, description="maximum active power", metadata={"opf": True}
    ),
    "min_p_mw": pa.Column(
        float, nullable=True, required=False, description="minimum active power", metadata={"opf": True}
    ),
    "vn_kv": pa.Column(
        float, nullable=True, required=False, description="rated voltage of the generator", metadata={"sc": True}
    ),
    "xdss_pu": pa.Column(
        float,
        pa.Check.gt(0),
        nullable=True,
        required=False,
        description="subtransient generator reactance in per unit",
        metadata={"sc": True},
    ),
    "rdss_ohm": pa.Column(
        float,
        pa.Check.gt(0),
        nullable=True,
        required=False,
        description="subtransient generator resistence in ohm",
        metadata={"sc": True},
    ),
    "cos_phi": pa.Column(
        float,
        pa.Check.between(min_value=0, max_value=1),
        nullable=True,
        required=False,
        description="rated generator cosine phi",
        metadata={"sc": True},
    ),
    "in_service": pa.Column(bool, description="specifies if the generator is in service"),
    "power_station_trafo": pa.Column(
        pd.Int64Dtype,
        nullable=True,
        required=False,
        description="index of the power station trafo (short-circuit relevant)",
        metadata={"sc": True},
    ),
    "id_q_capability_characteristic": pa.Column(
        pd.Int64Dtype,
        nullable=True,
        required=False,
        description="references the index of the characteristic from the q_capability_characteristic",
        metadata={"qcc": True},
    ),
    "curve_style": pa.Column(
        pd.StringDtype,
        pa.Check.isin(["straightLineYValues", "constantYValue"]),
        nullable=True,
        required=False,
        description="the style of the generator reactive power capability curve",
        metadata={"qcc": True},
    ),
    "reactive_capability_curve": pa.Column(
        pd.BooleanDtype,
        nullable=True,
        required=False,
        description="True if generator has dependency on q characteristic",
        metadata={"qcc": True},
    ),
    "slack_weight": pa.Column(
        float, nullable=True, required=False, description="weight of the slack when using multiple slacks"
    ),
    "slack": pa.Column(bool, description="use the gen as slack"),
    "controllable": pa.Column(
        pd.BooleanDtype, nullable=True, required=False, description="allow control for opf", metadata={"opf": True}
    ),
    "pg_percent": pa.Column(
        float,
        nullable=True,
        required=False,
        description="Rated pg (voltage control range) of the generator for short-circuit calculation",
        metadata={"sc": True},
    ),
    "min_vm_pu": pa.Column(
        float,
        pa.Check.gt(0),
        nullable=True,
        required=False,
        description="Minimum voltage magnitude. If not set, the bus voltage limit is taken - necessary for OPF.",
        metadata={"opf": True},
    ),
    "max_vm_pu": pa.Column(
        float,
        pa.Check.gt(0),
        nullable=True,
        required=False,
        description="Maximum voltage magnitude. If not set, the bus voltage limit is taken - necessary for OPF",
        metadata={"opf": True},
    ),
}
gen_checks = create_column_dependency_checks_from_metadata(
        [
            "opf",
            # "sc",
            "q_lim_enforced",
            "qcc",
        ],
        _gen_columns,
    )
gen_checks.append(create_lower_equals_column_check(first_element="min_q_mvar", second_element="max_q_mvar"))
gen_checks.append(create_lower_equals_column_check(first_element="min_p_mw", second_element="max_p_mw"))
gen_checks.append(create_lower_equals_column_check(first_element="min_vm_pu", second_element="max_vm_pu"))
gen_schema = pa.DataFrameSchema(
    _gen_columns,
    strict=False,
    checks=gen_checks,
)


res_gen_schema = pa.DataFrameSchema(
    {
        "p_mw": pa.Column(float, nullable=True, description="resulting active power production after scaling [MW]"),
        "q_mvar": pa.Column(
            float, nullable=True, description="resulting reactive power production after scaling [MVar]"
        ),
        "va_degree": pa.Column(float, nullable=True, description="generator voltage angle [degree]"),
        "vm_pu": pa.Column(float, nullable=True, description="voltage at the generator [p.u.]"),
    },
    strict=False,
)
