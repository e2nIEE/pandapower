import pandas as pd
import pandera.pandas as pa

from pandapower.network_schema.tools import (
    validate_column_group_dependency,
    create_column_dependency_checks_from_metadata,
)

_trafo_columns = {
    "name": pa.Column(pd.StringDtype, nullable=True, required=False, description="name of the transformer"),
    "std_type": pa.Column(str, nullable=True, required=False, description="transformer standard type name"),
    "hv_bus": pa.Column(
        int,
        pa.Check.ge(0),
        description="high voltage bus index of the transformer",
        metadata={"foreign_key": "bus.index"},
    ),
    "lv_bus": pa.Column(
        int, pa.Check.ge(0), description="low voltage bus index of the transformer", metadata={"foreign_key": "bus"}
    ),
    "sn_mva": pa.Column(float, pa.Check.gt(0), description="rated apparent power of the transformer [MVA]"),
    "vn_hv_kv": pa.Column(float, pa.Check.gt(0), description="rated voltage at high voltage bus [kV]"),
    "vn_lv_kv": pa.Column(float, pa.Check.gt(0), description="rated voltage at low voltage bus [kV]"),
    "vk_percent": pa.Column(float, pa.Check.gt(0), description="short circuit voltage [%]"),
    "vkr_percent": pa.Column(float, pa.Check.ge(0), description="real component of short circuit voltage [%]"),
    "pfe_kw": pa.Column(float, pa.Check.ge(0), description="iron losses [kW]"),
    "i0_percent": pa.Column(float, pa.Check.ge(0), description="open loop losses in [%]"),
    "vk0_percent": pa.Column(
        float,
        pa.Check.ge(0),
        nullable=True,
        required=False,
        description="zero sequence relative short-circuit voltage",
        metadata={"sc": True, "3ph": True},
    ),
    "vkr0_percent": pa.Column(
        float,
        pa.Check.ge(0),
        nullable=True,
        required=False,
        description="real part of zero sequence relative short-circuit voltage",
        metadata={"sc": True, "3ph": True},
    ),
    "mag0_percent": pa.Column(
        float,
        pa.Check.ge(0),
        nullable=True,
        required=False,
        description="z_mag0 / z0 ratio between magnetizing and short circuit impedance (zero sequence)",
        metadata={"sc": True, "3ph": True},
    ),
    "mag0_rx": pa.Column(
        float,
        nullable=True,
        required=False,
        description="zero sequence magnetizing r/x  ratio",
        metadata={"sc": True, "3ph": True},
    ),
    "si0_hv_partial": pa.Column(
        float,
        pa.Check.ge(0),
        nullable=True,
        required=False,
        description="zero sequence short circuit impedance  distribution in hv side",
        metadata={"sc": True, "3ph": True},
    ),
    "vector_group": pa.Column(
        str,
        nullable=True,
        required=False,
        description="Vector Groups ( required for zero sequence model of transformer )",
        metadata={"sc": True, "3ph": True},
    ),
    "shift_degree": pa.Column(float, description="transformer phase shift angle"),
    "tap_side": pa.Column(
        pd.StringDtype,
        pa.Check.isin(["hv", "lv"]),
        nullable=True,
        required=False,
        description="defines if tap changer is at the high- or low voltage side",
    ),
    "tap_neutral": pa.Column(
        float, nullable=True, required=False, description="rated tap position"
    ),  # TODO: different type in docu
    "tap_min": pa.Column(
        float, nullable=True, required=False, description="minimum tap position"
    ),  # TODO: different type in docu
    "tap_max": pa.Column(
        float, nullable=True, required=False, description="maximum tap position"
    ),  # TODO: different type in docu
    "tap_step_percent": pa.Column(
        float, pa.Check.gt(0), nullable=True, required=False, description="tap step size for voltage magnitude [%]"
    ),
    "tap_step_degree": pa.Column(
        float, pa.Check.ge(0), nullable=True, required=False, description="tap step size for voltage angle"
    ),
    "tap_pos": pa.Column(
        float, nullable=True, required=False, description="current position of tap changer"
    ),  # TODO: different type in docu
    "tap_changer_type": pa.Column(
        str,
        pa.Check.isin(["Ratio", "Symmetrical", "Ideal", "Tabular", "nan"]),
        nullable=True,
        required=False,  # TODO: according to test test_tap_changer_type_default it should be required
        description="specifies the tap changer type",
    ),
    "tap_dependency_table": pa.Column(
        bool,
        nullable=True,
        required=False,
        description="whether the transformer parameters (voltage ratio, angle, impedance) are adjusted dependent on the tap position of the transformer",
    ),
    "id_characteristic_table": pa.Column(
        pd.Int64Dtype,
        pa.Check.ge(0),
        nullable=True,
        required=False,
        description="references the id_characteristic index from the trafo_characteristic_table",
    ),
    "max_loading_percent": pa.Column(
        int,
        nullable=True,
        required=False,
        description="Maximum loading of the transformer with respect to sn_mva and its corresponding current at 1.0 p.u.",
        metadata={"opf": True},
    ),  # TODO: only in docu
    "parallel": pa.Column(
        int, pa.Check.ge(1), nullable=True, required=False, description="number of parallel transformers"
    ),  # TODO: was gt(0)
    "df": pa.Column(
        float,
        pa.Check.between(min_value=0, max_value=1, include_min=False),
        nullable=True,
        required=False,
        description="derating factor: maximum current of transformer in relation to nominal current of transformer (from 0 to 1)",
    ),
    "in_service": pa.Column(bool, description="specifies if the transformer is in service"),
    "oltc": pa.Column(
        bool,
        nullable=True,
        required=False,  # TODO: laut docu powerflow relevant, war aber nicht im alten network struct
        description="specifies if the transformer has an OLTC (short-circuit relevant)",
        metadata={"sc": True},
    ),
    "power_station_unit": pa.Column(
        bool,
        nullable=True,
        required=False,  # TODO: laut docu powerflow relevant, war aber nicht im alten network struct
        description="specifies if the transformer is part of a power_station_unit (short-circuit relevant)",
        metadata={"sc": True},
    ),  # TODO: not in create method call
    "tap2_side": pa.Column(str, pa.Check.isin(["hv", "lv", "nan"]), nullable=True, required=False, description=""),
    "tap2_neutral": pa.Column(pd.Float64Dtype, nullable=True, required=False, description="rated tap position"),
    "tap2_min": pa.Column(float, nullable=True, required=False, description="minimum tap position"),
    "tap2_max": pa.Column(float, nullable=True, required=False, description="maximum tap position"),
    "tap2_step_percent": pa.Column(
        float, pa.Check.gt(0), nullable=True, required=False, description="tap step size for voltage magnitude [%]"
    ),
    "tap2_step_degree": pa.Column(
        float, pa.Check.ge(0), nullable=True, required=False, description="tap step size for voltage angle"
    ),
    "tap2_pos": pa.Column(
        pd.Float64Dtype, nullable=True, required=False, description="current position of tap changer"
    ),
    "tap2_changer_type": pa.Column(
        str,
        pa.Check.isin(["Ratio", "Symmetrical", "Ideal", "nan"]),
        nullable=True,
        required=False,
        description="specifies the tap changer type",
    ),
    "leakage_resistance_ratio_hv": pa.Column(
        float,
        pa.Check.between(min_value=0, max_value=1),
        nullable=True,
        required=False,
        description="ratio of transformer short-circuit resistance on HV side (default 0.5)",
    ),  # TODO: not in create method call
    "leakage_reactance_ratio_hv": pa.Column(
        float,
        pa.Check.between(min_value=0, max_value=1),
        nullable=True,
        required=False,
        description="ratio of transformer short-circuit reactance on HV side (default 0.5)",
    ),  # TODO: not in create method call
    "xn_ohm": pa.Column(
        float,
        required=False,
        description="impedance of the grounding reactor (Z_N) for short circuit calculation",
        metadata={"sc": True},
    ),  # TODO: not in docu
    "pt_percent": pa.Column(float, required=False, description=""),  # TODO: not in docu
}
tap2_columns = ["tap2_pos", "tap2_neutral", "tap2_side", "tap2_step_percent", "tap2_step_degree"]
tap_columns = [
    "tap_pos",
    "tap_neutral",
    "tap_side",
    "tap_step_percent",
    "tap_step_degree",
]  # TODO: ideally tap_step_percent and tap_step_degree should not exist together
trafo_checks = [
    pa.Check(
        validate_column_group_dependency(tap_columns),
        error=f"Tap configuration columns have dependency violations. Please ensure {tap_columns} are present in the dataframe.",
    ),
    pa.Check(
        validate_column_group_dependency(tap2_columns),
        error=f"Tap2 configuration columns have dependency violations. Please ensure {tap2_columns} are present in the dataframe.",
    ),
]
trafo_checks += create_column_dependency_checks_from_metadata(["opf", "sc", "3ph"], _trafo_columns)
trafo_schema = pa.DataFrameSchema(
    _trafo_columns,
    checks=trafo_checks,
    strict=False,
)

res_trafo_schema = pa.DataFrameSchema(
    {
        "p_hv_mw": pa.Column(
            float, nullable=True, description="active power flow at the high voltage transformer bus [MW]"
        ),
        "q_hv_mvar": pa.Column(
            float, nullable=True, description="reactive power flow at the high voltage transformer bus [MVar]"
        ),
        "p_lv_mw": pa.Column(
            float, nullable=True, description="active power flow at the low voltage transformer bus [MW]"
        ),
        "q_lv_mvar": pa.Column(
            float, nullable=True, description="reactive power flow at the low voltage transformer bus [MVar]"
        ),
        "pl_mw": pa.Column(float, nullable=True, description="active power losses of the transformer [MW]"),
        "ql_mvar": pa.Column(float, nullable=True, description="reactive power consumption of the transformer [Mvar]"),
        "i_hv_ka": pa.Column(
            float, nullable=True, description="current at the high voltage side of the transformer [kA]"
        ),
        "i_lv_ka": pa.Column(
            float, nullable=True, description="current at the low voltage side of the transformer [kA]"
        ),
        "vm_hv_pu": pa.Column(float, nullable=True, description="voltage magnitude at the high voltage bus [pu]"),
        "va_hv_degree": pa.Column(float, nullable=True, description="voltage magnitude at the low voltage bus [pu]"),
        "vm_lv_pu": pa.Column(float, nullable=True, description="voltage angle at the high voltage bus [degrees]"),
        "va_lv_degree": pa.Column(float, nullable=True, description="voltage angle at the low voltage bus [degrees]"),
        "loading_percent": pa.Column(float, nullable=True, description="load utilization relative to rated power [%]"),
    },
    strict=False,
)

res_trafo_3ph_schema = pa.DataFrameSchema(
    {
        "p_a_hv_mw": pa.Column(
            float, nullable=True, description="active power flow at the high voltage transformer bus : Phase A [MW]"
        ),
        "q_a_hv_mvar": pa.Column(
            float, nullable=True, description="reactive power flow at the high voltage transformer bus : Phase A [MVar]"
        ),
        "p_b_hv_mw": pa.Column(
            float, nullable=True, description="active power flow at the high voltage transformer bus : Phase B [MW]"
        ),
        "q_b_hv_mvar": pa.Column(
            float, nullable=True, description="reactive power flow at the high voltage transformer bus : Phase B [MVar]"
        ),
        "p_c_hv_mw": pa.Column(
            float, nullable=True, description="active power flow at the high voltage transformer bus : Phase C [MW]"
        ),
        "q_c_hv_mvar": pa.Column(
            float, nullable=True, description="reactive power flow at the high voltage transformer bus : Phase C [MVar]"
        ),
        "p_a_lv_mw": pa.Column(
            float, nullable=True, description="active power flow at the low voltage transformer bus : Phase A [MW]"
        ),
        "q_a_lv_mvar": pa.Column(
            float, nullable=True, description="reactive power flow at the low voltage transformer bus : Phase A [MVar]"
        ),
        "p_b_lv_mw": pa.Column(
            float, nullable=True, description="active power flow at the low voltage transformer bus : Phase B [MW]"
        ),
        "q_b_lv_mvar": pa.Column(
            float, nullable=True, description="reactive power flow at the low voltage transformer bus : Phase B [MVar]"
        ),
        "p_c_lv_mw": pa.Column(
            float, nullable=True, description="active power flow at the low voltage transformer bus : Phase C [MW]"
        ),
        "q_c_lv_mvar": pa.Column(
            float, nullable=True, description="reactive power flow at the low voltage transformer bus : Phase C [MVar]"
        ),
        "pl_a_mw": pa.Column(float, nullable=True, description="active power losses of the transformer : Phase A [MW]"),
        "ql_a_mvar": pa.Column(
            float, nullable=True, description="reactive power consumption of the transformer : Phase A [Mvar]"
        ),
        "pl_b_mw": pa.Column(float, nullable=True, description="active power losses of the transformer : Phase B [MW]"),
        "ql_b_mvar": pa.Column(
            float, nullable=True, description="reactive power consumption of the transformer : Phase B [Mvar]"
        ),
        "pl_c_mw": pa.Column(float, nullable=True, description="active power losses of the transformer : Phase C [MW]"),
        "ql_c_mvar": pa.Column(
            float, nullable=True, description="reactive power consumption of the transformer : Phase C [Mvar]"
        ),
        "i_a_hv_ka": pa.Column(
            float, nullable=True, description="current at the high voltage side of the transformer : Phase A [kA]"
        ),
        "i_a_lv_ka": pa.Column(
            float, nullable=True, description="current at the low voltage side of the transformer : Phase A [kA]"
        ),
        "i_b_hv_ka": pa.Column(
            float, nullable=True, description="current at the high voltage side of the transformer : Phase B [kA]"
        ),
        "i_b_lv_ka": pa.Column(
            float, nullable=True, description="current at the low voltage side of the transformer : Phase B [kA]"
        ),
        "i_c_hv_ka": pa.Column(
            float, nullable=True, description="current at the high voltage side of the transformer : Phase C [kA]"
        ),
        "i_c_lv_ka": pa.Column(
            float, nullable=True, description="current at the low voltage side of the transformer : Phase C [kA]"
        ),
        "loading_a_percent": pa.Column(
            float, nullable=True, description="load utilization relative to rated power: Phase A [%]"
        ),
        "loading_b_percent": pa.Column(
            float, nullable=True, description="load utilization relative to rated power: Phase B [%]"
        ),
        "loading_c_percent": pa.Column(
            float, nullable=True, description="load utilization relative to rated power: Phase C [%]"
        ),
        "loading_percent": pa.Column(
            float,
            nullable=True,
            description="load utilization relative to rated power: Maximum of Phase A, B, C in [%]",
        ),
    },
    strict=False,
)

res_trafo_sc_schema = pa.DataFrameSchema(
    {
        "ikss_hv_ka": pa.Column(
            float, nullable=True, description="magnitude of the initial SC current at HV transformer bus [kA]"
        ),
        "ikss_hv_degree": pa.Column(
            float, nullable=True, description="degree of the initial SC current at HV transformer bus [degrees]"
        ),
        "ikss_lv_ka": pa.Column(
            float, nullable=True, description="magnitude of the initial SC current at LV transformer bus [kA]"
        ),
        "ikss_lv_degree": pa.Column(
            float, nullable=True, description="degree of the initial SC current at LV transformer bus [degrees]"
        ),
        "p_hv_mw": pa.Column(float, nullable=True, description="active SC power flow at HV transformer bus [MW]"),
        "q_hv_mvar": pa.Column(float, nullable=True, description="reactive SC power flow at HV transformer bus [MVAr]"),
        "p_lv_mw": pa.Column(float, nullable=True, description="active SC power flow at LV transformer bus [MW]"),
        "q_lv_mvar": pa.Column(float, nullable=True, description="reactive SC power flow at LV transformer bus [MVAr]"),
        "vm_hv_pu": pa.Column(
            float, nullable=True, description="voltage magnitude at the high voltage (HV) bus [p.u.]"
        ),
        "va_hv_degree": pa.Column(
            float, nullable=True, description="voltage angle at the high voltage (HV) bus [degrees]"
        ),
        "vm_lv_pu": pa.Column(float, nullable=True, description="voltage magnitude at the low voltage (LV) bus [p.u.]"),
        "va_lv_degree": pa.Column(
            float, nullable=True, description="voltage angle at the low voltage (LV) bus [degrees]"
        ),
    },
    strict=False,
)
