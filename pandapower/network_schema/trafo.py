import pandera.pandas as pa

schema = pa.DataFrameSchema(  # in methodcall but not parameter docu: xn_ohm, pt_percent
    {
        "name": pa.Column(str, description="name of the transformer"),
        "std_type": pa.Column(str, description="transformer standard type name"),
        "hv_bus": pa.Column(int, pa.Check.ge(0), description="high voltage bus index of the transformer"),
        "lv_bus": pa.Column(int, pa.Check.ge(0), description="low voltage bus index of the transformer"),
        "sn_mva": pa.Column(float, pa.Check.gt(0), description="rated apparent power of the transformer [MVA]"),
        "vn_hv_kv": pa.Column(float, pa.Check.gt(0), description="rated voltage at high voltage bus [kV]"),
        "vn_lv_kv": pa.Column(float, pa.Check.gt(0), description="rated voltage at low voltage bus [kV]"),
        "vk_percent": pa.Column(float, pa.Check.gt(0), description="short circuit voltage [%]"),
        "vkr_percent": pa.Column(float, pa.Check.ge(0), description="real component of short circuit voltage [%]"),
        "pfe_kw": pa.Column(float, pa.Check.ge(0), description="iron losses [kW]"),
        "i0_percent": pa.Column(float, pa.Check.ge(0), description="open loop losses in [%]"),
        "vk0_percent": pa.Column(
            float, pa.Check.ge(0), required=False, description="zero sequence relative short-circuit voltage"
        ),
        "vkr0_percent": pa.Column(
            float,
            pa.Check.ge(0),
            required=False,
            description="real part of zero sequence relative short-circuit voltage",
        ),
        "mag0_percent": pa.Column(
            float,
            pa.Check.ge(0),
            required=False,
            description="z_mag0 / z0 ratio between magnetizing and short circuit impedance (zero sequence)",
        ),
        "mag0_rx": pa.Column(float, required=False, description="zero sequence magnetizing r/x  ratio"),
        "si0_hv_partial": pa.Column(
            float,
            pa.Check.ge(0),
            required=False,
            description="zero sequence short circuit impedance  distribution in hv side",
        ),
        "vector_group": pa.Column(
            str, required=False, description="Vector Groups ( required for zero sequence model of transformer )"
        ),
        "shift_degree": pa.Column(float, description="transformer phase shift angle"),
        "tap_side": pa.Column(
            str, pa.Check.isin(["hv", "lv"]), description="defines if tap changer is at the high- or low voltage side"
        ),
        "tap_neutral": pa.Column(int, description="rated tap position"),
        "tap_min": pa.Column(int, description="minimum tap position"),
        "tap_max": pa.Column(int, description="maximum tap position"),
        "tap_step_percent": pa.Column(float, pa.Check.gt(0), description="tap step size for voltage magnitude [%]"),
        "tap_step_degree": pa.Column(
            float, pa.Check.ge(0), nullable=True, description="tap step size for voltage angle"
        ),
        "tap_pos": pa.Column(int, description="current position of tap changer"),
        "tap_changer_type": pa.Column(
            str,
            pa.Check.isin(["Ratio", "Symmetrical", "Ideal", "Tabular"]),
            description="specifies the tap changer type",
        ),
        "tap_dependency_table": pa.Column(
            bool,
            required=False,
            description="whether the transformer parameters (voltage ratio, angle, impedance) are adjusted dependent on the tap position of the transformer",
        ),
        "id_characteristic_table": pa.Column(
            int,
            pa.Check.ge(0),
            required=False,
            description="references the id_characteristic index from the trafo_characteristic_table",
        ),
        "max_loading_percent": pa.Column(
            int,
            required=False,
            description="Maximum loading of the transformer with respect to sn_mva and its corresponding current at 1.0 p.u.",
        ),
        "parallel": pa.Column(int, pa.Check.gt(0), description="number of parallel transformers"),
        "df": pa.Column(
            float,
            pa.Check.between(min_value=0, max_value=1),
            description="derating factor: maximum current of transformer in relation to nominal current of transformer (from 0 to 1)",
        ),
        "in_service": pa.Column(bool, description="specifies if the transformer is in service"),
        "oltc": pa.Column(
            bool, required=False, description="specifies if the transformer has an OLTC (short-circuit relevant)"
        ),
        "power_station_unit": pa.Column(bool, required=False, description=""),  # not in create method call
        "tap2_side": pa.Column(int, pa.Check.isin(["hv", "lv"]), required=False, description=""),
        "tap2_neutral": pa.Column(int, required=False, description="rated tap position"),
        "tap2_min": pa.Column(int, required=False, description="minimum tap position"),
        "tap2_max": pa.Column(int, required=False, description="maximum tap position"),
        "tap2_step_percent": pa.Column(
            float, pa.Check.gt(0), required=False, description="tap step size for voltage magnitude [%]"
        ),
        "tap2_step_degree": pa.Column(
            float, pa.Check.ge(0), required=False, description="tap step size for voltage angle"
        ),
        "tap2_pos": pa.Column(int, required=False, description="current position of tap changer"),
        "tap2_changer_type": pa.Column(
            float,
            pa.Check.isin(["Ratio", "Symmetrical", "Ideal"]),
            required=False,
            description="specifies the tap changer type",
        ),
        "leakage_resistance_ratio_hv": pa.Column(
            float,
            pa.Check.between(min_value=0, max_value=1),
            required=False,
            description="ratio of transformer short-circuit resistance on HV side (default 0.5)",
        ),  # not in create method call
        "leakage_reactance_ratio_hv": pa.Column(
            float,
            pa.Check.between(min_value=0, max_value=1),
            required=False,
            description="ratio of transformer short-circuit reactance on HV side (default 0.5)",
        ),  # not in create method call
    },
    strict=False,
)

res_schema = pa.DataFrameSchema(
    {
        "p_hv_mw": pa.Column(float, description="active power flow at the high voltage transformer bus [MW]"),
        "q_hv_mvar": pa.Column(float, description="reactive power flow at the high voltage transformer bus [MVar]"),
        "p_lv_mw": pa.Column(float, description="active power flow at the low voltage transformer bus [MW]"),
        "q_lv_mvar": pa.Column(float, description="reactive power flow at the low voltage transformer bus [MVar]"),
        "pl_mw": pa.Column(float, description="active power losses of the transformer [MW]"),
        "ql_mvar": pa.Column(float, description="reactive power consumption of the transformer [Mvar]"),
        "i_hv_ka": pa.Column(float, description="current at the high voltage side of the transformer [kA]"),
        "i_lv_ka": pa.Column(float, description="current at the low voltage side of the transformer [kA]"),
        "vm_hv_pu": pa.Column(float, description="voltage magnitude at the high voltage bus [pu]"),
        "va_hv_degree": pa.Column(float, description="voltage magnitude at the low voltage bus [pu]"),
        "vm_lv_pu": pa.Column(float, description="voltage angle at the high voltage bus [degrees]"),
        "va_lv_degree": pa.Column(float, description="voltage angle at the low voltage bus [degrees]"),
        "loading_percent": pa.Column(float, description="load utilization relative to rated power [%]"),
    },
)

res_schema_3ph = pa.DataFrameSchema(
    {
        "p_a_hv_mw": pa.Column(
            float, description="active power flow at the high voltage transformer bus : Phase A [MW]"
        ),
        "q_a_hv_mvar": pa.Column(
            float, description="reactive power flow at the high voltage transformer bus : Phase A [MVar]"
        ),
        "p_b_hv_mw": pa.Column(
            float, description="active power flow at the high voltage transformer bus : Phase B [MW]"
        ),
        "q_b_hv_mvar": pa.Column(
            float, description="reactive power flow at the high voltage transformer bus : Phase B [MVar]"
        ),
        "p_c_hv_mw": pa.Column(
            float, description="active power flow at the high voltage transformer bus : Phase C [MW]"
        ),
        "q_c_hv_mvar": pa.Column(
            float, description="reactive power flow at the high voltage transformer bus : Phase C [MVar]"
        ),
        "p_a_lv_mw": pa.Column(
            float, description="active power flow at the low voltage transformer bus : Phase A [MW]"
        ),
        "q_a_lv_mvar": pa.Column(
            float, description="reactive power flow at the low voltage transformer bus : Phase A [MVar]"
        ),
        "p_b_lv_mw": pa.Column(
            float, description="active power flow at the low voltage transformer bus : Phase B [MW]"
        ),
        "q_b_lv_mvar": pa.Column(
            float, description="reactive power flow at the low voltage transformer bus : Phase B [MVar]"
        ),
        "p_c_lv_mw": pa.Column(
            float, description="active power flow at the low voltage transformer bus : Phase C [MW]"
        ),
        "q_c_lv_mvar": pa.Column(
            float, description="reactive power flow at the low voltage transformer bus : Phase C [MVar]"
        ),
        "pl_a_mw": pa.Column(float, description="active power losses of the transformer : Phase A [MW]"),
        "ql_a_mvar": pa.Column(float, description="reactive power consumption of the transformer : Phase A [Mvar]"),
        "pl_b_mw": pa.Column(float, description="active power losses of the transformer : Phase B [MW]"),
        "ql_b_mvar": pa.Column(float, description="reactive power consumption of the transformer : Phase B [Mvar]"),
        "pl_c_mw": pa.Column(float, description="active power losses of the transformer : Phase C [MW]"),
        "ql_c_mvar": pa.Column(float, description="reactive power consumption of the transformer : Phase C [Mvar]"),
        "i_a_hv_ka": pa.Column(float, description="current at the high voltage side of the transformer : Phase A [kA]"),
        "i_a_lv_ka": pa.Column(float, description="current at the low voltage side of the transformer : Phase A [kA]"),
        "i_b_hv_ka": pa.Column(float, description="current at the high voltage side of the transformer : Phase B [kA]"),
        "i_b_lv_ka": pa.Column(float, description="current at the low voltage side of the transformer : Phase B [kA]"),
        "i_c_hv_ka": pa.Column(float, description="current at the high voltage side of the transformer : Phase C [kA]"),
        "i_c_lv_ka": pa.Column(float, description="current at the low voltage side of the transformer : Phase C [kA]"),
        "loading_a_percent": pa.Column(float, description="load utilization relative to rated power: Phase A [%]"),
        "loading_b_percent": pa.Column(float, description="load utilization relative to rated power: Phase B [%]"),
        "loading_c_percent": pa.Column(float, description="load utilization relative to rated power: Phase C [%]"),
        "loading_percent": pa.Column(
            float, description="load utilization relative to rated power: Maximum of Phase A, B, C in [%]"
        ),
    },
)

# TODO: was ist mit res_trafo_sc
