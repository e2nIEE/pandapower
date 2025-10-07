import pandas as pd
import pandera.pandas as pa

schema = pa.DataFrameSchema(
    {
        # TODO: in methodcall but not parameter docu: vector_group, vkr0_x, vk0_x, max_loading_percent, ahhh warum gibt es 2 create methoden???
        "name": pa.Column(str, description="name of the transformer"),
        "std_type": pa.Column(str, description="transformer standard type name"),
        "hv_bus": pa.Column(int, pa.Check.ge(0), description="high voltage bus index of the transformer"),
        "mv_bus": pa.Column(int, pa.Check.ge(0), description="medium voltage bus index of the transformer"),
        "lv_bus": pa.Column(int, pa.Check.ge(0), description="low voltage bus index of the transformer"),
        "vn_hv_kv": pa.Column(float, pa.Check.gt(0), description="rated voltage at high voltage bus [kV]"),
        "vn_mv_kv": pa.Column(float, pa.Check.gt(0), description="rated voltage at medium voltage bus [kV]"),
        "vn_lv_kv": pa.Column(float, pa.Check.gt(0), description="rated voltage at low voltage bus [kV]"),
        "sn_hv_mva": pa.Column(float, pa.Check.gt(0), description="rated apparent power on high voltage side [kVA]"),
        "sn_mv_mva": pa.Column(float, pa.Check.gt(0), description="rated apparent power on medium voltage side [kVA]"),
        "sn_lv_mva": pa.Column(float, pa.Check.gt(0), description="rated apparent power on low voltage side [kVA]"),
        "vk_hv_percent": pa.Column(
            float, pa.Check.gt(0), description="short circuit voltage from high to medium voltage [%]"
        ),
        "vk_mv_percent": pa.Column(
            float, pa.Check.gt(0), description="short circuit voltage from medium to low voltage [%]"
        ),
        "vk_lv_percent": pa.Column(
            float, pa.Check.gt(0), description="short circuit voltage from high to low voltage [%]"
        ),
        "vkr_hv_percent": pa.Column(
            float, pa.Check.ge(0), description="real part of short circuit voltage from high to medium voltage [%]"
        ),
        "vkr_mv_percent": pa.Column(
            float, pa.Check.ge(0), description="real part of short circuit voltage from medium to low voltage [%]"
        ),
        "vkr_lv_percent": pa.Column(
            float, pa.Check.ge(0), description="real part of short circuit voltage from high to low voltage [%]"
        ),
        "pfe_kw": pa.Column(float, description="iron losses [kW]"),
        "i0_percent": pa.Column(float, description="open loop losses [%]"),
        "shift_mv_degree": pa.Column(float, description="transformer phase shift angle at the MV side"),
        "shift_lv_degree": pa.Column(float, description="transformer phase shift angle at the LV side"),
        "tap_side": pa.Column(
            str,
            pa.Check.isin(["hv", "mv", "lv"]),
            description="defines if tap changer is positioned on high- medium- or low voltage side",
        ),
        "tap_neutral": pa.Column(int, description=""),
        "tap_min": pa.Column(int, description="minimum tap position"),
        "tap_max": pa.Column(int, description="maximum tap position"),
        "tap_step_percent": pa.Column(float, pa.Check.gt(0), description="tap step size [%]"),
        "tap_step_degree": pa.Column(float, description="tap step size for voltage angle"),
        "tap_at_star_point": pa.Column(
            bool, description="whether the tap changer is modelled at terminal or at star point"
        ),
        "tap_pos": pa.Column(float, description="current position of tap changer"),
        "tap_changer_type": pa.Column(
            str,
            pa.Check.isin(["Ratio", "Symmetrical", "Ideal", "Tabular"]),
            description="specifies the tap changer type",
        ),
        "tap_dependency_table": pa.Column(
            bool,
            description="whether the transformer parameters (voltage ratio, angle, impedance) are adjusted dependent on the tap position of the transformer",
        ),
        "id_characteristic_table": pa.Column(
            pd.Int64Dtype,
            pa.Check.ge(0),
            description="references the id_characteristic index from the trafo_characteristic_table",
        ),
        "in_service": pa.Column(bool, description="specifies if the transformer is in service."),
    },
    strict=False,
)

res_schema = pa.DataFrameSchema(
    {
        "p_hv_mw": pa.Column(float, description="active power flow at the high voltage transformer bus [MW]"),
        "q_hv_mvar": pa.Column(float, description="reactive power flow at the high voltage transformer bus [kVar]"),
        "p_mv_mw": pa.Column(float, description="active power flow at the medium voltage transformer bus [MW]"),
        "q_mv_mvar": pa.Column(float, description="reactive power flow at the medium voltage transformer bus [kVar]"),
        "p_lv_mw": pa.Column(float, description="active power flow at the low voltage transformer bus [MW]"),
        "q_lv_mvar": pa.Column(float, description="reactive power flow at the low voltage transformer bus [kVar]"),
        "pl_mw": pa.Column(float, description="active power losses of the transformer [MW]"),
        "ql_mvar": pa.Column(float, description="reactive power consumption of the transformer [Mvar]"),
        "i_hv_ka": pa.Column(float, description="current at the high voltage side of the transformer [kA]"),
        "i_mv_ka": pa.Column(float, description="current at the medium voltage side of the transformer [kA]"),
        "i_lv_ka": pa.Column(float, description="current at the low voltage side of the transformer [kA]"),
        "vm_hv_pu": pa.Column(float, description="voltage magnitude at the high voltage bus [pu]"),
        "va_hv_degree": pa.Column(float, description="voltage magnitude at the medium voltage bus [pu]"),
        "vm_mv_pu": pa.Column(float, description="voltage magnitude at the low voltage bus [pu]"),
        "va_mv_degree": pa.Column(float, description="voltage angle at the high voltage bus [degrees]"),
        "vm_lv_pu": pa.Column(float, description="voltage angle at the medium voltage bus [degrees]"),
        "va_lv_degree": pa.Column(float, description="voltage angle at the low voltage bus [degrees]"),
        "va_internal_degree": pa.Column(float, description=""),  # TODO: missing in docu
        "vm_internal_pu": pa.Column(float, description=""),  # TODO: missing in docu
        "loading_percent": pa.Column(float, description="transformer utilization [%]"),
    },
)
