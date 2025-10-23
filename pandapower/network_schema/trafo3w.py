import pandas as pd
import pandera.pandas as pa

from network_schema.tools import validate_column_group_dependency

_trafo3w_columns = {
    # TODO: in methodcall but not parameter docu: vector_group, vkr0_x, vk0_x, max_loading_percent, ahhh warum gibt es 2 create methoden???
    "name": pa.Column(pd.StringDtype, nullable=True, required=False, description="name of the transformer"),
    "std_type": pa.Column(str, nullable=True, required=False, description="transformer standard type name"),
    "hv_bus": pa.Column(
        int,
        pa.Check.ge(0),
        description="high voltage bus index of the transformer",
        metadata={"foreign_key": "bus.index"},
    ),
    "mv_bus": pa.Column(
        int,
        pa.Check.ge(0),
        description="medium voltage bus index of the transformer",
        metadata={"foreign_key": "bus.index"},
    ),
    "lv_bus": pa.Column(
        int,
        pa.Check.ge(0),
        description="low voltage bus index of the transformer",
        metadata={"foreign_key": "bus.index"},
    ),
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
    "shift_mv_degree": pa.Column(
        float, nullable=True, required=False, description="transformer phase shift angle at the MV side"
    ),
    "shift_lv_degree": pa.Column(
        float, nullable=True, required=False, description="transformer phase shift angle at the LV side"
    ),
    "tap_side": pa.Column(
        str,
        pa.Check.isin(["hv", "mv", "lv"]),
        nullable=True,
        required=False,
        description="defines if tap changer is positioned on high- medium- or low voltage side",
    ),
    "tap_neutral": pa.Column(float, nullable=True, required=False, description=""),  # TODO: different type in docu
    "tap_min": pa.Column(
        float, nullable=True, required=False, description="minimum tap position"
    ),  # TODO: different type in docu
    "tap_max": pa.Column(
        float, nullable=True, required=False, description="maximum tap position"
    ),  # TODO: different type in docu
    "tap_step_percent": pa.Column(
        float, pa.Check.gt(0), nullable=True, required=False, description="tap step size [%]"
    ),
    "tap_step_degree": pa.Column(
        float, nullable=True, required=False, description="tap step size for voltage angle"
    ),
    "tap_at_star_point": pa.Column(
        bool,
        nullable=True,
        required=False,
        description="whether the tap changer is modelled at terminal or at star point",
    ),
    "tap_pos": pa.Column(float, nullable=True, required=False, description="current position of tap changer"),
    "tap_changer_type": pa.Column(
        str,
        pa.Check.isin(["Ratio", "Symmetrical", "Ideal", "Tabular"]),
        nullable=True,
        required=False,
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
        int,  # TODO: guess from trafo2w
        nullable=True,
        required=False,
        description="",
    ),  # TODO: not in docu only create
    "in_service": pa.Column(bool, description="specifies if the transformer is in service."),
}
tap_columns = ["tap_pos", "tap_neutral", "tap_side", "tap_step_percent", "tap_step_degree"]  #TODO: which ones?
trafo3w_schema = pa.DataFrameSchema(
    _trafo3w_columns,
    checks=[pa.Check(
        validate_column_group_dependency(tap_columns),
        error=f"Tap configuration columns have dependency violations. Please ensure {tap_columns} are present in the dataframe.",
    ),],
    strict=False,
)

res_trafo3w_schema = pa.DataFrameSchema(
    {
        "p_hv_mw": pa.Column(
            float, nullable=True, description="active power flow at the high voltage transformer bus [MW]"
        ),
        "q_hv_mvar": pa.Column(
            float, nullable=True, description="reactive power flow at the high voltage transformer bus [kVar]"
        ),
        "p_mv_mw": pa.Column(
            float, nullable=True, description="active power flow at the medium voltage transformer bus [MW]"
        ),
        "q_mv_mvar": pa.Column(
            float, nullable=True, description="reactive power flow at the medium voltage transformer bus [kVar]"
        ),
        "p_lv_mw": pa.Column(
            float, nullable=True, description="active power flow at the low voltage transformer bus [MW]"
        ),
        "q_lv_mvar": pa.Column(
            float, nullable=True, description="reactive power flow at the low voltage transformer bus [kVar]"
        ),
        "pl_mw": pa.Column(float, nullable=True, description="active power losses of the transformer [MW]"),
        "ql_mvar": pa.Column(float, nullable=True, description="reactive power consumption of the transformer [Mvar]"),
        "i_hv_ka": pa.Column(
            float, nullable=True, description="current at the high voltage side of the transformer [kA]"
        ),
        "i_mv_ka": pa.Column(
            float, nullable=True, description="current at the medium voltage side of the transformer [kA]"
        ),
        "i_lv_ka": pa.Column(
            float, nullable=True, description="current at the low voltage side of the transformer [kA]"
        ),
        "vm_hv_pu": pa.Column(float, nullable=True, description="voltage magnitude at the high voltage bus [pu]"),
        "va_hv_degree": pa.Column(float, nullable=True, description="voltage magnitude at the medium voltage bus [pu]"),
        "vm_mv_pu": pa.Column(float, nullable=True, description="voltage magnitude at the low voltage bus [pu]"),
        "va_mv_degree": pa.Column(float, nullable=True, description="voltage angle at the high voltage bus [degrees]"),
        "vm_lv_pu": pa.Column(float, nullable=True, description="voltage angle at the medium voltage bus [degrees]"),
        "va_lv_degree": pa.Column(float, nullable=True, description="voltage angle at the low voltage bus [degrees]"),
        "va_internal_degree": pa.Column(float, nullable=True, description=""),  # TODO: missing in docu
        "vm_internal_pu": pa.Column(float, nullable=True, description=""),  # TODO: missing in docu
        "loading_percent": pa.Column(float, nullable=True, description="transformer utilization [%]"),
    },
    strict=False,
)

res_trafo3w_sc_schema = pa.DataFrameSchema(
    {
        "ikss_hv_ka": pa.Column(
            float,
            nullable=True,
            description="magnitude of the initial SC current at the high voltage side of the transformer [kA]",
        ),
        "ikss_mv_ka": pa.Column(
            float,
            nullable=True,
            description="magnitude of the initial SC current at the medium voltage side of the transformer [kA]",
        ),
        "ikss_lv_ka": pa.Column(
            float,
            nullable=True,
            description="magnitude of the initial SC current at the low voltage side of the transformer [kA]",
        ),
    },
    strict=False,
)
