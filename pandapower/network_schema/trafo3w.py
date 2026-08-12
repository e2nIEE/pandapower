import pandas as pd
import pandera.pandas as pa

from pandapower.network_schema.tools.validation.group_dependency import create_column_group_dependency_validation_func
from pandapower.network_schema.tools.validation.column_condition import create_lower_than_column_check

_trafo3w_columns = {
    "name": pa.Column(
        pd.StringDtype, nullable=True, required=False, description="name of the transformer", metadata={"cim": True}
    ),
    "std_type": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="transformer standard type name",
        metadata={"cim": True},
    ),
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
        float,
        pa.Check.gt(0),
        description="short circuit voltage from high to medium voltage [%]",
        metadata={"sc": True},
    ),
    "vk_mv_percent": pa.Column(
        float, pa.Check.gt(0), description="short circuit voltage from medium to low voltage [%]", metadata={"sc": True}
    ),
    "vk_lv_percent": pa.Column(
        float, pa.Check.gt(0), description="short circuit voltage from high to low voltage [%]", metadata={"sc": True}
    ),
    "vkr_hv_percent": pa.Column(
        float,
        pa.Check.ge(0),
        description="real part of short circuit voltage from high to medium voltage [%]",
        metadata={"sc": True},
    ),
    "vkr_mv_percent": pa.Column(
        float,
        pa.Check.ge(0),
        description="real part of short circuit voltage from medium to low voltage [%]",
        metadata={"sc": True},
    ),
    "vkr_lv_percent": pa.Column(
        float,
        pa.Check.ge(0),
        description="real part of short circuit voltage from high to low voltage [%]",
        metadata={"sc": True},
    ),
    "pfe_kw": pa.Column(float, description="iron losses [kW]"),
    "i0_percent": pa.Column(float, description="open loop losses [%]"),
    "shift_mv_degree": pa.Column(
        float, description="transformer phase shift angle at the MV side", metadata={"default": 0.0}
    ),
    "shift_lv_degree": pa.Column(
        float, description="transformer phase shift angle at the LV side", metadata={"default": 0.0}
    ),
    "tap_side": pa.Column(
        pd.StringDtype,
        pa.Check.isin(["hv", "mv", "lv"]),
        nullable=True,
        required=False,
        description="defines if tap changer is positioned on high- medium- or low voltage side",
        metadata={"cim": True},
    ),
    "tap_neutral": pa.Column(float, nullable=True, required=False, description="", metadata={"cim": True}),
    "tap_min": pa.Column(
        float, nullable=True, required=False, description="minimum tap position", metadata={"cim": True}
    ),
    "tap_max": pa.Column(
        float, nullable=True, required=False, description="maximum tap position", metadata={"cim": True}
    ),
    "tap_step_percent": pa.Column(
        float, pa.Check.gt(0), nullable=True, required=False, description="tap step size [%]", metadata={"cim": True}
    ),
    "tap_step_degree": pa.Column(
        float, nullable=True, required=False, description="tap step size for voltage angle", metadata={"cim": True}
    ),
    "tap_at_star_point": pa.Column(
        pd.BooleanDtype,
        nullable=True,
        required=False,
        description="whether the tap changer is modelled at terminal or at star point",
        metadata={"default": False, "cim": True},
    ),
    "tap_pos": pa.Column(
        float, nullable=True, required=False, description="current position of tap changer", metadata={"cim": True}
    ),
    "tap_changer_type": pa.Column(
        pd.StringDtype,
        pa.Check.isin(["Ratio", "Symmetrical", "Ideal", "Tabular"]),
        nullable=True,
        required=False,
        description="specifies the tap changer type",
    ),
    "tap_dependency_table": pa.Column(
        pd.BooleanDtype,
        nullable=True,
        required=False,
        description="whether the transformer parameters (voltage ratio, angle, impedance) are adjusted dependent on the tap position of the transformer",
        metadata={"tdt": True, "cim": True},
    ),
    "id_characteristic_table": pa.Column(
        pd.Int64Dtype,
        pa.Check.ge(0),
        nullable=True,
        required=False,
        description="references the id_characteristic index from the trafo_characteristic_table",
        metadata={"tdt": True, "cim": True},
    ),
    "max_loading_percent": pa.Column(
        float,
        nullable=True,
        required=False,
        description="maximum current loading (only needed for OPF)",
        metadata={"opf": True},
    ),
    "vector_group": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="vector group of the 3w-transformer",
        metadata={"sc": True, "cim": True},
    ),
    "vkr0_x": pa.Column(
        float,
        nullable=True,
        required=False,
        description="",
    ),
    "vk0_x": pa.Column(
        float,
        nullable=True,
        required=False,
        description="",
    ),
    "in_service": pa.Column(
        bool, description="specifies if the transformer is in service.", metadata={"default": True}
    ),
    "origin_id": pa.Column(
        pd.StringDtype, nullable=True, required=False, description="element rdfId from CIM", metadata={"cim": True, "doc": False}
    ),
    "origin_class": pa.Column(
        pd.StringDtype, nullable=True, required=False, description="origin_class rdfId from CIM", metadata={"cim": True, "doc": False}
    ),
    "terminal_hv": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="terminal_hv from converter, not relevant for calculations",
        metadata={"cim": True, "doc": False},
    ),
    "terminal_mv": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="terminal_mv from converter, not relevant for calculations",
        metadata={"cim": True, "doc": False},
    ),
    "terminal_lv": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="terminal_lv from converter, not relevant for calculations",
        metadata={"cim": True, "doc": False},
    ),
    "PowerTransformerEnd_id_hv": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="PowerTransformerEnd_id_hv from converter, not relevant for calculations",
        metadata={"cim": True, "doc": False},
    ),
    "PowerTransformerEnd_id_mv": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="PowerTransformerEnd_id_mv from converter, not relevant for calculations",
        metadata={"cim": True, "doc": False},
    ),
    "PowerTransformerEnd_id_lv": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="PowerTransformerEnd_id_lv from converter, not relevant for calculations",
        metadata={"cim": True, "doc": False},
    ),
    "tapchanger_class": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="tapchanger_class from converter, not relevant for calculations",
        metadata={"cim": True, "doc": False},
    ),
    "tapchanger_id": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="tapchanger_id from converter, not relevant for calculations",
        metadata={"cim": True, "doc": False},
    ),
    "description": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="description from converter, not relevant for calculations",
        metadata={"cim": True, "doc": False},
    ),
    "OperationalLimitType.limitType_hv": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="OperationalLimitType.limitType_hv from converter, not relevant for calculations",
        metadata={"cim": True, "doc": False},
    ),
    "OperationalLimitType.limitType_mv": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="OperationalLimitType.limitType_mv from converter, not relevant for calculations",
        metadata={"cim": True, "doc": False},
    ),
    "OperationalLimitType.limitType_lv": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="OperationalLimitType.limitType_lv from converter, not relevant for calculations",
        metadata={"cim": True, "doc": False},
    ),
    "vk0_hv_percent": pa.Column(  # TODO: check if really only cim
        float,
        nullable=True,
        required=False,
        description="vk0_hv_percent from converter, not relevant for calculations",
        metadata={"cim": True, "doc": False},
    ),
    "vk0_mv_percent": pa.Column(  # TODO: check if really only cim
        float,
        nullable=True,
        required=False,
        description="vk0_mv_percent from converter, not relevant for calculations",
        metadata={"cim": True, "doc": False},
    ),
    "vk0_lv_percent": pa.Column(  # TODO: check if really only cim
        float,
        nullable=True,
        required=False,
        description="vk0_lv_percent from converter, not relevant for calculations",
        metadata={"cim": True, "doc": False},
    ),
    "vkr0_hv_percent": pa.Column(  # TODO: check if really only cim
        float,
        nullable=True,
        required=False,
        description="vkr0_hv_percent from converter, not relevant for calculations",
        metadata={"cim": True, "doc": False},
    ),
    "vkr0_mv_percent": pa.Column(  # TODO: check if really only cim
        float,
        nullable=True,
        required=False,
        description="vkr0_mv_percent from converter, not relevant for calculations",
        metadata={"cim": True, "doc": False},
    ),
    "vkr0_lv_percent": pa.Column(  # TODO: check if really only cim
        float,
        nullable=True,
        required=False,
        description="vkr0_lv_percent from converter, not relevant for calculations",
        metadata={"cim": True, "doc": False},
    ),
    "CurrentLimit.value_hv": pa.Column(
        float,
        nullable=True,
        required=False,
        description="CurrentLimit.value_hv from converter, not relevant for calculations",
        metadata={"cim": True, "doc": False},
    ),
    "CurrentLimit.value_mv": pa.Column(
        float,
        nullable=True,
        required=False,
        description="CurrentLimit.value_mv from converter, not relevant for calculations",
        metadata={"cim": True, "doc": False},
    ),
    "CurrentLimit.value_lv": pa.Column(
        float,
        nullable=True,
        required=False,
        description="CurrentLimit.value_lv from converter, not relevant for calculations",
        metadata={"cim": True, "doc": False},
    ),
    "OperationalLimitType.acceptableDuration_hv": pa.Column(
        float,
        nullable=True,
        required=False,
        description="OperationalLimitType.acceptableDuration_hv from converter, not relevant for calculations",
        metadata={"cim": True, "doc": False},
    ),
    "OperationalLimitType.acceptableDuration_mv": pa.Column(
        float,
        nullable=True,
        required=False,
        description="OperationalLimitType.acceptableDuration_mv from converter, not relevant for calculations",
        metadata={"cim": True, "doc": False},
    ),
    "OperationalLimitType.acceptableDuration_lv": pa.Column(
        float,
        nullable=True,
        required=False,
        description="OperationalLimitType.acceptableDuration_lv from converter, not relevant for calculations",
        metadata={"cim": True, "doc": False},
    ),
    "power_station_unit": pa.Column(  # TODO: check if really only cim
        pd.BooleanDtype,
        nullable=True,
        required=False,
        description="power_station_unit from converter, not relevant for calculations",
        metadata={"cim": True, "doc": False},
    ),
}
# FIXME: either tap_step_percent or tap_step_degree for each row
tap_columns = ["tap_pos", "tap_neutral", "tap_side"]  # , "tap_step_percent", "tap_step_degree"]
trafo3w_checks = [
    pa.Check(
        create_column_group_dependency_validation_func(tap_columns),
        error=f"trafo3w tap configuration columns have dependency violations. Please ensure {tap_columns} are present in the dataframe.",
    ),
    create_lower_than_column_check(first_element="vkr_hv_percent", second_element="vk_hv_percent"),
    create_lower_than_column_check(first_element="vkr_mv_percent", second_element="vk_mv_percent"),
    create_lower_than_column_check(first_element="vkr_lv_percent", second_element="vk_lv_percent"),
]
trafo3w_schema = pa.DataFrameSchema(
    _trafo3w_columns,
    checks=trafo3w_checks,
    name="trafo3w",
    strict=False,
)

res_trafo3w_schema = res_trafo3w_est_schema = pa.DataFrameSchema(
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
        "va_hv_degree": pa.Column(float, nullable=True, description="voltage angle at the high voltage bus [degrees]"),
        "vm_mv_pu": pa.Column(float, nullable=True, description="voltage magnitude at the medium voltage bus [pu]"),
        "va_mv_degree": pa.Column(
            float, nullable=True, description="voltage angle at the medium voltage bus [degrees]"
        ),
        "vm_lv_pu": pa.Column(float, nullable=True, description="voltage magnitude at the low voltage bus [pu]"),
        "va_lv_degree": pa.Column(float, nullable=True, description="voltage angle at the low voltage bus [degrees]"),
        "va_internal_degree": pa.Column(float, nullable=True, description="voltage angle at internal bus"),
        "vm_internal_pu": pa.Column(float, nullable=True, description="voltage magnitude at internal bus"),
        "loading_percent": pa.Column(float, nullable=True, description="transformer utilization [%]"),
    },
    name="res_trafo3w",
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
    name="res_trafo3w_sc",
    strict=False,
)
