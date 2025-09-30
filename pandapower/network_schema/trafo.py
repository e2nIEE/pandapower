import pandera.pandas as pa

schema = pa.DataFrameSchema(  # in methodcall but not parameter docu: xn_ohm, pt_percent
    {
        "name": pa.Column(str),
        "std_type": pa.Column(str),
        "hv_bus": pa.Column(int, pa.Check.ge(0)),
        "lv_bus": pa.Column(int, pa.Check.ge(0)),
        "sn_mva": pa.Column(float, pa.Check.gt(0)),
        "vn_hv_kv": pa.Column(float, pa.Check.gt(0)),
        "vn_lv_kv": pa.Column(float, pa.Check.gt(0)),
        "vk_percent": pa.Column(float, pa.Check.gt(0)),
        "vkr_percent": pa.Column(float, pa.Check.ge(0)),
        "pfe_kw": pa.Column(float, pa.Check.ge(0)),
        "i0_percent": pa.Column(float, pa.Check.ge(0)),
        "vk0_percent": pa.Column(float, pa.Check.ge(0), required=False),
        "vkr0_percent": pa.Column(float, pa.Check.ge(0), required=False),
        "mag0_percent": pa.Column(float, pa.Check.ge(0), required=False),
        "mag0_rx": pa.Column(float, required=False),
        "si0_hv_partial": pa.Column(float, pa.Check.ge(0), required=False),
        "vector_group": pa.Column(str, required=False),
        "shift_degree": pa.Column(float),
        "tap_side": pa.Column(str, pa.Check.isin(["hv", "lv"])),
        "tap_neutral": pa.Column(int),
        "tap_min": pa.Column(int),
        "tap_max": pa.Column(int),
        "tap_step_percent": pa.Column(float, pa.Check.gt(0)),
        "tap_step_degree": pa.Column(float, pa.Check.ge(0), nullable=True),
        "tap_pos": pa.Column(int),
        "tap_changer_type": pa.Column(str, pa.Check.isin(["Ratio", "Symmetrical", "Ideal", "Tabular"])),
        "tap_dependency_table": pa.Column(bool, required=False),
        "id_characteristic_table": pa.Column(int, pa.Check.ge(0), required=False),
        "max_loading_percent": pa.Column(int, required=False),
        "parallel": pa.Column(int, pa.Check.gt(0)),
        "df": pa.Column(float, pa.Check.between(min_value=0, max_value=1)),
        "in_service": pa.Column(bool),
        "oltc": pa.Column(bool, required=False),
        "power_station_unit": pa.Column(bool, required=False),  # not in create method call
        "tap2_side": pa.Column(int, pa.Check.isin(["hv", "lv"]), required=False),
        "tap2_neutral": pa.Column(int, required=False),
        "tap2_min": pa.Column(int, required=False),
        "tap2_max": pa.Column(int, required=False),
        "tap2_step_percent": pa.Column(float, pa.Check.gt(0), required=False),
        "tap2_step_degree": pa.Column(float, pa.Check.ge(0), required=False),
        "tap2_pos": pa.Column(int, required=False),
        "tap2_changer_type": pa.Column(float, pa.Check.isin(["Ratio", "Symmetrical", "Ideal"]), required=False),
        "leakage_resistance_ratio_hv": pa.Column(
            float, pa.Check.between(min_value=0, max_value=1), required=False
        ),  # not in create method call
        "leakage_reactance_ratio_hv": pa.Column(
            float, pa.Check.between(min_value=0, max_value=1), required=False
        ),  # not in create method call
    },
    strict=False,
)
