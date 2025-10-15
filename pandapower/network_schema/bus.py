import pandera.pandas as pa

bus_schema = pa.DataFrameSchema(
    {
        "name": pa.Column(str, required=False, description="name of the bus"),
        "vn_kv": pa.Column(float, pa.Check.gt(0), description="rated voltage of the bus [kV]"),
        "type": pa.Column(
            str, pa.Check.isin(["n", "b", "m"]), required=False, description="type variable to classify buses"
        ),
        "zone": pa.Column(
            str, required=False, description="can be used to group buses, for example network groups / regions"
        ),
        "max_vm_pu": pa.Column(
            float, pa.Check.gt(0), required=False, description="Maximum voltage", metadata={"opf": True}
        ),
        "min_vm_pu": pa.Column(
            float, pa.Check.gt(0), required=False, description="Minimum voltage", metadata={"opf": True}
        ),
        "in_service": pa.Column(bool, description="specifies if the bus is in service."),
        "geo": pa.Column(str, nullable=True, required=False, description="geojson.Point as object or string"),
    },
    strict=False,
)


res_bus_schema = res_bus_est_schema = pa.DataFrameSchema(
    {
        "vm_pu": pa.Column(float, description="voltage magnitude [p.u]"),
        "va_degree": pa.Column(float, description="voltage angle [degree]"),
        "p_mw": pa.Column(float, description="resulting active power demand [MW]"),
        "q_mvar": pa.Column(float, description="resulting reactive power demand [Mvar]"),
    },
)

res_bus_3ph_schema = pa.DataFrameSchema(
    {
        "vm_a_pu": pa.Column(float, description="voltage magnitude:Phase A [p.u]"),
        "va_a_degree": pa.Column(float, description="voltage angle:Phase A [degree]"),
        "vm_b_pu": pa.Column(float, description="voltage magnitude:Phase B [p.u]"),
        "va_b_degree": pa.Column(float, description="voltage angle:Phase B [degree]"),
        "vm_c_pu": pa.Column(float, description="voltage magnitude:Phase C [p.u]"),
        "va_c_degree": pa.Column(float, description="voltage angle:Phase C [degree]"),
        "p_a_mw": pa.Column(float, description="resulting active power demand:Phase A [MW]"),
        "q_a_mvar": pa.Column(float, description="resulting reactive power demand:Phase A [Mvar]"),
        "p_b_mw": pa.Column(float, description="resulting active power demand:Phase B [MW]"),
        "q_b_mvar": pa.Column(float, description="resulting reactive power demand:Phase B [Mvar]"),
        "p_c_mw": pa.Column(float, description="resulting active power demand:Phase C [MW]"),
        "q_c_mvar": pa.Column(float, description="resulting reactive power demand:Phase C [Mvar]"),
        "unbalance_percent": pa.Column(
            float, description="unbalance in percent defined as the ratio of V2 and V1 according to IEC 62749"
        ),
    },
)


res_bus_sc_schema = pa.DataFrameSchema(
    {
        "ikss_ka": pa.Column(float, description="initial short-circuit current value [kA]"),
        "skss_mw": pa.Column(float, description="initial short-circuit power [MW]"),
        "ip_ka": pa.Column(float, description="peak value of the short-circuit current [kA]"),
        "ith_ka": pa.Column(float, description="equivalent thermal short-circuit current [kA]"),
        "rk_ohm": pa.Column(
            float, description="resistive part of equiv. (positive/negative sequence) SC impedance [Ohm]"
        ),
        "xk_ohm": pa.Column(
            float, description="reactive part of equiv. (positive/negative sequence) SC impedance [Ohm]"
        ),
        "rk0_ohm": pa.Column(float, description="resistive part of equiv. (zero sequence) SC impedance [Ohm]"),
        "xk0_ohm": pa.Column(float, description="reactive part of equiv. (zero sequence) SC impedance [Ohm]"),
    },
)
