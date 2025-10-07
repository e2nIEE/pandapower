import pandera.pandas as pa

schema = pa.DataFrameSchema(  # in methodcall but not parameter docu: geodata, coords #TODO:
    {
        "name": pa.Column(str, description="name of the bus"),
        "vn_kv": pa.Column(float, pa.Check.gt(0), description="rated voltage of the bus [kV]"),
        "type": pa.Column(str, pa.Check.isin(["n", "b", "m"]), description="type variable to classify buses"),
        "zone": pa.Column(
            str, nullable=True, description="can be used to group buses, for example network groups / regions"
        ),
        "max_vm_pu": pa.Column(
            float, pa.Check.gt(0), required=False, description="Maximum voltage", metadata={"opf": True}
        ),
        "min_vm_pu": pa.Column(
            float, pa.Check.gt(0), required=False, description="Minimum voltage", metadata={"opf": True}
        ),
        "in_service": pa.Column(bool, description="specifies if the bus is in service."),
        "geo": pa.Column(str, description="geojson.Point as object or string"),
    },
    strict=False,
)


res_schema = pa.DataFrameSchema(
    {
        "vm_pu": pa.Column(float, description="voltage magnitude [p.u]"),
        "va_degree": pa.Column(float, description="voltage angle [degree]"),
        "p_mw": pa.Column(float, description="resulting active power demand [MW]"),
        "q_mvar": pa.Column(float, description="resulting reactive power demand [Mvar]"),
    },
)

res_schema_3ph = pa.DataFrameSchema(
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
        # "unbalance_percent": pa.Column(
        #     float, description="unbalance in percent defined as the ratio of V2 and V1 according to IEC 62749"
        # ),  #TODO: was only in docu
    },
)

# TODO: was ist mit res_bus_est und res_bus_sc
