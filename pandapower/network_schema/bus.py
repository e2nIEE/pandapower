import pandera.pandas as pa

schema = pa.DataFrameSchema(  # in methodcall but not parameter docu: geodata, coords
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
