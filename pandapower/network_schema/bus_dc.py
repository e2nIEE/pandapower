import pandera.pandas as pa

schema = pa.DataFrameSchema(
    {
        "name": pa.Column(str, required=False, description="name of the dc bus"),
        "vn_kv": pa.Column(float, pa.Check.gt(0), description="rated voltage of the dc bus [kV]"),
        "type": pa.Column(
            str, pa.Check.isin(["n", "b", "m"]), required=False, description="type variable to classify buses"
        ),
        "zone": pa.Column(
            str, required=False, description="can be used to group dc buses, for example network groups / regions"
        ),
        "in_service": pa.Column(bool, description="specifies if the dc bus is in service"),
        "geo": pa.Column(str, description="geojson.Point as object or string"),

        # neu (Kommentar kann nach kontrolle gelöscht werden)
        "max_vm_pu": pa.Column(float, description="Maximum dc bus voltage in p.u. - necessary for OPF", metadata={"opf": True}),
        "min_vm_pu": pa.Column(float, description="Minimum dc bus voltage in p.u. - necessary for OPF", metadata={"opf": True})
    },
    strict=False,
)


res_schema = pa.DataFrameSchema(
    {
        "vm_pu": pa.Column(float, description="voltage magnitude [p.u]"),
        "p_mw": pa.Column(float, description="resulting active power demand [MW]"),
    },
)
