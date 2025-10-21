import pandera.pandas as pa
import pandas as pd

bus_dc_schema = pa.DataFrameSchema(
    {
        "name": pa.Column(pd.StringDtype, nullable=True, required=False, description="name of the dc bus"),
        "vn_kv": pa.Column(float, pa.Check.gt(0), description="rated voltage of the dc bus [kV]"),
        "type": pa.Column(str, required=False, description="type variable to classify buses"),
        "zone": pa.Column(
            str, required=False, description="can be used to group dc buses, for example network groups / regions"
        ),
        "in_service": pa.Column(bool, description="specifies if the dc bus is in service"),
        "geo": pa.Column(
            pd.StringDtype, nullable=True, required=False, description="geojson.Point as object or string"
        ),
        "max_vm_pu": pa.Column(
            float, description="Maximum dc bus voltage in p.u. - necessary for OPF", metadata={"opf": True}
        ),
        "min_vm_pu": pa.Column(
            float, description="Minimum dc bus voltage in p.u. - necessary for OPF", metadata={"opf": True}
        ),
    },
    strict=False,
)


res_bus_dc_schema = pa.DataFrameSchema(
    {
        "vm_pu": pa.Column(float, description="voltage magnitude [p.u]"),
        "p_mw": pa.Column(float, description="resulting active power demand [MW]"),
    },
)
