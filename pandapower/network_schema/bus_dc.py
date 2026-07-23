import pandas as pd
import pandera.pandas as pa

from pandapower.network_schema.tools.validation.group_dependency import create_column_dependency_checks_from_metadata
from pandapower.network_schema.tools.validation.column_condition import create_lower_equals_column_check
from pandapower.network_schema.tools.validation.column_condition import create_lower_equals_column_check

_bus_dc_columns = {
    "name": pa.Column(pd.StringDtype, nullable=True, required=False, description="name of the dc bus"),
    "vn_kv": pa.Column(float, pa.Check.gt(0), description="rated voltage of the dc bus [kV]"),
    "type": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="type variable to classify buses",
        metadata={"default": "b"},
    ),
    "zone": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="can be used to group dc buses, for example network groups / regions",
    ),
    "in_service": pa.Column(bool, description="specifies if the dc bus is in service", metadata={"default": True}),
    "geo": pa.Column(pd.StringDtype, nullable=True, required=False, description="geojson.Point as object or string"),
    "max_vm_pu": pa.Column(
        float,
        checks=[pa.Check.gt(0), pa.Check.le(2)],
        nullable=True,
        required=False,
        description="Maximum dc bus voltage in p.u. - necessary for OPF",
        metadata={"opf": True, "default": 2.0},
    ),
    "min_vm_pu": pa.Column(
        float,
        pa.Check.ge(0),
        nullable=True,
        required=False,
        description="Minimum dc bus voltage in p.u. - necessary for OPF",
        metadata={"opf": True, "default": 0.0},
    ),
}
bus_dc_schema = pa.DataFrameSchema(
    _bus_dc_columns,
    name="bus_dc",
    checks=create_lower_equals_column_check(first_element="min_vm_pu", second_element="max_vm_pu"),
    strict=False,
)


res_bus_dc_schema = pa.DataFrameSchema(
    {
        "vm_pu": pa.Column(float, nullable=True, description="voltage magnitude [p.u]"),
        "p_mw": pa.Column(float, nullable=True, description="resulting active power demand [MW]"),
    },
    name="res_bus_dc",
    strict=False,
)
