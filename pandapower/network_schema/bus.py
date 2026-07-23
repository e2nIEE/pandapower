import pandas as pd
import pandera.pandas as pa

from pandapower.network_schema.tools.validation.column_condition import create_lower_equals_column_check

_bus_columns = {
    "name": pa.Column(pd.StringDtype, nullable=True, required=True, description="name of the bus"),
    "vn_kv": pa.Column(float, pa.Check.gt(0), description="rated voltage of the bus [kV]"),
    "type": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="type variable to classify buses",
        metadata={"cim": True, "default": "b"},
    ),
    "zone": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="can be used to group buses, for example network groups / regions",
        metadata={"cim": True},
    ),
    "max_vm_pu": pa.Column(
        float,
        checks=[pa.Check.gt(0), pa.Check.le(2)],
        nullable=False,
        required=False,
        description="Maximum voltage",
        metadata={"opf": True, "default": 2.0},
    ),
    "min_vm_pu": pa.Column(
        float,
        pa.Check.ge(0),
        nullable=False,
        required=False,
        description="Minimum voltage",
        metadata={"opf": True, "default": 0.0},
    ),
    "in_service": pa.Column(bool, description="specifies if the bus is in service.", metadata={"default": True}),
    "geo": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="geojson.Point as object or string",
        metadata={"cim": True},
    ),
    "origin_id": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="element rdfId from CIM",
        metadata={"cim": True, "doc": False},
    ),
    "origin_class": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="origin_class rdfId from CIM",
        metadata={"cim": True, "doc": False},
    ),
    "origin_profile": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="origin_profile from converter, not relevant for calculations",
        metadata={"cim": True, "doc": False},
    ),
    "cim_topnode": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="cim_topnode from converter, not relevant for calculations",
        metadata={"cim": True, "doc": False},
    ),
    "ConnectivityNodeContainer_id": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="ConnectivityNodeContainer_id from converter, not relevant for calculations",
        metadata={"cim": True, "doc": False},
    ),
    "Substation_id": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="Substation_id from converter, not relevant for calculations",
        metadata={"cim": True, "doc": False},
    ),
    "description": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="description from converter, not relevant for calculations",
        metadata={"cim": True, "doc": False},
    ),
    "Busbar_id": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="Busbar_id from converter, not relevant for calculations",
        metadata={"cim": True, "doc": False},
    ),
    "Busbar_name": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="Busbar_name from converter, not relevant for calculations",
        metadata={"cim": True, "doc": False},
    ),
    "GeographicalRegion_id": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="GeographicalRegion_id from converter, not relevant for calculations",
        metadata={"cim": True, "doc": False},
    ),
    "GeographicalRegion_name": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="GeographicalRegion_name from converter, not relevant for calculations",
        metadata={"cim": True, "doc": False},
    ),
    "SubGeographicalRegion_id": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="SubGeographicalRegion_id from converter, not relevant for calculations",
        metadata={"cim": True, "doc": False},
    ),
    "SubGeographicalRegion_name": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="SubGeographicalRegion_name from converter, not relevant for calculations",
        metadata={"cim": True, "doc": False},
    ),
    "ucte_country": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="ucte_country from converter, not relevant for calculations",
        metadata={"ucte": True, "doc": False},
    ),
}
bus_schema = pa.DataFrameSchema(
    _bus_columns,
    name="bus",
    checks=[create_lower_equals_column_check(first_element="min_vm_pu", second_element="max_vm_pu")],
    strict=False,
)


res_bus_schema = res_bus_est_schema = pa.DataFrameSchema(
    {
        "vm_pu": pa.Column(float, nullable=True, description="voltage magnitude [p.u]"),
        "va_degree": pa.Column(float, nullable=True, description="voltage angle [degree]"),
        "p_mw": pa.Column(float, nullable=True, description="resulting active power demand [MW]"),
        "q_mvar": pa.Column(float, nullable=True, description="resulting reactive power demand [Mvar]"),
    },
    name="res_bus",
    strict=False,
)

res_bus_3ph_schema = pa.DataFrameSchema(
    {
        "vm_a_pu": pa.Column(float, nullable=True, description="voltage magnitude:Phase A [p.u]"),
        "va_a_degree": pa.Column(float, nullable=True, description="voltage angle:Phase A [degree]"),
        "vm_b_pu": pa.Column(float, nullable=True, description="voltage magnitude:Phase B [p.u]"),
        "va_b_degree": pa.Column(float, nullable=True, description="voltage angle:Phase B [degree]"),
        "vm_c_pu": pa.Column(float, nullable=True, description="voltage magnitude:Phase C [p.u]"),
        "va_c_degree": pa.Column(float, nullable=True, description="voltage angle:Phase C [degree]"),
        "p_a_mw": pa.Column(float, nullable=True, description="resulting active power demand:Phase A [MW]"),
        "q_a_mvar": pa.Column(float, nullable=True, description="resulting reactive power demand:Phase A [Mvar]"),
        "p_b_mw": pa.Column(float, nullable=True, description="resulting active power demand:Phase B [MW]"),
        "q_b_mvar": pa.Column(float, nullable=True, description="resulting reactive power demand:Phase B [Mvar]"),
        "p_c_mw": pa.Column(float, nullable=True, description="resulting active power demand:Phase C [MW]"),
        "q_c_mvar": pa.Column(float, nullable=True, description="resulting reactive power demand:Phase C [Mvar]"),
        "unbalance_percent": pa.Column(
            float,
            nullable=True,
            description="unbalance in percent defined as the ratio of V2 and V1 according to IEC 62749",
        ),
    },
    name="res_bus_3ph",
    strict=False,
)


res_bus_sc_schema = pa.DataFrameSchema(
    {
        "ikss_ka": pa.Column(float, nullable=True, description="initial short-circuit current value [kA]"),
        "skss_mw": pa.Column(float, nullable=True, description="initial short-circuit power [MW]"),
        "ip_ka": pa.Column(float, nullable=True, description="peak value of the short-circuit current [kA]"),
        "ith_ka": pa.Column(float, nullable=True, description="equivalent thermal short-circuit current [kA]"),
        "rk_ohm": pa.Column(
            float, nullable=True, description="resistive part of equiv. (positive/negative sequence) SC impedance [Ohm]"
        ),
        "xk_ohm": pa.Column(
            float, nullable=True, description="reactive part of equiv. (positive/negative sequence) SC impedance [Ohm]"
        ),
        "rk0_ohm": pa.Column(
            float, nullable=True, description="resistive part of equiv. (zero sequence) SC impedance [Ohm]"
        ),
        "xk0_ohm": pa.Column(
            float, nullable=True, description="reactive part of equiv. (zero sequence) SC impedance [Ohm]"
        ),
    },
    name="res_bus_sc",
    strict=False,
)
