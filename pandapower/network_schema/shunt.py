import pandera.pandas as pa

schema = pa.DataFrameSchema(
    {
        "name": pa.Column(str, description="name of the shunt"),
        "bus": pa.Column(int, pa.Check.ge(0), description="index of bus where the impedance starts"),
        "p_mw": pa.Column(float, pa.Check.ge(0), description="shunt active power in MW at v= 1.0 p.u. per step"),
        "q_mvar": pa.Column(float, description="shunt reactive power in MVAr at v= 1.0 p.u. per step"),
        "vn_kv": pa.Column(float, pa.Check.gt(0), description="rated voltage of the shunt element"),
        "step": pa.Column(
            int, pa.Check.ge(1), description="step position of the shunt with which power values are multiplied"
        ),
        "max_step": pa.Column(int, pa.Check.ge(1), description="maximum allowed step of shunt"),
        "in_service": pa.Column(bool, description="specifies if the shunt is in service"),
        "step_dependency_table": pa.Column(
            bool,
            description="whether the shunt parameters (q_mvar, p_mw) are adjusted dependent on the step of the shunt",
        ),
        "id_characteristic_table": pa.Column(
            int,
            pa.Check.ge(0),
            description="references the id_characteristic index from the shunt_characteristic_table",
        ),
    },
    strict=False,
)


res_schema = pa.DataFrameSchema(
    {
        "p_mw": pa.Column(float, description="shunt active power consumption [MW]"),
        "q_mvar": pa.Column(float, description="shunt reactive power consumption [MVAr]"),
        "vm_pu": pa.Column(float, description="voltage magnitude at shunt bus [pu]"),
    },
)
