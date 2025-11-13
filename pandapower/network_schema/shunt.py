import pandas as pd
import pandera.pandas as pa

shunt_schema = pa.DataFrameSchema(
    {
        "name": pa.Column(pd.StringDtype, nullable=True, required=False, description="name of the shunt"),
        "bus": pa.Column(
            int,
            pa.Check.ge(0),
            description="index of bus where the impedance starts",
            metadata={"foreign_key": "bus.index"},
        ),
        "p_mw": pa.Column(float, pa.Check.ge(0), description="shunt active power in MW at v= 1.0 p.u. per step"),
        "q_mvar": pa.Column(float, description="shunt reactive power in MVAr at v= 1.0 p.u. per step"),
        "vn_kv": pa.Column(float, pa.Check.gt(0), description="rated voltage of the shunt element"),
        "step": pa.Column(
            int, pa.Check.ge(1), description="step position of the shunt with which power values are multiplied"
        ),
        "max_step": pa.Column(
            pd.Int64Dtype, pa.Check.ge(1), nullable=True, required=False, description="maximum allowed step of shunt"
        ),
        "in_service": pa.Column(bool, description="specifies if the shunt is in service"),
        "step_dependency_table": pa.Column(
            pd.BooleanDtype,
            nullable=True,
            required=False,
            description="whether the shunt parameters (q_mvar, p_mw) are adjusted dependent on the step of the shunt",
        ),  # TODO: remove since it is implied by id_characteristic_table
        "id_characteristic_table": pa.Column(
            pd.Int64Dtype,
            pa.Check.ge(0),
            nullable=True,
            required=True,  # TODO: switch to false, when step_dependancy_table is gone
            description="references the id_characteristic index from the shunt_characteristic_table",
        ),
    },
    checks=[
        pa.Check(
            lambda df: (
                df["step"] <= df["max_step"] if all(col in df.columns for col in ["step", "max_step"]) else True
            ),
            error="Column 'step' must be <= column 'max_step'",
        )
    ],
    strict=False,
)


res_shunt_schema = pa.DataFrameSchema(
    {
        "p_mw": pa.Column(float, nullable=True, description="shunt active power consumption [MW]"),
        "q_mvar": pa.Column(float, nullable=True, description="shunt reactive power consumption [MVAr]"),
        "vm_pu": pa.Column(float, nullable=True, description="voltage magnitude at shunt bus [pu]"),
    },
    strict=False,
)
