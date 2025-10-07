import pandera.pandas as pa

schema = pa.DataFrameSchema(
    {
        "name": pa.Column(str, description="name of the generator"),
        "type": pa.Column(str, pa.Check.isin(["sync", "async"]), description="type variable to classify generators"),
        "bus": pa.Column(int, description="index of connected bus"),
        "p_mw": pa.Column(int, pa.Check.se(0), description="the real power of the generator [MW]"),
        "vm_pu": pa.Column(float, description="voltage set point of the generator [p.u.]"),
        "sn_mva": pa.Column(float, pa.Check.gt(0), description="nominal power of the generator [MVA]"),
        "max_q_mvar": pa.Column(float, description="maximum reactive power of the generator [MVAr]", metadata={"opf": True}),
        "min_q_mvar": pa.Column(float, description="minimum reactive power of the generator [MVAr]", metadata={"opf": True}),
        "scaling": pa.Column(float, pa.Check.se(0), description="scaling factor for the active power"),
        "max_p_mw": pa.Column(float, description="maximum active power", metadata={"opf": True}),
        "min_p_mw": pa.Column(float, description="minimum active power", metadata={"opf": True}),
        "vn_kv": pa.Column(float, description="rated voltage of the generator", metadata={"sc": True}),
        "xdss_pu": pa.Column(float, pa.Check.gt(0), description="subtransient generator reactance in per unit", metadata={"sc": True}),
        "rdss_ohm": pa.Column(float, pa.Check.gt(0), description="subtransient generator resistence in ohm", metadata={"sc": True}),
        "cos_phi": pa.Column(float, pa.Check.se(1), description="rated generator cosine phi", metadata={"sc": True}),
        "in_service": pa.Column(bool, description="specifies if the generator is in service"),
        "power_station_trafo": pa.Column(int, description="index of the power station trafo (short-circuit relevant)", metadata={"sc": True}),
        "id_q_capability_characteristic": pa.Column(int, description="references the index of the characteristic from the q_capability_characteristic"),
        "curve_style": pa.Column(str, pa.Check.isin(["straightLineYValues", "constantYValue"]), description="the style of the generator reactive power capability curve"),
        "reactive_capability_curve": pa.Column(bool, description="True if generator has dependency on q characteristic"),
        "slack_weight": pa.Column(bool, description=""),  # missing in docu
        "slack": pa.Column(bool, description=""),  # missing in docu
        "controllable": pa.Column(bool, description="")  # missing in docu
    },
    strict=False,
)



res_schema = pa.DataFrameSchema(
    {
        "p_mw": pa.Column(float, description="resulting active power demand after scaling [MW]"),
        "q_mvar": pa.Column(float, description="resulting reactive power demand after scaling [MVAr]"),
        "va_degree": pa.Column(float, description="generator voltage angle [degree]"),
        "vm_pu": pa.Column(float, description="voltage at the generator [p.u.]"),
    },
)
