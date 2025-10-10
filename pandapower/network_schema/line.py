import pandera.pandas as pa

schema = pa.DataFrameSchema(  # TODO: in methodcall but not parameter docu: geodata, alpha, temperature_degree_celsius
    {
        "name": pa.Column(str, required=False, description="name of the line"),
        "std_type": pa.Column(
            str,
            required=False,
            description="standard type which can be used to easily define line parameters with the pandapower standard type library",
        ),
        "from_bus": pa.Column(int, pa.Check.ge(0), description="Index of bus where the line starts"),
        "to_bus": pa.Column(int, pa.Check.ge(0), description="Index of bus where the line ends"),
        "length_km": pa.Column(float, pa.Check.gt(0), description="length of the line [km]"),
        "r_ohm_per_km": pa.Column(float, pa.Check.ge(0), description="resistance of the line [Ohm per km]"),
        "x_ohm_per_km": pa.Column(float, pa.Check.ge(0), description="reactance of the line [Ohm per km]"),
        "c_nf_per_km": pa.Column(
            float, pa.Check.ge(0), description="capacitance of the line (line-to-earth) [nano Farad per km]"
        ),
        "r0_ohm_per_km": pa.Column(
            float,
            pa.Check.ge(0),
            required=False,
            description="zero sequence resistance of the line [Ohm per km]",
            metadata={"sc": True, "3ph": True},
        ),
        "x0_ohm_per_km": pa.Column(
            float,
            pa.Check.ge(0),
            required=False,
            description="zero sequence reactance of the line [Ohm per km]",
            metadata={"sc": True, "3ph": True},
        ),
        "c0_nf_per_km": pa.Column(
            float,
            pa.Check.ge(0),
            required=False,
            description="zero sequence capacitance of the line [nano Farad per km]",
            metadata={"sc": True, "3ph": True},
        ),
        "g_us_per_km": pa.Column(
            float, pa.Check.ge(0), description="dielectric conductance of the line [micro Siemens per km]"
        ),
        "max_i_ka": pa.Column(float, pa.Check.gt(0), description="maximal thermal current [kilo Ampere]"),
        "parallel": pa.Column(int, pa.Check.ge(1), description="number of parallel line systems"),
        "df": pa.Column(
            float, pa.Check.between(min_value=0, max_value=1), description="derating factor (scaling) for max_i_ka"
        ),
        "type": pa.Column(str, pa.Check.isin(["ol", "cs"]), required=False, description="type of line"),
        "max_loading_percent": pa.Column(
            float, pa.Check.gt(0), required=False, description="Maximum loading of the line", metadata={"opf": True}
        ),
        "endtemp_degree": pa.Column(
            float,
            pa.Check.gt(0),
            required=False,
            description="Short-Circuit end temperature of the line",
            metadata={"sc": True, "tdpf": True},
        ),  # TODO: add all tdpf parameters from create documentation, bzw alle die in der methoden docu stehen
        "in_service": pa.Column(bool, description="specifies if the line is in service."),
        "geo": pa.Column(str, description="geojson.LineString object or its string representation"),
    },
    strict=False,
)

res_schema = pa.DataFrameSchema(
    {
        "p_from_mw": pa.Column(float, description="active power flow into the line at “from” bus [MW]"),
        "q_from_mvar": pa.Column(float, description="reactive power flow into the line at “from” bus [MVar]"),
        "p_to_mw": pa.Column(float, description="active power flow into the line at “to” bus [MW]"),
        "q_to_mvar": pa.Column(float, description="reactive power flow into the line at “to” bus [MVar]"),
        "pl_mw": pa.Column(float, description="active power losses of the line [MW]"),
        "ql_mvar": pa.Column(float, description="reactive power consumption of the line [MVar]"),
        "i_from_ka": pa.Column(float, description="Current at from bus [kA]"),
        "i_to_ka": pa.Column(float, description="Current at to bus [kA]"),
        "i_ka": pa.Column(float, description="Maximum of i_from_ka and i_to_ka [kA]"),
        "vm_from_pu": pa.Column(float, description="voltage magnitude at from bus"),
        "va_from_degree": pa.Column(float, description="voltage magnitude at to bus"),
        "vm_to_pu": pa.Column(float, description="voltage angle at from bus [degrees]"),
        "va_to_degree": pa.Column(float, description="voltage angle at to bus [degrees]"),
        "loading_percent": pa.Column(float, description="line loading [%]"),
    },
)

res_schema_3ph = pa.DataFrameSchema(
    {
        "p_a_from_mw": pa.Column(float, description="active power flow into the line at from bus: Phase A [MW]"),
        "q_a_from_mvar": pa.Column(float, description="reactive power flow into the line at from bus : Phase A [MVar]"),
        "p_b_from_mw": pa.Column(float, description="active power flow into the line at from bus: Phase B [MW]"),
        "q_b_from_mvar": pa.Column(float, description="reactive power flow into the line at from bus : Phase B[MVar]"),
        "p_c_from_mw": pa.Column(float, description="active power flow into the line at from bus: Phase C [MW]"),
        "q_c_from_mvar": pa.Column(float, description="reactive power flow into the line at from bus : Phase C[MVar]"),
        "p_a_to_mw": pa.Column(float, description="active power flow into the line at to bus: Phase A [MW]"),
        "q_a_to_mvar": pa.Column(float, description="reactive power flow into the line at to bus : Phase A[MVar]"),
        "p_b_to_mw": pa.Column(float, description="active power flow into the line at to bus: Phase B [MW]"),
        "q_b_to_mvar": pa.Column(float, description="reactive power flow into the line at to bus : Phase B[MVar]"),
        "p_c_to_mw": pa.Column(float, description="active power flow into the line at to bus: Phase C [MW]"),
        "q_c_to_mvar": pa.Column(float, description="reactive power flow into the line at to bus : Phase C[MVar]"),
        "pl_a_mw": pa.Column(float, description="active power losses of the line: Phase A [MW]"),
        "ql_a_mvar": pa.Column(float, description="reactive power consumption of the line: Phase A [MVar]"),
        "pl_b_mw": pa.Column(float, description="active power losses of the line: Phase B [MW]"),
        "ql_b_mvar": pa.Column(float, description="reactive power consumption of the line: Phase B [MVar]"),
        "pl_c_mw": pa.Column(float, description="active power losses of the line: Phase C [MW]"),
        "ql_c_mvar": pa.Column(float, description="reactive power consumption of the line: Phase C [MVar]"),
        "i_a_from_ka": pa.Column(float, description="Current at from bus: Phase A [kA]"),
        "i_a_to_ka": pa.Column(float, description="Current at to bus: Phase A [kA]"),
        "i_b_from_ka": pa.Column(float, description="Current at from bus: Phase B [kA]"),
        "i_b_to_ka": pa.Column(float, description="Current at to bus: Phase B [kA]"),
        "i_c_from_ka": pa.Column(float, description="Current at from bus: Phase C [kA]"),
        "i_c_to_ka": pa.Column(float, description="Current at to bus: Phase C [kA]"),
        "i_a_ka": pa.Column(float, description=""),  # TODO: missing in docu
        "i_b_ka": pa.Column(float, description=""),  # TODO: missing in docu
        "i_c_ka": pa.Column(float, description=""),  # TODO: missing in docu
        "i_n_from_ka": pa.Column(float, description="Current at from bus: Neutral [kA]"),
        "i_n_to_ka": pa.Column(float, description="Current at to bus: Neutral [kA]"),
        # "i_ka": pa.Column(float, description="Maximum of i_from_ka and i_to_ka [kA]"),  #TODO: was only in docu
        "i_n_ka": pa.Column(float, description=""),  # TODO: missing in docu
        "loading_a_percent": pa.Column(float, description="line a loading [%]"),
        "loading_b_percent": pa.Column(float, description="line b loading [%]"),
        "loading_c_percent": pa.Column(float, description="line c loading [%]"),
        # "loading_n_percent": pa.Column(float, description=""),  #TODO: was only in docu
    },
)

# TODO: was ist mit res_line_est und res_line_sc
