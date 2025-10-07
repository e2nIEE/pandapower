import pandera.pandas as pa

schema = pa.DataFrameSchema(  # in methodcall but not parameter docu: geodata, alpha, temperature_degree_celsius
    {
        "name": pa.Column(str, description="name of the line"),
        "std_type": pa.Column(
            str,
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
            float, pa.Check.ge(0), required=False, description="zero sequence resistance of the line [Ohm per km]"
        ),
        "x0_ohm_per_km": pa.Column(
            float, pa.Check.ge(0), required=False, description="zero sequence reactance of the line [Ohm per km]"
        ),
        "c0_nf_per_km": pa.Column(
            float,
            pa.Check.ge(0),
            required=False,
            description="zero sequence capacitance of the line [nano Farad per km]",
        ),
        "g_us_per_km": pa.Column(
            float, pa.Check.ge(0), description="dielectric conductance of the line [micro Siemens per km]"
        ),
        "max_i_ka": pa.Column(float, pa.Check.gt(0), description="maximal thermal current [kilo Ampere]"),
        "parallel": pa.Column(
            int, pa.Check.ge(1), description="number of parallel line systems"
        ),  # other elements have ge0
        "df": pa.Column(
            float, pa.Check.between(min_value=0, max_value=1), description="derating factor (scaling) for max_i_ka"
        ),
        "type": pa.Column(str, pa.Check.isin(["ol", "cs"]), description="type of line"),
        "max_loading_percent": pa.Column(
            float, pa.Check.gt(0), required=False, description="Maximum loading of the line"
        ),
        "endtemp_degree": pa.Column(
            float, pa.Check.gt(0), required=False, description="Short-Circuit end temperature of the line"
        ),  # not in create method call
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
