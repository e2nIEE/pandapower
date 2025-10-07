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
