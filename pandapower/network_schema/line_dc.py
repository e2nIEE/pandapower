import pandera.pandas as pa

schema = pa.DataFrameSchema(
    {  # in methodcall but not parameter docu: geodata, alpha, temperature_degree_celsius
        "name": pa.Column(str, description="name of the dc line"),
        "std_type": pa.Column(
            str,
            description="standard type which can be used to easily define dc line parameters with the pandapower standard type library",
        ),
        "from_bus_dc": pa.Column(int, pa.Check.ge(0), description="Index of dc bus where the dc line starts"),
        "to_bus_dc": pa.Column(int, pa.Check.ge(0), description="Index of dc bus where the dc line ends"),
        "length_km": pa.Column(float, pa.Check.gt(0), description="length of the line [km]"),
        "r_ohm_per_km": pa.Column(float, pa.Check.ge(0), description="resistance of the line [Ohm per km]"),
        "g_us_per_km": pa.Column(
            float, pa.Check.ge(0), description="dielectric conductance of the dc line [micro Siemens per km]"
        ),
        "max_i_ka": pa.Column(float, pa.Check.ge(0), description="maximal thermal current [kilo Ampere]"),
        "parallel": pa.Column(int, pa.Check.ge(1), description="number of parallel dc line systems"),
        "df": pa.Column(
            float, pa.Check.between(min_value=0, max_value=1), description="derating factor (scaling) for max_i_ka"
        ),
        "type": pa.Column(
            str,
            pa.Check.isin(["ol", "cs"]),
            description="type of dc line Naming conventions: “”ol”” - overhead dc line, “”cs”” - underground cable system”",
        ),  # docu broken
        "max_loading_percent": pa.Column(float, pa.Check.gt(0), description="Maximum loading of the dc line"),
        "in_service": pa.Column(bool, description="specifies if the dc line is in service."),
        "geo": pa.Column(str, description=""),  # missing in docu
    },
    strict=False,
)
