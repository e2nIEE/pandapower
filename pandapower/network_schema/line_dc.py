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

res_schema = pa.DataFrameSchema(
    {
        "p_from_mw": pa.Column(float, description="active power flow into the dc line at “”from”” dc bus [MW]"),
        "p_to_mw": pa.Column(float, description="active power flow into the dc line at “”to”” dc bus [MW]"),
        "pl_mw": pa.Column(float, description="active power losses of the dc line [MW]"),
        "i_from_ka": pa.Column(float, description="Current at from dc bus [kA]"),
        "i_to_ka": pa.Column(float, description="Current at to dc bus [kA]"),
        "i_ka": pa.Column(float, description="Maximum of i_from_ka and i_to_ka [kA]"),
        "vm_from_pu": pa.Column(float, description="voltage magnitude at from dc bus"),
        "vm_to_pu": pa.Column(float, description="voltage magnitude at to dc bus"),
        "loading_percent": pa.Column(float, description="line loading [%]"),
    },
)
