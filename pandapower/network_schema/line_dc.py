import pandera.pandas as pa

schema = pa.DataFrameSchema(
    {  # TODO: in methodcall but not parameter docu: geodata, alpha, temperature_degree_celsius
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
        ),  # TODO: docu broken
        "max_loading_percent": pa.Column(float, pa.Check.gt(0), description="Maximum loading of the dc line"),
        "in_service": pa.Column(bool, description="specifies if the dc line is in service."),
        "geo": pa.Column(str, description=""),  # TODO: missing in docu

        #neu (Kommentar kann nach kontrolle gelöscht werden)
        "alpha": pa.Column(float, description="temperature coefficient of resistance: R(T) = R(T_0) * (1 + alpha * (T - T_0))"),
        "temperature_degree_celsius": pa.Column(float, description="line temperature for which line resistance is adjusted"),
        "tdpf": pa.Column(bool, description="whether the line is considered in the TDPF calculation", metadata={"tdpf": True}),
        "wind_speed_m_per_s": pa.Column(float, description="wind speed at the line in m/s (TDPF)", metadata={"tdpf": True}),
        "wind_angle_degree": pa.Column(float, description="angle of attack between the wind direction and the line (TDPF)", metadata={"tdpf": True}),
        "conductor_outer_diameter_m": pa.Column(float, description="outer diameter of the line conductor in m (TDPF)", metadata={"tdpf": True}),
        "air_temperature_degree_celsius": pa.Column(float, description="ambient temperature in °C (TDPF)", metadata={"tdpf": True}),
        "reference_temperature_degree_celsius": pa.Column(float, description="reference temperature in °C for which r_ohm_per_km for the line_dc is specified (TDPF)", metadata={"tdpf": True}),
        "solar_radiation_w_per_sq_m": pa.Column(float, description="solar radiation on horizontal plane in W/m² (TDPF)", metadata={"tdpf": True}),
        "solar_absorptivity": pa.Column(float, description="Albedo factor for absorptivity of the lines (TDPF)", metadata={"tdpf": True}),
        "emissivity": pa.Column(float, description="Albedo factor for emissivity of the lines (TDPF)", metadata={"tdpf": True}),
        "r_theta_kelvin_per_mw": pa.Column(float, description="thermal resistance of the line (TDPF, only for simplified method)", metadata={"tdpf": True}),
        "mc_joule_per_m_k": pa.Column(float, description="specific mass of the conductor multiplied by the specific thermal capacity of the material (TDPF, only for thermal inertia consideration with tdpf_delay_s parameter)", metadata={"tdpf": True}),
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
