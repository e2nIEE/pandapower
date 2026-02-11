import pandas as pd
import pandera.pandas as pa

from pandapower.network_schema.tools.validation.group_dependency import create_column_dependency_checks_from_metadata

_line_dc_columns = {
    "name": pa.Column(pd.StringDtype, nullable=True, required=False, description="name of the dc line"),
    "std_type": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="standard type which can be used to easily define dc line parameters with the pandapower standard type library",
    ),
    "from_bus_dc": pa.Column(
        int,
        pa.Check.ge(0),
        description="Index of dc bus where the dc line starts",
        metadata={"foreign_key": "bus_dc.index"},
    ),
    "to_bus_dc": pa.Column(
        int,
        pa.Check.ge(0),
        description="Index of dc bus where the dc line ends",
        metadata={"foreign_key": "bus_dc.index"},
    ),
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
        pd.StringDtype,
        nullable=True,
        required=False,
        description="type of dc line Naming conventions: “”ol”” - overhead dc line, “”cs”” - underground cable system”",
    ),
    "max_loading_percent": pa.Column(
        float,
        pa.Check.gt(0),
        nullable=True,
        required=False,
        description="Maximum loading of the dc line",
        metadata={"opf": True},
    ),
    "in_service": pa.Column(bool, description="specifies if the dc line is in service."),
    "geo": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="geojson.LineString object or its string representation",
    ),
    "alpha": pa.Column(
        float,
        nullable=True,
        required=False,
        description="temperature coefficient of resistance: R(T) = R(T_0) * (1 + alpha * (T - T_0))",
    ),
    "temperature_degree_celsius": pa.Column(
        float, nullable=True, required=False, description="line temperature for which line resistance is adjusted"
    ),
    "tdpf": pa.Column(
        pd.BooleanDtype,
        nullable=True,
        required=False,
        description="whether the line is considered in the TDPF calculation",
        metadata={"tdpf": True},
    ),
    "wind_speed_m_per_s": pa.Column(
        float,
        nullable=True,
        required=False,
        description="wind speed at the line in m/s (TDPF)",
        metadata={"tdpf": True},
    ),
    "wind_angle_degree": pa.Column(
        float,
        nullable=True,
        required=False,
        description="angle of attack between the wind direction and the line (TDPF)",
        metadata={"tdpf": True},
    ),
    "conductor_outer_diameter_m": pa.Column(
        float,
        nullable=True,
        required=False,
        description="outer diameter of the line conductor in m (TDPF)",
        metadata={"tdpf": True},
    ),
    "air_temperature_degree_celsius": pa.Column(
        float,
        nullable=True,
        required=False,
        description="ambient temperature in °C (TDPF)",
        metadata={"tdpf": True},
    ),
    "reference_temperature_degree_celsius": pa.Column(
        float,
        nullable=True,
        required=False,
        description="reference temperature in °C for which r_ohm_per_km for the line_dc is specified (TDPF)",
        metadata={"tdpf": True},
    ),
    "solar_radiation_w_per_sq_m": pa.Column(
        float,
        nullable=True,
        required=False,
        description="solar radiation on horizontal plane in W/m² (TDPF)",
        metadata={"tdpf": True},
    ),
    "solar_absorptivity": pa.Column(
        float,
        nullable=True,
        required=False,
        description="Albedo factor for absorptivity of the lines (TDPF)",
        metadata={"tdpf": True},
    ),
    "emissivity": pa.Column(
        float,
        nullable=True,
        required=False,
        description="Albedo factor for emissivity of the lines (TDPF)",
        metadata={"tdpf": True},
    ),
    "r_theta_kelvin_per_mw": pa.Column(
        float,
        nullable=True,
        required=False,
        description="thermal resistance of the line (TDPF, only for simplified method)",
        metadata={"tdpf": True},
    ),
    "mc_joule_per_m_k": pa.Column(
        float,
        nullable=True,
        required=False,
        description="specific mass of the conductor multiplied by the specific thermal capacity of the material (TDPF, only for thermal inertia consideration with tdpf_delay_s parameter)",
        metadata={"tdpf": True},
    ),
}
line_dc_schema = pa.DataFrameSchema(
    _line_dc_columns,
    checks=create_column_dependency_checks_from_metadata(["tdpf"], _line_dc_columns),
    strict=False,
)

res_line_dc_schema = pa.DataFrameSchema(
    {
        "p_from_mw": pa.Column(
            float, nullable=True, description="active power flow into the dc line at “”from”” dc bus [MW]"
        ),
        "p_to_mw": pa.Column(
            float, nullable=True, description="active power flow into the dc line at “”to”” dc bus [MW]"
        ),
        "pl_mw": pa.Column(float, nullable=True, description="active power losses of the dc line [MW]"),
        "i_from_ka": pa.Column(float, nullable=True, description="Current at from dc bus [kA]"),
        "i_to_ka": pa.Column(float, nullable=True, description="Current at to dc bus [kA]"),
        "i_ka": pa.Column(float, nullable=True, description="Maximum of i_from_ka and i_to_ka [kA]"),
        "vm_from_pu": pa.Column(float, nullable=True, description="voltage magnitude at from dc bus"),
        "vm_to_pu": pa.Column(float, nullable=True, description="voltage magnitude at to dc bus"),
        "loading_percent": pa.Column(float, nullable=True, description="line loading [%]"),
    },
    strict=False,
)
