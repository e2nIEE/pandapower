import pandas as pd
import pandera.pandas as pa

from pandapower.network_schema.tools.validation.group_dependency import create_column_dependency_checks_from_metadata

_line_columns = {
    "name": pa.Column(pd.StringDtype, nullable=True, required=False, description="name of the line"),
    "std_type": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="standard type which can be used to easily define line parameters with the pandapower standard type library",
    ),
    "from_bus": pa.Column(
        int, pa.Check.ge(0), description="Index of bus where the line starts", metadata={"foreign_key": "bus.index"}
    ),
    "to_bus": pa.Column(
        int, pa.Check.ge(0), description="Index of bus where the line ends", metadata={"foreign_key": "bus.index"}
    ),
    "length_km": pa.Column(float, pa.Check.gt(0), description="length of the line [km]"),
    "r_ohm_per_km": pa.Column(float, pa.Check.ge(0), description="resistance of the line [Ohm per km]"),
    "x_ohm_per_km": pa.Column(float, pa.Check.ge(0), description="reactance of the line [Ohm per km]"),
    "c_nf_per_km": pa.Column(
        float, pa.Check.ge(0), description="capacitance of the line (line-to-earth) [nano Farad per km]"
    ),
    "g_us_per_km": pa.Column(
        float,
        pa.Check.ge(0),
        description="dielectric conductance in micro Siemens per km",
    ),
    "r0_ohm_per_km": pa.Column(
        float,
        pa.Check.ge(0),
        nullable=True,
        required=False,
        description="zero sequence resistance of the line [Ohm per km]",
        metadata={"sc": True, "3ph": True},
    ),
    "x0_ohm_per_km": pa.Column(
        float,
        pa.Check.ge(0),
        nullable=True,
        required=False,
        description="zero sequence reactance of the line [Ohm per km]",
        metadata={"sc": True, "3ph": True},
    ),
    "c0_nf_per_km": pa.Column(
        float,
        pa.Check.ge(0),
        nullable=True,
        required=False,
        description="zero sequence capacitance of the line [nano Farad per km]",
        metadata={"sc": True, "3ph": True},
    ),
    "g0_us_per_km": pa.Column(
        float,
        pa.Check.ge(0),
        nullable=True,
        required=False,
        description="dielectric conductance of the line [micro Siemens per km]",
        metadata={"sc": True, "3ph": True},
    ),
    "max_i_ka": pa.Column(float, pa.Check.gt(0), description="maximal thermal current [kilo Ampere]"),
    "parallel": pa.Column(int, pa.Check.ge(1), description="number of parallel line systems"),
    "df": pa.Column(
        float, pa.Check.between(min_value=0, max_value=1), description="derating factor (scaling) for max_i_ka"
    ),
    "type": pa.Column(
        pd.StringDtype,
        nullable=True,
        required=False,
        description="type of line normally “ol” - overhead line “cs” - underground cable system",
    ),
    "max_loading_percent": pa.Column(
        float,
        pa.Check.gt(0),
        nullable=True,
        required=False,
        description="Maximum loading of the line",
        metadata={"opf": True},
    ),
    "endtemp_degree": pa.Column(
        float,
        pa.Check.gt(-273),
        nullable=True,
        required=False,
        description="Short-Circuit end temperature of the line in degree Celsius",
        metadata={"sc": True, "tdpf": True},
    ),
    "in_service": pa.Column(bool, description="specifies if the line is in service."),
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
        float,
        pa.Check.gt(-273),
        nullable=True,
        required=False,
        description="line temperature for which line resistance is adjusted",
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
        description="reference temperature in °C for which r_ohm_per_km for the line is specified (TDPF)",
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
line_schema = pa.DataFrameSchema(
    _line_columns,
    strict=False,
    checks=create_column_dependency_checks_from_metadata(["opf", "sc", "tdpf", "3ph"], _line_columns),
)

res_line_schema = res_line_est_schema = pa.DataFrameSchema(
    {
        "p_from_mw": pa.Column(float, nullable=True, description="active power flow into the line at “from” bus [MW]"),
        "q_from_mvar": pa.Column(
            float, nullable=True, description="reactive power flow into the line at “from” bus [MVar]"
        ),
        "p_to_mw": pa.Column(float, nullable=True, description="active power flow into the line at “to” bus [MW]"),
        "q_to_mvar": pa.Column(
            float, nullable=True, description="reactive power flow into the line at “to” bus [MVar]"
        ),
        "pl_mw": pa.Column(float, nullable=True, description="active power losses of the line [MW]"),
        "ql_mvar": pa.Column(float, nullable=True, description="reactive power consumption of the line [MVar]"),
        "i_from_ka": pa.Column(float, nullable=True, description="Current at from bus [kA]"),
        "i_to_ka": pa.Column(float, nullable=True, description="Current at to bus [kA]"),
        "i_ka": pa.Column(float, nullable=True, description="Maximum of i_from_ka and i_to_ka [kA]"),
        "vm_from_pu": pa.Column(float, nullable=True, description="voltage magnitude at from bus"),
        "va_from_degree": pa.Column(float, nullable=True, description="voltage magnitude at to bus"),
        "vm_to_pu": pa.Column(float, nullable=True, description="voltage angle at from bus [degrees]"),
        "va_to_degree": pa.Column(float, nullable=True, description="voltage angle at to bus [degrees]"),
        "loading_percent": pa.Column(float, nullable=True, description="line loading [%]"),
    },
    strict=False,
)

res_line_3ph_schema = pa.DataFrameSchema(
    {
        "p_a_from_mw": pa.Column(
            float, nullable=True, description="active power flow into the line at from bus: Phase A [MW]"
        ),
        "q_a_from_mvar": pa.Column(
            float, nullable=True, description="reactive power flow into the line at from bus : Phase A [MVar]"
        ),
        "p_b_from_mw": pa.Column(
            float, nullable=True, description="active power flow into the line at from bus: Phase B [MW]"
        ),
        "q_b_from_mvar": pa.Column(
            float, nullable=True, description="reactive power flow into the line at from bus : Phase B[MVar]"
        ),
        "p_c_from_mw": pa.Column(
            float, nullable=True, description="active power flow into the line at from bus: Phase C [MW]"
        ),
        "q_c_from_mvar": pa.Column(
            float, nullable=True, description="reactive power flow into the line at from bus : Phase C[MVar]"
        ),
        "p_a_to_mw": pa.Column(
            float, nullable=True, description="active power flow into the line at to bus: Phase A [MW]"
        ),
        "q_a_to_mvar": pa.Column(
            float, nullable=True, description="reactive power flow into the line at to bus : Phase A[MVar]"
        ),
        "p_b_to_mw": pa.Column(
            float, nullable=True, description="active power flow into the line at to bus: Phase B [MW]"
        ),
        "q_b_to_mvar": pa.Column(
            float, nullable=True, description="reactive power flow into the line at to bus : Phase B[MVar]"
        ),
        "p_c_to_mw": pa.Column(
            float, nullable=True, description="active power flow into the line at to bus: Phase C [MW]"
        ),
        "q_c_to_mvar": pa.Column(
            float, nullable=True, description="reactive power flow into the line at to bus : Phase C[MVar]"
        ),
        "pl_a_mw": pa.Column(float, nullable=True, description="active power losses of the line: Phase A [MW]"),
        "ql_a_mvar": pa.Column(
            float, nullable=True, description="reactive power consumption of the line: Phase A [MVar]"
        ),
        "pl_b_mw": pa.Column(float, nullable=True, description="active power losses of the line: Phase B [MW]"),
        "ql_b_mvar": pa.Column(
            float, nullable=True, description="reactive power consumption of the line: Phase B [MVar]"
        ),
        "pl_c_mw": pa.Column(float, nullable=True, description="active power losses of the line: Phase C [MW]"),
        "ql_c_mvar": pa.Column(
            float, nullable=True, description="reactive power consumption of the line: Phase C [MVar]"
        ),
        "i_a_from_ka": pa.Column(float, nullable=True, description="Current at from bus: Phase A [kA]"),
        "i_a_to_ka": pa.Column(float, nullable=True, description="Current at to bus: Phase A [kA]"),
        "i_a_ka": pa.Column(float, nullable=True, description=""),
        "i_b_from_ka": pa.Column(float, nullable=True, description="Current at from bus: Phase B [kA]"),
        "i_b_to_ka": pa.Column(float, nullable=True, description="Current at to bus: Phase B [kA]"),
        "i_b_ka": pa.Column(float, nullable=True, description=""),
        "i_c_from_ka": pa.Column(float, nullable=True, description="Current at from bus: Phase C [kA]"),
        "i_c_to_ka": pa.Column(float, nullable=True, description="Current at to bus: Phase C [kA]"),
        "i_c_ka": pa.Column(float, nullable=True, description=""),
        "i_n_from_ka": pa.Column(
            float, nullable=True, description="Current at from bus: Neutral [kA]"
        ),  # TODO: muss mike schauen
        "i_n_to_ka": pa.Column(
            float, nullable=True, description="Current at to bus: Neutral [kA]"
        ),  # TODO: muss mike schauen
        "i_ka": pa.Column(float, nullable=True, description="Maximum of i_from_ka and i_to_ka [kA]"),
        "i_n_ka": pa.Column(float, nullable=True, description=""),  # TODO: missing in docu muss mike schauen
        "loading_a_percent": pa.Column(float, nullable=True, description="line a loading [%]"),
        "loading_b_percent": pa.Column(float, nullable=True, description="line b loading [%]"),
        "loading_c_percent": pa.Column(float, nullable=True, description="line c loading [%]"),
        "loading_n_percent": pa.Column(float, nullable=True, description=""),  # TODO: was only in docu
    },
    strict=False,
)

res_line_sc_schema = pa.DataFrameSchema(
    {
        "ikss_ka": pa.Column(float, nullable=True, description="initial short-circuit current value [kA]"),
        "ikss_from_ka": pa.Column(
            float, nullable=True, description="magnitude of the initial SC current at “”from”” bus [kA]"
        ),
        "ikss_from_degree": pa.Column(
            float, nullable=True, description="degree of the initial SC current at “”from”” bus [degrees]"
        ),
        "ikss_to_ka": pa.Column(
            float, nullable=True, description="magnitude of the initial SC current at “”to”” bus [kA]"
        ),
        "ikss_to_degree": pa.Column(
            float, nullable=True, description="degree of the initial SC current at “”to”” bus [degrees]"
        ),
        "p_from_mw": pa.Column(
            float, nullable=True, description="active SC power flow into the line at “”from”” bus [MW]"
        ),
        "q_from_mvar": pa.Column(
            float, nullable=True, description="reactive SC power flow into the line at “”from”” bus [MVar]"
        ),
        "p_to_mw": pa.Column(float, nullable=True, description="active SC power flow into the line at “”to”” bus [MW]"),
        "q_to_mvar": pa.Column(
            float, nullable=True, description="reactive SC power flow into the line at “”to”” bus [MVar]"
        ),
        "vm_from_pu": pa.Column(float, nullable=True, description="voltage magnitude at “”from”” bus [p.u.]"),
        "vm_to_pu": pa.Column(float, nullable=True, description="voltage magnitude at “”to”” bus [p.u.]"),
        "va_from_degree": pa.Column(float, nullable=True, description="voltage angle at “”from”” bus [degrees]"),
        "va_to_degree": pa.Column(float, nullable=True, description="voltage angle at “”to”” bus [degrees]"),
        "ip_ka": pa.Column(float, nullable=True, description="peak value of the short-circuit current [kA]"),
        "ith_ka": pa.Column(float, nullable=True, description="equivalent thermal short-circuit current [kA]"),
    },
    strict=False,
)
