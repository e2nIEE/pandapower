import pandera.pandas as pa

schema = pa.DataFrameSchema(
    {
        "name": pa.Column(str, description="name of the impedance"),
        "from_bus": pa.Column(int, pa.Check.ge(0), description="index of bus where the impedance starts"),
        "to_bus": pa.Column(int, pa.Check.ge(0), description="index of bus where the impedance ends"),
        "rft_pu": pa.Column(
            float, pa.Check.gt(0), description="resistance of the impedance from ‘from’ to ‘to’ bus [p.u.]"
        ),
        "xft_pu": pa.Column(
            float, pa.Check.gt(0), description="reactance of the impedance from ‘from’ to ‘to’ bus [p.u.]"
        ),
        "rtf_pu": pa.Column(
            float, pa.Check.gt(0), description="resistance of the impedance from ‘to’ to ‘from’ bus [p.u.]"
        ),
        "xtf_pu": pa.Column(
            float, pa.Check.gt(0), description="reactance of the impedance from ‘to’ to ‘from’ bus [p.u.]"
        ),
        "rft0_pu": pa.Column(
            float,
            pa.Check.gt(0),
            description="zero-sequence resistance of the impedance from ‘from’ to ‘to’ bus [p.u.]",
        ),
        "xft0_pu": pa.Column(
            float, pa.Check.gt(0), description="zero-sequence reactance of the impedance from ‘from’ to ‘to’ bus [p.u.]"
        ),
        "rtf0_pu": pa.Column(
            float,
            pa.Check.gt(0),
            description="zero-sequence resistance of the impedance from ‘to’ to ‘from’ bus [p.u.]",
        ),
        "xtf0_pu": pa.Column(
            float, pa.Check.gt(0), description="zero-sequence reactance of the impedance from ‘to’ to ‘from’ bus [p.u.]"
        ),
        "gf_pu": pa.Column(float, pa.Check.gt(1), description="conductance at the ‘from_bus’ [p.u.]"),
        "bf_pu": pa.Column(float, pa.Check.gt(2), description="susceptance at the ‘from_bus’ [p.u.]"),
        "gt_pu": pa.Column(
            float, pa.Check.gt(3), description="conductance at the ‘from_bus’ [p.u.]"
        ),  # TODO: duplicated description?
        "bt_pu": pa.Column(
            float, pa.Check.gt(4), description="susceptance at the ‘from_bus’ [p.u.]"
        ),  # TODO: duplicated description?
        "gf0_pu": pa.Column(float, pa.Check.gt(1), description="zero-sequence conductance at the ‘from_bus’ [p.u.]"),
        "bf0_pu": pa.Column(float, pa.Check.gt(2), description="zero-sequence susceptance at the ‘from_bus’ [p.u.]"),
        "gt0_pu": pa.Column(float, pa.Check.gt(3), description="zero-sequence conductance at the ‘from_bus’ [p.u.]"),
        "bt0_pu": pa.Column(float, pa.Check.gt(4), description="zero-sequence susceptance at the ‘from_bus’ [p.u.]"),
        "sn_mva": pa.Column(
            float, pa.Check.gt(0), description="reference apparent power for the impedance per unit values [MVA]"
        ),
        "in_service": pa.Column(bool, description="specifies if the impedance is in service."),

        # neu (Kommentar kann nach kontrolle gelöscht werden)
        "kwargs": pa.Column(dict, description="Additional arguments (for additional columns in net.impedance table)")
    },
    strict=False,
)


res_schema = pa.DataFrameSchema(
    {
        "p_from_mw": pa.Column(float, description="active power flow into the impedance at “from” bus [MW]"),
        "q_from_mvar": pa.Column(float, description="reactive power flow into the impedance at “from” bus [MVAr]"),
        "p_to_mw": pa.Column(float, description="active power flow into the impedance at “to” bus [MW]"),
        "q_to_mvar": pa.Column(float, description="reactive power flow into the impedance at “to” bus [MVAr]"),
        "pl_mw": pa.Column(float, description="active power losses of the impedance [MW]"),
        "ql_mvar": pa.Column(float, description="reactive power consumption of the impedance [MVar]"),
        "i_from_ka": pa.Column(float, description="current at from bus [kA]"),
        "i_to_ka": pa.Column(float, description="current at to bus [kA]"),
    },
)
