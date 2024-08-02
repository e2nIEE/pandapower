# -*- coding: utf-8 -*-

import logging

from pandapower.converter.sincal.sincal2pp.sincal_utility import *

logger = logging.getLogger(__name__)

try:
    import pyodbc
except:
    logger.debug(
        "Could not import pyodbc, which is needed for importing directly from access db."
    )
import pandapower as pp


def sincal2pp(filename, use_gis_data=False, variant=1, variant_type="flag"):
    """
    Creates pandapower network from sincal file (.mdb - format).

    :param filename: sincal network (database.mdb file)
    :type filename: str
    :param use_gis_data: flag from where to choose geodata: (hr / hh) if True, else (NodeStartX,\
        NodeStartY)
    :type use_gis_data: bool, default False
    :param variant: Can be flag, ID or name of the variant to choose, default is flag=1
    :type variant: int, str, default 1
    :param variant_type: Can be either "flag", "id" or "name"
    :type variant_type: str, deafult "flag"
    :return: net - The converted pandapower net
    """

    logger.warning(
        "The SINCAL converter was changed in 11/2020 and does not show the exact same "
        "behavior as before, as some element indices might be altered in the created pp"
        " tables. The reason is the usage of bulk create functions. To retain the "
        "previous behavior, please use the function sincal2pp_old."
    )

    connection = create_connection(filename)
    net = pp.create_empty_network()

    # NODE ####################################################################################
    variant_id = query_variant_id(connection, variant, variant_type)
    d = query_node_dict(connection, variant_id)
    d["lfnoderesult"] = pd.read_sql(sql_results(variant_id), connection)
    # d["lfnoderesult"].set_index("Node_ID", inplace=True)

    ddf = pd.merge(
        d["node"], d["voltagelevel"], on="VoltLevel_ID", suffixes=["", "_vl"]
    )
    d["graphicnode"] = d["graphicnode"][~d["graphicnode"].Node_ID.duplicated()]
    ddf = pd.merge(
        ddf, d["graphicnode"], on="Node_ID", suffixes=["", "_gn"], how="left"
    )
    ddf = ddf.merge(
        d["lfnoderesult"].loc[:, ["Node_ID", "U_Un"]],
        left_on="Node_ID",
        right_on="Node_ID",
        suffixes=["", "_res"],
        how="left",
    )
    ddf["u_sincal_pu"] = ddf["U_Un"] / 100.0

    bus_type_dict = {1: "n", 2: "b", 3: "m", 4: "b", 5: "b"}
    bus_types = ddf["Flag_Type"].apply(
        lambda flag: bus_type_dict[flag] if flag in bus_type_dict else "n"
    )
    names = ddf["Name"].apply(lambda n: n.strip() if n is not None else "")
    if use_gis_data:
        bus_geo = ddf.loc[:, ["hr", "hh"]].values
    else:
        bus_geo = ddf.loc[:, ["NodeStartX", "NodeStartY"]].values
    pp.create_buses(
        net,
        len(ddf),
        name=names.values,
        index=ddf["Node_ID"].values,
        vn_kv=ddf["Un_vl"].values,
        type=bus_types.values,
        in_service=True,
        geodata=bus_geo,
        u_sincal_pu=ddf["u_sincal_pu"].values,
    )

    # join Node, Terminal and Element
    nt = pd.merge(d["node"], d["terminal"], on="Node_ID")
    node_terminal_element = pd.merge(
        nt, d["element"], on="Element_ID", suffixes=["", "_element"]
    )
    node_terminal_element.sort_values('Element_ID', inplace=True)

    # LINE ####################################################################################
    df = pd.read_sql(sql_line(variant_id), connection)
    df["name"] = df["name"].str.strip().values
    lines_as_switches = df["lty"] == 3
    switch_line_df = df.loc[lines_as_switches, :]
    if not switch_line_df.empty:
        df_breaker = pd.read_sql(sql_breaker(variant_id), connection)
        switch_line_df = switch_line_df.merge(df_breaker, left_on='_index', right_on='element', how='left')
        closed = (
                (switch_line_df["in_service_tt"] == 1)
                & (switch_line_df["in_service_ft"] == 1)
                & switch_line_df["in_service"]
                & (switch_line_df["in_service_b"] == 1)
        )
        pp.create_switches(
            net,
            buses=switch_line_df["from_bus"].values,
            et="b",
            elements=switch_line_df["to_bus"].values,
            closed=closed.values,
            name=switch_line_df["name"].values,
            # _index is the Element_ID for lines -> sql_line "SELECT e.Element_ID as _index, "
            Sincal_Element_ID=switch_line_df["_index"].values,
        )
    line_df = df.loc[~lines_as_switches, :]
    line_df["max_i_ka"] = line_df.max_i_ka.replace({0: np.nan})
    line_indices = pp.create_lines_from_parameters(
        net,
        line_df.from_bus.values,
        line_df.to_bus.values,
        #index=line_df["_index"],
        **{
            c: line_df[c].values
            for c in line_df.columns
            if c
               not in [
                   "_index",
                   "from_bus",
                   "to_bus",
                   "in_service_tt",
                   "in_service_ft",
                   "lty",
               ]
        },
    )

    line_df["pp_ind"] = line_indices
    add_switches_from = line_df.loc[
        line_df.in_service_ft == 0, ["pp_ind", "from_bus"]
    ].values
    add_switches_to = line_df.loc[
        line_df.in_service_tt == 0, ["pp_ind", "to_bus"]
    ].values
    add_switches = np.concatenate([add_switches_from, add_switches_to])
    pp.create_switches(
        net, add_switches[:, 1], et="l", elements=add_switches[:, 0], closed=0
    )

    pp.replace_zero_branches_with_switches(
        net, in_service_only=False, min_length_km=0.0001, drop_affected=True
    )
    # LINE_GEODATA ########################################################################
    terminal_graphics_from = pd.read_sql(sql_line_geodata(1, variant_id), connection)
    terminal_graphics_to = pd.read_sql(sql_line_geodata(2, variant_id), connection)
    df = pd.merge(
        terminal_graphics_from,
        terminal_graphics_to,
        on="Element_ID",
        suffixes=["_from", "_to"],
    )
    graphic_buckle_df = pd.read_sql("SELECT * FROM GraphicBucklePoint", connection)

    pos_cols = {c: i for i, c in enumerate(graphic_buckle_df.columns)}
    pos_x, pos_y = pos_cols["PosX"], pos_cols["PosY"]

    def get_positions(gt):
        return list(gt.values[:, [pos_x, pos_y]])

    valid_geo = df.loc[df["Element_ID"].isin(net.line.index)]
    from_buckle = (
        graphic_buckle_df.sort_values("NoPoint", ascending=False)
        .groupby("GraphicTerminal_ID")
        .apply(lambda gt: get_positions(gt))
    )
    from_buckle.name = "xy_from"
    to_buckle = (
        graphic_buckle_df.sort_values("NoPoint", ascending=True)
        .groupby("GraphicTerminal_ID")
        .apply(lambda gt: get_positions(gt))
    )
    to_buckle.name = "xy_to"
    valid_geo = valid_geo.join(from_buckle, how="left", on="GraphicTerminal_ID_from")
    valid_geo = valid_geo.join(to_buckle, how="left", on="GraphicTerminal_ID_to")
    valid_geo.xy_from.loc[pd.isnull(valid_geo.xy_from)] = [[]] * sum(
        pd.isnull(valid_geo.xy_from)
    )
    valid_geo.xy_to.loc[pd.isnull(valid_geo.xy_to)] = [[]] * sum(
        pd.isnull(valid_geo.xy_to)
    )
    valid_geo["coords"] = valid_geo.apply(
        lambda geo: [[geo["tx_from"], geo["ty_from"]]]
                    + geo.xy_from
                    + geo.xy_to
                    + [[geo["tx_to"], geo["ty_to"]]],
        axis=1,
    )

    net.line_geodata["coords"] = valid_geo["coords"].values
    net.line_geodata.index = valid_geo["Element_ID"].values

    # TwoWindingTransformer #####################################################################
    df = pd.read_sql(sql_trafo_2w(variant_id), connection)
    df["tap_side"] = df["Flag_ConNode"].apply(lambda fl: ["hv", "lv"][fl - 1])
    df["name"] = df["name"].str.strip()
    vg = get_vector_group(df['VecGrp'].values)
    df["vector_group"] = vg[0]
    df["shift_degree"] = vg[1]


    trafo_dict = {
        "hv_buses": "hv_bus",
        "lv_buses": "lv_bus",
        "vn_hv_kv": "Un1",
        "vn_lv_kv": "Un2",
        "sn_mva": "Sn",
        "vk_percent": "uk",
        "vkr_percent": "ur",
        "pfe_kw": "Vfe",
        "i0_percent": "i0",
        "shift_degree": "AddRotate",
        "tap_min": "rohl",
        "tap_neutral": "rohm",
        "tap_max": "rohu",
        "tap_pos": "roh",
        "tap_step_percent": "ukr",
        "tap_side": "tap_side",
        "name": "name",
        "in_service": "fs",
        "Sincal_Element_ID": "Element_ID",
        "vector_group": "vector_group",
        "shift_degree": "shift_degree"
    }
    pp.create_transformers_from_parameters(
        net, **{k: df[v].values for k, v in trafo_dict.items()}
    )

    net.trafo.tap_neutral = net.trafo.tap_neutral.fillna(0)

    # ThreeWindingTransformer ###################################################################
    df = pd.read_sql(sql_trafo_3w(variant_id), connection)

    pp.create_transformers3w_from_parameters(
        net,
        hv_buses=df.hv_bus.values,
        mv_buses=df.mv_bus.values,
        lv_buses=df.lv_bus.values,
        vn_hv_kv=df.Un1.values,
        vn_mv_kv=df.Un2.values,
        vn_lv_kv=df.Un3.values,
        sn_hv_mva=df.Sn12.values,
        sn_mv_mva=df.Sn23.values,
        sn_lv_mva=df.Sn31.values,
        vk_hv_percent=df.uK12.values,
        vk_mv_percent=df.uk23.values,
        vk_lv_percent=df.uk31.values,
        vkr_hv_percent=df.ur12.values,
        vkr_mv_percent=df.ur23.values,
        vkr_lv_percent=df.ur31.values,
        pfe_kw=df.Vfe.values,
        i0_percent=df.i0.values,
        shift_mv_degree=0,
        shift_lv_degree=0,
        tap_side="hv",
        tap_pos=df.roh1.values,
        tap_step_percent=df.uk1.values,
        tap_neutral=df.rohm1.values,
        tap_max=df.rohu1.values,
        tap_min=df.rohl1.values,
        name=df.name.str.strip().values,
        in_service=df.fs.values == 1,
        Sincal_Element_ID=df.Element_ID,
    )

    # Infeeder ########################################################################
    df = pd.read_sql(sql_infeeder(variant_id), connection)

    for ii, (_, i) in enumerate(df.iterrows(), 1):
        updt(len(df), ii, "ext_grids")

        name = i["name"].strip()
        if i["flf"] == 2:
            logger.info("creating load instead of ext_grid '%s'" % i["name"])
            pp.create_load(
                net,
                bus=i["bus"],
                p_mw=-i["P"],
                q_mvar=-i["Q"],
                sn_mva=math.sqrt(i["P"] ** 2 + i["Q"] ** 2),
                name=name,
                in_service=i["Flag_State"],
            )
        else:
            if i["flf"] == 3:
                u = i["u"] / 100.0
            elif i["flf"] == 6:
                u = i["Ug"] / net.bus.loc[i["bus"]].vn_kv
            else:
                raise UserWarning("Unspupported Flag_Lf for infeeder")
            pp.create_ext_grid(
                net, bus=i["bus"], vm_pu=u, name=name, in_service=i["Flag_State"],
            )
    # ShuntCondensator ##########################################################################
    df = pd.read_sql(sql_shunt(variant_id), connection)

    for ii, (_, i) in enumerate(df.iterrows(), 1):
        updt(len(df), ii, "ShuntCondensator")

        if i["Vdi"] > 0:
            raise UserWarning("ShuntCondensator with Vdi > 0")
        pp.create_shunt_as_capacitor(
            net, bus=i["bus"], q_mvar=i["Sn"], loss_factor=0, name=i["name"].strip()
        )
    # LOAD #####################################################################################
    nteu = pd.merge(node_terminal_element, d["load"], on="Element_ID", suffixes=("", "_nt"))
    nteu.Mpl_ID = pd.to_numeric(nteu.Mpl_ID)
    df = pd.merge(
        nteu, d["manipulation"], on="Mpl_ID", how="left", suffixes=["", "_mpl"]
    )

    power_vals = calc_pqsc_vec(df, net)
    pp.create_loads(
        net,
        buses=df["Node_ID"].values,
        p_mw=power_vals["p"].values,
        q_mvar=power_vals["q"].values,
        name=df["Name_element"].str.strip(),
        scaling=power_vals["scaling"].values,
        in_service=df["Flag_State"].values,
        Sincal_Element_ID=df["Element_ID"].values,
    )

    # PowerUnit ####################################################################################

    nteu = pd.merge(
        node_terminal_element, d["powerunit"], on="Element_ID", suffixes=["_nt", ""]
    )
    if not nteu.empty:
        nteu.Mpl_ID = pd.to_numeric(nteu.Mpl_ID)
        nteum = pd.merge(
            nteu, d["manipulation"], on="Mpl_ID", how="left", suffixes=["", "_mpl"]
        )

        # handle Error in _preserve_dtypes pandapower/auxiliary.py
        nteum["Description"].fillna(value="", inplace=True)
        nteum["TextVal_element"].fillna(value="", inplace=True)

        for i, (_, l) in enumerate(nteum.iterrows(), 1):
            updt(len(nteum), i, "power units")
            p_mw, q_mvar, scaling = calc_pqsc(l, net)

            if l["Flag_LfCtrl"]:
                bid = pp.create_bus(
                    net, name=l["Name_element"].strip() + "_inner", vn_kv=l["Un1"]
                )
                l = dict(l)
                l.update(
                    {
                        "hv_bus": l["Node_ID_nt"],
                        "lv_bus": bid,
                        "vn_hv_kv": l["Un2"],
                        "vn_lv_kv": l["Un1"],
                        "sn_mva": l["Sn"],
                        "vk_percent": l["uk"],
                        "vkr_percent": l["ur"],
                        "pfe_kw": l["Vfe"],
                        "i0_percent": l["pG"],
                        "Sincal_Element_ID": l["Element_ID"],
                    }
                )
                l["name"] = l["Name_element"].strip() + "_trafo"
                pp.create_transformer_from_parameters(
                    net, in_service=l["Flag_State_element"], **l,
                )
            else:
                bid = l["Node_ID_nt"]
            loid = pp.create_sgen(
                net,
                bus=bid,
                p_mw=p_mw,
                q_mvar=q_mvar,
                name=l["Name_element"].strip(),
                scaling=scaling,
                in_service=l["Flag_State_element"],
            )
            net.sgen.loc[loid, "Sincal_Element_ID"] = l["Element_ID"]

            if l["Name_mpl"] is not None:
                net.load.loc[loid, "mpl"] = l["Name_mpl"].strip()
    # DCInfeeder ###############################################################################
    df = pd.merge(node_terminal_element, d["dcinfeeder"], on="Element_ID", suffixes=("", "_nt"))
    if not df.empty:
        df.Mpl_ID = pd.to_numeric(nteu.Mpl_ID)

        # nteum = pd.merge(
        #    nteu, d["manipulation"], on="Mpl_ID", how="left", suffixes=["", "_mpl"]
        # )

        def in_service(info):
            return info["Flag_State"] and info["Flag_State_element"]

        # Sincal 15 added column Node_ID to DCInfeeder
        #id_column = "Node_ID_x" if "Node_ID" in d["dcinfeeder"].columns else "Node_ID"
        id_column = "Node_ID"
        for i, (_, l) in enumerate(df.iterrows(), 1):
            updt(len(df), i, "dc infeeder as sgen")
            if l["Flag_Connect"] == 2:
                bid = pp.create_bus(
                    net,
                    name=l["Name_element"].strip() + "_inner",
                    vn_kv=l["Ur_Inverter"],
                    in_service=l["Flag_State_element"],
                )

                # TODO: maybe the calculation can be adapted to find a more precise formulation
                #  for pfe and i0??
                S_mva = l["Tr_Sr"] / 1000
                # Z_SC_abs = l["Tr_uk"] * 10 / S_mva
                # R_SC = math.sqrt(Z_SC_abs ** 2 * l["Tr_rx"] ** 2 / (1 + l["Tr_rx"] ** 2))
                # X_SC = math.sqrt(Z_SC_abs ** 2 - R_SC ** 2)
                # Y = 1 / complex(R_SC, X_SC)
                # Vkr = R_SC * S_mva / 10
                # PFE = S_mva ** 2 * Y.real * 1000
                # i0 = 100 * abs(Y)
                u_k = l["Tr_uk"] / 100
                rx = l["Tr_rx"] / 100
                Vkr = 100 * math.sqrt(u_k ** 2 * rx ** 2 / (1 + rx ** 2))
                PFE = 0.0
                i0 = 0.0

                pp.create_transformer_from_parameters(
                    net,
                    hv_bus=l[id_column],
                    lv_bus=bid,
                    name=l["Name_element"].strip() + "_trafo_inner",
                    sn_mva=S_mva,
                    vn_lv_kv=l["Ur_Inverter"],
                    vn_hv_kv=l["Tr_UrNet"],
                    vk_percent=l["Tr_uk"],
                    vkr_percent=Vkr,
                    pfe_kw=PFE,
                    i0_percent=i0,
                    in_service=l["Flag_State_element"],
                    Sincal_Element_ID=l["Element_ID"],
                )
            else:
                bid = l[id_column]
            p_mw, q_mvar = calc_pqinv(l)
            dc_loid = pp.create_sgen(
                net,
                bid,
                p_mw=p_mw,
                q_mvar=q_mvar,
                sn_mva=math.sqrt(p_mw ** 2 + q_mvar ** 2),
                name=l["Name_element"].strip(),
                in_service=in_service(l),
            )
            # no **kwargs in create_sgen
            net.sgen.loc[dc_loid, "Sincal_Element_ID"] = l["Element_ID"]
    # SynchronousMachine ####################################################################
    df = pd.read_sql(sql_synchronous_machine(), connection)

    for i, (_, l) in enumerate(df.iterrows(), 1):
        updt(len(df), i, "sgens")
        p_mw, q_mvar, scaling = calc_pqsc(l, net)
        if l['phase'] == 7:
            loid = pp.create_sgen(
                net,
                l["bus"],
                p_mw=p_mw,
                q_mvar=q_mvar,
                name=l["name"].strip(),
                scaling=scaling,
                sn_mva=l["Sn"],
                in_service=l["in_service"],
            )
            net.sgen.loc[loid, "Sincal_Element_ID"] = l["Element_ID"]
            if l["mpl"] is not None:
                net.sgen.loc[loid, "mpl"] = l["mpl"].strip()
        else:
            suffix = 'a' if l['phase'] == 1 else 'b' if l['phase'] == 2 else 'c'
            loid = pp.create_asymmetric_sgen(
                net,
                l["bus"],
                name=l["name"].strip(),
                scaling=scaling,
                sn_mva=l["Sn"],
                in_service=l["in_service"],
            )
            net.asymmetric_sgen.loc[loid, 'p_%s_mw' % suffix] = p_mw
            net.asymmetric_sgen.loc[loid, 'q_%s_mvar' % suffix] = q_mvar
            net.asymmetric_sgen.loc[loid, "Sincal_Element_ID"] = l["Element_ID"]
            if l["mpl"] is not None:
                net.asymmetric_sgen.loc[loid, "mpl"] = l["mpl"].strip()
        # no **kwargs in create_sgen

    connection.close()
    del connection
    logger.info("done")
    return net


def validate_sincal_conversion(filename, net, bus_lookup=None):
    connection = create_connection(filename)

    sql = "SELECT * FROM LFNodeResult"
    df = pd.read_sql(sql, connection)

    def get_lookup(x):
        return x if bus_lookup is None else bus_lookup[x]

    vs = {get_lookup(d.Node_ID): d for i, d in df.iterrows()}
    if not vs:
        raise UserWarning("couldnt load loadflow results")
    pp.runpp(net, trafo_model="pi")
    md = max(
        abs(r.vm_pu - vs[b]["U_Un"] / 100.0)
        for b, r in net.res_bus.iterrows()
        if r.vm_pu != 1.0 and b in vs
    )
    print("maximal voltage difference: %s" % md)
    connection.close()

    return md

def get_vector_group(sincal_id):
    vector_group = np.zeros_like(sincal_id).astype(str)
    shift = np.zeros_like(sincal_id)
    mask_yy0 = sincal_id == 6
    mask_ynd5 = sincal_id == 26
    mask_yzn5 = sincal_id == 30
    mask_dyn5 = sincal_id == 24
    vector_group[mask_yy0] = 'yy'
    vector_group[mask_ynd5] = 'ynd'
    vector_group[mask_yzn5] = 'ynz'
    vector_group[mask_dyn5] = 'dyn'
    shift[mask_yy0] = 0
    shift[mask_ynd5] = 150
    shift[mask_yzn5] = 150
    shift[mask_dyn5] = 150
    return vector_group, shift

if __name__ == "__main__":
    pass
