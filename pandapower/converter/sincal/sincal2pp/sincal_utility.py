# -*- coding: utf-8 -*-

import math
import pandas as pd
import numpy as np
import sys
import pandapower as pp

import logging

logger = logging.getLogger(__name__)

try:
    import pyodbc
except ImportError:
    logger.debug(
        "Could not import pyodbc, which is needed for importing directly from access db."
    )


def create_connection(filename):
    if "sqlite" in filename:
        import sqlite3

        return sqlite3.connect(filename)
    return pyodbc.connect(
        r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=%s" % filename
    )


def query_node_dict(connection, variant):
    d = dict()
    for v in [
        "Node",
        "Line",
        "Breaker",
        "Load",
        "VoltageLevel",
        "GraphicNode",
        "PowerUnit",
        "Element",
        "Terminal",
        "Manipulation",
        "DCInfeeder",
    ]:
        d[v.lower()] = pd.read_sql(
            "SELECT * FROM %s WHERE Variant_ID=%d" % (v, variant), connection
        )
    return d


def query_variant_id(connection, variant, variant_type):
    var_key = {"flag": "Flag_Variant", "id": "Variant_ID", "name": "Name"}[variant_type]
    var_df = pd.read_sql("SELECT * FROM VARIANT", connection)
    if isinstance(variant, str):
        cv = var_df.loc[var_df[var_key].str.strip() == variant]
    else:
        cv = var_df.loc[var_df[var_key] == variant]
    assert len(cv) == 1
    return cv["Variant_ID"].values[0]


def sql_results(variant_id):
    return "SELECT * FROM LFNodeResult WHERE Variant_ID=%s" % variant_id


def sql_line(variant_id):
    return (
        "SELECT e.Element_ID as _index, "
        "e.Name as name, "
        "fn.Node_ID as from_bus, "
        "tn.Node_ID as to_bus, "
        "l as length_km, "
        "r as r_ohm_per_km, "
        "x as x_ohm_per_km, "
        "c as c_nf_per_km, "
        "Ith as max_i_ka, "
        "e.Flag_State as in_service, "
        "ft.Flag_State as in_service_ft, "
        "tt.Flag_State as in_service_tt, "
        "l.Flag_LineTyp as lty, "
        "l.ParSys as parallel, "
        "l.fr as df "
        "FROM Line l, Element e, Node fn, Terminal ft, Node tn, Terminal tt "
        "WHERE "
        "e.Element_ID=l.Element_ID and "
        "ft.Node_ID=fn.Node_ID and "
        "tt.Node_ID=tn.Node_ID and "
        "ft.TerminalNo=1 and "
        "tt.TerminalNo=2 and "
        "tt.Element_ID=l.Element_ID and "
        "ft.Element_ID=l.Element_ID and "
        "l.Variant_ID=%d and "
        "e.Variant_ID=%d and "
        "fn.Variant_ID=%d and "
        "ft.Variant_ID=%d and "
        "tn.Variant_ID=%d and "
        "tt.Variant_ID=%d" % tuple([variant_id] * 6)
    )

def sql_breaker(variant_id):
    return (
        "SELECT b.Breaker_ID as _b_index, "
        "b.Name as b_name, "
        "b.Flag_State as in_service_b, "
        "l.Element_ID as element "
        "FROM Breaker b, Terminal t, Line l "
        "WHERE "
        "b.Terminal_ID=t.Terminal_ID and "
        "t.Element_ID=l.Element_ID and "
        "l.Variant_ID=%d and "
        "b.Variant_ID=%d and "
        "t.Variant_ID=%d" % tuple([variant_id] * 3)
    )

def sql_line_geodata(terminal_no, variant_id):
    return """
            SELECT
                l.Element_ID, gft.PosX as tx, gft.PosY as ty, ft.Terminal_ID,
                gft.GraphicTerminal_ID
            FROM
                Line l,
                Element e,
                Terminal ft,
                Node fn,
                GraphicTerminal gft
            WHERE
                l.Element_ID=e.Element_ID and
                ft.Element_ID=l.Element_ID and
                fn.Node_ID=ft.Node_ID and
                ft.TerminalNo=%s and
                gft.Terminal_ID=ft.Terminal_ID and
                l.Variant_ID=%d and
                e.Variant_ID=%d and
                ft.Variant_ID=%d and
                fn.Variant_ID=%d and
                gft.Variant_ID=%d
            """ % (
        terminal_no,
        *([variant_id] * 5),
    )


def sql_trafo_2w(variant_id):
    return """SELECT l.*, e.Name as name, fn.Node_ID as hv_bus, tn.Node_ID as lv_bus,
                e.Flag_State as fs
              FROM
                TwoWindingTransformer l,
                Element e,
                Node fn,
                Terminal ft,
                Node tn,
                Terminal tt
              WHERE
                e.Element_ID=l.Element_ID and
                ft.Node_ID=fn.Node_ID and
                tt.Node_ID=tn.Node_ID and
                ft.TerminalNo=1 and
                tt.TerminalNo=2 and
                tt.Element_ID=l.Element_ID and
                ft.Element_ID=l.Element_ID and
                l.Variant_ID=%d and
                e.Variant_ID=%d and
                fn.Variant_ID=%d and
                ft.Variant_ID=%d and
                tn.Variant_ID=%d and
                tt.Variant_ID=%d
            """ % tuple(
        [variant_id] * 6
    )


def sql_trafo_3w(variant_id):
    return (
        "SELECT l.*, e.Name as name, hn.Node_ID as hv_bus, "
        "mn.Node_ID as mv_bus, ln.Node_ID as lv_bus, e.Flag_State as fs "
        "FROM "
        "ThreeWindingTransformer l, "
        "Element e, "
        "Node hn, "
        "Terminal ht, "
        "Node mn, "
        "Terminal mt, "
        "Node ln, "
        "Terminal lt "
        "WHERE "
        "e.Element_ID=l.Element_ID and "
        "ht.Node_ID=hn.Node_ID and "
        "mt.Node_ID=mn.Node_ID and "
        "lt.Node_ID=ln.Node_ID and "
        "ht.TerminalNo=1 and "
        "mt.TerminalNo=2 and "
        "lt.TerminalNo=3 and "
        "ht.Element_ID=l.Element_ID and "
        "mt.Element_ID=l.Element_ID and "
        "lt.Element_ID=l.Element_ID and "
        "l.Variant_ID=%d and "
        "e.Variant_ID=%d and "
        "hn.Variant_ID=%d and "
        "ht.Variant_ID=%d and "
        "mn.Variant_ID=%d and "
        "mt.Variant_ID=%d and "
        "ln.Variant_ID=%d and "
        "lt.Variant_ID=%d" % tuple([variant_id] * 8)
    )


def sql_infeeder(variant_id):
    return """SELECT n.Node_ID as bus, u, Ug, P, Q, S, e.Name as name, i.Flag_Lf as flf,
                t.Flag_State
              FROM
                 Infeeder i,
                 Element E,
                 Node n,
                 Terminal t
             WHERE
                 E.Element_ID=i.Element_ID and
                 t.Node_ID=n.Node_ID and
                 t.Element_ID=i.Element_ID and
                 i.Variant_ID=%d and
                 E.Variant_ID=%d and
                 n.Variant_ID=%d and
                 t.Variant_ID=%d
            """ % tuple(
        [variant_id] * 4
    )


def sql_shunt(variant_id):
    return """SELECT n.Node_ID as bus, Sn, e.Name as name, Vdi
              FROM
                  ShuntCondensator s,
                  Element E,
                  Node n,
                  Terminal t
              WHERE
                  E.Element_ID=s.Element_ID and
                  t.Node_ID=n.Node_ID and
                  t.Element_ID=s.Element_ID and
                  s.Variant_ID=%d and
                  E.Variant_ID=%d and
                  n.Variant_ID=%d and
                  t.Variant_ID=%d
            """ % tuple(
        [variant_id] * 4
    )


def sql_synchronous_machine():
    return """SELECT t.Element_ID as Element_ID, n.Node_ID as bus, Flag_Lf, m.P, m.Q, m.fP, m.fQ, m.cosphi, m.S, m.fS, m.Sn,
               e.Name as name, t.Flag_State as in_service, t.Flag_Terminal as phase, mp.Name as mpl, ot.Name as got FROM
               (((((Node n INNER JOIN Terminal t ON  t.Node_ID=n.Node_ID)
               INNER JOIN Element e ON t.Element_ID=e.Element_ID)
               INNER JOIN SynchronousMachine m ON e.Element_ID=m.Element_ID)
               LEFT JOIN Manipulation mp ON m.Mpl_ID=mp.Mpl_ID)
               LEFT JOIN GraphicElement ge ON ge.Element_ID=e.Element_ID)
               LEFT JOIN GraphicObjectType ot ON ot.GraphicType_ID=ge.GraphicType_ID"""


def updt(total, progressn, what):
    """
    Displays or updates a console progress bar.

    Original source: https://stackoverflow.com/a/15860757/1391441
    """
    barLength, status = 20, "/ %s %s" % (total, what)
    progress = float(progressn) / float(total)
    if progress >= 1.0:
        progress, status = 1, "\r\n"
    block = int(round(barLength * progress))
    text = "\r[{}] {:.0f} {}".format(
        "#" * block + "-" * (barLength - block), progressn, status
    )
    sys.stdout.write(text)
    sys.stdout.flush()


def calc_pqsc(l, net):
    if l["Flag_Lf"] in [1, 2, 15]:
        if l["fP"] != l["fQ"] and l["Q"] != 0:
            raise UserWarning("fP and fQ differ !!")
        return l["P"], l["Q"], l["fP"]
    elif l["Flag_Lf"] in [3, 4]:
        return l["S"] * l["cosphi"], l["S"] * math.sqrt(1 - l["cosphi"] ** 2), l["fS"]
    elif l["Flag_Lf"] in [5, 6]:
        S = math.sqrt(3) * l["I"] * l["u"] / 100 * net.bus.vn_kv.at[l["Node_ID"]]
        return S * l["cosphi"], S * math.sqrt(1 - l["cosphi"] ** 2), l["fP"]
    elif l["Flag_Lf"] in [10, 11, 12]:
        if l["cosphi"] == 1:
            q = 0
        else:
            q = l["P"] / l["cosphi"] * math.sqrt(1.0 - l["cosphi"] ** 2)
        return l["P"], q, l["fP"]
    else:
        raise UserWarning("Unspupportet Flag_Lf")


def calc_pqsc_vec(df, net):
    all_flags = set(df["Flag_Lf"])
    res = pd.DataFrame(index=df.index, columns=["p", "q", "scaling"], dtype=float)

    isin_f1 = df["Flag_Lf"].isin([1, 2, 15])
    if np.any(isin_f1):
        if np.any((df["fP"] != df["fQ"]) & (df["Q"] != 0)):
            raise UserWarning("fP and fQ differ !!")
        res.loc[isin_f1, :] = df.loc[isin_f1, ["P", "Q", "fP"]].values
    isin_f2 = df["Flag_Lf"].isin([3, 4])
    if np.any(isin_f2):
        p = df.loc[isin_f2, "S"].values * df.loc[isin_f2, "cosphi"].values
        q = df.loc[isin_f2, "S"].values * np.sqrt(
            1 - df.loc[isin_f2, "cosphi"].values ** 2
        )
        res.loc[isin_f2, :] = list(zip(p, q, df.loc[isin_f2, "fS"].values))
    isin_f3 = df["Flag_Lf"].isin([5, 6])
    if np.any(isin_f3):
        s = (
            math.sqrt(3)
            * df.loc[isin_f3, "I"].values
            * df.loc[isin_f3, "u"].values
            / 100
            * net.bus.vn_kv.loc[df.loc[isin_f3, "Node_ID"].values].values
        )
        p = s * df.loc[isin_f3, "cosphi"].values
        q = s * np.sqrt(1 - df.loc[isin_f3, "cosphi"].values ** 2)
        res.loc[isin_f3, :] = list(zip(p, q, df.loc[isin_f3, "fP"].values))
    isin_f4 = df["Flag_Lf"].isin([10, 11, 12])
    if np.any(isin_f4):
        q = np.zeros(np.sum(isin_f4))
        not1 = (df["cosphi"].values != 1) & isin_f4
        q[not1[isin_f4.values]] = (
            df.loc[not1, "P"].values
            / df.loc[not1, "cosphi"].values
            * np.sqrt(1.0 - df.loc[not1, "cosphi"].values ** 2)
        )
        res.loc[isin_f4, :] = list(
            zip(df.loc[isin_f4, "P"].values, q, df.loc[isin_f4, "fP"].values)
        )
    if all_flags - set([1, 2, 3, 4, 5, 6, 10, 11, 12, 15]):
        raise UserWarning("Unspupportet Flag_Lf")
    return res


def calc_pqinv(l):
    if l["Flag_Lf"] in [1]:  # P und Q
        return l["P"] * l["fP"], l["Q"] * l["fQ"]
    elif l["Flag_Lf"] in [2]:  # P und cosphi
        return (
            l["P"] * l["fP"],
            l["P"] * l["fP"] / l["cosphi"] * math.sqrt(1 - l["cosphi"] ** 2),
        )  # WTF Sincal l["fQ"]
    elif l["Flag_Lf"] in [3]:  # Wechselrichter
        k = (1 - l["DC_losses"] / 100) * l["Eta_Inverter"] / 100 / 1000
        if l["DC_power"] < 0:
            k /= 1
        P = l["DC_power"] * l["fDC_power"] * k
        return P + l["Ctrl_power"] / 1e6, -abs(P) * (l["Q_Inverter"] / 100)
    else:
        raise UserWarning("Unspupportet Flag_Lf")
