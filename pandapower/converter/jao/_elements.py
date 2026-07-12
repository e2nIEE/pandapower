# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

"""
Creation of buses, lines and transformers from the corrected JAO sheets.

Buses are derived from the substations named in the Lines/Tielines sheets. Lines and tielines
are then created between those buses. Transformers are matched to their location by name and
voltage; where a suitable bus is missing (or its voltage deviates too much) new buses are
created.
"""

import difflib
import logging

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype

from pandapower.create import create_buses, create_lines_from_parameters, \
    create_transformers_from_parameters
from pandapower.io_utils import pandapowerNet
from pandapower.converter.jao import _schema
from pandapower.converter.jao._schema import SheetSchema
from pandapower.converter.jao._correction import normalize_trafo_name

logger = logging.getLogger(__name__)


# ==================================================================================================
# Small shared helpers
# ==================================================================================================

def get_bus_idx(net: pandapowerNet) -> pd.Series:
    """Series mapping ``(name, vn_kv) -> bus index``."""
    return net.bus[["name", "vn_kv"]].rename_axis("index").reset_index().set_index(
        ["name", "vn_kv"])["index"]


def _drop_duplicates_and_join_tso(bus_df: pd.DataFrame) -> pd.DataFrame:
    """Keep one bus per ``(name, vn_kv)``, joining the TSO strings of merged duplicates."""
    bus_df = bus_df.drop_duplicates(ignore_index=True)
    bus_df = bus_df.groupby(["name", "vn_kv"], as_index=False).agg({"TSO": lambda x: "/".join(x)})
    if bus_df.duplicated(["name", "vn_kv"]).any():
        raise AssertionError("bus_df contains duplicate names with identical vn_kv")
    return bus_df


def _get_float_column(df: pd.DataFrame, col_tuple: tuple, fill: float = 0) -> pd.Series:
    series = df.loc[:, col_tuple]
    series.loc[series == "\xa0"] = fill
    return series.astype(float).fillna(fill)


# ==================================================================================================
# Buses and lines
# ==================================================================================================

def create_buses_from_line_data(net: pandapowerNet, data: dict[str, pd.DataFrame]) -> None:
    """Create the buses named as substations in the Lines and Tielines sheets."""
    parts = []
    for key in ("Lines", "Tielines"):
        if key not in data:
            continue
        df = data[key]
        schema = SheetSchema(df)
        vn_col = schema.voltage_col()
        if vn_col is None:
            raise KeyError(f"{key}: no Voltage_level column found (fuzzy).")
        vn_kv = _schema.numeric_column(df, vn_col)
        for side in ("Substation_1", "Substation_2"):
            full = schema.fullname_col(side)
            if full is None:
                continue
            parts.append(pd.DataFrame({
                "name": schema.series(full).to_numpy(),
                "vn_kv": vn_kv,
                "TSO": schema.tso_series_for_side(side).to_numpy(),
            }))

    bus_df = pd.concat(parts, ignore_index=True) if parts else \
        pd.DataFrame({"name": [], "vn_kv": [], "TSO": []})
    bus_df = _drop_duplicates_and_join_tso(bus_df)
    new_bus_idx = create_buses(net, len(bus_df), vn_kv=bus_df.vn_kv, name=bus_df.name,
                               zone=bus_df.TSO)
    if not np.array_equal(new_bus_idx, bus_df.index):
        raise AssertionError("Created bus indices do not match the expected bus_df index.")


def create_lines(net: pandapowerNet, data: dict[str, pd.DataFrame],
                 max_i_ka_fillna: float) -> None:
    """Create lines and tielines between the previously created buses."""
    bus_idx = get_bus_idx(net)

    for key in ("Lines", "Tielines"):
        if key not in data:
            continue
        df = data[key]
        schema = SheetSchema(df)

        vn_col = schema.voltage_col()
        if vn_col is None:
            raise KeyError(f"{key}: Voltage_level column not found (fuzzy).")
        s1 = schema.fullname_col("Substation_1")
        s2 = schema.fullname_col("Substation_2")
        if s1 is None or s2 is None:
            raise KeyError(f"{key}: Substation_1/2 Full_name column not found (fuzzy).")

        vn_kvs = df.loc[:, vn_col].to_numpy()
        name1 = df[s1].astype(str).str.strip()
        name2 = df[s2].astype(str).str.strip()
        valid = (~pd.isna(vn_kvs)
                 & ~name1.isin(["", "NAN"]) & ~name2.isin(["", "NAN"]))
        n_invalid = len(df) - int(valid.sum())
        if n_invalid:
            logger.warning(f"{n_invalid} {key.lower()} were dropped due to missing or invalid data.")

        df = df[valid].copy()
        vn_kvs = vn_kvs[valid.to_numpy()]
        # rebuild the schema on the filtered rows so all column accessors align in length
        schema = SheetSchema(df)

        length_km = df[(_schema.ELECTRICAL_PARAMETERS, _schema.LENGTH)].to_numpy()
        zero_length = np.isclose(length_km, 0)
        no_length = np.isnan(length_km)
        if zero_length.any() or no_length.any():
            logger.warning(
                f"According to the data, {int(zero_length.sum())} {key.lower()} have zero length "
                f"and {int(no_length.sum())} have no length; both are set to 1 km.")
            length_km[zero_length | no_length] = 1

        from_bus = bus_idx.loc[list(zip(df[s1].astype(str), vn_kvs))].to_numpy()
        to_bus = bus_idx.loc[list(zip(df[s2].astype(str), vn_kvs))].to_numpy()

        r_ohm = _rxb_values(df, schema.resistance_col(),
                            (_schema.ELECTRICAL_PARAMETERS, _schema.RESISTANCE))
        x_ohm = _rxb_values(df, schema.reactance_col(),
                            (_schema.ELECTRICAL_PARAMETERS, _schema.REACTANCE))
        b_us = _rxb_values(df, schema.susceptance_col(), None)

        if _schema.LINE_IMAX in df.columns:
            i_ka = df[_schema.LINE_IMAX].fillna(max_i_ka_fillna * 1e3).to_numpy() / 1e3
        else:
            i_ka = np.full(len(df), max_i_ka_fillna)

        create_lines_from_parameters(
            net, from_bus, to_bus, length_km,
            r_ohm / length_km, x_ohm / length_km, b_us / length_km, i_ka,
            name=schema.string_values("NE name", tokens=["ne", "name"], default=None),
            EIC_Code=schema.string_values("EIC code", tokens=["eic", "code"], default=None),
            TSO=schema.line_tso_array(),
            Comment=schema.string_values("comment", tokens=["comment"], default=""),
            Tieline=(key == "Tielines"),
        )


def _rxb_values(df: pd.DataFrame, fuzzy_col: tuple | None,
                fallback_col: tuple | None) -> np.ndarray:
    """R/X/B values, preferring the fuzzy-matched column and falling back to the exact tuple."""
    if fuzzy_col is not None:
        return df.iloc[:, list(df.columns).index(fuzzy_col)].to_numpy()
    if fallback_col is not None and fallback_col in df.columns:
        return df[fallback_col].to_numpy()
    return np.zeros(len(df))


# ==================================================================================================
# Transformers
# ==================================================================================================

def create_transformers_and_buses(net: pandapowerNet, data: dict[str, pd.DataFrame],
                                  strict: bool = True, **kwargs) -> None:
    """Create transformers, matching them to buses and creating buses where necessary.

    Transformers whose location cannot be matched to a bus raise a ``ValueError`` when ``strict``
    is True (the default), or are skipped with a warning when ``strict`` is False.
    """
    df = data["Transformers"]
    schema = SheetSchema(df)

    bus_idx = get_bus_idx(net)
    vn_hv_kv, vn_lv_kv = _get_transformer_voltages(schema, bus_idx)
    trafo_connections, keep = _allocate_trafos_to_buses(
        net, schema, bus_idx, vn_hv_kv, vn_lv_kv, strict=strict, **kwargs)

    # restrict every per-transformer input to the transformers that were actually allocated
    if not keep.all():
        df = df.loc[keep].copy()
        schema = SheetSchema(df)
        vn_hv_kv = vn_hv_kv[keep]
        vn_lv_kv = vn_lv_kv[keep]

    max_fixed = pd.to_numeric(df.loc[:, ("Maximum Current Imax (A) primary", "Fixed")],
                              errors="coerce")
    max_max = pd.to_numeric(df.loc[:, ("Maximum Current Imax (A) primary", "Max")],
                            errors="coerce")
    max_i_a = np.asarray(max_fixed.fillna(max_max), dtype=float)

    vn_hv_arr = vn_hv_kv.astype(float)
    vn_lv_arr = vn_lv_kv.astype(float)
    sn_mva = np.sqrt(3.0) * max_i_a * vn_hv_arr / 1e3
    z_pu = vn_lv_arr ** 2 / sn_mva

    r_ohm = schema.numeric_values("resistance r ohm", tokens=["resistance"])
    x_ohm = schema.numeric_values("reactance x ohm", tokens=["reactance"])
    b_us = schema.numeric_values("susceptance b us", tokens=["susceptance", "b", "us"])
    g_us = schema.numeric_values("conductance g us", tokens=["conductance", "g", "us"])

    rk = r_ohm / z_pu
    xk = x_ohm / z_pu
    b0 = b_us * 1e-6 * z_pu
    g0 = g_us * 1e-6 * z_pu
    zk = np.sqrt(rk ** 2 + xk ** 2)
    vk_percent = np.sign(xk) * zk * 100
    vkr_percent = rk * 100
    pfe_kw = g0 * sn_mva * 1e3
    i0_percent = 100 * np.sqrt(b0 ** 2 + g0 ** 2) * net.sn_mva / sn_mva

    taps = df.loc[:, _schema.TAPS].str.split(";", expand=True).astype(int).set_axis(
        ["tap_min", "tap_max"], axis=1)
    du = _get_float_column(df, (_schema.PHASE_SHIFT_PROPERTIES, "Phase Regulation δu (%)"))
    dphi = _get_float_column(df, (_schema.PHASE_SHIFT_PROPERTIES, "Angle Regulation δu (%)"))
    phase_shifter = np.isclose(du, 0) & (~np.isclose(dphi, 0))

    comment = pd.Series(schema.string_values("comment", tokens=["comment"], default="")).replace(
        "\xa0", "").to_numpy()

    create_transformers_from_parameters(
        net,
        trafo_connections.hv_bus.values,
        trafo_connections.lv_bus.values,
        sn_mva, vn_hv_kv, vn_lv_kv, vkr_percent, vk_percent, pfe_kw, i0_percent,
        shift_degree=schema.numeric_values("theta degree", tokens=["theta"]),
        tap_pos=0, tap_neutral=0, tap_side="lv",
        tap_min=taps["tap_min"].values, tap_max=taps["tap_max"].values,
        tap_phase_shifter=phase_shifter,
        tap_step_percent=du, tap_step_degree=dphi,
        name=schema.transformer_location_series().values,
        EIC_Code=schema.string_values("eic code", tokens=["eic", "code"], default=None),
        TSO=schema.transformer_tso_series().values,
        Comment=comment,
    )


def _get_transformer_voltages(schema: SheetSchema,
                              bus_idx: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    col_p, col_s = schema.transformer_voltage_cols()
    if col_p is None or col_s is None:
        raise KeyError("Transformers: primary/secondary Voltage_level not found (fuzzy).")
    vn_p = _schema.to_numeric(schema.df.loc[:, col_p])
    vn_s = _schema.to_numeric(schema.df.loc[:, col_s])
    vn_hv_kv = np.maximum(vn_p, vn_s)
    vn_lv_kv = np.minimum(vn_p, vn_s)

    # keep the voltages integer if the bus index uses integer voltages
    try:
        if is_integer_dtype(list(bus_idx.index.dtypes)[1]):
            vn_hv_kv = vn_hv_kv.astype(int)
            vn_lv_kv = vn_lv_kv.astype(int)
    except Exception:
        pass
    return vn_hv_kv, vn_lv_kv


def _find_trafo_locations(trafo_bus_names: pd.Series,
                          bus_location_names: set[str]) -> tuple[pd.Series, pd.Series]:
    """Resolve each transformer location string to an existing bus location name.

    Each name is tokenized (keeping ``/`` intact, see :func:`normalize_trafo_name`) into a
    joined candidate and its longest token. Exact matches are preferred; otherwise a close
    difflib match (cutoff 0.8) is used.

    Returns ``(location_names, unmatched)`` where ``unmatched`` is a boolean Series flagging the
    transformers for which no suitable bus location name was found. The caller decides whether an
    unmatched transformer is a hard error or is skipped. The location string of an unmatched
    transformer is left at its longest token (unused unless the caller skips it).
    """
    normalized = trafo_bus_names.map(normalize_trafo_name)
    joined = normalized.map(lambda t: t[0])
    longest = normalized.map(lambda t: t[1])

    joined_hit = joined.isin(bus_location_names)
    longest_hit = longest.isin(bus_location_names)
    unmatched = ~(joined_hit | longest_hit)

    if unmatched.any():
        candidates = list(bus_location_names)
        for i in unmatched[unmatched].index:
            query = joined.at[i] or longest.at[i]
            if not query:
                continue
            match = difflib.get_close_matches(query, candidates, n=1, cutoff=0.8)
            if match:
                joined.at[i] = match[0]
                joined_hit.at[i] = True
                unmatched.at[i] = False

    location_names = longest.copy()
    location_names.loc[joined_hit] = joined.loc[joined_hit]
    return location_names, unmatched


def _report_unmatched_trafos(trafo_bus_names: pd.Series, unmatched: pd.Series) -> str:
    """Build a human-readable, one-per-line list of the unmatched transformer locations."""
    items = trafo_bus_names.loc[unmatched]
    listing = "\n".join(f"  - {name}" for name in items)
    return (f"For {int(unmatched.sum())} transformers, no suitable bus location names were found "
            f"(missing buses or inconsistent naming). Affected transformers:\n{listing}")


def _allocate_trafos_to_buses(net: pandapowerNet, schema: SheetSchema, bus_idx: pd.Series,
                              vn_hv_kv: np.ndarray, vn_lv_kv: np.ndarray,
                              rel_deviation_threshold_for_trafo_bus_creation: float = 0.2,
                              log_rel_vn_deviation: float = 0.12, strict: bool = True,
                              **kwargs) -> tuple[pd.DataFrame, np.ndarray]:
    """Allocate transformers to bus pairs by location and voltage, creating buses when needed.

    For each side, transformers are connected to the bus at their location with the matching
    voltage. If only a differing voltage exists, a new bus is created when the relative deviation
    exceeds ``rel_deviation_threshold_for_trafo_bus_creation`` (a warning is logged above
    ``log_rel_vn_deviation``). Transformers whose HV and LV side would land on the same bus get a
    duplicated LV bus (suffix ``" (2)"``).

    Transformers whose location cannot be matched to any bus raise a ``ValueError`` (listing all
    of them) when ``strict`` is True, or are dropped with a warning when ``strict`` is False.

    Returns ``(trafo_connections, keep)`` where ``keep`` is a boolean array over the original
    transformer rows telling the caller which transformers were kept (all True when strict).
    """
    if rel_deviation_threshold_for_trafo_bus_creation < log_rel_vn_deviation:
        logger.warning(
            f"Given parameters violate {rel_deviation_threshold_for_trafo_bus_creation=} >= "
            f"{log_rel_vn_deviation=}. Therefore, "
            f"rel_deviation_threshold_for_trafo_bus_creation={log_rel_vn_deviation} is assumed.")
        rel_deviation_threshold_for_trafo_bus_creation = log_rel_vn_deviation

    bus_location_names = set(net.bus.name)
    trafo_bus_names = schema.transformer_location_series()
    trafo_location_names, unmatched = _find_trafo_locations(trafo_bus_names, bus_location_names)

    # a transformer without a (valid) voltage on either side cannot be placed on a bus either
    no_voltage = pd.Series(
        np.isnan(vn_hv_kv.astype(float)) | np.isnan(vn_lv_kv.astype(float)),
        index=unmatched.index)
    unmatched = unmatched | no_voltage

    # TSO per transformer, kept aligned to trafo_connections (used when new buses are created)
    trafo_tso = schema.transformer_tso_series().reset_index(drop=True)

    keep = ~unmatched.to_numpy()
    if unmatched.any():
        message = _report_unmatched_trafos(trafo_bus_names, unmatched)
        if strict:
            raise ValueError(
                message + "\n\nPass strict=False to from_jao() to skip these transformers and "
                "still perform the conversion (the resulting network will be incomplete).")
        logger.warning(message + "\n\nThese transformers are skipped (strict=False).")
        # drop the unmatched transformers from every per-transformer input
        trafo_location_names = trafo_location_names.loc[keep].reset_index(drop=True)
        trafo_bus_names = trafo_bus_names.loc[keep].reset_index(drop=True)
        trafo_tso = trafo_tso.loc[keep].reset_index(drop=True)
        vn_hv_kv = vn_hv_kv[keep]
        vn_lv_kv = vn_lv_kv[keep]

    # use a clean 0..N-1 index so trafo_connections, trafo_tso and the vn arrays stay aligned
    trafo_location_names = trafo_location_names.reset_index(drop=True)
    empties = -1 * np.ones(len(vn_hv_kv), dtype=int)
    trafo_connections = pd.DataFrame({
        "name": trafo_location_names,
        "hv_bus": empties,
        "lv_bus": empties,
        "vn_hv_kv": vn_hv_kv,
        "vn_lv_kv": vn_lv_kv,
        "vn_hv_kv_next_bus": vn_hv_kv,
        "vn_lv_kv_next_bus": vn_lv_kv,
        "hv_rel_deviation": np.zeros(len(vn_hv_kv)),
        "lv_rel_deviation": np.zeros(len(vn_hv_kv)),
    })
    trafo_connections[["hv_bus", "lv_bus"]] = \
        trafo_connections[["hv_bus", "lv_bus"]].astype(np.int64)

    for side in ("hv", "lv"):
        bus_col = f"{side}_bus"
        trafo_vn_col = f"vn_{side}_kv"
        next_col = f"vn_{side}_kv_next_bus"
        rel_dev_col = f"{side}_rel_deviation"
        has_dev_col = f"trafo_{side}_to_bus_deviation"

        name_vn = pd.Series(tuple(zip(trafo_location_names, trafo_connections[trafo_vn_col])))
        isin = name_vn.isin(bus_idx.index)
        trafo_connections[has_dev_col] = ~isin
        trafo_connections.loc[isin, bus_col] = bus_idx.loc[name_vn.loc[isin]].values

        # for locations without the exact voltage, take the nearest available voltage
        next_vn = np.array([
            bus_idx.loc[row.name].index.values[
                (pd.Series(bus_idx.loc[row.name].index) - getattr(row, trafo_vn_col))
                .abs().idxmin()]
            for row in trafo_connections.loc[~isin, ["name", trafo_vn_col]].itertuples()])
        trafo_connections.loc[~isin, next_col] = next_vn
        rel_dev = np.abs(next_vn - trafo_connections.loc[~isin, trafo_vn_col].values) / next_vn
        trafo_connections.loc[~isin, rel_dev_col] = rel_dev
        trafo_connections.loc[~isin, bus_col] = bus_idx.loc[list(zip(
            trafo_connections.loc[~isin, "name"],
            trafo_connections.loc[~isin, next_col]))].values

        need_bus = trafo_connections[rel_dev_col] > rel_deviation_threshold_for_trafo_bus_creation
        if need_bus.any():
            new_bus_data = pd.DataFrame({
                "vn_kv": trafo_connections.loc[need_bus, trafo_vn_col].values,
                "name": trafo_connections.loc[need_bus, "name"].values,
                "TSO": trafo_tso.loc[need_bus].values,
            })
            new_bus_dd = _drop_duplicates_and_join_tso(new_bus_data)
            new_bus_idx = create_buses(net, len(new_bus_dd), vn_kv=new_bus_dd.vn_kv,
                                       name=new_bus_dd.name, zone=new_bus_dd.TSO)
            trafo_connections.loc[need_bus, bus_col] = net.bus.loc[
                new_bus_idx, ["name", "vn_kv"]].reset_index().set_index(["name", "vn_kv"]).loc[
                list(new_bus_data[["name", "vn_kv"]].itertuples(index=False, name=None))].values
            trafo_connections.loc[need_bus, next_col] = \
                trafo_connections.loc[need_bus, trafo_vn_col].values
            trafo_connections.loc[need_bus, rel_dev_col] = 0
            trafo_connections.loc[need_bus, has_dev_col] = False

    _duplicate_same_bus_connections(net, trafo_connections, trafo_location_names)

    for side in ("hv", "lv"):
        rel_dev_col = f"{side}_rel_deviation"
        has_dev_col = f"trafo_{side}_to_bus_deviation"
        next_col = f"vn_{side}_kv_next_bus"
        trafo_vn_col = f"vn_{side}_kv"
        need_logging = trafo_connections.loc[trafo_connections[has_dev_col],
                                             rel_dev_col] > log_rel_vn_deviation
        if n := int(need_logging.sum()):
            idx_max = trafo_connections[rel_dev_col].idxmax()
            logger.warning(
                f"For {n} transformers ({side} side), only locations with relative deviation > "
                f"{log_rel_vn_deviation} were found. Max deviation "
                f"{trafo_connections[rel_dev_col].max()} at "
                f"{trafo_connections.at[idx_max, trafo_vn_col]} kV vs bus "
                f"{trafo_connections.at[idx_max, next_col]} kV.")

    assert (trafo_connections.hv_bus > -1).all()
    assert (trafo_connections.lv_bus > -1).all()
    assert (trafo_connections.hv_bus != trafo_connections.lv_bus).all()
    return trafo_connections, keep


def _duplicate_same_bus_connections(net: pandapowerNet, trafo_connections: pd.DataFrame,
                                    trafo_location_names: pd.Series) -> None:
    """Give transformers whose HV and LV side share a bus a duplicated LV bus (suffix " (2)")."""
    same_bus = trafo_connections.hv_bus == trafo_connections.lv_bus
    duplicated = net.bus.loc[trafo_connections.loc[same_bus, "lv_bus"]].copy()
    if duplicated.empty:
        return
    duplicated["name"] += " (2)"
    start = net.bus.index.max() + 1
    duplicated.index = list(range(start, start + len(duplicated)))
    trafo_connections.loc[same_bus, "lv_bus"] = duplicated.index
    net.bus = pd.concat([net.bus, duplicated])

    tr_names = trafo_location_names.loc[same_bus]
    are_psts = tr_names.str.contains("PST")
    logger.info(
        f"{len(duplicated)} additional buses created to avoid same-bus transformers. "
        f"Of {len(tr_names)} transformers, {int(are_psts.sum())} contain 'PST'.")
