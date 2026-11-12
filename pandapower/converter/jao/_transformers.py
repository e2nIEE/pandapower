# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

"""Creation of the transformers and their bus allocation from the Transformers sheet."""

import difflib
import logging
import re

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype

from pandapower.converter.jao._common import (
    _drop_duplicates_and_join_TSO,
    _get_bus_idx,
    _get_float_column,
)
from pandapower.converter.jao._locnames import (
    _normalize_locname,
    _parse_voltage_kv,
    _split_name_voltage,
)
from pandapower.create import create_buses, create_transformers_from_parameters
from pandapower.network import pandapowerNet

logger = logging.getLogger(__name__)


def _create_transformers_and_buses(net: pandapowerNet, data: dict[str, pd.DataFrame], **kwargs) -> None:
    """
    Creates transformers to the pandapower net using information from the transformers sheet (excel file).

    :param pandapowerNet net: net to be filled by buses
    :param dict[str, pd.DataFrame] data: data provided by the excel file which will be corrected
    """

    # --- data preparations
    key = "Transformers"
    bus_idx = _get_bus_idx(net)
    vn_hv_kv, vn_lv_kv = _get_transformer_voltages(data, bus_idx)
    trafo_connections = _allocate_trafos_to_buses_and_create_buses(net, data, bus_idx, vn_hv_kv, vn_lv_kv, **kwargs)
    max_i_a = (
        data[key]
        .loc[:, ("Maximum Current Imax (A) primary", "Fixed")]
        .fillna(data[key].loc[:, ("Maximum Current Imax (A) primary", "Max")])
    )
    sn_mva = np.sqrt(3) * max_i_a * vn_hv_kv / 1e3
    z_pu = vn_hv_kv**2 / sn_mva
    rk = data[key].xs("Resistance_R(Ω)", level=1, axis=1).values[:, 0] / z_pu
    xk = data[key].xs("Reactance_X(Ω)", level=1, axis=1).values[:, 0] / z_pu
    b0 = data[key].xs("Susceptance_B (µS)", level=1, axis=1).values[:, 0] * 1e-6 * z_pu  # type: ignore[operator]
    g0 = data[key].xs("Conductance_G (µS)", level=1, axis=1).values[:, 0] * 1e-6 * z_pu  # type: ignore[operator]
    zk = np.sqrt(rk**2 + xk**2)
    vk_percent = np.sign(xk) * zk * 100
    vkr_percent = rk * 100
    pfe_kw = g0 * sn_mva * 1e3
    i0_percent = 100 * np.sqrt(b0**2 + g0**2) * net.sn_mva / sn_mva
    taps = (
        data[key]
        .loc[:, ("Phase Shifting Properties", "Taps used for RAO")]
        .str.split(";", expand=True)
        .astype(int)
        .set_axis(["tap_min", "tap_max"], axis=1)
    )

    du = _get_float_column(data[key], ("Phase Shifting Properties", "Phase Regulation δu (%)"))
    dphi = _get_float_column(data[key], ("Phase Shifting Properties", "Angle Regulation δu (%)"))
    phase_shifter = np.isclose(du, 0) & (~np.isclose(dphi, 0))  # Symmetrical/Asymmetrical not considered

    _ = create_transformers_from_parameters(
        net,
        trafo_connections.hv_bus.values,  # type: ignore[arg-type]
        trafo_connections.lv_bus.values,  # type: ignore[arg-type]
        sn_mva,
        vn_hv_kv,
        vn_lv_kv,
        vkr_percent,
        vk_percent,
        pfe_kw,
        i0_percent,
        shift_degree=data[key].xs("Theta θ (°)", level=1, axis=1).values[:, 0],  # type: ignore[call-overload]
        tap_pos=0,
        tap_neutral=0,
        tap_side="lv",
        tap_min=taps["tap_min"].values,
        tap_max=taps["tap_max"].values,
        tap_phase_shifter=phase_shifter,
        tap_step_percent=du,
        tap_step_degree=dphi,
        name=data[key].loc[:, ("Location", "Full Name")].str.strip().values,
        EIC_Code=data[key].xs("EIC_Code", level=1, axis=1).values[:, 0],  # type: ignore[call-overload]
        TSO=data[key].xs("TSO", level=1, axis=1).values[:, 0],  # type: ignore[call-overload]
        Comment=data[key].xs("Comment", level=1, axis=1).replace("\xa0", "").values[:, 0],  # type: ignore[call-overload]
    )


def _get_transformer_voltages(data: dict[str, pd.DataFrame], bus_idx: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Returns the hv and lv nominal voltages of the transformers."""
    key = "Transformers"
    vn = data[key].loc[:, [("Voltage_level(kV)", "Primary"), ("Voltage_level(kV)", "Secondary")]].values
    vn_hv_kv = np.max(vn, axis=1)
    vn_lv_kv = np.min(vn, axis=1)
    if is_integer_dtype(list(bus_idx.index.dtypes)[1]):  # type: ignore[attr-defined]
        vn_hv_kv = vn_hv_kv.astype(int)
        vn_lv_kv = vn_lv_kv.astype(int)

    return vn_hv_kv, vn_lv_kv


def _allocate_trafos_to_buses_and_create_buses(
    net: pandapowerNet,
    data: dict[str, pd.DataFrame],
    bus_idx: pd.Series,
    vn_hv_kv: np.ndarray,
    vn_lv_kv: np.ndarray,
    rel_deviation_threshold_for_trafo_bus_creation: float = 0.2,
    log_rel_vn_deviation: float = 0.12,
    **kwargs,
) -> pd.DataFrame:
    """Provides a DataFrame of data to allocate transformers to the buses according to their
    location names. If locations of transformers do not exist due to the data of the lines and
    tielines sheets, additional buses are created. If locations exist but have a far different
    voltage level than the transformer, either a warning is logged or additional buses are created
    according to rel_deviation_threshold_for_trafo_bus_creation and log_rel_vn_deviation.

    :param pandapowerNet net: pandapower net
    :param dict[str, pd.DataFrame] data: _description_
    :param pd.Series bus_idx: Series of indices and corresponding location names and voltage levels in the MultiIndex of
        the Series
    :param np.ndarray vn_hv_kv: nominal voltages of the hv side of the transformers
    :param np.ndarray vn_lv_kv: Nominal voltages of the lv side of the transformers
    :param Optional[float] rel_deviation_threshold_for_trafo_bus_creation: If the voltage level of transformer locations
        is far different than the transformer data,
        additional buses are created. rel_deviation_threshold_for_trafo_bus_creation defines the
        tolerance in which no additional buses are created. (default: 0.2)
    :param Optional[float] log_rel_vn_deviation: This parameter allows a range below rel_deviation_threshold_for_trafo_bus_creation in which
        a warning is logged instead of a creating additional buses. (default: 0.12)

    :rtype: pd.DataFrame
    :return: information to which bus the trafos should be connected to. Columns are
        ["name", "hv_bus", "lv_bus", "vn_hv_kv", "vn_lv_kv", ...]
    """

    if rel_deviation_threshold_for_trafo_bus_creation < log_rel_vn_deviation:
        logger.warning(
            f"Given parameters violates the ineqation "
            f"{rel_deviation_threshold_for_trafo_bus_creation=} >= {log_rel_vn_deviation=}. "
            f"Therefore, rel_deviation_threshold_for_trafo_bus_creation={log_rel_vn_deviation} "
            "is assumed."
        )
        rel_deviation_threshold_for_trafo_bus_creation = log_rel_vn_deviation

    key = "Transformers"
    bus_location_names = set(net.bus.name)
    trafo_bus_names = data[key].loc[:, ("Location", "Full Name")]
    trafo_location_names = _find_trafo_locations(trafo_bus_names, bus_location_names)

    # --- construct DataFrame trafo_connections including all information on trafo allocation to
    # --- buses
    empties: np.typing.NDArray[np.int_] = -1 * np.ones(len(vn_hv_kv), dtype=int)
    trafo_connections = pd.DataFrame(
        {
            "name": trafo_location_names,
            "hv_bus": empties,
            "lv_bus": empties,
            "vn_hv_kv": vn_hv_kv,
            "vn_lv_kv": vn_lv_kv,
            "vn_hv_kv_next_bus": vn_hv_kv,
            "vn_lv_kv_next_bus": vn_lv_kv,
            "hv_rel_deviation": np.zeros(len(vn_hv_kv)),
            "lv_rel_deviation": np.zeros(len(vn_hv_kv)),
        }
    )
    trafo_connections[["hv_bus", "lv_bus"]] = trafo_connections[["hv_bus", "lv_bus"]].astype(np.int64)

    for side in ["hv", "lv"]:
        bus_col, trafo_vn_col, next_col, rel_dev_col, has_dev_col = (
            f"{side}_bus",
            f"vn_{side}_kv",
            f"vn_{side}_kv_next_bus",
            f"{side}_rel_deviation",
            f"trafo_{side}_to_bus_deviation",
        )
        name_vn_series = pd.Series(tuple(zip(trafo_location_names, trafo_connections[trafo_vn_col])))
        isin = name_vn_series.isin(bus_idx.index).to_numpy()
        has_name = trafo_location_names.notna().to_numpy()
        trafo_connections[has_dev_col] = ~isin
        trafo_connections.loc[isin, bus_col] = bus_idx.loc[name_vn_series.loc[isin]].values

        # --- unresolved locations: a new bus at the trafo's rated voltage is created later
        unresolved = ~has_name
        trafo_connections.loc[unresolved, next_col] = trafo_connections.loc[unresolved, trafo_vn_col].values
        trafo_connections.loc[unresolved, rel_dev_col] = np.inf

        # --- resolved but no exact (name, vn) match: code to find bus locations with vn deviation
        need_nearest = ~isin & has_name
        next_vn = np.array(
            [
                bus_idx.loc[tln.name].index.values[
                    (pd.Series(bus_idx.loc[tln.name].index) - getattr(tln, trafo_vn_col)).abs().idxmin()
                ]
                for tln in trafo_connections.loc[need_nearest, ["name", trafo_vn_col]].itertuples()
            ]
        )
        trafo_connections.loc[need_nearest, next_col] = next_vn
        rel_dev = np.abs(next_vn - trafo_connections.loc[need_nearest, trafo_vn_col].values) / next_vn
        trafo_connections.loc[need_nearest, rel_dev_col] = rel_dev
        trafo_connections.loc[need_nearest, bus_col] = bus_idx.loc[
            list(zip(trafo_connections.loc[need_nearest, "name"], trafo_connections.loc[need_nearest, next_col]))
        ].values

        # --- create buses to avoid too large vn deviations between nodes and transformers
        need_bus_creation = trafo_connections[rel_dev_col] > rel_deviation_threshold_for_trafo_bus_creation
        fallback_names = pd.Series(trafo_bus_names.astype(str).str.strip().to_numpy(), index=trafo_connections.index)
        new_bus_data = pd.DataFrame(
            {
                "vn_kv": trafo_connections.loc[need_bus_creation, trafo_vn_col].values,
                "name": trafo_connections.loc[need_bus_creation, "name"].fillna(fallback_names).values,
                "TSO": data[key].loc[need_bus_creation, ("Location", "TSO")].values,
            }
        )
        new_bus_data_dd = _drop_duplicates_and_join_TSO(new_bus_data)
        new_bus_idx = create_buses(
            net, len(new_bus_data_dd), vn_kv=new_bus_data_dd.vn_kv, name=new_bus_data_dd.name, zone=new_bus_data_dd.TSO
        )
        trafo_connections.loc[need_bus_creation, bus_col] = (
            net.bus.loc[new_bus_idx, ["name", "vn_kv"]]
            .reset_index()
            .set_index(["name", "vn_kv"])
            .loc[list(new_bus_data[["name", "vn_kv"]].itertuples(index=False, name=None))]
            .values
        )
        trafo_connections.loc[need_bus_creation, next_col] = trafo_connections.loc[
            need_bus_creation, trafo_vn_col
        ].values
        trafo_connections.loc[need_bus_creation, rel_dev_col] = 0
        trafo_connections.loc[need_bus_creation, has_dev_col] = False

    # --- create buses for trafos that are connected to the same bus at both sides (possible if
    # --- vn_hv_kv < vn_lv_kv *(1+rel_deviation_threshold_for_trafo_bus_creation) which usually
    # --- occurs for PSTs only)
    same_bus_connection = trafo_connections.hv_bus == trafo_connections.lv_bus
    duplicated_buses = net.bus.loc[trafo_connections.loc[same_bus_connection, "lv_bus"]].copy()
    duplicated_buses["name"] += " (2)"
    duplicated_buses.index = list(range(net.bus.index.max() + 1, net.bus.index.max() + 1 + len(duplicated_buses)))
    if same_bus_connection.any():
        trafo_connections.loc[same_bus_connection, "lv_bus"] = duplicated_buses.index
    net.bus = pd.concat([net.bus, duplicated_buses])
    if n_add_buses := len(duplicated_buses):
        tr_names = data[key].loc[trafo_connections.index[same_bus_connection], ("Location", "Full Name")]
        are_PSTs = tr_names.str.contains("PST")
        logger.info(
            f"{n_add_buses} additional buses were created to avoid that transformers are "
            f"connected to the same bus at both side, hv and lv. Of the causing "
            f"{len(tr_names)} transformers, {sum(are_PSTs)} contain 'PST' in their name. "
            f"According to this converter, the power flows over all these transformers will"
            f" end at the additional buses. Please consider to connect lines with the "
            f"additional buses, so that the power flow is over the (PST) transformers into "
            f"the lines."
        )

    # --- log according to log_rel_vn_deviation
    for side in ["hv", "lv"]:
        need_logging = trafo_connections.loc[trafo_connections[has_dev_col], rel_dev_col] > log_rel_vn_deviation
        if n_need_logging := sum(need_logging):
            max_dev = trafo_connections[rel_dev_col].max()
            idx_max_dev = trafo_connections[rel_dev_col].idxmax()
            logger.warning(
                f"For {n_need_logging} Transformers ({side} side), only locations were found (orig"
                f"in are the line and tieline data) that have a higher relative deviation than "
                f"{log_rel_vn_deviation}. The maximum relative deviation is {max_dev} which "
                f"results from a Transformer rated voltage of "
                f"{trafo_connections.at[idx_max_dev, trafo_vn_col]} and a bus "
                f"rated voltage (taken from Lines/Tielines data sheet) of "
                f"{trafo_connections.at[idx_max_dev, next_col]}. The best locations were "
                f"nevertheless applied, due to {rel_deviation_threshold_for_trafo_bus_creation=}"
            )

    if (
        not (trafo_connections.hv_bus > -1).all()
        or not (trafo_connections.lv_bus > -1).all()
        or not (trafo_connections.hv_bus != trafo_connections.lv_bus).all()
    ):
        raise AssertionError("A trafo")

    return trafo_connections


def _find_trafo_locations(trafo_bus_names, bus_location_names):
    """Identifies the bus location name for each transformer from its "Location / Full Name".

    The transformer names are reduced to the impeding-part-free location (joined and longest part)
    and compared to the bus location names on normalized keys. Remaining names are resolved by
    fuzzy matching (only clear best matches) and, if the name contains a voltage level, by matching
    the name base with that voltage. Unresolved transformers are returned as ``None``; the caller
    then creates a new bus at the transformer's rated voltage.

    Returns a Series of bus location names (or ``None``) aligned with ``trafo_bus_names``.
    """
    bus_location_names = sorted(set(bus_location_names))
    bus_key_to_name: dict[str, str] = {}
    for name in bus_location_names:
        bus_key_to_name.setdefault(_normalize_locname(name), name)

    # --- split strings at " " separators to remove impeding parts for identifying the location
    # --- names
    trafo_bus_names_expended = (
        trafo_bus_names.str.split(r"[ ]+|-A[0-9]+|-TD[0-9]+|-PF[0-9]+", expand=True).fillna("").replace(" ", "")
    )
    trafo_bus_names_expended_lower = (
        trafo_bus_names.str.lower()
        .str.split(r"[ ]+|-A[0-9]+|-TD[0-9]+|-PF[0-9]+", expand=True)
        .fillna("")
        .replace(" ", "")
    )

    # --- identify impeding parts
    contains_number = trafo_bus_names_expended.map(lambda x: any(char.isdigit() for char in x))
    to_drop = (
        (trafo_bus_names_expended_lower == "tr")
        | (trafo_bus_names_expended_lower == "pst")
        | (trafo_bus_names_expended == "")
        | (trafo_bus_names_expended == "/")
        | (trafo_bus_names_expended == "LIPST")
        | (trafo_bus_names_expended == "EHPST")
        | (trafo_bus_names_expended == "TFO")
        | (trafo_bus_names_expended_lower == "trafo")
        | (trafo_bus_names_expended_lower == "kv")
        | contains_number
    )
    trafo_bus_names_expended[to_drop] = ""

    # --- reconstruct name strings for identification and compare on normalized keys
    trafo_bus_names_joined = trafo_bus_names_expended.where(~to_drop).fillna("").agg(" ".join, axis=1).str.strip()
    trafo_bus_names_longest_part = trafo_bus_names_expended.apply(lambda row: max(row, key=len), axis=1)
    joined_keys = trafo_bus_names_joined.map(_normalize_locname)
    longest_keys = trafo_bus_names_longest_part.map(_normalize_locname)
    joined_in_buses = joined_keys.isin(bus_key_to_name)
    longest_part_in_buses = longest_keys.isin(bus_key_to_name)

    # --- resolve glued noise prefixes, e.g. "PSTMIKULOWA" -> "MIKULOWA"
    def _strip_glued_prefix(part: str) -> str:
        stripped = re.sub(r"^(?:PST|TR|TFO)(?=[A-Za-z]{3,})", "", part)
        return stripped if stripped != part else ""

    unresolved = ~(joined_in_buses | longest_part_in_buses)
    resolved_from_glued = pd.Series("", index=trafo_bus_names_longest_part.index)
    for idx in trafo_bus_names_longest_part.index[unresolved]:
        for part in [p for p in trafo_bus_names_expended.loc[idx] if p]:
            stripped_key = _normalize_locname(_strip_glued_prefix(part))
            if stripped_key in bus_key_to_name:
                resolved_from_glued.at[idx] = bus_key_to_name[stripped_key]
                break
    glued_in_buses = resolved_from_glued.astype(bool)

    # --- further attempts for still unresolved names: fuzzy matching and voltage-aware matching
    unresolved = ~(joined_in_buses | longest_part_in_buses | glued_in_buses)
    bus_by_base: dict[str, list[tuple[int | None, str]]] = {}
    for bus_key, bus_name in bus_key_to_name.items():
        base, voltage = _split_name_voltage(bus_key)
        bus_by_base.setdefault(base, []).append((voltage, bus_name))

    resolved_extra = pd.Series("", index=trafo_bus_names_longest_part.index)
    unresolved_names: list[str] = []
    if unresolved.any():
        bus_keys = list(bus_key_to_name)
        for idx in trafo_bus_names_longest_part.index[unresolved]:
            raw_name = trafo_bus_names.at[idx]
            search_key = longest_keys.at[idx] or _normalize_locname(raw_name)

            # --- fuzzy matching: accept only a clear best match
            scored = sorted(
                ((difflib.SequenceMatcher(None, search_key, key).ratio(), key) for key in bus_keys), reverse=True
            )
            best_ratio, best_key = scored[0]
            second_ratio = scored[1][0] if len(scored) > 1 else 0.0
            if best_ratio >= 0.85 and best_ratio - second_ratio >= 0.03:
                resolved_extra.at[idx] = bus_key_to_name[best_key]
                logger.info(
                    f"Transformer {raw_name!r} was matched to bus location {bus_key_to_name[best_key]!r} "
                    f"by fuzzy matching (similarity {best_ratio:.3f})."
                )
                continue

            # --- voltage-aware matching: same name base, bus voltage equals the voltage in the name
            voltage = _parse_voltage_kv(raw_name)
            if voltage is not None:
                base, _ = _split_name_voltage(search_key)
                candidates = sorted({name for v, name in bus_by_base.get(base, []) if v == voltage})
                if len(candidates) == 1:
                    resolved_extra.at[idx] = candidates[0]
                    logger.info(
                        f"Transformer {raw_name!r} was matched to bus location {candidates[0]!r} "
                        f"by name and voltage level ({voltage} kV)."
                    )
                    continue

            unresolved_names.append(raw_name)

    if unresolved_names:
        logger.warning(
            f"For {len(unresolved_names)} transformer(s), no bus location could be determined from the "
            f"transformer location names. A new bus at the transformer's rated voltage will be created for "
            f"each of them:\n" + "\n".join(f"  {name!r}" for name in unresolved_names)
        )

    # --- set the trafo location names (actual bus names) to be used for the bus allocation
    trafo_location_names = longest_keys.map(bus_key_to_name).where(longest_part_in_buses)
    trafo_location_names = trafo_location_names.where(~joined_in_buses, joined_keys.map(bus_key_to_name))
    trafo_location_names = trafo_location_names.where(~glued_in_buses, resolved_from_glued)
    trafo_location_names = trafo_location_names.where(~resolved_extra.astype(bool), resolved_extra)

    return trafo_location_names
