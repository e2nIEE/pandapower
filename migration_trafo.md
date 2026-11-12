# Migration guide: transformer model with mandatory characteristic tables

This guide describes how to move pandapower from the 3.x transformer tap model to the
characteristic-table model of pandapower 4. Part A is for pandapower developers who implement
the change. Part B is for users who have pandapower 3 nets, scripts and controllers.

The design itself is in the planning document
[pandapower 4: Transformer model redesign - planning](https://claude.ai/artifact/TrVtBty9PuAvCDebZfjrjP).
This guide does not repeat the reasoning, only what has to be done.

**Status.** Nothing described here is implemented yet. Class, function and column names are the
working names from the planning document. Line numbers refer to branch `feature/imprv_rundcpp`
at commit `2b0b08d39`.

## Contents

- [1. Decisions and assumptions](#1-decisions-and-assumptions)
- [2. What changes](#2-what-changes)
- [3. The model](#3-the-model)
- [Part A: implementing the change](#part-a-implementing-the-change)
- [Part B: migrating nets and scripts](#part-b-migrating-nets-and-scripts)
- [Checklist](#checklist)

## 1. Decisions and assumptions

Decided:

| Decision | Consequence for the migration |
| --- | --- |
| The characteristic table is mandatory for every tap changer | The formula path in the power flow is deleted, not kept as a fallback |
| Slots are named by winding: `tap_hv`, `tap_lv` (2W), `tap_hv`, `tap_mv`, `tap_lv` (3W) | `tap_side` is removed; the old `tap`/`tap2` pair is distributed to slots by its side |
| A second changer on one winding goes into `tap2`, stored referred to the HV side | LV-side second changers are converted once; `pfe_scaling` and `i0_scaling` must be read by the power flow |
| Impedance is stored as scaling factors | Existing tables are converted from absolute values; `from_measurement_report` and `to_measurement_report` are part of the first release |
| Every transformer type gets a documentation page with formulas | Documentation is a deliverable of the migration, not a follow-up |

Assumed, because the planning document still lists them as open. **Each of these changes steps
below if decided differently.**

| # | Assumption | Affects |
| --- | --- | --- |
| A1 | Hard break. Old tap columns and old tap keyword arguments are removed in 4.0. Old nets are converted on load by `convert_format` | A4, A5, B2, B3 |
| A2 | `convert_format` reproduces the 3.x numbers. In particular a 3.x `Symmetrical` changer stays a quadrature booster after conversion | A4, B7 |
| A3 | The CGMES and UCTE converters switch to the true symmetrical formula for fresh imports. This is a separate, documented behaviour change | A8, B7 |
| A4 | No tap changer on a winding is a null reference and a null position, not a one-row characteristic | A1, A3, A4 |
| A5 | `tap_min`, `tap_max` and `tap_neutral` are derived from the table, not stored | A6, B4 |
| A6 | 3W star-point changers are baked into the characteristic by the builder; `tap_at_star_point` is removed | A4, A7 |
| A7 | Fractional tap positions interpolate linearly; the spline tables are removed | A3, A6, B7 |
| A8 | The 3W pair columns (`vk_hv_percent` and so on) keep their names | A7 |
| A9 | The zero-sequence factor columns exist in the schema with default 1.0, but 4.0 does not read them yet | A1 |
| A10 | Short-circuit mode keeps the 3.x behaviour: ratio and angle are ignored unless `use_pre_fault_voltage` is set, impedance factors are applied | A5 |

## 2. What changes

### `net.trafo`

| 3.x | 4.0 |
| --- | --- |
| `tap_pos` with `tap_side == "hv"` | `tap_hv_pos` |
| `tap_pos` with `tap_side == "lv"` | `tap_lv_pos` |
| `tap2_pos`, other side than `tap_side` | the other winding's `tap_*_pos` |
| `tap2_pos`, same side as `tap_side` | `tap2_pos`, characteristic referred to HV |
| `id_characteristic_table` | `tap_hv_characteristic`, `tap_lv_characteristic`, `tap2_characteristic` |
| `tap_side`, `tap_neutral`, `tap_min`, `tap_max`, `tap_step_percent`, `tap_step_degree`, `tap_changer_type` | removed, arguments of the builder |
| `tap2_side`, `tap2_neutral`, `tap2_min`, `tap2_max`, `tap2_step_percent`, `tap2_step_degree`, `tap2_changer_type` | removed, arguments of the builder |
| `tap_dependency_table` | removed, always true |
| `tap_phase_shifter`, `tap_dependent_impedance`, `vk_percent_characteristic`, `vkr_percent_characteristic` | removed (pre-3.0 legacy) |
| `id_characteristic_spline` | removed |
| `tapchanger_class`, `tapchanger_id`, `tapchanger2_class`, `tapchanger2_id` | `tapchanger_hv_class`, `tapchanger_hv_id`, `tapchanger_lv_class`, `tapchanger_lv_id` |

All other columns (`sn_mva`, `vn_hv_kv`, `vk_percent`, `shift_degree`, `oltc`, ...) are unchanged.

### `net.trafo3w`

| 3.x | 4.0 |
| --- | --- |
| `tap_pos` with `tap_side` | `tap_hv_pos`, `tap_mv_pos` or `tap_lv_pos` |
| `id_characteristic_table` | `tap_hv_characteristic`, `tap_mv_characteristic`, `tap_lv_characteristic` |
| `tap_side`, `tap_neutral`, `tap_min`, `tap_max`, `tap_step_percent`, `tap_step_degree`, `tap_changer_type`, `tap_at_star_point`, `tap_dependency_table` | removed |

### Tables

| 3.x | 4.0 |
| --- | --- |
| `net.trafo_characteristic_table`, 2W and 3W rows mixed | `net.trafo_characteristic_table` (2W) and `net.trafo3w_characteristic_table` (3W) |
| `id_characteristic` as index level or column, depending on who built the table | always a plain integer column |
| `step` | `tap_pos` |
| `voltage_ratio` | `voltage_ratio` |
| `angle_deg` | `angle_degree` |
| `vk_percent`, `vkr_percent` | `vk_scaling`, `vkr_scaling` |
| `vk_hv_percent` ... `vkr_lv_percent` | `vk_hv_scaling` ... `vkr_lv_scaling`, in the 3W table |
| none | `pfe_scaling`, `i0_scaling`, `vk0_scaling`, `vkr0_scaling` (and the 3W pair versions) |
| `net.trafo_characteristic_spline` | removed |

### Functions

| 3.x | 4.0 |
| --- | --- |
| `create_trafo_characteristic(net, characteristic_values)` | `TapCharacteristic.tabular(...).add_to(net)` |
| `pandapower.control.trafo_characteristic_table_diagnostic` | removed, the pandera schema validates |
| `pandapower.control.create_trafo_characteristic_object` | removed, interpolation is built in |
| none | `TapCharacteristic` with builders, `from_measurement_report`, `to_measurement_report` |
| none | accessors for tap range and step size |

## 3. The model

Each slot delivers, at its tap position, a ratio `rho`, an angle `alpha` and a factor `k` per
nameplate value. An empty slot delivers `rho = 1`, `alpha = 0`, `k = 1`.

```text
V_hv   = vn_hv_kv * rho_hv * rho_2
V_lv   = vn_lv_kv * rho_lv
theta  = shift_degree + alpha_hv + alpha_2 - alpha_lv
q_eff  = q * k_hv * k_lv * k_2        for q in vk_percent, vkr_percent, pfe_kw, i0_percent
```

From there the 3.x computation is unchanged. It already scales the series impedance with
`(V_lv / V_bus_lv)^2` and the magnetizing admittance with `(vn_lv_kv / V_lv)^2`.

A fractional position `p` between table positions `p0` and `p1` interpolates every column
linearly. A position outside the characteristic is an error.

A second changer that physically sits on the LV winding is stored in `tap2` as:

```text
rho_2   = 1 / rho          alpha_2  = -alpha
k_vk_2  = k_vk * rho^2     k_vkr_2  = k_vkr * rho^2
k_pfe_2 = k_pfe / rho^2    k_i0_2   = k_i0 / rho^2
```

For a 3W the same equations apply to each of the three legs after the split at the star bus.

---

# Part A: implementing the change

The work is split into steps that can each be reviewed and merged on their own. Steps A1 to A3
add code without touching behaviour. Step A5 is the switch.

## A0. Before starting: freeze reference results

The regression rule is: after conversion, every net gives the same power flow results as before.
That needs reference results taken from the unchanged code.

1. On the base commit, run `runpp` for every net in `pandapower.networks` and every net file
   under `test/test_files` and `test/converter/testfiles` that contains transformers.
2. For each net, also sweep each tap changer over its full range (`tap_min` to `tap_max`),
   one transformer at a time, with `calculate_voltage_angles=True`.
3. Repeat for `trafo_model="pi"` and `trafo_model="t"`, and for `runpp_3ph` and
   `calc_sc` where the net supports it.
4. Store `res_bus`, `res_trafo` and `res_trafo3w` per case.

Existing fixtures that already serve this purpose and must keep passing:
`test/test_files/test_trafo3w_tap.json` with
`test/test_files/test_results_files/trafo_3w_tap_results.csv` and
`trafo_3w_tap_results_neutral_not_zero.csv`, `test/converter/testfiles/trafo_tap_model.json`
and `test/converter/testfiles/test_ucte_res_trafo.csv`.

Three 3.x defects make exact equality impossible in specific situations. Write a failing test
for each before the migration, so the change in results is intentional and documented:

| Defect in 3.x | Where | Result after migration |
| --- | --- | --- |
| Two transformers that share an `id_characteristic` but have different tap positions get the same table values | `build_branch.py:790-798`, `1040-1041` | Each transformer gets the row of its own position |
| A (`id_characteristic`, `step`) pair missing from the table silently yields ratio 1, angle 1 degree, `vk_percent` 1 | `.get(id_val, 1)` in the same places | Error |
| A fractional `tap_pos` never matches a table row | same | Linear interpolation |

## A1. Schemas

Files: new `pandapower/network_schema/trafo_characteristic.py`, `pandapower/network_schema/__init__.py`,
`pandapower/network_structure.py`.

1. Add `trafo_characteristic_table_schema` and `trafo3w_characteristic_table_schema`.
   There is no schema for the table today (`TODO` at `create/trafo_characteristic_create.py:48`).
2. Column rules: `id_characteristic` int, `tap_pos` int, `voltage_ratio` float `> 0`,
   `angle_degree` float, all `*_scaling` columns float `> 0` with `metadata={"default": 1.0}`.
   No column is nullable.
3. Frame-level checks: (`id_characteristic`, `tap_pos`) unique, and rows sorted by that pair.
4. Register both tables where `trafo_schema` is registered (`network_structure.py:62`, `221`).

Do not change `network_schema/trafo.py` or `trafo3w.py` yet. That happens in A5.

Done when: schema tests in `test/network_schema/elements/` cover valid tables, duplicates,
unsorted rows, nulls and non-positive ratios.

## A2. Utility class

File: new module, for example `pandapower/trafo_characteristic.py`. It must not import from
`build_branch.py`.

1. `TapCharacteristic` holds the rows of one characteristic as numpy arrays.
2. Builders, one per transformer case of the planning document:

   | Builder | Formula |
   | --- | --- |
   | `ratio(tap_min, tap_max, tap_neutral, step_percent)` | `rho = 1 + dn * du` |
   | `ideal(tap_min, tap_max, tap_neutral, step_degree)` | `rho = 1`, `alpha = dn * step_degree` |
   | `symmetrical(tap_min, tap_max, tap_neutral, step_percent)` | `rho = 1`, `alpha = 2 * atan(dn * du / 2)` |
   | `asymmetrical(tap_min, tap_max, tap_neutral, step_percent, theta_degree=90)` | see below |
   | `tabular(tap_pos, voltage_ratio, angle_degree, **scaling)` | rows as given |
   | `from_measurement_report(report, nameplate, measured_side)` | true values to factors |
   | `from_deviation(tap_pos, ratio, angle, r, x, g, b, nameplate)` | CGMES and PowSyBl percent deviations |
   | `from_pandapower3(row)` | the 3.x formulas, see A4 |

   with `dn = tap_pos - tap_neutral` and `du = step_percent / 100`.

   For `asymmetrical`, decide and document the ratio convention explicitly. ENTSO-E and UCTE
   state the ratio as a reciprocal (`1/rho = sqrt((dn*du*sin(theta))^2 + (1 + dn*du*cos(theta))^2)`).
   pandapower 3 multiplies the rated voltage of the tap winding by that square root. The stored
   `voltage_ratio` must follow the pandapower convention of section 3.
3. Transforms: `referred_to_hv()` (section 3), `at_star_point(nameplate)` for 3W,
   `with_reactance_law(x_zero, x_max, kind)`.
4. Queries: `tap_min`, `tap_max`, `tap_neutral`, `step_size_at(tap_pos)`.
5. `to_measurement_report(nameplate)`, the inverse of `from_measurement_report`.
6. `add_to(net)` appends the rows and returns the `id_characteristic`. If an identical
   characteristic already exists, it returns that id instead. It must keep the table sorted.
7. `validate()` applies the same rules as the schema of A1.

Done when:
- each builder is tested against the formula of its source (ENTSO-E guide, UCTE-DEF section 4);
- `from_measurement_report` followed by `to_measurement_report` returns the input;
- `referred_to_hv()` is tested through a power flow: one net with the second changer on LV
  built the 3.x way, one with `tap2`, equal results for `trafo_model` `pi` and `t`.

## A3. Lookup function

File: the module of A2 or `build_branch.py`.

```python
def lookup_characteristic(table, ids, tap_pos, columns) -> dict[str, np.ndarray]:
    """Return one value per transformer for each requested column."""
```

1. Build a combined sort key from `id_characteristic` and `tap_pos` of the table once per call.
2. `numpy.searchsorted` with the key of each transformer gives the upper neighbour row.
3. Exact hit: take the row. Fractional position: interpolate between the two neighbours of the
   same characteristic.
4. Null `ids`: return 1.0 for ratio and factors, 0.0 for the angle.
5. Raise, naming the transformer indices, when a reference does not exist in the table, when a
   reference is set but the position is null, or when the position is outside the characteristic.
6. Before the search, assert cheaply that the key is non-decreasing. An unsorted table would
   otherwise give wrong results without an error.

No pandas merge, no dict, no object dtype.

Done when: unit tests cover exact hits, interpolation, null slots, all three error cases, an
unsorted table, and two transformers sharing one characteristic at different positions.

## A4. `convert_format`

File: `pandapower/convert_format.py`.

**Version gate.** `__format_version__` is already `"4.0.0"` on `develop`
(`pandapower/_version.py`), and a `< 4.0.0` block exists (`convert_format.py:68`). Nets saved
from the current `develop` therefore carry format 4.0.0 with the old tap columns. Bump the
format version for this change, and additionally detect old nets by the presence of `tap_side`
in `net.trafo` or `net.trafo3w`.

**Order.** Run the new conversion after the existing steps that normalise older nets:
`convert_trafo_pst_logic` (`tap_phase_shifter` to `tap_changer_type`) and
`_update_characteristics` (`vk_percent_characteristic` columns to a table).

**Per two-winding transformer:**

1. Decide which 3.x changers are active.

   | 3.x state | Meaning |
   | --- | --- |
   | `tap_dependency_table` True | Table changer. The table is used whatever `tap_changer_type` says |
   | `tap_changer_type` is `Ratio` or `Symmetrical`, flag not True | Formula changer, complex formula |
   | `tap_changer_type` is `Ideal`, flag not True | Formula changer, ideal formula |
   | `tap_changer_type` null, or `Tabular` without the flag | No changer, even if `tap_pos` is set |

   `tap2` can only be a formula changer.
2. Build the characteristic with `TapCharacteristic.from_pandapower3`:

   | 3.x changer | Rows |
   | --- | --- |
   | `Ratio`, `Symmetrical` | `voltage_ratio = sqrt((1 + d*cos(t))^2 + (d*sin(t))^2)`, `angle_degree = atan(d*sin(t) / (1 + d*cos(t)))` with `d = dn * tap_step_percent / 100`, `t = tap_step_degree`; NaN counts as 0 |
   | `Ideal` with `tap_step_degree` | `voltage_ratio = 1`, `angle_degree = dn * tap_step_degree` |
   | `Ideal` with `tap_step_percent` | `voltage_ratio = 1`, `angle_degree = 2 * asin(dn * tap_step_percent / 100 / 2)` |
   | Table | `voltage_ratio` and `angle_degree` from the rows; `vk_scaling = vk_percent_row / vk_percent_nameplate`, same for `vkr` |

   These are the formulas of `_get_trafo_shift` (`build_branch.py:850-978`). Do not "correct"
   them here (assumption A2).
3. Positions run from `tap_min` to `tap_max`. See the edge cases below when they are missing.
4. Assign slots.

   | 3.x | Slot |
   | --- | --- |
   | `tap` only, on side `s` | `tap_<s>` |
   | `tap` on `s1`, `tap2` on the other side | `tap_<s1>` and `tap_<s2>` |
   | `tap` and `tap2` on HV | `tap_hv` and `tap2` |
   | `tap` and `tap2` on LV | `tap_lv` and `tap2` after `referred_to_hv()` |
5. `add_to(net)` each characteristic and write position and reference into the slot columns.

**Existing 3.x tables.** A 3.x table holds absolute impedances, so one `id_characteristic` can
be referenced by transformers with different nameplates. Group the referencing transformers by
(`id_characteristic`, `vk_percent`, `vkr_percent`) and create one 4.0 characteristic per group.
Normalise the table to plain columns first; it may arrive with a MultiIndex
(`create_trafo_characteristic`) or with columns (CIM and PowerFactory converters).

**Per three-winding transformer:**

1. `tap_side` selects the leg slot.
2. Build the characteristic as for 2W. The 3W table rows carry six pair impedances; convert each
   to its pair factor against the nameplate pair value.
3. If `tap_at_star_point` is True, apply `at_star_point(nameplate)`. 3.x implements star-point
   changers in two places: the formula path rewrites the step (`build_branch.py:1890-1900`), the
   table path inverts ratio and angle (`build_branch.py:800-809`). Both must be reproduced.
   `doc/elements/trafo.rst:259` claims the flag has no effect with a table; the code is the reference.
4. Write the rows to `net.trafo3w_characteristic_table`.

**Edge cases.** Decide each one explicitly and test it.

| Case in a 3.x net | Suggested handling |
| --- | --- |
| `tap_min` or `tap_max` is NaN | Use the range spanned by `tap_pos` and `tap_neutral`, warn |
| `tap_pos` outside `tap_min`..`tap_max` (3.x never checked) | Extend the characteristic to include it, warn |
| `tap_neutral` is NaN | 3.x computes NaN here; treat as no changer, warn |
| `tap_pos` is NaN with an active changer | No changer on that slot, warn |
| `Ideal` with both step values set | 3.x raises in `runpp`; raise in the conversion with the transformer index |
| Table lacks the row of the current `tap_pos` | Keep the table as it is, warn; `runpp` will raise until the user fixes it |
| `tap_dependency_table` True but `id_characteristic_table` null | 3.x raises in `runpp`; raise in the conversion |
| Shared `id_characteristic` at different positions | Converted correctly; results differ from 3.x because 3.x was wrong (A0) |

**Finally.** Drop every removed column of section 2, delete `net.trafo_characteristic_spline`,
rename the CIM bookkeeping columns, and extend `_rename_columns` and `_add_missing_columns`
accordingly.

Done when: `test/api/test_convert_format.py` loads the frozen 3.x nets of A0 and reproduces
their results within tolerance, with the three documented exceptions.

## A5. The switch: element schemas, create functions, std types, power flow

These parts cannot be merged separately, because the columns cannot have both meanings at once.

**Element schemas** (`network_schema/trafo.py`, `network_schema/trafo3w.py`):
remove the tap columns of section 2, add the slot columns (`tap_*_pos` float, nullable;
`tap_*_characteristic` `Int64`, nullable), replace the `tap_columns` and `tap2_columns` group
checks and the `tdt` dependency check by one check per slot: position and reference are both
null or both set. While there, fix the issues listed in the planning document: `loss_side`
missing from the 3W schema, `vk0_*_percent` described as not relevant although the
zero-sequence code reads them, `sn_*_mva` described in kVA.

**Create functions** (`create/trafo_create.py`, eight functions from `create_transformer` at
line 45 to `create_transformers3w_from_parameters` at line 1241):

- New arguments per slot: `tap_hv`, `tap_hv_pos`, `tap_lv`, `tap_lv_pos`, `tap2`, `tap2_pos`
  (2W) and `tap_hv`, `tap_mv`, `tap_lv` with positions (3W). `tap_*` accepts a
  `TapCharacteristic` or an existing `id_characteristic`.
- Remove the old tap arguments (assumption A1). Give a clear error that names the builder to use.
- `create_transformer` and `create_transformer3w` (std type based) call `ratio(...)` or the
  matching builder with the std type's parameters. `add_to` deduplicates, so all transformers of
  one std type share one characteristic.
- Delete `create/trafo_characteristic_create.py` or reduce it to a thin wrapper around `tabular`.

**Std types** (`std_types.py`): the library keeps `tap_side`, `tap_neutral`, `tap_min`,
`tap_max`, `tap_step_percent`, `tap_step_degree` and `tap_changer_type`, because they are
builder input. Remove the unused `"trafo_characteristic_table": False` entries (from line 1035).
Check `parameter_from_std_type` and `change_std_type`: they copy std type values into element
columns that no longer exist.

**Power flow** (`build_branch.py`):

| Function | Action |
| --- | --- |
| `_calc_tap_from_dataframe` (724) | Replace the body by one `lookup_characteristic` call per slot and the equations of section 3 |
| `_get_trafo_shift` (850) | Delete; its formulas live in `from_pandapower3` |
| `_get_vk_values_from_table` (982), `_get_vk_values` (1052), `_calc_tap_dependent_value` (1130) | Delete |
| `_calc_r_x_from_dataframe` (1152) | Multiply `vk_percent` and `vkr_percent` by the looked-up factors; remove the table and dict branches |
| `_calc_y_from_dataframe` (685) | Multiply `pfe_kw` and `i0_percent` by the looked-up factors |
| `_calculate_3w_tap_changers` (1868) | Delete |
| `_trafo_df_from_trafo3w` (1690) | Pass the three leg slots through; no side remapping, no `tap_at_star_point` |
| `_calculate_sc_voltages_of_equivalent_transformers` (1766) | Multiply the pair values by the pair factors before the delta-star conversion |
| `get_trafo_values` (496) | The NA replacement and object-dtype round trip are no longer needed for tap columns |

Keep two 3.x behaviours unless decided otherwise: in short-circuit mode without
`use_pre_fault_voltage` ratio and angle are skipped (`build_branch.py:750`) while impedance
factors still apply (assumption A10), and with `calculate_voltage_angles=False` only
`shift_degree` is zeroed while tap angles are still added.

Do the lookup once per element and pass the arrays down. Today `_calc_tap_from_dataframe`,
`_calc_r_x_from_dataframe` and `_calculate_sc_voltages_of_equivalent_transformers` each query
the table on their own.

Done when: the regression suite of A0 passes on converted nets, `test/loadflow/test_runpp.py`
(including `test_tap_table_order`), `test/create/test_trafo_create.py` and
`test/shortcircuit/test_trafo3w.py` are ported, and a `runpp` benchmark on a large net is
recorded before and after.

## A6. Consumers of the removed columns

| Consumer | Location | Change |
| --- | --- | --- |
| Tap controllers | `control/controller/trafo_control.py:105-127`, `156-180` | Read range and step from the accessors. Add a `slot` argument (`"hv"`, `"mv"`, `"lv"`, `"2"`); default to the only populated slot, raise if ambiguous. Derive the control direction from the sign of the ratio change per step instead of `tap_side` and the sign of `tap_step_percent` |
| `DiscreteTapControl` | `control/controller/trafo/discrete_tap_control.py:63`, `131` | `vm_delta_pu` from `step_size_at` |
| `ContinuousTapControl` | `control/controller/trafo/continuous_tap_control.py:92-95` | Step size from `step_size_at`; clip to the characteristic's range |
| `TapDependentImpedance` | `control/controller/trafo/tap_dependent_impedance.py` | Review. Its purpose is covered by the table |
| Stored controllers in old nets | `convert_format._convert_trafo_controller_parameter_names` | Extend for the new attribute names |
| Short circuit, power station units | `shortcircuit/ppc_conversion.py:237-248` | The fallback for `pt_percent` becomes `voltage_ratio` at `tap_max` minus 1 |
| Contingency, lightsim2grid workaround | `contingency/contingency.py:250-293`, `400-414` | Add the looked-up angle to `shift_degree` instead of testing for `Ideal` |
| Subnet selection, merge, reindex | `toolbox/grid_modification.py:155-158`, `280-282`; `toolbox/data_modification.py:357-410` | Use the three reference columns and both tables |
| Grid equivalents | `grid_equivalents/rei_generation.py:193-198`, `grid_equivalents/get_equivalent.py:489` | Same |
| Diagnostic and splines | `control/util/diagnostic.py:62-180`, `control/util/auxiliary.py:221-300`, exports in `control/__init__.py` | Delete |
| Plotting hover text | `plotting/plotly/simple_plotly.py` | Show the slot positions |

Add the accessors next to `TapCharacteristic`, for example `tap_range(net, element, index, slot)`
and `tap_step_size(net, element, index, slot)`.

## A7. Three-winding specifics

- One slot per leg. Three changers on one 3W become possible; add a test with two.
- `at_star_point(nameplate)` produces pair factors, so such a characteristic depends on the
  nameplate and is not shared between different 3W types.
- Pair factors from changers on different legs multiply.
- Reference results: the two `trafo_3w_tap_results*.csv` files named in A0.

## A8. Converters and built-in networks

| Converter | Location | Change |
| --- | --- | --- |
| CIM/CGMES | `converter/cim/cim2pp/converter_classes/transformers/power_transformers_cim16.py:63-400`, `486-600`, `737-860` | `from_deviation` for table points, a builder per tap changer class. HV end to `tap_hv`, LV end to `tap_lv`. `PhaseTapChangerSymmetrical` to `symmetrical` (assumption A3) |
| PowerFactory | `converter/powerfactory/pp_import_functions.py:2838-2900`, `2960-3260`, `3360-3700` | `from_measurement_report`, including the measured side (`iMeasLoc`, line 3220). Second changer by its own side, to `tap2` when on the same winding |
| UCTE | `converter/ucte/ucte_converter.py:410-497` | Phase regulation to `ratio`, ASYM to `asymmetrical`, SYMM to `symmetrical` (assumption A3). Both regulations in one record: winding slot plus `tap2`. Optional: build a tabular characteristic from `##TT`, which the parser already reads (`ucte_parser.py:274`) |
| JAO | `converter/jao/from_jao.py` | Builders instead of tap columns |
| PYPOWER/MATPOWER | `converter/pypower/from_ppc.py` | A one-row characteristic per distinct ratio and shift, or nameplate voltages; decide which |
| Built-in networks | `networks/create_examples.py`, `cigre_networks.py`, `mv_oberrhein.py`, `tutorial_networks.py` | New create arguments |

Converter results change for symmetrical phase shifters under assumption A3. Update the
reference files (`test/converter/testfiles/test_ucte_res_trafo.csv`, the CIM test expectations
in `test/converter/test_from_cim.py`) in a commit of their own, so the change is visible.

## A9. Documentation

1. One common page: equations of section 3, slots, both table schemas, interpolation, the `tap2`
   conversion, measurement report conversion.
2. One page per two-winding case T1 to T11 and per three-winding case W1 to W6 of the planning
   document. Each page: the device, the equivalent circuit, the formulas with their source, the
   builder call with the rows it produces, the mapping to CGMES, UCTE-DEF and PowerFactory, and
   a numerical example that is also a test.
3. Replace the tap sections of `doc/elements/trafo.rst` (lines 160-285) and
   `doc/elements/trafo3w.rst` (lines 107-160), and the CSV tables they include.
4. Part B of this guide goes into the release notes and `CHANGELOG`.

## A10. Tests to port

| Test file | What changes |
| --- | --- |
| `test/create/test_trafo_create.py`, `test/create/test_trafo_characteristic_create.py` | New create arguments, `TapCharacteristic` |
| `test/network_schema/elements/test_pandera_trafo_elements.py`, `test_pandera_trafo3w_elements.py` | Slot columns and checks |
| `test/loadflow/test_runpp.py` | Table tests use the new columns |
| `test/control/test_tap_dependent_impedance.py` | Rewrite against the table, or delete with the controller |
| `test/api/test_convert_format.py`, `test/api/test_auxiliary.py` | Conversion of 3.x nets |
| `test/toolbox/test_grid_modification.py`, `test/grid_equivalents/test_get_equivalent.py` | Reference columns and both tables |
| `test/converter/test_from_cim.py` and the UCTE and PowerFactory converter tests | New expectations |
| `test/shortcircuit/test_trafo3w.py` | Slot columns |

---

# Part B: migrating nets and scripts

## B1. Loading existing nets

Nets saved with pandapower 3 are converted when they are loaded (`from_json`, `from_excel`,
`from_sqlite`, `from_pickle`) or by calling `pp.convert_format(net)`. Power flow results stay
the same, with the exceptions in B7. Save the net again to keep it in the new format.

A net built in memory by a pandapower 3 script is not converted. Update the script (B3).

## B2. Reading and writing tap positions

The position column depends on the winding the tap changer is on.

```python
# pandapower 3
net.trafo.at[0, "tap_pos"] = 3

# pandapower 4, tap changer on the HV winding
net.trafo.at[0, "tap_hv_pos"] = 3
# tap changer on the LV winding
net.trafo.at[0, "tap_lv_pos"] = 3
```

To find the slot of a converted transformer, look for the non-null reference:

```python
net.trafo.loc[0, ["tap_hv_characteristic", "tap_lv_characteristic", "tap2_characteristic"]]
```

For three-winding transformers the columns are `tap_hv_pos`, `tap_mv_pos` and `tap_lv_pos`.

## B3. Creating transformers

```python
# pandapower 3
pp.create_transformer_from_parameters(
    net, hv_bus, lv_bus, sn_mva=40, vn_hv_kv=110, vn_lv_kv=20, vk_percent=12, vkr_percent=0.3,
    pfe_kw=20, i0_percent=0.05,
    tap_side="hv", tap_neutral=0, tap_min=-9, tap_max=9, tap_step_percent=1.5,
    tap_pos=2, tap_changer_type="Ratio",
)

# pandapower 4
oltc = TapCharacteristic.ratio(tap_min=-9, tap_max=9, tap_neutral=0, step_percent=1.5)
pp.create_transformer_from_parameters(
    net, hv_bus, lv_bus, sn_mva=40, vn_hv_kv=110, vn_lv_kv=20, vk_percent=12, vkr_percent=0.3,
    pfe_kw=20, i0_percent=0.05,
    tap_hv=oltc, tap_hv_pos=2,
)
```

`create_transformer(net, hv_bus, lv_bus, std_type=...)` keeps working. It builds the
characteristic from the standard type; pass the position with `tap_hv_pos` or `tap_lv_pos`.

Which builder replaces which 3.x setting:

| pandapower 3 | pandapower 4 |
| --- | --- |
| `tap_changer_type="Ratio"`, no `tap_step_degree` | `TapCharacteristic.ratio(...)` |
| `tap_changer_type="Ratio"` with `tap_step_degree` | `TapCharacteristic.asymmetrical(..., theta_degree=tap_step_degree)` |
| `tap_changer_type="Symmetrical"` (step angle 90 degrees) | `TapCharacteristic.asymmetrical(..., theta_degree=90)` to keep the 3.x results; `TapCharacteristic.symmetrical(...)` for a real symmetrical phase shifter, see B7 |
| `tap_changer_type="Ideal"` with `tap_step_degree` | `TapCharacteristic.ideal(...)` |
| `tap_changer_type="Tabular"` with a table | `TapCharacteristic.tabular(...)` or `from_measurement_report(...)` |
| `tap2_*` on the other winding | the other winding's slot |
| `tap2_*` on the same winding | `tap2=...`, `tap2_pos=...` |
| `tap_at_star_point=True` (3W) | `.at_star_point(nameplate)` on the characteristic |

## B4. Tap range, neutral position and step size

`tap_min`, `tap_max`, `tap_neutral` and `tap_step_percent` are no longer columns. Use the
accessors, or read the characteristic:

```python
tap_min, tap_max, tap_neutral = tap_range(net, "trafo", 0, slot="hv")
```

## B5. Tap-dependent impedance and measurement reports

The table stores factors relative to the nameplate, not `vk_percent` per tap. To work with true
values, convert in both directions:

```python
# measured values per tap (a "measurement report") to a characteristic
char = TapCharacteristic.from_measurement_report(report_df, nameplate, measured_side="hv")
net.trafo.at[0, "tap_hv_characteristic"] = char.add_to(net)

# true values per tap for a transformer in the net
report_df = char.to_measurement_report(nameplate)
```

`from_measurement_report` raises if the neutral row of the report does not match the nameplate.

`create_trafo_characteristic`, `tap_dependency_table` and `id_characteristic_table` are gone.

## B6. Controllers and time series

- `DiscreteTapControl` and `ContinuousTapControl` take the slot they control. With one tap
  changer on the transformer the default is correct.
- `ConstControl` and time series that write `tap_pos` must write `tap_hv_pos`, `tap_lv_pos` or
  `tap2_pos`. The same applies to `OutputWriter` variables.
- `create_trafo_characteristic_object` and `net.trafo_characteristic_spline` are removed.
  Fractional tap positions are interpolated by the power flow itself.

## B7. Behaviour changes

| Change | What you may notice |
| --- | --- |
| A tap position outside the characteristic raises | Scripts that step beyond `tap_max` fail instead of extrapolating |
| A missing table row raises | pandapower 3 silently used ratio 1; such nets now need a complete table |
| Fractional positions interpolate linearly | Small differences against the spline-based results |
| Shared characteristics are resolved per transformer | Results change where transformers shared a table but had different positions. The old results were wrong |
| A characteristic can be shared by many transformers | Editing its rows changes all of them. Create a new characteristic to change one transformer |
| Fresh CGMES and UCTE imports of symmetrical phase shifters use ratio 1 and `2*atan(n*du/2)` | Different flows on those phase shifters than with pandapower 3 imports. Converted pandapower 3 nets keep their old values |
| `trafo_characteristic_table_diagnostic` is removed | Invalid tables are reported by schema validation |

## B8. Typical errors after upgrading

| Message mentions | Cause | Fix |
| --- | --- | --- |
| unknown argument `tap_side`, `tap_step_percent`, ... | pandapower 3 create call | B3 |
| column `tap_pos` not found | script or controller writes the old column | B2, B6 |
| position outside characteristic | `tap_*_pos` beyond the table | Correct the position or extend the characteristic |
| reference without position, or position without reference | one of the two slot columns is null | Set both or clear both |
| table not sorted | rows appended by hand | Use `add_to`, or sort by `id_characteristic`, `tap_pos` |

---

# Checklist

Implementation:

- [ ] A0 reference results frozen; failing tests written for the three 3.x defects
- [ ] A1 schemas for both tables
- [ ] A2 `TapCharacteristic` with builders, transforms, measurement report conversion
- [ ] A3 lookup function
- [ ] A4 `convert_format`, format version bumped
- [ ] A5 element schemas, create functions, std types, `build_branch.py` switched; benchmark recorded
- [ ] A6 controllers, short circuit, contingency, toolbox, grid equivalents; diagnostic and splines removed
- [ ] A7 three-winding slots and star-point conversion
- [ ] A8 converters and built-in networks
- [ ] A9 documentation pages per transformer type
- [ ] A10 tests ported

To confirm before starting (section 1): A1 hard break, A2 and A3 handling of `Symmetrical`,
A4 null reference, A5 derived range, A6 star point, A7 interpolation, A8 3W column names,
A9 zero-sequence scope, A10 short-circuit behaviour, and the new format version number.
