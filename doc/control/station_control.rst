##################
Station Controller
##################

The station controller (:class:`~pandapower.control.controller.station_control.BinarySearchControl`,
short *BSC*) adjusts the reactive power of a group of generation units ("a station") until a
measured quantity reaches a set point. It supports reactive power control (``Q_ctrl``), voltage
control (``V_ctrl``), power factor control (``PF_ctrl_ind`` / ``PF_ctrl_cap``), ``tan_phi_ctrl``
and droop variants of the Q and V control modi, mirroring the station controllers known from
PowerFactory.

This page describes the reworked implementation: how to use the controller correctly (single
station and the new multi-station mode), the opt-in Jacobian sensitivity update, and the
convergence and runtime improvements over the previous implementation.

.. contents::
    :local:
    :depth: 2


Controlling a single station
============================

The constructor API is unchanged and fully backward compatible: one
``BinarySearchControl`` instance controls one station. Saved nets (JSON) created with earlier
pandapower versions and nets imported from PowerFactory keep working without modification,
including chained :class:`~pandapower.control.controller.station_control.DroopControl` /
:class:`~pandapower.control.controller.station_control.VDroopControl_local` controllers.

::

    from pandapower.control.controller.station_control import BinarySearchControl
    from pandapower.run import runpp

    BinarySearchControl(
        net, name="station_1", ctrl_in_service=True,
        output_element="sgen", output_variable="q_mvar",
        output_element_index=[0, 1], output_element_in_service=[True, True],
        output_values_distribution=[0.6, 0.4],       # Q share of each output element
        input_element="res_line", input_variable=["q_to_mvar"], input_element_index=0,
        set_point=1.0, control_modus="Q_ctrl", tol=1e-6)

    runpp(net, run_control=True)

Notes on correct usage:

- ``output_values_distribution`` is normalized internally; it defines how the total station
  output is shared among the output elements.
- The measurement (``input_element`` / ``input_variable`` / ``input_element_index``) must
  actually respond to the controlled elements. If it does not (for example a line on a
  different feeder), the controller now keeps its outputs bounded and logs a stagnation
  warning instead of blowing the values up, but it can of course not converge.
- ``input_inverted=True`` flips the sign of the measurement (needed when the measured branch
  orientation is opposite to the control direction, e.g. for PowerFactory imports).
- Reactive power limits (``min_q_mvar`` / ``max_q_mvar`` columns, or Q capability
  characteristics) are respected when the powerflow is run with ``enforce_q_lims=True``.
- ``damping_factor`` (e.g. 0.9) softens the first probing step and the fallback steps; it is
  deliberately *not* applied to regular secant steps, so well-behaved controllers converge at
  full speed.


Controlling many stations: ``for_stations``
===========================================

For grids with tens to hundreds of stations, creating one controller instance per station
wastes most of the run time in per-controller bookkeeping. The recommended API is one
controller instance that manages all stations of one output table:

::

    from pandapower.control.controller.station_control import BinarySearchControl

    stations = [
        # a voltage-controlled station with two sgens
        dict(control_modus="V_ctrl", set_point=1.02,
             input_element="res_bus", input_variable="vm_pu", input_element_index=2,
             output_element_index=[0, 1], output_values_distribution=[0.6, 0.4]),
        # a Q-controlled station measuring a line flow
        dict(control_modus="Q_ctrl", set_point=0.5,
             input_element="res_line", input_variable="q_to_mvar", input_element_index=4,
             output_element_index=[2, 3], output_values_distribution=[0.5, 0.5]),
        # a voltage-controlled station with Q droop: in the converged state
        # vm(bus 7) == set_point + q_hv_mvar(trafo 1) / q_droop_mvar
        dict(control_modus="V_ctrl_Q_droop", set_point=1.02,
             input_element="res_trafo", input_variable="q_hv_mvar", input_element_index=1,
             droop=dict(q_droop_mvar=40, bus_idx=7),
             output_element_index=[4, 5], output_values_distribution=[1, 1]),
    ]
    BinarySearchControl.for_stations(net, stations, output_element="sgen",
                                     output_variable="q_mvar", tol=1e-6)

Each station dict carries its own configuration:

.. list-table::
    :widths: 25 75
    :header-rows: 1

    * - key
      - meaning
    * - ``control_modus``
      - ``"Q_ctrl"``, ``"V_ctrl"``, ``"PF_ctrl_ind"``, ``"PF_ctrl_cap"``, ``"tan_phi_ctrl"``,
        ``"Q_ctrl_V_droop"``, ``"V_ctrl_Q_droop"`` or ``"V_ctrl_Q_droop_local"``
    * - ``set_point``
      - target value; for droop modi the *base* set point of the droop characteristic
    * - ``input_element`` / ``input_variable`` / ``input_element_index``
      - the measurement, e.g. ``"res_line"`` / ``"q_to_mvar"`` / ``4``; plain ``V_ctrl``
        measures ``"res_bus"`` / ``"vm_pu"`` at the controlled bus. Lists are allowed for
        multiple measurement elements per station
    * - ``input_inverted``
      - bool or list of bool, flips the measurement sign
    * - ``output_element_index`` / ``output_values_distribution``
      - controlled elements in the shared output table and their share of the station total
    * - ``tol``
      - optional per-station tolerance (defaults to the instance tolerance)
    * - ``droop``
      - required for droop modi, see below
    * - ``name``
      - optional, used in log messages

Droop is part of the station, not a second controller
------------------------------------------------------

In the legacy API, droop behaviour required chaining a separate ``DroopControl`` that
rewrites the BSC set point between iterations -- two coupled fixed-point loops that iterate
against each other. In multi-station mode, the droop characteristic is evaluated inside the
station residual, so there is only one loop. The ``droop`` dict supports:

.. list-table::
    :widths: 30 70
    :header-rows: 1

    * - key
      - meaning
    * - ``q_droop_mvar``
      - droop constant in Mvar/pu (all droop modi)
    * - ``bus_idx``
      - the voltage-measured bus (all droop modi)
    * - ``vm_set_lb`` / ``vm_set_ub``
      - deadband borders for ``Q_ctrl_V_droop``: inside the band the station holds
        ``set_point``, outside it the Q set point follows the droop line
    * - ``vm_set_pu``
      - voltage reference for ``Q_ctrl_V_droop`` *without* deadband:
        ``q_set = set_point + (vm_set_pu - vm) * q_droop_mvar``
    * - ``q_set_mvar``
      - local Q reference for ``V_ctrl_Q_droop_local``:
        ``vm = set_point - (q_meas - q_set_mvar) / q_droop_mvar``

Converged droop states:

- ``V_ctrl_Q_droop``: ``vm(bus_idx) = set_point + q_meas / q_droop_mvar``
- ``Q_ctrl_V_droop`` (deadband): ``q_meas = set_point`` inside the band,
  ``q_meas = set_point ± (band border - vm) * q_droop_mvar`` outside
- ``V_ctrl_Q_droop_local``: ``vm(bus_idx) = set_point - (q_meas - q_set_mvar) / q_droop_mvar``

Constraints and behaviour
-------------------------

- One instance writes one output table and column (``output_element`` /
  ``output_variable``). Stations writing to a different table (e.g. shunt steps) need a
  second instance. Output elements should not be shared between stations.
- Stations whose input or output elements are all out of service are skipped with a warning;
  the remaining stations keep working.
- Stations that reach their reactive power limits (with ``enforce_q_lims=True``) freeze the
  limited outputs at the limit and redistribute the remainder; if all outputs are limited,
  the station is reported converged-at-limit.
- Already-converged stations are frozen while the others keep iterating.
- ``for_stations`` controllers serialize with :func:`pandapower.to_json` and restore with
  :func:`pandapower.from_json` like any other controller.


Jacobian sensitivities: ``update_method="jacobian"``
====================================================

By default (``update_method="secant"``) every station iterates independently on its own
residual. For voltage control this has a fundamental limitation: two ``V_ctrl`` stations that
are electrically close (for example behind the same transformer) see each other's actions as
measurement noise, and the interleaved iteration may oscillate without converging -- the
previous implementation could not solve such configurations either.

With ``update_method="jacobian"``, plain ``V_ctrl`` stations take **coupled Newton steps**
instead: after each powerflow, the sensitivities ``dVm/dQ`` of all measured buses to all
controlled injections are extracted from the final Newton-Raphson Jacobian (which pandapower
keeps in ``net._ppc["internal"]["J"]`` -- no extra powerflow is needed), the cross-station
sensitivity matrix is built and one linear solve updates all stations simultaneously:

::

    BinarySearchControl.for_stations(net, stations, tol=1e-6,
                                     update_method="jacobian")

Effects:

- fewer powerflows for voltage control (3 instead of 5 in the benchmark below, i.e. one
  Newton step typically suffices), which matters most when each powerflow is expensive;
- convergence of electrically coupled V_ctrl stations that independent iterations cannot
  solve (see ``test_jacobian_coupled_same_feeder``).

Scope and safety:

- The Newton step is applied to plain ``V_ctrl`` stations. Q/PF/tan(phi) modi gain nothing
  from it (their residual already follows the station output nearly 1:1) and droop modi
  currently keep the secant update as well.
- The controller falls back to the safeguarded secant automatically whenever no Jacobian is
  available (non-NR solver, stripped internals), a controlled or measured bus is not a PQ bus
  in the powerflow, the linear system is unusable, or a Newton step increased the residual.
  A run can therefore never get *worse* by enabling the flag.
- The underlying sensitivity function is available as a standalone utility::

      from pandapower.control.util.sensitivity import calc_dvm_dq
      sensitivity = calc_dvm_dq(net, q_bus_idx=[4, 5], vm_bus_idx=[2, 3])  # pu per Mvar

  Its sign, scaling and index mapping are pinned against finite differences in
  ``test_dvm_dq_matches_finite_difference``.


Convergence improvements
========================

The update rule was rebuilt around a safeguarded secant method (all of this applies to the
legacy single-station API as well):

- **Residual-scaled first step** for Q/PF/tan(phi) modi: the first perturbation is sized by
  the residual (nearly a Newton step) instead of a fixed 1 kvar probe.
- **Bracketing**: once two iterates with opposite residual signs are known, an
  Illinois-damped regula falsi keeps every further iterate inside the bracket -- no more
  overshooting oscillations. Brackets that were collected while *other* controllers still
  moved the operating point are detected via stagnation and discarded.
- **No more blow-ups on flat responses**: if the measurement does not react to the output,
  the legacy code divided by a 1e-6 dummy slope and multiplied the outputs into the millions
  of Mvar. The new update takes bounded fallback steps and logs a stagnation warning.
- **Bug fixes**: the power factor set point is no longer permanently overwritten by the
  near-zero clipping; internal output state is re-synchronized with the net before every
  step (robust against other controllers writing the same elements); stations with partially
  out-of-service output elements no longer crash.

The characterization suite (``pandapower/test/control/test_stactrl_characterization.py``)
pins the converged results and powerflow counts of 15 scenarios against the pre-refactor
implementation: all scenarios converge to the same results with the same or fewer powerflows
(e.g. ``tan_phi_ctrl`` 4 → 3, ``Q_ctrl_V_droop`` with deadband 5 → 4).

A known limitation is documented as an expected failure: several *independently iterating*
V_ctrl stations behind the same transformer do not converge with the secant method (the
pre-refactor implementation fails there too). Use one ``for_stations`` instance with
``update_method="jacobian"`` for such configurations.


Timing results
==============

Measured with the committed benchmark harness on the synthetic benchmark grid (one HV slack,
n MV feeders, one station with two sgens per feeder; recorded in
``pandapower/test/control/benchmarks/baseline.json``). "Overhead" is the controller
bookkeeping time of a full ``run_control``, i.e. wall time minus the time spent inside the
powerflows -- the metric that scales with the number of stations.

Controller overhead, n = 500 stations:

.. list-table::
    :widths: 30 20 20 20 12
    :header-rows: 1

    * - mode
      - before (s)
      - per-station instances, after (s)
      - one ``for_stations`` instance (s)
      - speed-up
    * - ``Q_ctrl``
      - 0.69
      - 0.43
      - 0.06
      - 11x
    * - ``V_ctrl``
      - 1.03
      - 0.67
      - 0.08
      - 13x
    * - ``V_ctrl_Q_droop``
      - 1.75
      - 0.89
      - 0.08
      - 22x

Controller overhead, n = 200 stations:

.. list-table::
    :widths: 30 20 20 20 12
    :header-rows: 1

    * - mode
      - before (s)
      - per-station instances, after (s)
      - one ``for_stations`` instance (s)
      - speed-up
    * - ``Q_ctrl``
      - 0.21
      - 0.16
      - 0.02
      - 9x
    * - ``V_ctrl``
      - 0.52
      - 0.27
      - 0.04
      - 13x
    * - ``V_ctrl_Q_droop``
      - 0.58
      - 0.31
      - 0.04
      - 15x

The gain has three sources: an O(n²) scan of all controllers in every iteration was removed,
per-iteration pandas element access was replaced by positional numpy lookups precomputed in
``initialize_control``, and the multi-station mode reduces n controller objects (plus n
chained droop controllers) to a handful of table reads and one table write per iteration.

Number of powerflows (n = 200, multi-station):

.. list-table::
    :widths: 34 22 22 22
    :header-rows: 1

    * - mode
      - before
      - secant (after)
      - ``jacobian``
    * - ``Q_ctrl``
      - 3
      - 2
      - 2 (secant path)
    * - ``V_ctrl``
      - 5
      - 5
      - 3
    * - ``V_ctrl_Q_droop``
      - 5
      - 4
      - 4 (secant path)

On the small benchmark feeders a powerflow is cheap, so the Jacobian mode mainly saves
iterations; on large real grids, where each ``runpp`` dominates the run time, the reduction
from 5 to 3 powerflows translates directly into wall time.

Reproducing the numbers::

    python -m pandapower.test.control.benchmarks.bench_station_control \
        --n 50 200 500 --mode Q_ctrl V_ctrl V_ctrl_Q_droop --label my_run
    python -m pandapower.test.control.benchmarks.bench_station_control \
        --n 200 --mode V_ctrl --multistation --update-method jacobian --label my_run_jac


Backward compatibility
======================

- The ``BinarySearchControl`` constructor signature, the ``DroopControl`` /
  ``VDroopControl_local`` chaining mechanism and all ``ControlModusEnum`` values are
  unchanged. Legacy nets keep their behaviour; the single-station code path reproduces the
  pre-refactor results bit-exactly (verified by the characterization suite).
- Nets saved with earlier pandapower versions load and solve unchanged; this is guarded by a
  frozen pre-refactor fixture
  (``pandapower/test/control/testfiles/stactrl_prerefactor_v1.json``).
- Mixed operation is supported: legacy controllers and ``for_stations`` instances can run in
  the same net (``test_multistation_prerefactor_interop``).

API reference
=============

See the class documentation of
:class:`~pandapower.control.controller.station_control.BinarySearchControl` (including
:meth:`~pandapower.control.controller.station_control.BinarySearchControl.for_stations`),
:class:`~pandapower.control.controller.station_control.DroopControl` and
:class:`~pandapower.control.controller.station_control.VDroopControl_local` in
:doc:`controller`.
