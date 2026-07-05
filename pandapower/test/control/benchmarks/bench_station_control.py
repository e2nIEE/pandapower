# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

"""
Benchmark harness for the station controller (BinarySearchControl / DroopControl).

Builds a synthetic HV/MV net with n independent stations (one MV feeder each, two sgens per
station) and measures, for a full run_control:

- wall time
- number of powerflow (runpp) calls
- controller overhead: wall time minus the time spent inside runpp

Usage (record a baseline, from the repo root):

    python -m pandapower.test.control.benchmarks.bench_station_control \
        --n 50 200 500 --mode Q_ctrl V_ctrl V_ctrl_Q_droop --label baseline --out baseline.json

Results are appended to the JSON file under the given label, together with the git revision.
The committed ``baseline.json`` pins the pre-refactor performance; refactor steps are gated
against it (see the project plan: overhead reduction, runpp calls <= baseline).
"""

import argparse
import json
import os
import subprocess
import time

from pandapower.auxiliary import ControllerNotConverged
from pandapower.control.controller.station_control import BinarySearchControl, DroopControl
from pandapower.control.run_control import run_control
from pandapower.create import (
    create_empty_network, create_bus, create_ext_grid, create_transformer, create_load,
    create_line, create_sgen)
from pandapower.run import runpp

MODES = ("Q_ctrl", "V_ctrl", "V_ctrl_Q_droop")
DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "baseline.json")


class TimingRunpp:
    """runpp wrapper for run_control(net, run=...) counting calls and accumulated runpp time."""

    def __init__(self):
        self.count = 0
        self.time_s = 0.0

    def __call__(self, net, **kwargs):
        kwargs.pop("run", None)
        self.count += 1
        t0 = time.perf_counter()
        runpp(net, **kwargs)
        self.time_s += time.perf_counter() - t0


def build_bench_net(n_stations, mode):
    """HV slack with ``n_stations`` identical MV feeders; one station controller per feeder.

    Each feeder: 110/20 kV trafo, MV load, MV->LV line, two sgens on the LV bus controlled
    with distribution [0.6, 0.4]. ``mode`` selects the controller configuration per station.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    net = create_empty_network()
    hv = create_bus(net, 110)
    create_ext_grid(net, hv)
    for k in range(n_stations):
        mv = create_bus(net, 20)
        lv = create_bus(net, 20)
        trafo = create_transformer(net, hv, mv, "63 MVA 110/20 kV")
        create_load(net, mv, 3, 0.1)
        sgen_a = create_sgen(net, lv, p_mw=2., sn_mva=10)
        sgen_b = create_sgen(net, lv, p_mw=1., sn_mva=10)
        line = create_line(net, mv, lv, length_km=0.1, std_type="NAYY 4x50 SE")
        outputs = dict(
            output_element="sgen", output_variable="q_mvar",
            output_element_index=[sgen_a, sgen_b], output_element_in_service=[True, True],
            output_values_distribution=[0.6, 0.4])
        if mode == "Q_ctrl":
            BinarySearchControl(
                net, name=f"q_ctrl_{k}", ctrl_in_service=True, input_element="res_line",
                damping_factor=0.9, input_variable=["q_to_mvar"], input_element_index=line,
                set_point=0.5, control_modus="Q_ctrl", tol=1e-6, **outputs)
        elif mode == "V_ctrl":
            BinarySearchControl(
                net, name=f"v_ctrl_{k}", ctrl_in_service=True, input_element="res_bus",
                input_variable="vm_pu", input_element_index=[mv], set_point=1.02,
                control_modus="V_ctrl", tol=1e-6, **outputs)
        else:  # V_ctrl_Q_droop
            bsc = BinarySearchControl(
                net, name=f"v_droop_bsc_{k}", ctrl_in_service=True, input_element="res_trafo",
                input_variable="q_hv_mvar", input_element_index=[trafo], set_point=1.02,
                control_modus="V_ctrl_Q_droop", bus_idx=mv, tol=1e-6, **outputs)
            DroopControl(net, name=f"v_droop_{k}", q_droop_mvar=40, bus_idx=mv, vm_set_pu=1.02,
                         controller_idx=bsc.index, control_modus="V_ctrl_Q_droop", tol=1e-6)
    return net


def station_dicts(mode, n_stations):
    """Station dicts for BinarySearchControl.for_stations matching build_bench_net(mode)."""
    stations = []
    for k in range(n_stations):
        outputs = dict(output_element_index=[2 * k, 2 * k + 1],
                       output_values_distribution=[0.6, 0.4])
        if mode == "Q_ctrl":
            stations.append(dict(control_modus="Q_ctrl", set_point=0.5,
                                 input_element="res_line", input_variable="q_to_mvar",
                                 input_element_index=k, **outputs))
        elif mode == "V_ctrl":
            stations.append(dict(control_modus="V_ctrl", set_point=1.02,
                                 input_element="res_bus", input_variable="vm_pu",
                                 input_element_index=1 + 2 * k, **outputs))
        else:  # V_ctrl_Q_droop
            stations.append(dict(control_modus="V_ctrl_Q_droop", set_point=1.02,
                                 input_element="res_trafo", input_variable="q_hv_mvar",
                                 input_element_index=k,
                                 droop=dict(q_droop_mvar=40, bus_idx=1 + 2 * k), **outputs))
    return stations


def build_bench_net_multistation(n_stations, mode, update_method="secant"):
    """Same net as build_bench_net, but all stations in one for_stations instance."""
    net = build_bench_net(n_stations, mode)
    net.controller.drop(net.controller.index, inplace=True)
    BinarySearchControl.for_stations(net, station_dicts(mode, n_stations),
                                     name="bench_multi", tol=1e-6,
                                     update_method=update_method)
    return net


def bench(n_stations, mode, max_iter=30, multistation=False, update_method="secant"):
    """Run one benchmark case, return a result dict."""
    if multistation:
        net = build_bench_net_multistation(n_stations, mode, update_method=update_method)
    else:
        net = build_bench_net(n_stations, mode)
    timer = TimingRunpp()
    t0 = time.perf_counter()
    converged = True
    try:
        run_control(net, run=timer, max_iter=max_iter)
    except ControllerNotConverged:
        converged = False
    wall_s = time.perf_counter() - t0
    return {
        "n_stations": n_stations,
        "mode": mode,
        "multistation": multistation,
        "update_method": update_method if multistation else None,
        "converged": converged,
        "wall_s": round(wall_s, 4),
        "runpp_calls": timer.count,
        "runpp_time_s": round(timer.time_s, 4),
        "overhead_s": round(wall_s - timer.time_s, 4),
    }


def _git_rev():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(__file__)).decode().strip()
    except Exception:  # noqa: BLE001 - benchmark metadata only
        return "unknown"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=[50, 200, 500])
    parser.add_argument("--mode", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--label", default="baseline",
                        help="section name in the output JSON, e.g. baseline/post_vectorization")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--multistation", action="store_true",
                        help="all stations in one for_stations controller instance")
    parser.add_argument("--update-method", default="secant", choices=("secant", "jacobian"),
                        help="update method for --multistation runs")
    args = parser.parse_args(argv)

    results = []
    for mode in args.mode:
        for n in args.n:
            result = bench(n, mode, max_iter=args.max_iter, multistation=args.multistation,
                           update_method=args.update_method)
            results.append(result)
            print(json.dumps(result))

    data = {}
    if os.path.isfile(args.out):
        with open(args.out) as f:
            data = json.load(f)
    data[args.label] = {"git_rev": _git_rev(), "results": results}
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"written section {args.label!r} to {args.out}")


if __name__ == "__main__":
    main()
