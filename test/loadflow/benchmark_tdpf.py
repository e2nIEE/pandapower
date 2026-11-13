# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

"""Benchmark for the temperature-dependent power flow (``runpp(net, tdpf=True)``).

This script is not collected by pytest. Run it manually from the repository root::

    python test/loadflow/benchmark_tdpf.py
    python test/loadflow/benchmark_tdpf.py --cases case30 mv_oberrhein --repeat 5
    python test/loadflow/benchmark_tdpf.py --profile case1354pegase

It reports wall time (best of ``--repeat`` runs), the number of Newton iterations and the peak
traced memory for plain ``runpp`` and for TDPF with all combinations of numba,
``tdpf_update_r_theta`` and ``distributed_slack``.

The thermal time constant ``tdpf_delay_s`` is set to ``None`` by default, matching ``runpp``'s
default (steady-state conductor temperature).

Large transmission cases may not converge thermally when TDPF is enabled for every line (the
generic thermal parameters and the ratings of the case can lead to a thermal runaway), so the
fraction of lines that take part in TDPF can be reduced with ``--tdpf-fraction``.
"""

from __future__ import annotations

import argparse
import copy
import cProfile
import io
import pstats
import sys
import tracemalloc
from pathlib import Path
from time import perf_counter

# allow importing the test helpers when the script is run directly
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pandapower.pf.create_jacobian_tdpf import calc_r_theta_from_t_rise
from pandapower.run import runpp


def _has_numba() -> bool:
    try:
        import numba  # noqa: F401
    except ImportError:
        return False
    return True


def _prepare_tdpf_net(net, tdpf_update_r_theta: bool, tdpf_fraction: float = 1.0):
    """Fill the line columns that TDPF needs and enable TDPF for a fraction of the lines."""
    net = copy.deepcopy(net)
    line = net.line
    if len(line) == 0:
        return net

    line["tdpf"] = line.r_ohm_per_km.fillna(0.0) != 0.0
    if tdpf_fraction < 1.0:
        step = max(round(1.0 / tdpf_fraction), 1)
        selected = set(line.index[::step])
        line.loc[~line.index.isin(selected), "tdpf"] = False
    line["temperature_degree_celsius"] = 20.0
    line["reference_temperature_degree_celsius"] = 20.0
    line["air_temperature_degree_celsius"] = 35.0
    line["alpha"] = 4.03e-3
    line["mc_joule_per_m_k"] = 1490.0
    line["wind_speed_m_per_s"] = 0.6
    line["wind_angle_degree"] = 45.0
    line["solar_radiation_w_per_sq_m"] = 900.0
    line["solar_absorptivity"] = 0.5
    line["emissivity"] = 0.5

    if tdpf_update_r_theta:
        line["conductor_outer_diameter_m"] = 30.6e-3
    else:
        line["r_theta_kelvin_per_mw"] = calc_r_theta_from_t_rise(net, 25.0)

    return net


def load_case(name: str):
    if name == "case30":
        from test.loadflow.test_tdpf import prepare_case_30

        return prepare_case_30()
    if name == "case1354pegase":
        from pandapower.networks import case1354pegase

        return case1354pegase()
    if name == "case9241pegase":
        from pandapower.networks import case9241pegase

        return case9241pegase()
    if name == "mv_oberrhein":
        from pandapower.networks import mv_oberrhein

        return mv_oberrhein()
    raise ValueError(f"unknown benchmark case {name!r}")


def _run_once(net, tdpf: bool, **options):
    """Run one power flow on a deep copy and return (wall time, iterations, peak memory in MB)."""
    net = copy.deepcopy(net)
    tracemalloc.start()
    start = perf_counter()
    runpp(net, tdpf=tdpf, **options)
    elapsed = perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    iterations = net._ppc.get("iterations", None)
    return elapsed, iterations, peak / 1e6


def benchmark_case(name: str, repeat: int, numba_values, tdpf_fraction: float = 1.0):
    net = load_case(name)

    print(f"\n=== {name} ===")
    print(f"buses: {len(net.bus)}, lines: {len(net.line)}, trafos: {len(net.trafo)}")

    _run_once(net, tdpf=False)  # warm-up (numba compilation, imports)
    elapsed, iterations, peak = _run_once(net, tdpf=False)
    for _ in range(repeat - 1):
        t, it, mem = _run_once(net, tdpf=False)
        elapsed, iterations, peak = min(elapsed, t), it, min(peak, mem)
    print(f"  {'plain runpp':<60} {elapsed * 1e3:9.2f} ms  it={iterations}  peak={peak:7.1f} MB")

    for numba in numba_values:
        for update_r_theta in (True, False):
            tdpf_net = _prepare_tdpf_net(net, tdpf_update_r_theta=update_r_theta, tdpf_fraction=tdpf_fraction)
            for dist_slack in (False, True):
                label = f"tdpf numba={numba} update_r_theta={update_r_theta} dist_slack={dist_slack}"
                try:
                    _run_once(
                        tdpf_net,
                        tdpf=True,
                        numba=numba,
                        tdpf_update_r_theta=update_r_theta,
                        distributed_slack=dist_slack,
                    )  # warm-up
                    elapsed, iterations, peak = _run_once(
                        tdpf_net,
                        tdpf=True,
                        numba=numba,
                        tdpf_update_r_theta=update_r_theta,
                        distributed_slack=dist_slack,
                    )
                    for _ in range(repeat - 1):
                        t, it, mem = _run_once(
                            tdpf_net,
                            tdpf=True,
                            numba=numba,
                            tdpf_update_r_theta=update_r_theta,
                            distributed_slack=dist_slack,
                        )
                        elapsed, iterations, peak = min(elapsed, t), it, min(peak, mem)
                except Exception as exc:  # noqa: BLE001 - benchmarks report, not abort, on failures
                    print(f"  {label:<60} failed: {type(exc).__name__}: {exc}")
                    continue
                print(f"  {label:<60} {elapsed * 1e3:9.2f} ms  it={iterations}  peak={peak:7.1f} MB")


def profile_case(name: str, tdpf_fraction: float = 1.0):
    net = _prepare_tdpf_net(load_case(name), tdpf_update_r_theta=True, tdpf_fraction=tdpf_fraction)
    runpp(copy.deepcopy(net), tdpf=True, numba=False)  # warm-up without numba compilation
    profiler = cProfile.Profile()
    profiler.enable()
    runpp(net, tdpf=True, numba=False)
    profiler.disable()

    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(30)
    print(stream.getvalue())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["case30", "mv_oberrhein", "case1354pegase", "case9241pegase"],
        choices=["case30", "mv_oberrhein", "case1354pegase", "case9241pegase"],
        help="grids to benchmark",
    )
    parser.add_argument("--repeat", type=int, default=3, help="number of runs per configuration (best is reported)")
    parser.add_argument("--no-numba", action="store_true", help="benchmark the numpy path only")
    parser.add_argument(
        "--tdpf-fraction",
        type=float,
        default=1.0,
        help="fraction of the lines that take part in TDPF (use < 1 if a large case does not converge)",
    )
    parser.add_argument("--profile", metavar="CASE", help="profile one TDPF run with cProfile instead of benchmarking")
    args = parser.parse_args(argv)

    if args.profile:
        profile_case(args.profile, tdpf_fraction=args.tdpf_fraction)
        return 0

    numba_values = [False] if args.no_numba else [False, True]
    if not _has_numba():
        print("numba is not installed, benchmarking only the numpy path")
        numba_values = [False]

    for name in args.cases:
        benchmark_case(
            name,
            repeat=max(args.repeat, 1),
            numba_values=numba_values,
            tdpf_fraction=args.tdpf_fraction,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
