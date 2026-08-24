"""Compare Benders decomposition against the direct multi-commodity MILP solve
across a sample of the multi-commodity dissertation scenarios."""

import argparse
import glob
import os
import time

import gurobipy as gp

from benders_multicommodity import load_scenario, solve_benders
from multicommodity_milp import build_model

SCENARIO_DIR = "synthetic_data_generator/multicommodity_scenarios"


def default_scenario_paths():
    families = sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(SCENARIO_DIR, "*")) if os.path.isdir(p)
    )
    return [os.path.join(SCENARIO_DIR, fam, "seed_001", f"{fam}_seed_001.json") for fam in families]


def all_scenario_paths():
    return sorted(glob.glob(os.path.join(SCENARIO_DIR, "*", "seed_*", "*.json")))


def run_direct(path):
    data = load_scenario(path)
    model, _ = build_model(data)
    model.Params.OutputFlag = 0
    t0 = time.perf_counter()
    model.optimize()
    wall = time.perf_counter() - t0
    obj = model.ObjVal if model.SolCount > 0 else None
    return obj, wall


def run_benders(path, max_iters, tol):
    data = load_scenario(path)
    t0 = time.perf_counter()
    result = solve_benders(data, max_iters=max_iters, tol=tol)
    wall = time.perf_counter() - t0
    return result, wall


def fmt(value, spec):
    return format(value, spec) if value is not None else "n/a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="run the full 120-scenario battery instead of the seed_001 sample")
    parser.add_argument("--limit", type=int, default=None, help="cap the number of scenarios run (smoke testing)")
    parser.add_argument("--max-iters", type=int, default=150)
    parser.add_argument("--tol", type=float, default=1e-3, help="relative optimality gap for Benders")
    args = parser.parse_args()

    paths = all_scenario_paths() if args.all else default_scenario_paths()
    if args.limit:
        paths = paths[: args.limit]

    header = (
        f"{'scenario':<34} {'direct_obj':>14} {'benders_obj':>14} {'gap_%':>8} "
        f"{'status':<16} {'iters':>6} {'feas':>5} {'opt':>4} {'direct_s':>9} {'benders_s':>10}"
    )
    print(header)
    print("-" * len(header))
    for path in paths:
        name = os.path.relpath(path, SCENARIO_DIR)

        try:
            direct_obj, direct_wall = run_direct(path)
        except gp.GurobiError as exc:
            print(f"{name:<34} ERROR (direct): {exc}")
            continue

        try:
            result, benders_wall = run_benders(path, args.max_iters, args.tol)
        except gp.GurobiError as exc:
            print(f"{name:<34} ERROR (benders): {exc}")
            continue

        benders_obj = result["best_objective"]
        gap_pct = (
            100 * (direct_obj - benders_obj) / direct_obj
            if direct_obj not in (None, 0) and benders_obj is not None
            else None
        )
        print(
            f"{name:<34} {fmt(direct_obj, ',.1f'):>14} {fmt(benders_obj, ',.1f'):>14} "
            f"{fmt(gap_pct, '.3f'):>7}% {result['status']:<16} {result['iterations']:>6} "
            f"{result['feasibility_cuts']:>5} {result['optimality_cuts']:>4} "
            f"{direct_wall:>9.2f} {benders_wall:>10.2f}"
        )


if __name__ == "__main__":
    main()
