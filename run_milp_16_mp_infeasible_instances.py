#!/usr/bin/env python3
"""
Run the current multi-commodity monolithic MILP on only the 16 instances
reported as infeasible in the Benders master problem with no-overlap.

Purpose
-------
This is a diagnostic run.

The current direct MILP does NOT enforce the same no-overlap condition as the
Benders master problem. Therefore this script does not modify the MILP.

It checks whether the 16 instances that are infeasible in the Benders MP
are feasible under the CURRENT monolithic MILP formulation.

Parallelisation
---------------
Several independent MILP instances are solved simultaneously using
ProcessPoolExecutor.

Default:
    4 parallel workers
    1 Gurobi thread per worker
    1800 seconds per instance

Results are written to CSV every time an instance finishes, so partial
results are preserved if the run is interrupted.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import gurobipy as gp


# =============================================================================
# Project paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multicommodity_milp import build_model, load_scenario


# =============================================================================
# Default configuration
# =============================================================================

DEFAULT_SCENARIO_DIR = (
    PROJECT_ROOT
    / "synthetic_data_generator"
    / "multicommodity_scenarios"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "milp_16_mp_infeasible"
)

DEFAULT_TIME_LIMIT = 1800.0

# Gurobi threads PER MILP instance
DEFAULT_THREADS = 1

# Number of MILP instances solved simultaneously
DEFAULT_WORKERS = 16

DEFAULT_SEED = 1


# =============================================================================
# 16 instances identified as infeasible in the Benders MP with no-overlap
# =============================================================================

TARGET_INSTANCES = (
    "F1_system_scale_H_seed_006",
    "F1_system_scale_H_seed_009",

    "F2_mode_tradeoff_H_seed_008",
    "F2_mode_tradeoff_M_seed_008",

    "F3_yard_flow_M_seed_002",
    "F3_yard_flow_M_seed_010",

    "F4_operational_conflict_H_seed_002",
    "F4_operational_conflict_H_seed_003",
    "F4_operational_conflict_H_seed_005",
    "F4_operational_conflict_H_seed_006",
    "F4_operational_conflict_H_seed_009",

    "F4_operational_conflict_L_seed_005",
    "F4_operational_conflict_L_seed_009",

    "F4_operational_conflict_M_seed_001",
    "F4_operational_conflict_M_seed_005",
    "F4_operational_conflict_M_seed_007",
)


# =============================================================================
# Gurobi status names
# =============================================================================

GUROBI_STATUS = {
    gp.GRB.LOADED: "LOADED",
    gp.GRB.OPTIMAL: "OPTIMAL",
    gp.GRB.INFEASIBLE: "INFEASIBLE",
    gp.GRB.INF_OR_UNBD: "INF_OR_UNBD",
    gp.GRB.UNBOUNDED: "UNBOUNDED",
    gp.GRB.CUTOFF: "CUTOFF",
    gp.GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
    gp.GRB.NODE_LIMIT: "NODE_LIMIT",
    gp.GRB.TIME_LIMIT: "TIME_LIMIT",
    gp.GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
    gp.GRB.INTERRUPTED: "INTERRUPTED",
    gp.GRB.NUMERIC: "NUMERIC",
    gp.GRB.SUBOPTIMAL: "SUBOPTIMAL",
    gp.GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
}


# =============================================================================
# CSV fields
# =============================================================================

RESULT_FIELDS = [
    "instance",
    "scenario_file",
    "mp_no_overlap_status",
    "milp_status",
    "sol_count",
    "objective",
    "best_bound",
    "mip_gap",
    "runtime_seconds",
    "node_count",
    "num_vars",
    "num_binary_vars",
    "num_integer_vars",
    "num_constraints",
    "time_limit_seconds",
    "threads",
    "seed",
    "error",
]


# =============================================================================
# Discover target JSON files
# =============================================================================

def discover_target_instances(
    scenario_dir: Path,
) -> dict[str, Path]:

    if not scenario_dir.exists():
        raise FileNotFoundError(
            f"Scenario directory does not exist:\n"
            f"{scenario_dir}"
        )

    targets = set(TARGET_INSTANCES)

    found: dict[str, Path] = {}

    for path in scenario_dir.rglob("*.json"):

        instance_name = path.stem

        if instance_name not in targets:
            continue

        if instance_name in found:
            raise RuntimeError(
                f"Duplicate instance found for {instance_name}:\n"
                f"  {found[instance_name]}\n"
                f"  {path}"
            )

        found[instance_name] = path

    return found


# =============================================================================
# Numerical helper
# =============================================================================

def safe_float(
    value: Any,
) -> float | None:

    try:
        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return None

        return value

    except (TypeError, ValueError):
        return None


# =============================================================================
# Solve ONE instance
#
# This function runs inside a worker process.
# =============================================================================

def solve_instance(
    instance_name: str,
    scenario_path: Path,
    *,
    time_limit: float,
    threads: int,
    seed: int,
    output_flag: int,
    output_dir: Path,
) -> dict[str, Any]:

    result: dict[str, Any] = {
        "instance": instance_name,
        "scenario_file": str(scenario_path),
        "mp_no_overlap_status": "INFEASIBLE",
        "milp_status": "",
        "sol_count": 0,
        "objective": None,
        "best_bound": None,
        "mip_gap": None,
        "runtime_seconds": None,
        "node_count": None,
        "num_vars": None,
        "num_binary_vars": None,
        "num_integer_vars": None,
        "num_constraints": None,
        "time_limit_seconds": time_limit,
        "threads": threads,
        "seed": seed,
        "error": "",
    }

    try:

        # ---------------------------------------------------------------------
        # Load scenario
        # ---------------------------------------------------------------------

        data = load_scenario(
            scenario_path
        )

        # ---------------------------------------------------------------------
        # Build CURRENT MILP unchanged
        # ---------------------------------------------------------------------

        model_result = build_model(
            data
        )

        # Support both possibilities:
        #
        #     model = build_model(data)
        #
        # and
        #
        #     model, variables = build_model(data)

        if isinstance(model_result, tuple):
            model = model_result[0]
        else:
            model = model_result

        # ---------------------------------------------------------------------
        # Solver settings
        # ---------------------------------------------------------------------

        model.Params.TimeLimit = time_limit
        model.Params.Threads = threads
        model.Params.Seed = seed
        model.Params.OutputFlag = output_flag

        # Makes infeasibility status clearer instead of returning INF_OR_UNBD
        # in cases where reductions obscure the distinction.
        model.Params.DualReductions = 0

        model.update()

        # ---------------------------------------------------------------------
        # Model size
        # ---------------------------------------------------------------------

        result["num_vars"] = model.NumVars
        result["num_constraints"] = model.NumConstrs
        result["num_binary_vars"] = model.NumBinVars
        result["num_integer_vars"] = model.NumIntVars

        # ---------------------------------------------------------------------
        # Solve
        # ---------------------------------------------------------------------

        start_time = time.perf_counter()

        model.optimize()

        wall_runtime = (
            time.perf_counter()
            - start_time
        )

        status_code = model.Status

        status_name = GUROBI_STATUS.get(
            status_code,
            f"STATUS_{status_code}",
        )

        result["milp_status"] = status_name
        result["sol_count"] = model.SolCount

        result["runtime_seconds"] = safe_float(
            model.Runtime
        )

        if result["runtime_seconds"] is None:
            result["runtime_seconds"] = wall_runtime

        result["node_count"] = safe_float(
            model.NodeCount
        )

        # ---------------------------------------------------------------------
        # Incumbent
        # ---------------------------------------------------------------------

        if model.SolCount > 0:

            result["objective"] = safe_float(
                model.ObjVal
            )

        # ---------------------------------------------------------------------
        # Best bound
        # ---------------------------------------------------------------------

        if status_code not in (
            gp.GRB.INFEASIBLE,
            gp.GRB.INF_OR_UNBD,
            gp.GRB.UNBOUNDED,
        ):

            result["best_bound"] = safe_float(
                model.ObjBound
            )

        # ---------------------------------------------------------------------
        # Gap
        # ---------------------------------------------------------------------

        if model.SolCount > 0:

            result["mip_gap"] = safe_float(
                model.MIPGap
            )

        # ---------------------------------------------------------------------
        # Write IIS if the CURRENT MILP is genuinely infeasible
        # ---------------------------------------------------------------------

        if status_code == gp.GRB.INFEASIBLE:

            try:

                model.computeIIS()

                iis_dir = (
                    output_dir
                    / "iis"
                )

                iis_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                iis_path = (
                    iis_dir
                    / f"{instance_name}.ilp"
                )

                model.write(
                    str(iis_path)
                )

            except Exception as iis_error:

                result["error"] = (
                    f"IIS write warning: {iis_error}"
                )

    except Exception as exc:

        result["milp_status"] = "ERROR"

        result["error"] = (
            f"{type(exc).__name__}: {exc}"
        )

    return result


# =============================================================================
# CSV writer
# =============================================================================

def write_results_csv(
    path: Path,
    results: list[dict[str, Any]],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temp_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=RESULT_FIELDS,
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    # Atomic-ish replacement:
    # avoids leaving a half-written CSV if interrupted during writing.
    temp_path.replace(
        path
    )


# =============================================================================
# Formatting
# =============================================================================

def format_number(
    value: Any,
    digits: int = 3,
) -> str:

    if value is None:
        return "-"

    try:
        return f"{float(value):,.{digits}f}"

    except (TypeError, ValueError):
        return str(value)


# =============================================================================
# Print one result
# =============================================================================

def print_result(
    result: dict[str, Any],
) -> None:

    print(
        f"  Status:    "
        f"{result['milp_status']}"
    )

    print(
        f"  Solutions: "
        f"{result['sol_count']}"
    )

    print(
        f"  Objective: "
        f"{format_number(result['objective'])}"
    )

    print(
        f"  Bound:     "
        f"{format_number(result['best_bound'])}"
    )

    if result["mip_gap"] is not None:

        print(
            f"  Gap:       "
            f"{100.0 * result['mip_gap']:.4f}%"
        )

    else:

        print(
            "  Gap:       -"
        )

    print(
        f"  Runtime:   "
        f"{format_number(result['runtime_seconds'], 2)} s"
    )

    if result["error"]:

        print(
            f"  Warning:   "
            f"{result['error']}"
        )


# =============================================================================
# Final summary
# =============================================================================

def print_summary(
    results: list[dict[str, Any]],
) -> None:

    print()
    print("=" * 95)
    print("FINAL SUMMARY")
    print("=" * 95)

    header = (
        f"{'Instance':<43}"
        f"{'Status':<16}"
        f"{'Objective':>18}"
        f"{'Runtime (s)':>16}"
    )

    print(
        header
    )

    print(
        "-" * len(header)
    )

    for result in results:

        if result["objective"] is not None:

            objective = format_number(
                result["objective"],
                2,
            )

        else:

            objective = "-"

        if result["runtime_seconds"] is not None:

            runtime = format_number(
                result["runtime_seconds"],
                2,
            )

        else:

            runtime = "-"

        print(
            f"{result['instance']:<43}"
            f"{result['milp_status']:<16}"
            f"{objective:>18}"
            f"{runtime:>16}"
        )

    print()

    print(
        "Status counts:"
    )

    statuses = sorted(
        {
            result["milp_status"]
            for result in results
        }
    )

    for status in statuses:

        count = sum(
            result["milp_status"] == status
            for result in results
        )

        print(
            f"  {status:<20}: "
            f"{count}"
        )

    feasible_count = sum(
        result["sol_count"] > 0
        for result in results
    )

    optimal_count = sum(
        result["milp_status"] == "OPTIMAL"
        for result in results
    )

    infeasible_count = sum(
        result["milp_status"] == "INFEASIBLE"
        for result in results
    )

    time_limit_count = sum(
        result["milp_status"] == "TIME_LIMIT"
        for result in results
    )

    print()

    print(
        f"Instances with a feasible MILP solution : "
        f"{feasible_count}/{len(results)}"
    )

    print(
        f"Proven optimal                       : "
        f"{optimal_count}/{len(results)}"
    )

    print(
        f"Proven MILP infeasible               : "
        f"{infeasible_count}/{len(results)}"
    )

    print(
        f"Reached time limit                    : "
        f"{time_limit_count}/{len(results)}"
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Run the current multi-commodity MILP on the "
            "16 instances reported infeasible in the "
            "Benders MP with no-overlap."
        )
    )

    parser.add_argument(
        "--scenario-dir",
        type=Path,
        default=DEFAULT_SCENARIO_DIR,
        help=(
            "Root directory containing the multi-commodity "
            "scenario JSON files."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory in which diagnostic results are stored."
        ),
    )

    parser.add_argument(
        "--time-limit",
        type=float,
        default=DEFAULT_TIME_LIMIT,
        help=(
            "Gurobi time limit for each individual MILP "
            "instance, in seconds."
        ),
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help=(
            "Number of Gurobi threads used by EACH MILP solve."
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            "Number of MILP instances solved simultaneously."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            "Gurobi random seed."
        ),
    )

    parser.add_argument(
        "--solver-log",
        action="store_true",
        help=(
            "Show Gurobi output from each worker."
        ),
    )

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> int:

    args = parse_args()

    # -------------------------------------------------------------------------
    # Basic validation
    # -------------------------------------------------------------------------

    if args.workers < 1:
        raise ValueError(
            "--workers must be at least 1"
        )

    if args.threads < 1:
        raise ValueError(
            "--threads must be at least 1"
        )

    if args.time_limit <= 0:
        raise ValueError(
            "--time-limit must be positive"
        )

    scenario_dir = (
        args.scenario_dir
        .resolve()
    )

    output_dir = (
        args.output_dir
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_csv = (
        output_dir
        / "milp_16_mp_infeasible_results.csv"
    )

    # -------------------------------------------------------------------------
    # Header
    # -------------------------------------------------------------------------

    print(
        "=" * 95
    )

    print(
        "MILP diagnostic: "
        "16 instances infeasible in Benders MP"
    )

    print(
        "=" * 95
    )

    print(
        f"Scenario directory : "
        f"{scenario_dir}"
    )

    print(
        f"Results CSV        : "
        f"{results_csv}"
    )

    print(
        f"Time limit         : "
        f"{args.time_limit:.0f} seconds / instance"
    )

    print(
        f"Parallel workers   : "
        f"{args.workers}"
    )

    print(
        f"Threads per MILP   : "
        f"{args.threads}"
    )

    print(
        f"Maximum CPU threads: "
        f"{args.workers * args.threads}"
    )

    print(
        f"Gurobi seed        : "
        f"{args.seed}"
    )

    print()

    # -------------------------------------------------------------------------
    # Find all 16 JSON files
    # -------------------------------------------------------------------------

    found = discover_target_instances(
        scenario_dir
    )

    missing = [
        instance
        for instance in TARGET_INSTANCES
        if instance not in found
    ]

    print(
        f"Found "
        f"{len(found)}/{len(TARGET_INSTANCES)} "
        f"target instances."
    )

    if missing:

        print()
        print(
            "Missing instances:"
        )

        for instance in missing:

            print(
                f"  - {instance}"
            )

        print()
        print(
            "The experiment has NOT started."
        )

        print(
            "Check the --scenario-dir path."
        )

        return 1

    # -------------------------------------------------------------------------
    # Parallel solving
    # -------------------------------------------------------------------------

    print()
    print(
        f"Starting {len(TARGET_INSTANCES)} MILP solves "
        f"using {args.workers} workers..."
    )
    print()

    results: list[dict[str, Any]] = []

    instance_order = {
        name: index
        for index, name
        in enumerate(TARGET_INSTANCES)
    }

    jobs = {}

    with ProcessPoolExecutor(
        max_workers=args.workers
    ) as executor:

        # ---------------------------------------------------------------------
        # Submit all jobs
        # ---------------------------------------------------------------------

        for instance_name in TARGET_INSTANCES:

            scenario_path = found[
                instance_name
            ]

            future = executor.submit(
                solve_instance,
                instance_name,
                scenario_path,
                time_limit=args.time_limit,
                threads=args.threads,
                seed=args.seed,
                output_flag=(
                    1
                    if args.solver_log
                    else 0
                ),
                output_dir=output_dir,
            )

            jobs[future] = (
                instance_name,
                scenario_path,
            )

        # ---------------------------------------------------------------------
        # Process results as workers finish
        # ---------------------------------------------------------------------

        completed = 0

        for future in as_completed(
            jobs
        ):

            instance_name, scenario_path = (
                jobs[future]
            )

            completed += 1

            print()
            print(
                "=" * 95
            )

            print(
                f"[{completed:02d}/{len(TARGET_INSTANCES):02d}] "
                f"COMPLETED: {instance_name}"
            )

            print(
                f"File: {scenario_path}"
            )

            print(
                "-" * 95
            )

            try:

                result = future.result()

            except Exception as exc:

                result = {
                    "instance": instance_name,
                    "scenario_file": str(scenario_path),
                    "mp_no_overlap_status": "INFEASIBLE",
                    "milp_status": "ERROR",
                    "sol_count": 0,
                    "objective": None,
                    "best_bound": None,
                    "mip_gap": None,
                    "runtime_seconds": None,
                    "node_count": None,
                    "num_vars": None,
                    "num_binary_vars": None,
                    "num_integer_vars": None,
                    "num_constraints": None,
                    "time_limit_seconds": args.time_limit,
                    "threads": args.threads,
                    "seed": args.seed,
                    "error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                }

            results.append(
                result
            )

            # -----------------------------------------------------------------
            # Keep deterministic ordering in the CSV even though workers finish
            # in arbitrary order.
            # -----------------------------------------------------------------

            results.sort(
                key=lambda row: instance_order[
                    row["instance"]
                ]
            )

            # -----------------------------------------------------------------
            # Save immediately after EVERY completed instance.
            # -----------------------------------------------------------------

            write_results_csv(
                results_csv,
                results,
            )

            print_result(
                result
            )

            print(
                f"  Progress:  "
                f"{completed}/{len(TARGET_INSTANCES)}"
            )

            print(
                f"  Saved:     "
                f"{results_csv}"
            )

    # -------------------------------------------------------------------------
    # Final output
    # -------------------------------------------------------------------------

    results.sort(
        key=lambda row: instance_order[
            row["instance"]
        ]
    )

    write_results_csv(
        results_csv,
        results,
    )

    print_summary(
        results
    )

    print()

    print(
        "Final results written to:"
    )

    print(
        results_csv
    )

    print()

    return 0


# =============================================================================
# Required for multiprocessing on Windows
# =============================================================================

if __name__ == "__main__":
    raise SystemExit(
        main()
    )