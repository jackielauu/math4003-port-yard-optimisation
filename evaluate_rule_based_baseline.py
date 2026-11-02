#!/usr/bin/env python3
"""
Evaluate COMPLETE rule-based maintenance schedules with the same operational MILP.

Purpose
-------
The rule-based baseline constructs maintenance decisions only.  This script:
  1. reads the already-generated *_rule_based_schedule.json payloads;
  2. skips INCOMPLETE schedules;
  3. loads the corresponding original benchmark instance;
  4. builds the same monolithic MILP used for the benchmark;
  5. fixes every maintenance start/mode variable x[m,k,d] to the rule-based decision;
  6. optimises only the remaining operational decisions;
  7. writes a full JSON payload containing objective, inventory, overflow,
     throughput, and the fixed maintenance schedule.

Default model modules
---------------------
single commodity -> monolithic_milp.py
multi commodity  -> multicommodity_milp.py

Both modules are expected to expose:
    load_scenario(path)
    build_model(data, strict_capacity=False)

and build_model must return:
    (model, variables)

where variables contains at least:
    x, z, o, mu

Typical usage
-------------
Single commodity:
python evaluate_rule_based_baseline.py ^
  --commodity single ^
  --schedule-dir results\\rule_based_baseline_schedules_single ^
  --scenario-dir synthetic_data_generator\\singlecommodity_scenarios ^
  --output-dir results\\rule_based_baseline_evaluated_single

Multi commodity:
python evaluate_rule_based_baseline.py ^
  --commodity multi ^
  --schedule-dir results\\rule_based_baseline_schedules ^
  --scenario-dir synthetic_data_generator\\multicommodity_scenarios ^
  --output-dir results\\rule_based_baseline_evaluated_multi
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import time
from pathlib import Path
from typing import Any

import gurobipy as gp


STATUS_NAMES = {
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


def safe_float(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def discover_schedule_files(schedule_dir: Path) -> list[Path]:
    if not schedule_dir.exists():
        raise FileNotFoundError(f"schedule directory does not exist: {schedule_dir}")
    return sorted(schedule_dir.rglob("*_rule_based_schedule.json"))


def instance_id_from_schedule_file(path: Path) -> str:
    suffix = "_rule_based_schedule"
    stem = path.stem
    return stem[:-len(suffix)] if stem.endswith(suffix) else stem


def resolve_scenario_path(
    schedule_payload: dict[str, Any],
    schedule_file: Path,
    scenario_dir: Path | None,
) -> Path:
    # First use the exact source path recorded by rule_based_baseline.py.
    recorded = schedule_payload.get("source_instance_file")
    if recorded:
        candidate = Path(recorded)
        if candidate.exists():
            return candidate

    # Otherwise locate the benchmark instance by its full instance ID.
    if scenario_dir is None:
        raise FileNotFoundError(
            f"could not resolve source instance for {schedule_file}; "
            "supply --scenario-dir"
        )

    instance_id = instance_id_from_schedule_file(schedule_file)
    matches = list(scenario_dir.rglob(f"{instance_id}.json"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"no scenario JSON found for {instance_id} below {scenario_dir}"
        )
    raise RuntimeError(
        f"multiple scenario JSON files found for {instance_id}: "
        + ", ".join(str(p) for p in matches)
    )


def load_model_functions(commodity: str, model_module: str | None):
    module_name = model_module
    if module_name is None:
        module_name = "monolithic_milp" if commodity == "single" else "multicommodity_milp"

    module = importlib.import_module(module_name)
    if not hasattr(module, "load_scenario") or not hasattr(module, "build_model"):
        raise AttributeError(
            f"{module_name} must expose load_scenario() and build_model()"
        )
    return module_name, module.load_scenario, module.build_model


def build_model_robust(build_model, data, strict_capacity: bool):
    try:
        built = build_model(data, strict_capacity=strict_capacity)
    except TypeError:
        # Fallback only for a local model implementation without the keyword.
        built = build_model(data)

    if not isinstance(built, tuple) or len(built) < 2:
        raise TypeError(
            "build_model() must return (model, variables) so the rule-based "
            "maintenance variables can be fixed and operational variables extracted"
        )

    model, variables = built[0], built[1]
    required = {"x", "z", "o", "mu"}
    missing = required - set(variables)
    if missing:
        raise KeyError(
            f"build_model() variables dictionary is missing: {sorted(missing)}"
        )
    return model, variables


def normalise_key(key: Any) -> tuple:
    if isinstance(key, tuple):
        return key
    try:
        return tuple(key)
    except TypeError:
        return (key,)


def fix_maintenance_schedule(
    x_vars,
    schedule: dict[str, Any],
) -> int:
    """
    Fix every x[m,k,d] variable:
      chosen rule-based (mode,start) -> 1
      all other combinations         -> 0
    """
    chosen: dict[str, tuple[str, int]] = {}
    for task_id, decision in schedule.items():
        chosen[str(task_id)] = (
            str(decision["mode"]),
            int(decision["start_day"]),
        )

    model_tasks = set()
    chosen_keys_found = set()

    for key, var in x_vars.items():
        parts = normalise_key(key)
        if len(parts) != 3:
            raise ValueError(f"unexpected x variable key: {key!r}")

        task_id, mode, day = str(parts[0]), str(parts[1]), int(parts[2])
        model_tasks.add(task_id)

        if task_id not in chosen:
            # A COMPLETE rule payload should contain every maintenance task.
            var.LB = 0.0
            var.UB = 0.0
            continue

        selected_mode, selected_day = chosen[task_id]
        selected = (mode == selected_mode and day == selected_day)
        value = 1.0 if selected else 0.0
        var.LB = value
        var.UB = value

        if selected:
            chosen_keys_found.add(task_id)

    missing_from_payload = model_tasks - set(chosen)
    extra_in_payload = set(chosen) - model_tasks
    selected_not_found = set(chosen) - chosen_keys_found

    if missing_from_payload:
        raise ValueError(
            "COMPLETE rule schedule is missing model task(s): "
            + ", ".join(sorted(missing_from_payload))
        )
    if extra_in_payload:
        raise ValueError(
            "rule schedule contains task(s) not present in model: "
            + ", ".join(sorted(extra_in_payload))
        )
    if selected_not_found:
        raise ValueError(
            "chosen (mode,start) not found in x variables for task(s): "
            + ", ".join(sorted(selected_not_found))
        )

    return len(chosen)


def extract_day_series(var_dict) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, var in var_dict.items():
        parts = normalise_key(key)
        if len(parts) != 1:
            raise ValueError(f"expected one-dimensional day variable, got {key!r}")
        day = str(int(parts[0]))
        result[day] = float(var.X)
    return dict(sorted(result.items(), key=lambda kv: int(kv[0])))


def extract_zone_day_series(var_dict) -> dict[str, dict[str, float]]:
    """
    Supports:
      single commodity z[p,d]
      multi commodity  z[p,g,d]

    Multi-commodity inventory is aggregated across products by zone/day so the
    payload directly supports the same stockyard-inventory visualisation.
    """
    aggregated: dict[str, dict[int, float]] = {}

    for key, var in var_dict.items():
        parts = normalise_key(key)

        if len(parts) == 2:
            zone, day = str(parts[0]), int(parts[1])
        elif len(parts) == 3:
            zone, _product, day = str(parts[0]), str(parts[1]), int(parts[2])
        else:
            raise ValueError(f"unexpected inventory variable key: {key!r}")

        aggregated.setdefault(zone, {})
        aggregated[zone][day] = aggregated[zone].get(day, 0.0) + float(var.X)

    return {
        zone: {str(day): values[day] for day in sorted(values)}
        for zone, values in sorted(aggregated.items())
    }


def extract_zone_day_raw_by_product(var_dict) -> dict[str, dict[str, dict[str, float]]]:
    """
    Returns product-level inventory only when z has keys (zone, product, day).
    Returns {} for a single-commodity two-dimensional z.
    """
    output: dict[str, dict[str, dict[int, float]]] = {}

    for key, var in var_dict.items():
        parts = normalise_key(key)
        if len(parts) != 3:
            continue
        zone, product, day = str(parts[0]), str(parts[1]), int(parts[2])
        output.setdefault(zone, {}).setdefault(product, {})[day] = float(var.X)

    return {
        zone: {
            product: {str(day): values[day] for day in sorted(values)}
            for product, values in sorted(products.items())
        }
        for zone, products in sorted(output.items())
    }


def extract_overflow_series(o_vars) -> dict[str, dict[str, float]]:
    output: dict[str, dict[int, float]] = {}
    for key, var in o_vars.items():
        parts = normalise_key(key)
        if len(parts) != 2:
            raise ValueError(f"unexpected overflow variable key: {key!r}")
        zone, day = str(parts[0]), int(parts[1])
        output.setdefault(zone, {})[day] = float(var.X)

    return {
        zone: {str(day): values[day] for day in sorted(values)}
        for zone, values in sorted(output.items())
    }


def scenario_plot_parameters(scenario_path: Path) -> dict[str, Any]:
    """
    Pull the stockyard bounds directly from the frozen instance JSON so the
    evaluator output contains everything needed for the dashed max lines.
    """
    raw = read_json(scenario_path)
    model_ready = raw.get("model_ready_data", {})
    return {
        "zone_min_inventory_tonnes": model_ready.get(
            "zone_min_inventory_tonnes", {}
        ),
        "zone_max_inventory_tonnes": model_ready.get(
            "zone_max_inventory_tonnes", {}
        ),
        "minimum_throughput_tpd": model_ready.get(
            "minimum_throughput_tpd", {}
        ),
    }


def evaluate_one(
    schedule_file: Path,
    *,
    schedule_root: Path,
    scenario_dir: Path | None,
    output_dir: Path,
    load_scenario,
    build_model,
    module_name: str,
    commodity: str,
    strict_capacity: bool,
    time_limit: float,
    threads: int,
    seed: int,
    output_flag: int,
) -> dict[str, Any]:
    rule_payload = read_json(schedule_file)
    instance_id = instance_id_from_schedule_file(schedule_file)

    relative_parent = schedule_file.parent.relative_to(schedule_root)
    output_path = (
        output_dir
        / relative_parent
        / f"{instance_id}_rule_based_evaluated.json"
    )

    base_result: dict[str, Any] = {
        "schema_version": "rule_based_operational_evaluation_v1",
        "instance": instance_id,
        "commodity_representation": commodity,
        "method": "RULE_BASED_FIXED_MILP",
        "construction_status": rule_payload.get("status"),
        "operational_status": "NOT_EVALUATED",
        "source_schedule_file": str(schedule_file),
        "source_instance_file": None,
        "model_module": module_name,
        "model_variant": (
            "strict_no_overflow" if strict_capacity else "penalised_overflow"
        ),
        "task_count": rule_payload.get("task_count"),
        "scheduled_count": rule_payload.get("scheduled_count"),
        "unscheduled_count": rule_payload.get("unscheduled_count"),
        "maintenance_schedule": rule_payload.get("schedule", {}),
        "unscheduled_tasks": rule_payload.get("unscheduled_tasks", []),
        "objective": None,
        "cumulative_throughput_tonnes": None,
        "total_overflow_tonnes": None,
        "daily_throughput_tonnes": {},
        "stockyard_inventory_tonnes": {},
        "stockyard_inventory_by_product_tonnes": {},
        "stockyard_overflow_tonnes": {},
        "plot_parameters": {},
        "solver": {
            "status": None,
            "sol_count": 0,
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
        },
        "error": "",
    }

    # Do not invent operational results for an incomplete maintenance plan.
    if rule_payload.get("status") != "COMPLETE":
        base_result["operational_status"] = "SKIPPED_INCOMPLETE_CONSTRUCTION"
        write_json(output_path, base_result)
        return {
            "instance": instance_id,
            "construction_status": rule_payload.get("status"),
            "operational_status": base_result["operational_status"],
            "objective": None,
            "cumulative_throughput_tonnes": None,
            "total_overflow_tonnes": None,
            "runtime_seconds": None,
            "output_file": str(output_path),
            "error": "",
        }

    try:
        scenario_path = resolve_scenario_path(
            rule_payload, schedule_file, scenario_dir
        )
        base_result["source_instance_file"] = str(scenario_path)
        base_result["plot_parameters"] = scenario_plot_parameters(scenario_path)

        data = load_scenario(str(scenario_path))
        model, variables = build_model_robust(
            build_model, data, strict_capacity
        )

        fixed_count = fix_maintenance_schedule(
            variables["x"],
            rule_payload["schedule"],
        )
        if fixed_count != int(rule_payload["task_count"]):
            raise ValueError(
                f"fixed {fixed_count} tasks but payload task_count="
                f"{rule_payload['task_count']}"
            )

        model.Params.TimeLimit = float(time_limit)
        model.Params.Threads = int(threads)
        model.Params.Seed = int(seed)
        model.Params.OutputFlag = int(output_flag)
        model.Params.DualReductions = 0
        model.update()

        solver = base_result["solver"]
        solver["num_vars"] = model.NumVars
        solver["num_binary_vars"] = model.NumBinVars
        solver["num_integer_vars"] = model.NumIntVars
        solver["num_constraints"] = model.NumConstrs

        start = time.perf_counter()
        model.optimize()
        wall_runtime = time.perf_counter() - start

        status_code = model.Status
        status_name = STATUS_NAMES.get(status_code, f"STATUS_{status_code}")
        solver["status"] = status_name
        solver["sol_count"] = model.SolCount
        solver["runtime_seconds"] = safe_float(model.Runtime) or wall_runtime
        solver["node_count"] = safe_float(model.NodeCount)

        if status_code not in (
            gp.GRB.INFEASIBLE,
            gp.GRB.INF_OR_UNBD,
            gp.GRB.UNBOUNDED,
        ):
            solver["best_bound"] = safe_float(model.ObjBound)

        if model.SolCount > 0:
            solver["mip_gap"] = safe_float(model.MIPGap)
            base_result["objective"] = safe_float(model.ObjVal)

            daily_throughput = extract_day_series(variables["mu"])
            inventory = extract_zone_day_series(variables["z"])
            inventory_by_product = extract_zone_day_raw_by_product(
                variables["z"]
            )
            overflow = extract_overflow_series(variables["o"])

            base_result["daily_throughput_tonnes"] = daily_throughput
            base_result["stockyard_inventory_tonnes"] = inventory
            base_result[
                "stockyard_inventory_by_product_tonnes"
            ] = inventory_by_product
            base_result["stockyard_overflow_tonnes"] = overflow

            base_result["cumulative_throughput_tonnes"] = sum(
                daily_throughput.values()
            )
            base_result["total_overflow_tonnes"] = sum(
                value
                for zone_values in overflow.values()
                for value in zone_values.values()
            )

            if status_code == gp.GRB.OPTIMAL:
                base_result["operational_status"] = "OPTIMAL"
            else:
                base_result["operational_status"] = "FEASIBLE"
        else:
            if status_code == gp.GRB.INFEASIBLE:
                base_result["operational_status"] = "INFEASIBLE"
            elif status_code == gp.GRB.INF_OR_UNBD:
                base_result["operational_status"] = "INF_OR_UNBD"
            else:
                base_result["operational_status"] = "NO_SOLUTION"

    except Exception as exc:
        base_result["operational_status"] = "ERROR"
        base_result["error"] = f"{type(exc).__name__}: {exc}"

    write_json(output_path, base_result)

    return {
        "instance": instance_id,
        "construction_status": base_result["construction_status"],
        "operational_status": base_result["operational_status"],
        "objective": base_result["objective"],
        "cumulative_throughput_tonnes": base_result[
            "cumulative_throughput_tonnes"
        ],
        "total_overflow_tonnes": base_result["total_overflow_tonnes"],
        "runtime_seconds": base_result["solver"]["runtime_seconds"],
        "output_file": str(output_path),
        "error": base_result["error"],
    }


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "instance",
        "construction_status",
        "operational_status",
        "objective",
        "cumulative_throughput_tonnes",
        "total_overflow_tonnes",
        "runtime_seconds",
        "output_file",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate rule-based maintenance schedules by fixing them in the "
            "same operational MILP."
        )
    )
    parser.add_argument(
        "--commodity",
        required=True,
        choices=("single", "multi"),
    )
    parser.add_argument(
        "--schedule-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--scenario-dir",
        type=Path,
        default=None,
        help=(
            "Fallback benchmark root if source_instance_file recorded in the "
            "rule payload is not directly accessible."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--model-module",
        default=None,
        help=(
            "Optional Python module override. Defaults to monolithic_milp for "
            "single and multicommodity_milp for multi."
        ),
    )
    parser.add_argument("--time-limit", type=float, default=1800.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-flag", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict no-overflow model variant.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    module_name, load_scenario, build_model = load_model_functions(
        args.commodity, args.model_module
    )

    schedule_files = discover_schedule_files(args.schedule_dir)
    if not schedule_files:
        raise FileNotFoundError(
            f"no *_rule_based_schedule.json files found below {args.schedule_dir}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Rule-based operational evaluation")
    print(f"  commodity    : {args.commodity}")
    print(f"  model module : {module_name}")
    print(f"  schedules    : {len(schedule_files)}")
    print(f"  input        : {args.schedule_dir}")
    print(f"  output       : {args.output_dir}")
    print(
        "  variant      : "
        + ("strict no-overflow" if args.strict else "penalised overflow")
    )
    print()

    rows: list[dict[str, Any]] = []

    for schedule_file in schedule_files:
        row = evaluate_one(
            schedule_file,
            schedule_root=args.schedule_dir,
            scenario_dir=args.scenario_dir,
            output_dir=args.output_dir,
            load_scenario=load_scenario,
            build_model=build_model,
            module_name=module_name,
            commodity=args.commodity,
            strict_capacity=args.strict,
            time_limit=args.time_limit,
            threads=args.threads,
            seed=args.seed,
            output_flag=args.output_flag,
        )
        rows.append(row)

        print(
            f"{row['construction_status']:<10} "
            f"{row['operational_status']:<34} "
            f"{row['instance']}"
        )

    summary_path = args.output_dir / "summary.csv"
    write_summary(summary_path, rows)

    complete = sum(r["construction_status"] == "COMPLETE" for r in rows)
    solved = sum(
        r["operational_status"] in {"OPTIMAL", "FEASIBLE"} for r in rows
    )
    infeasible = sum(r["operational_status"] == "INFEASIBLE" for r in rows)
    errors = sum(r["operational_status"] == "ERROR" for r in rows)
    skipped = sum(
        r["operational_status"] == "SKIPPED_INCOMPLETE_CONSTRUCTION"
        for r in rows
    )

    print()
    print("Evaluation complete")
    print(f"  total schedules     : {len(rows)}")
    print(f"  construction COMPLETE: {complete}")
    print(f"  operational solved  : {solved}")
    print(f"  operational infeasible: {infeasible}")
    print(f"  skipped incomplete  : {skipped}")
    print(f"  errors              : {errors}")
    print(f"  summary             : {summary_path}")

    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
