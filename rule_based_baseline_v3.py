#!/usr/bin/env python3
"""
Deterministic rule-based maintenance baseline for the MATH4003 benchmark.

This version is aligned with the current MILP's maintenance-capacity logic.

Heuristic
---------
1. Sort maintenance tasks by EDD using maintenance_latest_start l_m.
2. For each task, try maintenance modes in capacity-preserving order:
       P70 -> P30 -> F
3. For the first mode that can be placed, choose the earliest feasible start
   in [e_m, l_m].
4. A candidate placement is accepted only if:
     - the full maintenance interval fits in the planning horizon;
     - it does not touch a maintenance/activity-incompatible asset-day; and
     - after adding the task, the cumulative maintenance capacity loss on the
       asset does not exceed 1.0 on any day.
5. If no mode/start pair is feasible, mark the task unscheduled.

Important
---------
Unlike v1/v2, this script does NOT impose an extra "no two maintenance tasks may
overlap on the same asset" rule.  The current MILP does not contain that
restriction.  Instead, it mirrors the MILP's additive maintenance-capacity
logic: overlapping tasks are allowed only while total capacity loss remains
<= 100%.

There is no mathematical optimisation, no backtracking, no local search, no
throughput calculation, no inventory calculation, no objective calculation,
no scenario-family logic, and no randomness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODE_PRIORITY = ("P70", "P30", "F")
DEFAULT_OUTPUT_DIR = Path("results") / "rule_based_baseline_schedules_v3"
PAYLOAD_SCHEMA_VERSION = "rule_based_baseline_v3"
LOSS_TOL = 1e-9


class BaselineSchemaError(ValueError):
    """Raised when an input JSON file does not match the frozen benchmark schema."""


def load_instance(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if not isinstance(raw, dict) or "model_ready_data" not in raw:
        raise BaselineSchemaError(
            f"{path}: expected top-level 'model_ready_data'"
        )

    data = raw["model_ready_data"]
    required = {
        "D",
        "M",
        "M_a",
        "K",
        "maintenance_duration_days",
        "maintenance_available_capacity_fraction",
        "maintenance_earliest_start",
        "maintenance_latest_start",
        "maintenance_activity_incompatible",
    }

    missing = sorted(required.difference(data))
    if missing:
        raise BaselineSchemaError(
            f"{path}: missing model_ready_data keys: {', '.join(missing)}"
        )

    missing_modes = [mode for mode in MODE_PRIORITY if mode not in data["K"]]
    if missing_modes:
        raise BaselineSchemaError(
            f"{path}: required maintenance mode(s) absent: "
            + ", ".join(missing_modes)
        )

    return raw


def task_to_asset_map(data: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}

    for asset_id, task_ids in data["M_a"].items():
        for task_id in task_ids:
            if task_id in result:
                raise BaselineSchemaError(
                    f"maintenance task {task_id!r} is assigned to more than one asset"
                )
            result[task_id] = asset_id

    missing = [task_id for task_id in data["M"] if task_id not in result]
    if missing:
        raise BaselineSchemaError(
            "maintenance tasks missing from M_a: " + ", ".join(missing)
        )

    return result


def tuple_key(task_id: str, mode: str) -> str:
    return f"{task_id}::{mode}"


def duration_days(
    data: dict[str, Any],
    task_id: str,
    mode: str,
) -> int:
    key = tuple_key(task_id, mode)
    durations = data["maintenance_duration_days"]

    if key not in durations:
        raise BaselineSchemaError(
            f"missing maintenance duration for {key!r}"
        )

    value = int(durations[key])
    if value <= 0:
        raise BaselineSchemaError(
            f"maintenance duration must be positive for {key!r}"
        )
    return value


def retained_capacity_fraction(
    data: dict[str, Any],
    task_id: str,
    mode: str,
) -> float:
    key = tuple_key(task_id, mode)
    values = data["maintenance_available_capacity_fraction"]

    if key not in values:
        raise BaselineSchemaError(
            f"missing available-capacity fraction for {key!r}"
        )

    eta = float(values[key])
    if not 0.0 <= eta <= 1.0:
        raise BaselineSchemaError(
            f"invalid available-capacity fraction for {key!r}: {eta}"
        )
    return eta


def capacity_loss_fraction(
    data: dict[str, Any],
    task_id: str,
    mode: str,
) -> float:
    return 1.0 - retained_capacity_fraction(data, task_id, mode)


def activity_incompatible(
    data: dict[str, Any],
    asset_id: str,
    day: int,
) -> bool:
    key = f"{asset_id}::{day}"
    return int(
        data["maintenance_activity_incompatible"].get(key, 0)
    ) == 1


def active_days(start_day: int, duration: int) -> range:
    return range(start_day, start_day + duration)


def build_rule_based_schedule(
    raw: dict[str, Any],
    source_path: Path,
) -> dict[str, Any]:
    data = raw["model_ready_data"]
    task_asset = task_to_asset_map(data)

    days = sorted(int(day) for day in data["D"])
    if not days:
        raise BaselineSchemaError("D is empty")
    horizon = set(days)

    earliest = {
        task_id: int(data["maintenance_earliest_start"][task_id])
        for task_id in data["M"]
    }
    latest = {
        task_id: int(data["maintenance_latest_start"][task_id])
        for task_id in data["M"]
    }

    # EDD: l_m is the deadline proxy available in the frozen schema.
    ordered_tasks = sorted(
        data["M"],
        key=lambda task_id: (
            latest[task_id],
            earliest[task_id],
            task_id,
        ),
    )

    # Current cumulative maintenance loss by asset/day.
    # This replaces the overly restrictive occupied-day set used in v1/v2.
    loss_by_asset_day: dict[str, dict[int, float]] = {}

    schedule: dict[str, dict[str, Any]] = {}
    unscheduled: list[dict[str, Any]] = []

    for task_id in ordered_tasks:
        asset_id = task_asset[task_id]
        asset_loss = loss_by_asset_day.setdefault(asset_id, {})

        chosen_mode: str | None = None
        chosen_start: int | None = None
        chosen_duration: int | None = None
        chosen_loss: float | None = None
        chosen_eta: float | None = None

        for mode in MODE_PRIORITY:
            duration = duration_days(data, task_id, mode)
            eta = retained_capacity_fraction(data, task_id, mode)
            loss = 1.0 - eta

            for start_day in range(
                earliest[task_id],
                latest[task_id] + 1,
            ):
                interval = tuple(active_days(start_day, duration))

                # Candidate must fit inside the planning horizon.
                if any(day not in horizon for day in interval):
                    continue

                # Match the MILP's fixed-activity incompatibility logic.
                if any(
                    activity_incompatible(data, asset_id, day)
                    for day in interval
                ):
                    continue

                # Match the MILP's additive maintenance-loss structure.
                # We reject a placement if it would make 1 - total_loss < 0.
                if any(
                    asset_loss.get(day, 0.0) + loss > 1.0 + LOSS_TOL
                    for day in interval
                ):
                    continue

                chosen_mode = mode
                chosen_start = start_day
                chosen_duration = duration
                chosen_loss = loss
                chosen_eta = eta
                break

            if chosen_start is not None:
                break

        if chosen_start is None:
            unscheduled.append(
                {
                    "task_id": task_id,
                    "asset_id": asset_id,
                    "earliest_start_day": earliest[task_id],
                    "latest_start_day": latest[task_id],
                    "mode_priority_tried": list(MODE_PRIORITY),
                    "reason": "no_capacity_compatible_mode_start_in_window",
                }
            )
            continue

        assert chosen_mode is not None
        assert chosen_duration is not None
        assert chosen_loss is not None
        assert chosen_eta is not None

        interval = tuple(active_days(chosen_start, chosen_duration))

        for day in interval:
            asset_loss[day] = asset_loss.get(day, 0.0) + chosen_loss

        schedule[task_id] = {
            "asset_id": asset_id,
            "mode": chosen_mode,
            "start_day": chosen_start,
            "duration_days": chosen_duration,
            "end_day": chosen_start + chosen_duration - 1,
            "available_capacity_fraction": chosen_eta,
            "capacity_loss_fraction": chosen_loss,
        }

    metadata = raw.get("metadata", {})
    instance_name = (
        metadata.get("scenario_name")
        or metadata.get("instance_name")
        or source_path.stem
    )

    mode_counts = {
        mode: sum(
            1
            for decision in schedule.values()
            if decision["mode"] == mode
        )
        for mode in MODE_PRIORITY
    }

    max_daily_loss = 0.0
    for per_day in loss_by_asset_day.values():
        if per_day:
            max_daily_loss = max(max_daily_loss, max(per_day.values()))

    return {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "instance": instance_name,
        "source_instance_file": str(source_path),
        "heuristic": {
            "name": "EDD_capacity_loss_aware_earliest_feasible",
            "ordering_rule": (
                "ascending maintenance_latest_start; ties by "
                "maintenance_earliest_start then task_id"
            ),
            "mode_rule": (
                "prefer P70, then P30, then F; "
                "choose earliest feasible start within the first schedulable mode"
            ),
            "mode_priority": list(MODE_PRIORITY),
            "placement_rule": "earliest feasible start",
            "conflict_avoidance": [
                "maintenance_activity_incompatible",
                "cumulative_same_asset_maintenance_loss_must_not_exceed_1",
            ],
            "allows_same_asset_overlap": True,
            "uses_optimisation": False,
            "uses_throughput_information": False,
            "uses_inventory_information": False,
            "uses_demand_information": False,
            "uses_scenario_family_logic": False,
            "uses_randomness": False,
            "uses_backtracking": False,
        },
        "status": "COMPLETE" if not unscheduled else "INCOMPLETE",
        "task_count": len(data["M"]),
        "scheduled_count": len(schedule),
        "unscheduled_count": len(unscheduled),
        "mode_counts": mode_counts,
        "maximum_cumulative_maintenance_loss": round(max_daily_loss, 10),
        "edd_order": ordered_tasks,
        "schedule": schedule,
        "unscheduled_tasks": unscheduled,
    }


def write_payload(
    payload: dict[str, Any],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=False,
        )
        handle.write("\n")

    return output_path


def discover_instance_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(
            f"input path does not exist: {input_path}"
        )

    excluded_names = {
        "manifest.json",
        "benchmark_manifest.json",
        "notebook_validation.json",
    }

    files: list[Path] = []

    for path in sorted(input_path.rglob("*.json")):
        if path.name in excluded_names:
            continue
        if path.name.endswith("_rule_based_schedule.json"):
            continue
        files.append(path)

    return files


def output_path_for(
    instance_path: Path,
    input_path: Path,
    output_dir: Path,
) -> Path:
    if input_path.is_file():
        relative_parent = Path()
    else:
        relative_parent = instance_path.parent.relative_to(input_path)

    return (
        output_dir
        / relative_parent
        / f"{instance_path.stem}_rule_based_schedule.json"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic EDD/capacity-loss-aware "
            "rule-based maintenance schedules."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="One scenario JSON file or a directory containing scenario JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    instance_files = discover_instance_files(args.input)

    if not instance_files:
        print(f"No scenario JSON files found under {args.input}")
        return 1

    complete = 0
    incomplete = 0
    failed = 0

    for instance_path in instance_files:
        try:
            raw = load_instance(instance_path)
            payload = build_rule_based_schedule(raw, instance_path)

            output_path = output_path_for(
                instance_path,
                args.input,
                args.output_dir,
            )
            write_payload(payload, output_path)

            if payload["status"] == "COMPLETE":
                complete += 1
            else:
                incomplete += 1

            mode_counts = payload["mode_counts"]

            print(
                f"{payload['status']:10s} "
                f"{payload['scheduled_count']:>3}/{payload['task_count']:<3} "
                f"P70={mode_counts['P70']:<3} "
                f"P30={mode_counts['P30']:<3} "
                f"F={mode_counts['F']:<3} "
                f"maxloss={payload['maximum_cumulative_maintenance_loss']:.2f} "
                f"{instance_path} -> {output_path}"
            )

        except (
            OSError,
            json.JSONDecodeError,
            BaselineSchemaError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            failed += 1
            print(f"ERROR      {instance_path}: {exc}")

    print()
    print("Rule-based baseline generation complete")
    print(f"  instances : {len(instance_files)}")
    print(f"  complete  : {complete}")
    print(f"  incomplete: {incomplete}")
    print(f"  errors    : {failed}")
    print(f"  output    : {args.output_dir}")

    return 0 if incomplete == 0 and failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
