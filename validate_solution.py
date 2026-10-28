"""Independently validate maintenance schedules and operational results.

Three layers of checks, each reported separately:

  A. Schedule rules, from the instance data alone (no solver):
       every job scheduled exactly once, with a valid mode; start inside its window;
       finishes inside the horizon; no maintenance on a fixed-activity day (h=1);
       at most one job per asset per day (same-asset no-overlap / "no-alignment");
       cumulative capacity loss per asset-day <= 100% (implied by the asset-capacity
       constraint, since flow cannot be negative).
  B. Reported operational values, if the source has them: daily throughput >= mu_min,
       Zmin <= z <= Zmax + o, overflow >= 0, and objective = sum(mu) - sum(alpha * o).
  C. Independent re-solve: fix the schedule in the full model, re-optimise the flows,
       and confirm the model is feasible and (if reported) the objective matches.
       The re-solve allows overlap, so A's no-overlap check is the only place that
       rule is judged and C isolates flow feasibility.

Sources:
  - Ponpot's rule-based payload JSONs (payload_rule_based_baseline/...)
  - compare_benders_vs_gurobi.py JSON-lines records (one per method per instance)

  python3 validate_solution.py payload_rule_based_baseline/rule_based_baseline_evaluated_single_v3
  python3 validate_solution.py results/core5_single.jsonl --resolve
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

TOL = 1e-6
REL_TOL = 1e-6


def scenario_path(representation, instance):
    family_level, seed = instance.rsplit("_seed_", 1)
    kind = "singlecommodity" if representation == "single" else "multicommodity"
    return f"synthetic_data_generator/{kind}_scenarios/{family_level}/seed_{seed}/{instance}.json"


def load(representation, path):
    if representation == "single":
        import monolithic_milp as module
    else:
        import multicommodity_milp as module
    return module, module.load_scenario(path)


# ---------------------------------------------------------------- sources

def from_payload(path):
    d = json.load(open(path))
    rep = d["commodity_representation"]
    sched = d.get("maintenance_schedule") or {}
    ops = {
        "label": f"{d['instance']} [{rep}] rule-based",
        "instance": d["instance"],
        "representation": rep,
        "schedule": [(m, v["mode"], int(v["start_day"])) for m, v in sched.items()],
        "unscheduled": list(d.get("unscheduled_tasks") or []),
        "source_status": f"{d.get('construction_status')}/{d.get('operational_status')}",
    }
    if d.get("operational_status") == "OPTIMAL":
        ops["throughput"] = {int(k): v for k, v in d["daily_throughput_tonnes"].items()}
        ops["inventory"] = {z: {int(k): v for k, v in days.items()} for z, days in d["stockyard_inventory_tonnes"].items()}
        ops["overflow"] = {z: {int(k): v for k, v in days.items()} for z, days in d["stockyard_overflow_tonnes"].items()}
        ops["objective"] = d["objective"]
    return [ops]


def from_compare_jsonl(path):
    out = []
    for line in open(path):
        r = json.loads(line)
        if not r.get("schedule"):
            continue
        instance = f"{r['family']}_seed_{int(r['seed']):03d}"
        out.append({
            "label": f"{instance} [{r['model']}] {r['method']}",
            "instance": instance,
            "representation": r["model"],
            "schedule": [tuple(s) for s in r["schedule"]],
            "unscheduled": [],
            "objective": r.get("objective"),
            "source_status": r.get("status"),
        })
    return out


def collect(inputs):
    items = []
    for item in inputs:
        if os.path.isdir(item):
            for p in sorted(glob.glob(os.path.join(item, "**", "*.json"), recursive=True)):
                items.extend(from_payload(p))
        elif item.endswith(".jsonl"):
            items.extend(from_compare_jsonl(item))
        else:
            items.extend(from_payload(item))
    return items


# ---------------------------------------------------------------- checks

def check_schedule(data, schedule):
    """Layer A. Returns {check name: list of violation strings}."""
    v = defaultdict(list)
    D, K = data["D"], set(data["K"])
    asset_of = {m: a for a, ms in data["M_a"].items() for m in ms}
    count = defaultdict(int)
    load = defaultdict(float)
    busy = defaultdict(list)

    for m, k, start in schedule:
        count[m] += 1
        if m not in asset_of:
            v["unknown_job"].append(m)
            continue
        if k not in K:
            v["invalid_mode"].append(f"{m}:{k}")
            continue
        a, dur = asset_of[m], data["p"][(m, k)]
        if not data["e_m"][m] <= start <= data["l_m"][m]:
            v["start_outside_window"].append(f"{m} starts {start}, window {data['e_m'][m]}-{data['l_m'][m]}")
        if start + dur - 1 > max(D):
            v["ends_after_horizon"].append(f"{m} ends {start + dur - 1}")
        for t in range(start, start + dur):
            if data["h"].get((a, t), 0):
                v["fixed_activity_day"].append(f"{m} on {a} day {t}")
            busy[(a, t)].append(m)
            load[(a, t)] += 1.0 - data["eta"][(m, k)]

    for m in data["M"]:
        if count[m] == 0:
            v["not_scheduled"].append(m)
        elif count[m] > 1:
            v["scheduled_more_than_once"].append(m)
    for (a, t), ms in sorted(busy.items()):
        if len(ms) > 1:
            v["same_asset_overlap"].append(f"{a} day {t}: {len(ms)} jobs")
    for (a, t), l in sorted(load.items()):
        if l > 1 + 1e-9:
            v["capacity_loss_over_100pct"].append(f"{a} day {t}: {100 * l:.0f}%")
    return v


def check_operations(data, item):
    """Layer B. Only runs if the source reported operational values."""
    v = defaultdict(list)
    if "throughput" not in item:
        return v
    mu, z, o = item["throughput"], item["inventory"], item["overflow"]
    for d in data["D"]:
        if mu[d] < data["mu_min"].get(d, 0.0) - 1e-3:
            v["throughput_below_minimum"].append(f"day {d}: {mu[d]:.0f} < {data['mu_min'][d]:.0f}")
    for p in data["P_"]:
        for d in data["D"]:
            if z[p][d] < data["Z_min"][p] - 1e-3:
                v["inventory_below_minimum"].append(f"{p} day {d}")
            if z[p][d] > data["Z_max"][p] + o[p][d] + 1e-3:
                v["inventory_above_max_plus_overflow"].append(f"{p} day {d}")
            if o[p][d] < -1e-6:
                v["negative_overflow"].append(f"{p} day {d}")
    recomputed = sum(mu.values()) - sum(data["alpha"][p] * o[p][d] for p in data["P_"] for d in data["D"])
    if abs(recomputed - item["objective"]) > max(1e-3, REL_TOL * abs(item["objective"])):
        v["objective_inconsistent"].append(f"reported {item['objective']:,.1f} vs recomputed {recomputed:,.1f}")
    return v


def resolve(module, data, schedule, threads=1, time_limit=120):
    """Layer C. Fix the schedule, re-optimise flows; overlap allowed (judged in layer A)."""
    import gurobipy as gp
    from gurobipy import GRB
    from schedule_constraints import ScheduleOptions

    chosen = {(m, k, s) for m, k, s in schedule}
    model, variables = module.build_model(data, schedule=ScheduleOptions.none())
    model.Params.OutputFlag = 0
    model.Params.Threads = threads
    model.Params.TimeLimit = time_limit
    model.Params.DualReductions = 0
    for key, var in variables["x"].items():
        var.LB = var.UB = 1.0 if key in chosen else 0.0
    model.optimize()
    status = {GRB.OPTIMAL: "OPTIMAL", GRB.INFEASIBLE: "INFEASIBLE", GRB.TIME_LIMIT: "TIME_LIMIT"}.get(model.Status, str(model.Status))
    obj = model.ObjVal if model.SolCount else None
    model.dispose()
    return status, obj


# ---------------------------------------------------------------- driver

SCHEDULE_CHECKS = ["not_scheduled", "scheduled_more_than_once", "unknown_job", "invalid_mode", "start_outside_window",
                   "ends_after_horizon", "fixed_activity_day", "same_asset_overlap", "capacity_loss_over_100pct"]
OPERATION_CHECKS = ["throughput_below_minimum", "inventory_below_minimum", "inventory_above_max_plus_overflow",
                    "negative_overflow", "objective_inconsistent"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="payload JSON files/directories, or compare_*.jsonl files")
    ap.add_argument("--resolve", action="store_true", help="also run layer C (re-optimise flows with the schedule fixed)")
    ap.add_argument("--exclude-family", default="F3_yard_flow", help="instance prefix to skip ('' keeps all)")
    ap.add_argument("--out", default=None, help="CSV of per-schedule results")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    items = collect(args.inputs)
    if args.exclude_family:
        items = [i for i in items if not i["instance"].startswith(args.exclude_family)]
    rows, cache = [], {}
    for item in items:
        key = (item["representation"], item["instance"])
        if key not in cache:
            cache[key] = load(item["representation"], scenario_path(*key))
        module, data = cache[key]

        sched_v = check_schedule(data, item["schedule"])
        ops_v = check_operations(data, item)
        row = dict(label=item["label"], source_status=item["source_status"])
        for c in SCHEDULE_CHECKS + OPERATION_CHECKS:
            row[c] = len((sched_v if c in SCHEDULE_CHECKS else ops_v).get(c, []))
        rules = [c for c in SCHEDULE_CHECKS if c != "same_asset_overlap"]
        row["schedule_valid_excl_overlap"] = all(row[c] == 0 for c in rules)
        row["schedule_valid_with_no_overlap"] = row["schedule_valid_excl_overlap"] and row["same_asset_overlap"] == 0
        row["operations_valid"] = all(row[c] == 0 for c in OPERATION_CHECKS) if "throughput" in item else None

        if args.resolve and not sched_v.get("not_scheduled") and not any(sched_v.get(c) for c in
                                                                         ("invalid_mode", "unknown_job")):
            status, obj = resolve(module, data, item["schedule"])
            row["resolve_status"] = status
            row["resolve_objective"] = obj
            reported = item.get("objective")
            row["resolve_matches_reported"] = (
                None if reported is None or obj is None
                else abs(obj - reported) <= max(1e-3, 1e-6 * abs(reported))
            )
        rows.append(row)
        if args.verbose:
            for c, msgs in list(sched_v.items()) + list(ops_v.items()):
                print(f"  {item['label']}: {c}: {msgs[:3]}{' ...' if len(msgs) > 3 else ''}")

    n = len(rows)
    print(f"Validated {n} schedules.")
    for c in SCHEDULE_CHECKS + OPERATION_CHECKS:
        bad = sum(1 for r in rows if r[c])
        if bad:
            print(f"  {c:36} violated in {bad}/{n} schedules")
    print(f"  schedule valid (all rules except no-overlap): {sum(r['schedule_valid_excl_overlap'] for r in rows)}/{n}")
    print(f"  schedule valid (all rules incl. no-overlap):  {sum(r['schedule_valid_with_no_overlap'] for r in rows)}/{n}")
    ops = [r for r in rows if r["operations_valid"] is not None]
    if ops:
        print(f"  reported operations valid: {sum(r['operations_valid'] for r in ops)}/{len(ops)}")
    if args.resolve:
        st = defaultdict(int)
        for r in rows:
            st[r.get("resolve_status", "not run")] += 1
        print(f"  re-solve with schedule fixed: {dict(st)}")
        m = [r for r in rows if r.get("resolve_matches_reported") is not None]
        if m:
            print(f"  re-solved objective matches reported: {sum(r['resolve_matches_reported'] for r in m)}/{len(m)}")
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
