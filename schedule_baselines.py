"""Non-optimised maintenance schedules, for section 4.1 (part 2) of the results.

Three baselines, each judged by validate_solution.py (schedule rules, then the flows
re-optimised with the schedule fixed):

  earliest_random   every job starts on its earliest allowed day in a uniformly random
                    mode, with no conflict avoidance; repeated over several seeds
                    ("how many simple schedules are even feasible?")
  edd               Ponpot's earliest-deadline-first priority rule (Rule-Based
                    Maintenance Baseline, Algorithm 1): jobs by (l_m, e_m, id), modes
                    P70 -> P30 -> F, earliest admissible start, no backtracking.
                    overlap="cumulative" reproduces his version (jobs may overlap on an
                    asset while total loss <= 100%); overlap="strict" enforces the agreed
                    same-asset no-overlap rule.
  fixed_mode        the full MILP with every job forced into one mode (start days still
                    optimised) -- compared with the free-mode optimum, this measures what
                    the mode decision is worth.

  python3 schedule_baselines.py --instances core5 --model single
"""

import argparse
import csv
import json
import os
import random
import time

from validate_solution import check_schedule, load, resolve, scenario_path

MODE_PRIORITY = ("P70", "P30", "F")
CORE5 = ["F1_system_scale_L_seed_001", "F1_system_scale_M_seed_001", "F1_system_scale_H_seed_001",
         "F2_mode_tradeoff_H_seed_001", "F4_operational_conflict_H_seed_001"]


def earliest_random(data, rng):
    return [(m, rng.choice(list(data["K"])), data["e_m"][m]) for m in data["M"]]


def edd(data, overlap="strict"):
    """Returns (schedule, unscheduled jobs)."""
    asset_of = {m: a for a, ms in data["M_a"].items() for m in ms}
    load = {}
    schedule, unscheduled = [], []
    horizon_end = max(data["D"])
    for m in sorted(data["M"], key=lambda m: (data["l_m"][m], data["e_m"][m], m)):
        a = asset_of[m]
        placed = False
        for k in MODE_PRIORITY:
            if k not in data["K"]:
                continue
            loss, dur = 1.0 - data["eta"][(m, k)], data["p"][(m, k)]
            for s in range(data["e_m"][m], data["l_m"][m] + 1):
                days = range(s, s + dur)
                if s + dur - 1 > horizon_end or any(data["h"].get((a, t), 0) for t in days):
                    continue
                if overlap == "strict":
                    ok = all(load.get((a, t), 0.0) == 0.0 for t in days)
                else:
                    ok = all(load.get((a, t), 0.0) + loss <= 1.0 + 1e-9 for t in days)
                if ok:
                    for t in days:
                        load[(a, t)] = load.get((a, t), 0.0) + loss
                    schedule.append((m, k, s))
                    placed = True
                    break
            if placed:
                break
        if not placed:
            unscheduled.append(m)
    return schedule, unscheduled


def fixed_mode_optimum(module, data, mode, time_limit, threads):
    """Solve the full model (agreed formulation, no-overlap on) with every job in `mode`."""
    from gurobipy import GRB
    model, variables = module.build_model(data)
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit
    model.Params.Threads = threads
    model.Params.MIPGap = 1e-3
    model.Params.DualReductions = 0  # report INFEASIBLE rather than INF_OR_UNBD (the model is bounded)
    for (m, k, d), var in variables["x"].items():
        if k != mode:
            var.UB = 0.0
    started = time.perf_counter()
    model.optimize()
    status = {GRB.OPTIMAL: "OPTIMAL", GRB.INFEASIBLE: "INFEASIBLE", GRB.TIME_LIMIT: "TIME_LIMIT"}.get(model.Status, str(model.Status))
    result = dict(status=status, objective=model.ObjVal if model.SolCount else None,
                  gap=model.MIPGap if model.SolCount else None, seconds=time.perf_counter() - started)
    model.dispose()
    return result


def judge(module, data, schedule, threads):
    """Schedule-rule violations (layer A) plus a flow re-solve (layer C)."""
    v = {k: len(x) for k, x in check_schedule(data, schedule).items() if x}
    rules_ok = not v
    status, obj = resolve(module, data, schedule, threads=threads)
    return dict(rule_violations=";".join(f"{k}:{n}" for k, n in sorted(v.items())), rules_ok=rules_ok,
                flow_status=status, objective=obj, feasible=rules_ok and status == "OPTIMAL")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=("single", "multi"), default="single")
    ap.add_argument("--instances", default="core5", help="core5 or comma-separated instance names")
    ap.add_argument("--random-seeds", type=int, default=10)
    ap.add_argument("--fixed-mode-time-limit", type=float, default=600.0)
    ap.add_argument("--skip-fixed-mode", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    instances = CORE5 if args.instances == "core5" else args.instances.split(",")
    rows = []
    for inst in instances:
        module, data = load(args.model, scenario_path(args.model, inst))

        for seed in range(1, args.random_seeds + 1):
            sched = earliest_random(data, random.Random(seed))
            rows.append(dict(instance=inst, model=args.model, baseline="earliest_random", variant=f"seed {seed}",
                             unscheduled=0, **judge(module, data, sched, args.threads)))

        for overlap in ("cumulative", "strict"):
            sched, unscheduled = edd(data, overlap)
            row = dict(instance=inst, model=args.model, baseline="edd", variant=f"overlap={overlap}",
                       unscheduled=len(unscheduled))
            if unscheduled:
                row.update(rule_violations=f"not_scheduled:{len(unscheduled)}", rules_ok=False,
                           flow_status="NOT_EVALUATED", objective=None, feasible=False)
            else:
                row.update(judge(module, data, sched, args.threads))
            rows.append(row)

        if not args.skip_fixed_mode:
            for mode in data["K"]:
                r = fixed_mode_optimum(module, data, mode, args.fixed_mode_time_limit, args.threads)
                rows.append(dict(instance=inst, model=args.model, baseline="fixed_mode", variant=f"all {mode}",
                                 unscheduled=0, rule_violations="", rules_ok=r["status"] != "INFEASIBLE",
                                 flow_status=r["status"], objective=r["objective"],
                                 feasible=r["objective"] is not None, gap=r["gap"], seconds=r["seconds"]))
        done = [r for r in rows if r["instance"] == inst]
        print(f"{inst}: random feasible {sum(r['feasible'] for r in done if r['baseline'] == 'earliest_random')}"
              f"/{args.random_seeds}; " + "; ".join(f"{r['variant']} {r['flow_status']} {r['objective']}"
                                                    for r in done if r["baseline"] != "earliest_random"), flush=True)

    out = args.out or f"results/baselines_{args.model}.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
