"""Compare Benders decomposition with Gurobi on the full model, like for like.

Both methods get the same schedule constraints, the same relative gap target, the
same thread count and the same wall-clock budget per instance. Model building counts
against each method's budget; loading the JSON does not. Instances run in separate
processes. Same-asset no-overlap ("no-alignment") is ON in both methods by default, as
agreed on 7/10; --allow-overlap turns it off. F3_yard_flow is excluded by default
(dropped at the 7/10 meeting).

Each record also stores what the convergence figures and the validator need, so a run
never has to be repeated: Gurobi's incumbent/bound timeline (via callback), Benders'
per-iteration history, and each method's final maintenance schedule.

  python3 compare_benders_vs_gurobi.py --model single --instances core5
  python3 compare_benders_vs_gurobi.py --model multi --instances core5 --time-limit 600
  python3 compare_benders_vs_gurobi.py --summarize-only results/compare_single_....jsonl
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import statistics
import time

BASE = "synthetic_data_generator/{}_scenarios"
EXCLUDED_FAMILIES = ("F3_yard_flow",)

# Size scaling (F1 L/M/H: 30/60/100 jobs) plus the hardest level of the two other
# medium-network families; seed 1 of each has a valid schedule under no-overlap.
PRESETS = {
    "core5": "F1_system_scale_L:1,F1_system_scale_M:1,F1_system_scale_H:1,"
             "F2_mode_tradeoff_H:1,F4_operational_conflict_H:1",
}


def scenario_path(model, family, seed):
    kind = "singlecommodity" if model == "single" else "multicommodity"
    return f"{BASE.format(kind)}/{family}/seed_{seed:03d}/{family}_seed_{seed:03d}.json"


def modules(model):
    if model == "single":
        import monolithic_milp as milp
        import benders_monolithic as benders
    else:
        import multicommodity_milp as milp
        import benders_multicommodity as benders
    return milp, benders


def schedule_options(no_overlap, cap):
    from schedule_constraints import ScheduleOptions
    return ScheduleOptions(same_asset_no_overlap=no_overlap, max_concurrent_ops=cap)


def families(model):
    kind = "singlecommodity" if model == "single" else "multicommodity"
    root = BASE.format(kind)
    return sorted(os.path.basename(p) for p in glob.glob(f"{root}/*")
                  if os.path.isdir(p) and not os.path.basename(p).startswith(EXCLUDED_FAMILIES))


def master_has_a_schedule(model, family, seed, no_overlap, cap):
    import gurobipy as gp
    milp, benders = modules(model)
    data = milp.load_scenario(scenario_path(model, family, seed))
    master, _ = benders.build_master(data, schedule_options(no_overlap, cap))
    master.Params.OutputFlag = 0
    master.Params.TimeLimit = 60
    master.optimize()
    ok = master.SolCount > 0
    master.dispose()
    return ok


def select_instances(model, spec, no_overlap, cap):
    """'sample' = first seed per family/severity that has a valid schedule; 'all' = every
    instance with a valid schedule; or an explicit comma list of family:seed."""
    spec = PRESETS.get(spec, spec)
    if spec not in ("sample", "all"):
        return [(item.split(":")[0], int(item.split(":")[1])) for item in spec.split(",")]
    chosen, skipped = [], []
    for family in families(model):
        for seed in range(1, 11):
            if not master_has_a_schedule(model, family, seed, no_overlap, cap):
                skipped.append((family, seed))
                continue
            chosen.append((family, seed))
            if spec == "sample":
                break
    if skipped:
        print("skipped (no valid schedule under these constraints):", ", ".join(f"{f}:{s}" for f, s in skipped))
    return chosen


def chosen_schedule(x_values):
    """[(operation, mode, start_day), ...] from a {(m, k, d): value} mapping."""
    return sorted([m, k, int(d)] for (m, k, d), v in x_values.items() if v > 0.5)


def gurobi_timeline_callback(started, timeline, min_interval=1.0):
    """Record (elapsed s, incumbent, bound) on every new incumbent and at most once per
    `min_interval` seconds otherwise -- enough for a gap-vs-time figure."""
    from gurobipy import GRB
    last = [-min_interval]

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            t = time.perf_counter() - started
            # OBJBST is the incumbent *before* this solution (-1e100 at first), so take the better of the two.
            inc = max(model.cbGet(GRB.Callback.MIPSOL_OBJ), model.cbGet(GRB.Callback.MIPSOL_OBJBST))
            timeline.append([t, inc, model.cbGet(GRB.Callback.MIPSOL_OBJBND)])
            last[0] = t
        elif where == GRB.Callback.MIP:
            t = time.perf_counter() - started
            if t - last[0] >= min_interval:
                inc, bnd = model.cbGet(GRB.Callback.MIP_OBJBST), model.cbGet(GRB.Callback.MIP_OBJBND)
                if abs(inc) < GRB.INFINITY:
                    timeline.append([t, inc, bnd])
                    last[0] = t
    return callback


def run_task(task):
    import gurobipy as gp
    from gurobipy import GRB

    gp.setParam("Threads", task["threads"])
    milp, benders = modules(task["model"])
    options = schedule_options(task["no_overlap"], task["cap"])
    record = dict(task, status="ERROR", objective=None, bound=None, gap=None, wall_seconds=None,
                  iterations=None, feasibility_cuts=None, optimality_cuts=None, master_seconds=None,
                  sub_seconds=None, nodes=None, error=None, timeline=None, history=None, schedule=None)
    try:
        data = milp.load_scenario(scenario_path(task["model"], task["family"], task["seed"]))
        started = time.perf_counter()
        if task["method"] == "direct":
            model, variables = milp.build_model(data, schedule=options)
            model.Params.OutputFlag = 0
            model.Params.MIPGap = task["tol"]
            model.Params.TimeLimit = max(task["time_limit"] - (time.perf_counter() - started), 1.0)
            timeline = []
            model.optimize(gurobi_timeline_callback(started, timeline))
            record["timeline"] = timeline
            names = {GRB.OPTIMAL: "OPTIMAL", GRB.TIME_LIMIT: "TIME_LIMIT", GRB.INFEASIBLE: "INFEASIBLE"}
            record["status"] = names.get(model.Status, f"STATUS_{model.Status}")
            if model.SolCount > 0:
                record.update(objective=model.ObjVal, bound=model.ObjBound, gap=model.MIPGap,
                              schedule=chosen_schedule({key: v.X for key, v in variables["x"].items()}))
            record["nodes"] = model.NodeCount
            record["wall_seconds"] = time.perf_counter() - started
        else:
            result = benders.solve_benders(data, tol=task["tol"], schedule=options, time_limit=task["time_limit"],
                                           master_time_limit=task["master_time_limit"], max_iters=10**6)
            record.update(
                status=result["status"], objective=result["best_objective"], bound=result["best_bound"],
                gap=result["gap"], wall_seconds=time.perf_counter() - started, iterations=result["iterations"],
                feasibility_cuts=result["feasibility_cuts"], optimality_cuts=result["optimality_cuts"],
                master_seconds=result["master_time"], sub_seconds=result["sub_time"],
                history=[[h["iter"], h["type"][0], h["elapsed"], h["ub"], h["lb"], h["best_ub"], h["best_lb"]]
                         for h in result["history"]],
                schedule=chosen_schedule(result["best_x"]) if result["best_x"] else None,
            )
            if result["status"].startswith("MASTER_3"):
                record["status"] = "INFEASIBLE"
    except Exception as exc:  # noqa: BLE001 -- keep the batch going, record per-instance failures
        record["error"] = str(exc)
    return record


def summarize(path):
    rows = [json.loads(line) for line in open(path)]
    by_instance = {}
    for r in rows:
        by_instance.setdefault((r["family"], r["seed"]), {})[r["method"]] = r

    def cell(r):
        if r is None:
            return ("–",) * 4
        obj = "–" if r["objective"] is None else f"{r['objective']:,.0f}"
        gap = "–" if r["gap"] is None else f"{100 * r['gap']:.3f}%"
        return f"{r['status']}", obj, gap, f"{(r['wall_seconds'] or 0):.0f}s"

    def benders_cuts(r):
        if r is None or r.get("feasibility_cuts") is None:
            return "–"
        return f"{r['feasibility_cuts']} feas / {r['optimality_cuts']} opt ({r['iterations']} iters)"

    lines = ["| Instance | Gurobi status | Gurobi obj | Gurobi gap | Gurobi time | Benders status | Benders obj | Benders gap | Benders time | Benders cuts | Faster / better |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    tally = dict(direct=0, benders=0, tie=0)
    solved = dict(direct=0, benders=0)
    both_times, obj_diffs = [], []
    for (family, seed), methods in sorted(by_instance.items()):
        d, b = methods.get("direct"), methods.get("benders")
        dc, bc = cell(d), cell(b)
        winner = "–"
        d_ok = bool(d) and d["status"] == "OPTIMAL"
        b_ok = bool(b) and b["status"] == "OPTIMAL"
        solved["direct"] += d_ok
        solved["benders"] += b_ok
        if d and b:
            if d_ok and b_ok:
                winner = "Gurobi" if d["wall_seconds"] < b["wall_seconds"] else "Benders"
                both_times.append((d["wall_seconds"], b["wall_seconds"]))
                if d["objective"] and b["objective"]:
                    obj_diffs.append(abs(d["objective"] - b["objective"]) / abs(d["objective"]))
            elif d_ok != b_ok:
                winner = "Gurobi" if d_ok else "Benders"
            elif d["status"] == "INFEASIBLE" and b["status"] == "INFEASIBLE":
                winner = "both infeasible"
            elif d["gap"] is not None and b["gap"] is not None:
                winner = "Gurobi (smaller gap)" if d["gap"] < b["gap"] else "Benders (smaller gap)"
            if winner.startswith("Gurobi"):
                tally["direct"] += 1
            elif winner.startswith("Benders"):
                tally["benders"] += 1
        lines.append(f"| {family} seed {seed} | " + " | ".join(dc) + " | " + " | ".join(bc) + f" | {benders_cuts(b)} | {winner} |")
    lines.append("")
    n = len(by_instance)
    lines.append(f"Instances: {n}. Solved to the gap target: Gurobi {solved['direct']}, Benders {solved['benders']}. "
                 f"Wins: Gurobi {tally['direct']}, Benders {tally['benders']}.")
    if both_times:
        ratios = [b / d for d, b in both_times if d > 0]
        lines.append(f"Where both solved: median Benders/Gurobi time ratio {statistics.median(ratios):.1f}x.")
    if obj_diffs:
        lines.append(f"Largest objective difference where both solved: {100 * max(obj_diffs):.4f}%.")
    total_feas = sum(r["feasibility_cuts"] for r in rows if r["method"] == "benders" and r.get("feasibility_cuts") is not None)
    total_opt = sum(r["optimality_cuts"] for r in rows if r["method"] == "benders" and r.get("optimality_cuts") is not None)
    lines.append(f"Total Benders cuts across all runs: {total_feas} feasibility, {total_opt} optimality.")
    text = "\n".join(lines)
    out = os.path.splitext(path)[0] + ".md"
    open(out, "w").write(text + "\n")
    print(text)
    print("\nwrote", out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=("single", "multi"), default="single")
    parser.add_argument("--instances", default="sample", help="core5 | sample | all | family:seed,family:seed")
    parser.add_argument("--methods", nargs="+", default=["direct", "benders"], choices=("direct", "benders"))
    parser.add_argument("--time-limit", type=float, default=600.0, help="wall-clock budget per instance per method, seconds")
    parser.add_argument("--master-time-limit", type=float, default=60.0,
                        help="cap on a single Benders master solve, seconds -- keeps one slow iteration "
                             "from eating the whole --time-limit budget; 0 disables the cap")
    parser.add_argument("--allow-overlap", action="store_true",
                        help="drop the same-asset no-overlap constraint from both methods (default: enforced)")
    parser.add_argument("--cap", type=int, default=None, help="re-add a max-operations-per-day cap (default: off)")
    parser.add_argument("--tol", type=float, default=1e-3, help="relative gap target for both methods")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2, help="Gurobi threads per run")
    parser.add_argument("--out", default=None)
    parser.add_argument("--summarize-only", default=None)
    args = parser.parse_args()

    if args.summarize_only:
        summarize(args.summarize_only)
        return

    master_time_limit = None if args.master_time_limit <= 0 else args.master_time_limit
    instances = select_instances(args.model, args.instances, not args.allow_overlap, args.cap)
    tasks = [dict(model=args.model, family=f, seed=s, method=m, no_overlap=not args.allow_overlap, cap=args.cap,
                  tol=args.tol, time_limit=args.time_limit, master_time_limit=master_time_limit,
                  threads=args.threads)
             for f, s in instances for m in args.methods]
    os.makedirs("results", exist_ok=True)
    out = args.out or f"results/compare_{args.model}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    print(f"{len(instances)} instances x {len(args.methods)} methods = {len(tasks)} runs, "
          f"{args.workers} workers x {args.threads} threads, {args.time_limit:.0f}s each -> {out}", flush=True)

    with mp.get_context("spawn").Pool(args.workers) as pool, open(out, "w") as sink:
        for i, record in enumerate(pool.imap_unordered(run_task, tasks), 1):
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            gap = "-" if record["gap"] is None else f"{100 * record['gap']:.3f}%"
            cuts = ""
            if record.get("feasibility_cuts") is not None:
                cuts = f"cuts={record['feasibility_cuts']}feas/{record['optimality_cuts']}opt iters={record['iterations']}"
            print(f"[{i}/{len(tasks)}] {record['family']} seed {record['seed']} {record['method']:8} "
                  f"{record['status']:12} gap={gap:>8} {record['wall_seconds'] or 0:6.0f}s {cuts} "
                  f"{record['error'] or ''}", flush=True)
    summarize(out)


if __name__ == "__main__":
    main()
