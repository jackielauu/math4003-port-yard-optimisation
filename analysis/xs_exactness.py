"""Exactness check on a designed suite of extra-small instances: the full MILP and Benders,
both run to proved optimality, must return the same objective.

Instances: the F1 system-scale topology at three extra-small sizes (4, 6 and 8 machines with
8, 14 and 20 maintenance jobs), seeds 1-3, single- and multi-commodity, on a 60-day horizon
(the generator's minimum). The sizes are registered as extra levels at run time, so the
generator file itself is unchanged; instances are built and validated by its normal path.

Each instance is solved with the same-asset no-overlap constraint on and off. The full model
runs with MIPGap 0. Benders uses exact masters (MIPGap 0, no per-master time cap, unlike the
60 s cap on the main benchmark) and stops at the dissertation's relative-gap tolerance of 1e-3
(or --tol). Agreement is judged by the objective difference, not the stopping rule. Both have
a 600 s safety limit.

Run from the repository root (outputs are never overwritten):
  python3 analysis/xs_exactness.py results/xs_exactness              # tol 1e-3 -> xs_exactness.csv
  python3 analysis/xs_exactness.py results/xs_exactness --tol 1e-6   # -> xs_exactness_tol1e-06.csv
"""
import csv, json, multiprocessing as mp, os, sys, time
sys.path.insert(0, os.getcwd())

XS_LEVELS = {  # level: (machines per stage, maintenance jobs)
    "XS4": ({"train_unloader": 1, "stacker": 1, "reclaimer": 1, "shiploader": 1}, 8),
    "XS6": ({"train_unloader": 1, "stacker": 2, "reclaimer": 2, "shiploader": 1}, 14),
    "XS8": ({"train_unloader": 2, "stacker": 2, "reclaimer": 2, "shiploader": 2}, 20),
}
SEEDS = (1, 2, 3)
HORIZON = 60
TIME_LIMIT = 600.0


def generate(out_dir):
    sys.path.insert(0, os.path.join(os.getcwd(), "synthetic_data_generator"))
    import generate_dissertation_scenarios as g
    names = tuple(f"F1_system_scale_{level}" for level in XS_LEVELS)
    for level, (profile, jobs) in XS_LEVELS.items():
        g.SYSTEM_SCALE_PROFILES[level] = profile
        g.MAINTENANCE_OPERATION_COUNTS[level] = jobs
    g.SCENARIO_NAMES = g.SCENARIO_NAMES + names
    paths = {}
    for model, count in (("single", 1), ("multi", 4)):
        root = os.path.join(out_dir, "instances", model)
        gen = g.generate_benchmark_suite(root, config=g.GeneratorConfig(commodity_count=count, horizon_days=HORIZON),
                                         scenario_names=names, seeds=SEEDS)
        for key, path in gen.items():
            if key != "benchmark_manifest":
                paths[(model, key)] = str(path)
    return paths


def solve(task):
    import gurobipy as gp
    from gurobipy import GRB
    from schedule_constraints import ScheduleOptions
    gp.setParam("OutputFlag", 0)
    gp.setParam("Threads", 1)
    gp.setParam("MIPGap", 0.0)  # applies to the full model and to every Benders master
    if task["model"] == "single":
        import monolithic_milp as milp, benders_monolithic as benders
    else:
        import multicommodity_milp as milp, benders_multicommodity as benders
    options = ScheduleOptions(same_asset_no_overlap=task["no_overlap"], max_concurrent_ops=None)
    data = milp.load_scenario(task["path"])
    row = dict(instance=task["instance"], model=task["model"], no_overlap=task["no_overlap"],
               machines=len(data["A"]), jobs=len(data["M"]), days=len(data["D"]),
               benders_tol=task["tol"], masters="exact (MIPGap 0, no time cap)")

    m, _ = milp.build_model(data, schedule=options)
    m.Params.TimeLimit = TIME_LIMIT
    m.Params.DualReductions = 0  # report INFEASIBLE rather than INF_OR_UNBD (the model is bounded)
    t = time.perf_counter(); m.optimize(); dt = time.perf_counter() - t
    names = {GRB.OPTIMAL: "OPTIMAL", GRB.INFEASIBLE: "INFEASIBLE", GRB.TIME_LIMIT: "TIME_LIMIT", GRB.INF_OR_UNBD: "INF_OR_UNBD"}
    row.update(direct_status=names.get(m.Status, str(m.Status)), direct_objective=m.ObjVal if m.SolCount else None,
               direct_bound=m.ObjBound if m.SolCount else None, direct_seconds=round(dt, 3))

    t = time.perf_counter()
    b = benders.solve_benders(milp.load_scenario(task["path"]), max_iters=10**6, tol=task["tol"], schedule=options,
                              time_limit=TIME_LIMIT, master_time_limit=None)
    row.update(benders_status=b["status"], benders_objective=b["best_objective"], benders_bound=b["best_bound"],
               benders_gap=b["gap"], benders_iterations=b["iterations"], feasibility_cuts=b["feasibility_cuts"],
               optimality_cuts=b["optimality_cuts"], benders_seconds=round(time.perf_counter() - t, 3))
    if row["direct_objective"] is not None and row["benders_objective"] is not None:
        diff = row["benders_objective"] - row["direct_objective"]
        row.update(abs_diff=diff, rel_diff=diff / abs(row["direct_objective"]))
    else:
        row.update(abs_diff=None, rel_diff=None)
    return row


def main():
    out_dir = sys.argv[1]
    tol = float(sys.argv[sys.argv.index("--tol") + 1]) if "--tol" in sys.argv else 1e-3
    out_csv = os.path.join(out_dir, "xs_exactness.csv" if tol == 1e-3 else f"xs_exactness_tol{tol:.0e}.csv")
    assert not os.path.exists(out_csv), "refusing to overwrite"
    paths = generate(out_dir)
    tasks = [dict(model=model, instance=instance, path=path, no_overlap=no_overlap, tol=tol)
             for (model, instance), path in sorted(paths.items()) for no_overlap in (True, False)]
    print(f"{len(paths)} instances, {len(tasks)} configurations, {2 * len(tasks)} solves", flush=True)
    with mp.get_context("spawn").Pool(4) as pool:
        rows = []
        for row in pool.imap_unordered(solve, tasks):
            rows.append(row)
            rel = row["rel_diff"]
            print(f"{row['instance']:32} {row['model']:6} no_overlap={str(row['no_overlap']):5} | direct {row['direct_status']:10} "
                  f"| benders {row['benders_status']:10} {row['benders_iterations']:>4} iters "
                  f"| rel diff {'%.2e' % rel if rel is not None else '-'}", flush=True)
    rows.sort(key=lambda r: (r["model"], r["instance"], not r["no_overlap"]))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote", out_csv)


if __name__ == "__main__":
    main()
