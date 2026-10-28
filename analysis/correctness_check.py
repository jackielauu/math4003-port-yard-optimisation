"""B6 correctness check: old-data F1_L seed 1, single and multi, Benders vs direct MILP (old-branch code).

Run from the repository root (outputs are never overwritten):
  git worktree add --detach /tmp/olddata enhance-data-generator
  python3 analysis/correctness_check.py /tmp/olddata results/correctness_check_olddata_F1_L.json
"""
import json, os, sys, time
OLD = sys.argv[1]; OUT = os.path.abspath(sys.argv[2])
assert not os.path.exists(OUT), "refusing to overwrite"
os.chdir(OLD); sys.path.insert(0, OLD)
import monolithic_milp, multicommodity_milp, benders_monolithic, benders_multicommodity
from gurobipy import GRB
rows = []
for model_name, milp, benders in (("single", monolithic_milp, benders_monolithic), ("multi", multicommodity_milp, benders_multicommodity)):
    kind = "singlecommodity" if model_name == "single" else "multicommodity"
    path = f"synthetic_data_generator/{kind}_scenarios/F1_system_scale_L/seed_001/F1_system_scale_L_seed_001.json"
    data = milp.load_scenario(path)
    m, _ = milp.build_model(data)
    m.Params.OutputFlag = 0; m.Params.MIPGap = 1e-9; m.Params.Threads = 2
    t = time.perf_counter(); m.optimize(); dt = time.perf_counter() - t
    direct = dict(status="OPTIMAL" if m.Status == GRB.OPTIMAL else str(m.Status), objective=m.ObjVal, bound=m.ObjBound, seconds=dt)
    t = time.perf_counter(); b = benders.solve_benders(milp.load_scenario(path), max_iters=500, tol=1e-6); bt = time.perf_counter() - t
    keep = {k: v for k, v in b.items() if isinstance(v, (int, float, str)) or v is None}
    row = dict(model=model_name, data="enhance-data-generator @ 3ef0440 (earlier dataset)", instance="F1_system_scale_L_seed_001",
               n_ops=len(data["M"]), direct=direct, benders=dict(keep, seconds=bt))
    rows.append(row)
    print(model_name, "ops", len(data["M"]), "| direct", direct["status"], f"{direct['objective']:,.4f}", f"{dt:.2f}s",
          "| benders", b["status"], b.get("iterations"), "iters", f"{b['best_objective']:,.4f}" if b.get("best_objective") is not None else None,
          f"{bt:.2f}s", "| diff", None if b.get("best_objective") is None else f"{b['best_objective'] - direct['objective']:.6f}")
json.dump(rows, open(OUT, "w"), indent=1)
print("wrote", OUT)
