"""LP-relaxation and root-node (NodeLimit=0) bounds vs the proved optimum, core5, no-overlap on.

Run from the repository root (outputs are never overwritten):
  python3 analysis/root_gap.py results/root_relaxation_gap.csv
"""
import csv, json, sys, time
sys.path.insert(0, ".")
import monolithic_milp, multicommodity_milp
from validate_solution import scenario_path
OUT = sys.argv[1]
rows = []
for model_name, module in (("single", monolithic_milp), ("multi", multicommodity_milp)):
    opt = {r["family"]: r for r in map(json.loads, open(f"results/core5_{model_name}.jsonl")) if r["method"] == "direct"}
    for fam in ("F1_system_scale_L", "F1_system_scale_M", "F1_system_scale_H", "F2_mode_tradeoff_H", "F4_operational_conflict_H"):
        data = module.load_scenario(scenario_path(model_name, f"{fam}_seed_001"))
        m, _ = module.build_model(data)
        m.Params.OutputFlag = 0; m.Params.Threads = 2
        lp = m.relax(); lp.Params.OutputFlag = 0; lp.Params.Threads = 2
        t = time.perf_counter(); lp.optimize(); lp_s = time.perf_counter() - t
        m.Params.NodeLimit = 0
        t = time.perf_counter(); m.optimize(); root_s = time.perf_counter() - t
        best = opt[fam]["objective"]; proved_bound = opt[fam]["bound"]
        row = dict(model=model_name, instance=f"{fam}_seed_001", optimum=best, optimum_proved_bound=proved_bound,
                   lp_bound=lp.ObjVal, lp_gap_pct=100 * (lp.ObjVal - best) / best, lp_seconds=round(lp_s, 2),
                   root_bound=m.ObjBound, root_gap_pct=100 * (m.ObjBound - best) / best,
                   root_incumbent=m.ObjVal if m.SolCount else None, root_seconds=round(root_s, 2))
        rows.append(row)
        print(f"{model_name:6} {fam:26} opt {best:,.0f} | LP bound {lp.ObjVal:,.0f} ({row['lp_gap_pct']:.2f}%) | "
              f"root bound {m.ObjBound:,.0f} ({row['root_gap_pct']:.2f}%) | root incumbent "
              f"{'%s' % (f'{m.ObjVal:,.0f}' if m.SolCount else None)}", flush=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("wrote", OUT)
