"""Optimal (direct MILP, no-overlap) schedule for F1_system_scale_M single, in the rule-based plot style:
maintenance schedule on top, stockyard inventory below, no figure title.

Run from the repository root (outputs are never overwritten):
  python3 analysis/plot_optimal_schedule.py results/plots_slides/F1_system_scale_M_optimal_single.png
"""
import json, os, sys
sys.path.insert(0, os.getcwd())
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import monolithic_milp
import solution_payloads as sp
from validate_solution import scenario_path

OUT = sys.argv[1]
rec = next(r for r in map(json.loads, open("results/core5_single.jsonl"))
           if r["method"] == "direct" and r["family"] == "F1_system_scale_M" and int(r["seed"]) == 1)
data = monolithic_milp.load_scenario(scenario_path("single", "F1_system_scale_M_seed_001"))
model, variables = monolithic_milp.build_model(data)   # default schedule options: no-overlap on, no cap
model.Params.OutputFlag = 0
model.Params.Threads = 2
chosen = {tuple(s) for s in rec["schedule"]}
for key, var in variables["x"].items():
    var.LB = var.UB = 1.0 if key in chosen else 0.0
model.optimize()
print("schedule from core5_single.jsonl:", len(chosen), "jobs | reported", f"{rec['objective']:,.3f}",
      "| re-solved", f"{model.ObjVal:,.3f}")

result = dict(model="single_commodity", scenario="F1_system_scale_M", family="F1_system_scale_M", seed=1,
              status="OPTIMAL", objective=model.ObjVal, mip_gap=None, best_bound=None, runtime_seconds=None,
              strict_capacity=False, time_limit_seconds=None, threads_per_solve=2, scenario_path=None)
payload = sp.build_solution_payload(result, data, variables)
# Match the rule-based figure: rows only for assets with maintenance, and the same inventory scale.
rb = json.load(open("payload_rule_based_baseline/rule_based_baseline_evaluated_single_v3/F1_system_scale_M/seed_001/"
                    "F1_system_scale_M_seed_001_rule_based_evaluated.json"))
rb_assets = sorted({v["asset_id"] for v in rb["maintenance_schedule"].values()})
payload["sets"]["assets"] = sorted({item["asset"] for item in payload["solution"]["maintenance_schedule"]})
print("rows match rule-based figure:", payload["sets"]["assets"] == rb_assets)
rb_top = 1.15 * max(v for days in rb["stockyard_inventory_tonnes"].values() for v in days.values())
fig, (ax_gantt, ax_inv) = plt.subplots(2, 1, figsize=(11, 7), facecolor=sp.SURFACE)
sp._plot_maintenance_gantt(ax_gantt, payload, mpatches)
sp._plot_inventory(ax_inv, payload)
drawn_top = ax_inv.get_ylim()[1]
zone_max = payload["parameters"]["zone_max_inventory_tonnes"]
colors = dict(zip(sorted(zone_max), sp.CATEGORICAL))
for zone, zmax in zone_max.items():
    if drawn_top < zmax <= rb_top:
        ax_inv.axhline(zmax, color=colors[zone], linewidth=0.8, linestyle="--", alpha=0.5)
ax_inv.set_ylim(0, rb_top)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
assert not os.path.exists(OUT), "refusing to overwrite"
fig.savefig(OUT, dpi=150, facecolor=sp.SURFACE)
print("wrote", OUT)
