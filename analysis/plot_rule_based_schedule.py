"""Ponpot's rule-based F1_system_scale_M single schedule, redrawn to match the optimal figure:
maintenance schedule on top, stockyard inventory below, no figure title, same inventory scale.

Run from the repository root (outputs are never overwritten):
  python3 analysis/plot_rule_based_schedule.py results/plots_slides/F1_system_scale_M_rule_based_single.png
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
rb = json.load(open("payload_rule_based_baseline/rule_based_baseline_evaluated_single_v3/F1_system_scale_M/seed_001/"
                    "F1_system_scale_M_seed_001_rule_based_evaluated.json"))
data = monolithic_milp.load_scenario(scenario_path("single", "F1_system_scale_M_seed_001"))
sched = [dict(asset=v["asset_id"], mode=v["mode"], start_day=int(v["start_day"]), duration_days=int(v["duration_days"]))
         for v in rb["maintenance_schedule"].values()]
inv = {z: [dict(day=int(d), inventory_tonnes=t) for d, t in sorted(days.items(), key=lambda kv: int(kv[0]))]
       for z, days in rb["stockyard_inventory_tonnes"].items()}
payload = dict(sets=dict(days=list(data["D"]), assets=sorted({s["asset"] for s in sched})),
               parameters=dict(zone_max_inventory_tonnes=dict(data["Z_max"])),
               solution=dict(maintenance_schedule=sched, zone_inventory=inv))
top = 1.15 * max(p["inventory_tonnes"] for pts in inv.values() for p in pts)
fig, (ax_gantt, ax_inv) = plt.subplots(2, 1, figsize=(11, 7), facecolor=sp.SURFACE)
sp._plot_maintenance_gantt(ax_gantt, payload, mpatches)
sp._plot_inventory(ax_inv, payload)
ax_inv.set_ylim(0, top)
fig.tight_layout()
assert not os.path.exists(OUT), "refusing to overwrite"
fig.savefig(OUT, dpi=150, facecolor=sp.SURFACE)
print("jobs", len(sched), "| inventory axis top", round(top), "| wrote", OUT)
