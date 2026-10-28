"""Benders' best schedule at 600 s (core5 single, F1_system_scale_M seed 1), in the same style and scale as
the optimal and rule-based figures, plus checks against the instance file.

Run from the repository root (outputs are never overwritten):
  python3 analysis/plot_benders_schedule.py results/plots_slides/F1_system_scale_M_benders_single.png results/plots_slides/F1_system_scale_M_benders_single_notitle.png
"""
import json, os, sys
from collections import Counter
sys.path.insert(0, os.getcwd())
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import monolithic_milp
import solution_payloads as sp
from validate_solution import check_schedule, scenario_path

OUT_TITLED, OUT_PLAIN = sys.argv[1], sys.argv[2]
recs = {r["method"]: r for r in map(json.loads, open("results/core5_single.jsonl"))
        if r["family"] == "F1_system_scale_M" and int(r["seed"]) == 1}
bend, opt = recs["benders"], recs["direct"]
data = monolithic_milp.load_scenario(scenario_path("single", "F1_system_scale_M_seed_001"))
schedule = [tuple(s) for s in bend["schedule"]]

# 3a. schedule rules (incl. same-asset no-overlap) straight from the instance data
viol = {k: v for k, v in check_schedule(data, schedule).items() if v}
print("schedule rule violations:", viol or "none")

model, v = monolithic_milp.build_model(data)
model.Params.OutputFlag = 0; model.Params.Threads = 2
chosen = set(schedule)
for key, var in v["x"].items():
    var.LB = var.UB = 1.0 if key in chosen else 0.0
model.optimize()
assert model.Status == 2, model.Status
print(f"objective: logged {bend['objective']:,.3f} | re-solved {model.ObjVal:,.3f}")

result = dict(model="single_commodity", scenario="F1_system_scale_M", family="F1_system_scale_M", seed=1,
              status="OPTIMAL", objective=model.ObjVal, mip_gap=None, best_bound=None, runtime_seconds=None,
              strict_capacity=False, time_limit_seconds=None, threads_per_solve=2, scenario_path=None)
payload = sp.build_solution_payload(result, data, v)
inv = payload["solution"]["zone_inventory"]

# 3b/3c. stock within zone min/max, no overflow, throughput floor met, every day
tol = 1e-6
bad_stock = [(z, p["day"], p["inventory_tonnes"]) for z, pts in inv.items() for p in pts
             if not data["Z_min"][z] - tol <= p["inventory_tonnes"] <= data["Z_max"][z] + tol]
overflow = sum(x.X for x in v["o"].values())
mu = {d: v["mu"][d].X for d in data["D"]}
bad_floor = [(d, mu[d], data["mu_min"].get(d, 0.0)) for d in data["D"] if mu[d] < data["mu_min"].get(d, 0.0) - tol]
print("stock outside [Zmin, Zmax]:", len(bad_stock), bad_stock[:3], "| total overflow", round(overflow, 6))
print("days below throughput floor:", len(bad_floor), bad_floor[:3])

# 4. report numbers
shipped = sum(mu.values())
opt_shipped = 19_644_203.746   # optimal schedule's flows re-solved to optimality (F1_system_scale_M_optimal_single.png)
print(f"shipped {shipped:,.0f} t | vs logged optimum {opt['objective']:,.0f}: {opt['objective']-shipped:,.0f} t "
      f"({100*(opt['objective']-shipped)/opt['objective']:.2f}%) | vs re-solved optimum {opt_shipped:,.0f}: "
      f"{opt_shipped-shipped:,.0f} t ({100*(opt_shipped-shipped)/opt_shipped:.2f}%)")
print("mode mix benders:", dict(Counter(k for _, k, _ in schedule)), "| optimal:", dict(Counter(s[1] for s in opt["schedule"])))
peak = max(((z, p["day"], p["inventory_tonnes"]) for z, pts in inv.items() for p in pts), key=lambda t: t[2])
print("peak stock:", peak, "| zone max", data["Z_max"][peak[0]])
for z, pts in inv.items():
    hi = [p["day"] for p in pts if p["inventory_tonnes"] > 0.8 * peak[2]]
    if hi: print("  ", z, "above 80% of peak on days", hi)
print("per-zone peaks:", {z: round(max(p["inventory_tonnes"] for p in pts)) for z, pts in inv.items()})
starts = sorted(s for _, _, s in schedule)
print("benders start days by fortnight:", Counter(s // 15 for s in starts),
      "| optimal:", Counter(s[2] // 15 for s in opt["schedule"]))
print("benders jobs active per day (max):", max(sum(1 for m,k,s in schedule if s <= d < s + data['p'][(m,k)]) for d in data["D"]),
      "| optimal:", max(sum(1 for m,k,s in opt['schedule'] if s <= d < s + data['p'][(m,k)]) for d in data["D"]))

# 2. figure: same rows, size, scale as the optimal / rule-based figures
rb = json.load(open("payload_rule_based_baseline/rule_based_baseline_evaluated_single_v3/F1_system_scale_M/seed_001/"
                    "F1_system_scale_M_seed_001_rule_based_evaluated.json"))
rb_assets = sorted({x["asset_id"] for x in rb["maintenance_schedule"].values()})
rb_top = 1.15 * max(t for days in rb["stockyard_inventory_tonnes"].values() for t in days.values())
payload["sets"]["assets"] = sorted({item["asset"] for item in payload["solution"]["maintenance_schedule"]})
print("rows match the other two figures:", payload["sets"]["assets"] == rb_assets)

def draw(out, title):
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
    if title:
        fig.suptitle(title, color=sp.INK_PRIMARY, fontsize=13, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()
    assert not os.path.exists(out), "refusing to overwrite"
    fig.savefig(out, dpi=150, facecolor=sp.SURFACE)
    plt.close(fig)
    print("wrote", out)

draw(OUT_TITLED, "Benders (best schedule at 600 s)")
draw(OUT_PLAIN, None)
