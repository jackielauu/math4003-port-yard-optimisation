"""Rule-based, optimal and Benders schedules for F1_system_scale_M single, drawn by one function so the
panels line up: legend above the schedule panel, no figure title, same rows, size, margins and stock scale.

Run from the repository root (outputs are never overwritten):
  python3 analysis/plot_three_schedules.py results/plots_slides
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

OUT_DIR = sys.argv[1]
data = monolithic_milp.load_scenario(scenario_path("single", "F1_system_scale_M_seed_001"))
rb = json.load(open("payload_rule_based_baseline/rule_based_baseline_evaluated_single_v3/F1_system_scale_M/seed_001/"
                    "F1_system_scale_M_seed_001_rule_based_evaluated.json"))
recs = {r["method"]: r for r in map(json.loads, open("results/core5_single.jsonl"))
        if r["family"] == "F1_system_scale_M" and int(r["seed"]) == 1}
TOP = 1.15 * max(t for days in rb["stockyard_inventory_tonnes"].values() for t in days.values())


def rule_based_payload():
    sched = [dict(asset=v["asset_id"], mode=v["mode"], start_day=int(v["start_day"]), duration_days=int(v["duration_days"]))
             for v in rb["maintenance_schedule"].values()]
    inv = {z: [dict(day=int(d), inventory_tonnes=t) for d, t in sorted(days.items(), key=lambda kv: int(kv[0]))]
           for z, days in rb["stockyard_inventory_tonnes"].items()}
    return dict(sets=dict(days=list(data["D"]), assets=sorted({s["asset"] for s in sched})),
                parameters=dict(zone_max_inventory_tonnes=dict(data["Z_max"])),
                solution=dict(maintenance_schedule=sched, zone_inventory=inv))


def fixed_schedule_payload(rec):
    model, v = monolithic_milp.build_model(data)
    model.Params.OutputFlag = 0; model.Params.Threads = 2
    chosen = {tuple(s) for s in rec["schedule"]}
    for key, var in v["x"].items():
        var.LB = var.UB = 1.0 if key in chosen else 0.0
    model.optimize()
    print(f"  {rec['method']}: objective {model.ObjVal:,.3f}")
    result = dict(model="single_commodity", scenario="F1_system_scale_M", family="F1_system_scale_M", seed=1,
                  status="OPTIMAL", objective=model.ObjVal, mip_gap=None, best_bound=None, runtime_seconds=None,
                  strict_capacity=False, time_limit_seconds=None, threads_per_solve=2, scenario_path=None)
    p = sp.build_solution_payload(result, data, v)
    p["sets"]["assets"] = sorted({item["asset"] for item in p["solution"]["maintenance_schedule"]})
    return p


def draw(payload, out):
    fig, (ax_gantt, ax_inv) = plt.subplots(2, 1, figsize=(11, 7), facecolor=sp.SURFACE)
    sp._plot_maintenance_gantt(ax_gantt, payload, mpatches)
    # Legend above the schedule panel, right-aligned on the title line, so it never covers a bar.
    leg = ax_gantt.get_legend()
    if leg is not None:
        handles = [mpatches.Patch(color=sp.MODE_COLORS[m], label=sp.MODE_LABELS[m]) for m in sp.MODE_LABELS
                   if any(i["mode"] == m for i in payload["solution"]["maintenance_schedule"])]
        leg.remove()
        ax_gantt.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8,
                        labelcolor=sp.INK_SECONDARY, ncol=3, borderaxespad=0.2)
    sp._plot_inventory(ax_inv, payload)
    drawn_top = ax_inv.get_ylim()[1]
    zone_max = payload["parameters"]["zone_max_inventory_tonnes"]
    colors = dict(zip(sorted(zone_max), sp.CATEGORICAL))
    for zone, zmax in zone_max.items():
        if drawn_top < zmax <= TOP:
            ax_inv.axhline(zmax, color=colors[zone], linewidth=0.8, linestyle="--", alpha=0.5)
    ax_inv.set_ylim(0, TOP)
    # Fixed margins (not tight_layout) so all three figures have identical panel positions.
    fig.subplots_adjust(left=0.075, right=0.98, top=0.95, bottom=0.07, hspace=0.32)
    assert not os.path.exists(out), f"refusing to overwrite {out}"
    fig.savefig(out, dpi=150, facecolor=sp.SURFACE)
    pos = [tuple(round(x, 4) for x in ax.get_position().bounds) for ax in (ax_gantt, ax_inv)]
    plt.close(fig)
    print("  wrote", out, "| panel positions", pos)
    return pos


positions = {}
for name, make in (("rule_based", rule_based_payload),
                   ("optimal", lambda: fixed_schedule_payload(recs["direct"])),
                   ("benders", lambda: fixed_schedule_payload(recs["benders"]))):
    print(name)
    positions[name] = draw(make(), os.path.join(OUT_DIR, f"F1_system_scale_M_{name}_single_legendtop.png"))
print("identical panel positions:", len(set(map(tuple, positions.values()))) == 1)
