"""Solve dissertation scenarios with monolithic_milp's model and plot the
resulting maintenance schedule (Gantt, one row per asset), any fixed special
activities, and stockyard inventory profile (one line per zone) for each,
saving each as a PNG."""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from monolithic_milp import build_model, load_scenario

SCENARIO_DIR = "synthetic_data_generator/singlecommodity_scenarios"
OUTPUT_DIR = "synthetic_data_generator/scenario_plots"

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#008300", "#e34948"]

MODE_LABELS = {
    "full_shutdown": "Full shutdown",
    "partial_30": "Partial (30%)",
    "partial_70": "Partial (70%)",
}
MODE_COLORS = dict(zip(MODE_LABELS, CATEGORICAL))


def default_scenario_paths():
    """seed_001 across every family/severity combination -- a fast, representative sample."""
    families = sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(SCENARIO_DIR, "*")) if os.path.isdir(p)
    )
    return [os.path.join(SCENARIO_DIR, fam, "seed_001", f"{fam}_seed_001.json") for fam in families]


def all_scenario_paths():
    return sorted(glob.glob(os.path.join(SCENARIO_DIR, "*", "seed_*", "*.json")))


def extract_special_activities(path):
    """Special activities are fixed data (scheduled_start, not a decision
    variable) -- read straight from the raw scenario JSON, which load_scenario
    doesn't otherwise expose since constraint (3) only needs the derived h."""
    with open(path) as f:
        m = json.load(f)["model_ready_data"]
    starts = m.get("special_activity_scheduled_start", {})
    durations = m.get("special_activity_duration_days", {})
    assets = m.get("special_activity_required_assets", {})
    zones = m.get("special_activity_required_zone", {})
    products = m.get("special_activity_product", {})
    return [
        dict(
            activity=h, start=starts[h], duration=durations[h],
            assets=assets.get(h, []), zone=zones.get(h), product=products.get(h),
        )
        for h in m.get("H", [])
    ]


def extract_maintenance_schedule(data, variables):
    op_asset = {op: asset for asset, ops in data["M_a"].items() for op in ops}
    p = data["p"]
    schedule = []
    for (mm, k, d), var in variables["x"].items():
        if var.X > 0.5:
            schedule.append(dict(asset=op_asset[mm], mode=k, start=d, duration=p[(mm, k)]))
    return schedule


def extract_inventory(data, variables):
    z = variables["z"]
    return {pz: [z[pz, d].X for d in data["D"]] for pz in data["P_"]}


def style_axes(ax, keep_spines):
    ax.set_facecolor(SURFACE)
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep_spines)
        if side in keep_spines:
            spine.set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=8)


def plot_maintenance_gantt(ax, data, schedule):
    assets = sorted(data["A"])
    y_pos = {a: i for i, a in enumerate(assets)}
    for item in schedule:
        ax.broken_barh(
            [(item["start"], item["duration"])],
            (y_pos[item["asset"]] - 0.35, 0.7),
            facecolors=MODE_COLORS[item["mode"]],
            edgecolors=SURFACE,
            linewidth=1,
        )
    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels(assets, color=INK_SECONDARY, fontsize=9)
    ax.set_ylim(-0.5, len(assets) - 0.5)
    ax.invert_yaxis()
    ax.set_xlim(data["D"][0], data["D"][-1] + 1)
    ax.set_xlabel("Day", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Maintenance schedule", color=INK_PRIMARY, loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    style_axes(ax, keep_spines={"bottom"})
    handles = [mpatches.Patch(color=c, label=MODE_LABELS[k]) for k, c in MODE_COLORS.items()]
    legend = ax.legend(handles=handles, loc="upper right", fontsize=8, labelcolor=INK_SECONDARY, ncol=3)
    legend.get_frame().set_facecolor(SURFACE)
    legend.get_frame().set_edgecolor(BASELINE)
    legend.get_frame().set_linewidth(0.6)


SPECIAL_ACTIVITY_COLOR = CATEGORICAL[5]  # distinct from the three mode colors


def plot_special_activities(ax, data, activities):
    ax.set_facecolor(SURFACE)
    if not activities:
        ax.axis("off")
        ax.text(0.5, 0.5, "No special activities in this scenario", ha="center", va="center",
                transform=ax.transAxes, color=INK_MUTED, fontsize=9.5)
        return

    names = [a["activity"] for a in activities]
    y_pos = {n: i for i, n in enumerate(names)}
    for a in activities:
        ax.broken_barh(
            [(a["start"], a["duration"])],
            (y_pos[a["activity"]] - 0.35, 0.7),
            facecolors=SPECIAL_ACTIVITY_COLOR, edgecolors=SURFACE, linewidth=1,
        )
        label = a["product"] or ""
        if a["zone"]:
            label += f" @ {a['zone']}"
        if a["assets"]:
            label += f"  ({', '.join(a['assets'])})"
        ax.text(a["start"] + a["duration"] + 0.8, y_pos[a["activity"]], label,
                va="center", fontsize=7.5, color=INK_SECONDARY)

    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels(names, color=INK_SECONDARY, fontsize=8.5)
    ax.set_ylim(-0.5, len(names) - 0.5)
    ax.invert_yaxis()
    ax.set_xlim(data["D"][0], data["D"][-1] + 1)
    ax.set_xlabel("Day", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Special activities (fixed windows)", color=INK_PRIMARY, loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    style_axes(ax, keep_spines={"bottom"})


def plot_inventory(ax, data, inventory):
    zones = sorted(inventory)
    colors = dict(zip(zones, CATEGORICAL))
    observed_max = max(v for vals in inventory.values() for v in vals)
    ylim_top = observed_max * 1.15

    # Only draw a zone's capacity ceiling if inventory gets close enough to it
    # to be a meaningful reference -- otherwise it just flattens the real data.
    relevant_max = [data["Z_max"][pz] for pz in zones if data["Z_max"][pz] <= observed_max * 1.5]
    if relevant_max:
        ylim_top = max(ylim_top, max(relevant_max) * 1.02)

    for pz in zones:
        ax.plot(data["D"], inventory[pz], color=colors[pz], linewidth=2, label=pz)
        if data["Z_max"][pz] <= ylim_top:
            ax.axhline(data["Z_max"][pz], color=colors[pz], linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_xlim(data["D"][0], data["D"][-1])
    ax.set_ylim(0, ylim_top)
    ax.set_xlabel("Day", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("Tonnes", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Stockyard inventory (dashed = zone max)", color=INK_PRIMARY, loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    style_axes(ax, keep_spines={"bottom", "left"})
    ax.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=INK_SECONDARY, ncol=len(zones))


def plot_scenario(name, data, variables, activities, out_path):
    schedule = extract_maintenance_schedule(data, variables)
    inventory = extract_inventory(data, variables)

    if activities:
        fig, (ax_gantt, ax_special, ax_inv) = plt.subplots(
            3, 1, figsize=(11, 10), facecolor=SURFACE, gridspec_kw={"height_ratios": [1, 0.45, 1]}
        )
        plot_special_activities(ax_special, data, activities)
    else:
        fig, (ax_gantt, ax_inv) = plt.subplots(2, 1, figsize=(11, 8), facecolor=SURFACE)

    fig.suptitle(name.replace("_", " ").title(), color=INK_PRIMARY, fontsize=13, fontweight="bold")
    plot_maintenance_gantt(ax_gantt, data, schedule)
    plot_inventory(ax_inv, data, inventory)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="plot the full 120-scenario battery instead of the seed_001 sample")
    parser.add_argument("--limit", type=int, default=None, help="cap the number of scenarios plotted")
    args = parser.parse_args()

    paths = all_scenario_paths() if args.all else default_scenario_paths()
    if args.limit:
        paths = paths[: args.limit]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        data = load_scenario(path)
        model, variables = build_model(data)
        model.Params.OutputFlag = 0
        model.optimize()

        if model.SolCount == 0:
            print(f"{name}: no solution found (status {model.Status}) -- skipping plot")
            continue

        activities = extract_special_activities(path)
        out_path = os.path.join(OUTPUT_DIR, f"{name}.png")
        plot_scenario(name, data, variables, activities, out_path)
        print(f"{name}: saved {out_path}")


if __name__ == "__main__":
    main()
