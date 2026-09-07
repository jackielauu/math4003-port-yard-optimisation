"""Solve every generated dissertation scenario with multicommodity_milp's model
and plot the resulting maintenance schedule (Gantt, one row per asset) and
stockyard inventory profile (one line per zone, summed across products) for
each, saving each as a PNG."""

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from multicommodity_milp import build_model, load_scenario

SCENARIO_DIR = "synthetic_data_generator/generated_dissertation_scenarios"
OUTPUT_DIR = "synthetic_data_generator/scenario_plots_multicommodity"

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


def scenario_paths():
    paths = sorted(glob.glob(os.path.join(SCENARIO_DIR, "*.json")))
    return [p for p in paths if os.path.basename(p) != "manifest.json"]


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
    return {pz: [sum(z[pz, g, d].X for g in data["PRODUCTS"]) for d in data["D"]] for pz in data["P_"]}


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
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8, labelcolor=INK_SECONDARY, ncol=3)


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


def plot_scenario(name, data, variables, out_path):
    schedule = extract_maintenance_schedule(data, variables)
    inventory = extract_inventory(data, variables)

    fig, (ax_gantt, ax_inv) = plt.subplots(2, 1, figsize=(11, 8), facecolor=SURFACE)
    fig.suptitle(name.replace("_", " ").title(), color=INK_PRIMARY, fontsize=13, fontweight="bold")
    plot_maintenance_gantt(ax_gantt, data, schedule)
    plot_inventory(ax_inv, data, inventory)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for path in scenario_paths():
        name = os.path.splitext(os.path.basename(path))[0]
        data = load_scenario(path)
        model, variables = build_model(data)
        model.Params.OutputFlag = 0
        model.optimize()

        if model.SolCount == 0:
            print(f"{name}: no solution found (status {model.Status}) -- skipping plot")
            continue

        out_path = os.path.join(OUTPUT_DIR, f"{name}.png")
        plot_scenario(name, data, variables, out_path)
        print(f"{name}: saved {out_path}")


if __name__ == "__main__":
    main()
