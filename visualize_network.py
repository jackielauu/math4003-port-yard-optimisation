"""Render the port-yard network topology (assets, stockyard zones, arcs) for
each family/severity combination. Topology is constant across seeds within a
family/severity (only capacities/maintenance/etc. vary by seed), so one
diagram per combination is enough -- uses seed_001 as the representative."""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SCENARIO_DIR = "synthetic_data_generator/singlecommodity_scenarios"
OUTPUT_DIR = "synthetic_data_generator/network_plots"

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
BASELINE = "#c3c2b7"

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#008300", "#e34948"]

TYPE_COLOR = {
    "source": INK_MUTED, "TU": CATEGORICAL[0], "ST": CATEGORICAL[1],
    "Z": CATEGORICAL[2], "RC": CATEGORICAL[3], "SL": CATEGORICAL[4], "sink": INK_MUTED,
}
TYPE_LABEL = {
    "source": "Source", "TU": "Train unloader", "ST": "Stacker",
    "Z": "Stockyard zone", "RC": "Reclaimer", "SL": "Ship loader", "sink": "Sink",
}
TYPE_ORDER = ["source", "TU", "ST", "Z", "RC", "SL", "sink"]


def default_scenario_paths():
    families = sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(SCENARIO_DIR, "*")) if os.path.isdir(p)
    )
    return [os.path.join(SCENARIO_DIR, fam, "seed_001", f"{fam}_seed_001.json") for fam in families]


def node_type(node, zones):
    if node in zones:
        return "Z"
    for prefix in ("TU", "ST", "RC", "SL"):
        if node.startswith(prefix):
            return prefix
    return "source" if "SOURCE" in node else "sink"


def compute_layers(nodes, edges):
    """Longest path from any source, via repeated relaxation -- graphs here
    are tiny DAGs (<30 nodes), so a topological sort isn't worth the code."""
    successors = {n: [] for n in nodes}
    predecessors = {n: [] for n in nodes}
    for i, j in edges:
        successors[i].append(j)
        predecessors[j].append(i)

    layer = {n: 0 for n in nodes if not predecessors[n]}
    changed = True
    while changed:
        changed = False
        for n in nodes:
            for j in successors[n]:
                candidate = layer.get(n, 0) + 1
                if candidate > layer.get(j, -1):
                    layer[j] = candidate
                    changed = True
    return layer


def plot_network(path, out_path, title):
    m = json.load(open(path))["model_ready_data"]
    nodes = m["N"]
    zones = set(m["Z"])
    arc_ids = list(m["arc_from"].keys())
    edges = [(m["arc_from"][a], m["arc_to"][a]) for a in arc_ids]
    cap_by_edge = {(m["arc_from"][a], m["arc_to"][a]): m["arc_capacity_tpd"][a] for a in arc_ids}

    layer = compute_layers(nodes, edges)
    by_layer = {}
    for n in nodes:
        by_layer.setdefault(layer[n], []).append(n)
    for ns in by_layer.values():
        ns.sort()

    pos = {}
    for l, ns in by_layer.items():
        for i, n in enumerate(ns):
            pos[n] = (l, (len(ns) - 1) / 2 - i)

    max_layer = max(by_layer)
    max_rows = max(len(v) for v in by_layer.values())

    fig, ax = plt.subplots(
        figsize=(2.3 * (max_layer + 1) + 1.5, 0.85 * max_rows + 2), facecolor=SURFACE
    )
    ax.set_facecolor(SURFACE)

    caps = list(cap_by_edge.values())
    cmin, cmax = min(caps), max(caps)

    def linewidth(cap):
        return 1.6 if cmax == cmin else 1.0 + 3.0 * (cap - cmin) / (cmax - cmin)

    for (i, j) in edges:
        x1, y1 = pos[i]
        x2, y2 = pos[j]
        ax.add_patch(FancyArrowPatch(
            (x1 + 0.3, y1), (x2 - 0.3, y2),
            arrowstyle="-|>", mutation_scale=9, color=BASELINE,
            linewidth=linewidth(cap_by_edge[(i, j)]),
            connectionstyle="arc3,rad=0.06", zorder=1,
        ))

    for n, (x, y) in pos.items():
        color = TYPE_COLOR[node_type(n, zones)]
        ax.add_patch(FancyBboxPatch(
            (x - 0.3, y - 0.19), 0.6, 0.38,
            boxstyle="round,pad=0.02,rounding_size=0.07",
            linewidth=1.5, edgecolor=color, facecolor=SURFACE, zorder=2,
        ))
        ax.text(x, y, n, ha="center", va="center", fontsize=8.5, color=INK_PRIMARY, zorder=3)

    ax.set_xlim(-0.7, max_layer + 0.7)
    ax.set_ylim(-max_rows / 2 - 0.7, max_rows / 2 + 0.7)
    ax.set_aspect("equal")
    ax.axis("off")

    handles = [
        Line2D([0], [0], marker="s", linestyle="", markerfacecolor=SURFACE,
               markeredgecolor=TYPE_COLOR[t], markeredgewidth=1.6, markersize=10, label=TYPE_LABEL[t])
        for t in TYPE_ORDER
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=len(TYPE_ORDER),
              frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)
    ax.set_title(f"{title} -- arrow width ∝ arc capacity (tpd)",
                 color=INK_PRIMARY, fontsize=12, fontweight="bold", loc="left", pad=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="cap the number of diagrams generated")
    args = parser.parse_args()

    paths = default_scenario_paths()
    if args.limit:
        paths = paths[: args.limit]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for path in paths:
        family_severity = os.path.basename(os.path.dirname(os.path.dirname(path)))
        out_path = os.path.join(OUTPUT_DIR, f"{family_severity}.png")
        plot_network(path, out_path, family_severity.replace("_", " ").title())
        print(f"{family_severity}: saved {out_path}")


if __name__ == "__main__":
    main()
