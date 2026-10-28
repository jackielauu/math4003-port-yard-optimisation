"""Gap-versus-time convergence figures from compare_benders_vs_gurobi.py records.

Both methods use the same gap definition, (upper bound - best feasible) / |best feasible|
by default, or / |upper bound| with --gap-denominator bound (matching the results tables),
so the curves are directly comparable: Gurobi's from its incumbent/bound timeline,
Benders' from its best upper bound (master) and best lower bound (subproblem) after each
iteration. One panel per instance; feasibility-cut iterations are marked on the Benders
curve.

  python3 plot_convergence.py results/core5_multi.jsonl --instances F1_system_scale_H:1
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
GUROBI, BENDERS = "#2a78d6", "#eb6834"


def gap(ub, lb, denominator="best"):
    """denominator="best": (bound - best) / |best|, as Gurobi reports it;
    denominator="bound": (bound - best) / |bound|, as run_benders reports it (the results tables)."""
    if ub is None or lb is None:
        return None
    base = abs(lb) if denominator == "best" else abs(ub)
    if base < 1e-9:
        return None
    return 100.0 * max(ub - lb, 0.0) / base


def gurobi_curve(r, denominator="best"):
    pts = [(t, gap(bnd, inc, denominator)) for t, inc, bnd in (r.get("timeline") or [])]
    return [(t, g) for t, g in pts if g is not None]


def benders_curve(r, denominator="best"):
    pts, feas = [], []
    for it, kind, t, ub, lb, best_ub, best_lb in (r.get("history") or []):
        g = gap(best_ub, best_lb, denominator)
        if g is not None:
            pts.append((t, g))
            if kind == "f":
                feas.append((t, g))
    return pts, feas


def load(paths):
    by_instance = {}
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            by_instance.setdefault((r["model"], r["family"], int(r["seed"])), {})[r["method"]] = r
    return by_instance


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records", nargs="+")
    ap.add_argument("--instances", default=None, help="family:seed,... (default: all in the records)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--gap-denominator", choices=("best", "bound"), default="best",
                    help="divide the gap by the best feasible objective (Gurobi's convention, default) "
                         "or by the bound (the convention in the results tables)")
    args = ap.parse_args()

    data = load(args.records)
    # Families in name order, levels by size (L, M, H) rather than alphabetically.
    level = {"L": 0, "M": 1, "H": 2}
    keys = sorted(data, key=lambda k: (k[0], k[1].rsplit("_", 1)[0], level.get(k[1].rsplit("_", 1)[1], 9), k[2]))
    if args.instances:
        wanted = {(f, int(s)) for f, s in (x.split(":") for x in args.instances.split(","))}
        keys = [k for k in keys if (k[1], k[2]) in wanted]
    if not keys:
        raise SystemExit("no matching instances")

    cols = min(len(keys), 3)
    rows = (len(keys) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.6 * rows), facecolor=SURFACE, squeeze=False)
    for ax in axes.flat[len(keys):]:
        ax.set_visible(False)

    for ax, key in zip(axes.flat, keys):
        model, family, seed = key
        methods = data[key]
        ax.set_facecolor(SURFACE)
        limit = max((m.get("time_limit") or 0) for m in methods.values()) or None
        if "direct" in methods:
            pts = gurobi_curve(methods["direct"], args.gap_denominator)
            if pts:
                xs, ys = zip(*pts)
                if limit and methods["direct"]["status"] != "OPTIMAL":
                    xs, ys = xs + (limit,), ys + (ys[-1],)
                ax.step(xs, ys, where="post", color=GUROBI, linewidth=2, label="Gurobi (full model)")
        if "benders" in methods:
            pts, feas = benders_curve(methods["benders"], args.gap_denominator)
            if pts:
                xs, ys = zip(*pts)
                ax.step(xs, ys, where="post", color=BENDERS, linewidth=2, label="Benders")
            if feas:
                fx, fy = zip(*feas)
                ax.plot(fx, fy, "|", color=BENDERS, markersize=7, alpha=0.6, label="feasibility cut")
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_ylim(bottom=0)
        if limit:
            ax.set_xlim(0, limit)
        ax.set_title(f"{family.replace('_', ' ')} ({model}-commodity)", loc="left", fontsize=10, fontweight="bold", color=INK)
        ax.set_xlabel("Wall-clock time (s)", color=INK2, fontsize=9)
        ax.set_ylabel("Optimality gap (%)", color=INK2, fontsize=9)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c3c2b7")
        ax.tick_params(colors=MUTED, labelsize=8)

    handles, labels = [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = args.out or os.path.splitext(args.records[0])[0] + "_convergence.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
