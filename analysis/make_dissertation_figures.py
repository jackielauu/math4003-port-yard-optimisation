"""Figures 1, 2 and 8 of the dissertation: the F1 medium network, the feasibility-cut schematic
and the scalability plot. The network is read from the F1_system_scale_M seed 1 instance
(multicommodity file, which also carries the special-product route); the timings and gaps are
read from results/core5_{single,multi}.jsonl. The cut schematic is illustrative (no data).

Run from the repository root:
  python3 analysis/make_dissertation_figures.py
"""
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch
import numpy as np

BLUE, ORANGE = '#1f77b4', '#ff7f0e'
INK, MUTED, GRID = '#1a1a1a', '#5a5a5a', '#d8d8d8'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#9a9a9a',
                     'axes.labelcolor': INK, 'text.color': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED,
                     'font.family': 'DejaVu Sans'})

INSTANCE = "synthetic_data_generator/{}_scenarios/F1_system_scale_{}/seed_001/F1_system_scale_{}_seed_001.json"


def instance(kind, level):
    return json.load(open(INSTANCE.format(kind, level, level)))


# ---------------------------------------------------------------- Figure: network
network = instance("multicommodity", "M")["model_ready_data"]
ARCS = [f"{network['arc_from'][e]}__{network['arc_to'][e]}" for e in network["E"]]
SPECIAL = {(network['arc_from'][k.split("::", 1)[1]], network['arc_to'][k.split("::", 1)[1]])
           for k, v in network["product_arc_compatible"].items() if k.startswith("S1::") and v}
nodes = {n for arc in ARCS for n in arc.split('__', 1)}
natural = lambda n: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', n)]
members = lambda prefix: sorted((n for n in nodes if re.fullmatch(prefix + r'\d+', n)), key=natural)
COLS = [('Rail\nsource', ['RAIL_SOURCE']),
        ('Train\nunloaders', members('TU')),
        ('Stackers', members('ST')),
        ('Stockyard\nzones', members('Z')),
        ('Reclaimers', members('RC')),
        ('Ship\nloaders', members('SL')),
        ('Vessel\nsink', ['VESSEL_SINK'])]
pos, SPAN = {}, 6.0
for ci, (_, members) in enumerate(COLS):
    n = len(members)
    ys = [SPAN / 2] if n == 1 else list(np.linspace(SPAN, 0, n))
    for m, y in zip(members, ys):
        pos[m] = (ci * 2.3, y)

fig, ax = plt.subplots(figsize=(9.6, 4.6))
for arc in ARCS:
    i, j = arc.split('__', 1)
    x1, y1 = pos[i]; x2, y2 = pos[j]
    sp = (i, j) in SPECIAL
    ax.annotate('', xy=(x2 - 0.46, y2), xytext=(x1 + 0.46, y1),
                arrowprops=dict(arrowstyle='-|>', shrinkA=0, shrinkB=0,
                                color=ORANGE if sp else '#b9c6d2',
                                lw=1.6 if sp else 0.9,
                                alpha=1.0 if sp else 0.85, zorder=2 if sp else 1))
for ci, (label, members) in enumerate(COLS):
    for m in members:
        x, y = pos[m]
        zone, term = m.startswith('Z'), m in ('RAIL_SOURCE', 'VESSEL_SINK')
        face = '#fdf0e3' if zone else ('#f2f2f2' if term else '#eaf1f7')
        edge = '#e0a260' if zone else ('#b0b0b0' if term else '#7fa8c9')
        style = 'round,pad=0.10,rounding_size=0.30' if zone else 'square,pad=0.10'
        ax.add_patch(FancyBboxPatch((x - 0.44, y - 0.26), 0.88, 0.52,
                                    boxstyle=style, linewidth=1.1,
                                    facecolor=face, edgecolor=edge, zorder=3))
        ax.text(x, y, m.replace('_SOURCE', '').replace('VESSEL_SINK', 'Vessels'),
                ha='center', va='center', fontsize=7.4, zorder=4, color=INK)
    ax.text(ci * 2.3, SPAN + 0.95, label, ha='center', va='center',
            fontsize=8.2, color=MUTED, linespacing=1.3)
h1 = plt.Line2D([], [], color='#7fa8c9', marker='s', ls='', ms=8,
                markerfacecolor='#eaf1f7', label='Capacity-constrained asset')
h2 = plt.Line2D([], [], color='#e0a260', marker='o', ls='', ms=8,
                markerfacecolor='#fdf0e3', label='Stockyard zone (storage)')
h3 = plt.Line2D([], [], color=ORANGE, lw=1.8, label='Line reserved for the special product S1')
ax.legend(handles=[h1, h2, h3], loc='lower center', bbox_to_anchor=(0.5, -0.13),
          ncol=3, frameon=False, fontsize=8)
ax.set_xlim(-1.1, (len(COLS) - 1) * 2.3 + 1.1); ax.set_ylim(-1.5, SPAN + 1.6)
ax.axis('off')
fig.tight_layout()
fig.savefig('analysis/figures/fig_network.png', dpi=220, bbox_inches='tight', facecolor='white')
plt.close(fig)

# ------------------------------------------------------- Figure: feasibility cut
fig, ax = plt.subplots(figsize=(6.6, 4.3))
xs = np.linspace(0, 1, 60)
ax.fill_between(xs, 0.92 - xs, 1.05, facecolor=ORANGE, alpha=0.07, zorder=0)
ax.fill_between(xs, 0.92 - xs, 1.05, facecolor='none', hatch='////',
                edgecolor=ORANGE, linewidth=0.0, alpha=0.22, zorder=0)
ax.plot(xs, 0.92 - xs, color=ORANGE, lw=2.2, zorder=3)
region = Polygon([(0, 0), (0.62, 0), (0.62, 0.30), (0.30, 0.62), (0, 0.62)],
                 closed=True, facecolor='#e4eef6', edgecolor='#5b8db5', lw=1.5, zorder=4)
ax.add_patch(region)
ax.text(0.26, 0.24, 'schedules whose flow\nproblem is feasible',
        fontsize=8.6, color='#2c5d85', ha='center', va='center', zorder=6)
ax.plot([0.70], [0.62], marker='o', ms=9, color='#c0392b', zorder=7)
ax.annotate('schedule the master proposed',
            xy=(0.70, 0.62), xytext=(0.62, 0.93), fontsize=8.6, color='#c0392b',
            ha='center', zorder=7,
            arrowprops=dict(arrowstyle='-|>', color='#c0392b', lw=1.1))
ax.annotate('the cut removes this entire\nhalf-space, not only that point',
            xy=(0.80, 0.30), xytext=(0.78, 0.17), fontsize=8.6, color='#b35c0a',
            ha='center', va='center', zorder=7,
            arrowprops=dict(arrowstyle='-|>', color='#b35c0a', lw=1.0))
ax.set_xlabel('capacity loss at asset $a$ over the window')
ax.set_ylabel('capacity loss at asset $b$ over the window')
ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.0)
ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0])
ax.grid(True, color=GRID, lw=0.6, alpha=0.5); ax.set_axisbelow(True)
for s_ in ('top', 'right'):
    ax.spines[s_].set_visible(False)
fig.tight_layout()
fig.savefig('analysis/figures/fig_cut.png', dpi=220, bbox_inches='tight', facecolor='white')
plt.close(fig)

# ------------------------------------------------------------ Figure: scalability
LEVELS = ("L", "M", "H")
assets = [instance("singlecommodity", lv)["diagnostics"]["asset_count"] for lv in LEVELS]


def core5(model, method, field, scale=1.0):
    recs = {r["family"]: r for r in map(json.loads, open(f"results/core5_{model}.jsonl")) if r["method"] == method}
    return [scale * recs[f"F1_system_scale_{lv}"][field] for lv in LEVELS]


mono_s, mono_m = core5("single", "direct", "wall_seconds"), core5("multi", "direct", "wall_seconds")
gap_s, gap_m = core5("single", "benders", "gap", 100), core5("multi", "benders", "gap", 100)
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
a = axes[0]
a.plot(assets, mono_s, color=BLUE, lw=2, marker='o', ms=7, label='single-commodity')
a.plot(assets, mono_m, color=BLUE, lw=2, marker='s', ms=7, ls='--', label='multicommodity')
a.set_yscale('log'); a.set_ylabel('time to proved optimality (s)')
a.set_title('(a)  Monolithic model', fontsize=9.5, loc='left', color=INK)
for x, y, t in [(assets[-1], mono_s[-1], f'{mono_s[-1]:.1f} s'), (assets[-1], mono_m[-1], f'{mono_m[-1]:.1f} s')]:
    a.annotate(t, xy=(x, y), xytext=(-6, 9), textcoords='offset points',
               fontsize=8.2, color=MUTED, ha='right')
b = axes[1]
b.plot(assets, gap_s, color=ORANGE, lw=2, marker='o', ms=7, label='single-commodity')
b.plot(assets, gap_m, color=ORANGE, lw=2, marker='s', ms=7, ls='--', label='multicommodity')
b.set_ylabel('certified gap at 600 s (%)'); b.set_ylim(0, 45)
b.set_title('(b)  Benders decomposition', fontsize=9.5, loc='left', color=INK)
for x, y, t in [(assets[-1], gap_s[-1], f'{gap_s[-1]:.1f}%'), (assets[-1], gap_m[-1], f'{gap_m[-1]:.1f}%')]:
    b.annotate(t, xy=(x, y), xytext=(-6, 9), textcoords='offset points',
               fontsize=8.2, color=MUTED, ha='right')
for ax_ in axes:
    ax_.set_xlabel('capacity-constrained assets')
    ax_.set_xticks(assets); ax_.set_xlim(9, 35)
    ax_.grid(True, color=GRID, lw=0.6, alpha=0.6); ax_.set_axisbelow(True)
    for s in ('top', 'right'):
        ax_.spines[s].set_visible(False)
    ax_.legend(frameon=False, fontsize=8.4, loc='upper left')
fig.tight_layout()
fig.savefig('analysis/figures/fig_scalability.png', dpi=220, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('figures written')
