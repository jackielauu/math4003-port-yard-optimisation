"""Maintenance-schedule constraints shared by the direct MILPs and the Benders master.

Real constraints (change the feasible set, so they must be in BOTH the direct models
and the Benders master -- otherwise the two methods solve different problems):
  no_overlap[a,d]     at most one operation under maintenance per asset per day (the
                      "no-alignment" constraint). ON by default: agreed at the 7/10
                      meeting that it is added to every decomposition and every
                      non-decomposition formulation.
  concurrency_cap[d]  at most `max_concurrent_ops` operations under maintenance per
                      day across the network. OFF by default (not part of the agreed
                      formulation; kept for sensitivity runs).

Valid inequalities (implied by the flow constraints, so they do not change the
optimum; added to the Benders master, which has no flow variables, so that it
stops proposing schedules that break the minimum-flow requirement):
  stage_cap[l,d]      downstream of the stockyards, no storage sits between
                      the yard and the sink, so on every day the stage's
                      remaining capacity must cover mu_min(d).
  cum_cap[l,T]        upstream of the stockyards, the yard can buffer, but over
                      days 0..T the stage must still deliver the throughput the
                      initial stock cannot cover (prefix windows only, because
                      overflow makes the stock at later window starts unbounded).

Both stage families rely on the network being a strict layered chain
source -> assets -> stockyards -> assets -> sink; `layered_stages` returns None
(and they are skipped) if it is not.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import gurobipy as gp
from gurobipy import GRB


@dataclass(frozen=True)
class ScheduleOptions:
    same_asset_no_overlap: bool = True
    max_concurrent_ops: Optional[int] = None
    downstream_stage_cuts: bool = True
    upstream_prefix_cuts: bool = True

    @classmethod
    def none(cls):
        return cls(False, None, False, False)


DEFAULT_SCHEDULE = ScheduleOptions()


def _starts(data, mm, k, d):
    lo = max(data["e_m"][mm], d - data["p"][(mm, k)] + 1)
    hi = min(d, data["l_m"][mm])
    return range(int(lo), int(hi) + 1)


def _possible_ops(data, ops, d):
    return [mm for mm in ops if any(len(_starts(data, mm, k, d)) for k in data["K"])]


def _active(x, data, ops, d):
    return gp.quicksum(x[mm, k, tau] for mm in ops for k in data["K"] for tau in _starts(data, mm, k, d))


def layered_stages(data):
    """Return {"up": [layer, ...], "down": [layer, ...]} (lists of asset names, ordered
    away from the stockyards) if the network is a strict layered chain with all
    stockyards on one level, else None."""
    N, E, P = data["N"], data["E"], set(data["P_"])
    preds, succs = defaultdict(list), defaultdict(list)
    for i, j in E:
        preds[j].append(i)
        succs[i].append(j)

    level, remaining = {}, {n: len(preds[n]) for n in N}
    frontier = [n for n in N if remaining[n] == 0]
    for n in frontier:
        level[n] = 0
    while frontier:
        nxt = []
        for i in frontier:
            for j in succs[i]:
                level[j] = max(level.get(j, 0), level[i] + 1)
                remaining[j] -= 1
                if remaining[j] == 0:
                    nxt.append(j)
        frontier = nxt
    if len(level) != len(N) or any(level[j] != level[i] + 1 for i, j in E):
        return None

    zone_levels = {level[p] for p in P}
    if len(zone_levels) != 1:
        return None
    lz, lmax = zone_levels.pop(), max(level.values())
    sinks_ok = all(level[n] == lmax for n in N if not succs[n])
    sources_ok = all(level[n] == 0 for n in N if not preds[n])
    a_ns = set(data["A_NS"])
    by_level = defaultdict(list)
    for n in N:
        by_level[level[n]].append(n)
    middle_ok = all(n in a_ns for lvl in range(1, lmax) if lvl != lz for n in by_level[lvl])
    if not (sinks_ok and sources_ok and middle_ok and 0 < lz < lmax):
        return None
    return dict(
        up=[sorted(by_level[lvl]) for lvl in range(1, lz)],
        down=[sorted(by_level[lvl]) for lvl in range(lz + 1, lmax)],
    )


def add_schedule_constraints(model, x, delta, data, options=None, valid_inequalities=False):
    """Add the schedule constraints to `model`; returns a dict of row counts."""
    options = DEFAULT_SCHEDULE if options is None else options
    D, M_a = data["D"], data["M_a"]
    asset_of = {mm: a for a, ops in M_a.items() for mm in ops}
    counts = dict(no_overlap=0, concurrency_cap=0, stage_cap=0, cum_cap=0)

    if options.same_asset_no_overlap:
        for a, ops in M_a.items():
            if len(ops) < 2:
                continue
            for d in D:
                possible = _possible_ops(data, ops, d)
                if len(possible) >= 2:
                    model.addConstr(_active(x, data, possible, d) <= 1, name=f"no_overlap[{a},{d}]")
                    counts["no_overlap"] += 1

    cap = options.max_concurrent_ops
    if cap is not None:
        for d in D:
            possible = _possible_ops(data, list(data["M"]), d)
            distinct_assets = len({asset_of[mm] for mm in possible})
            # With no-overlap on, one op per asset already bounds the count by the asset count.
            bound = distinct_assets if options.same_asset_no_overlap else len(possible)
            if bound > cap:
                model.addConstr(_active(x, data, possible, d) <= cap, name=f"concurrency_cap[{d}]")
                counts["concurrency_cap"] += 1

    if valid_inequalities and (options.downstream_stage_cuts or options.upstream_prefix_cuts):
        stages = layered_stages(data)
        if stages is not None:
            _add_stage_inequalities(model, delta, data, options, stages, counts)
    return counts


def _capacity_terms(data, delta, layer, d):
    """(full capacity, capacity-loss expression) of a layer of assets on day d."""
    C, sigma, M_a = data["C"], data["sigma"], data["M_a"]
    full = sum(C[a] * sigma.get((a, d), 1.0) for a in layer)
    loss = gp.quicksum(C[a] * sigma.get((a, d), 1.0) * delta[mm, d] for a in layer for mm in M_a.get(a, []))
    return full, loss


def _add_stage_inequalities(model, delta, data, options, stages, counts):
    D, mu_min = data["D"], data["mu_min"]

    if options.downstream_stage_cuts:
        for li, layer in enumerate(stages["down"]):
            for d in D:
                full, loss = _capacity_terms(data, delta, layer, d)
                need = mu_min.get(d, 0.0)
                if need <= 0:
                    continue
                if loss.size() == 0 and full >= need:
                    continue
                model.addConstr(loss <= full - need, name=f"stage_cap[{li},{d}]")
                counts["stage_cap"] += 1

    if options.upstream_prefix_cuts:
        stock = sum(data["z0_pg"].values()) - sum(data["Z_min"].values())
        needs, running = {}, 0.0
        for d in D:
            running += mu_min.get(d, 0.0)
            needs[d] = running - stock
        first = next((d for d in D if needs[d] > 0), None)
        if first is None:
            return
        for li, layer in enumerate(stages["up"]):
            prev = None
            for d in D:
                full, loss = _capacity_terms(data, delta, layer, d)
                if d < first:
                    continue
                cum = model.addVar(lb=needs[d] if needs[d] > 0 else -GRB.INFINITY, name=f"cum_cap[{li},{d}]")
                if prev is None:
                    early = gp.LinExpr()
                    for dd in D:
                        if dd > d:
                            break
                        f_dd, l_dd = _capacity_terms(data, delta, layer, dd)
                        early += f_dd - l_dd
                    model.addConstr(cum == early, name=f"cum_def[{li},{d}]")
                else:
                    model.addConstr(cum == prev + full - loss, name=f"cum_def[{li},{d}]")
                prev = cum
                counts["cum_cap"] += 1
