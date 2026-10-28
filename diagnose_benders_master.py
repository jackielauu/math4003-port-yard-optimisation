"""Per-iteration profile of the single-commodity Benders master.

Runs the same loop as benders_monolithic.solve_benders, but records, for every
iteration: master solve time / nodes / gap, cut counts, subproblem time, and
the size and coefficient range of the cut just added. Writes one JSON object per
line. Use --root to point at a different checkout (e.g. a git worktree holding
another data version); --drop-no-overlap removes the master's same-asset
no-overlap constraints as a diagnostic ablation (the resulting schedules may be
physically invalid -- it is only for timing)."""

import argparse
import json
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="checkout providing benders_monolithic.py and the scenario data")
    ap.add_argument("--scenario", required=True, help="family_severity, e.g. F1_system_scale_L")
    ap.add_argument("--seed", default="seed_001")
    ap.add_argument("--out", required=True, help="JSON-lines output path")
    ap.add_argument("--drop-no-overlap", action="store_true")
    ap.add_argument("--max-iters", type=int, default=60)
    ap.add_argument("--wall-cap", type=float, default=240.0, help="stop starting new iterations after this many seconds")
    ap.add_argument("--master-time-limit", type=float, default=10.0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()

    sys.path.insert(0, args.root)
    os.chdir(args.root)
    import gurobipy as gp
    from gurobipy import GRB

    gp.setParam("Threads", args.threads)
    from benders_monolithic import (
        build_master, build_subproblem, load_scenario, make_cut, make_feasibility_cut,
    )

    path = f"synthetic_data_generator/singlecommodity_scenarios/{args.scenario}/{args.seed}/{args.scenario}_{args.seed}.json"
    data = load_scenario(path)
    master, mvars = build_master(data)
    delta, theta = mvars["delta"], mvars["theta"]
    master.Params.OutputFlag = 0
    master.Params.TimeLimit = args.master_time_limit

    n_removed = 0
    if args.drop_no_overlap:
        doomed = [c for c in master.getConstrs() if c.ConstrName.startswith("no_overlap")]
        n_removed = len(doomed)
        master.remove(doomed)
        master.update()

    out = open(args.out, "w")

    def emit(obj):
        out.write(json.dumps(obj) + "\n")
        out.flush()

    emit(dict(meta=dict(
        root=args.root, scenario=args.scenario, seed=args.seed, drop_no_overlap=args.drop_no_overlap,
        n_assets=len(data["A"]), n_ops=len(data["M"]),
        assets_with_multiple_ops=sum(1 for ops in data["M_a"].values() if len(ops) > 1),
        master_bin_vars=master.NumBinVars, master_constrs_start=master.NumConstrs,
        no_overlap_constrs_removed=n_removed, master_time_limit=args.master_time_limit,
        threads=args.threads, wall_cap=args.wall_cap,
    )))

    n_feas = n_opt = 0
    t_start = time.perf_counter()
    reason = "max_iters"
    for k in range(args.max_iters):
        if time.perf_counter() - t_start > args.wall_cap:
            reason = "wall_cap"
            break

        cuts_before = n_feas + n_opt
        master.optimize()
        if master.SolCount == 0:
            reason = f"master_no_solution_status_{master.Status}"
            break
        ub = master.ObjBound
        try:
            master_gap = master.MIPGap
        except gp.GurobiError:
            master_gap = None
        row = dict(
            iter=k, cuts_before=cuts_before, n_feas=n_feas, n_opt=n_opt,
            master_time=master.Runtime, master_status=master.Status,
            master_nodes=master.NodeCount, master_simplex_iters=master.IterCount, master_gap=master_gap,
            ub=ub,
        )

        delta_value = {key: v.X for key, v in delta.items()}
        sub, svars = build_subproblem(data, delta_value)
        sub.Params.OutputFlag = 0
        sub.Params.InfUnbdInfo = 1
        sub.Params.Method = 1
        sub.optimize()
        row.update(sub_time=sub.Runtime, sub_status=sub.Status)

        if sub.Status == GRB.INFEASIBLE:
            cut = make_feasibility_cut(delta, data, sub, svars["cap7"])
            master.addConstr(cut >= 0, name=f"feasibility_cut[{k}]")
            n_feas += 1
            row.update(cut_type="feasibility", lb=None)
        elif sub.Status == GRB.OPTIMAL:
            lb = sub.ObjVal
            row.update(lb=lb)
            if ub - lb <= args.tol * max(abs(ub), 1.0):
                row.update(cut_type="converged")
                emit(row)
                reason = "converged"
                break
            cut = make_cut(delta, data, delta_value, sub, svars["cap7"])
            master.addConstr(theta <= cut, name=f"optimality_cut[{k}]")
            n_opt += 1
            row.update(cut_type="optimality")
        else:
            reason = f"sub_status_{sub.Status}"
            emit(row)
            break

        coefs = [abs(cut.getCoeff(i)) for i in range(cut.size()) if cut.getCoeff(i) != 0.0]
        row.update(
            cut_nnz=len(coefs), cut_max_abs=max(coefs) if coefs else None,
            cut_min_abs=min(coefs) if coefs else None, cut_const=cut.getConstant(),
        )
        emit(row)

    emit(dict(done=True, reason=reason, elapsed=time.perf_counter() - t_start, cuts=n_feas + n_opt))
    out.close()


if __name__ == "__main__":
    main()
