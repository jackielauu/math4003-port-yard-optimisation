"""
Benders decomposition for monolithic_milp.py's single-commodity MILP.

Master problem (constraints 1, 2, 3, and "7" meaning model_ideas.pdf section
1.5's same-asset no-overlap constraint -- distinct from the original doc's
constraint 7): binary mode-and-start x, derived capacity-loss delta, and the
Benders value estimate theta. Pure scheduling feasibility -- no flow variables.

Subproblem (constraints 4, 5, 6, 7, 8, 9, where "7" now means the original
doc's asset-capacity constraint): continuous flow f, inventory z, overflow o,
throughput mu, for a *fixed* delta handed down from the master. Solved as an
LP; its constraint-7 duals drive the Benders optimality cut.

Only constraint 7 has an RHS that depends on the master's decision (through
delta) -- constraints 4, 5, 6, 8, 9 have fixed data on the RHS, so their
contribution to the subproblem's optimal value is already folded into
SubObj(delta_bar) and doesn't need to be re-derived as a function of x:

    theta <= SubObj(delta_bar)
             + sum_{a,d} pi_bar[a,d] * C_a * sigma[a,d] * (loss_bar[a,d] - loss[a,d])

where pi_bar is constraint 7's dual at the subproblem optimum for delta_bar,
and loss[a,d] = sum_{m in M_a[a]} delta[m,d].

Feasibility cuts (constraints 6 and 9 can make the subproblem infeasible,
which happens on nearly every scenario's first, cut-free master solve) use
Gurobi's Farkas-certificate duals rather than an explicit dual LP:
https://docs.gurobi.com/projects/optimizer/en/current/reference/attributes/constraintlinear.html#attrfarkasdual
https://support.gurobi.com/hc/en-us/community/posts/32075232484369-Dual-simplex-VS-Barrier-for-Bender-s-feasibility-cuts
FarkasDual gives lambda with lambda^T A y <= lambda^T b(delta_bar) "trivially
infeasible within the bounds of the variables" -- since y >= 0 with no finite
upper bound, that reduces to lambda^T b(delta_bar) < 0. As with the optimality
cut, only constraint 7's RHS depends on delta, so this becomes a linear cut on
the master's delta variables: lambda^T b(delta) >= 0. Farkas duals need a
simplex basis (not barrier), hence Method=1 and InfUnbdInfo=1 on the subproblem.
"""

import sys

import gurobipy as gp
from gurobipy import GRB

from monolithic_milp import load_scenario

__all__ = ["load_scenario", "build_master", "build_subproblem", "make_cut", "solve_benders"]


def _active_starts(e_m, l_m, mm, dur, d):
    lo, hi = max(e_m[mm], d - dur + 1), min(d, l_m[mm])
    return range(int(lo), int(hi) + 1) if lo <= hi else range(0)


def build_master(data):
    D, N, M, M_a, K, A = data["D"], data["N"], data["M"], data["M_a"], data["K"], data["A"]
    U, predecessors, successors = data["U"], data["predecessors"], data["successors"]
    e_m, l_m, p, eta, h = data["e_m"], data["l_m"], data["p"], data["eta"], data["h"]

    model = gp.Model("benders_master_monolithic")

    x = model.addVars(((mm, k, d) for mm in M for k in K for d in D), vtype=GRB.BINARY, name="x")
    delta = model.addVars(M, D, lb=0.0, ub=1.0, name="delta")

    # theta's initial bound: total throughput can never exceed the total
    # inbound-to-sink arc capacity summed over the horizon (constraint 9's own
    # bound, ignoring overflow penalty) -- keeps iteration 0 finite pre-cuts.
    sink_nodes = [n for n in N if len(successors.get(n, [])) == 0]
    daily_sink_capacity = sum(
        U[(i, s)] for s in sink_nodes for i in predecessors.get(s, []) if (i, s) in U
    )
    theta = model.addVar(lb=0.0, ub=daily_sink_capacity * len(D), name="theta")

    # --- (1) each maintenance op assigned exactly one mode and start day ---
    for mm in M:
        model.addConstr(
            gp.quicksum(x[mm, k, d] for k in K for d in D if e_m[mm] <= d <= l_m[mm]) == 1,
            name=f"assign_once[{mm}]",
        )

    # --- (2) delta_{m,d} derived from selected start/mode/duration ---
    for mm in M:
        for d in D:
            expr = gp.LinExpr()
            for k in K:
                for tau in _active_starts(e_m, l_m, mm, p[(mm, k)], d):
                    expr += (1 - eta[(mm, k)]) * x[mm, k, tau]
            model.addConstr(delta[mm, d] == expr, name=f"delta_def[{mm},{d}]")

    # --- (3) maintenance respects fixed high-priority production days ---
    for a in A:
        for mm in M_a.get(a, []):
            for d in D:
                expr = gp.LinExpr()
                for k in K:
                    for tau in _active_starts(e_m, l_m, mm, p[(mm, k)], d):
                        expr += x[mm, k, tau]
                model.addConstr(expr + h.get((a, d), 0) <= 1, name=f"fixed_activity[{a},{mm},{d}]")

    # --- (7-master) same-asset no-overlap (model_ideas.pdf section 1.5) ---
    # Only binds when an asset has more than one maintenance operation -- a
    # no-op otherwise, matching the note in monolithic_milp.py.
    for a in A:
        ops = M_a.get(a, [])
        if len(ops) < 2:
            continue
        for d in D:
            expr = gp.LinExpr()
            for mm in ops:
                for k in K:
                    for tau in _active_starts(e_m, l_m, mm, p[(mm, k)], d):
                        expr += x[mm, k, tau]
            model.addConstr(expr <= 1, name=f"no_overlap[{a},{d}]")

    model.setObjective(theta, GRB.MAXIMIZE)
    model.update()
    return model, dict(x=x, delta=delta, theta=theta)


def build_subproblem(data, delta_value):
    D, A, A_NS, P_, N = data["D"], data["A"], data["A_NS"], data["P_"], data["N"]
    E, U = data["E"], data["U"]
    predecessors, successors = data["predecessors"], data["successors"]
    C, sigma, M_a = data["C"], data["sigma"], data["M_a"]
    Z_min, Z_max, alpha, z0 = data["Z_min"], data["Z_max"], data["alpha"], data["z0"]
    mu_min = data["mu_min"]

    model = gp.Model("benders_subproblem_monolithic")

    f = model.addVars(E, D, lb=0.0, name="f")
    z = model.addVars(P_, D, lb=0.0, name="z")
    o = model.addVars(P_, D, lb=0.0, name="o")
    mu = model.addVars(D, lb=0.0, name="mu")

    # --- (4) flow conservation at non-storage processing assets ---
    for n in A_NS:
        for d in D:
            inflow = gp.quicksum(f[i, n, d] for i in predecessors.get(n, []) if (i, n) in U)
            outflow = gp.quicksum(f[n, j, d] for j in successors.get(n, []) if (n, j) in U)
            model.addConstr(inflow == outflow, name=f"flow_conservation[{n},{d}]")

    # --- (5) stockyard inventory balance ---
    for pz in P_:
        for d in D:
            prev = z0[pz] if d == D[0] else z[pz, d - 1]
            inflow = gp.quicksum(f[i, pz, d] for i in predecessors.get(pz, []) if (i, pz) in U)
            outflow = gp.quicksum(f[pz, j, d] for j in successors.get(pz, []) if (pz, j) in U)
            model.addConstr(z[pz, d] == prev + inflow - outflow, name=f"inventory_balance[{pz},{d}]")

    # --- (6) stockyard capacity ---
    for pz in P_:
        for d in D:
            model.addConstr(z[pz, d] >= Z_min[pz], name=f"zone_min[{pz},{d}]")
            model.addConstr(z[pz, d] <= Z_max[pz] + o[pz, d], name=f"zone_max[{pz},{d}]")

    # --- (7) asset capacity, with delta fixed at the value handed down by the master ---
    cap7 = {}
    for a in A:
        for d in D:
            outflow = gp.quicksum(f[a, j, d] for j in successors.get(a, []) if (a, j) in U)
            loss = sum(delta_value.get((mm, d), 0.0) for mm in M_a.get(a, []))
            rhs = C[a] * sigma.get((a, d), 1.0) * (1 - loss)
            cap7[a, d] = model.addConstr(outflow <= rhs, name=f"asset_capacity[{a},{d}]")

    # --- (8) arc flow capacity ---
    for (i, j) in E:
        for d in D:
            model.addConstr(f[i, j, d] <= U[(i, j)], name=f"arc_capacity[{i},{j},{d}]")

    # --- (9) throughput definition + minimum daily target ---
    sink_nodes = [n for n in N if len(successors.get(n, [])) == 0]
    for d in D:
        inflow_to_sink = gp.quicksum(
            f[i, s, d] for s in sink_nodes for i in predecessors.get(s, []) if (i, s) in U
        )
        model.addConstr(mu[d] == inflow_to_sink, name=f"throughput_def[{d}]")
        model.addConstr(mu[d] >= mu_min.get(d, 0.0), name=f"min_throughput[{d}]")

    model.setObjective(
        gp.quicksum(mu[d] for d in D) - gp.quicksum(alpha[pz] * o[pz, d] for pz in P_ for d in D),
        GRB.MAXIMIZE,
    )

    model.update()
    return model, dict(f=f, z=z, o=o, mu=mu, cap7=cap7)


def make_cut(delta, data, delta_value, subproblem, cap7):
    C, sigma, M_a = data["C"], data["sigma"], data["M_a"]
    cut = gp.LinExpr(subproblem.ObjVal)
    for (a, d), constr in cap7.items():
        pi = constr.Pi
        if pi == 0.0:
            continue
        coef = pi * C[a] * sigma.get((a, d), 1.0)
        loss_bar = sum(delta_value.get((mm, d), 0.0) for mm in M_a.get(a, []))
        loss_expr = gp.quicksum(delta[mm, d] for mm in M_a.get(a, []))
        cut += coef * (loss_bar - loss_expr)
    return cut


def make_feasibility_cut(delta, data, subproblem, cap7):
    """Farkas-certificate feasibility cut for an infeasible subproblem.

    Gurobi's FarkasDual gives lambda such that lambda^T A y <= lambda^T b is
    "trivially infeasible within the bounds of the variables" (see
    https://docs.gurobi.com/projects/optimizer/en/current/reference/attributes/constraintlinear.html#attrfarkasdual)
    -- since all of our subproblem variables are y >= 0 with no finite upper
    bound, that reduces to lambda^T b(delta_bar) < 0. As with the optimality
    cut, only constraint 7's RHS depends on delta, so lambda^T b(delta) is a
    constant (read off each constraint's live .RHS, valid regardless of solve
    status) plus a symbolic term over the cap7 constraints. The cut excludes
    exactly the delta region that produced this particular infeasibility:
    lambda^T b(delta) >= 0.
    """
    C, sigma, M_a = data["C"], data["sigma"], data["M_a"]
    cap7_constrs = set(cap7.values())
    cut = gp.LinExpr()
    for constr in subproblem.getConstrs():
        lam = constr.FarkasDual
        if lam == 0.0 or constr in cap7_constrs:
            continue
        cut += lam * constr.RHS
    for (a, d), constr in cap7.items():
        lam = constr.FarkasDual
        if lam == 0.0:
            continue
        coef = lam * C[a] * sigma.get((a, d), 1.0)
        loss_expr = gp.quicksum(delta[mm, d] for mm in M_a.get(a, []))
        cut += coef * (1 - loss_expr)
    return cut


def solve_benders(data, max_iters=200, tol=1e-4, verbose=False):
    """tol is a relative optimality gap (Gurobi's own MIPGap default is 1e-4)
    -- objective values here run in the tens of millions of tonnes, so a fixed
    absolute tolerance would be unrealistically tight."""
    master, mvars = build_master(data)
    x, delta, theta = mvars["x"], mvars["delta"], mvars["theta"]
    master.Params.OutputFlag = 0

    history = []
    best_lb = float("-inf")
    best = dict(delta=None, x=None, subproblem_vars=None)
    n_feasibility_cuts = 0
    n_optimality_cuts = 0

    for k in range(max_iters):
        master.optimize()
        if master.Status != GRB.OPTIMAL:
            return dict(
                status=f"MASTER_{master.Status}", iterations=k, history=history, best_objective=None,
                feasibility_cuts=n_feasibility_cuts, optimality_cuts=n_optimality_cuts,
            )

        ub = master.ObjVal
        delta_value = {key: v.X for key, v in delta.items()}

        subproblem, svars = build_subproblem(data, delta_value)
        subproblem.Params.OutputFlag = 0
        # Farkas duals for feasibility cuts need a simplex basis, not barrier.
        subproblem.Params.InfUnbdInfo = 1
        subproblem.Params.Method = 1  # dual simplex
        subproblem.optimize()

        if subproblem.Status == GRB.INFEASIBLE:
            cut = make_feasibility_cut(delta, data, subproblem, svars["cap7"])
            master.addConstr(cut >= 0, name=f"feasibility_cut[{k}]")
            n_feasibility_cuts += 1
            history.append(dict(iter=k, type="feasibility", ub=ub, lb=None))
            if verbose:
                print(f"iter {k}: UB={ub:.2f} subproblem infeasible -- added feasibility cut")
            continue

        if subproblem.Status != GRB.OPTIMAL:
            return dict(
                status=f"SUBPROBLEM_{subproblem.Status}", iterations=k, history=history,
                best_objective=None if best_lb == float("-inf") else best_lb,
                best_delta=best["delta"], best_x=best["x"], best_subproblem_vars=best["subproblem_vars"],
                feasibility_cuts=n_feasibility_cuts, optimality_cuts=n_optimality_cuts,
            )

        lb = subproblem.ObjVal
        if lb > best_lb:
            best_lb = lb
            best["delta"] = delta_value
            best["x"] = {key: v.X for key, v in x.items()}
            best["subproblem_vars"] = {
                name: {key: v.X for key, v in vardict.items()}
                for name, vardict in svars.items() if name != "cap7"
            }

        history.append(dict(iter=k, type="optimality", ub=ub, lb=lb))
        if verbose:
            print(f"iter {k}: UB={ub:.2f} LB={lb:.2f} gap={ub - lb:.2f}")

        if ub - lb <= tol * max(abs(ub), 1.0):
            return dict(
                status="OPTIMAL", iterations=k + 1, history=history, best_objective=best_lb,
                best_delta=best["delta"], best_x=best["x"], best_subproblem_vars=best["subproblem_vars"],
                feasibility_cuts=n_feasibility_cuts, optimality_cuts=n_optimality_cuts,
            )

        cut = make_cut(delta, data, delta_value, subproblem, svars["cap7"])
        master.addConstr(theta <= cut, name=f"optimality_cut[{k}]")
        n_optimality_cuts += 1

    return dict(
        status="MAX_ITERS", iterations=max_iters, history=history, best_objective=best_lb,
        best_delta=best["delta"], best_x=best["x"], best_subproblem_vars=best["subproblem_vars"],
        feasibility_cuts=n_feasibility_cuts, optimality_cuts=n_optimality_cuts,
    )


if __name__ == "__main__":
    scenario_path = sys.argv[1] if len(sys.argv) > 1 else \
        "synthetic_data_generator/singlecommodity_scenarios/F1_system_scale_H/seed_001/F1_system_scale_H_seed_001.json"

    data = load_scenario(scenario_path)
    result = solve_benders(data, verbose=True)
    print(f"Scenario: {scenario_path}")
    print(f"status={result['status']} iterations={result['iterations']} objective={result['best_objective']}")
