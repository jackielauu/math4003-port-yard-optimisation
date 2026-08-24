"""Multi-commodity monolithic MILP for the frozen canonical schema.

This is the same physical model as monolithic_milp.py, with commodity index g
only on material flow and stockyard inventory variables. Maintenance variables,
overflow, total throughput, and physical capacities remain shared.
"""

import sys

import gurobipy as gp
from gurobipy import GRB

from model_data_loader import load_canonical_scenario


def load_scenario(path):
    data = load_canonical_scenario(path)
    data["z0"] = data["z0_pg"]
    data["is_multicommodity"] = True
    return data


def build_model(data, strict_capacity=False):
    D, N, A, A_NS, P_, G = data["D"], data["N"], data["A"], data["A_NS"], data["P_"], data["G"]
    M, M_a, K = data["M"], data["M_a"], data["K"]
    E, U = data["E"], data["U"]
    predecessors, successors = data["predecessors"], data["successors"]
    C, p, eta = data["C"], data["p"], data["eta"]
    e_m, l_m = data["e_m"], data["l_m"]
    sigma, h = data["sigma"], data["h"]
    Z_min, Z_max, alpha, z0 = data["Z_min"], data["Z_max"], data["alpha"], data["z0"]
    mu_min = data["mu_min"]

    model = gp.Model("math4003_multicommodity_milp")

    # --- Decision variables ---
    valid_starts = [(mm, k, d) for mm in M for k in K for d in D if e_m[mm] <= d <= l_m[mm]]
    x = model.addVars(valid_starts, vtype=GRB.BINARY, name="x")
    delta = model.addVars(M, D, lb=0.0, ub=1.0, name="delta")

    f = model.addVars(E, G, D, lb=0.0, name="f")
    z = model.addVars(P_, G, D, lb=0.0, name="z")
    o = model.addVars(P_, D, lb=0.0, name="o")   # aggregate overflow, per zone
    mu = model.addVars(D, lb=0.0, name="mu")     # aggregate throughput, per day

    if strict_capacity:
        for pz in P_:
            for d in D:
                o[pz, d].ub = 0.0

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
                dur = p[(mm, k)]
                lo, hi = max(e_m[mm], d - dur + 1), min(d, l_m[mm])
                if lo <= hi:
                    expr += (1 - eta[(mm, k)]) * gp.quicksum(
                        x[mm, k, tau] for tau in range(int(lo), int(hi) + 1)
                    )
            model.addConstr(delta[mm, d] == expr, name=f"delta_def[{mm},{d}]")

    # --- (3) maintenance respects fixed high-priority production days ---
    for a in A:
        for mm in M_a.get(a, []):
            for d in D:
                expr = gp.LinExpr()
                for k in K:
                    dur = p[(mm, k)]
                    lo, hi = max(e_m[mm], d - dur + 1), min(d, l_m[mm])
                    if lo <= hi:
                        expr += gp.quicksum(x[mm, k, tau] for tau in range(int(lo), int(hi) + 1))
                model.addConstr(expr + h.get((a, d), 0) <= 1, name=f"fixed_activity[{a},{mm},{d}]")

    # --- (4) flow conservation at non-storage processing assets -- PER PRODUCT ---
    for n in A_NS:
        for g in G:
            for d in D:
                inflow = gp.quicksum(f[i, n, g, d] for i in predecessors.get(n, []) if (i, n) in U)
                outflow = gp.quicksum(f[n, j, g, d] for j in successors.get(n, []) if (n, j) in U)
                model.addConstr(inflow == outflow, name=f"flow_conservation[{n},{g},{d}]")

    # --- (5) stockyard inventory balance -- PER PRODUCT ---
    for pz in P_:
        for g in G:
            for d in D:
                prev = z0.get((pz, g), 0.0) if d == D[0] else z[pz, g, d - 1]
                inflow = gp.quicksum(f[i, pz, g, d] for i in predecessors.get(pz, []) if (i, pz) in U)
                outflow = gp.quicksum(f[pz, j, g, d] for j in successors.get(pz, []) if (pz, j) in U)
                model.addConstr(z[pz, g, d] == prev + inflow - outflow, name=f"inventory_balance[{pz},{g},{d}]")

    # --- (6) stockyard capacity -- shared physical space, aggregated across products ---
    for pz in P_:
        for d in D:
            total_inv = gp.quicksum(z[pz, g, d] for g in G)
            model.addConstr(total_inv >= Z_min[pz], name=f"zone_min[{pz},{d}]")
            model.addConstr(total_inv <= Z_max[pz] + o[pz, d], name=f"zone_max[{pz},{d}]")

    # --- (7) asset capacity -- shared across products, reduced by maintenance mode ---
    for a in A:
        for d in D:
            outflow = gp.quicksum(f[a, j, g, d] for j in successors.get(a, []) for g in G if (a, j) in U)
            loss = gp.quicksum(delta[mm, d] for mm in M_a.get(a, []))
            model.addConstr(
                outflow <= C[a] * sigma.get((a, d), 1.0) * (1 - loss),
                name=f"asset_capacity[{a},{d}]",
            )

    # --- (8) arc flow capacity -- shared across products ---
    for (i, j) in E:
        for d in D:
            model.addConstr(
                gp.quicksum(f[i, j, g, d] for g in G) <= U[(i, j)],
                name=f"arc_capacity[{i},{j},{d}]",
            )

    # --- (9) throughput definition + minimum daily target -- aggregate across products ---
    sink_nodes = [n for n in N if len(successors.get(n, [])) == 0]
    for d in D:
        inflow_to_sink = gp.quicksum(
            f[i, s, g, d] for s in sink_nodes for g in G
            for i in predecessors.get(s, []) if (i, s) in U
        )
        model.addConstr(mu[d] == inflow_to_sink, name=f"throughput_def[{d}]")
        model.addConstr(mu[d] >= mu_min.get(d, 0.0), name=f"min_throughput[{d}]")

    # --- (10) variable domains: set at creation above ---

    # --- Objective: max sum_d mu_d - sum_p sum_d alpha_p * o_{p,d} ---
    model.setObjective(
        gp.quicksum(mu[d] for d in D)
        - gp.quicksum(alpha[pz] * o[pz, d] for pz in P_ for d in D),
        GRB.MAXIMIZE,
    )

    model.update()
    return model, dict(x=x, delta=delta, f=f, z=z, o=o, mu=mu)


if __name__ == "__main__":
    scenario_path = sys.argv[1] if len(sys.argv) > 1 else \
        "synthetic_data_generator/generated_dissertation_scenarios/balanced_control.json"
    strict = "--strict" in sys.argv

    data = load_scenario(scenario_path)
    model, variables = build_model(data, strict_capacity=strict)
    print(f"Scenario: {scenario_path}")
    print(f"Variant: {'strict no-overflow' if strict else 'penalised overflow'}")
    print(f"Built model: {model.NumVars} variables, {model.NumConstrs} constraints")
    model.optimize()
