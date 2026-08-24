"""Single-commodity monolithic MILP for the frozen canonical schema.

The scenario JSON is loaded through model_data_loader.py. The loader validates
that G has exactly one commodity, then this model uses aggregate flow and
stockyard inventory variables: x, delta, f, z, o, and mu.
"""

import sys

import gurobipy as gp
from gurobipy import GRB

from model_data_loader import load_canonical_scenario


def load_scenario(path):
    data = load_canonical_scenario(path, require_single_commodity=True)
    g0 = data["G"][0]
    data["g0"] = g0
    data["z0"] = {zone: data["z0_pg"][(zone, g0)] for zone in data["P_"]}
    data["is_multicommodity"] = False
    return data


def build_model(data, strict_capacity=False):
    """Build the Gurobi model. strict_capacity=True implements the strict
    no-overflow variant (o fixed to 0) mentioned alongside constraint (6);
    strict_capacity=False is the penalised-overflow base model implemented
    first, per Section 2.4."""
    D, N, A, A_NS, P_ = data["D"], data["N"], data["A"], data["A_NS"], data["P_"]
    M, M_a, K = data["M"], data["M_a"], data["K"]
    E, U = data["E"], data["U"]
    predecessors, successors = data["predecessors"], data["successors"]
    C, p, eta = data["C"], data["p"], data["eta"]
    e_m, l_m = data["e_m"], data["l_m"]
    sigma, h = data["sigma"], data["h"]
    Z_min, Z_max, alpha, z0 = data["Z_min"], data["Z_max"], data["alpha"], data["z0"]
    mu_min = data["mu_min"]

    model = gp.Model("math4003_monolithic_milp")

    # --- Decision variables (Section 2.3) ---
    valid_starts = [(mm, k, d) for mm in M for k in K for d in D if e_m[mm] <= d <= l_m[mm]]
    x = model.addVars(valid_starts, vtype=GRB.BINARY, name="x")
    delta = model.addVars(M, D, lb=0.0, ub=1.0, name="delta")
    f = model.addVars(E, D, lb=0.0, name="f")
    z = model.addVars(P_, D, lb=0.0, name="z")
    o = model.addVars(P_, D, lb=0.0, name="o")
    mu = model.addVars(D, lb=0.0, name="mu")

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
    # See MODELLING DECISIONS note (2) at top of file re: mode-dependent duration.
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

    # --- (7) asset capacity reduced by selected maintenance mode ---
    for a in A:
        for d in D:
            outflow = gp.quicksum(f[a, j, d] for j in successors.get(a, []) if (a, j) in U)
            loss = gp.quicksum(delta[mm, d] for mm in M_a.get(a, []))
            model.addConstr(
                outflow <= C[a] * sigma.get((a, d), 1.0) * (1 - loss),
                name=f"asset_capacity[{a},{d}]",
            )

    # --- (8) arc flow capacity ---
    for (i, j) in E:
        for d in D:
            model.addConstr(f[i, j, d] <= U[(i, j)], name=f"arc_capacity[{i},{j},{d}]")

    # --- (9) throughput definition + minimum daily target ---
    # See MODELLING DECISIONS note (1) at top of file re: mu_d definition.
    sink_nodes = [n for n in N if len(successors.get(n, [])) == 0]
    for d in D:
        inflow_to_sink = gp.quicksum(
            f[i, s, d] for s in sink_nodes for i in predecessors.get(s, []) if (i, s) in U
        )
        model.addConstr(mu[d] == inflow_to_sink, name=f"throughput_def[{d}]")
        model.addConstr(mu[d] >= mu_min.get(d, 0.0), name=f"min_throughput[{d}]")

    # --- (10) variable domains: x binary; delta, f, z, o, mu continuous >= 0,
    # 0 <= delta <= 1 -- all set at variable creation above.

    # --- Objective (Section 2.4): max sum_d mu_d - sum_p sum_d alpha_p * o_{p,d} ---
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
