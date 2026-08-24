"""
Superseded by multicommodity_milp.py -- kept for reference only.

Monolithic MILP for MATH4003 Chapter 3
Integrated Mid-Term Port Yard Planning in Iron Ore Supply Chains

Implements the compact formulation from MATH4003_Mathematical_Model_21467181.docx
(Sections 2.1-2.5) exactly: sets, parameters, decision variables, objective, and
all ten constraints. This is the monolithic MILP that Section 3 of that document
names as the first implementation step, ahead of Benders decomposition.

SCOPE NOTE: Chapter 3's formulation is single-commodity (aggregate tonnes flow
f_{i,j,d}), while generate_dissertation_scenarios.py exports multi-product data
(P1, P2, P3, X1). This script aggregates across products so the model matches
Chapter 3 exactly. Extending to a multi-commodity flow (f_{i,j,g,d}) is the
natural next step once this base model is validated -- that's also where the
X1 special-product research question plugs in.

MODELLING DECISIONS NOT FULLY SPECIFIED IN THE SOURCE DOCUMENT (flagged for
supervisor confirmation, not silently assumed):
  1. mu_d (throughput) has no defining equation in Chapter 3, only mu_d >= mu_min.
     Defined here as total flow into the network's sink node(s) (nodes with no
     successors -- VESSEL_SINK in the generator's topology).
  2. Constraint (3)'s source-document image writes the x-summation window using
     a single p_m (not mode-indexed), while constraint (2) uses mode-dependent
     p_{m,k}. This implementation uses p_{m,k} throughout for consistency with
     (2). Worth a quick confirmation with Hoa/Ponpot that this is intended.

KEY NAMING NOTE: Chapter 3 uses P for the stockyard node set. In the current
generate_dissertation_scenarios.py export, model_ready_data['P'] is actually the
*product* set (P1, P2, P3, X1) -- a legacy key repurposed for the multi-product
extension. The stockyard set matching Chapter 3's P is model_ready_data['Z']
(equivalently 'P_STOCKYARD'). This script uses 'Z' throughout and calls it P_
in variable names to match the paper's notation without colliding with Python.
"""

import json
import sys

import gurobipy as gp
from gurobipy import GRB


def load_scenario(path):
    """Load one generate_dissertation_scenarios.py JSON output and translate
    its model_ready_data into Chapter 3's notation, aggregating per-product
    quantities to a single commodity where the generator exports them by
    product."""
    with open(path) as f:
        raw = json.load(f)
    m = raw["model_ready_data"]

    D = m["D"]
    N = m["N"]
    A = m["A"]
    A_NS = m["A_NS"]
    P_ = m["Z"]  # stockyard set -- see KEY NAMING NOTE above
    M = m["M_OPERATIONS"]
    M_a = m["M_a"]
    K = m["K"]  # ['full_shutdown', 'partial_30', 'partial_70'] == {F, P30, P70}

    arc_ids = list(m["arc_from"].keys())
    E = [(m["arc_from"][a], m["arc_to"][a]) for a in arc_ids]
    U = {(m["arc_from"][a], m["arc_to"][a]): m["arc_capacity_tpd"][a] for a in arc_ids}

    predecessors = m["predecessors"]
    successors = m["successors"]

    C = m["asset_capacity_tpd"]

    p, eta = {}, {}
    for key, val in m["maintenance_duration_days"].items():
        mm, kk = key.split("::")
        p[(mm, kk)] = val
    for key, val in m["maintenance_available_capacity_fraction"].items():
        mm, kk = key.split("::")
        eta[(mm, kk)] = val

    e_m = m["maintenance_earliest_start"]
    l_m = m["maintenance_latest_start"]

    sigma, h = {}, {}
    for key, val in m["special_activity_capacity_factor"].items():
        a, d = key.split("::")
        sigma[(a, int(d))] = val
    for key, val in m["maintenance_activity_incompatible"].items():
        a, d = key.split("::")
        h[(a, int(d))] = val

    Z_min = m["zone_min_inventory_tonnes"]
    Z_max = m["zone_max_inventory_tonnes"]
    alpha = m["overflow_penalty_per_tonne"]

    # z_p^0: generator tracks initial inventory per (zone, product) in the
    # multi-commodity scenario export, or pre-aggregated per zone in the
    # single-commodity export ("Z1" vs "Z1::P1") -- split(...)[0] handles both.
    z0 = {zp: 0.0 for zp in P_}
    for key, val in m["initial_inventory_tonnes"].items():
        zz = key.split("::")[0]
        if zz in z0:
            z0[zz] += val

    mu_min = {int(d): v for d, v in m["minimum_throughput_tpd"].items()}

    return dict(
        D=D, N=N, A=A, A_NS=A_NS, P_=P_, M=M, M_a=M_a, K=K,
        E=E, U=U, predecessors=predecessors, successors=successors,
        C=C, p=p, eta=eta, e_m=e_m, l_m=l_m, sigma=sigma, h=h,
        Z_min=Z_min, Z_max=Z_max, alpha=alpha, z0=z0, mu_min=mu_min,
    )


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
    x = model.addVars(((mm, k, d) for mm in M for k in K for d in D),
                       vtype=GRB.BINARY, name="x")
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
