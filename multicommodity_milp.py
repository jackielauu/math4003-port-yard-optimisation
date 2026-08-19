"""
Monolithic MILP for MATH4003 Chapter 3 -- multi-commodity extension
Integrated Mid-Term Port Yard Planning in Iron Ore Supply Chains

Extends monolithic_milp.py to a multi-commodity flow

WHAT CHANGED FROM THE SINGLE-COMMODITY VERSION:
  - Flow, and everything downstream of flow, now carries a product index g
    drawn from G = {P1, P2, P3, X1}: f_{i,j,d} -> f_{i,j,g,d}.
  - Constraints that are genuinely about material movement now hold per
    product: flow conservation (4), stockyard inventory balance (5),
    throughput definition (9).
  - Constraints that represent SHARED PHYSICAL CAPACITY stay aggregated
    across products, since an asset/arc/zone's capacity is a property of
    the infrastructure, not of any one commodity: stockyard capacity (6),
    asset capacity (7), arc capacity (8). Overflow o_{p,d} likewise stays
    per-zone rather than per-(zone,product) for the same reason.
  - Maintenance-side constraints (1), (2), (3) are unchanged -- maintenance
    decisions aren't product-specific.
  - Product-routing restrictions (e.g. X1 only through TU3-ST4-Z4-RC3-SL2,
    only permitted in zone Z4) are enforced by fixing the upper bound of
    f[i,j,g,d] and z[p,g,d] to 0 wherever the generator's compatibility
    data says (arc, g) or (zone, g) isn't allowed, rather than adding new
    constraints -- this keeps the constraint count matching Chapter 3.

NOT YET IN SCOPE: per-product demand satisfaction, X1's protected-campaign
footprint/buffer/cleanout rules. The generator already exports the data
for these (demand_tonnes, special_activity_* fields) if/when Chapter 4/5
extends the formulation to cover them.
"""

import json
import sys

import gurobipy as gp
from gurobipy import GRB


def load_scenario(path):
    with open(path) as f:
        raw = json.load(f)
    m = raw["model_ready_data"]

    D = m["D"]
    N = m["N"]
    A = m["A"]
    A_NS = m["A_NS"]
    P_ = m["Z"]              # stockyard set (Chapter 3's P) -- see naming note in monolithic_milp.py
    G = m["G"]                # product set: P1, P2, P3, X1
    M = m["M_OPERATIONS"]
    M_a = m["M_a"]
    K = m["K"]

    arc_ids = list(m["arc_from"].keys())
    E = [(m["arc_from"][a], m["arc_to"][a]) for a in arc_ids]
    arc_id_of = {(m["arc_from"][a], m["arc_to"][a]): a for a in arc_ids}
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
    mu_min = {int(d): v for d, v in m["minimum_throughput_tpd"].items()}

    # z0[p, g]: initial inventory per zone per product -- the generator
    # already tracks this granularly, so no aggregation needed here.
    z0 = {}
    for key, val in m["initial_inventory_tonnes"].items():
        zz, gg = key.split("::")
        z0[(zz, gg)] = val

    # Product-compatibility lookups, used to fix incompatible (arc,g) and
    # (zone,g) variables to 0 rather than adding new constraints.
    product_arc_compatible = m["product_arc_compatible"]      # "g::arc_id" -> 1/0
    zone_allowed_product = m["zone_allowed_product"]           # "zone::g"   -> 1/0

    def arc_ok(i, j, g):
        arc_id = arc_id_of.get((i, j))
        if arc_id is None:
            return False
        if not product_arc_compatible.get(f"{g}::{arc_id}", 0):
            return False
        if i in P_ and not zone_allowed_product.get(f"{i}::{g}", 0):
            return False
        if j in P_ and not zone_allowed_product.get(f"{j}::{g}", 0):
            return False
        return True

    return dict(
        D=D, N=N, A=A, A_NS=A_NS, P_=P_, G=G, M=M, M_a=M_a, K=K,
        E=E, U=U, predecessors=predecessors, successors=successors,
        C=C, p=p, eta=eta, e_m=e_m, l_m=l_m, sigma=sigma, h=h,
        Z_min=Z_min, Z_max=Z_max, alpha=alpha, z0=z0, mu_min=mu_min,
        arc_ok=arc_ok, zone_allowed_product=zone_allowed_product,
    )


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
    arc_ok = data["arc_ok"]
    zone_allowed_product = data["zone_allowed_product"]

    model = gp.Model("math4003_multicommodity_milp")

    # --- Decision variables ---
    x = model.addVars(((mm, k, d) for mm in M for k in K for d in D),
                       vtype=GRB.BINARY, name="x")
    delta = model.addVars(M, D, lb=0.0, ub=1.0, name="delta")

    # f[i,j,g,d]: only create/allow flow where the product is actually
    # permitted on that arc -- everything else is fixed to 0 rather than
    # left to the solver to discover.
    f = model.addVars(E, G, D, lb=0.0, name="f")
    for (i, j) in E:
        for g in G:
            if not arc_ok(i, j, g):
                for d in D:
                    f[i, j, g, d].ub = 0.0

    z = model.addVars(P_, G, D, lb=0.0, name="z")
    for pz in P_:
        for g in G:
            if not zone_allowed_product.get(f"{pz}::{g}", 0):
                for d in D:
                    z[pz, g, d].ub = 0.0

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
