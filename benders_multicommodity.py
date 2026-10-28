"""
Benders decomposition for multicommodity_milp.py's multi-commodity MILP.

The master problem (constraints 1, 2, 3, 7-same-asset-no-overlap) is entirely
about maintenance scheduling -- it never touches products -- so it's identical
in shape to the single-commodity case and is reused directly from
benders_monolithic.py, along with the optimality/feasibility cut builders
(which likewise only depend on the asset-capacity constraint 7 and delta, not
on any per-product structure). See benders_monolithic.py's module docstring
for the full derivation of both cut types.

Only the subproblem changes: flow conservation (4) and inventory balance (5)
become per-product, while stockyard capacity (6), asset capacity (7), and arc
capacity (8) stay aggregated across products -- exactly mirroring how
multicommodity_milp.py's build_model extends monolithic_milp.py's.
"""

import sys

import gurobipy as gp
from gurobipy import GRB

from benders_monolithic import build_master, make_cut, make_feasibility_cut, run_benders
from multicommodity_milp import load_scenario, apply_routing_bounds

__all__ = ["load_scenario", "build_master", "build_subproblem", "make_cut", "make_feasibility_cut", "solve_benders"]


def build_subproblem(data, delta_value):
    D, N, A, A_NS, P_, G = data["D"], data["N"], data["A"], data["A_NS"], data["P_"], data["PRODUCTS"]
    E, U = data["E"], data["U"]
    predecessors, successors = data["predecessors"], data["successors"]
    C, sigma, M_a = data["C"], data["sigma"], data["M_a"]
    Z_min, Z_max, alpha, z0 = data["Z_min"], data["Z_max"], data["alpha"], data["z0"]
    mu_min = data["mu_min"]
    model = gp.Model("benders_subproblem_multicommodity")

    f = model.addVars(E, G, D, lb=0.0, name="f")
    z = model.addVars(P_, G, D, lb=0.0, name="z")
    apply_routing_bounds(data, f, z)

    o = model.addVars(P_, D, lb=0.0, name="o")
    mu = model.addVars(D, lb=0.0, name="mu")

    # --- (4) flow conservation at non-storage processing assets -- per product ---
    for n in A_NS:
        for g in G:
            for d in D:
                inflow = gp.quicksum(f[i, n, g, d] for i in predecessors.get(n, []) if (i, n) in U)
                outflow = gp.quicksum(f[n, j, g, d] for j in successors.get(n, []) if (n, j) in U)
                model.addConstr(inflow == outflow, name=f"flow_conservation[{n},{g},{d}]")

    # --- (5) stockyard inventory balance -- per product ---
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

    # --- (7) asset capacity -- aggregated across products, delta fixed by the master ---
    cap7 = {}
    for a in A:
        for d in D:
            outflow = gp.quicksum(f[a, j, g, d] for j in successors.get(a, []) for g in G if (a, j) in U)
            loss = sum(delta_value.get((mm, d), 0.0) for mm in M_a.get(a, []))
            rhs = C[a] * sigma.get((a, d), 1.0) * (1 - loss)
            cap7[a, d] = model.addConstr(outflow <= rhs, name=f"asset_capacity[{a},{d}]")

    # --- (8) arc flow capacity -- aggregated across products ---
    for (i, j) in E:
        for d in D:
            model.addConstr(
                gp.quicksum(f[i, j, g, d] for g in G) <= U[(i, j)],
                name=f"arc_capacity[{i},{j},{d}]",
            )

    # --- (9) throughput definition + minimum daily target -- aggregated across products ---
    sink_nodes = [n for n in N if len(successors.get(n, [])) == 0]
    for d in D:
        inflow_to_sink = gp.quicksum(
            f[i, s, g, d] for s in sink_nodes for g in G
            for i in predecessors.get(s, []) if (i, s) in U
        )
        model.addConstr(mu[d] == inflow_to_sink, name=f"throughput_def[{d}]")
        model.addConstr(mu[d] >= mu_min.get(d, 0.0), name=f"min_throughput[{d}]")

    model.setObjective(
        gp.quicksum(mu[d] for d in D) - gp.quicksum(alpha[pz] * o[pz, d] for pz in P_ for d in D),
        GRB.MAXIMIZE,
    )

    model.update()
    return model, dict(f=f, z=z, o=o, mu=mu, cap7=cap7)


def solve_benders(data, max_iters=200, tol=1e-4, verbose=False, schedule=None,
                  time_limit=None, master_time_limit=None):
    """See benders_monolithic.run_benders; only the subproblem differs."""
    return run_benders(data, build_subproblem, max_iters, tol, verbose, schedule, time_limit, master_time_limit)


if __name__ == "__main__":
    scenario_path = sys.argv[1] if len(sys.argv) > 1 else \
        "synthetic_data_generator/multicommodity_scenarios/F1_system_scale_H/seed_001/F1_system_scale_H_seed_001.json"

    data = load_scenario(scenario_path)
    result = solve_benders(data, verbose=True)
    print(f"Scenario: {scenario_path}")
    print(f"status={result['status']} iterations={result['iterations']} objective={result['best_objective']}")
