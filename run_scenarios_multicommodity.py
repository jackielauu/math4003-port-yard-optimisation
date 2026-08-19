"""Run multicommodity_milp's model against every generated dissertation
scenario and report objective value, solve status, and any errors."""

import glob
import os

import gurobipy as gp

from multicommodity_milp import build_model, load_scenario

SCENARIO_DIR = "synthetic_data_generator/generated_dissertation_scenarios"

STATUS_NAMES = {
    gp.GRB.OPTIMAL: "OPTIMAL",
    gp.GRB.INFEASIBLE: "INFEASIBLE",
    gp.GRB.INF_OR_UNBD: "INF_OR_UNBD",
    gp.GRB.UNBOUNDED: "UNBOUNDED",
    gp.GRB.SUBOPTIMAL: "SUBOPTIMAL",
    gp.GRB.TIME_LIMIT: "TIME_LIMIT",
}


def scenario_paths():
    paths = sorted(glob.glob(os.path.join(SCENARIO_DIR, "*.json")))
    return [p for p in paths if os.path.basename(p) != "manifest.json"]


def run_scenario(path):
    try:
        data = load_scenario(path)
        model, _ = build_model(data)
        model.Params.OutputFlag = 0
        model.optimize()
        status = STATUS_NAMES.get(model.Status, str(model.Status))
        obj = model.ObjVal if model.SolCount > 0 else None
        return status, obj, None
    except gp.GurobiError as exc:
        return "ERROR", None, str(exc)
    except Exception as exc:  # noqa: BLE001 -- surfacing any load/build error per-scenario
        return "ERROR", None, str(exc)


def main():
    results = []
    for path in scenario_paths():
        name = os.path.basename(path)
        status, obj, err = run_scenario(path)
        results.append((name, status, obj, err))

    header = f"{'scenario':<34} {'status':<12} {'objective':>16}  error"
    print(header)
    print("-" * len(header))
    for name, status, obj, err in results:
        obj_str = f"{obj:,.2f}" if obj is not None else "-"
        print(f"{name:<34} {status:<12} {obj_str:>16}  {err or ''}")


if __name__ == "__main__":
    main()
