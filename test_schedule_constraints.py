import random
import unittest
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

import benders_monolithic
import benders_multicommodity
import monolithic_milp
import multicommodity_milp
from schedule_constraints import DEFAULT_SCHEDULE, ScheduleOptions, add_schedule_constraints, layered_stages

SCENARIOS = Path(__file__).parent / "synthetic_data_generator"
SINGLE = SCENARIOS / "singlecommodity_scenarios/F1_system_scale_L/seed_001/F1_system_scale_L_seed_001.json"
MULTI = SCENARIOS / "multicommodity_scenarios/F1_system_scale_L/seed_001/F1_system_scale_L_seed_001.json"


def chosen_starts(x):
    return [key for key, var in x.items() if var.X > 0.5]


def active_ops_by_day(data, starts):
    active = {}
    for mm, k, tau in starts:
        for day in range(tau, tau + data["p"][(mm, k)]):
            active.setdefault(day, []).append(mm)
    return active


class ScheduleConstraintTests(unittest.TestCase):
    def solve_direct(self, module, path, options, seconds=60, objective=None):
        data = module.load_scenario(path)
        model, variables = module.build_model(data, schedule=options)
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = seconds
        model.Params.Threads = 4
        if objective is not None:
            model.setObjective(objective(variables["x"]), GRB.MAXIMIZE)
        model.optimize()
        self.assertGreater(model.SolCount, 0, f"no schedule found (status {model.Status})")
        return data, model, variables

    def test_defaults_match_the_agreed_setup(self):
        # 7/10 meeting: the no-alignment (same-asset no-overlap) constraint goes into every
        # formulation; the concurrency cap is not part of the agreed formulation.
        self.assertTrue(DEFAULT_SCHEDULE.same_asset_no_overlap)
        self.assertIsNone(DEFAULT_SCHEDULE.max_concurrent_ops)
        self.assertTrue(DEFAULT_SCHEDULE.downstream_stage_cuts)
        self.assertTrue(DEFAULT_SCHEDULE.upstream_prefix_cuts)
        self.assertEqual(ScheduleOptions.none(), ScheduleOptions(False, None, False, False))
        self.assertNotEqual(ScheduleOptions.none(), DEFAULT_SCHEDULE)

    def test_network_is_layered_and_stages_are_found(self):
        data = monolithic_milp.load_scenario(SINGLE)
        stages = layered_stages(data)
        self.assertEqual([layer[0][:2] for layer in stages["up"]], ["TU", "ST"])
        self.assertEqual([layer[0][:2] for layer in stages["down"]], ["RC", "SL"])

    def test_cap_and_no_overlap_hold_in_direct_solutions(self):
        for module, path in ((monolithic_milp, SINGLE), (multicommodity_milp, MULTI)):
            with self.subTest(model=module.__name__):
                options = ScheduleOptions(same_asset_no_overlap=True, max_concurrent_ops=3)
                data, model, variables = self.solve_direct(module, path, options)
                starts = chosen_starts(variables["x"])
                active = active_ops_by_day(data, starts)
                asset_of = {mm: a for a, ops in data["M_a"].items() for mm in ops}
                self.assertLessEqual(max(len(ops) for ops in active.values()), 3)
                for day, ops in active.items():
                    assets = [asset_of[mm] for mm in ops]
                    self.assertEqual(len(assets), len(set(assets)), f"two ops on one asset on day {day}")
                model.dispose()

    def test_without_the_constraints_the_direct_optimum_stacks_operations(self):
        # Guards the "different problem" finding: with no schedule constraints the direct
        # optimum runs several operations on one asset at the same time.
        data, model, variables = self.solve_direct(monolithic_milp, SINGLE, ScheduleOptions.none())
        asset_of = {mm: a for a, ops in data["M_a"].items() for mm in ops}
        active = active_ops_by_day(data, chosen_starts(variables["x"]))
        self.assertTrue(any(len({asset_of[mm] for mm in ops}) < len(ops) for ops in active.values()))
        model.dispose()

    def test_master_inequalities_never_reject_a_schedule_the_full_model_accepts(self):
        """The stage inequalities must be valid: every schedule feasible for the full
        model (flows included) must be feasible for the master. Random objectives on the
        start variables give diverse feasible schedules, not just the best one."""
        for module, path in ((monolithic_milp, SINGLE), (multicommodity_milp, MULTI)):
            data = module.load_scenario(path)
            for seed in range(4):
                with self.subTest(model=module.__name__, seed=seed):
                    rng = random.Random(seed)
                    weights = {}
                    _, model, variables = self.solve_direct(
                        module, path, DEFAULT_SCHEDULE, seconds=30,
                        objective=lambda x: gp.quicksum(weights.setdefault(key, rng.uniform(-1, 1)) * var
                                                        for key, var in x.items()),
                    )
                    starts = set(chosen_starts(variables["x"]))
                    model.dispose()

                    master, mvars = benders_monolithic.build_master(data)
                    master.Params.OutputFlag = 0
                    for key, var in mvars["x"].items():
                        var.LB = var.UB = 1.0 if key in starts else 0.0
                    master.optimize()
                    self.assertEqual(master.Status, GRB.OPTIMAL,
                                     "master rejected a schedule that is feasible for the full model")
                    master.dispose()

    def test_stage_inequalities_reject_only_schedules_the_flow_model_cannot_serve(self):
        """Sharper validity check: sample schedules that the master accepts without the
        stage inequalities (objectives that favour heavy simultaneous downtime), and
        whenever the stage inequalities reject one, the flow subproblem for that
        schedule must really be infeasible. Also proves the inequalities bite."""
        base = ScheduleOptions(downstream_stage_cuts=False, upstream_prefix_cuts=False)
        for module, bmodule, path in ((monolithic_milp, benders_monolithic, SINGLE),
                                      (multicommodity_milp, benders_multicommodity, MULTI)):
            data = module.load_scenario(path)
            rejected = 0
            for seed in range(6):
                rng = random.Random(seed)
                master, mvars = benders_monolithic.build_master(data, base)
                master.Params.OutputFlag = 0
                master.Params.TimeLimit = 20
                master.setObjective(gp.quicksum(rng.uniform(0, 1) * v for v in mvars["delta"].values()), GRB.MAXIMIZE)
                master.optimize()
                starts = {key: round(var.X) for key, var in mvars["x"].items()}
                delta_value = {key: var.X for key, var in mvars["delta"].items()}
                master.dispose()

                full, fvars = benders_monolithic.build_master(data)
                full.Params.OutputFlag = 0
                for key, var in fvars["x"].items():
                    var.LB = var.UB = starts[key]
                full.optimize()
                if full.Status != GRB.OPTIMAL:
                    rejected += 1
                    sub, _ = bmodule.build_subproblem(data, delta_value)
                    sub.Params.OutputFlag = 0
                    sub.Params.DualReductions = 0
                    sub.optimize()
                    self.assertEqual(sub.Status, GRB.INFEASIBLE,
                                     f"{module.__name__} seed {seed}: inequality rejected a servable schedule")
                    sub.dispose()
                full.dispose()
            self.assertGreater(rejected, 0, f"{module.__name__}: the stage inequalities never rejected anything")

    def test_stage_inequalities_are_actually_added(self):
        data = monolithic_milp.load_scenario(SINGLE)

        def fresh():
            model = gp.Model()
            x = model.addVars([(mm, k, d) for mm in data["M"] for k in data["K"] for d in data["D"]
                               if data["e_m"][mm] <= d <= data["l_m"][mm]], vtype=GRB.BINARY, name="x")
            delta = model.addVars(data["M"], data["D"], lb=0.0, ub=1.0, name="delta")
            return model, x, delta

        # no_overlap/concurrency_cap are off by default, but the mechanism must still work
        # if a future comparison re-enables them explicitly.
        model, x, delta = fresh()
        on = ScheduleOptions(same_asset_no_overlap=True, max_concurrent_ops=5,
                             downstream_stage_cuts=True, upstream_prefix_cuts=True)
        counts = add_schedule_constraints(model, x, delta, data, on, valid_inequalities=True)
        for family in ("no_overlap", "concurrency_cap", "stage_cap", "cum_cap"):
            self.assertGreater(counts[family], 0, family)
        model.dispose()

        # With the agreed defaults: no-overlap and the valid inequalities, no cap.
        model, x, delta = fresh()
        default_counts = add_schedule_constraints(model, x, delta, data, valid_inequalities=True)
        self.assertGreater(default_counts["no_overlap"], 0)
        self.assertEqual(default_counts["concurrency_cap"], 0)
        self.assertGreater(default_counts["stage_cap"], 0)
        self.assertGreater(default_counts["cum_cap"], 0)
        model.dispose()

        model, x, delta = fresh()
        direct_counts = add_schedule_constraints(model, x, delta, data, valid_inequalities=False)
        self.assertEqual((direct_counts["stage_cap"], direct_counts["cum_cap"]), (0, 0))
        model.dispose()


if __name__ == "__main__":
    unittest.main()
