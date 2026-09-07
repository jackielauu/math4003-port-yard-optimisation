import inspect
import copy
import json
from types import SimpleNamespace

import benders_multicommodity
import tempfile
import unittest
from pathlib import Path

import gurobipy as gp

import model_data_loader
import monolithic_milp
import multicommodity_milp
from synthetic_data_generator import generate_dissertation_scenarios as generator


FORBIDDEN_SOLVER_FIELD_NAMES = (
    "product_asset_compatible",
    "arc_allowed_product",
    "product_is_special",
    "M_OPERATIONS",
)


class CanonicalMilpLoaderTests(unittest.TestCase):
    def write_instance(self, tmp, scenario_name, commodity_count, seed=1):
        scenario = generator.build_scenario(
            scenario_name,
            generator.GeneratorConfig(seed=seed, commodity_count=commodity_count),
        )
        path = Path(tmp) / f"{scenario_name}_{commodity_count}.json"
        generator.write_scenario(scenario, path)
        return scenario, path

    def test_single_and_multi_load_new_canonical_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            _single_scenario, single_path = self.write_instance(tmp, "F1_system_scale_L", 1)
            _multi_scenario, multi_path = self.write_instance(tmp, "F1_system_scale_L", 4)

            single_data = monolithic_milp.load_scenario(single_path)
            multi_data = multicommodity_milp.load_scenario(multi_path)

            self.assertEqual(single_data["G"], ["G1"])
            self.assertEqual(multi_data["G"], ["G1", "G2", "G3", "G4"])
            self.assertEqual(single_data["P_"], multi_data["P_"])
            self.assertEqual(single_data["C"], multi_data["C"])
            self.assertEqual(single_data["p"], multi_data["p"])
            self.assertEqual(single_data["eta"], multi_data["eta"])

    def test_routing_schema_rejects_missing_invalid_and_inconsistent_data(self):
        original = generator.build_scenario("F1_system_scale_L", generator.GeneratorConfig()).to_json_dict()["model_ready_data"]
        for field in ("SPECIAL_PRODUCTS", "product_arc_compatible", "zone_allowed_product", "product_zone_compatible"):
            data = copy.deepcopy(original)
            del data[field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                model_data_loader.validate_canonical_model_ready_data(data)
        for field in ("product_arc_compatible", "zone_allowed_product", "product_zone_compatible"):
            key = next(iter(original[field]))
            for mutation in ("missing", "extra", "nonbinary"):
                data = copy.deepcopy(original)
                if mutation == "missing":
                    del data[field][key]
                elif mutation == "extra":
                    data[field]["UNKNOWN::UNKNOWN"] = 1
                else:
                    data[field][key] = 0.5
                with self.subTest(field=field, mutation=mutation), self.assertRaises(ValueError):
                    model_data_loader.validate_canonical_model_ready_data(data)
        for mutation in ("zone_disagreement", "forbidden_inventory", "overlapping_products"):
            data = copy.deepcopy(original)
            if mutation == "zone_disagreement":
                data["product_zone_compatible"]["S1::Z1"] = 1
            elif mutation == "forbidden_inventory":
                data["initial_inventory_tonnes"]["Z1::S1"] = 1
            else:
                data["SPECIAL_PRODUCTS"].append("G1")
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                model_data_loader.validate_canonical_model_ready_data(data)

    def test_routing_bounds_forbid_transit_even_when_arc_matrix_allows_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            scenario, path = self.write_instance(tmp, "F1_system_scale_L", 4)
            raw = scenario.to_json_dict()
            # Zone permissions must override permissive incident-arc data.
            raw["model_ready_data"]["product_arc_compatible"] = {
                key: 1 for key in raw["model_ready_data"]["product_arc_compatible"]
            }
            path.write_text(json.dumps(raw))
            data = multicommodity_milp.load_scenario(path)
            self.assertEqual(data["PRODUCTS"], ["G1", "G2", "G3", "G4", "S1"])
            f = {(i, j, g, 0): SimpleNamespace(ub=float("inf"))
                 for i, j in data["E"] for g in data["PRODUCTS"]}
            z = {(zone, g, 0): SimpleNamespace(ub=float("inf"))
                 for zone in data["P_"] for g in data["PRODUCTS"]}
            multicommodity_milp.apply_routing_bounds(data, f, z)
            for (i, j, g, day), var in f.items():
                blocked = any(not data["zone_allowed_product"][f"{node}::{g}"]
                              for node in (i, j) if node in data["P_"])
                self.assertEqual(var.ub == 0, blocked)
            self.assertEqual(z["Z1", "S1", 0].ub, 0)
            self.assertEqual(z["Z4", "G1", 0].ub, 0)
            self.assertEqual(z["Z4", "S1", 0].ub, float("inf"))
            del data["arc_ok"]
            with self.assertRaises(KeyError):
                multicommodity_milp.apply_routing_bounds(data, f, z)

    def test_single_model_rejects_restricted_routing(self):
        scenario = generator.build_scenario("F1_system_scale_L", generator.GeneratorConfig(commodity_count=1))
        data = scenario.optimizer_objects()
        key = next(iter(data["product_arc_compatible"]))
        data["product_arc_compatible"][key] = 0
        with self.assertRaisesRegex(ValueError, "unrestricted routing"):
            model_data_loader.validate_canonical_model_ready_data(data, require_single_commodity=True)

    def test_solution_inventory_totals_include_special_product(self):
        import solution_payloads
        data = {"G": ["G1"], "PRODUCTS": ["G1", "S1"], "P_": ["Z4"], "D": [0], "is_multicommodity": True}
        variables = {"z": {("Z4", "G1", 0): SimpleNamespace(X=0.0), ("Z4", "S1", 0): SimpleNamespace(X=17.0)}}
        total = solution_payloads._extract_zone_inventory(data, variables)
        products = solution_payloads._extract_product_inventory(data, variables)
        self.assertEqual(total["Z4"][0]["inventory_tonnes"], 17.0)
        self.assertEqual(products["Z4"]["S1"][0]["inventory_tonnes"], 17.0)

    def test_direct_and_benders_routing_bounds_and_optima_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, path = self.write_instance(tmp, "F1_system_scale_L", 4)
            data = multicommodity_milp.load_scenario(path)
        # One operational day without maintenance isolates the routing LP.
        data.update(D=[0], M=[], M_a={a: [] for a in data["A"]}, mu_min={0: 0})
        try:
            direct, dv = multicommodity_milp.build_model(data)
        except gp.GurobiError as exc:
            self.skipTest(f"Gurobi model construction unavailable: {exc}")
        try:
            sub, sv = benders_multicommodity.build_subproblem(data, {})
            try:
                for name in ("f", "z"):
                    self.assertEqual(set(dv[name]), set(sv[name]))
                    for key in dv[name]:
                        self.assertEqual(dv[name][key].UB, sv[name][key].UB)
                forbidden = next(key for key, var in dv["f"].items() if var.UB == 0 and key[2] == "S1")
                allowed = next(key for key, var in dv["f"].items() if var.UB > 0 and key[2] == "S1")
                for model, variables in ((direct, dv), (sub, sv)):
                    model.Params.OutputFlag = 0
                    model.Params.DualReductions = 0
                    model.addConstr(variables["f"][allowed] >= 1)
                    model.optimize()
                    self.assertEqual(model.Status, gp.GRB.OPTIMAL)
                self.assertAlmostEqual(direct.ObjVal, sub.ObjVal, places=5)
                for model, variables in ((direct, dv), (sub, sv)):
                    model.addConstr(variables["f"][forbidden] >= 1)
                    model.optimize()
                    self.assertEqual(model.Status, gp.GRB.INFEASIBLE)
            finally:
                sub.dispose()
        finally:
            direct.dispose()

    def test_milp_sources_do_not_request_deleted_legacy_fields(self):
        for module in (monolithic_milp, multicommodity_milp):
            source = inspect.getsource(module)
            with self.subTest(module=module.__name__):
                for field_name in FORBIDDEN_SOLVER_FIELD_NAMES:
                    self.assertNotIn(field_name, source)
                self.assertNotIn('["Z"]', source)
                self.assertNotIn("['Z']", source)

    def test_model_ready_schema_is_exact_and_rejects_legacy_aliases(self):
        scenario = generator.build_scenario(
            "F2_mode_tradeoff_M",
            generator.GeneratorConfig(seed=3, commodity_count=4),
        )
        data = scenario.optimizer_objects()
        self.assertEqual(set(data), set(model_data_loader.CANONICAL_MODEL_READY_KEYS))

        mutated = scenario.to_json_dict()["model_ready_data"]
        mutated["Z"] = mutated["P"]
        with self.assertRaisesRegex(ValueError, "canonical schema"):
            model_data_loader.validate_canonical_model_ready_data(mutated)

    def test_builds_single_and_multi_models_for_low_medium_high_instances(self):
        cases = (
            ("F1_system_scale_L", 1),
            ("F2_mode_tradeoff_M", 2),
            ("F3_yard_flow_H", 3),
        )
        with tempfile.TemporaryDirectory() as tmp:
            for scenario_name, seed in cases:
                for commodity_count, module in (
                    (1, monolithic_milp),
                    (4, multicommodity_milp),
                ):
                    with self.subTest(scenario=scenario_name, commodity_count=commodity_count):
                        _scenario, path = self.write_instance(tmp, scenario_name, commodity_count, seed)
                        data = module.load_scenario(path)
                        try:
                            model, variables = module.build_model(data)
                        except gp.GurobiError as exc:
                            self.skipTest(f"Gurobi model construction unavailable: {exc}")
                        try:
                            self.assertGreater(model.NumVars, 0)
                            self.assertGreater(model.NumConstrs, 0)
                            self.assertEqual(model.NumBinVars, len(variables["x"]))
                        finally:
                            model.dispose()

    def test_matched_initial_inventory_totals_are_equal_after_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            _single_scenario, single_path = self.write_instance(tmp, "F4_operational_conflict_H", 1, 7)
            _multi_scenario, multi_path = self.write_instance(tmp, "F4_operational_conflict_H", 4, 7)
            single_data = monolithic_milp.load_scenario(single_path)
            multi_data = multicommodity_milp.load_scenario(multi_path)

            for zone in single_data["P_"]:
                multi_total = sum(multi_data["z0"][(zone, commodity)] for commodity in multi_data["PRODUCTS"])
                self.assertAlmostEqual(single_data["z0"][zone], multi_total, places=3)


if __name__ == "__main__":
    unittest.main()
