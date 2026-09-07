import inspect
import tempfile
import unittest
from pathlib import Path

import gurobipy as gp

import model_data_loader
import monolithic_milp
import multicommodity_milp
from synthetic_data_generator import generate_dissertation_scenarios as generator


FORBIDDEN_SOLVER_FIELD_NAMES = (
    "product_arc_compatible",
    "zone_allowed_product",
    "product_zone_compatible",
    "product_asset_compatible",
    "arc_allowed_product",
    "SPECIAL_PRODUCTS",
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
                multi_total = sum(multi_data["z0"][(zone, commodity)] for commodity in multi_data["G"])
                self.assertAlmostEqual(single_data["z0"][zone], multi_total, places=3)


if __name__ == "__main__":
    unittest.main()
