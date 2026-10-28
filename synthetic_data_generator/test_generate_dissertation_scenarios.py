import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("generate_dissertation_scenarios.py")
spec = importlib.util.spec_from_file_location("scenario_generator", MODULE_PATH)
assert spec and spec.loader
scenario_generator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = scenario_generator
spec.loader.exec_module(scenario_generator)


class FrozenSchemaGeneratorTests(unittest.TestCase):
    def build(self, scenario_name, seed=1, commodity_count=4):
        return scenario_generator.build_scenario(
            scenario_name,
            scenario_generator.GeneratorConfig(seed=seed, commodity_count=commodity_count),
        )

    def test_scenario_set_is_four_family_design(self):
        self.assertEqual(len(scenario_generator.SCENARIO_NAMES), 12)
        self.assertEqual(
            scenario_generator.SCENARIO_NAMES,
            (
                "F1_system_scale_L",
                "F1_system_scale_M",
                "F1_system_scale_H",
                "F2_mode_tradeoff_L",
                "F2_mode_tradeoff_M",
                "F2_mode_tradeoff_H",
                "F3_yard_flow_L",
                "F3_yard_flow_M",
                "F3_yard_flow_H",
                "F4_operational_conflict_L",
                "F4_operational_conflict_M",
                "F4_operational_conflict_H",
            ),
        )

    def test_all_scenarios_build_and_use_only_canonical_model_ready_keys(self):
        forbidden = scenario_generator.FORBIDDEN_MODEL_READY_KEYS
        canonical = set(scenario_generator.CANONICAL_MODEL_READY_KEYS)
        for name in scenario_generator.SCENARIO_NAMES:
            for commodity_count in (1, 4):
                with self.subTest(name=name, commodity_count=commodity_count):
                    scenario = self.build(name, commodity_count=commodity_count)
                    scenario.validate()
                    data = scenario.optimizer_objects()
                    self.assertEqual(set(data), canonical)
                    self.assertFalse(set(data) & forbidden)
                    self.assertEqual(data["G"], [f"G{i}" for i in range(1, commodity_count + 1)])
                    self.assertEqual(data["K"], ["F", "P30", "P70"])

    def test_special_product_routing_is_exported(self):
        scenario = self.build("F4_operational_conflict_H", commodity_count=4)
        data = scenario.optimizer_objects()
        self.assertEqual(data["SPECIAL_PRODUCTS"], ["S1"])
        self.assertNotIn("X1", json.dumps(scenario.to_json_dict()))
        self.assertEqual(data["zone_allowed_product"][("Z4", "S1")], 1)
        for zone in ("Z1", "Z2", "Z3"):
            self.assertEqual(data["zone_allowed_product"][(zone, "S1")], 0)
        for product in data["G"]:
            self.assertEqual(data["zone_allowed_product"][("Z4", product)], 0)

    def test_special_route_is_complete_at_every_scale(self):
        for level in ("L", "M", "H"):
            data = self.build(f"F1_system_scale_{level}").optimizer_objects()
            route = {arc for arc in data["E"] if data["product_arc_compatible"][("S1", arc)]}
            node = "RAIL_SOURCE"
            visited = []
            used = set()
            while node != "VESSEL_SINK":
                self.assertNotIn(node, visited)
                visited.append(node)
                outgoing = [arc for arc in route if data["arc_from"][arc] == node]
                self.assertEqual(len(outgoing), 1)
                used.add(outgoing[0])
                node = data["arc_to"][outgoing[0]]
            self.assertIn("Z4", visited)
            self.assertEqual(used, route)
            self.assertEqual(len(route), 6)
            for (zone, product), value in data["initial_inventory_tonnes"].items():
                if not data["zone_allowed_product"][(zone, product)]:
                    self.assertEqual(value, 0)

    def test_f1_scale_ladder_has_requested_asset_and_maintenance_counts(self):
        expected = {
            "F1_system_scale_L": (12, 30),
            "F1_system_scale_M": (20, 60),
            "F1_system_scale_H": (32, 100),
        }
        for name, (asset_count, maintenance_count) in expected.items():
            with self.subTest(name=name):
                scenario = self.build(name)
                self.assertEqual(scenario.diagnostics["asset_count"], asset_count)
                self.assertEqual(
                    scenario.diagnostics["maintenance_operation_count"],
                    maintenance_count,
                )
                self.assertEqual(
                    sum(scenario.diagnostics["stage_asset_counts"].values()),
                    asset_count,
                )

    def test_extended_f1_levels_scale_up_and_stay_out_of_the_benchmark(self):
        expected = {
            "F1_system_scale_XL": (48, 160),
            "F1_system_scale_XXL": (64, 250),
        }
        self.assertEqual(tuple(expected), scenario_generator.EXTENDED_SCALE_SCENARIO_NAMES)
        for name, (asset_count, maintenance_count) in expected.items():
            self.assertNotIn(name, scenario_generator.SCENARIO_NAMES)
            for commodity_count in (1, 4):
                with self.subTest(name=name, commodity_count=commodity_count):
                    scenario = self.build(name, seed=2, commodity_count=commodity_count)
                    self.assertEqual(scenario.diagnostics["asset_count"], asset_count)
                    self.assertEqual(scenario.diagnostics["maintenance_operation_count"], maintenance_count)
                    scenario.validate()
                    self.assertEqual(set(scenario.optimizer_objects()), set(scenario_generator.CANONICAL_MODEL_READY_KEYS))
                    for task in scenario.maintenance_tasks:
                        self.assertLessEqual(task.earliest_start_day, task.latest_start_day)
                        self.assertLess(task.mode_durations["F"], task.mode_durations["P30"])
                        self.assertLess(task.mode_durations["P30"], task.mode_durations["P70"])
                        self.assertLessEqual(
                            task.latest_start_day + max(task.mode_durations.values()),
                            scenario.config.horizon_days,
                        )

    def test_non_f1_families_use_reference_medium_topology(self):
        for name in scenario_generator.SCENARIO_NAMES:
            if name.startswith("F1_"):
                continue
            with self.subTest(name=name):
                scenario = self.build(name)
                self.assertEqual(scenario.metadata["system_scale_level"], "M")
                self.assertEqual(scenario.diagnostics["asset_count"], 20)
                self.assertEqual(scenario.diagnostics["maintenance_operation_count"], 60)

    def test_maintenance_is_concentrated_on_bottleneck_assets(self):
        scenario = self.build("F1_system_scale_H", seed=3)
        counts = list(scenario.diagnostics["maintenance_operations_per_asset"].values())
        self.assertGreaterEqual(max(counts), 8)
        self.assertGreaterEqual(sum(1 for value in counts if value >= 3), 8)
        self.assertGreater(
            scenario.diagnostics["maintenance_window_overlap_pairs"],
            len(scenario.maintenance_tasks) * 4,
        )

    def test_mode_tradeoff_centres_avoid_p70_dominance(self):
        scenario = self.build("F2_mode_tradeoff_M", seed=4)
        scores = scenario.diagnostics["mode_effective_outage_days"]
        self.assertAlmostEqual(scores["F"]["mean"], scores["P30"]["mean"], delta=1.2)
        self.assertAlmostEqual(scores["F"]["mean"], scores["P70"]["mean"], delta=1.5)
        for task in scenario.maintenance_tasks:
            self.assertLess(task.mode_durations["F"], task.mode_durations["P30"])
            self.assertLess(task.mode_durations["P30"], task.mode_durations["P70"])

    def test_f2_duration_levels_follow_requested_order(self):
        built = {
            level: self.build(f"F2_mode_tradeoff_{level}", seed=5)
            for level in ("L", "M", "H")
        }
        common_task = built["L"].maintenance_tasks[0].task_id
        durations = {}
        for level, scenario in built.items():
            task = next(task for task in scenario.maintenance_tasks if task.task_id == common_task)
            durations[level] = task.mode_durations
        self.assertLessEqual(durations["L"]["P30"], durations["M"]["P30"])
        self.assertLessEqual(durations["M"]["P30"], durations["H"]["P30"])
        self.assertLess(durations["L"]["P70"], durations["M"]["P70"])
        self.assertLess(durations["M"]["P70"], durations["H"]["P70"])

    def test_f1_f2_and_f4_normalized_throughput_profiles_are_fixed_by_family(self):
        for family in ("F1_system_scale", "F2_mode_tradeoff", "F4_operational_conflict"):
            built = {
                level: self.build(f"{family}_{level}", seed=6)
                for level in ("L", "M", "H")
            }
            ratios = {}
            for level, scenario in built.items():
                q0 = scenario.diagnostics["no_maintenance_throughput_tpd"]
                ratios[level] = [
                    round(value / q0, 6)
                    for value in scenario.minimum_throughput_tpd.values()
                ]
            with self.subTest(family=family):
                for left, right in (("L", "M"), ("M", "H")):
                    max_difference = max(
                        abs(a - b)
                        for a, b in zip(ratios[left], ratios[right])
                    )
                    self.assertLessEqual(max_difference, 1e-5)

    def test_f3_throughput_profiles_are_variable_and_pressure_increases(self):
        low = self.build("F3_yard_flow_L")
        medium = self.build("F3_yard_flow_M")
        high = self.build("F3_yard_flow_H")
        self.assertGreater(
            high.diagnostics["minimum_throughput_ratio_to_q0"]["max"],
            medium.diagnostics["minimum_throughput_ratio_to_q0"]["max"],
        )
        self.assertGreater(
            medium.diagnostics["minimum_throughput_ratio_to_q0"]["max"],
            low.diagnostics["minimum_throughput_ratio_to_q0"]["max"],
        )
        for scenario in (low, medium, high):
            ratio = scenario.diagnostics["minimum_throughput_ratio_to_q0"]
            self.assertGreater(ratio["max"] - ratio["min"], 0.12)
            self.assertLessEqual(ratio["max"], 0.95)

    def test_yard_initial_inventory_respects_bounds_and_f3_is_tighter(self):
        low = self.build("F3_yard_flow_L")
        high = self.build("F3_yard_flow_H")
        self.assertLess(
            sum(zone.capacity_tonnes for zone in high.zones),
            sum(zone.capacity_tonnes for zone in low.zones),
        )
        self.assertLess(
            max(high.diagnostics["initial_yard_fill_after_min_ratio"].values()),
            max(low.diagnostics["initial_yard_fill_after_min_ratio"].values()),
        )
        self.assertLess(
            high.diagnostics["average_usable_storage_days"],
            low.diagnostics["average_usable_storage_days"],
        )
        self.assertGreater(
            high.diagnostics["fixed_capacity_reduction_severity"],
            low.diagnostics["fixed_capacity_reduction_severity"],
        )
        for scenario in (low, high):
            data = scenario.optimizer_objects()
            for zone in data["P"]:
                total = sum(
                    value
                    for key, value in data["initial_inventory_tonnes"].items()
                    if key[0] == zone
                )
                self.assertGreaterEqual(total, data["zone_min_inventory_tonnes"][zone])
                self.assertLessEqual(total, data["zone_max_inventory_tonnes"][zone])

    def test_maintenance_windows_and_mode_durations_are_valid(self):
        for name in scenario_generator.SCENARIO_NAMES:
            scenario = self.build(name, seed=2)
            horizon = scenario.config.horizon_days
            for task in scenario.maintenance_tasks:
                with self.subTest(name=name, task=task.task_id):
                    self.assertLessEqual(task.earliest_start_day, task.latest_start_day)
                    self.assertGreaterEqual(task.latest_start_day - task.earliest_start_day + 1, 1)
                    self.assertGreaterEqual(task.mode_durations["F"], 1)
                    self.assertGreater(task.mode_durations["P30"], task.mode_durations["F"])
                    self.assertGreater(task.mode_durations["P70"], task.mode_durations["P30"])
                    self.assertLessEqual(
                        task.latest_start_day + max(task.mode_durations.values()),
                        horizon,
                    )

    def test_f4_fixed_activity_conflict_increases_by_level(self):
        built = {level: self.build(f"F4_operational_conflict_{level}") for level in ("L", "M", "H")}
        self.assertEqual(built["L"].diagnostics["fixed_activity_window_count"], 2)
        self.assertEqual(built["M"].diagnostics["fixed_activity_window_count"], 4)
        self.assertEqual(built["H"].diagnostics["fixed_activity_window_count"], 6)
        self.assertLess(
            built["L"].diagnostics["fixed_activity_incompatible_asset_days"],
            built["M"].diagnostics["fixed_activity_incompatible_asset_days"],
        )
        self.assertLess(
            built["M"].diagnostics["fixed_activity_incompatible_asset_days"],
            built["H"].diagnostics["fixed_activity_incompatible_asset_days"],
        )
        self.assertGreater(len(built["H"].optimizer_objects()["special_activity_capacity_factor"]), 0)
        self.assertGreater(len(built["H"].optimizer_objects()["maintenance_activity_incompatible"]), 0)

    def test_single_and_multi_instances_are_matched_except_commodity_count(self):
        for name in scenario_generator.SCENARIO_NAMES:
            with self.subTest(name=name):
                single = self.build(name, seed=7, commodity_count=1)
                multi = self.build(name, seed=7, commodity_count=4)
                single_data = single.optimizer_objects()
                multi_data = multi.optimizer_objects()
                for key in scenario_generator.CANONICAL_MODEL_READY_KEYS:
                    if key in {"G", "initial_inventory_tonnes", "SPECIAL_PRODUCTS", "product_arc_compatible", "zone_allowed_product", "product_zone_compatible"}:
                        continue
                    self.assertEqual(single_data[key], multi_data[key])
                for zone in single_data["P"]:
                    single_total = sum(
                        value
                        for key, value in single_data["initial_inventory_tonnes"].items()
                        if key[0] == zone
                    )
                    multi_total = sum(
                        value
                        for key, value in multi_data["initial_inventory_tonnes"].items()
                        if key[0] == zone
                    )
                    self.assertAlmostEqual(single_total, multi_total, places=3)

    def test_generation_is_deterministic_and_seeded(self):
        first = self.build("F4_operational_conflict_H", seed=8)
        second = self.build("F4_operational_conflict_H", seed=8)
        third = self.build("F4_operational_conflict_H", seed=9)
        self.assertEqual(first.optimizer_objects(), second.optimizer_objects())
        self.assertEqual(first.to_json_dict(), second.to_json_dict())
        self.assertNotEqual(first.optimizer_objects(), third.optimizer_objects())

    def test_generate_benchmark_suite_writes_120_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            generated = scenario_generator.generate_benchmark_suite(
                tmp,
                config=scenario_generator.GeneratorConfig(commodity_count=1),
            )
            instance_keys = [key for key in generated if key != "benchmark_manifest"]
            self.assertEqual(len(instance_keys), 120)
            manifest = json.loads(Path(generated["benchmark_manifest"]).read_text())
            self.assertEqual(manifest["instance_count"], 120)
            self.assertEqual(manifest["expected_instance_count"], 120)
            self.assertEqual(manifest["commodity_count"], 1)
            self.assertEqual(
                set(manifest["canonical_model_ready_keys"]),
                set(scenario_generator.CANONICAL_MODEL_READY_KEYS),
            )
            self.assertTrue((Path(tmp) / "benchmark_validation.csv").exists())
            self.assertTrue((Path(tmp) / "family_validation.csv").exists())
            self.assertTrue((Path(tmp) / "seed_diversity.csv").exists())


if __name__ == "__main__":
    unittest.main()
