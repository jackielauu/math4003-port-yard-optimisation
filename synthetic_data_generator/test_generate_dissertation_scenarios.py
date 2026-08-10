import importlib.util
import json
import tempfile
import unittest
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("generate_dissertation_scenarios.py")
spec = importlib.util.spec_from_file_location("scenario_generator", MODULE_PATH)
assert spec and spec.loader
scenario_generator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = scenario_generator
spec.loader.exec_module(scenario_generator)


class DissertationScenarioGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = scenario_generator.GeneratorConfig(horizon_days=90, seed=42)

    def test_all_occasion_names_build_and_validate(self) -> None:
        for name in scenario_generator.OCCASION_NAMES:
            with self.subTest(name=name):
                scenario = scenario_generator.build_scenario(name, self.config)
                scenario.validate()
                self.assertEqual(scenario.metadata["scenario_name"], name)
                self.assertEqual(len(scenario.days), 90)
                self.assertTrue(
                    scenario.design_evidence["occasion_checks"]["all_checks_pass"]
                )

    def test_generator_is_deterministic_for_same_seed(self) -> None:
        first = scenario_generator.build_scenario("peak_export_conflict", self.config)
        second = scenario_generator.build_scenario("peak_export_conflict", self.config)
        self.assertEqual(first.to_json_dict(), second.to_json_dict())

    def test_different_seed_changes_generated_profiles(self) -> None:
        first = scenario_generator.build_scenario(
            "balanced_control",
            scenario_generator.GeneratorConfig(seed=1),
        )
        second = scenario_generator.build_scenario(
            "balanced_control",
            scenario_generator.GeneratorConfig(seed=2),
        )
        self.assertNotEqual(first.inbound_supply, second.inbound_supply)
        self.assertNotEqual(first.outbound_demand, second.outbound_demand)

    def test_optimizer_objects_have_complete_core_sets(self) -> None:
        scenario = scenario_generator.build_scenario("balanced_control", self.config)
        data = scenario.optimizer_objects()
        self.assertEqual(len(data["T"]), 90)
        self.assertEqual(len(data["P"]), 4)
        self.assertEqual(len(data["A"]), 12)
        self.assertEqual(len(data["Z"]), 4)
        self.assertEqual(len(data["J"]), 12)
        self.assertIn(("P1", 0), data["supply_tonnes"])
        self.assertIn(("P1", 0), data["demand_tonnes"])
        self.assertIn(("Z1", "P1"), data["initial_inventory_tonnes"])
        # Rate-based model checks
        self.assertIn("asset_rate_tph", data)
        self.assertIn("arc_rate_tph", data)
        self.assertIn("minimum_throughput_rate_tph", data)
        self.assertIn("operating_hours_per_day", data)
        self.assertEqual(data["operating_hours_per_day"], 24.0)
        # Product metadata checks
        self.assertIn("product_name", data)
        self.assertIn("product_share_of_volume", data)
        self.assertEqual(data["SPECIAL_PRODUCTS"], ["X1"])
        self.assertEqual(data["product_is_special"]["X1"], 1)
        self.assertEqual(data["product_blending_allowed"]["X1"], 0)
        # Verify removed fields are absent
        self.assertNotIn("product_commercial_weight", data)
        self.assertNotIn("product_priority_eligible", data)
        # Verify product shares sum to 1.0
        total_share = sum(data["product_share_of_volume"].values())
        self.assertAlmostEqual(total_share, 1.0, places=6)
        # Grade specification checks
        self.assertIn("product_grade_specs", data)
        self.assertEqual(len(data["product_grade_specs"]), 4)
        for product_id in data["P"]:
            self.assertIn(product_id, data["product_grade_specs"])
            grade_specs = data["product_grade_specs"][product_id]
            # Each product should have chemical grade elements
            self.assertIn("FE", grade_specs)
            self.assertIn("SIO2", grade_specs)
            self.assertIn("AL2O3", grade_specs)
            # Each element should have target, min, max, unit
            fe_spec = grade_specs["FE"]
            self.assertIn("target", fe_spec)
            self.assertIn("min", fe_spec)
            self.assertIn("max", fe_spec)
            self.assertIn("unit", fe_spec)
            # FE target should be in reasonable iron ore range (55-62%)
            self.assertGreater(fe_spec["target"], 55.0)
            self.assertLess(fe_spec["target"], 62.0)
        self.assertNotIn("reference_schedule", data)
        self.assertNotIn("design_evidence", data)

    def test_special_product_identity_is_first_class(self) -> None:
        scenario = scenario_generator.build_scenario("balanced_control", self.config)
        special = [product for product in scenario.products if product.is_special]
        self.assertEqual(len(special), 1)
        self.assertEqual(special[0].product_id, "X1")
        self.assertEqual(special[0].name, "Special Product X1")
        self.assertFalse(special[0].blending_allowed)
        self.assertTrue(all(not p.is_special for p in scenario.products if p.product_id != "X1"))
        data = scenario.optimizer_objects()
        self.assertEqual(data["SPECIAL_PRODUCTS"], ["X1"])
        self.assertEqual(data["product_is_special"], {"P1": 0, "P2": 0, "P3": 0, "X1": 1})

    def test_special_product_requirements_and_segregation_are_exported(self) -> None:
        scenario = scenario_generator.build_scenario("balanced_control", self.config)
        requirement = scenario.special_product_requirements[0]
        data = scenario.optimizer_objects()
        self.assertEqual(requirement.product_id, "X1")
        self.assertGreater(requirement.tonnes_per_square_metre, 0)
        self.assertGreater(requirement.minimum_reserved_footprint_m2, 0)
        self.assertEqual(requirement.compatible_zones, ("Z4",))
        self.assertEqual(data["special_product_compatible_zones"]["X1"], ["Z4"])
        self.assertEqual(data["product_zone_compatible"][("X1", "Z4")], 1)
        self.assertEqual(data["product_zone_compatible"][("X1", "Z3")], 0)
        for ordinary in ("P1", "P2", "P3"):
            self.assertEqual(data["product_may_share_zone"][("X1", ordinary)], 0)
            self.assertEqual(data["product_may_share_zone"][(ordinary, "X1")], 0)
        self.assertEqual(data["product_may_share_zone"][("X1", "X1")], 1)

    def test_pdf_maintenance_modes_and_duration_tradeoff(self) -> None:
        scenario = scenario_generator.build_scenario("balanced_control", self.config)
        expected_factors = {
            scenario_generator.FULL_SHUTDOWN_MODE: 0.0,
            scenario_generator.PARTIAL_30_MODE: 0.30,
            scenario_generator.PARTIAL_70_MODE: 0.70,
        }
        for task in scenario.maintenance_tasks:
            with self.subTest(task=task.task_id):
                mode_by_id = {mode.mode_id: mode for mode in task.modes}
                self.assertEqual(set(mode_by_id), set(expected_factors))
                for mode_id, factor in expected_factors.items():
                    self.assertAlmostEqual(
                        mode_by_id[mode_id].capacity_factor_during_work,
                        factor,
                    )
                self.assertLess(
                    mode_by_id[scenario_generator.FULL_SHUTDOWN_MODE].duration_days,
                    mode_by_id[scenario_generator.PARTIAL_30_MODE].duration_days,
                )
                self.assertLess(
                    mode_by_id[scenario_generator.PARTIAL_30_MODE].duration_days,
                    mode_by_id[scenario_generator.PARTIAL_70_MODE].duration_days,
                )

    def test_optimizer_objects_include_current_pdf_notation(self) -> None:
        scenario = scenario_generator.build_scenario("balanced_control", self.config)
        data = scenario.optimizer_objects()
        for key in (
            "D",
            "N",
            "E",
            "A",
            "A_NS",
            "P_STOCKYARD",
            "G",
            "M_OPERATIONS",
            "M_a",
            "H",
            "W_m",
            "W_h",
            "K",
            "predecessors",
            "successors",
        ):
            self.assertIn(key, data)
        self.assertEqual(data["K"], list(scenario_generator.MAINTENANCE_MODE_ORDER))
        self.assertEqual(data["K_symbol"][scenario_generator.PARTIAL_30_MODE], "P30")
        self.assertIn("asset_capacity_tpd", data)
        self.assertIn("arc_capacity_tpd", data)
        self.assertIn("zone_min_inventory_tonnes", data)
        self.assertIn("zone_max_inventory_tonnes", data)
        self.assertIn("overflow_penalty_per_tonne", data)
        self.assertIn("minimum_throughput_tpd", data)
        self.assertAlmostEqual(
            data["asset_capacity_tpd"]["RC1"],
            self.config.reclaimer_rate_tph * self.config.operating_hours_per_day,
        )
        mode_key = ("M_RC2", scenario_generator.PARTIAL_30_MODE)
        self.assertAlmostEqual(
            data["maintenance_available_capacity_fraction"][mode_key], 0.30
        )
        self.assertAlmostEqual(
            data["maintenance_capacity_loss_fraction"][mode_key], 0.70
        )

    def test_priority_activity_exports_sigma_and_incompatibility_inputs(self) -> None:
        scenario = scenario_generator.build_scenario(
            "priority_campaign_conflict", self.config
        )
        data = scenario.optimizer_objects()
        activity = scenario.priority_activities[0]
        self.assertEqual(data["H"], [activity.activity_id])
        self.assertEqual(
            data["special_activity_duration_days"][activity.activity_id],
            activity.end_day - activity.start_day,
        )
        self.assertEqual(
            data["W_h"][activity.activity_id],
            [activity.start_day],
        )
        self.assertEqual(data["special_activity_product"][activity.activity_id], "X1")
        self.assertEqual(data["special_activity_required_zone"][activity.activity_id], "Z4")
        self.assertEqual(
            data["special_activity_minimum_total_tonnes"][activity.activity_id],
            activity.minimum_total_tonnes,
        )
        self.assertEqual(
            data["special_product_campaign_minimum_tonnes"][activity.activity_id],
            activity.minimum_total_tonnes,
        )
        for asset_id in activity.required_assets:
            for day in range(
                activity.start_day - activity.pre_campaign_buffer_days,
                activity.end_day + activity.post_campaign_buffer_days,
            ):
                self.assertEqual(
                    data["maintenance_activity_incompatible"][(asset_id, day)],
                    1,
                )
            for day in range(activity.start_day, activity.end_day):
                self.assertEqual(
                    data["maintenance_activity_incompatible"][(asset_id, day)],
                    1,
                )
                self.assertAlmostEqual(
                    data["special_activity_capacity_factor"][(asset_id, day)],
                    activity.capacity_factor_during_activity,
                )
        for day in range(activity.start_day, activity.end_day):
            self.assertEqual(
                data["special_product_minimum_tpd"][("X1", day)],
                activity.minimum_tonnes_per_day,
            )

    def test_special_product_route_is_connected_for_two_and_one_shiploaders(self) -> None:
        for config in (
            self.config,
            scenario_generator.GeneratorConfig(
                horizon_days=90,
                seed=42,
                shiploader_count=1,
                shiploader_rate_tph=9166.0,
            ),
        ):
            with self.subTest(shiploader_count=config.shiploader_count):
                scenario = scenario_generator.build_scenario("balanced_control", config)
                data = scenario.optimizer_objects()
                route = data["special_product_compatible_arcs"]["X1"]
                assets = set(data["A"])
                for asset in data["special_product_required_assets"]["X1"]:
                    self.assertIn(asset, assets)
                    self.assertEqual(data["product_asset_compatible"][("X1", asset)], 1)
                self.assertEqual(data["arc_from"][route[0]], "RAIL_SOURCE")
                self.assertEqual(data["arc_to"][route[-1]], "VESSEL_SINK")
                for left, right in zip(route, route[1:]):
                    self.assertEqual(data["arc_to"][left], data["arc_from"][right])
                self.assertIn("Z4", {data["arc_from"][arc] for arc in route} | {data["arc_to"][arc] for arc in route})
                self.assertTrue(any(data["arc_to"][arc].startswith("SL") for arc in route))
                if config.shiploader_count == 1:
                    self.assertFalse(any("SL2" in item for item in route))
                    self.assertNotIn("SL2", data["special_product_required_assets"]["X1"])

    def test_single_shiploader_topology_matches_pdf_case_study_option(self) -> None:
        config = scenario_generator.GeneratorConfig(
            horizon_days=90,
            seed=42,
            shiploader_count=1,
            shiploader_rate_tph=9166.0,
        )
        for name in scenario_generator.OCCASION_NAMES:
            with self.subTest(name=name):
                scenario = scenario_generator.build_scenario(name, config)
                asset_ids = {asset.asset_id for asset in scenario.assets}
                task_ids = {task.task_id for task in scenario.maintenance_tasks}
                self.assertIn("SL1", asset_ids)
                self.assertNotIn("SL2", asset_ids)
                self.assertIn("M_SL1", task_ids)
                self.assertNotIn("M_SL2", task_ids)
                self.assertEqual(len(scenario.assets), 11)
                self.assertEqual(len(scenario.maintenance_tasks), 11)
                self.assertTrue(
                    scenario.design_evidence["occasion_checks"]["all_checks_pass"]
                )
                scenario.validate()
        campaign = scenario_generator.build_scenario(
            "priority_campaign_conflict", config
        )
        self.assertIn("SL1", campaign.priority_activities[0].required_assets)

    def test_control_case_is_feasible_and_unchanged(self) -> None:
        scenario = scenario_generator.build_scenario("balanced_control", self.config)
        evidence = scenario.design_evidence
        self.assertEqual(scenario.manual_baseline, scenario.reference_schedule)
        self.assertEqual(
            evidence["manual_baseline_metrics"]["minimum_throughput_violation_days"],
            0,
        )
        self.assertEqual(
            evidence["manual_baseline_metrics"],
            evidence["reference_schedule_metrics"],
        )

    def test_peak_conflict_has_a_better_reference_schedule(self) -> None:
        scenario = scenario_generator.build_scenario(
            "peak_export_conflict", self.config
        )
        baseline = scenario.design_evidence["manual_baseline_metrics"]
        reference = scenario.design_evidence["reference_schedule_metrics"]
        self.assertGreater(
            reference["total_throughput_tonnes"], baseline["total_throughput_tonnes"]
        )
        self.assertLess(
            reference["unmet_demand_tonnes"], baseline["unmet_demand_tonnes"]
        )
        self.assertEqual(reference["minimum_throughput_violation_days"], 0)
        self.assertGreater(reference["objective_proxy"], baseline["objective_proxy"])

    def test_overlapping_and_tight_windows_create_feasibility_value(self) -> None:
        for name in ("overlapping_maintenance", "tight_lifecycle_windows"):
            with self.subTest(name=name):
                scenario = scenario_generator.build_scenario(name, self.config)
                baseline = scenario.design_evidence["manual_baseline_metrics"]
                reference = scenario.design_evidence["reference_schedule_metrics"]
                self.assertGreater(baseline["minimum_throughput_violation_days"], 0)
                self.assertEqual(reference["minimum_throughput_violation_days"], 0)
                self.assertGreater(
                    reference["objective_proxy"], baseline["objective_proxy"]
                )

    def test_priority_campaign_is_blocked_only_by_baseline(self) -> None:
        scenario = scenario_generator.build_scenario(
            "priority_campaign_conflict", self.config
        )
        baseline = scenario.design_evidence["manual_baseline_metrics"]
        reference = scenario.design_evidence["reference_schedule_metrics"]
        self.assertGreater(baseline["priority_shortfall_tonnes"], 0)
        self.assertGreater(baseline["special_product_shortfall_tonnes"], 0)
        self.assertGreater(
            baseline["special_product_campaign_volume_shortfall_tonnes"], 0
        )
        self.assertGreater(baseline["total_throughput_tonnes"], 252_000.0)
        self.assertEqual(reference["priority_shortfall_tonnes"], 0)
        self.assertEqual(reference["special_product_shortfall_tonnes"], 0)
        self.assertEqual(
            reference["special_product_campaign_volume_shortfall_tonnes"], 0
        )
        self.assertGreaterEqual(
            reference["special_product_campaign_volume_tonnes"], 252_000.0
        )
        self.assertEqual(reference["special_product_maintenance_conflict_days"], 0)
        self.assertGreater(reference["objective_proxy"], baseline["objective_proxy"])

    def test_special_product_footprint_is_checked(self) -> None:
        scenario = scenario_generator.build_scenario(
            "priority_campaign_conflict", self.config
        )
        data = scenario.optimizer_objects()
        self.assertIn("zone_footprint_capacity_m2", data)
        reference = scenario.design_evidence["reference_schedule_metrics"]
        self.assertGreater(reference["special_product_maximum_footprint_m2"], 0)
        self.assertEqual(reference["special_product_footprint_violation_days"], 0)

        oversized = scenario_generator.build_scenario("balanced_control", self.config)
        oversized.initial_inventory["Z4"]["X1"] = 540_000.0
        oversized_metrics = scenario_generator.evaluate_schedule(
            oversized, oversized.manual_baseline
        )
        self.assertGreater(
            oversized_metrics["special_product_footprint_violation_days"], 0
        )
        self.assertGreater(oversized_metrics["special_product_footprint_excess_m2"], 0)

    def test_incompatible_special_product_zone_use_fails_validation(self) -> None:
        scenario = scenario_generator.build_scenario("balanced_control", self.config)
        scenario.initial_inventory["Z3"]["X1"] = 1_000.0
        with self.assertRaisesRegex(ValueError, "incompatible with zone Z3"):
            scenario.validate()

    def test_new_model_ready_keys_and_json_serialization(self) -> None:
        scenario = scenario_generator.build_scenario(
            "priority_campaign_conflict", self.config
        )
        data = scenario.optimizer_objects()
        for key in (
            "SPECIAL_PRODUCTS",
            "product_is_special",
            "product_blending_allowed",
            "special_product_requirements",
            "special_product_tonnes_per_square_metre",
            "special_product_minimum_reserved_footprint_m2",
            "special_product_segregation_buffer_m2",
            "special_product_required_assets",
            "special_product_compatible_arcs",
            "special_product_compatible_zones",
            "zone_footprint_capacity_m2",
            "product_zone_compatible",
            "product_asset_compatible",
            "product_arc_compatible",
            "product_may_share_zone",
            "special_activity_product",
            "special_activity_required_zone",
            "special_activity_minimum_tpd",
            "special_activity_minimum_total_tonnes",
            "special_activity_pre_buffer_days",
            "special_activity_post_buffer_days",
            "special_activity_cleanout_before_days",
            "special_activity_cleanout_after_days",
            "special_product_minimum_tpd",
            "special_product_campaign_minimum_tonnes",
        ):
            self.assertIn(key, data)
        self.assertNotIn("reference_schedule", data)
        self.assertNotIn("design_evidence", data)
        payload = scenario.to_json_dict(include_reference_schedule=False)
        self.assertIn("model_ready_data", payload)
        self.assertNotIn("reference_schedule", payload)
        self.assertNotIn("design_evidence", payload)
        json.dumps(payload)

    def test_storage_pressure_is_created_and_removed(self) -> None:
        scenario = scenario_generator.build_scenario("yard_overflow_risk", self.config)
        baseline = scenario.design_evidence["manual_baseline_metrics"]
        reference = scenario.design_evidence["reference_schedule_metrics"]
        self.assertGreater(baseline["storage_pressure_days"], 0)
        self.assertEqual(reference["storage_pressure_days"], 0)
        self.assertLess(
            reference["unaccepted_supply_tonnes"], baseline["unaccepted_supply_tonnes"]
        )

    def test_inventory_starvation_reference_improves_throughput(self) -> None:
        scenario = scenario_generator.build_scenario(
            "inventory_starvation_risk", self.config
        )
        baseline = scenario.design_evidence["manual_baseline_metrics"]
        reference = scenario.design_evidence["reference_schedule_metrics"]
        self.assertGreater(
            reference["total_throughput_tonnes"], baseline["total_throughput_tonnes"]
        )
        self.assertGreater(reference["objective_proxy"], baseline["objective_proxy"])

    def test_partial_shutdown_mode_preserves_minimum_throughput(self) -> None:
        scenario = scenario_generator.build_scenario(
            "partial_shutdown_choice", self.config
        )
        baseline = {d.task_id: d for d in scenario.manual_baseline}
        reference = {d.task_id: d for d in scenario.reference_schedule}
        self.assertEqual(baseline["M_RC2"].mode_id, "full_shutdown")
        self.assertEqual(
            reference["M_RC2"].mode_id, scenario_generator.PARTIAL_70_MODE
        )
        baseline_metrics = scenario.design_evidence["manual_baseline_metrics"]
        reference_metrics = scenario.design_evidence["reference_schedule_metrics"]
        self.assertGreater(baseline_metrics["minimum_throughput_violation_days"], 0)
        self.assertEqual(reference_metrics["minimum_throughput_violation_days"], 0)

    def test_fixed_schedule_is_a_negative_control(self) -> None:
        scenario = scenario_generator.build_scenario(
            "fixed_schedule_control", self.config
        )
        baseline = {d.task_id: d for d in scenario.manual_baseline}
        self.assertEqual(scenario.manual_baseline, scenario.reference_schedule)
        for task in scenario.maintenance_tasks:
            self.assertEqual(task.earliest_start_day, task.latest_start_day)
            self.assertEqual(task.earliest_start_day, baseline[task.task_id].start_day)
        self.assertEqual(
            scenario.design_evidence["manual_baseline_metrics"],
            scenario.design_evidence["reference_schedule_metrics"],
        )

    def test_structural_infeasibility_cannot_be_scheduled_away(self) -> None:
        scenario = scenario_generator.build_scenario(
            "structurally_infeasible", self.config
        )
        self.assertEqual(scenario.metadata["expected_feasibility"], "infeasible")
        baseline = scenario.design_evidence["manual_baseline_metrics"]
        reference = scenario.design_evidence["reference_schedule_metrics"]
        self.assertGreater(baseline["minimum_throughput_violation_days"], 0)
        self.assertGreater(reference["minimum_throughput_violation_days"], 0)

    def test_generate_suite_writes_manifest_and_all_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            generated = scenario_generator.generate_suite(
                temporary_directory,
                self.config,
            )
            self.assertEqual(
                set(generated), set(scenario_generator.OCCASION_NAMES) | {"manifest"}
            )
            manifest = json.loads(Path(generated["manifest"]).read_text())
            self.assertEqual(
                manifest["scenario_count"], len(scenario_generator.OCCASION_NAMES)
            )
            self.assertTrue(
                all(case["all_design_checks_pass"] for case in manifest["cases"])
            )


if __name__ == "__main__":
    unittest.main()

