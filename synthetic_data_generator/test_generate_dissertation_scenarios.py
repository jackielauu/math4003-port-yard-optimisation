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


class DissertationScenarioGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = scenario_generator.GeneratorConfig(horizon_days=90, seed=42)

    def maintenance_signature(self, scenario, include_modes: bool = False):
        signature = {}
        for task in scenario.maintenance_tasks:
            values = {
                "asset_id": task.asset_id,
                "earliest_start_day": task.earliest_start_day,
                "latest_start_day": task.latest_start_day,
                "lifecycle_due_day": task.lifecycle_due_day,
                "full_duration_days": next(
                    mode.duration_days
                    for mode in task.modes
                    if mode.mode_id == scenario_generator.FULL_SHUTDOWN_MODE
                ),
                "task_class": task.task_class,
            }
            if include_modes:
                values["modes"] = tuple(
                    (mode.mode_id, mode.duration_days)
                    for mode in task.modes
                )
            signature[task.task_id] = values
        return signature

    def stage_counts(self, scenario):
        return {
            stage: sum(1 for asset in scenario.assets if asset.stage == stage)
            for stage in scenario_generator.STAGES
        }

    def arc_signature(self, scenario):
        return {
            arc.arc_id: (arc.capacity_tph, arc.allowed_products)
            for arc in scenario.arcs
        }

    def reachable_nodes(self, scenario, start_node: str) -> set[str]:
        adjacency = {}
        for arc in scenario.arcs:
            adjacency.setdefault(arc.from_node, set()).add(arc.to_node)
        seen = {start_node}
        stack = [start_node]
        while stack:
            node = stack.pop()
            for successor in adjacency.get(node, set()):
                if successor not in seen:
                    seen.add(successor)
                    stack.append(successor)
        return seen

    def total_by_day(self, records):
        totals = {}
        for record in records:
            totals[record.day] = totals.get(record.day, 0.0) + record.tonnes
        return totals

    def average_total(self, records):
        totals = self.total_by_day(records)
        return sum(totals.values()) / len(totals)

    def test_suite_names_are_new_four_family_design(self) -> None:
        self.assertEqual(scenario_generator.BASE_CONTROL_NAME, "base_control")
        self.assertEqual(len(scenario_generator.EXPERIMENTAL_SCENARIO_NAMES), 12)
        self.assertEqual(
            scenario_generator.EXPERIMENTAL_SCENARIO_NAMES,
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
        self.assertIn("structurally_infeasible", scenario_generator.SCENARIO_NAMES)

    def test_all_scenarios_build_validate_and_pass_design_checks(self) -> None:
        for name in scenario_generator.SCENARIO_NAMES:
            with self.subTest(name=name):
                scenario = scenario_generator.build_scenario(name, self.config)
                scenario.validate()
                self.assertEqual(scenario.metadata["scenario_name"], name)
                self.assertEqual(len(scenario.days), 90)
                self.assertTrue(
                    scenario.design_evidence["occasion_checks"]["all_checks_pass"]
                )

    def test_base_control_yard_and_inventory_calibration(self) -> None:
        scenario = scenario_generator.build_scenario("base_control", self.config)
        zone_by_id = {zone.zone_id: zone for zone in scenario.zones}
        self.assertEqual(zone_by_id["Z1"].capacity_tonnes, 650_000.0)
        self.assertEqual(zone_by_id["Z2"].capacity_tonnes, 750_000.0)
        self.assertEqual(zone_by_id["Z3"].capacity_tonnes, 580_000.0)
        self.assertEqual(zone_by_id["Z4"].capacity_tonnes, 500_000.0)

        target_utilisation = scenario_generator.get_zone_initial_inventory_ratios(
            "base_control"
        )
        for zone_id, target in target_utilisation.items():
            total = sum(scenario.initial_inventory[zone_id].values())
            zone = zone_by_id[zone_id]
            self.assertGreaterEqual(total, zone.minimum_inventory_tonnes)
            self.assertLess(total, zone.capacity_tonnes)
            self.assertLessEqual(
                abs(total / zone.capacity_tonnes - target),
                scenario_generator.INITIAL_INVENTORY_FILL_RATIO_JITTER + 1e-6,
            )
            self.assertTrue(
                set(scenario.initial_inventory[zone_id]).issubset(
                    set(zone.allowed_products)
                )
            )
            self.assertGreaterEqual(
                zone.minimum_inventory_tonnes / zone.capacity_tonnes,
                0.05,
            )
            self.assertLessEqual(
                zone.minimum_inventory_tonnes / zone.capacity_tonnes,
                0.08,
            )
        self.assertEqual(set(scenario.initial_inventory["Z4"]), {"X1"})

    def test_active_model_ready_inputs_are_exported(self) -> None:
        scenario = scenario_generator.build_scenario("base_control", self.config)
        data = scenario.optimizer_objects()
        for key in (
            "D",
            "A",
            "Z",
            "G",
            "M_OPERATIONS",
            "M_a",
            "K",
            "maintenance_duration_days",
            "maintenance_available_capacity_fraction",
            "maintenance_earliest_start",
            "maintenance_latest_start",
            "maintenance_activity_incompatible",
            "special_activity_capacity_factor",
            "initial_inventory_tonnes",
            "zone_min_inventory_tonnes",
            "zone_max_inventory_tonnes",
            "asset_capacity_tpd",
            "arc_capacity_tpd",
            "minimum_throughput_tpd",
            "product_arc_compatible",
            "zone_allowed_product",
        ):
            self.assertIn(key, data)
        self.assertEqual(data["K"], list(scenario_generator.MAINTENANCE_MODE_ORDER))
        self.assertNotIn("reference_schedule", data)
        self.assertNotIn("design_evidence", data)

    def test_base_maintenance_windows_are_spread_and_varied(self) -> None:
        scenario = scenario_generator.build_scenario("base_control", self.config)
        starts = [task.earliest_start_day for task in scenario.maintenance_tasks]
        widths = [
            task.latest_start_day - task.earliest_start_day + 1
            for task in scenario.maintenance_tasks
        ]
        self.assertGreaterEqual(min(starts), 0)
        self.assertGreater(max(starts) - min(starts), 45)
        self.assertLessEqual(min(widths), 10)
        self.assertGreaterEqual(max(widths), 20)
        for task in scenario.maintenance_tasks:
            max_duration = max(mode.duration_days for mode in task.modes)
            self.assertLessEqual(task.latest_start_day, 90 - max_duration)

    def test_mode_duration_capacity_tradeoff(self) -> None:
        scenario = scenario_generator.build_scenario("base_control", self.config)
        task = max(
            scenario.maintenance_tasks,
            key=lambda item: next(
                mode.duration_days
                for mode in item.modes
                if mode.mode_id == scenario_generator.FULL_SHUTDOWN_MODE
            ),
        )
        mode_by_id = {mode.mode_id: mode for mode in task.modes}
        full_duration = mode_by_id["full_shutdown"].duration_days
        self.assertEqual(
            mode_by_id["partial_30"].duration_days,
            max(
                full_duration + 1,
                int(full_duration * scenario_generator.BASE_MODE_DURATION_RATIOS["partial_30"] + 0.999999),
            ),
        )
        self.assertGreater(
            mode_by_id["partial_70"].duration_days,
            mode_by_id["partial_30"].duration_days,
        )
        self.assertEqual(mode_by_id["full_shutdown"].capacity_factor_during_work, 0.0)
        self.assertEqual(mode_by_id["partial_30"].capacity_factor_during_work, 0.30)
        self.assertEqual(mode_by_id["partial_70"].capacity_factor_during_work, 0.70)

    def test_system_scale_changes_topology_and_operation_count(self) -> None:
        expected = {
            "F1_system_scale_L": (
                {"train_unloader": 2, "stacker": 3, "reclaimer": 2, "shiploader": 1},
                8,
                8,
            ),
            "F1_system_scale_M": (
                {"train_unloader": 3, "stacker": 4, "reclaimer": 3, "shiploader": 2},
                12,
                12,
            ),
            "F1_system_scale_H": (
                {"train_unloader": 4, "stacker": 5, "reclaimer": 4, "shiploader": 3},
                16,
                16,
            ),
        }
        for name, (stage_counts, asset_count, task_count) in expected.items():
            with self.subTest(name=name):
                scenario = scenario_generator.build_scenario(name, self.config)
                data = scenario.optimizer_objects()
                self.assertEqual(self.stage_counts(scenario), stage_counts)
                self.assertEqual(len(scenario.assets), asset_count)
                self.assertEqual(len(scenario.maintenance_tasks), task_count)
                self.assertEqual(data["K"], list(scenario_generator.MAINTENANCE_MODE_ORDER))
                self.assertNotIn("partial_50", json.dumps(scenario.to_json_dict()))
                self.assertEqual(scenario.metadata["scenario_family"], "F1_system_scale")

    def test_f1_network_arcs_are_nested_and_x1_uses_common_core_route(self) -> None:
        built = {
            name: scenario_generator.build_scenario(name, self.config)
            for name in (
                "F1_system_scale_L",
                "F1_system_scale_M",
                "F1_system_scale_H",
            )
        }
        signatures = {
            name: self.arc_signature(scenario)
            for name, scenario in built.items()
        }
        low_arcs = set(signatures["F1_system_scale_L"])
        medium_arcs = set(signatures["F1_system_scale_M"])
        high_arcs = set(signatures["F1_system_scale_H"])
        self.assertLess(low_arcs, medium_arcs)
        self.assertLess(medium_arcs, high_arcs)
        for arc_id in low_arcs:
            self.assertEqual(
                signatures["F1_system_scale_L"][arc_id],
                signatures["F1_system_scale_M"][arc_id],
            )
            self.assertEqual(
                signatures["F1_system_scale_L"][arc_id],
                signatures["F1_system_scale_H"][arc_id],
            )
        for arc_id in medium_arcs:
            self.assertEqual(
                signatures["F1_system_scale_M"][arc_id],
                signatures["F1_system_scale_H"][arc_id],
            )

        expected_assets = ("TU2", "ST3", "RC2", "SL1")
        expected_arcs = (
            "RAIL_SOURCE__TU2",
            "TU2__ST3",
            "ST3__Z4",
            "Z4__RC2",
            "RC2__SL1",
            "SL1__VESSEL_SINK",
        )
        for scenario in built.values():
            requirement = scenario.special_product_requirements[0]
            self.assertEqual(requirement.required_assets, expected_assets)
            self.assertEqual(requirement.compatible_arcs, expected_arcs)
            self.assertTrue(set(expected_arcs).issubset(self.arc_signature(scenario)))

    def test_f1_effective_network_capacity_increases_by_scale(self) -> None:
        capacities = {
            name: scenario_generator.build_scenario(name, self.config).metadata[
                "effective_network_capacity_tph"
            ]
            for name in (
                "F1_system_scale_L",
                "F1_system_scale_M",
                "F1_system_scale_H",
            )
        }
        self.assertLess(
            capacities["F1_system_scale_L"],
            capacities["F1_system_scale_M"],
        )
        self.assertLess(
            capacities["F1_system_scale_M"],
            capacities["F1_system_scale_H"],
        )
        self.assertAlmostEqual(capacities["F1_system_scale_L"], 4583.0)
        self.assertAlmostEqual(capacities["F1_system_scale_M"], 7708.0)
        self.assertAlmostEqual(capacities["F1_system_scale_H"], 10416.0)

    def test_mode_tradeoff_family_uses_requested_ratios(self) -> None:
        built = {
            name: scenario_generator.build_scenario(name, self.config)
            for name in (
                "F2_mode_tradeoff_L",
                "F2_mode_tradeoff_M",
                "F2_mode_tradeoff_H",
            )
        }
        task_id = built["F2_mode_tradeoff_L"].maintenance_tasks[0].task_id
        durations = {}
        for name, scenario in built.items():
            task = next(task for task in scenario.maintenance_tasks if task.task_id == task_id)
            durations[name] = {
                mode.mode_id: mode.duration_days for mode in task.modes
            }
        self.assertEqual(
            {
                values[scenario_generator.FULL_SHUTDOWN_MODE]
                for values in durations.values()
            },
            {durations["F2_mode_tradeoff_M"][scenario_generator.FULL_SHUTDOWN_MODE]},
        )
        self.assertLessEqual(
            durations["F2_mode_tradeoff_L"][scenario_generator.PARTIAL_30_MODE],
            durations["F2_mode_tradeoff_M"][scenario_generator.PARTIAL_30_MODE],
        )
        self.assertLess(
            durations["F2_mode_tradeoff_M"][scenario_generator.PARTIAL_30_MODE],
            durations["F2_mode_tradeoff_H"][scenario_generator.PARTIAL_30_MODE],
        )
        self.assertLess(
            durations["F2_mode_tradeoff_L"][scenario_generator.PARTIAL_70_MODE],
            durations["F2_mode_tradeoff_M"][scenario_generator.PARTIAL_70_MODE],
        )
        self.assertLess(
            durations["F2_mode_tradeoff_M"][scenario_generator.PARTIAL_70_MODE],
            durations["F2_mode_tradeoff_H"][scenario_generator.PARTIAL_70_MODE],
        )

    def test_yard_flow_tightness_changes_initial_inventory_minimums_and_peak(self) -> None:
        built = {
            name: scenario_generator.build_scenario(name, self.config)
            for name in (
                "F3_yard_flow_L",
                "F3_yard_flow_M",
                "F3_yard_flow_H",
            )
        }
        utilisations = {}
        peaks = {}
        minimum_inventory_ratios = {}
        fill_deviations = {}
        for name, scenario in built.items():
            total_inventory = sum(
                sum(by_product.values()) for by_product in scenario.initial_inventory.values()
            )
            total_capacity = sum(zone.capacity_tonnes for zone in scenario.zones)
            utilisations[name] = total_inventory / total_capacity
            targets = scenario_generator.get_zone_initial_inventory_ratios(name)
            fill_deviations[name] = {
                zone.zone_id: (
                    sum(scenario.initial_inventory[zone.zone_id].values())
                    / zone.capacity_tonnes
                    - targets[zone.zone_id]
                )
                for zone in scenario.zones
            }
            peaks[name] = (
                max(scenario.minimum_throughput_rate_tph.values())
                * scenario.operating_hours_per_day
            )
            minimum_inventory_ratios[name] = min(
                zone.minimum_inventory_tonnes / zone.capacity_tonnes
                for zone in scenario.zones
            )
        self.assertGreater(utilisations["F3_yard_flow_L"], utilisations["F3_yard_flow_M"])
        self.assertGreater(utilisations["F3_yard_flow_M"], utilisations["F3_yard_flow_H"])
        self.assertLess(peaks["F3_yard_flow_L"], peaks["F3_yard_flow_M"])
        self.assertLess(peaks["F3_yard_flow_M"], peaks["F3_yard_flow_H"])
        self.assertLess(
            minimum_inventory_ratios["F3_yard_flow_L"],
            minimum_inventory_ratios["F3_yard_flow_H"],
        )
        for zone_id in fill_deviations["F3_yard_flow_L"]:
            self.assertAlmostEqual(
                fill_deviations["F3_yard_flow_L"][zone_id],
                fill_deviations["F3_yard_flow_M"][zone_id],
                places=6,
            )
            self.assertAlmostEqual(
                fill_deviations["F3_yard_flow_M"][zone_id],
                fill_deviations["F3_yard_flow_H"][zone_id],
                places=6,
            )

    def test_operational_conflict_family_exports_h_and_sigma(self) -> None:
        for name in (
            "F4_operational_conflict_L",
            "F4_operational_conflict_M",
            "F4_operational_conflict_H",
        ):
            with self.subTest(name=name):
                scenario = scenario_generator.build_scenario(name, self.config)
                data = scenario.optimizer_objects()
                self.assertGreaterEqual(len(data["H"]), 2)
                route_window = next(
                    activity
                    for activity in scenario.priority_activities
                    if activity.activity_id == "PA_X1_ROUTE_WINDOW"
                )
                for asset_id in route_window.required_assets:
                    for day in range(route_window.start_day, route_window.end_day):
                        self.assertEqual(
                            data["maintenance_activity_incompatible"][(asset_id, day)],
                            1,
                        )
                        self.assertLess(
                            data["special_activity_capacity_factor"][(asset_id, day)],
                            1.0,
                        )
                if name.endswith(("_M", "_H")):
                    capacity_only = next(
                        activity
                        for activity in scenario.priority_activities
                        if activity.activity_id == "PA_EXPORT_WINDOW_3"
                    )
                    for asset_id in capacity_only.required_assets:
                        for day in range(capacity_only.start_day, capacity_only.end_day):
                            self.assertEqual(
                                data["maintenance_activity_incompatible"][(asset_id, day)],
                                0,
                            )
                            self.assertLess(
                                data["special_activity_capacity_factor"][(asset_id, day)],
                                1.0,
                            )

    def test_f4_activity_timing_is_paired_across_levels(self) -> None:
        built = {
            name: scenario_generator.build_scenario(name, self.config)
            for name in (
                "F4_operational_conflict_L",
                "F4_operational_conflict_M",
                "F4_operational_conflict_H",
            )
        }
        timing_by_level = {
            name: {
                activity.activity_id: (activity.start_day, activity.end_day)
                for activity in scenario.priority_activities
            }
            for name, scenario in built.items()
        }
        for activity_id in ("PA_EXPORT_WINDOW_1", "PA_X1_ROUTE_WINDOW"):
            self.assertEqual(
                timing_by_level["F4_operational_conflict_L"][activity_id],
                timing_by_level["F4_operational_conflict_M"][activity_id],
            )
            self.assertEqual(
                timing_by_level["F4_operational_conflict_M"][activity_id],
                timing_by_level["F4_operational_conflict_H"][activity_id],
            )
        self.assertNotIn(
            "PA_EXPORT_WINDOW_3",
            timing_by_level["F4_operational_conflict_L"],
        )
        self.assertEqual(
            timing_by_level["F4_operational_conflict_M"]["PA_EXPORT_WINDOW_3"],
            timing_by_level["F4_operational_conflict_H"]["PA_EXPORT_WINDOW_3"],
        )

    def test_generator_is_deterministic_for_same_seed(self) -> None:
        for seed in (7, 42, 91):
            config = scenario_generator.GeneratorConfig(seed=seed)
            first = scenario_generator.build_scenario("F4_operational_conflict_H", config)
            second = scenario_generator.build_scenario("F4_operational_conflict_H", config)
            self.assertEqual(first.optimizer_objects(), second.optimizer_objects())
            self.assertEqual(first.to_json_dict(), second.to_json_dict())

    def test_different_seed_changes_multiple_realized_attributes(self) -> None:
        first = scenario_generator.build_scenario(
            "base_control",
            scenario_generator.GeneratorConfig(seed=42),
        )
        second = scenario_generator.build_scenario(
            "base_control",
            scenario_generator.GeneratorConfig(seed=43),
        )
        self.assertNotEqual(first.inbound_supply, second.inbound_supply)
        self.assertNotEqual(first.outbound_demand, second.outbound_demand)
        self.assertNotEqual(first.initial_inventory, second.initial_inventory)
        first_task_sequence = [task.task_id for task in first.maintenance_tasks]
        second_task_sequence = [task.task_id for task in second.maintenance_tasks]
        first_window_signature = {
            task.task_id: (task.earliest_start_day, task.latest_start_day)
            for task in first.maintenance_tasks
        }
        second_window_signature = {
            task.task_id: (task.earliest_start_day, task.latest_start_day)
            for task in second.maintenance_tasks
        }
        self.assertNotEqual(first_task_sequence, second_task_sequence)
        self.assertNotEqual(first_window_signature, second_window_signature)

    def test_common_streams_pair_data_across_families(self) -> None:
        built = {
            name: scenario_generator.build_scenario(name, self.config)
            for name in (
                "base_control",
                "F1_system_scale_M",
                "F2_mode_tradeoff_L",
                "F2_mode_tradeoff_M",
                "F2_mode_tradeoff_H",
                "F3_yard_flow_L",
                "F3_yard_flow_M",
                "F3_yard_flow_H",
                "F4_operational_conflict_M",
            )
        }
        supply = built["base_control"].inbound_supply
        demand = built["base_control"].outbound_demand
        for name, scenario in built.items():
            with self.subTest(name=name, component="supply_demand"):
                self.assertEqual(scenario.inbound_supply, supply)
                self.assertEqual(scenario.outbound_demand, demand)

        base_maintenance = self.maintenance_signature(built["base_control"])
        for name in (
            "F1_system_scale_M",
            "F2_mode_tradeoff_L",
            "F2_mode_tradeoff_M",
            "F2_mode_tradeoff_H",
            "F3_yard_flow_L",
            "F3_yard_flow_M",
            "F3_yard_flow_H",
        ):
            with self.subTest(name=name, component="maintenance"):
                self.assertEqual(
                    self.maintenance_signature(built[name]),
                    base_maintenance,
                )

        self.assertEqual(
            built["base_control"].minimum_throughput_rate_tph,
            built["F2_mode_tradeoff_M"].minimum_throughput_rate_tph,
        )
        self.assertNotEqual(
            built["base_control"].minimum_throughput_rate_tph,
            built["F3_yard_flow_H"].minimum_throughput_rate_tph,
        )

    def test_f1_maintenance_pool_is_nested_by_level(self) -> None:
        low = scenario_generator.build_scenario("F1_system_scale_L", self.config)
        medium = scenario_generator.build_scenario("F1_system_scale_M", self.config)
        high = scenario_generator.build_scenario("F1_system_scale_H", self.config)
        low_signature = self.maintenance_signature(low)
        medium_signature = self.maintenance_signature(medium)
        high_signature = self.maintenance_signature(high)
        self.assertLess(set(low_signature), set(medium_signature))
        self.assertLess(set(medium_signature), set(high_signature))
        for task_id, values in low_signature.items():
            self.assertEqual(medium_signature[task_id], values)
            self.assertEqual(high_signature[task_id], values)
        for task_id, values in medium_signature.items():
            self.assertEqual(high_signature[task_id], values)

    def test_f1_scaled_operating_quantities_preserve_relative_utilisation(self) -> None:
        for seed in range(1, 21):
            config = scenario_generator.GeneratorConfig(seed=seed)
            built = {
                name: scenario_generator.build_scenario(name, config)
                for name in (
                    "F1_system_scale_L",
                    "F1_system_scale_M",
                    "F1_system_scale_H",
                )
            }
            normalized_supply_by_name = {}
            normalized_demand_by_name = {}
            average_supply_ratio = {}
            average_demand_ratio = {}
            yard_utilisation = {}
            for name, scenario in built.items():
                effective_daily_capacity = (
                    scenario.metadata["effective_network_capacity_tph"]
                    * scenario.operating_hours_per_day
                )
                supply_by_day = self.total_by_day(scenario.inbound_supply)
                demand_by_day = self.total_by_day(scenario.outbound_demand)
                normalized_supply_by_name[name] = {
                    day: round(total / effective_daily_capacity, 8)
                    for day, total in supply_by_day.items()
                }
                normalized_demand_by_name[name] = {
                    day: round(total / effective_daily_capacity, 8)
                    for day, total in demand_by_day.items()
                }
                average_supply_ratio[name] = (
                    self.average_total(scenario.inbound_supply)
                    / effective_daily_capacity
                )
                average_demand_ratio[name] = (
                    self.average_total(scenario.outbound_demand)
                    / effective_daily_capacity
                )
                yard_utilisation[name] = (
                    sum(
                        sum(by_product.values())
                        for by_product in scenario.initial_inventory.values()
                    )
                    / sum(zone.capacity_tonnes for zone in scenario.zones)
                )

            for day in normalized_supply_by_name["F1_system_scale_M"]:
                self.assertAlmostEqual(
                    normalized_supply_by_name["F1_system_scale_L"][day],
                    normalized_supply_by_name["F1_system_scale_M"][day],
                    places=7,
                )
                self.assertAlmostEqual(
                    normalized_supply_by_name["F1_system_scale_M"][day],
                    normalized_supply_by_name["F1_system_scale_H"][day],
                    places=7,
                )
                self.assertAlmostEqual(
                    normalized_demand_by_name["F1_system_scale_L"][day],
                    normalized_demand_by_name["F1_system_scale_M"][day],
                    places=7,
                )
                self.assertAlmostEqual(
                    normalized_demand_by_name["F1_system_scale_M"][day],
                    normalized_demand_by_name["F1_system_scale_H"][day],
                    places=7,
                )
            self.assertAlmostEqual(
                average_supply_ratio["F1_system_scale_L"],
                average_supply_ratio["F1_system_scale_H"],
                places=7,
            )
            self.assertAlmostEqual(
                average_demand_ratio["F1_system_scale_L"],
                average_demand_ratio["F1_system_scale_H"],
                places=7,
            )
            self.assertAlmostEqual(
                yard_utilisation["F1_system_scale_L"],
                yard_utilisation["F1_system_scale_H"],
                places=6,
            )

    def test_f2_f3_f4_use_reference_m_topology(self) -> None:
        reference_topology = self.arc_signature(
            scenario_generator.build_scenario("F1_system_scale_M", self.config)
        )
        for name in (
            "F2_mode_tradeoff_L",
            "F2_mode_tradeoff_M",
            "F2_mode_tradeoff_H",
            "F3_yard_flow_L",
            "F3_yard_flow_M",
            "F3_yard_flow_H",
            "F4_operational_conflict_L",
            "F4_operational_conflict_M",
            "F4_operational_conflict_H",
        ):
            with self.subTest(name=name):
                scenario = scenario_generator.build_scenario(name, self.config)
                self.assertEqual(
                    self.stage_counts(scenario),
                    scenario_generator.SYSTEM_SCALE_PROFILES["M"],
                )
                self.assertEqual(len(scenario.assets), 12)
                self.assertEqual(self.arc_signature(scenario), reference_topology)

    def test_dynamic_network_connects_zones_assets_and_x1_routes(self) -> None:
        for name in (
            "F1_system_scale_L",
            "F1_system_scale_M",
            "F1_system_scale_H",
            "F2_mode_tradeoff_M",
            "F3_yard_flow_M",
            "F4_operational_conflict_M",
        ):
            with self.subTest(name=name):
                scenario = scenario_generator.build_scenario(name, self.config)
                data = scenario.optimizer_objects()
                asset_ids = {asset.asset_id for asset in scenario.assets}
                outgoing = {arc.from_node for arc in scenario.arcs}
                incoming = {arc.to_node for arc in scenario.arcs}
                for asset_id in asset_ids:
                    self.assertIn(asset_id, outgoing | incoming)

                reachable_from_source = self.reachable_nodes(scenario, "RAIL_SOURCE")
                for asset_id in asset_ids:
                    self.assertIn(asset_id, reachable_from_source)
                    self.assertIn(
                        "VESSEL_SINK",
                        self.reachable_nodes(scenario, asset_id),
                    )
                for zone in scenario.zones:
                    self.assertIn(zone.zone_id, reachable_from_source)
                    self.assertIn(
                        "VESSEL_SINK",
                        self.reachable_nodes(scenario, zone.zone_id),
                    )

                route = data["special_product_compatible_arcs"]["X1"]
                self.assertEqual(data["arc_from"][route[0]], "RAIL_SOURCE")
                self.assertEqual(data["arc_to"][route[-1]], "VESSEL_SINK")
                for left, right in zip(route, route[1:]):
                    self.assertEqual(data["arc_to"][left], data["arc_from"][right])
                self.assertEqual(
                    tuple(data["special_product_required_assets"]["X1"]),
                    ("TU2", "ST3", "RC2", "SL1"),
                )
                self.assertTrue(
                    set(data["special_product_required_assets"]["X1"]).issubset(
                        asset_ids
                    )
                )
                self.assertEqual(
                    data["special_product_compatible_zones"]["X1"],
                    ["Z4"],
                )

    def test_json_serialization_excludes_reference_when_requested(self) -> None:
        scenario = scenario_generator.build_scenario("F4_operational_conflict_M", self.config)
        payload = scenario.to_json_dict(include_reference_schedule=False)
        self.assertIn("model_ready_data", payload)
        self.assertNotIn("reference_schedule", payload)
        self.assertNotIn("design_evidence", payload)
        json.dumps(payload)

    def test_generation_metadata_records_seeded_realization(self) -> None:
        scenario = scenario_generator.build_scenario("base_control", self.config)
        generation = scenario.metadata["generation"]
        realization = scenario.metadata["realization"]
        self.assertEqual(generation["master_seed"], self.config.seed)
        self.assertEqual(
            generation["randomization_version"],
            scenario_generator.GENERATOR_RANDOMIZATION_VERSION,
        )
        self.assertEqual(
            set(generation["rng_stream_names"]),
            set(scenario_generator.RNG_STREAM_NAMES),
        )
        self.assertEqual(
            realization["maintenance_operation_count"],
            len(scenario.maintenance_tasks),
        )
        self.assertEqual(
            sum(realization["maintenance_by_stage_counts"].values()),
            len(scenario.maintenance_tasks),
        )
        self.assertEqual(
            set(realization["initial_inventory_fill_ratios"]),
            {zone.zone_id for zone in scenario.zones},
        )

    def test_generate_suite_writes_manifest_and_all_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            generated = scenario_generator.generate_suite(
                temporary_directory,
                self.config,
            )
            self.assertEqual(
                set(generated), set(scenario_generator.SCENARIO_NAMES) | {"manifest"}
            )
            manifest = json.loads(Path(generated["manifest"]).read_text())
            self.assertEqual(
                manifest["scenario_count"], len(scenario_generator.SCENARIO_NAMES)
            )
            self.assertTrue(
                all(case["all_design_checks_pass"] for case in manifest["cases"])
            )
            self.assertEqual(
                {
                    case["scenario_name"]
                    for case in manifest["cases"]
                    if case["scenario_name"].startswith("F")
                },
                set(scenario_generator.EXPERIMENTAL_SCENARIO_NAMES),
            )
            self.assertEqual(
                manifest["randomization_version"],
                scenario_generator.GENERATOR_RANDOMIZATION_VERSION,
            )
            self.assertTrue(
                all("realization" in case for case in manifest["cases"])
            )


if __name__ == "__main__":
    unittest.main()
