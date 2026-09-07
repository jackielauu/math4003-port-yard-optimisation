#!/usr/bin/env python3
"""Canonical frozen-schema benchmark generator for MATH4003 port-yard instances.

This module deliberately generates only the data used by the frozen MILP:
sets/topology, shared capacities, maintenance mode/window data, stockyard
bounds and initial inventory, minimum throughput, fixed activity capacity
profiles, fixed maintenance/activity incompatibility profiles, and product routing.

Single- and multi-commodity instances use the same implementation. A
single-commodity instance is the special case ``G = ["G1"]``; a
multi-commodity instance uses ``G = ["G1", "G2", "G3", "G4"]``
plus ``SPECIAL_PRODUCTS = ["S1"]``. Routing matrices are mandatory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


# Support direct script execution as well as package imports.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_data_loader import (
    CANONICAL_MODEL_READY_KEYS, CANONICAL_MODEL_READY_KEY_SET,
    validate_canonical_model_ready_data,
)

EXPERIMENTAL_SCENARIO_NAMES = (
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
)
SCENARIO_NAMES = EXPERIMENTAL_SCENARIO_NAMES
OCCASION_NAMES = SCENARIO_NAMES

GENERATOR_RANDOMIZATION_VERSION = "frozen_schema_routing_restricted_v2"
RNG_STREAM_VERSION = "matched_single_multi_streams_v1"
RNG_STREAM_NAMES = (
    "topology",
    "maintenance_pool",
    "maintenance_windows",
    "mode_duration_variation",
    "throughput_profile",
    "initial_inventory",
    "commodity_split",
    "fixed_activity_profiles",
)

STAGES = ("train_unloader", "stacker", "reclaimer", "shiploader")
STAGE_PREFIX = {
    "train_unloader": "TU",
    "stacker": "ST",
    "reclaimer": "RC",
    "shiploader": "SL",
}
MODE_ORDER = ("F", "P30", "P70")
MODE_AVAILABLE_CAPACITY = {"F": 0.0, "P30": 0.30, "P70": 0.70}
BASE_MODE_DURATION_RATIOS = {"F": 1.00, "P30": 1.45, "P70": 3.30}
F2_MODE_DURATION_RATIOS = {
    "L": {"F": 1.00, "P30": 1.35, "P70": 2.85},
    "M": {"F": 1.00, "P30": 1.45, "P70": 3.30},
    "H": {"F": 1.00, "P30": 1.60, "P70": 3.80},
}

SYSTEM_SCALE_PROFILES = {
    "L": {"train_unloader": 3, "stacker": 3, "reclaimer": 4, "shiploader": 2},
    "M": {"train_unloader": 5, "stacker": 5, "reclaimer": 6, "shiploader": 4},
    "H": {"train_unloader": 8, "stacker": 8, "reclaimer": 10, "shiploader": 6},
}
MAINTENANCE_OPERATION_COUNTS = {"L": 30, "M": 60, "H": 100}
REFERENCE_SCALE_LEVEL = "M"
YARD_ZONES = ("Z1", "Z2", "Z3", "Z4")
DEFAULT_SEEDS = tuple(range(1, 11))

# Preserve the former general-zone permissions; G4 is a new general grade.
ZONE_ALLOWED_PRODUCTS = {
    "Z1": ("G1", "G2", "G4"),
    "Z2": ("G1", "G2", "G3", "G4"),
    "Z3": ("G2", "G3", "G4"),
    "Z4": ("S1",),
}


FORBIDDEN_MODEL_READY_KEYS = {
    "supply_tonnes",
    "demand_tonnes",
    "maintenance_relative_cost",
    "product_is_special",
    "product_grade_specs",
    "product_blending_allowed",
    "product_volume_share",
    "arc_allowed_product",
    "product_asset_compatible",
    "product_may_share_zone",
    "special_product_required_assets",
    "special_product_compatible_arcs",
    "special_product_compatible_zones",
    "special_product_tonnes_per_square_metre",
    "special_product_minimum_reserved_footprint_m2",
    "special_product_segregation_buffer_m2",
    "special_activity_product",
    "special_activity_minimum_tpd",
    "special_activity_minimum_total_tonnes",
    "special_activity_required_zone",
    "special_product_minimum_tpd",
    "special_product_campaign_minimum_tonnes",
    "H",
    "W_h",
}


@dataclass(frozen=True)
class GeneratorConfig:
    horizon_days: int = 90
    seed: int = 1
    commodity_count: int = 4
    operating_hours_per_day: float = 24.0
    train_unloader_rate_tph: float = 2_900.0
    stacker_rate_tph: float = 2_300.0
    reclaimer_rate_tph: float = 2_700.0
    shiploader_rate_tph: float = 4_600.0
    secondary_arc_rate_fraction: float = 0.80

    def validate(self) -> None:
        if self.horizon_days < 60:
            raise ValueError("horizon_days must be at least 60")
        if self.commodity_count not in (1, 4):
            raise ValueError("commodity_count must be 1 or 4")
        if not 0.0 < self.operating_hours_per_day <= 24.0:
            raise ValueError("operating_hours_per_day must be in (0, 24]")
        for value_name in (
            "train_unloader_rate_tph",
            "stacker_rate_tph",
            "reclaimer_rate_tph",
            "shiploader_rate_tph",
            "secondary_arc_rate_fraction",
        ):
            if getattr(self, value_name) <= 0:
                raise ValueError(f"{value_name} must be positive")


@dataclass(frozen=True)
class Asset:
    asset_id: str
    stage: str
    capacity_tpd: float


@dataclass(frozen=True)
class Arc:
    arc_id: str
    from_node: str
    to_node: str
    capacity_tpd: float


@dataclass(frozen=True)
class YardZone:
    zone_id: str
    minimum_inventory_tonnes: float
    capacity_tonnes: float
    expected_flow_share: float
    usable_storage_days: float


@dataclass(frozen=True)
class MaintenanceTask:
    task_id: str
    asset_id: str
    full_duration_days: int
    mode_durations: dict[str, int]
    earliest_start_day: int
    latest_start_day: int
    stress_cluster_day: int


@dataclass(frozen=True)
class FixedActivityWindow:
    window_id: str
    assets: tuple[str, ...]
    start_day: int
    end_day_exclusive: int
    capacity_factor: float
    maintenance_incompatible: bool


@dataclass(frozen=True)
class DissertationScenario:
    scenario_name: str
    config: GeneratorConfig
    assets: tuple[Asset, ...]
    arcs: tuple[Arc, ...]
    zones: tuple[YardZone, ...]
    maintenance_tasks: tuple[MaintenanceTask, ...]
    minimum_throughput_tpd: dict[int, float]
    initial_inventory_tonnes: dict[tuple[str, str], float]
    sigma: dict[tuple[str, int], float]
    h: dict[tuple[str, int], int]
    activity_windows: tuple[FixedActivityWindow, ...]
    metadata: dict[str, Any]
    diagnostics: dict[str, Any]

    @property
    def days(self) -> list[int]:
        return list(range(self.config.horizon_days))

    @property
    def commodities(self) -> list[str]:
        return [f"G{i}" for i in range(1, self.config.commodity_count + 1)]

    def optimizer_objects(self) -> dict[str, Any]:
        days = self.days
        assets = [asset.asset_id for asset in self.assets]
        zones = [zone.zone_id for zone in self.zones]
        nodes = ["RAIL_SOURCE", *assets, *zones, "VESSEL_SINK"]
        arcs = [arc.arc_id for arc in self.arcs]
        predecessors, successors = build_adjacency(nodes, self.arcs)
        maintenance_by_asset = {asset_id: [] for asset_id in assets}
        for task in self.maintenance_tasks:
            maintenance_by_asset[task.asset_id].append(task.task_id)

        model_ready_data = {
            "D": days,
            "N": nodes,
            "E": arcs,
            "A": assets,
            "A_NS": assets,
            "P": zones,
            "G": self.commodities,
            **build_routing_data(self.arcs, zones, self.config),
            "M": [task.task_id for task in self.maintenance_tasks],
            "M_a": maintenance_by_asset,
            "K": list(MODE_ORDER),
            "arc_from": {arc.arc_id: arc.from_node for arc in self.arcs},
            "arc_to": {arc.arc_id: arc.to_node for arc in self.arcs},
            "predecessors": predecessors,
            "successors": successors,
            "asset_capacity_tpd": {asset.asset_id: asset.capacity_tpd for asset in self.assets},
            "arc_capacity_tpd": {arc.arc_id: arc.capacity_tpd for arc in self.arcs},
            "maintenance_duration_days": {
                (task.task_id, mode): task.mode_durations[mode]
                for task in self.maintenance_tasks
                for mode in MODE_ORDER
            },
            "maintenance_available_capacity_fraction": {
                (task.task_id, mode): MODE_AVAILABLE_CAPACITY[mode]
                for task in self.maintenance_tasks
                for mode in MODE_ORDER
            },
            "maintenance_earliest_start": {
                task.task_id: task.earliest_start_day for task in self.maintenance_tasks
            },
            "maintenance_latest_start": {
                task.task_id: task.latest_start_day for task in self.maintenance_tasks
            },
            "zone_min_inventory_tonnes": {
                zone.zone_id: zone.minimum_inventory_tonnes for zone in self.zones
            },
            "zone_max_inventory_tonnes": {
                zone.zone_id: zone.capacity_tonnes for zone in self.zones
            },
            "initial_inventory_tonnes": self.initial_inventory_tonnes,
            "overflow_penalty_per_tonne": {
                zone.zone_id: round(850.0 + 25.0 * index, 3)
                for index, zone in enumerate(self.zones)
            },
            "minimum_throughput_tpd": self.minimum_throughput_tpd,
            "special_activity_capacity_factor": self.sigma,
            "maintenance_activity_incompatible": self.h,
        }
        validate_model_ready_schema(model_ready_data)
        return model_ready_data

    def validate(self) -> None:
        data = self.optimizer_objects()
        assets = set(data["A"])
        zones = set(data["P"])
        nodes = set(data["N"])
        arc_ids = set(data["E"])
        tasks = set(data["M"])
        days = set(data["D"])

        if set(data) != CANONICAL_MODEL_READY_KEY_SET:
            raise ValueError("model_ready_data does not match the canonical schema")
        if not set(data["arc_from"]) == set(data["arc_to"]) == arc_ids:
            raise ValueError("arc_from/arc_to keys must match E")
        for arc_id in arc_ids:
            if data["arc_from"][arc_id] not in nodes or data["arc_to"][arc_id] not in nodes:
                raise ValueError(f"arc {arc_id} references inactive nodes")
        for asset_id, task_ids in data["M_a"].items():
            if asset_id not in assets:
                raise ValueError(f"M_a references inactive asset {asset_id}")
            if not set(task_ids).issubset(tasks):
                raise ValueError(f"M_a contains unknown task for {asset_id}")
        for task in self.maintenance_tasks:
            max_duration = max(task.mode_durations.values())
            if task.asset_id not in assets:
                raise ValueError(f"maintenance task {task.task_id} has invalid asset")
            if not 0 <= task.earliest_start_day <= task.latest_start_day:
                raise ValueError(f"maintenance task {task.task_id} has invalid window")
            if task.latest_start_day + max_duration > self.config.horizon_days:
                raise ValueError(f"maintenance task {task.task_id} cannot fit")
        inventory_by_zone = {zone_id: 0.0 for zone_id in zones}
        for (zone_id, commodity), value in self.initial_inventory_tonnes.items():
            if zone_id not in zones:
                raise ValueError(f"initial inventory references invalid zone {zone_id}")
            if commodity not in data["G"] + data["SPECIAL_PRODUCTS"]:
                raise ValueError(f"initial inventory references invalid commodity {commodity}")
            inventory_by_zone[zone_id] += value
        for zone in self.zones:
            total = round(inventory_by_zone[zone.zone_id], 3)
            if not zone.minimum_inventory_tonnes <= total <= zone.capacity_tonnes:
                raise ValueError(f"initial inventory outside bounds for {zone.zone_id}")
        for (asset_id, day), value in self.sigma.items():
            if asset_id not in assets or day not in days or not 0.0 <= value <= 1.0:
                raise ValueError("invalid sigma entry")
        for (asset_id, day), value in self.h.items():
            if asset_id not in assets or day not in days or value not in (0, 1):
                raise ValueError("invalid h entry")
        if max(self.minimum_throughput_tpd.values()) > self.diagnostics["no_maintenance_throughput_tpd"] * 0.97:
            raise ValueError("minimum throughput exceeds plausible physical benchmark")

    def to_json_dict(self, include_reference_schedule: bool = False) -> dict[str, Any]:
        del include_reference_schedule
        return {
            "metadata": _json_safe(self.metadata),
            "model_ready_data": _json_safe(self.optimizer_objects()),
            "diagnostics": _json_safe(self.diagnostics),
        }


def build_scenario(scenario_name: str, config: GeneratorConfig | None = None) -> DissertationScenario:
    if scenario_name not in SCENARIO_NAMES:
        raise ValueError(f"unknown scenario_name: {scenario_name}")
    config = config or GeneratorConfig()
    config.validate()
    family = get_scenario_family(scenario_name)
    level = get_scenario_level(scenario_name)
    scale_level = level if family == "F1_system_scale" else REFERENCE_SCALE_LEVEL
    profile = SYSTEM_SCALE_PROFILES[scale_level]

    assets = tuple(build_assets(profile, config))
    arcs = tuple(build_arcs(assets, config))
    q0 = calculate_no_maintenance_throughput_tpd(assets)
    zones = tuple(build_zones(scenario_name, q0, arcs))
    activity_windows = tuple(build_fixed_activity_windows(scenario_name, assets, config))
    sigma, h = build_activity_profiles(activity_windows, config.horizon_days)
    maintenance_tasks = tuple(
        build_maintenance_tasks(scenario_name, assets, activity_windows, config)
    )
    minimum_throughput_tpd = build_minimum_throughput_profile(scenario_name, q0, config)
    single_inventory_by_zone = build_single_inventory_by_zone(scenario_name, zones, config)
    initial_inventory = split_initial_inventory(single_inventory_by_zone, config)

    diagnostics = build_diagnostics(
        scenario_name,
        config,
        assets,
        arcs,
        zones,
        maintenance_tasks,
        q0,
        minimum_throughput_tpd,
        single_inventory_by_zone,
        activity_windows,
    )
    metadata = {
        "scenario_name": scenario_name,
        "scenario_family": family,
        "level": level,
        "seed": config.seed,
        "commodity_count": config.commodity_count,
        "generator_randomization_version": GENERATOR_RANDOMIZATION_VERSION,
        "rng_stream_version": RNG_STREAM_VERSION,
        "rng_stream_names": list(RNG_STREAM_NAMES),
        "system_scale_level": scale_level,
        "system_scale_profile": profile,
        "canonical_model_ready_keys": list(CANONICAL_MODEL_READY_KEYS),
        "notes": (
            "Canonical benchmark with mandatory product routing. Multi-commodity instances "
            "add S1 on a dedicated route through Z4; G4 may use all general zones. "
            "Single-commodity instances remain unrestricted. Supply, demand, grade, "
            "footprint, and campaign fields are not exported."
        ),
    }
    scenario = DissertationScenario(
        scenario_name=scenario_name,
        config=config,
        assets=assets,
        arcs=arcs,
        zones=zones,
        maintenance_tasks=maintenance_tasks,
        minimum_throughput_tpd=minimum_throughput_tpd,
        initial_inventory_tonnes=initial_inventory,
        sigma=sigma,
        h=h,
        activity_windows=activity_windows,
        metadata=metadata,
        diagnostics=diagnostics,
    )
    scenario.validate()
    return scenario


def build_assets(profile: Mapping[str, int], config: GeneratorConfig) -> list[Asset]:
    rates = {
        "train_unloader": config.train_unloader_rate_tph,
        "stacker": config.stacker_rate_tph,
        "reclaimer": config.reclaimer_rate_tph,
        "shiploader": config.shiploader_rate_tph,
    }
    assets: list[Asset] = []
    for stage in STAGES:
        for index in range(1, profile[stage] + 1):
            assets.append(
                Asset(
                    asset_id=f"{STAGE_PREFIX[stage]}{index}",
                    stage=stage,
                    capacity_tpd=round(rates[stage] * config.operating_hours_per_day, 3),
                )
            )
    return assets


def build_arcs(assets: Sequence[Asset], config: GeneratorConfig) -> list[Arc]:
    assets_by_stage = {
        stage: [asset for asset in assets if asset.stage == stage]
        for stage in STAGES
    }
    zones = list(YARD_ZONES)
    arcs: list[Arc] = []

    def add_arc(from_node: str, to_node: str, capacity_tpd: float) -> None:
        arc_id = f"{from_node}__{to_node}"
        if arc_id not in {arc.arc_id for arc in arcs}:
            arcs.append(Arc(arc_id, from_node, to_node, round(capacity_tpd, 3)))

    for asset in assets_by_stage["train_unloader"]:
        add_arc("RAIL_SOURCE", asset.asset_id, asset.capacity_tpd)

    connect_layers(
        [asset.asset_id for asset in assets_by_stage["train_unloader"]],
        [asset.asset_id for asset in assets_by_stage["stacker"]],
        min(config.train_unloader_rate_tph, config.stacker_rate_tph)
        * config.operating_hours_per_day
        * config.secondary_arc_rate_fraction,
        add_arc,
    )
    connect_layers(
        [asset.asset_id for asset in assets_by_stage["stacker"]],
        zones,
        config.stacker_rate_tph * config.operating_hours_per_day * 0.90,
        add_arc,
    )
    connect_layers(
        zones,
        [asset.asset_id for asset in assets_by_stage["reclaimer"]],
        config.reclaimer_rate_tph * config.operating_hours_per_day * 0.90,
        add_arc,
    )
    connect_layers(
        [asset.asset_id for asset in assets_by_stage["reclaimer"]],
        [asset.asset_id for asset in assets_by_stage["shiploader"]],
        min(config.reclaimer_rate_tph, config.shiploader_rate_tph)
        * config.operating_hours_per_day
        * config.secondary_arc_rate_fraction,
        add_arc,
    )
    for asset in assets_by_stage["shiploader"]:
        add_arc(asset.asset_id, "VESSEL_SINK", asset.capacity_tpd)
    return arcs


def connect_layers(
    left_nodes: Sequence[str],
    right_nodes: Sequence[str],
    capacity_tpd: float,
    add_arc,
) -> None:
    for index, left in enumerate(left_nodes):
        add_arc(left, right_nodes[index % len(right_nodes)], capacity_tpd)
        add_arc(left, right_nodes[(index + 1) % len(right_nodes)], capacity_tpd)
    for index, right in enumerate(right_nodes):
        add_arc(left_nodes[index % len(left_nodes)], right, capacity_tpd)


def build_zones(scenario_name: str, q0_tpd: float, arcs: Sequence[Arc]) -> list[YardZone]:
    family = get_scenario_family(scenario_name)
    level = get_scenario_level(scenario_name)
    flow_shares = estimate_stockyard_flow_shares(arcs, YARD_ZONES)
    usable_storage_days = {zone_id: 5.6 for zone_id in YARD_ZONES}
    minimum_storage_days = {zone_id: 0.45 for zone_id in YARD_ZONES}
    if family == "F3_yard_flow":
        usable_by_level = {
            "L": {"Z1": 6.3, "Z2": 5.8, "Z3": 6.6, "Z4": 5.4},
            "M": {"Z1": 4.7, "Z2": 4.2, "Z3": 4.5, "Z4": 3.8},
            "H": {"Z1": 3.2, "Z2": 2.7, "Z3": 3.0, "Z4": 2.3},
        }
        min_by_level = {
            "L": {"Z1": 0.42, "Z2": 0.46, "Z3": 0.40, "Z4": 0.44},
            "M": {"Z1": 0.52, "Z2": 0.56, "Z3": 0.50, "Z4": 0.54},
            "H": {"Z1": 0.62, "Z2": 0.66, "Z3": 0.60, "Z4": 0.64},
        }
        usable_storage_days = usable_by_level[level]
        minimum_storage_days = min_by_level[level]
    zones = []
    for zone_id in YARD_ZONES:
        zone_flow = q0_tpd * flow_shares[zone_id]
        minimum = round(minimum_storage_days[zone_id] * zone_flow, 3)
        capacity = round(minimum + usable_storage_days[zone_id] * zone_flow, 3)
        zones.append(
            YardZone(
                zone_id=zone_id,
                minimum_inventory_tonnes=minimum,
                capacity_tonnes=capacity,
                expected_flow_share=round(flow_shares[zone_id], 8),
                usable_storage_days=usable_storage_days[zone_id],
            )
        )
    return zones


def estimate_stockyard_flow_shares(
    arcs: Sequence[Arc],
    zones: Sequence[str],
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for zone_id in zones:
        inbound = sum(arc.capacity_tpd for arc in arcs if arc.to_node == zone_id)
        outbound = sum(arc.capacity_tpd for arc in arcs if arc.from_node == zone_id)
        if inbound <= 0.0 or outbound <= 0.0:
            raise ValueError(f"cannot estimate flow share for disconnected stockyard {zone_id}")
        scores[zone_id] = math.sqrt(inbound * outbound)
    total = sum(scores.values())
    if total <= 0.0:
        raise ValueError("stockyard flow shares cannot be normalised")
    return {zone_id: scores[zone_id] / total for zone_id in zones}


def build_single_inventory_by_zone(
    scenario_name: str,
    zones: Sequence[YardZone],
    config: GeneratorConfig,
) -> dict[str, float]:
    family = get_scenario_family(scenario_name)
    level = get_scenario_level(scenario_name)
    fill_after_min = 0.22
    if family == "F3_yard_flow":
        fill_after_min = {"L": 0.28, "M": 0.22, "H": 0.18}[level]
    rng = rng_for(config.seed, scenario_name, "initial_inventory")
    totals = {}
    for zone in zones:
        usable = zone.capacity_tonnes - zone.minimum_inventory_tonnes
        ratio = fill_after_min + rng.uniform(-0.02, 0.02)
        if family == "F3_yard_flow":
            ratio = min(0.30, max(0.15, ratio))
        value = zone.minimum_inventory_tonnes + ratio * usable
        value = min(max(value, zone.minimum_inventory_tonnes), zone.capacity_tonnes)
        totals[zone.zone_id] = round(value, 3)
    return totals


def split_initial_inventory(
    single_inventory_by_zone: Mapping[str, float],
    config: GeneratorConfig,
) -> dict[tuple[str, str], float]:
    if config.commodity_count == 1:
        return {(zone_id, "G1"): value for zone_id, value in single_inventory_by_zone.items()}

    result: dict[tuple[str, str], float] = {}
    products = [f"G{i}" for i in range(1, config.commodity_count + 1)] + ["S1"]
    for zone_id, total in single_inventory_by_zone.items():
        commodities = list(ZONE_ALLOWED_PRODUCTS[zone_id])
        result.update({(zone_id, product): 0.0 for product in products})
        rng = rng_for(config.seed, zone_id, "commodity_split")
        weights = [rng.uniform(0.6, 1.4) for _ in commodities]
        weight_sum = sum(weights)
        assigned = 0.0
        for commodity, weight in zip(commodities[:-1], weights[:-1]):
            value = round(total * weight / weight_sum, 3)
            result[(zone_id, commodity)] = value
            assigned += value
        result[(zone_id, commodities[-1])] = round(total - assigned, 3)
    return result


def build_maintenance_tasks(
    scenario_name: str,
    assets: Sequence[Asset],
    activity_windows: Sequence[FixedActivityWindow],
    config: GeneratorConfig,
) -> list[MaintenanceTask]:
    family = get_scenario_family(scenario_name)
    level = get_scenario_level(scenario_name)
    scale_level = level if family == "F1_system_scale" else REFERENCE_SCALE_LEVEL
    task_count = MAINTENANCE_OPERATION_COUNTS[scale_level]
    ratio_centres = F2_MODE_DURATION_RATIOS[level] if family == "F2_mode_tradeoff" else BASE_MODE_DURATION_RATIOS
    maintenance_stream_key = (
        "F2_mode_tradeoff"
        if family == "F2_mode_tradeoff"
        else scenario_name
    )
    pool_rng = rng_for(config.seed, maintenance_stream_key, "maintenance_pool")
    window_rng = rng_for(config.seed, maintenance_stream_key, "maintenance_windows")
    duration_rng = rng_for(config.seed, maintenance_stream_key, "mode_duration_variation")

    critical_stage = bottleneck_stage(assets)
    tasks_by_asset: dict[str, int] = {asset.asset_id: 0 for asset in assets}
    tasks: list[MaintenanceTask] = []
    stress_clusters = stress_cluster_days(scenario_name)
    weighted_assets = build_weighted_asset_pool(assets, critical_stage)

    for index in range(1, task_count + 1):
        asset = weighted_choice(weighted_assets, pool_rng)
        tasks_by_asset[asset.asset_id] += 1
        full_duration = pool_rng.randint(3, 6)
        if asset.stage in {"reclaimer", "shiploader"}:
            full_duration += pool_rng.choice((0, 1))
        mode_durations = build_mode_durations(full_duration, ratio_centres, duration_rng)
        max_duration = max(mode_durations.values())
        cluster = pool_rng.choice(stress_clusters)
        width = window_rng.randint(8, 24)
        centre = cluster + window_rng.randint(-7, 7)
        earliest, latest = fit_start_window(centre, width, max_duration, config.horizon_days)
        task = MaintenanceTask(
            task_id=f"M_{asset.asset_id}_{tasks_by_asset[asset.asset_id]:02d}",
            asset_id=asset.asset_id,
            full_duration_days=full_duration,
            mode_durations=mode_durations,
            earliest_start_day=earliest,
            latest_start_day=latest,
            stress_cluster_day=cluster,
        )
        tasks.append(task)

    if activity_windows:
        tasks = retime_tasks_to_activity_conflicts(tasks, activity_windows, config.horizon_days)
    return sorted(tasks, key=lambda task: natural_task_key(task.task_id))


def build_mode_durations(
    full_duration: int,
    ratio_centres: Mapping[str, float],
    rng: random.Random,
) -> dict[str, int]:
    durations = {"F": full_duration}
    p30_ratio = ratio_centres["P30"] * rng.uniform(0.88, 1.12)
    p70_ratio = ratio_centres["P70"] * rng.uniform(0.88, 1.12)
    durations["P30"] = max(full_duration + 1, math.ceil(full_duration * p30_ratio))
    durations["P70"] = max(durations["P30"] + 1, math.ceil(full_duration * p70_ratio))
    return durations


def fit_start_window(
    centre_day: int,
    requested_width: int,
    max_duration: int,
    horizon_days: int,
) -> tuple[int, int]:
    latest_start_limit = horizon_days - max_duration
    if latest_start_limit < 0:
        raise ValueError("maintenance duration exceeds horizon")
    width = max(1, min(requested_width, latest_start_limit + 1))
    earliest = centre_day - width // 2
    earliest = max(0, min(earliest, latest_start_limit - width + 1))
    latest = earliest + width - 1
    return earliest, latest


def retime_tasks_to_activity_conflicts(
    tasks: Sequence[MaintenanceTask],
    activity_windows: Sequence[FixedActivityWindow],
    horizon_days: int,
) -> list[MaintenanceTask]:
    tasks_by_id = {task.task_id: task for task in tasks}
    used: set[str] = set()
    for window in activity_windows:
        candidates = [
            task for task in tasks
            if task.asset_id in window.assets and task.task_id not in used
        ]
        for task in candidates[: max(1, min(3, len(candidates)))]:
            max_duration = max(task.mode_durations.values())
            centre = (window.start_day + window.end_day_exclusive) // 2
            activity_width = max(8, min(24, window.end_day_exclusive - window.start_day + 12))
            earliest, latest = fit_start_window(centre, activity_width, max_duration, horizon_days)
            if latest >= earliest:
                tasks_by_id[task.task_id] = replace(
                    task,
                    earliest_start_day=earliest,
                    latest_start_day=latest,
                    stress_cluster_day=window.start_day,
                )
                used.add(task.task_id)
    return repair_conflict_trapped_task_windows(list(tasks_by_id.values()), activity_windows, horizon_days)


def repair_conflict_trapped_task_windows(
    tasks: Sequence[MaintenanceTask],
    activity_windows: Sequence[FixedActivityWindow],
    horizon_days: int,
) -> list[MaintenanceTask]:
    conflict_days_by_asset: dict[str, set[int]] = {}
    for window in activity_windows:
        if not window.maintenance_incompatible:
            continue
        for asset_id in window.assets:
            conflict_days_by_asset.setdefault(asset_id, set()).update(
                range(window.start_day, min(window.end_day_exclusive, horizon_days))
            )

    repaired: list[MaintenanceTask] = []
    for task in tasks:
        conflict_days = conflict_days_by_asset.get(task.asset_id, set())
        if not conflict_days or has_nonconflicting_start(task, conflict_days):
            repaired.append(task)
            continue
        shortest_duration = min(task.mode_durations.values())
        max_duration = max(task.mode_durations.values())
        latest_start_limit = horizon_days - max_duration
        candidates = [
            start
            for start in range(0, latest_start_limit + 1)
            if all(day not in conflict_days for day in range(start, start + shortest_duration))
        ]
        if not candidates:
            repaired.append(task)
            continue
        centre = (task.earliest_start_day + task.latest_start_day) // 2
        selected_start = min(candidates, key=lambda start: abs(start - centre))
        earliest, latest = fit_start_window(selected_start, 18, max_duration, horizon_days)
        repaired.append(
            replace(
                task,
                earliest_start_day=earliest,
                latest_start_day=latest,
            )
        )
    return repaired


def has_nonconflicting_start(task: MaintenanceTask, conflict_days: set[int]) -> bool:
    for duration in task.mode_durations.values():
        for start_day in range(task.earliest_start_day, task.latest_start_day + 1):
            if all(day not in conflict_days for day in range(start_day, start_day + duration)):
                return True
    return False


def build_minimum_throughput_profile(
    scenario_name: str,
    q0_tpd: float,
    config: GeneratorConfig,
) -> dict[int, float]:
    family = get_scenario_family(scenario_name)
    level = get_scenario_level(scenario_name)
    pressure_class = "medium"
    stream_key = family
    if family == "F4_operational_conflict":
        pressure_class = "high"
    if family == "F3_yard_flow":
        pressure_class = {"L": "medium", "M": "high", "H": "very_high"}[level]
        stream_key = scenario_name

    segment_ratios = {
        "low": (0.48, 0.54, 0.60, 0.52, 0.58, 0.55),
        "medium": (0.54, 0.62, 0.68, 0.59, 0.65, 0.61),
        "high": (0.58, 0.66, 0.72, 0.62, 0.70, 0.65),
        "very_high": (0.60, 0.70, 0.78, 0.66, 0.74, 0.68),
    }[pressure_class]
    rng = rng_for(config.seed, stream_key, "throughput_profile")
    profile: dict[int, float] = {}
    segment_length = math.ceil(config.horizon_days / len(segment_ratios))
    for day in range(config.horizon_days):
        segment = min(day // segment_length, len(segment_ratios) - 1)
        seasonal = 0.012 * math.sin(2.0 * math.pi * day / max(config.horizon_days, 1))
        jitter = rng.uniform(-0.012, 0.012)
        ratio = min(0.95, max(0.45, segment_ratios[segment] + seasonal + jitter))
        profile[day] = round(q0_tpd * ratio, 3)
    return profile


def build_fixed_activity_windows(
    scenario_name: str,
    assets: Sequence[Asset],
    config: GeneratorConfig,
) -> list[FixedActivityWindow]:
    family = get_scenario_family(scenario_name)
    level = get_scenario_level(scenario_name)
    if family not in {"F4_operational_conflict", "F3_yard_flow"}:
        return []

    rng = rng_for(config.seed, scenario_name, "fixed_activity_profiles")
    assets_by_stage = {stage: [asset.asset_id for asset in assets if asset.stage == stage] for stage in STAGES}
    critical_stage = bottleneck_stage(assets)
    critical_assets = assets_by_stage[critical_stage]
    windows: list[FixedActivityWindow] = []

    if family == "F3_yard_flow":
        if level == "L":
            return []
        count = {"M": 1, "H": 2}[level]
        for index in range(count):
            start = 32 + index * 9 + rng.randint(-2, 2)
            duration = 6 + index
            selected = tuple(critical_assets[index % len(critical_assets): index % len(critical_assets) + 2])
            if len(selected) < 2:
                selected = tuple(critical_assets[:2])
            windows.append(
                FixedActivityWindow(
                    window_id=f"F3_UPSTREAM_RESTRICTION_{index + 1}",
                    assets=selected,
                    start_day=start,
                    end_day_exclusive=min(config.horizon_days, start + duration),
                    capacity_factor=0.78 if level == "M" else 0.68,
                    maintenance_incompatible=False,
                )
            )
        return windows

    count = {"L": 2, "M": 4, "H": 6}[level]
    start_centres = (20, 32, 44, 56, 66, 75)
    severity = {"L": 0.84, "M": 0.74, "H": 0.64}[level]
    for index in range(count):
        centre = start_centres[index] + rng.randint(-3, 3)
        duration = rng.randint(5, 9)
        if index % 3 == 0:
            selected_pool = critical_assets
        elif index % 3 == 1:
            selected_pool = assets_by_stage["reclaimer"]
        else:
            selected_pool = assets_by_stage["shiploader"]
        selected = tuple(selected_pool[: min(len(selected_pool), 2 + index % 3)])
        windows.append(
            FixedActivityWindow(
                window_id=f"F4_FIXED_WINDOW_{index + 1}",
                assets=selected,
                start_day=max(0, centre - duration // 2),
                end_day_exclusive=min(config.horizon_days, centre - duration // 2 + duration),
                capacity_factor=max(0.50, severity + rng.uniform(-0.04, 0.04)),
                maintenance_incompatible=(index < {"L": 1, "M": 3, "H": 5}[level]),
            )
        )
    return windows


def build_activity_profiles(
    activity_windows: Sequence[FixedActivityWindow],
    horizon_days: int,
) -> tuple[dict[tuple[str, int], float], dict[tuple[str, int], int]]:
    sigma: dict[tuple[str, int], float] = {}
    h: dict[tuple[str, int], int] = {}
    for window in activity_windows:
        for asset_id in window.assets:
            for day in range(window.start_day, min(window.end_day_exclusive, horizon_days)):
                key = (asset_id, day)
                sigma[key] = round(min(sigma.get(key, 1.0), window.capacity_factor), 6)
                if window.maintenance_incompatible:
                    h[key] = 1
    return sigma, h


def build_diagnostics(
    scenario_name: str,
    config: GeneratorConfig,
    assets: Sequence[Asset],
    arcs: Sequence[Arc],
    zones: Sequence[YardZone],
    tasks: Sequence[MaintenanceTask],
    q0: float,
    minimum_throughput_tpd: Mapping[int, float],
    single_inventory_by_zone: Mapping[str, float],
    activity_windows: Sequence[FixedActivityWindow],
) -> dict[str, Any]:
    by_asset = {asset.asset_id: 0 for asset in assets}
    for task in tasks:
        by_asset[task.asset_id] += 1
    widths = [task.latest_start_day - task.earliest_start_day + 1 for task in tasks]
    effective_scores = {
        mode: [
            round((1.0 - MODE_AVAILABLE_CAPACITY[mode]) * task.mode_durations[mode], 6)
            for task in tasks
        ]
        for mode in MODE_ORDER
    }
    throughput_ratios = [value / q0 for value in minimum_throughput_tpd.values()]
    overlap_pairs = count_overlapping_windows(tasks)
    stage_counts = {
        stage: sum(1 for asset in assets if asset.stage == stage)
        for stage in STAGES
    }
    return {
        "asset_count": len(assets),
        "arc_count": len(arcs),
        "stage_asset_counts": stage_counts,
        "maintenance_operation_count": len(tasks),
        "maintenance_operations_per_asset": by_asset,
        "maintenance_window_width": summary_stats(widths),
        "maintenance_window_overlap_pairs": overlap_pairs,
        "mode_effective_outage_days": {
            mode: summary_stats(values) for mode, values in effective_scores.items()
        },
        "no_maintenance_throughput_tpd": round(q0, 3),
        "minimum_throughput_ratio_to_q0": summary_stats(throughput_ratios),
        "initial_yard_fill_ratio": {
            zone.zone_id: round(single_inventory_by_zone[zone.zone_id] / zone.capacity_tonnes, 6)
            for zone in zones
        },
        "initial_yard_fill_after_min_ratio": {
            zone.zone_id: round(
                (single_inventory_by_zone[zone.zone_id] - zone.minimum_inventory_tonnes)
                / (zone.capacity_tonnes - zone.minimum_inventory_tonnes),
                6,
            )
            for zone in zones
        },
        "stockyard_flow_share": {
            zone.zone_id: zone.expected_flow_share
            for zone in zones
        },
        "usable_storage_days": {
            zone.zone_id: zone.usable_storage_days
            for zone in zones
        },
        "average_usable_storage_days": round(
            sum(zone.usable_storage_days for zone in zones) / len(zones),
            6,
        ),
        "fixed_activity_window_count": len(activity_windows),
        "fixed_activity_capacity_days": sum(
            len(window.assets) * (window.end_day_exclusive - window.start_day)
            for window in activity_windows
        ),
        "fixed_capacity_reduction_severity": round(
            sum(
                (1.0 - window.capacity_factor)
                * len(window.assets)
                * (window.end_day_exclusive - window.start_day)
                for window in activity_windows
            ),
            6,
        ),
        "fixed_activity_incompatible_asset_days": sum(
            len(window.assets) * (window.end_day_exclusive - window.start_day)
            for window in activity_windows
            if window.maintenance_incompatible
        ),
        "estimated_model_size": estimate_model_size(config, assets, arcs, zones, tasks),
        "schema_key_count": len(CANONICAL_MODEL_READY_KEYS),
        "scenario_name": scenario_name,
    }


def write_scenario(
    scenario: DissertationScenario,
    output_path: str | Path,
    *,
    include_reference_schedule: bool = False,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Path.write_text(newline=...) requires Python 3.10; open supports 3.9.
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                scenario.to_json_dict(include_reference_schedule=include_reference_schedule),
                indent=2,
                sort_keys=False,
            )
            + "\n"
        )
    return path


def generate_benchmark_suite(
    output_dir: str | Path,
    config: GeneratorConfig | None = None,
    scenario_names: Sequence[str] = SCENARIO_NAMES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> dict[str, Path]:
    base_config = config or GeneratorConfig()
    base_config.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    generated: dict[str, Path] = {}
    manifest_files: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []

    for scenario_name in scenario_names:
        for seed in seeds:
            scenario_config = replace(base_config, seed=seed)
            scenario = build_scenario(scenario_name, scenario_config)
            relative = Path(scenario_name) / f"seed_{seed:03d}" / f"{scenario_name}_seed_{seed:03d}.json"
            path = write_scenario(scenario, output / relative)
            generated[f"{scenario_name}_seed_{seed:03d}"] = path
            manifest_files.append(
                {
                    "scenario_name": scenario_name,
                    "seed": seed,
                    "file": str(relative).replace("\\", "/"),
                    "asset_count": scenario.diagnostics["asset_count"],
                    "arc_count": scenario.diagnostics["arc_count"],
                    "maintenance_operation_count": scenario.diagnostics["maintenance_operation_count"],
                    "no_maintenance_throughput_tpd": scenario.diagnostics["no_maintenance_throughput_tpd"],
                    "mu_min_ratio_min": scenario.diagnostics["minimum_throughput_ratio_to_q0"]["min"],
                    "mu_min_ratio_max": scenario.diagnostics["minimum_throughput_ratio_to_q0"]["max"],
                    "fixed_activity_window_count": scenario.diagnostics["fixed_activity_window_count"],
                }
            )
            validation_rows.append(flatten_validation_row(scenario))

    manifest = {
        "generator_file": "generate_dissertation_scenarios.py",
        "generator_randomization_version": GENERATOR_RANDOMIZATION_VERSION,
        "rng_stream_version": RNG_STREAM_VERSION,
        "rng_stream_names": list(RNG_STREAM_NAMES),
        "commodity_count": base_config.commodity_count,
        "commodity_representation": (
            "single commodity G=['G1']"
            if base_config.commodity_count == 1
            else "multi commodity G=['G1','G2','G3','G4'], SPECIAL_PRODUCTS=['S1']"
        ),
        "scenario_names": list(scenario_names),
        "seeds": list(seeds),
        "expected_instance_count": len(scenario_names) * len(seeds),
        "instance_count": len(manifest_files),
        "canonical_model_ready_keys": list(CANONICAL_MODEL_READY_KEYS),
        "forbidden_model_ready_keys": sorted(FORBIDDEN_MODEL_READY_KEYS),
        "files": manifest_files,
    }
    manifest_path = output / "benchmark_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    generated["benchmark_manifest"] = manifest_path
    write_validation_csv(output / "benchmark_validation.csv", validation_rows)
    write_family_validation_csv(output / "family_validation.csv", validation_rows)
    write_seed_diversity_csv(output / "seed_diversity.csv", validation_rows)
    return generated


def generate_suite(
    output_dir: str | Path,
    config: GeneratorConfig | None = None,
    scenario_names: Sequence[str] = SCENARIO_NAMES,
) -> dict[str, Path]:
    return generate_benchmark_suite(output_dir, config, scenario_names, DEFAULT_SEEDS)


def flatten_validation_row(scenario: DissertationScenario) -> dict[str, Any]:
    throughput = scenario.diagnostics["minimum_throughput_ratio_to_q0"]
    widths = scenario.diagnostics["maintenance_window_width"]
    return {
        "scenario_name": scenario.scenario_name,
        "seed": scenario.config.seed,
        "commodity_count": scenario.config.commodity_count,
        "asset_count": scenario.diagnostics["asset_count"],
        "arc_count": scenario.diagnostics["arc_count"],
        "maintenance_operation_count": scenario.diagnostics["maintenance_operation_count"],
        "window_width_min": widths["min"],
        "window_width_mean": widths["mean"],
        "window_width_max": widths["max"],
        "window_overlap_pairs": scenario.diagnostics["maintenance_window_overlap_pairs"],
        "q0_tpd": scenario.diagnostics["no_maintenance_throughput_tpd"],
        "mu_ratio_min": throughput["min"],
        "mu_ratio_mean": throughput["mean"],
        "mu_ratio_max": throughput["max"],
        "fixed_activity_window_count": scenario.diagnostics["fixed_activity_window_count"],
        "fixed_activity_incompatible_asset_days": scenario.diagnostics[
            "fixed_activity_incompatible_asset_days"
        ],
    }


def write_validation_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_family_validation_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(get_scenario_family(row["scenario_name"]), []).append(row)
    output_rows = []
    for family, family_rows in grouped.items():
        output_rows.append(
            {
                "family": family,
                "case_count": len(family_rows),
                "asset_count_min": min(row["asset_count"] for row in family_rows),
                "asset_count_max": max(row["asset_count"] for row in family_rows),
                "maintenance_count_min": min(row["maintenance_operation_count"] for row in family_rows),
                "maintenance_count_max": max(row["maintenance_operation_count"] for row in family_rows),
                "mu_ratio_max": max(row["mu_ratio_max"] for row in family_rows),
            }
        )
    write_validation_csv(path, output_rows)


def write_seed_diversity_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output_rows = []
    for seed in sorted({row["seed"] for row in rows}):
        seed_rows = [row for row in rows if row["seed"] == seed]
        output_rows.append(
            {
                "seed": seed,
                "case_count": len(seed_rows),
                "window_overlap_pairs_total": sum(row["window_overlap_pairs"] for row in seed_rows),
                "fixed_activity_windows_total": sum(row["fixed_activity_window_count"] for row in seed_rows),
            }
        )
    write_validation_csv(path, output_rows)


def validate_model_ready_schema(model_ready_data: Mapping[str, Any]) -> None:
    validate_canonical_model_ready_data(model_ready_data)


def build_routing_data(arcs: Sequence[Arc], zones: Sequence[str], config: GeneratorConfig) -> dict[str, Any]:
    """Restrict S1 to one existing source-Z4-sink route on the current topology.

    The old fixed asset IDs do not span the newer scale profiles. Select the
    first incoming/outgoing arc at each stage deterministically instead.
    Single-commodity scenarios retain their unrestricted aggregate meaning.
    """
    general = [f"G{i}" for i in range(1, config.commodity_count + 1)]
    special = ["S1"] if config.commodity_count > 1 else []
    products = general + special
    route = set()
    if special:
        for start, terminal, backwards in (("Z4", "RAIL_SOURCE", True), ("Z4", "VESSEL_SINK", False)):
            node = start
            visited = set()
            while node != terminal:
                if node in visited:
                    raise ValueError("cycle in special-product route")
                visited.add(node)
                candidates = [arc for arc in arcs if (arc.to_node if backwards else arc.from_node) == node]
                if not candidates:
                    raise ValueError("no complete special-product route through Z4")
                arc = candidates[0]
                route.add(arc.arc_id)
                node = arc.from_node if backwards else arc.to_node
    zone_allowed = {
        (zone, product): int(not special or product in ZONE_ALLOWED_PRODUCTS[zone])
        for zone in zones for product in products
    }
    return {
        "SPECIAL_PRODUCTS": special,
        "product_arc_compatible": {
            (product, arc.arc_id): int(
                (product not in special or arc.arc_id in route)
                and all(zone_allowed[(node, product)] for node in (arc.from_node, arc.to_node) if node in zones)
            )
            for product in products for arc in arcs
        },
        "zone_allowed_product": zone_allowed,
        "product_zone_compatible": {(product, zone): value for (zone, product), value in zone_allowed.items()},
    }


def build_adjacency(nodes: Sequence[str], arcs: Sequence[Arc]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    predecessors = {node: [] for node in nodes}
    successors = {node: [] for node in nodes}
    for arc in arcs:
        predecessors.setdefault(arc.to_node, []).append(arc.from_node)
        successors.setdefault(arc.from_node, []).append(arc.to_node)
    return predecessors, successors


def calculate_no_maintenance_throughput_tpd(assets: Sequence[Asset]) -> float:
    return round(
        min(
            sum(asset.capacity_tpd for asset in assets if asset.stage == stage)
            for stage in STAGES
        ),
        3,
    )


def bottleneck_stage(assets: Sequence[Asset]) -> str:
    return min(
        STAGES,
        key=lambda stage: sum(asset.capacity_tpd for asset in assets if asset.stage == stage),
    )


def build_weighted_asset_pool(assets: Sequence[Asset], critical_stage: str) -> list[tuple[Asset, float]]:
    weighted = []
    for asset in assets:
        weight = 1.0
        if asset.stage == critical_stage:
            weight = 5.0
        elif asset.stage in {"train_unloader", "reclaimer"}:
            weight = 2.5
        weighted.append((asset, weight))
    return weighted


def weighted_choice(weighted_assets: Sequence[tuple[Asset, float]], rng: random.Random) -> Asset:
    total = sum(weight for _asset, weight in weighted_assets)
    draw = rng.uniform(0.0, total)
    upto = 0.0
    for asset, weight in weighted_assets:
        upto += weight
        if draw <= upto:
            return asset
    return weighted_assets[-1][0]


def stress_cluster_days(scenario_name: str) -> tuple[int, ...]:
    family = get_scenario_family(scenario_name)
    if family == "F3_yard_flow":
        return (24, 35, 43, 57, 70)
    if family == "F4_operational_conflict":
        return (20, 32, 44, 56, 68, 76)
    return (18, 32, 46, 60, 74)


def count_overlapping_windows(tasks: Sequence[MaintenanceTask]) -> int:
    count = 0
    for left_index, left in enumerate(tasks):
        for right in tasks[left_index + 1:]:
            if left.earliest_start_day <= right.latest_start_day and right.earliest_start_day <= left.latest_start_day:
                count += 1
    return count


def estimate_model_size(
    config: GeneratorConfig,
    assets: Sequence[Asset],
    arcs: Sequence[Arc],
    zones: Sequence[YardZone],
    tasks: Sequence[MaintenanceTask],
) -> dict[str, int]:
    days = config.horizon_days
    commodities = config.commodity_count + int(config.commodity_count > 1)
    maintenance_start_variables = sum(
        (task.latest_start_day - task.earliest_start_day + 1) * len(MODE_ORDER)
        for task in tasks
    )
    variables = (
        maintenance_start_variables
        + len(tasks) * days
        + len(arcs) * commodities * days
        + len(zones) * commodities * days
        + len(zones) * days
        + days
    )
    constraints = (
        len(tasks)
        + len(tasks) * days
        + len(assets) * commodities * days
        + len(zones) * commodities * days
        + 2 * len(zones) * days
        + len(assets) * days
        + len(arcs) * days
        + 2 * days
    )
    return {"variables": variables, "constraints": constraints}


def summary_stats(values: Sequence[float | int]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": round(float(min(values)), 6),
        "mean": round(float(sum(values) / len(values)), 6),
        "max": round(float(max(values)), 6),
    }


def get_scenario_family(scenario_name: str) -> str:
    return "_".join(scenario_name.split("_")[:3])


def get_scenario_level(scenario_name: str) -> str:
    return scenario_name.rsplit("_", 1)[1]


def natural_task_key(task_id: str) -> tuple[str, int]:
    prefix, number = task_id.rsplit("_", 1)
    return prefix, int(number)


def rng_for(seed: int, *parts: object) -> random.Random:
    material = "::".join(str(part) for part in (seed, *parts, GENERATOR_RANDOMIZATION_VERSION))
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        converted = {}
        for key, item in value.items():
            safe_key = "::".join(str(part) for part in key) if isinstance(key, tuple) else str(key)
            converted[safe_key] = _json_safe(item)
        return converted
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def parse_seeds(value: str) -> tuple[int, ...]:
    if "-" in value:
        start, end = value.split("-", 1)
        return tuple(range(int(start), int(end) + 1))
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate frozen-schema dissertation benchmark data.")
    parser.add_argument(
        "--scenario",
        "--occasion",
        dest="scenario",
        default="all",
        choices=("all", *SCENARIO_NAMES),
        help="Scenario family/level to generate, or all.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("generated_dissertation_scenarios"))
    parser.add_argument("--commodity-count", type=int, choices=(1, 4), default=4)
    parser.add_argument("--horizon-days", type=int, default=90)
    parser.add_argument("--seeds", default="1-10", help="Seed list such as 1-10 or 1,2,3.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    scenario_names = SCENARIO_NAMES if args.scenario == "all" else (args.scenario,)
    config = GeneratorConfig(
        horizon_days=args.horizon_days,
        commodity_count=args.commodity_count,
    )
    generated = generate_benchmark_suite(
        args.output_dir,
        config=config,
        scenario_names=scenario_names,
        seeds=parse_seeds(args.seeds),
    )
    print(f"Generated {len(generated) - 1} instance file(s) in {args.output_dir}")
    print(f"Manifest: {generated['benchmark_manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
