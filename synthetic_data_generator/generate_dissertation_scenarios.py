#!/usr/bin/env python3
"""Standalone synthetic data generator for the dissertation optimisation model.

The generator creates a complete 90-day integrated maintenance-and-flow planning
instance from first principles.  It has no dependency on an operational database
or another software project.

Two layers are deliberately separated:

1. Base system generation
   Creates products, equipment, network connections, capacities, supply, demand,
   inventory, maintenance jobs, and planning rules.
2. Occasion generation
   Applies a controlled event profile such as a demand peak, overlapping
   maintenance, a priority campaign, storage pressure, inventory starvation,
   narrow lifecycle windows, alternative maintenance modes, or structural
   infeasibility.

Every scenario contains:

* ``manual_baseline``: a plausible but non-optimised maintenance plan;
* ``reference_schedule``: a design witness used to check that an improved plan
  exists.  It is not an optimiser result and should not be passed to the solver
  during the experiment;
* ``design_evidence``: a lightweight deterministic comparison that confirms the
  intended stress mechanism is present before running the MILP or Benders model.

Example
-------
Generate all benchmark occasions::

    python generate_dissertation_scenarios.py --occasion all --output-dir generated

Import one scenario directly::

    from generate_dissertation_scenarios import GeneratorConfig, build_scenario

    scenario = build_scenario(
        "peak_export_conflict",
        GeneratorConfig(horizon_days=90, seed=42),
    )
    model_data = scenario.optimizer_objects()
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

OCCASION_NAMES = (
    "balanced_control",
    "peak_export_conflict",
    "overlapping_maintenance",
    "priority_campaign_conflict",
    "yard_overflow_risk",
    "inventory_starvation_risk",
    "tight_lifecycle_windows",
    "partial_shutdown_choice",
    "fixed_schedule_control",
    "structurally_infeasible",
)

STAGES = ("train_unloader", "stacker", "reclaimer", "shiploader")

# Maintenance modes used by the current mathematical model.
FULL_SHUTDOWN_MODE = "full_shutdown"
PARTIAL_30_MODE = "partial_30"
PARTIAL_70_MODE = "partial_70"
MAINTENANCE_MODE_ORDER = (
    FULL_SHUTDOWN_MODE,
    PARTIAL_30_MODE,
    PARTIAL_70_MODE,
)
PDF_MODE_SYMBOL = {
    FULL_SHUTDOWN_MODE: "F",
    PARTIAL_30_MODE: "P30",
    PARTIAL_70_MODE: "P70",
}


# ---------------------------------------------------------------------------
# Public data objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneratorConfig:
    """Controls the size and random variation of generated instances."""

    horizon_days: int = 90
    seed: int = 42
    minimum_throughput_ratio: float = 0.55
    base_supply_tpd: float = 178_000.0
    base_demand_tpd: float = 184_000.0
    noise_fraction: float = 0.035
    initial_inventory_fill_ratio: float = 0.48
    number_of_products: int = 4
    operating_hours_per_day: float = 24.0  # Hours of operation per day
    shiploader_count: int = 2

    # The current PDF's initial case-study sentence mentions one shiploader,
    # while the existing synthetic benchmark uses two. Both topologies are
    # supported; set shiploader_count=1 to mirror the PDF case study exactly.

    # Asset flow rates in tonnes per hour (t/hr)
    train_unloader_rate_tph: float = 2917.0
    stacker_rate_tph: float = 2292.0
    reclaimer_rate_tph: float = 2708.0
    shiploader_rate_tph: float = 4583.0

    # Arc flow rates in tonnes per hour (t/hr) - secondary connections
    arc_secondary_rate_tph: float = 1875.0  # For cross-connections between stages

    # Yard zone capacity in tonnes
    zone_capacity_tonnes: float = 800_000.0

    def validate(self) -> None:
        if self.horizon_days < 60:
            raise ValueError("horizon_days must be at least 60 for these occasions")
        if self.number_of_products != 4:
            raise ValueError("the benchmark currently defines exactly four products")
        if not 0.0 < self.minimum_throughput_ratio <= 1.0:
            raise ValueError("minimum_throughput_ratio must be in (0, 1]")
        if not 0.0 <= self.noise_fraction <= 0.25:
            raise ValueError("noise_fraction must be between 0 and 0.25")
        if not 0.0 < self.initial_inventory_fill_ratio < 1.0:
            raise ValueError("initial_inventory_fill_ratio must be in (0, 1)")
        if not 0.0 < self.operating_hours_per_day <= 24.0:
            raise ValueError("operating_hours_per_day must be in (0, 24]")
        if self.shiploader_count not in (1, 2):
            raise ValueError("shiploader_count must be either 1 or 2")
        if self.train_unloader_rate_tph <= 0:
            raise ValueError("train_unloader_rate_tph must be positive")
        if self.stacker_rate_tph <= 0:
            raise ValueError("stacker_rate_tph must be positive")
        if self.reclaimer_rate_tph <= 0:
            raise ValueError("reclaimer_rate_tph must be positive")
        if self.shiploader_rate_tph <= 0:
            raise ValueError("shiploader_rate_tph must be positive")
        if self.arc_secondary_rate_tph <= 0:
            raise ValueError("arc_secondary_rate_tph must be positive")
        if self.zone_capacity_tonnes <= 0:
            raise ValueError("zone_capacity_tonnes must be positive")


@dataclass(frozen=True)
class GradeSpec:
    """Chemical grade specification for a product.

    Defines target values and acceptable ranges for chemical elements.
    Typical iron ore grades include:
    - FE: Iron content (target ~57-60%)
    - SIO2: Silica content (target ~4-7%)
    - AL2O3: Alumina content (target ~2-4%)
    - MN: Manganese content (target ~0.02-1.5%)
    - P: Phosphorus content (target ~0.03-0.14%)
    - AS: Arsenic content (target ~0.002-0.01%)
    - LOI: Loss on Ignition (target ~5-9%)
    """

    element: str
    target: float
    min_value: float
    max_value: float
    unit: str = "%"  # Percentage by default


@dataclass(frozen=True)
class Product:
    product_id: str
    name: str
    share_of_volume: float
    grade_specs: tuple[GradeSpec, ...] = ()  # Chemical grade specifications
    is_special: bool = False
    blending_allowed: bool = True


@dataclass(frozen=True)
class SpecialProductRequirements:
    """Synthetic aggregate requirements for a benchmark special product."""

    product_id: str

    # Spatial requirements
    tonnes_per_square_metre: float
    minimum_reserved_footprint_m2: float
    segregation_buffer_m2: float
    compatible_zones: tuple[str, ...]

    # Routing requirements
    required_assets: tuple[str, ...]
    compatible_arcs: tuple[str, ...]

    # Campaign requirements
    minimum_campaign_tonnes: float
    minimum_contiguous_days: int
    earliest_campaign_start_day: int
    latest_campaign_start_day: int

    # Operational requirements
    pre_campaign_buffer_days: int
    post_campaign_buffer_days: int
    cleanout_before_days: int
    cleanout_after_days: int
    maintenance_incompatible: bool
    blending_allowed: bool
    minimum_build_tonnes: float
    build_complete_before_reclaim: bool


@dataclass(frozen=True)
class YardZone:
    zone_id: str
    capacity_tonnes: float
    allowed_products: tuple[str, ...]
    footprint_capacity_m2: float
    minimum_inventory_tonnes: float = 0.0
    overflow_penalty_per_tonne: float = 1.0


@dataclass(frozen=True)
class Asset:
    asset_id: str
    stage: str
    nominal_rate_tph: float  # Continuous flow rate (tonnes per hour)
    description: str


@dataclass(frozen=True)
class Arc:
    arc_id: str
    from_node: str
    to_node: str
    capacity_tph: float  # Continuous flow rate (tonnes per hour)
    allowed_products: tuple[str, ...]


@dataclass(frozen=True)
class DailyProductQuantity:
    day: int
    product_id: str
    tonnes: float


@dataclass(frozen=True)
class MaintenanceMode:
    mode_id: str
    duration_days: int
    capacity_factor_during_work: float
    relative_cost: float = 1.0


@dataclass(frozen=True)
class MaintenanceTask:
    task_id: str
    asset_id: str
    earliest_start_day: int
    latest_start_day: int
    lifecycle_due_day: int
    modes: tuple[MaintenanceMode, ...]
    mandatory: bool = True
    task_class: str = "major"


@dataclass(frozen=True)
class MaintenanceDecision:
    task_id: str
    start_day: int
    mode_id: str


@dataclass(frozen=True)
class PriorityActivity:
    activity_id: str
    product_id: str
    start_day: int
    end_day: int
    minimum_tonnes_per_day: float
    required_assets: tuple[str, ...]
    hard_requirement: bool = True
    earliest_start_day: int | None = None
    latest_start_day: int | None = None
    required_zone: str | None = None
    minimum_total_tonnes: float = 0.0
    minimum_contiguous_days: int = 1
    pre_campaign_buffer_days: int = 0
    post_campaign_buffer_days: int = 0
    cleanout_before_days: int = 0
    cleanout_after_days: int = 0
    minimum_build_tonnes: float = 0.0
    build_complete_before_reclaim: bool = False
    capacity_factor_during_activity: float = 1.0
    maintenance_incompatible: bool = True

    @property
    def duration_days(self) -> int:
        return self.end_day - self.start_day

    @property
    def feasible_start_days(self) -> tuple[int, ...]:
        earliest = self.start_day if self.earliest_start_day is None else self.earliest_start_day
        latest = self.start_day if self.latest_start_day is None else self.latest_start_day
        return tuple(range(earliest, latest + 1))


@dataclass(frozen=True)
class CommittedOutage:
    outage_id: str
    asset_id: str
    start_day: int
    end_day: int
    capacity_factor: float
    reason: str


@dataclass
class DissertationScenario:
    metadata: dict[str, Any]
    products: list[Product]
    special_product_requirements: list[SpecialProductRequirements]
    days: list[int]
    zones: list[YardZone]
    assets: list[Asset]
    arcs: list[Arc]
    initial_inventory: dict[str, dict[str, float]]
    inbound_supply: list[DailyProductQuantity]
    outbound_demand: list[DailyProductQuantity]
    minimum_throughput_rate_tph: dict[int, float]  # Minimum flow rate constraint (t/hr)
    operating_hours_per_day: float  # Hours of operation per day
    maintenance_tasks: list[MaintenanceTask]
    committed_outages: list[CommittedOutage]
    priority_activities: list[PriorityActivity]
    manual_baseline: list[MaintenanceDecision]
    reference_schedule: list[MaintenanceDecision]
    objective_parameters: dict[str, float]
    design_evidence: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Raise ``ValueError`` when the generated instance is inconsistent."""
        horizon = len(self.days)
        if self.days != list(range(horizon)):
            raise ValueError("days must be consecutive and zero based")
        if horizon <= 0:
            raise ValueError("scenario horizon must be positive")

        product_ids = {p.product_id for p in self.products}
        if len(product_ids) != len(self.products):
            raise ValueError("product IDs must be unique")
        if not math.isclose(
            sum(p.share_of_volume for p in self.products), 1.0, abs_tol=1e-8
        ):
            raise ValueError("product shares must sum to one")
        special_products = [p.product_id for p in self.products if p.is_special]
        if len(special_products) != 1:
            raise ValueError("exactly one benchmark special product must exist")
        special_product_ids = set(special_products)

        zone_ids = {z.zone_id for z in self.zones}
        if len(zone_ids) != len(self.zones):
            raise ValueError("zone IDs must be unique")
        for zone in self.zones:
            if zone.capacity_tonnes <= 0:
                raise ValueError(f"zone {zone.zone_id} has non-positive capacity")
            if zone.footprint_capacity_m2 <= 0:
                raise ValueError(f"zone {zone.zone_id} has non-positive footprint")
            if not 0.0 <= zone.minimum_inventory_tonnes <= zone.capacity_tonnes:
                raise ValueError(f"zone {zone.zone_id} has invalid inventory bounds")
            if zone.overflow_penalty_per_tonne < 0:
                raise ValueError(f"zone {zone.zone_id} has a negative overflow penalty")
            if not set(zone.allowed_products).issubset(product_ids):
                raise ValueError(f"zone {zone.zone_id} references an unknown product")

        asset_ids = {a.asset_id for a in self.assets}
        if len(asset_ids) != len(self.assets):
            raise ValueError("asset IDs must be unique")
        for asset in self.assets:
            if asset.stage not in STAGES:
                raise ValueError(f"unknown stage for asset {asset.asset_id}")
            if asset.nominal_rate_tph <= 0:
                raise ValueError(f"asset {asset.asset_id} has non-positive rate")

        valid_nodes = asset_ids | zone_ids | {"RAIL_SOURCE", "VESSEL_SINK"}
        arc_ids: set[str] = set()
        for arc in self.arcs:
            if arc.arc_id in arc_ids:
                raise ValueError(f"duplicate arc ID {arc.arc_id}")
            arc_ids.add(arc.arc_id)
            if arc.from_node not in valid_nodes or arc.to_node not in valid_nodes:
                raise ValueError(f"arc {arc.arc_id} references an unknown node")
            if arc.capacity_tph <= 0:
                raise ValueError(f"arc {arc.arc_id} has non-positive rate")
            if not set(arc.allowed_products).issubset(product_ids):
                raise ValueError(f"arc {arc.arc_id} references an unknown product")

        self._validate_special_product_requirements(
            product_ids,
            special_product_ids,
            zone_ids,
            asset_ids,
            arc_ids,
            horizon,
        )

        for zone_id, by_product in self.initial_inventory.items():
            if zone_id not in zone_ids:
                raise ValueError(f"initial inventory uses unknown zone {zone_id}")
            if not set(by_product).issubset(product_ids):
                raise ValueError(f"initial inventory uses unknown product in {zone_id}")
            zone_allowed = next(
                zone.allowed_products for zone in self.zones if zone.zone_id == zone_id
            )
            if not set(by_product).issubset(set(zone_allowed)):
                raise ValueError(
                    f"initial inventory uses product incompatible with zone {zone_id}"
                )
            total = sum(by_product.values())
            zone_capacity = next(
                z.capacity_tonnes for z in self.zones if z.zone_id == zone_id
            )
            if any(v < 0 for v in by_product.values()):
                raise ValueError(f"negative initial inventory in {zone_id}")
            if total > zone_capacity + 1e-6:
                raise ValueError(f"initial inventory exceeds capacity in {zone_id}")

        for record in (*self.inbound_supply, *self.outbound_demand):
            if record.product_id not in product_ids:
                raise ValueError(f"unknown product in daily quantity {record}")
            if not 0 <= record.day < horizon:
                raise ValueError(f"day outside horizon in daily quantity {record}")
            if record.tonnes < 0:
                raise ValueError(f"negative tonnes in daily quantity {record}")

        if set(self.minimum_throughput_rate_tph) != set(self.days):
            raise ValueError("minimum throughput rate must be provided for every day")
        if any(v < 0 for v in self.minimum_throughput_rate_tph.values()):
            raise ValueError("minimum throughput rate cannot be negative")

        task_ids: set[str] = set()
        task_by_id: dict[str, MaintenanceTask] = {}
        for task in self.maintenance_tasks:
            if task.task_id in task_ids:
                raise ValueError(f"duplicate task ID {task.task_id}")
            task_ids.add(task.task_id)
            task_by_id[task.task_id] = task
            if task.asset_id not in asset_ids:
                raise ValueError(f"task {task.task_id} references unknown asset")
            if not task.modes:
                raise ValueError(f"task {task.task_id} has no maintenance mode")
            mode_ids = {m.mode_id for m in task.modes}
            if len(mode_ids) != len(task.modes):
                raise ValueError(f"task {task.task_id} has duplicate modes")
            if mode_ids != set(MAINTENANCE_MODE_ORDER):
                raise ValueError(
                    f"task {task.task_id} must define F, P30 and P70 modes"
                )
            max_duration = max(m.duration_days for m in task.modes)
            if not (
                0
                <= task.earliest_start_day
                <= task.latest_start_day
                <= horizon - max_duration
            ):
                raise ValueError(f"invalid maintenance window for {task.task_id}")
            if not task.earliest_start_day <= task.lifecycle_due_day < horizon:
                raise ValueError(f"invalid lifecycle due day for {task.task_id}")
            for mode in task.modes:
                if mode.duration_days <= 0:
                    raise ValueError(f"non-positive duration for {task.task_id}")
                if not 0.0 <= mode.capacity_factor_during_work <= 1.0:
                    raise ValueError(f"invalid capacity factor for {task.task_id}")

        self._validate_schedule(self.manual_baseline, task_by_id, "manual_baseline")
        self._validate_schedule(
            self.reference_schedule, task_by_id, "reference_schedule"
        )

        for outage in self.committed_outages:
            if outage.asset_id not in asset_ids:
                raise ValueError(f"outage {outage.outage_id} references unknown asset")
            if not 0 <= outage.start_day < outage.end_day <= horizon:
                raise ValueError(f"invalid outage interval for {outage.outage_id}")
            if not 0.0 <= outage.capacity_factor <= 1.0:
                raise ValueError(f"invalid outage factor for {outage.outage_id}")

        for activity in self.priority_activities:
            if activity.product_id not in product_ids:
                raise ValueError(
                    f"priority activity {activity.activity_id} has unknown product"
                )
            if activity.product_id not in special_product_ids:
                raise ValueError(
                    f"priority activity {activity.activity_id} must refer to a special product"
                )
            if not 0 <= activity.start_day < activity.end_day <= horizon:
                raise ValueError(
                    f"invalid priority interval for {activity.activity_id}"
                )
            earliest = (
                activity.start_day
                if activity.earliest_start_day is None
                else activity.earliest_start_day
            )
            latest = (
                activity.start_day
                if activity.latest_start_day is None
                else activity.latest_start_day
            )
            if not 0 <= earliest <= activity.start_day <= latest:
                raise ValueError(
                    f"invalid priority start window for {activity.activity_id}"
                )
            if latest + activity.duration_days > horizon:
                raise ValueError(
                    f"priority activity {activity.activity_id} cannot finish in horizon"
                )
            if activity.minimum_tonnes_per_day < 0:
                raise ValueError(f"negative priority volume for {activity.activity_id}")
            if activity.minimum_total_tonnes < 0:
                raise ValueError(
                    f"negative priority total volume for {activity.activity_id}"
                )
            if activity.minimum_contiguous_days <= 0:
                raise ValueError(
                    f"invalid contiguous-day requirement for {activity.activity_id}"
                )
            if activity.minimum_contiguous_days > activity.duration_days:
                raise ValueError(
                    f"contiguous days exceed duration for {activity.activity_id}"
                )
            if activity.minimum_total_tonnes + 1e-6 < (
                activity.minimum_tonnes_per_day * activity.minimum_contiguous_days
            ):
                raise ValueError(
                    f"priority total volume is inconsistent with daily minimum for {activity.activity_id}"
                )
            protected_start = activity.start_day - activity.pre_campaign_buffer_days
            protected_end = activity.end_day + activity.post_campaign_buffer_days
            if protected_start < 0 or protected_end > horizon:
                raise ValueError(
                    f"priority buffers leave the horizon for {activity.activity_id}"
                )
            if activity.cleanout_before_days < 0 or activity.cleanout_after_days < 0:
                raise ValueError(
                    f"negative cleanout duration for {activity.activity_id}"
                )
            if activity.cleanout_before_days > activity.pre_campaign_buffer_days:
                raise ValueError(
                    f"cleanout before exceeds pre-buffer for {activity.activity_id}"
                )
            if activity.cleanout_after_days > activity.post_campaign_buffer_days:
                raise ValueError(
                    f"cleanout after exceeds post-buffer for {activity.activity_id}"
                )
            if not 0.0 <= activity.capacity_factor_during_activity <= 1.0:
                raise ValueError(
                    f"invalid activity capacity factor for {activity.activity_id}"
                )
            unknown = set(activity.required_assets) - asset_ids
            if unknown:
                raise ValueError(
                    f"priority activity {activity.activity_id} has unknown assets {unknown}"
                )
            requirement = next(
                req
                for req in self.special_product_requirements
                if req.product_id == activity.product_id
            )
            disallowed_assets = set(activity.required_assets) - set(
                requirement.required_assets
            )
            if disallowed_assets:
                raise ValueError(
                    f"priority activity {activity.activity_id} uses assets outside special-product route"
                )
            if activity.required_zone is None:
                raise ValueError(
                    f"priority activity {activity.activity_id} must define a required zone"
                )
            if activity.required_zone not in requirement.compatible_zones:
                raise ValueError(
                    f"priority activity {activity.activity_id} uses incompatible zone"
                )

    def _validate_special_product_requirements(
        self,
        product_ids: set[str],
        special_product_ids: set[str],
        zone_ids: set[str],
        asset_ids: set[str],
        arc_ids: set[str],
        horizon: int,
    ) -> None:
        if len(self.special_product_requirements) != len(special_product_ids):
            raise ValueError("special products must have one matching requirement record")
        requirement_by_product = {
            req.product_id: req for req in self.special_product_requirements
        }
        if set(requirement_by_product) != special_product_ids:
            raise ValueError(
                "special-product requirement product IDs must match special products"
            )

        arc_by_id = {arc.arc_id: arc for arc in self.arcs}
        asset_stage = {asset.asset_id: asset.stage for asset in self.assets}
        for requirement in self.special_product_requirements:
            product_id = requirement.product_id
            if product_id not in product_ids:
                raise ValueError("special-product requirement references unknown product")
            if requirement.tonnes_per_square_metre <= 0:
                raise ValueError(f"{product_id} footprint density must be positive")
            if requirement.minimum_reserved_footprint_m2 <= 0:
                raise ValueError(f"{product_id} reserved footprint must be positive")
            if requirement.segregation_buffer_m2 < 0:
                raise ValueError(f"{product_id} segregation buffer cannot be negative")
            if requirement.minimum_campaign_tonnes <= 0:
                raise ValueError(f"{product_id} campaign tonnes must be positive")
            if requirement.minimum_contiguous_days <= 0:
                raise ValueError(f"{product_id} contiguous days must be positive")
            if not 0 <= requirement.earliest_campaign_start_day <= requirement.latest_campaign_start_day < horizon:
                raise ValueError(f"{product_id} campaign start window is outside horizon")
            if requirement.pre_campaign_buffer_days < 0 or requirement.post_campaign_buffer_days < 0:
                raise ValueError(f"{product_id} campaign buffers cannot be negative")
            if requirement.cleanout_before_days < 0 or requirement.cleanout_after_days < 0:
                raise ValueError(f"{product_id} cleanout durations cannot be negative")
            if requirement.minimum_build_tonnes < 0:
                raise ValueError(f"{product_id} minimum build tonnes cannot be negative")

            unknown_zones = set(requirement.compatible_zones) - zone_ids
            if unknown_zones:
                raise ValueError(
                    f"{product_id} has unknown compatible zones {unknown_zones}"
                )
            unknown_assets = set(requirement.required_assets) - asset_ids
            if unknown_assets:
                raise ValueError(f"{product_id} has unknown required assets {unknown_assets}")
            unknown_arcs = set(requirement.compatible_arcs) - arc_ids
            if unknown_arcs:
                raise ValueError(f"{product_id} has unknown compatible arcs {unknown_arcs}")

            for zone_id in requirement.compatible_zones:
                zone = next(zone for zone in self.zones if zone.zone_id == zone_id)
                if product_id not in zone.allowed_products:
                    raise ValueError(
                        f"{product_id} compatible zone {zone_id} does not allow the product"
                    )
                if zone.footprint_capacity_m2 < requirement.minimum_reserved_footprint_m2:
                    raise ValueError(
                        f"{product_id} reserved footprint exceeds zone {zone_id} footprint capacity"
                    )

            route_arcs = [arc_by_id[arc_id] for arc_id in requirement.compatible_arcs]
            for left, right in zip(route_arcs, route_arcs[1:]):
                if left.to_node != right.from_node:
                    raise ValueError(f"{product_id} route is not connected")
            if route_arcs[0].from_node != "RAIL_SOURCE":
                raise ValueError(f"{product_id} route must start at RAIL_SOURCE")
            if route_arcs[-1].to_node != "VESSEL_SINK":
                raise ValueError(f"{product_id} route must reach VESSEL_SINK")
            route_nodes = {route_arcs[0].from_node}
            for arc in route_arcs:
                route_nodes.add(arc.to_node)
                if product_id not in arc.allowed_products:
                    raise ValueError(f"{product_id} route arc {arc.arc_id} disallows product")
            if not set(requirement.compatible_zones).issubset(route_nodes):
                raise ValueError(f"{product_id} route does not include its stockyard zone")
            if not set(requirement.required_assets).issubset(route_nodes):
                raise ValueError(f"{product_id} route omits a required asset")
            shiploaders = [
                asset for asset in requirement.required_assets if asset_stage[asset] == "shiploader"
            ]
            if len(shiploaders) != 1:
                raise ValueError(f"{product_id} route must include exactly one shiploader")
            if "SL2" not in asset_ids and any(
                token == "SL2" or "__SL2" in token or "SL2__" in token
                for token in (*requirement.required_assets, *requirement.compatible_arcs)
            ):
                raise ValueError("single-shiploader scenario must not reference SL2")

    @staticmethod
    def _validate_schedule(
        schedule: Sequence[MaintenanceDecision],
        task_by_id: Mapping[str, MaintenanceTask],
        name: str,
    ) -> None:
        decisions = {d.task_id: d for d in schedule}
        if set(decisions) != set(task_by_id):
            missing = set(task_by_id) - set(decisions)
            extra = set(decisions) - set(task_by_id)
            raise ValueError(f"{name} task mismatch; missing={missing}, extra={extra}")
        for decision in schedule:
            task = task_by_id[decision.task_id]
            mode_by_id = {m.mode_id: m for m in task.modes}
            if decision.mode_id not in mode_by_id:
                raise ValueError(f"unknown mode in {name}: {decision}")
            mode = mode_by_id[decision.mode_id]
            if (
                not task.earliest_start_day
                <= decision.start_day
                <= task.latest_start_day
            ):
                raise ValueError(
                    f"start outside maintenance window in {name}: {decision}"
                )
            if decision.start_day + mode.duration_days - 1 > task.lifecycle_due_day:
                raise ValueError(f"lifecycle due day violated in {name}: {decision}")

    def to_json_dict(self, include_reference_schedule: bool = True) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the scenario."""
        self.validate()
        payload = asdict(self)
        payload["model_ready_data"] = _json_safe(self.optimizer_objects())
        if not include_reference_schedule:
            payload.pop("reference_schedule", None)
            payload.pop("design_evidence", None)
        return payload

    def optimizer_objects(self) -> dict[str, Any]:
        """Return model-ready data aligned with the current mathematical PDF.

        The canonical PDF notation is exported alongside the legacy keys used by
        earlier code. The reference schedule and design evidence are deliberately
        excluded so they cannot leak into the optimisation experiment.
        """
        self.validate()
        products = [p.product_id for p in self.products]
        special_products = [p.product_id for p in self.products if p.is_special]
        requirement_by_product = {
            req.product_id: req for req in self.special_product_requirements
        }
        assets = [a.asset_id for a in self.assets]
        zones = [z.zone_id for z in self.zones]
        arc_ids = [arc.arc_id for arc in self.arcs]
        tasks = [task.task_id for task in self.maintenance_tasks]
        activities = [activity.activity_id for activity in self.priority_activities]
        nodes = ["RAIL_SOURCE", *assets, *zones, "VESSEL_SINK"]
        modes = {
            task.task_id: [mode.mode_id for mode in task.modes]
            for task in self.maintenance_tasks
        }

        supply = _records_to_dense(self.inbound_supply, products, self.days)
        demand = _records_to_dense(self.outbound_demand, products, self.days)
        asset_rate = {
            (asset.asset_id, day): asset.nominal_rate_tph
            for asset in self.assets
            for day in self.days
        }
        asset_capacity_tpd = {
            asset.asset_id: asset.nominal_rate_tph * self.operating_hours_per_day
            for asset in self.assets
        }
        arc_capacity_tpd = {
            arc.arc_id: arc.capacity_tph * self.operating_hours_per_day
            for arc in self.arcs
        }
        initial_inventory = {
            (zone.zone_id, product): self.initial_inventory[zone.zone_id].get(
                product, 0.0
            )
            for zone in self.zones
            for product in products
        }
        maintenance_duration: dict[tuple[str, str], int] = {}
        maintenance_factor: dict[tuple[str, str], float] = {}
        maintenance_loss: dict[tuple[str, str], float] = {}
        maintenance_cost: dict[tuple[str, str], float] = {}
        for task in self.maintenance_tasks:
            for mode in task.modes:
                key = (task.task_id, mode.mode_id)
                maintenance_duration[key] = mode.duration_days
                maintenance_factor[key] = mode.capacity_factor_during_work
                maintenance_loss[key] = 1.0 - mode.capacity_factor_during_work
                maintenance_cost[key] = mode.relative_cost

        priority_minimum = {
            (activity.activity_id, day): activity.minimum_tonnes_per_day
            for activity in self.priority_activities
            for day in range(activity.start_day, activity.end_day)
        }
        sigma, incompatible = _special_activity_profiles(self)
        predecessors, successors = _network_neighbours(self.arcs, nodes)
        maintenance_by_asset = {
            asset: [
                task.task_id
                for task in self.maintenance_tasks
                if task.asset_id == asset
            ]
            for asset in assets
        }
        feasible_maintenance_starts = {
            task.task_id: list(
                range(task.earliest_start_day, task.latest_start_day + 1)
            )
            for task in self.maintenance_tasks
        }
        feasible_activity_starts = {
            activity.activity_id: list(activity.feasible_start_days)
            for activity in self.priority_activities
        }
        product_zone_compatible = {
            (product, zone.zone_id): int(product in zone.allowed_products)
            for product in products
            for zone in self.zones
        }
        special_product_zone_compatible = {
            (product, zone): int(zone in requirement_by_product[product].compatible_zones)
            for product in special_products
            for zone in zones
        }
        product_may_share_zone = {
            (left, right): int(
                left == right or (left not in special_products and right not in special_products)
            )
            for left in products
            for right in products
        }
        product_arc_compatible = {
            (product, arc.arc_id): int(
                arc.arc_id in requirement_by_product[product].compatible_arcs
                if product in requirement_by_product
                else product in arc.allowed_products
            )
            for product in products
            for arc in self.arcs
        }
        assets_adjacent_to_product: dict[str, set[str]] = {product: set() for product in products}
        for arc in self.arcs:
            for product in products:
                if product_arc_compatible[(product, arc.arc_id)]:
                    if arc.from_node in assets:
                        assets_adjacent_to_product[product].add(arc.from_node)
                    if arc.to_node in assets:
                        assets_adjacent_to_product[product].add(arc.to_node)
        product_asset_compatible = {
            (product, asset): int(asset in assets_adjacent_to_product[product])
            for product in products
            for asset in assets
        }
        special_product_minimum_tpd = {
            (activity.product_id, day): activity.minimum_tonnes_per_day
            for activity in self.priority_activities
            for day in range(activity.start_day, activity.end_day)
        }

        return {
            # Canonical mathematical-model sets from the current PDF.
            "D": list(self.days),
            "N": nodes,
            "E": arc_ids,
            "A": assets,
            "A_NS": assets,
            "P_STOCKYARD": zones,
            "G": products,
            "M_OPERATIONS": tasks,
            "M_a": maintenance_by_asset,
            "H": activities,
            "W_m": feasible_maintenance_starts,
            "W_h": feasible_activity_starts,
            "K": list(MAINTENANCE_MODE_ORDER),
            "K_symbol": dict(PDF_MODE_SYMBOL),
            "predecessors": predecessors,
            "successors": successors,
            # Legacy aliases retained for existing Gurobi/Pyomo code.
            "T": list(self.days),
            "P": products,
            "Z": zones,
            "J": tasks,
            "M": modes,
            "product_name": {p.product_id: p.name for p in self.products},
            "product_share_of_volume": {
                p.product_id: p.share_of_volume for p in self.products
            },
            "SPECIAL_PRODUCTS": special_products,
            "product_is_special": {p.product_id: int(p.is_special) for p in self.products},
            "product_blending_allowed": {
                p.product_id: int(p.blending_allowed) for p in self.products
            },
            "product_grade_specs": {
                p.product_id: {
                    gs.element: {
                        "target": gs.target,
                        "min": gs.min_value,
                        "max": gs.max_value,
                        "unit": gs.unit,
                    }
                    for gs in p.grade_specs
                }
                for p in self.products
            },
            "stage": {asset.asset_id: asset.stage for asset in self.assets},
            "asset_rate_tph": asset_rate,
            "asset_capacity_tpd": asset_capacity_tpd,
            "arc_from": {arc.arc_id: arc.from_node for arc in self.arcs},
            "arc_to": {arc.arc_id: arc.to_node for arc in self.arcs},
            "arc_rate_tph": {arc.arc_id: arc.capacity_tph for arc in self.arcs},
            "arc_capacity_tpd": arc_capacity_tpd,
            "arc_allowed_product": {
                (arc.arc_id, product): int(product in arc.allowed_products)
                for arc in self.arcs
                for product in products
            },
            "product_arc_compatible": product_arc_compatible,
            "zone_min_inventory_tonnes": {
                zone.zone_id: zone.minimum_inventory_tonnes for zone in self.zones
            },
            "zone_max_inventory_tonnes": {
                zone.zone_id: zone.capacity_tonnes for zone in self.zones
            },
            "zone_capacity_tonnes": {
                zone.zone_id: zone.capacity_tonnes for zone in self.zones
            },
            "zone_footprint_capacity_m2": {
                zone.zone_id: zone.footprint_capacity_m2 for zone in self.zones
            },
            "overflow_penalty_per_tonne": {
                zone.zone_id: zone.overflow_penalty_per_tonne for zone in self.zones
            },
            "zone_allowed_product": {
                (zone.zone_id, product): int(product in zone.allowed_products)
                for zone in self.zones
                for product in products
            },
            "product_zone_compatible": product_zone_compatible,
            "product_asset_compatible": product_asset_compatible,
            "product_may_share_zone": product_may_share_zone,
            "special_product_requirements": {
                product: asdict(requirement_by_product[product])
                for product in special_products
            },
            "special_product_tonnes_per_square_metre": {
                product: requirement_by_product[product].tonnes_per_square_metre
                for product in special_products
            },
            "special_product_minimum_reserved_footprint_m2": {
                product: requirement_by_product[product].minimum_reserved_footprint_m2
                for product in special_products
            },
            "special_product_segregation_buffer_m2": {
                product: requirement_by_product[product].segregation_buffer_m2
                for product in special_products
            },
            "special_product_required_assets": {
                product: list(requirement_by_product[product].required_assets)
                for product in special_products
            },
            "special_product_compatible_arcs": {
                product: list(requirement_by_product[product].compatible_arcs)
                for product in special_products
            },
            "special_product_compatible_zones": {
                product: list(requirement_by_product[product].compatible_zones)
                for product in special_products
            },
            "special_product_zone_compatible": special_product_zone_compatible,
            "initial_inventory_tonnes": initial_inventory,
            "supply_tonnes": supply,
            "demand_tonnes": demand,
            "minimum_throughput_rate_tph": dict(self.minimum_throughput_rate_tph),
            "minimum_throughput_tpd": {
                day: self.minimum_throughput_rate_tph[day]
                * self.operating_hours_per_day
                for day in self.days
            },
            "operating_hours_per_day": self.operating_hours_per_day,
            "maintenance_asset": {
                task.task_id: task.asset_id for task in self.maintenance_tasks
            },
            "maintenance_earliest_start": {
                task.task_id: task.earliest_start_day for task in self.maintenance_tasks
            },
            "maintenance_latest_start": {
                task.task_id: task.latest_start_day for task in self.maintenance_tasks
            },
            "maintenance_lifecycle_due": {
                task.task_id: task.lifecycle_due_day for task in self.maintenance_tasks
            },
            "maintenance_duration_days": maintenance_duration,
            "maintenance_capacity_factor": maintenance_factor,
            "maintenance_available_capacity_fraction": maintenance_factor,
            "maintenance_capacity_loss_fraction": maintenance_loss,
            "maintenance_relative_cost": maintenance_cost,
            "special_activity_duration_days": {
                activity.activity_id: activity.duration_days
                for activity in self.priority_activities
            },
            "special_activity_product": {
                activity.activity_id: activity.product_id
                for activity in self.priority_activities
            },
            "special_activity_scheduled_start": {
                activity.activity_id: activity.start_day
                for activity in self.priority_activities
            },
            "special_activity_required_assets": {
                activity.activity_id: list(activity.required_assets)
                for activity in self.priority_activities
            },
            "special_activity_required_zone": {
                activity.activity_id: activity.required_zone
                for activity in self.priority_activities
            },
            "special_activity_minimum_tpd": {
                activity.activity_id: activity.minimum_tonnes_per_day
                for activity in self.priority_activities
            },
            "special_activity_minimum_total_tonnes": {
                activity.activity_id: activity.minimum_total_tonnes
                for activity in self.priority_activities
            },
            "special_activity_minimum_contiguous_days": {
                activity.activity_id: activity.minimum_contiguous_days
                for activity in self.priority_activities
            },
            "special_activity_pre_buffer_days": {
                activity.activity_id: activity.pre_campaign_buffer_days
                for activity in self.priority_activities
            },
            "special_activity_post_buffer_days": {
                activity.activity_id: activity.post_campaign_buffer_days
                for activity in self.priority_activities
            },
            "special_activity_cleanout_before_days": {
                activity.activity_id: activity.cleanout_before_days
                for activity in self.priority_activities
            },
            "special_activity_cleanout_after_days": {
                activity.activity_id: activity.cleanout_after_days
                for activity in self.priority_activities
            },
            "special_activity_minimum_build_tonnes": {
                activity.activity_id: activity.minimum_build_tonnes
                for activity in self.priority_activities
            },
            "special_activity_build_complete_before_reclaim": {
                activity.activity_id: int(activity.build_complete_before_reclaim)
                for activity in self.priority_activities
            },
            "special_activity_capacity_factor": sigma,
            "maintenance_activity_incompatible": incompatible,
            "committed_outages": [asdict(outage) for outage in self.committed_outages],
            "priority_activities": [
                asdict(activity) for activity in self.priority_activities
            ],
            "priority_minimum_tpd": priority_minimum,
            "special_product_minimum_tpd": special_product_minimum_tpd,
            "special_product_campaign_minimum_tonnes": {
                activity.activity_id: activity.minimum_total_tonnes
                for activity in self.priority_activities
            },
            "manual_baseline": {
                decision.task_id: {
                    "start_day": decision.start_day,
                    "mode_id": decision.mode_id,
                }
                for decision in self.manual_baseline
            },
            "objective_parameters": dict(self.objective_parameters),
            "objective_definition": {
                "sense": "maximize",
                "primary_term": "sum_d throughput_tonnes[d]",
                "penalty_term": (
                    "sum_p,d overflow_penalty_per_tonne[p] * overflow_tonnes[p,d]"
                ),
                "strict_capacity_variant": "fix overflow_tonnes[p,d] = 0",
            },
            "model_notation": {
                "D": "days",
                "N": "network nodes",
                "E": "directed arcs",
                "A": "capacity-constrained assets",
                "A_NS": "non-storage processing assets",
                "P_STOCKYARD": "stockyard nodes (P in the PDF)",
                "G": "material products (generator extension)",
                "M_OPERATIONS": "maintenance operations (M in the PDF)",
                "H": "high-priority activities",
                "K": "maintenance modes F, P30 and P70",
            },
        }


# ---------------------------------------------------------------------------
# Base system generation
# ---------------------------------------------------------------------------


def build_scenario(
    occasion: str,
    config: GeneratorConfig | None = None,
) -> DissertationScenario:
    """Build one complete synthetic scenario for a named occasion."""
    config = config or GeneratorConfig()
    config.validate()
    if occasion not in OCCASION_NAMES:
        raise ValueError(
            f"unknown occasion {occasion!r}; choose one of {', '.join(OCCASION_NAMES)}"
        )

    rng = random.Random(_stable_seed(config.seed, occasion))
    scenario = _build_base_scenario(config, rng, occasion)
    scenario = _apply_occasion(scenario, occasion, config, rng)
    scenario.validate()

    baseline_metrics = evaluate_schedule(scenario, scenario.manual_baseline)
    reference_metrics = evaluate_schedule(scenario, scenario.reference_schedule)
    scenario.design_evidence = {
        "purpose": (
            "Construction check only. The reference schedule is a known witness, "
            "not an optimal solution. Final claims must compare the solver result "
            "against the manual baseline."
        ),
        "manual_baseline_metrics": baseline_metrics,
        "reference_schedule_metrics": reference_metrics,
        "reference_minus_baseline": _metric_delta(reference_metrics, baseline_metrics),
        "occasion_checks": _occasion_checks(
            occasion, scenario, baseline_metrics, reference_metrics
        ),
    }
    return scenario


def _build_base_scenario(
    config: GeneratorConfig,
    rng: random.Random,
    occasion: str,
) -> DissertationScenario:
    # Define grade specifications for iron ore products
    # Based on typical iron ore chemical composition ranges
    normal_1_grades = (  # Synthetic normal product 1 grade envelope
        GradeSpec("FE", 57.5, 57.0, 58.0),
        GradeSpec("SIO2", 5.5, 5.0, 6.0),
        GradeSpec("AL2O3", 2.1, 1.8, 2.4),
        GradeSpec("MN", 0.75, 0.4, 1.1),
        GradeSpec("P", 0.063, 0.055, 0.070),
        GradeSpec("AS", 0.005, 0.002, 0.010),
        GradeSpec("LOI", 7.7, 7.2, 8.2),
    )
    normal_2_grades = (  # Synthetic normal product 2 grade envelope
        GradeSpec("FE", 56.8, 56.3, 57.3),
        GradeSpec("SIO2", 5.9, 5.5, 6.5),
        GradeSpec("AL2O3", 2.7, 2.4, 3.2),
        GradeSpec("MN", 0.80, 0.5, 1.3),
        GradeSpec("P", 0.061, 0.055, 0.068),
        GradeSpec("AS", 0.006, 0.002, 0.012),
        GradeSpec("LOI", 7.4, 7.0, 8.0),
    )
    normal_3_grades = (  # Synthetic normal product 3 grade envelope
        GradeSpec("FE", 60.3, 59.7, 60.8),
        GradeSpec("SIO2", 4.3, 3.8, 4.8),
        GradeSpec("AL2O3", 2.4, 2.1, 2.7),
        GradeSpec("MN", 0.06, 0.04, 0.09),
        GradeSpec("P", 0.133, 0.120, 0.145),
        GradeSpec("AS", 0.003, 0.002, 0.005),
        GradeSpec("LOI", 5.3, 5.0, 5.7),
    )
    special_x1_grades = (  # Synthetic special product X1 grade envelope
        GradeSpec("FE", 57.2, 56.8, 57.6),
        GradeSpec("SIO2", 5.7, 5.2, 6.2),
        GradeSpec("AL2O3", 1.8, 1.6, 2.0),
        GradeSpec("MN", 0.025, 0.020, 0.035),
        GradeSpec("P", 0.072, 0.065, 0.080),
        GradeSpec("AS", 0.003, 0.002, 0.005),
        GradeSpec("LOI", 9.6, 9.0, 10.2),
    )

    products = [
        Product("P1", "Normal Product 1", 0.38, normal_1_grades),
        Product("P2", "Normal Product 2", 0.30, normal_2_grades),
        Product("P3", "Normal Product 3", 0.20, normal_3_grades),
        Product(
            "X1",
            "Special Product X1",
            0.12,
            special_x1_grades,
            is_special=True,
            blending_allowed=False,
        ),
    ]
    product_ids = tuple(p.product_id for p in products)

    zones = [
        YardZone("Z1", config.zone_capacity_tonnes, ("P1", "P2"), 260_000.0),
        YardZone("Z2", config.zone_capacity_tonnes, ("P1", "P2", "P3"), 240_000.0),
        YardZone("Z3", config.zone_capacity_tonnes, ("P2", "P3"), 220_000.0),
        YardZone("Z4", config.zone_capacity_tonnes, ("X1",), 185_000.0),
    ]

    # Rates in tonnes per hour (t/hr) for continuous flow model
    tu_rate = config.train_unloader_rate_tph
    st_rate = config.stacker_rate_tph
    rc_rate = config.reclaimer_rate_tph
    sl_rate = config.shiploader_rate_tph

    assets = [
        Asset("TU1", "train_unloader", tu_rate, "Train unloader 1"),
        Asset("TU2", "train_unloader", tu_rate, "Train unloader 2"),
        Asset("TU3", "train_unloader", tu_rate, "Train unloader 3"),
        Asset("ST1", "stacker", st_rate, "Stacker serving zone 1"),
        Asset("ST2", "stacker", st_rate, "Stacker serving zone 2"),
        Asset("ST3", "stacker", st_rate, "Stacker serving zone 3"),
        Asset("ST4", "stacker", st_rate, "Stacker serving zone 4"),
        Asset("RC1", "reclaimer", rc_rate, "Reclaimer serving zones 1 and 2"),
        Asset("RC2", "reclaimer", rc_rate, "Reclaimer serving zones 2 and 3"),
        Asset("RC3", "reclaimer", rc_rate, "Reclaimer serving zones 3 and 4"),
    ]
    assets.extend(
        Asset(
            f"SL{index}",
            "shiploader",
            sl_rate,
            f"Shiploader {index}",
        )
        for index in range(1, config.shiploader_count + 1)
    )

    arcs = _build_network_arcs(product_ids, config)
    special_requirements = _build_special_product_requirements(config)
    days = list(range(config.horizon_days))

    supply = _generate_daily_profile(
        days=days,
        products=products,
        base_total=config.base_supply_tpd,
        noise_fraction=config.noise_fraction,
        rng=rng,
        phase=0.4,
    )
    demand = _generate_daily_profile(
        days=days,
        products=products,
        base_total=config.base_demand_tpd,
        noise_fraction=config.noise_fraction,
        rng=rng,
        phase=2.1,
    )

    initial_inventory = _generate_initial_inventory(
        zones,
        products,
        config.initial_inventory_fill_ratio,
    )

    tasks = _base_maintenance_tasks(config.horizon_days, config.shiploader_count)
    spread_schedule = _spread_schedule(tasks)

    nominal_stage_rate = _nominal_stage_rate(assets)
    nominal_system_rate = min(nominal_stage_rate.values())
    minimum_throughput_rate = {
        day: round(config.minimum_throughput_ratio * nominal_system_rate, 3)
        for day in days
    }

    return DissertationScenario(
        metadata={
            "scenario_name": occasion,
            "scenario_family": "synthetic_dissertation_benchmark",
            "horizon_days": config.horizon_days,
            "seed": config.seed,
            "time_granularity": "day",
            "flow_model": "continuous_rate_based",
            "data_origin": "generated_from_first_principles",
            "nominal_stage_rate_tph": nominal_stage_rate,
            "nominal_system_rate_tph": nominal_system_rate,
            "operating_hours_per_day": config.operating_hours_per_day,
            "shiploader_count": config.shiploader_count,
            "mathematical_model_version": "MATH4003 Mathematical Model - current PDF",
            "maintenance_modes": {
                "F": {"mode_id": FULL_SHUTDOWN_MODE, "available_capacity": 0.0},
                "P30": {"mode_id": PARTIAL_30_MODE, "available_capacity": 0.30},
                "P70": {"mode_id": PARTIAL_70_MODE, "available_capacity": 0.70},
            },
            "objective_formulation": (
                "maximise cumulative throughput minus penalised stockyard overflow; "
                "a strict no-overflow variant fixes overflow to zero"
            ),
            "occasion_description": "",
            "expected_model_value": "",
            "expected_feasibility": "feasible",
        },
        products=products,
        special_product_requirements=special_requirements,
        days=days,
        zones=zones,
        assets=assets,
        arcs=arcs,
        initial_inventory=initial_inventory,
        inbound_supply=supply,
        outbound_demand=demand,
        minimum_throughput_rate_tph=minimum_throughput_rate,
        operating_hours_per_day=config.operating_hours_per_day,
        maintenance_tasks=tasks,
        committed_outages=[],
        priority_activities=[],
        manual_baseline=list(spread_schedule),
        reference_schedule=list(spread_schedule),
        objective_parameters={
            "throughput_reward_per_tonne": 1.0,
            "unmet_demand_penalty_per_tonne": 3.0,
            "minimum_throughput_shortfall_penalty_per_tonne": 8.0,
            "unaccepted_supply_penalty_per_tonne": 1.0,
            "priority_shortfall_penalty_per_tonne": 20.0,
            "maintenance_mode_cost_weight": 1_000.0,
        },
    )


def _build_network_arcs(
    products: tuple[str, ...], config: GeneratorConfig
) -> list[Arc]:
    """Build network arcs with rates in tonnes per hour (t/hr)."""
    arcs: list[Arc] = []

    # Get rates from config
    tu_rate = config.train_unloader_rate_tph
    st_rate = config.stacker_rate_tph
    rc_rate = config.reclaimer_rate_tph
    sl_rate = config.shiploader_rate_tph
    secondary_rate = config.arc_secondary_rate_tph

    def add(
        from_node: str, to_node: str, rate_tph: float, allowed: Iterable[str]
    ) -> None:
        arcs.append(
            Arc(
                arc_id=f"{from_node}__{to_node}",
                from_node=from_node,
                to_node=to_node,
                capacity_tph=rate_tph,
                allowed_products=tuple(allowed),
            )
        )

    # All rates in tonnes per hour (t/hr)
    for tu in ("TU1", "TU2", "TU3"):
        add("RAIL_SOURCE", tu, tu_rate, products)

    # Train unloader to stacker arcs (primary and secondary connections)
    add("TU1", "ST1", st_rate, ("P1", "P2"))
    add("TU1", "ST2", secondary_rate, ("P1", "P2", "P3"))
    add("TU2", "ST2", st_rate, ("P1", "P2", "P3"))
    add("TU2", "ST3", st_rate, ("P2", "P3", "X1"))
    add("TU3", "ST3", secondary_rate, ("P2", "P3", "X1"))
    add("TU3", "ST4", st_rate, ("P3", "X1"))

    # Stacker to zone arcs
    add("ST1", "Z1", st_rate, ("P1", "P2"))
    add("ST2", "Z2", st_rate, ("P1", "P2", "P3"))
    add("ST3", "Z3", st_rate, ("P2", "P3", "X1"))
    add("ST4", "Z4", st_rate, ("P3", "X1"))

    # Zone to reclaimer arcs (primary and secondary connections)
    add("Z1", "RC1", rc_rate, ("P1", "P2"))
    add("Z2", "RC1", secondary_rate, ("P1", "P2", "P3"))
    add("Z2", "RC2", secondary_rate, ("P1", "P2", "P3"))
    add("Z3", "RC2", rc_rate, ("P2", "P3", "X1"))
    add("Z3", "RC3", secondary_rate, ("P2", "P3", "X1"))
    add("Z4", "RC3", rc_rate, ("P3", "X1"))

    # Reclaimer to shiploader arcs. The one-shiploader topology mirrors the
    # current PDF case study; the two-shiploader topology preserves the earlier
    # benchmark network.
    if config.shiploader_count == 1:
        add("RC1", "SL1", rc_rate, products)
        add("RC2", "SL1", secondary_rate, products)
        add("RC3", "SL1", rc_rate, products)
    else:
        add("RC1", "SL1", rc_rate, products)
        add("RC2", "SL1", secondary_rate, products)
        add("RC2", "SL2", secondary_rate, products)
        add("RC3", "SL2", rc_rate, products)

    # Shiploader to vessel sink arcs
    for index in range(1, config.shiploader_count + 1):
        add(f"SL{index}", "VESSEL_SINK", sl_rate, products)
    return arcs


def _build_special_product_requirements(
    config: GeneratorConfig,
) -> list[SpecialProductRequirements]:
    """Return deterministic synthetic requirements for benchmark special-product handling."""
    shiploader = "SL2" if config.shiploader_count == 2 else "SL1"
    route_nodes = ("RAIL_SOURCE", "TU3", "ST4", "Z4", "RC3", shiploader, "VESSEL_SINK")
    route_arcs = tuple(
        f"{from_node}__{to_node}"
        for from_node, to_node in zip(route_nodes, route_nodes[1:])
    )
    return [
        SpecialProductRequirements(
            product_id="X1",
            tonnes_per_square_metre=3.0,
            minimum_reserved_footprint_m2=90_000.0,
            segregation_buffer_m2=8_000.0,
            compatible_zones=("Z4",),
            required_assets=("TU3", "ST4", "RC3", shiploader),
            compatible_arcs=route_arcs,
            minimum_campaign_tonnes=252_000.0,
            minimum_contiguous_days=9,
            earliest_campaign_start_day=_day(config.horizon_days, 0.52),
            latest_campaign_start_day=_day(config.horizon_days, 0.60),
            pre_campaign_buffer_days=2,
            post_campaign_buffer_days=2,
            cleanout_before_days=1,
            cleanout_after_days=1,
            maintenance_incompatible=True,
            blending_allowed=False,
            minimum_build_tonnes=168_000.0,
            build_complete_before_reclaim=True,
        )
    ]


def _generate_daily_profile(
    *,
    days: Sequence[int],
    products: Sequence[Product],
    base_total: float,
    noise_fraction: float,
    rng: random.Random,
    phase: float,
) -> list[DailyProductQuantity]:
    records: list[DailyProductQuantity] = []
    for day in days:
        weekly = 0.035 * math.sin(2.0 * math.pi * day / 7.0 + phase)
        monthly = 0.025 * math.sin(2.0 * math.pi * day / 30.0 + phase / 2.0)
        noise = rng.uniform(-noise_fraction, noise_fraction)
        total = max(0.0, base_total * (1.0 + weekly + monthly + noise))
        shares = _jittered_product_shares(products, rng)
        for product, share in zip(products, shares, strict=True):
            records.append(
                DailyProductQuantity(day, product.product_id, round(total * share, 3))
            )
    return records


def _jittered_product_shares(
    products: Sequence[Product], rng: random.Random
) -> list[float]:
    raw = [
        max(0.02, p.share_of_volume * (1.0 + rng.uniform(-0.05, 0.05)))
        for p in products
    ]
    total = sum(raw)
    return [v / total for v in raw]


def _generate_initial_inventory(
    zones: Sequence[YardZone],
    products: Sequence[Product],
    fill_ratio: float,
) -> dict[str, dict[str, float]]:
    product_share = {p.product_id: p.share_of_volume for p in products}
    inventory: dict[str, dict[str, float]] = {}
    for zone in zones:
        allowed = list(zone.allowed_products)
        total_weight = sum(product_share[p] for p in allowed)
        zone_total = zone.capacity_tonnes * fill_ratio
        inventory[zone.zone_id] = {
            product: round(zone_total * product_share[product] / total_weight, 3)
            for product in allowed
        }
    return inventory


def _base_maintenance_tasks(
    horizon: int, shiploader_count: int = 2
) -> list[MaintenanceTask]:
    def scaled(day: int) -> int:
        return min(horizon - 1, max(0, round(day * horizon / 90)))

    specifications = [
        ("M_TU1", "TU1", 5, 34, 39, 6, "major"),
        ("M_TU2", "TU2", 18, 54, 60, 7, "major"),
        ("M_TU3", "TU3", 48, 79, 84, 5, "major"),
        ("M_ST1", "ST1", 7, 38, 43, 4, "modular"),
        ("M_ST2", "ST2", 14, 49, 55, 5, "modular"),
        ("M_ST3", "ST3", 32, 64, 70, 4, "modular"),
        ("M_ST4", "ST4", 52, 80, 85, 5, "modular"),
        ("M_RC1", "RC1", 8, 43, 50, 7, "major"),
        ("M_RC2", "RC2", 23, 59, 66, 6, "major"),
        ("M_RC3", "RC3", 46, 80, 87, 7, "major"),
        ("M_SL1", "SL1", 14, 53, 59, 5, "major"),
    ]
    if shiploader_count == 2:
        specifications.append(("M_SL2", "SL2", 43, 80, 86, 5, "major"))
    tasks: list[MaintenanceTask] = []
    for (
        task_id,
        asset_id,
        earliest,
        latest,
        due,
        duration,
        task_class,
    ) in specifications:
        e = scaled(earliest)
        standard_duration = max(2, scaled(duration) - scaled(0))
        partial_30_duration = standard_duration + max(1, standard_duration // 3)
        partial_70_duration = standard_duration + max(2, standard_duration // 2)
        latest_scaled = min(scaled(latest), horizon - partial_70_duration)
        due_scaled = min(scaled(due), horizon - 1)
        if due_scaled < latest_scaled + standard_duration - 1:
            due_scaled = min(horizon - 1, latest_scaled + partial_70_duration - 1)
        tasks.append(
            MaintenanceTask(
                task_id=task_id,
                asset_id=asset_id,
                earliest_start_day=e,
                latest_start_day=latest_scaled,
                lifecycle_due_day=due_scaled,
                modes=(
                    MaintenanceMode(FULL_SHUTDOWN_MODE, standard_duration, 0.0, 1.0),
                    MaintenanceMode(PARTIAL_30_MODE, partial_30_duration, 0.30, 1.15),
                    MaintenanceMode(PARTIAL_70_MODE, partial_70_duration, 0.70, 1.35),
                ),
                task_class=task_class,
            )
        )
    return tasks


def _spread_schedule(tasks: Sequence[MaintenanceTask]) -> list[MaintenanceDecision]:
    """Create a reasonable non-overlapping-by-stage starting schedule."""
    stage_offsets = {
        "TU": 0.10,
        "ST": 0.35,
        "RC": 0.60,
        "SL": 0.82,
    }
    schedule: list[MaintenanceDecision] = []
    for task in tasks:
        prefix = task.asset_id[:2]
        fraction = stage_offsets[prefix]
        span = task.latest_start_day - task.earliest_start_day
        start = task.earliest_start_day + round(span * fraction)
        schedule.append(MaintenanceDecision(task.task_id, start, FULL_SHUTDOWN_MODE))
    return schedule


# ---------------------------------------------------------------------------
# Occasion overlays
# ---------------------------------------------------------------------------


def _apply_occasion(
    scenario: DissertationScenario,
    occasion: str,
    config: GeneratorConfig,
    rng: random.Random,
) -> DissertationScenario:
    handlers = {
        "balanced_control": _occasion_balanced_control,
        "peak_export_conflict": _occasion_peak_export_conflict,
        "overlapping_maintenance": _occasion_overlapping_maintenance,
        "priority_campaign_conflict": _occasion_priority_campaign_conflict,
        "yard_overflow_risk": _occasion_yard_overflow_risk,
        "inventory_starvation_risk": _occasion_inventory_starvation_risk,
        "tight_lifecycle_windows": _occasion_tight_lifecycle_windows,
        "partial_shutdown_choice": _occasion_partial_shutdown_choice,
        "fixed_schedule_control": _occasion_fixed_schedule_control,
        "structurally_infeasible": _occasion_structurally_infeasible,
    }
    return handlers[occasion](scenario, config, rng)


def _occasion_balanced_control(
    scenario: DissertationScenario,
    config: GeneratorConfig,
    rng: random.Random,
) -> DissertationScenario:
    scenario.metadata.update(
        occasion_description=(
            "Normal supply and demand with maintenance already distributed across "
            "the horizon."
        ),
        expected_model_value=(
            "A control case: the optimiser should remain feasible and find only "
            "small gains over the baseline."
        ),
    )
    return scenario


def _occasion_peak_export_conflict(
    scenario: DissertationScenario,
    config: GeneratorConfig,
    rng: random.Random,
) -> DissertationScenario:
    peak_start, peak_end = _period(config.horizon_days, 0.34, 0.50)
    # Create a genuine low-demand shoulder and a pronounced export peak.
    # This makes moving maintenance away from the peak economically meaningful
    # rather than merely spreading the same capacity loss over more days.
    scenario.outbound_demand = _multiply_records(
        scenario.outbound_demand,
        0,
        config.horizon_days,
        factor=0.86,
    )
    scenario.outbound_demand = _multiply_records(
        scenario.outbound_demand,
        peak_start,
        peak_end,
        factor=1.62,
    )
    scenario.inbound_supply = _multiply_records(
        scenario.inbound_supply,
        peak_start - 3,
        peak_end,
        factor=1.12,
    )

    baseline = _schedule_map(scenario.manual_baseline)
    reference = _schedule_map(scenario.reference_schedule)
    _set_decision(baseline, scenario.maintenance_tasks, "M_RC1", peak_start + 1)
    _set_decision(baseline, scenario.maintenance_tasks, "M_RC2", peak_start + 3)
    _set_decision(baseline, scenario.maintenance_tasks, "M_SL1", peak_start + 2)
    _set_decision(reference, scenario.maintenance_tasks, "M_RC1", 10)
    _set_decision(reference, scenario.maintenance_tasks, "M_RC2", peak_end + 2)
    # Cross-stage overlap in a low-demand shoulder concentrates the outage
    # without violating the minimum-throughput threshold.
    _set_decision(reference, scenario.maintenance_tasks, "M_SL1", 10)

    scenario.manual_baseline = list(baseline.values())
    scenario.reference_schedule = list(reference.values())
    scenario.metadata.update(
        occasion_description=(
            f"Export demand rises by 30% on days {peak_start}-{peak_end - 1}, while "
            "the manual plan places two reclaimers and one shiploader in maintenance."
        ),
        expected_model_value=(
            "Move shutdowns away from the export peak to increase loaded tonnes, "
            "reduce demand shortfall, and protect the minimum daily throughput."
        ),
        stress_period=[peak_start, peak_end],
    )
    return scenario


def _occasion_overlapping_maintenance(
    scenario: DissertationScenario,
    config: GeneratorConfig,
    rng: random.Random,
) -> DissertationScenario:
    overlap_day = _day(config.horizon_days, 0.40)
    baseline = _schedule_map(scenario.manual_baseline)
    reference = _schedule_map(scenario.reference_schedule)

    _set_decision(baseline, scenario.maintenance_tasks, "M_RC1", overlap_day)
    _set_decision(baseline, scenario.maintenance_tasks, "M_RC2", overlap_day)
    _set_decision(baseline, scenario.maintenance_tasks, "M_SL1", overlap_day)

    _set_decision(reference, scenario.maintenance_tasks, "M_RC1", overlap_day - 12)
    _set_decision(reference, scenario.maintenance_tasks, "M_RC2", overlap_day + 1)
    _set_decision(reference, scenario.maintenance_tasks, "M_SL1", overlap_day + 9)

    scenario.manual_baseline = list(baseline.values())
    scenario.reference_schedule = list(reference.values())
    scenario.metadata.update(
        occasion_description=(
            "The manual plan starts maintenance on two reclaimers and a shiploader "
            f"at approximately day {overlap_day}."
        ),
        expected_model_value=(
            "Stagger the same mandatory maintenance workload so downstream capacity "
            "does not fall below the minimum operational threshold."
        ),
        stress_period=[overlap_day, overlap_day + 8],
    )
    return scenario


def _occasion_priority_campaign_conflict(
    scenario: DissertationScenario,
    config: GeneratorConfig,
    rng: random.Random,
) -> DissertationScenario:
    start, end = _period(config.horizon_days, 0.56, 0.67)
    requirement = scenario.special_product_requirements[0]
    campaign_shiploader = "SL2" if config.shiploader_count == 2 else "SL1"
    campaign_shiploader_task = (
        "M_SL2" if config.shiploader_count == 2 else "M_SL1"
    )
    activity = PriorityActivity(
        activity_id="PA_X1_SEGREGATED_CAMPAIGN",
        product_id="X1",
        start_day=start,
        end_day=end,
        minimum_tonnes_per_day=28_000.0,
        required_assets=requirement.required_assets,
        hard_requirement=True,
        earliest_start_day=start,
        latest_start_day=start,
        required_zone="Z4",
        minimum_total_tonnes=requirement.minimum_campaign_tonnes,
        minimum_contiguous_days=requirement.minimum_contiguous_days,
        pre_campaign_buffer_days=requirement.pre_campaign_buffer_days,
        post_campaign_buffer_days=requirement.post_campaign_buffer_days,
        cleanout_before_days=requirement.cleanout_before_days,
        cleanout_after_days=requirement.cleanout_after_days,
        minimum_build_tonnes=requirement.minimum_build_tonnes,
        build_complete_before_reclaim=requirement.build_complete_before_reclaim,
        capacity_factor_during_activity=1.0,
        maintenance_incompatible=requirement.maintenance_incompatible,
    )
    scenario.priority_activities = [activity]

    # Give the three campaign-path jobs enough flexibility to be placed before
    # or after the protected activity. The manual baseline will deliberately
    # choose starts inside the campaign.
    replacements: dict[str, MaintenanceTask] = {}
    for task in scenario.maintenance_tasks:
        if task.task_id == "M_TU3":
            replacements[task.task_id] = _retime_task(
                task, start - 12, end + 12, end + 22
            )
        elif task.task_id == "M_ST4":
            replacements[task.task_id] = _retime_task(
                task, start - 12, end + 12, end + 20
            )
        elif task.task_id == "M_RC3":
            replacements[task.task_id] = _retime_task(
                task, start - 12, end + 12, end + 21
            )
        elif task.task_id == campaign_shiploader_task:
            replacements[task.task_id] = _retime_task(
                task, start - 12, end + 12, end + 20
            )
    scenario.maintenance_tasks = [
        replacements.get(task.task_id, task) for task in scenario.maintenance_tasks
    ]

    scenario.outbound_demand = _add_product_demand(
        scenario.outbound_demand,
        product_id="X1",
        start=start,
        end=end,
        additional_tpd=18_000.0,
    )
    scenario.inbound_supply = _multiply_records(
        scenario.inbound_supply,
        start - 10,
        end,
        factor=1.28,
        product_id="X1",
    )

    baseline = _schedule_map(scenario.manual_baseline)
    reference = _schedule_map(scenario.reference_schedule)
    _set_decision(baseline, scenario.maintenance_tasks, "M_TU3", start + 1)
    _set_decision(baseline, scenario.maintenance_tasks, "M_ST4", start)
    _set_decision(baseline, scenario.maintenance_tasks, "M_RC3", start + 1)
    _set_decision(baseline, scenario.maintenance_tasks, campaign_shiploader_task, start + 2)
    _set_decision(reference, scenario.maintenance_tasks, "M_TU3", end + 8)
    _set_decision(reference, scenario.maintenance_tasks, "M_ST4", start - 10)
    _set_decision(reference, scenario.maintenance_tasks, "M_RC3", end + 3)
    _set_decision(reference, scenario.maintenance_tasks, campaign_shiploader_task, start - 10)

    scenario.manual_baseline = list(baseline.values())
    scenario.reference_schedule = list(reference.values())
    scenario.metadata.update(
        occasion_description=(
            f"A protected synthetic X1 campaign runs on days {start}-{end - 1}. "
            "The manual plan shuts its dedicated route, buffer, or cleanout interval."
        ),
        expected_model_value=(
            "Schedule maintenance around the protected X1 campaign, maintain the "
            "dedicated Z4 footprint and route, and avoid special-product shortfall."
        ),
        stress_period=[start, end],
    )
    return scenario


def _occasion_yard_overflow_risk(
    scenario: DissertationScenario,
    config: GeneratorConfig,
    rng: random.Random,
) -> DissertationScenario:
    scenario.initial_inventory = _generate_initial_inventory(
        scenario.zones,
        scenario.products,
        fill_ratio=0.885,
    )
    start, end = _period(config.horizon_days, 0.24, 0.43)
    scenario.inbound_supply = _multiply_records(
        scenario.inbound_supply,
        start,
        end,
        factor=1.38,
    )
    scenario.outbound_demand = _multiply_records(
        scenario.outbound_demand,
        start,
        end,
        factor=1.18,
    )

    baseline = _schedule_map(scenario.manual_baseline)
    reference = _schedule_map(scenario.reference_schedule)
    _set_decision(baseline, scenario.maintenance_tasks, "M_RC1", start + 2)
    _set_decision(baseline, scenario.maintenance_tasks, "M_RC2", start + 4)
    _set_decision(baseline, scenario.maintenance_tasks, "M_SL1", start + 3)
    if config.shiploader_count == 1:
        # With a single shiploader, overlapping its outage with RC1 would stop the
        # entire export path. Place both after the inbound surge instead.
        _set_decision(reference, scenario.maintenance_tasks, "M_RC1", end)
        _set_decision(reference, scenario.maintenance_tasks, "M_RC2", end + 4)
        _set_decision(reference, scenario.maintenance_tasks, "M_SL1", end)
    else:
        _set_decision(reference, scenario.maintenance_tasks, "M_RC1", 10)
        _set_decision(reference, scenario.maintenance_tasks, "M_RC2", end + 4)
        _set_decision(reference, scenario.maintenance_tasks, "M_SL1", 10)

    scenario.manual_baseline = list(baseline.values())
    scenario.reference_schedule = list(reference.values())
    scenario.metadata.update(
        occasion_description=(
            f"The yard starts at 88.5% full and inbound supply increases by 38% on "
            f"days {start}-{end - 1}. The manual plan simultaneously reduces "
            "reclaiming and shiploading capacity."
        ),
        expected_model_value=(
            "Move downstream maintenance away from the supply surge to reduce rail "
            "curtailment and storage-pressure violations."
        ),
        stress_period=[start, end],
    )
    return scenario


def _occasion_inventory_starvation_risk(
    scenario: DissertationScenario,
    config: GeneratorConfig,
    rng: random.Random,
) -> DissertationScenario:
    scenario.initial_inventory = _generate_initial_inventory(
        scenario.zones,
        scenario.products,
        fill_ratio=0.08,
    )
    start, end = _period(config.horizon_days, 0.20, 0.40)
    # Demand is deliberately low outside the peak. This creates useful shoulder
    # periods in which maintenance can be performed while inventory is rebuilt.
    scenario.outbound_demand = _multiply_records(
        scenario.outbound_demand,
        0,
        config.horizon_days,
        factor=0.82,
    )
    scenario.outbound_demand = _multiply_records(
        scenario.outbound_demand,
        start,
        end,
        factor=1.65,
    )
    scenario.inbound_supply = _multiply_records(
        scenario.inbound_supply,
        start - 6,
        end,
        factor=0.94,
    )

    baseline = _schedule_map(scenario.manual_baseline)
    reference = _schedule_map(scenario.reference_schedule)
    _set_decision(baseline, scenario.maintenance_tasks, "M_TU1", start - 4)
    _set_decision(baseline, scenario.maintenance_tasks, "M_TU2", start - 2)
    _set_decision(baseline, scenario.maintenance_tasks, "M_ST1", start - 3)
    _set_decision(baseline, scenario.maintenance_tasks, "M_ST2", start - 1)

    _set_decision(reference, scenario.maintenance_tasks, "M_TU1", 7)
    _set_decision(reference, scenario.maintenance_tasks, "M_TU2", end)
    _set_decision(reference, scenario.maintenance_tasks, "M_ST1", 8)
    _set_decision(reference, scenario.maintenance_tasks, "M_ST2", end)

    scenario.manual_baseline = list(baseline.values())
    scenario.reference_schedule = list(reference.values())
    scenario.metadata.update(
        occasion_description=(
            f"The yard starts at only 8% full, demand increases sharply on days {start}-"
            f"{end - 1}, and the manual plan overlaps upstream unloader and stacker "
            "maintenance immediately before and during the peak."
        ),
        expected_model_value=(
            "Coordinate upstream maintenance with inventory buffering to prevent "
            "stock starvation and downstream throughput loss."
        ),
        stress_period=[start, end],
    )
    return scenario


def _occasion_tight_lifecycle_windows(
    scenario: DissertationScenario,
    config: GeneratorConfig,
    rng: random.Random,
) -> DissertationScenario:
    center = _day(config.horizon_days, 0.43)
    replacements: dict[str, MaintenanceTask] = {}
    for task in scenario.maintenance_tasks:
        if task.task_id == "M_RC1":
            replacements[task.task_id] = _retime_task(
                task, center - 8, center + 2, center + 9
            )
        elif task.task_id == "M_RC2":
            replacements[task.task_id] = _retime_task(
                task, center - 2, center + 8, center + 15
            )
        elif task.task_id == "M_SL1":
            replacements[task.task_id] = _retime_task(
                task, center - 4, center + 8, center + 14
            )
    scenario.maintenance_tasks = [
        replacements.get(t.task_id, t) for t in scenario.maintenance_tasks
    ]

    baseline = _schedule_map(scenario.manual_baseline)
    reference = _schedule_map(scenario.reference_schedule)
    _set_decision(baseline, scenario.maintenance_tasks, "M_RC1", center - 2)
    _set_decision(baseline, scenario.maintenance_tasks, "M_RC2", center - 2)
    _set_decision(baseline, scenario.maintenance_tasks, "M_SL1", center - 2)

    _set_decision(reference, scenario.maintenance_tasks, "M_RC1", center - 8)
    _set_decision(reference, scenario.maintenance_tasks, "M_RC2", center)
    _set_decision(reference, scenario.maintenance_tasks, "M_SL1", center + 6)

    scenario.manual_baseline = list(baseline.values())
    scenario.reference_schedule = list(reference.values())
    scenario.metadata.update(
        occasion_description=(
            "Three downstream jobs have narrow, partially overlapping lifecycle "
            "windows. A naive common start causes a capacity collapse."
        ),
        expected_model_value=(
            "Find a feasible ordering inside tight windows that a manual earliest-"
            "start or common-start rule misses."
        ),
        stress_period=[center - 8, center + 14],
    )
    return scenario


def _occasion_partial_shutdown_choice(
    scenario: DissertationScenario,
    config: GeneratorConfig,
    rng: random.Random,
) -> DissertationScenario:
    start, end = _period(config.horizon_days, 0.37, 0.53)

    scenario.outbound_demand = _multiply_records(
        scenario.outbound_demand,
        start,
        end,
        factor=1.18,
    )
    baseline = _schedule_map(scenario.manual_baseline)
    reference = _schedule_map(scenario.reference_schedule)
    _set_decision(
        baseline,
        scenario.maintenance_tasks,
        "M_RC2",
        start + 1,
        mode_id=FULL_SHUTDOWN_MODE,
    )
    _set_decision(
        reference,
        scenario.maintenance_tasks,
        "M_RC2",
        start + 1,
        mode_id=PARTIAL_70_MODE,
    )
    scenario.manual_baseline = list(baseline.values())
    scenario.reference_schedule = list(reference.values())
    scenario.metadata.update(
        occasion_description=(
            f"A reclaimer maintenance job must occur during sustained strong demand "
            f"on days {start}-{end - 1}. It can use a short full shutdown, a "
            "longer 30%-available mode, or the longest 70%-available mode."
        ),
        expected_model_value=(
            "Select the maintenance mode that provides the best trade-off between "
            "maintenance duration, retained capacity, and throughput."
        ),
        stress_period=[start, end],
    )
    return scenario


def _occasion_fixed_schedule_control(
    scenario: DissertationScenario,
    config: GeneratorConfig,
    rng: random.Random,
) -> DissertationScenario:
    decisions = _schedule_map(scenario.manual_baseline)
    fixed_tasks: list[MaintenanceTask] = []
    for task in scenario.maintenance_tasks:
        decision = decisions[task.task_id]
        mode = next(m for m in task.modes if m.mode_id == decision.mode_id)
        fixed_tasks.append(
            replace(
                task,
                earliest_start_day=decision.start_day,
                latest_start_day=decision.start_day,
                lifecycle_due_day=max(
                    task.lifecycle_due_day,
                    decision.start_day + mode.duration_days - 1,
                ),
            )
        )
    scenario.maintenance_tasks = fixed_tasks
    scenario.reference_schedule = list(scenario.manual_baseline)
    scenario.metadata.update(
        occasion_description=(
            "Every maintenance start is fixed by setting earliest and latest start "
            "to the same day."
        ),
        expected_model_value=(
            "A negative control: the integrated optimiser must not claim scheduling "
            "benefit when no maintenance flexibility exists."
        ),
    )
    return scenario


def _occasion_structurally_infeasible(
    scenario: DissertationScenario,
    config: GeneratorConfig,
    rng: random.Random,
) -> DissertationScenario:
    start, end = _period(config.horizon_days, 0.44, 0.55)
    scenario.committed_outages = [
        CommittedOutage(
            "CO_RC1_FORCED",
            "RC1",
            start,
            end,
            0.0,
            "Non-movable emergency outage",
        ),
        CommittedOutage(
            "CO_RC2_FORCED",
            "RC2",
            start,
            end,
            0.0,
            "Non-movable emergency outage",
        ),
    ]
    for day in range(start, end):
        scenario.minimum_throughput_rate_tph[day] = 5000.0  # ~120,000 tpd equivalent
    scenario.metadata.update(
        occasion_description=(
            f"Two of three reclaimers have non-movable outages on days {start}-"
            f"{end - 1}; remaining downstream capacity is below the hard minimum."
        ),
        expected_model_value=(
            "Correctly prove infeasibility instead of returning an apparently valid "
            "but physically impossible schedule."
        ),
        expected_feasibility="infeasible",
        stress_period=[start, end],
    )
    return scenario


# ---------------------------------------------------------------------------
# Lightweight design evaluator
# ---------------------------------------------------------------------------


def evaluate_schedule(
    scenario: DissertationScenario,
    schedule: Sequence[MaintenanceDecision],
) -> dict[str, float | int]:
    """Evaluate a schedule with a transparent aggregate flow approximation.

    Uses rate-based continuous flow model: throughput = rate Ã— operating_hours.
    The bottleneck rate determines the maximum flow through the system.

    This function is only a scenario-construction check.  The dissertation MILP
    or Benders subproblem remains the authoritative optimisation model.
    """
    scenario.validate()
    products = [p.product_id for p in scenario.products]
    requirement_by_product = {
        req.product_id: req for req in scenario.special_product_requirements
    }
    special_products = set(requirement_by_product)
    supply = _records_to_dense(scenario.inbound_supply, products, scenario.days)
    demand = _records_to_dense(scenario.outbound_demand, products, scenario.days)
    capacity_factor = _effective_capacity_factors(scenario, schedule)
    _, maintenance_activity_incompatible = _special_activity_profiles(scenario)
    hours_per_day = scenario.operating_hours_per_day

    inventory = {
        product: sum(
            scenario.initial_inventory.get(zone.zone_id, {}).get(product, 0.0)
            for zone in scenario.zones
        )
        for product in products
    }
    special_zone_inventory = {
        (product, zone): scenario.initial_inventory.get(zone, {}).get(product, 0.0)
        for product, requirement in requirement_by_product.items()
        for zone in requirement.compatible_zones
    }
    incompatible_special_zone_inventory = {
        product: sum(
            tonnes
            for zone_id, by_product in scenario.initial_inventory.items()
            for stored_product, tonnes in by_product.items()
            if stored_product == product
            and zone_id not in requirement_by_product[product].compatible_zones
        )
        for product in special_products
    }
    total_yard_capacity = sum(z.capacity_tonnes for z in scenario.zones)
    zone_footprint_capacity = {
        zone.zone_id: zone.footprint_capacity_m2 for zone in scenario.zones
    }
    asset_rate = {asset.asset_id: asset.nominal_rate_tph for asset in scenario.assets}
    task_by_id = {task.task_id: task for task in scenario.maintenance_tasks}

    total_throughput = 0.0
    unmet_demand = 0.0
    unaccepted_supply = 0.0
    priority_shortfall = 0.0
    special_product_throughput = {product: 0.0 for product in special_products}
    special_product_daily_shortfall = {product: 0.0 for product in special_products}
    special_product_build_shortfall = {product: 0.0 for product in special_products}
    campaign_volume = {
        activity.activity_id: 0.0 for activity in scenario.priority_activities
    }
    minimum_shortfall = 0.0
    minimum_violation_days = 0
    storage_pressure_days = 0
    overflow_tonnes = 0.0
    footprint_violation_days: set[tuple[str, int]] = set()
    footprint_excess = 0.0
    maximum_special_footprint = 0.0
    incompatible_zone_violation_days = 0
    maintenance_conflict_days: set[tuple[str, int]] = set()
    maximum_inventory = sum(inventory.values())

    for decision in schedule:
        task = task_by_id[decision.task_id]
        mode = next(mode for mode in task.modes if mode.mode_id == decision.mode_id)
        for day in range(decision.start_day, decision.start_day + mode.duration_days):
            if maintenance_activity_incompatible.get((task.asset_id, day), 0):
                maintenance_conflict_days.add((task.asset_id, day))

    activities_by_day: dict[int, list[PriorityActivity]] = {
        day: [] for day in scenario.days
    }
    for activity in scenario.priority_activities:
        for day in range(activity.start_day, activity.end_day):
            activities_by_day[day].append(activity)

    for day in scenario.days:
        # Calculate effective stage rates (t/hr) with maintenance/outage factors
        stage_rate = {
            stage: sum(
                asset.nominal_rate_tph * capacity_factor[(asset.asset_id, day)]
                for asset in scenario.assets
                if asset.stage == stage
            )
            for stage in STAGES
        }
        # Bottleneck rate determines maximum flow
        inbound_rate = min(stage_rate["train_unloader"], stage_rate["stacker"])
        outbound_rate = min(stage_rate["reclaimer"], stage_rate["shiploader"])

        # Convert rates to daily capacity: rate Ã— hours
        inbound_capacity = inbound_rate * hours_per_day
        outbound_capacity = outbound_rate * hours_per_day

        daily_supply = {p: supply[(p, day)] for p in products}
        supply_total = sum(daily_supply.values())
        accepted_total = min(supply_total, inbound_capacity)
        supply_scale = accepted_total / supply_total if supply_total > 0 else 0.0
        for product in products:
            accepted_product = daily_supply[product] * supply_scale
            inventory[product] += accepted_product
            if product in special_products:
                requirement = requirement_by_product[product]
                zone = requirement.compatible_zones[0]
                special_zone_inventory[(product, zone)] += accepted_product
        unaccepted_supply += supply_total - accepted_total

        demand_remaining = {p: demand[(p, day)] for p in products}
        delivered = {p: 0.0 for p in products}
        capacity_remaining = outbound_capacity

        for activity in activities_by_day[day]:
            requirement = requirement_by_product[activity.product_id]
            route_daily_capacity = min(
                asset_rate[asset_id] * capacity_factor[(asset_id, day)] * hours_per_day
                for asset_id in activity.required_assets
            )
            dedicated_inventory = sum(
                special_zone_inventory[(activity.product_id, zone)]
                for zone in requirement.compatible_zones
            )
            if (
                activity.build_complete_before_reclaim
                and day == activity.start_day
                and dedicated_inventory + 1e-6 < activity.minimum_build_tonnes
            ):
                special_product_build_shortfall[activity.product_id] += (
                    activity.minimum_build_tonnes - dedicated_inventory
                )
            route_factor = min(
                capacity_factor[(asset_id, day)]
                for asset_id in activity.required_assets
            )
            target = activity.minimum_tonnes_per_day
            possible = min(
                target,
                inventory[activity.product_id],
                dedicated_inventory,
                capacity_remaining,
                route_daily_capacity,
                target * route_factor,
            )
            delivered[activity.product_id] += possible
            inventory[activity.product_id] -= possible
            remaining_withdrawal = possible
            for zone in requirement.compatible_zones:
                key = (activity.product_id, zone)
                withdrawal = min(special_zone_inventory[key], remaining_withdrawal)
                special_zone_inventory[key] -= withdrawal
                remaining_withdrawal -= withdrawal
                if remaining_withdrawal <= 1e-9:
                    break
            capacity_remaining -= possible
            demand_remaining[activity.product_id] = max(
                0.0, demand_remaining[activity.product_id] - possible
            )
            activity_shortfall = target - possible
            priority_shortfall += activity_shortfall
            special_product_daily_shortfall[activity.product_id] += activity_shortfall
            campaign_volume[activity.activity_id] += possible

        # Proportional allocation of remaining capacity to normal demand.
        # Allocate based on min(demand_remaining, inventory) for each product.
        # This is deterministic and does not favour any product by ID.
        allocatable = {p: min(demand_remaining[p], inventory[p]) for p in products}
        total_allocatable = sum(allocatable.values())
        if total_allocatable > 0 and capacity_remaining > 0:
            # Scale factor: fraction of allocatable demand we can satisfy
            scale = min(1.0, capacity_remaining / total_allocatable)
            for product in products:
                allocation = allocatable[product] * scale
                delivered[product] += allocation
                inventory[product] -= allocation
                if product in special_products:
                    requirement = requirement_by_product[product]
                    remaining_withdrawal = allocation
                    for zone in requirement.compatible_zones:
                        key = (product, zone)
                        withdrawal = min(
                            special_zone_inventory[key], remaining_withdrawal
                        )
                        special_zone_inventory[key] -= withdrawal
                        remaining_withdrawal -= withdrawal
                        if remaining_withdrawal <= 1e-9:
                            break
            capacity_remaining -= total_allocatable * scale

        day_throughput = sum(delivered.values())
        for product in special_products:
            special_product_throughput[product] += delivered[product]
        day_demand = sum(demand[(p, day)] for p in products)
        total_throughput += day_throughput
        unmet_demand += max(0.0, day_demand - day_throughput)

        # Minimum throughput constraint: compare achieved rate vs minimum rate
        # achieved_rate = day_throughput / hours_per_day
        # min_rate = scenario.minimum_throughput_rate_tph[day]
        # Convert to daily comparison: min_daily = min_rate Ã— hours_per_day
        minimum_daily = scenario.minimum_throughput_rate_tph[day] * hours_per_day
        shortfall = max(0.0, minimum_daily - day_throughput)
        minimum_shortfall += shortfall
        if shortfall > 1e-6:
            minimum_violation_days += 1

        total_inventory = sum(inventory.values())
        if total_inventory > total_yard_capacity:
            excess = total_inventory - total_yard_capacity
            storage_pressure_days += 1
            overflow_tonnes += excess
            unaccepted_supply += excess
            _remove_inventory_proportionally(inventory, excess)
            total_inventory = total_yard_capacity
        maximum_inventory = max(maximum_inventory, total_inventory)

        active_products = {
            activity.product_id
            for activity in scenario.priority_activities
            if (
                activity.start_day - activity.pre_campaign_buffer_days
                <= day
                < activity.end_day + activity.post_campaign_buffer_days
            )
        }
        for product, requirement in requirement_by_product.items():
            if incompatible_special_zone_inventory[product] > 1e-6:
                incompatible_zone_violation_days += 1
            for zone in requirement.compatible_zones:
                tonnes = special_zone_inventory[(product, zone)]
                physical_footprint = (
                    tonnes / requirement.tonnes_per_square_metre
                    + (requirement.segregation_buffer_m2 if tonnes > 1e-6 else 0.0)
                )
                reserved_footprint = (
                    requirement.minimum_reserved_footprint_m2
                    if product in active_products
                    else 0.0
                )
                footprint_used = max(physical_footprint, reserved_footprint)
                maximum_special_footprint = max(
                    maximum_special_footprint, footprint_used
                )
                excess_m2 = max(
                    0.0, footprint_used - zone_footprint_capacity[zone]
                )
                if excess_m2 > 1e-6:
                    footprint_violation_days.add((product, day))
                    footprint_excess += excess_m2

    overflow_penalty = (
        sum(zone.overflow_penalty_per_tonne for zone in scenario.zones)
        / len(scenario.zones)
    )
    penalized_throughput_objective = total_throughput - overflow_penalty * overflow_tonnes
    campaign_volume_shortfall = {
        activity.activity_id: max(
            0.0, activity.minimum_total_tonnes - campaign_volume[activity.activity_id]
        )
        for activity in scenario.priority_activities
    }
    special_campaign_shortfall_by_product = {product: 0.0 for product in special_products}
    for activity in scenario.priority_activities:
        special_campaign_shortfall_by_product[activity.product_id] += (
            campaign_volume_shortfall[activity.activity_id]
        )
    special_product_shortfall = {
        product: (
            special_product_daily_shortfall[product]
            + special_product_build_shortfall[product]
        )
        for product in special_products
    }

    objective_proxy = (
        scenario.objective_parameters["throughput_reward_per_tonne"] * total_throughput
        - scenario.objective_parameters["unmet_demand_penalty_per_tonne"] * unmet_demand
        - scenario.objective_parameters[
            "minimum_throughput_shortfall_penalty_per_tonne"
        ]
        * minimum_shortfall
        - scenario.objective_parameters["unaccepted_supply_penalty_per_tonne"]
        * unaccepted_supply
        - scenario.objective_parameters["priority_shortfall_penalty_per_tonne"]
        * (
            priority_shortfall
            + sum(special_product_build_shortfall.values())
            + sum(campaign_volume_shortfall.values())
        )
    )

    return {
        "total_throughput_tonnes": round(total_throughput, 3),
        "unmet_demand_tonnes": round(unmet_demand, 3),
        "unaccepted_supply_tonnes": round(unaccepted_supply, 3),
        "priority_shortfall_tonnes": round(priority_shortfall, 3),
        "special_product_throughput_tonnes": round(
            sum(special_product_throughput.values()), 3
        ),
        "special_product_shortfall_tonnes": round(
            sum(special_product_shortfall.values()), 3
        ),
        "special_product_campaign_volume_tonnes": round(
            sum(campaign_volume.values()), 3
        ),
        "special_product_campaign_volume_shortfall_tonnes": round(
            sum(campaign_volume_shortfall.values()), 3
        ),
        "special_product_build_shortfall_tonnes": round(
            sum(special_product_build_shortfall.values()), 3
        ),
        "special_product_footprint_violation_days": len(footprint_violation_days),
        "special_product_footprint_excess_m2": round(footprint_excess, 3),
        "special_product_maximum_footprint_m2": round(maximum_special_footprint, 3),
        "special_product_incompatible_zone_violation_days": incompatible_zone_violation_days,
        "special_product_maintenance_conflict_days": len(maintenance_conflict_days),
        "minimum_throughput_shortfall_tonnes": round(minimum_shortfall, 3),
        "minimum_throughput_violation_days": minimum_violation_days,
        "storage_pressure_days": storage_pressure_days,
        "overflow_tonnes": round(overflow_tonnes, 3),
        "penalized_throughput_objective": round(
            penalized_throughput_objective, 3
        ),
        "strict_capacity_feasible": int(
            overflow_tonnes <= 1e-6 and minimum_violation_days == 0
        ),
        "maximum_inventory_tonnes": round(maximum_inventory, 3),
        "ending_inventory_tonnes": round(sum(inventory.values()), 3),
        "objective_proxy": round(objective_proxy, 3),
    }


def _network_neighbours(
    arcs: Sequence[Arc], nodes: Sequence[str]
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Return predecessor and successor node sets used by flow conservation."""
    predecessors = {node: [] for node in nodes}
    successors = {node: [] for node in nodes}
    for arc in arcs:
        predecessors[arc.to_node].append(arc.from_node)
        successors[arc.from_node].append(arc.to_node)
    return predecessors, successors


def _special_activity_profiles(
    scenario: DissertationScenario,
) -> tuple[dict[tuple[str, int], float], dict[tuple[str, int], int]]:
    """Build sigma[a,d] and h[a,d] from fixed high-priority activities."""
    sigma = {
        (asset.asset_id, day): 1.0
        for asset in scenario.assets
        for day in scenario.days
    }
    incompatible = {
        (asset.asset_id, day): 0
        for asset in scenario.assets
        for day in scenario.days
    }
    for activity in scenario.priority_activities:
        for asset_id in activity.required_assets:
            protected_start = activity.start_day - activity.pre_campaign_buffer_days
            protected_end = activity.end_day + activity.post_campaign_buffer_days
            cleanout_before_start = activity.start_day - activity.cleanout_before_days
            cleanout_after_end = activity.end_day + activity.cleanout_after_days
            for day in range(max(0, protected_start), min(len(scenario.days), protected_end)):
                key = (asset_id, day)
                if activity.maintenance_incompatible:
                    incompatible[key] = 1
            for day in range(activity.start_day, activity.end_day):
                key = (asset_id, day)
                sigma[key] = min(
                    sigma[key], activity.capacity_factor_during_activity
                )
            for day in range(cleanout_before_start, activity.start_day):
                if day in scenario.days:
                    sigma[(asset_id, day)] = min(sigma[(asset_id, day)], 0.0)
            for day in range(activity.end_day, cleanout_after_end):
                if day in scenario.days:
                    sigma[(asset_id, day)] = min(sigma[(asset_id, day)], 0.0)
    return sigma, incompatible


def _effective_capacity_factors(
    scenario: DissertationScenario,
    schedule: Sequence[MaintenanceDecision],
) -> dict[tuple[str, int], float]:
    factors = {
        (asset.asset_id, day): 1.0 for asset in scenario.assets for day in scenario.days
    }
    sigma, _ = _special_activity_profiles(scenario)
    for key, activity_factor in sigma.items():
        factors[key] = min(factors[key], activity_factor)

    for outage in scenario.committed_outages:
        for day in range(outage.start_day, outage.end_day):
            key = (outage.asset_id, day)
            factors[key] = min(factors[key], outage.capacity_factor)

    task_by_id = {task.task_id: task for task in scenario.maintenance_tasks}
    for decision in schedule:
        task = task_by_id[decision.task_id]
        mode = next(mode for mode in task.modes if mode.mode_id == decision.mode_id)
        for day in range(decision.start_day, decision.start_day + mode.duration_days):
            key = (task.asset_id, day)
            factors[key] = min(factors[key], mode.capacity_factor_during_work)
    return factors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stable_seed(seed: int, occasion: str) -> int:
    digest = hashlib.sha256(f"{seed}:{occasion}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _day(horizon: int, fraction: float) -> int:
    return min(horizon - 1, max(0, round(fraction * (horizon - 1))))


def _period(
    horizon: int, start_fraction: float, end_fraction: float
) -> tuple[int, int]:
    start = _day(horizon, start_fraction)
    end = min(horizon, max(start + 1, _day(horizon, end_fraction)))
    return start, end


def _nominal_stage_rate(assets: Sequence[Asset]) -> dict[str, float]:
    """Calculate the total nominal rate (t/hr) per processing stage."""
    return {
        stage: sum(a.nominal_rate_tph for a in assets if a.stage == stage)
        for stage in STAGES
    }


def _records_to_dense(
    records: Sequence[DailyProductQuantity],
    products: Sequence[str],
    days: Sequence[int],
) -> dict[tuple[str, int], float]:
    dense = {(p, day): 0.0 for p in products for day in days}
    for record in records:
        dense[(record.product_id, record.day)] += record.tonnes
    return dense


def _json_safe(value: Any) -> Any:
    """Convert tuple-key model dictionaries into stable JSON-safe structures."""
    if isinstance(value, dict):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, tuple):
                safe_key = "::".join(str(part) for part in key)
            else:
                safe_key = str(key)
            converted[safe_key] = _json_safe(item)
        return converted
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _multiply_records(
    records: Sequence[DailyProductQuantity],
    start: int,
    end: int,
    factor: float,
    product_id: str | None = None,
) -> list[DailyProductQuantity]:
    return [
        (
            replace(
                record,
                tonnes=round(record.tonnes * factor, 3),
            )
            if start <= record.day < end
            and (product_id is None or record.product_id == product_id)
            else record
        )
        for record in records
    ]


def _add_product_demand(
    records: Sequence[DailyProductQuantity],
    *,
    product_id: str,
    start: int,
    end: int,
    additional_tpd: float,
) -> list[DailyProductQuantity]:
    return [
        (
            replace(record, tonnes=round(record.tonnes + additional_tpd, 3))
            if record.product_id == product_id and start <= record.day < end
            else record
        )
        for record in records
    ]


def _schedule_map(
    schedule: Sequence[MaintenanceDecision],
) -> dict[str, MaintenanceDecision]:
    return {decision.task_id: decision for decision in schedule}


def _task_by_id(tasks: Sequence[MaintenanceTask], task_id: str) -> MaintenanceTask:
    try:
        return next(task for task in tasks if task.task_id == task_id)
    except StopIteration as exc:
        raise KeyError(f"unknown maintenance task {task_id}") from exc


def _set_decision(
    schedule: dict[str, MaintenanceDecision],
    tasks: Sequence[MaintenanceTask],
    task_id: str,
    preferred_start: int,
    mode_id: str = FULL_SHUTDOWN_MODE,
) -> None:
    task = _task_by_id(tasks, task_id)
    mode = next((m for m in task.modes if m.mode_id == mode_id), None)
    if mode is None:
        raise ValueError(f"task {task_id} does not support mode {mode_id}")
    latest_by_due = task.lifecycle_due_day - mode.duration_days + 1
    latest = min(task.latest_start_day, latest_by_due)
    start = min(max(preferred_start, task.earliest_start_day), latest)
    schedule[task_id] = MaintenanceDecision(task_id, start, mode_id)


def _retime_task(
    task: MaintenanceTask,
    earliest: int,
    latest: int,
    due: int,
) -> MaintenanceTask:
    max_duration = max(mode.duration_days for mode in task.modes)
    latest = max(earliest, latest)
    due = max(due, latest + max_duration - 1)
    return replace(
        task,
        earliest_start_day=earliest,
        latest_start_day=latest,
        lifecycle_due_day=due,
    )


def _remove_inventory_proportionally(
    inventory: dict[str, float], amount: float
) -> None:
    total = sum(inventory.values())
    if total <= 0 or amount <= 0:
        return
    scale = max(0.0, (total - amount) / total)
    for product in inventory:
        inventory[product] *= scale


def _metric_delta(
    reference: Mapping[str, float | int],
    baseline: Mapping[str, float | int],
) -> dict[str, float]:
    return {
        key: round(float(reference[key]) - float(baseline[key]), 3)
        for key in reference
        if isinstance(reference[key], (int, float))
        and isinstance(baseline.get(key), (int, float))
    }


def _occasion_checks(
    occasion: str,
    scenario: DissertationScenario,
    baseline: Mapping[str, float | int],
    reference: Mapping[str, float | int],
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    if occasion == "balanced_control":
        checks["baseline_and_reference_are_identical"] = (
            scenario.manual_baseline == scenario.reference_schedule
        )
    elif occasion == "peak_export_conflict":
        checks["reference_has_more_throughput"] = (
            reference["total_throughput_tonnes"] > baseline["total_throughput_tonnes"]
        )
        checks["reference_has_less_unmet_demand"] = (
            reference["unmet_demand_tonnes"] < baseline["unmet_demand_tonnes"]
        )
    elif occasion in {"overlapping_maintenance", "tight_lifecycle_windows"}:
        checks["reference_has_fewer_minimum_violations"] = (
            reference["minimum_throughput_violation_days"]
            < baseline["minimum_throughput_violation_days"]
        )
    elif occasion == "priority_campaign_conflict":
        checks["reference_reduces_priority_shortfall"] = (
            reference["priority_shortfall_tonnes"]
            < baseline["priority_shortfall_tonnes"]
        )
        checks["baseline_has_special_product_shortfall"] = (
            baseline["special_product_shortfall_tonnes"] > 0
            or baseline["special_product_campaign_volume_shortfall_tonnes"] > 0
        )
        checks["reference_has_no_special_product_shortfall"] = (
            reference["special_product_shortfall_tonnes"] == 0
            and reference["special_product_campaign_volume_shortfall_tonnes"] == 0
        )
        checks["reference_has_no_footprint_violation"] = (
            reference["special_product_footprint_violation_days"] == 0
        )
        checks["reference_has_no_maintenance_conflict"] = (
            reference["special_product_maintenance_conflict_days"] == 0
        )
    elif occasion == "yard_overflow_risk":
        checks["reference_reduces_unaccepted_supply"] = (
            reference["unaccepted_supply_tonnes"] < baseline["unaccepted_supply_tonnes"]
        )
    elif occasion == "inventory_starvation_risk":
        checks["reference_has_more_throughput"] = (
            reference["total_throughput_tonnes"] > baseline["total_throughput_tonnes"]
        )
    elif occasion == "partial_shutdown_choice":
        checks["reference_has_more_throughput"] = (
            reference["total_throughput_tonnes"] > baseline["total_throughput_tonnes"]
        )
        checks["reference_has_fewer_minimum_violations"] = (
            reference["minimum_throughput_violation_days"]
            < baseline["minimum_throughput_violation_days"]
        )
    elif occasion == "fixed_schedule_control":
        checks["baseline_and_reference_are_identical"] = (
            scenario.manual_baseline == scenario.reference_schedule
        )
        checks["metrics_are_identical"] = baseline == reference
    elif occasion == "structurally_infeasible":
        checks["baseline_has_minimum_violation"] = (
            baseline["minimum_throughput_violation_days"] > 0
        )
        checks["reference_has_minimum_violation"] = (
            reference["minimum_throughput_violation_days"] > 0
        )
    return {
        "checks": checks,
        "all_checks_pass": all(checks.values()) if checks else True,
    }


# ---------------------------------------------------------------------------
# Persistence and CLI
# ---------------------------------------------------------------------------


def write_scenario(
    scenario: DissertationScenario,
    output_path: str | Path,
    *,
    include_reference_schedule: bool = True,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            scenario.to_json_dict(
                include_reference_schedule=include_reference_schedule
            ),
            indent=2,
            sort_keys=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def generate_suite(
    output_dir: str | Path,
    config: GeneratorConfig | None = None,
    occasions: Sequence[str] = OCCASION_NAMES,
) -> dict[str, Path]:
    config = config or GeneratorConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    generated: dict[str, Path] = {}
    manifest_cases: list[dict[str, Any]] = []
    for occasion in occasions:
        scenario = build_scenario(occasion, config)
        special_products = [product.product_id for product in scenario.products if product.is_special]
        special_requirement = next(
            req for req in scenario.special_product_requirements if req.product_id == "X1"
        )
        path = write_scenario(scenario, output / f"{occasion}.json")
        generated[occasion] = path
        manifest_cases.append(
            {
                "scenario_name": occasion,
                "file": path.name,
                "description": scenario.metadata["occasion_description"],
                "expected_model_value": scenario.metadata["expected_model_value"],
                "expected_feasibility": scenario.metadata["expected_feasibility"],
                "special_product_ids": special_products,
                "special_product_id": "X1",
                "special_product_compatible_zone": special_requirement.compatible_zones[0],
                "special_product_required_route_assets": list(special_requirement.required_assets),
                "special_product_footprint_assumptions": {
                    "tonnes_per_square_metre": special_requirement.tonnes_per_square_metre,
                    "minimum_reserved_footprint_m2": special_requirement.minimum_reserved_footprint_m2,
                    "segregation_buffer_m2": special_requirement.segregation_buffer_m2,
                },
                "contains_active_protected_campaign": bool(
                    scenario.priority_activities
                ),
                "all_design_checks_pass": scenario.design_evidence["occasion_checks"][
                    "all_checks_pass"
                ],
                "design_check_status": scenario.design_evidence["occasion_checks"],
            }
        )

    manifest = {
        "generator": "generate_dissertation_scenarios.py",
        "mathematical_model_version": "MATH4003 current PDF",
        "horizon_days": config.horizon_days,
        "seed": config.seed,
        "shiploader_count": config.shiploader_count,
        "maintenance_modes": list(MAINTENANCE_MODE_ORDER),
        "special_product_ids": ["X1"],
        "special_product_id": "X1",
        "special_product_synthetic_assumptions": {
            "benchmark_only": True,
            "not_real_operational_data": True,
            "footprint_formula": (
                "special-product inventory tonnes / tonnes_per_square_metre + segregation buffer; "
                "protected campaigns reserve at least minimum_reserved_footprint_m2"
            ),
        },
        "scenario_count": len(manifest_cases),
        "cases": manifest_cases,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    generated["manifest"] = manifest_path
    return generated


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate standalone dissertation maintenance-planning data."
    )
    parser.add_argument(
        "--occasion",
        default="all",
        choices=("all", *OCCASION_NAMES),
        help="Named occasion to generate, or all benchmark occasions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("generated_dissertation_scenarios"),
        help="Directory for generated JSON files.",
    )
    parser.add_argument("--horizon-days", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--minimum-throughput-ratio",
        type=float,
        default=0.55,
    )
    parser.add_argument(
        "--shiploader-count",
        type=int,
        choices=(1, 2),
        default=2,
        help=(
            "Use one shiploader to mirror the current PDF case study, or two "
            "to preserve the original benchmark topology."
        ),
    )
    parser.add_argument(
        "--shiploader-rate-tph",
        type=float,
        default=4583.0,
        help="Nominal hourly rate of each shiploader.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = GeneratorConfig(
        horizon_days=args.horizon_days,
        seed=args.seed,
        minimum_throughput_ratio=args.minimum_throughput_ratio,
        shiploader_count=args.shiploader_count,
        shiploader_rate_tph=args.shiploader_rate_tph,
    )
    occasions = OCCASION_NAMES if args.occasion == "all" else (args.occasion,)
    generated = generate_suite(args.output_dir, config, occasions)
    for name, path in generated.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

