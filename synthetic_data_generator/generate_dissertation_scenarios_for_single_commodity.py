#!/usr/bin/env python3
"""Synthetic data generator for the dissertation optimisation scenarios.

The file is organised as an experimental specification.  Scenario generation is:

    build_scenario(name)
        -> structural/config parameters
        -> family treatment tables
        -> named seeded realization streams
        -> derived model parameters
        -> apply_scenario_family()
        -> validate()
        -> build design evidence
        -> optimizer_objects()
        -> JSON

MODEL PARAMETER MAP

Parameter                         Control source
-----------------------------------------------------------------------
Products G                        structural/config controlled
Assets A, capacities C_a          F1 system-scale controlled
Network arcs E, capacities U_ij   derived from active topology
Yard bounds Zmin/Zmax             F1 scaled or F3 family controlled
Initial inventory z0              F3 target ratios + common seed stream
Supply and demand                 common seed streams
Maintenance operation count       F1 family controlled
Maintenance assets/durations      common maintenance-pool seed stream
Maintenance windows e_m,l_m       common maintenance-window seed stream
Maintenance p_mk                  common full duration + family ratios
Maintenance eta_mk                fixed F/P30/P70 capacity fractions
Minimum throughput mu_min         F3 peak ratio + common throughput stream
Activity sigma_ad and h_ad        F4 family tables + F4 seed stream

Fixed structural/config values include products and grades, product/arc
and product/zone compatibility rules, nominal asset and arc rates,
maintenance capacity factors, and the F2/F3/F4 level definitions.  F1
controls physical system size and maintenance-operation count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections.abc import Mapping as MappingABC
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


# ============================================================
# 1. SCENARIO DEFINITIONS
# ============================================================

BASE_CONTROL_NAME = "base_control"
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
VALIDATION_SCENARIO_NAMES = ("structurally_infeasible",)
SCENARIO_NAMES = (
    BASE_CONTROL_NAME,
    *EXPERIMENTAL_SCENARIO_NAMES,
    *VALIDATION_SCENARIO_NAMES,
)
# Backward-compatible name used by the existing notebook.
OCCASION_NAMES = SCENARIO_NAMES

GENERATOR_RANDOMIZATION_VERSION = "seeded_realization_v1"
RNG_STREAM_VERSION = "named_streams_v1"
RNG_STREAM_NAMES = (
    "common/supply",
    "common/demand",
    "common/initial_inventory",
    "common/maintenance_pool",
    "common/maintenance_windows",
    "common/throughput",
    "F4/activity_windows",
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

MAINTENANCE_MODE_AVAILABLE_CAPACITY = {
    FULL_SHUTDOWN_MODE: 0.0,
    PARTIAL_30_MODE: 0.30,
    PARTIAL_70_MODE: 0.70,
}

BASE_MODE_DURATION_RATIOS = {
    FULL_SHUTDOWN_MODE: 1.00,
    PARTIAL_30_MODE: 1.50,
    PARTIAL_70_MODE: 3.05,
}
F2_MODE_DURATION_RATIOS = {
    "L": {
        FULL_SHUTDOWN_MODE: 1.00,
        PARTIAL_30_MODE: 1.35,
        PARTIAL_70_MODE: 2.45,
    },
    "M": {
        FULL_SHUTDOWN_MODE: 1.00,
        PARTIAL_30_MODE: 1.50,
        PARTIAL_70_MODE: 3.00,
    },
    "H": {
        FULL_SHUTDOWN_MODE: 1.00,
        PARTIAL_30_MODE: 1.75,
        PARTIAL_70_MODE: 3.45,
    },
}

SYSTEM_SCALE_PROFILES = {
    "L": {
        "train_unloader": 2,
        "stacker": 3,
        "reclaimer": 2,
        "shiploader": 1,
    },
    "M": {
        "train_unloader": 3,
        "stacker": 4,
        "reclaimer": 3,
        "shiploader": 2,
    },
    "H": {
        "train_unloader": 4,
        "stacker": 5,
        "reclaimer": 4,
        "shiploader": 3,
    },
}

BASE_ZONE_CAPACITIES_TONNES = {
    "Z1": 650_000.0,
    "Z2": 750_000.0,
    "Z3": 580_000.0,
    "Z4": 500_000.0,
}
BASE_ZONE_MINIMUM_INVENTORY_RATIOS = {
    "Z1": 0.070,
    "Z2": 0.060,
    "Z3": 0.060,
    "Z4": 0.065,
}
BASE_INITIAL_INVENTORY_RATIOS = {
    "Z1": 0.220,
    "Z2": 0.250,
    "Z3": 0.180,
    "Z4": 0.160,
}
ZONE_ALLOWED_PRODUCTS = {
    "Z1": ("P1", "P2"),
    "Z2": ("P1", "P2", "P3"),
    "Z3": ("P2", "P3"),
    "Z4": ("X1",),
}
ZONE_FOOTPRINT_CAPACITY_M2 = {
    "Z1": 260_000.0,
    "Z2": 240_000.0,
    "Z3": 220_000.0,
    "Z4": 185_000.0,
}
ARC_ALLOWED_ALL_PRODUCTS = "__all_products__"
NETWORK_ARC_LAYERS = {
    "L": (
        ("RAIL_SOURCE", "TU1", "train_unloader", ARC_ALLOWED_ALL_PRODUCTS),
        ("RAIL_SOURCE", "TU2", "train_unloader", ARC_ALLOWED_ALL_PRODUCTS),
        ("TU1", "ST1", "stacker", ("P1", "P2", "P3")),
        ("TU2", "ST2", "stacker", ("P1", "P2", "P3")),
        ("TU1", "ST2", "secondary", ("P1", "P2", "P3")),
        ("TU2", "ST3", "stacker", ("P2", "P3", "X1")),
        ("TU1", "ST3", "secondary", ("P2", "P3", "X1")),
        ("ST1", "Z1", "stacker", ZONE_ALLOWED_PRODUCTS["Z1"]),
        ("ST1", "Z2", "secondary", ZONE_ALLOWED_PRODUCTS["Z2"]),
        ("ST2", "Z2", "stacker", ZONE_ALLOWED_PRODUCTS["Z2"]),
        ("ST2", "Z3", "secondary", ZONE_ALLOWED_PRODUCTS["Z3"]),
        ("ST3", "Z3", "stacker", ZONE_ALLOWED_PRODUCTS["Z3"]),
        ("ST3", "Z4", "stacker", ZONE_ALLOWED_PRODUCTS["Z4"]),
        ("Z1", "RC1", "reclaimer", ZONE_ALLOWED_PRODUCTS["Z1"]),
        ("Z1", "RC2", "secondary", ZONE_ALLOWED_PRODUCTS["Z1"]),
        ("Z2", "RC2", "reclaimer", ZONE_ALLOWED_PRODUCTS["Z2"]),
        ("Z3", "RC2", "reclaimer", ZONE_ALLOWED_PRODUCTS["Z3"]),
        ("Z4", "RC2", "reclaimer", ZONE_ALLOWED_PRODUCTS["Z4"]),
        ("RC1", "SL1", "reclaimer", ARC_ALLOWED_ALL_PRODUCTS),
        ("RC2", "SL1", "reclaimer", ARC_ALLOWED_ALL_PRODUCTS),
        ("SL1", "VESSEL_SINK", "shiploader", ARC_ALLOWED_ALL_PRODUCTS),
    ),
    "M": (
        ("RAIL_SOURCE", "TU3", "train_unloader", ARC_ALLOWED_ALL_PRODUCTS),
        ("TU3", "ST4", "stacker", ("P2", "P3", "X1")),
        ("TU2", "ST4", "secondary", ("P2", "P3", "X1")),
        ("ST4", "Z3", "secondary", ZONE_ALLOWED_PRODUCTS["Z3"]),
        ("ST4", "Z4", "stacker", ZONE_ALLOWED_PRODUCTS["Z4"]),
        ("Z2", "RC3", "secondary", ZONE_ALLOWED_PRODUCTS["Z2"]),
        ("Z3", "RC3", "reclaimer", ZONE_ALLOWED_PRODUCTS["Z3"]),
        ("Z4", "RC3", "secondary", ZONE_ALLOWED_PRODUCTS["Z4"]),
        ("RC2", "SL2", "secondary", ARC_ALLOWED_ALL_PRODUCTS),
        ("RC3", "SL1", "secondary", ARC_ALLOWED_ALL_PRODUCTS),
        ("RC3", "SL2", "reclaimer", ARC_ALLOWED_ALL_PRODUCTS),
        ("SL2", "VESSEL_SINK", "shiploader", ARC_ALLOWED_ALL_PRODUCTS),
    ),
    "H": (
        ("RAIL_SOURCE", "TU4", "train_unloader", ARC_ALLOWED_ALL_PRODUCTS),
        ("TU4", "ST5", "stacker", ("P2", "P3", "X1")),
        ("TU3", "ST5", "secondary", ("P2", "P3", "X1")),
        ("ST5", "Z3", "secondary", ZONE_ALLOWED_PRODUCTS["Z3"]),
        ("ST5", "Z4", "stacker", ZONE_ALLOWED_PRODUCTS["Z4"]),
        ("Z2", "RC4", "secondary", ZONE_ALLOWED_PRODUCTS["Z2"]),
        ("Z3", "RC4", "secondary", ZONE_ALLOWED_PRODUCTS["Z3"]),
        ("Z4", "RC4", "reclaimer", ZONE_ALLOWED_PRODUCTS["Z4"]),
        ("RC3", "SL3", "secondary", ARC_ALLOWED_ALL_PRODUCTS),
        ("RC4", "SL2", "secondary", ARC_ALLOWED_ALL_PRODUCTS),
        ("RC4", "SL3", "reclaimer", ARC_ALLOWED_ALL_PRODUCTS),
        ("SL3", "VESSEL_SINK", "shiploader", ARC_ALLOWED_ALL_PRODUCTS),
    ),
}
F3_YARD_FLOW_LEVELS = {
    "L": {
        "zone_capacities_tonnes": {
            "Z1": 690_000.0,
            "Z2": 790_000.0,
            "Z3": 610_000.0,
            "Z4": 500_000.0,
        },
        "minimum_inventory_ratios": {
            "Z1": 0.050,
            "Z2": 0.052,
            "Z3": 0.050,
            "Z4": 0.055,
        },
        "initial_inventory_ratios": {
            "Z1": 0.280,
            "Z2": 0.300,
            "Z3": 0.260,
            "Z4": 0.220,
        },
        "throughput_peak_ratio": 0.70,
    },
    "M": {
        "zone_capacities_tonnes": BASE_ZONE_CAPACITIES_TONNES,
        "minimum_inventory_ratios": {
            "Z1": 0.070,
            "Z2": 0.070,
            "Z3": 0.068,
            "Z4": 0.070,
        },
        "initial_inventory_ratios": {
            "Z1": 0.220,
            "Z2": 0.245,
            "Z3": 0.190,
            "Z4": 0.170,
        },
        "throughput_peak_ratio": 0.80,
    },
    "H": {
        "zone_capacities_tonnes": {
            "Z1": 620_000.0,
            "Z2": 720_000.0,
            "Z3": 560_000.0,
            "Z4": 500_000.0,
        },
        "minimum_inventory_ratios": {
            "Z1": 0.090,
            "Z2": 0.085,
            "Z3": 0.092,
            "Z4": 0.090,
        },
        "initial_inventory_ratios": {
            "Z1": 0.160,
            "Z2": 0.185,
            "Z3": 0.145,
            "Z4": 0.160,
        },
        "throughput_peak_ratio": 0.885,
    },
}

BASE_THROUGHPUT_PEAK_RATIO = 0.855
THROUGHPUT_PROFILE_JITTER_FRACTION = 0.006
THROUGHPUT_SEGMENT_END_DAY_FOR_90_DAY_HORIZON = (15, 35, 50, 65, 78, 90)
THROUGHPUT_SEGMENT_RATIOS_BY_PEAK_CLASS = {
    "LOW_PEAK": (0.525, 0.635, "PEAK", 0.595, 0.670, 0.575),
    "MEDIUM_PEAK": (0.530, 0.670, "PEAK", 0.625, 0.765, 0.600),
    "HIGH_PEAK": (0.535, 0.685, "PEAK", 0.635, 0.795, 0.610),
}

INITIAL_INVENTORY_FILL_RATIO_JITTER = 0.012
INITIAL_INVENTORY_PRODUCT_MIX_JITTER_FRACTION = 0.08

MAINTENANCE_LEVEL_ORDER = {"L": 0, "M": 1, "H": 2}
MAINTENANCE_WINDOW_EM_JITTER_DAYS = 4
MAINTENANCE_WINDOW_DUE_BUFFER_RANGE_DAYS = (0, 3)
MAINTENANCE_WINDOW_SAFETY_DURATION_RATIO = max(
    max(BASE_MODE_DURATION_RATIOS.values()),
    max(value for ratios in F2_MODE_DURATION_RATIOS.values() for value in ratios.values()),
)

# The maintenance template pool preserves the old stage composition and
# effective counts.  entry_level defines the nested F1 pool:
# L subset M subset H.  Durations/windows are calibrated around the old
# calendar placements but realized by the seed.
MAINTENANCE_TEMPLATE_POOL = (
    {
        "template_id": "LOW_TU_EARLY",
        "entry_level": "L",
        "stage": "train_unloader",
        "task_class": "major",
        "calendar_region": (10, 30),
        "full_duration_candidates": (4, 5, 6),
        "window_width_range": (18, 24),
    },
    {
        "template_id": "LOW_TU_MID",
        "entry_level": "L",
        "stage": "train_unloader",
        "task_class": "major",
        "calendar_region": (14, 36),
        "full_duration_candidates": (4, 5, 6),
        "window_width_range": (20, 26),
    },
    {
        "template_id": "LOW_ST_EARLY",
        "entry_level": "L",
        "stage": "stacker",
        "task_class": "modular",
        "calendar_region": (18, 32),
        "full_duration_candidates": (3, 4, 5),
        "window_width_range": (12, 17),
    },
    {
        "template_id": "LOW_ST_LATE",
        "entry_level": "L",
        "stage": "stacker",
        "task_class": "modular",
        "calendar_region": (55, 62),
        "full_duration_candidates": (2, 3, 4),
        "window_width_range": (7, 10),
    },
    {
        "template_id": "LOW_RC_EARLY",
        "entry_level": "L",
        "stage": "reclaimer",
        "task_class": "major",
        "calendar_region": (24, 38),
        "full_duration_candidates": (4, 5, 6),
        "window_width_range": (12, 17),
    },
    {
        "template_id": "LOW_RC_MID",
        "entry_level": "L",
        "stage": "reclaimer",
        "task_class": "major",
        "calendar_region": (35, 55),
        "full_duration_candidates": (5, 6, 7),
        "window_width_range": (18, 23),
    },
    {
        "template_id": "LOW_RC_LATE",
        "entry_level": "L",
        "stage": "reclaimer",
        "task_class": "major",
        "calendar_region": (60, 67),
        "full_duration_candidates": (2, 3, 4),
        "window_width_range": (7, 10),
    },
    {
        "template_id": "LOW_SL",
        "entry_level": "L",
        "stage": "shiploader",
        "task_class": "major",
        "calendar_region": (42, 57),
        "full_duration_candidates": (3, 4),
        "window_width_range": (13, 18),
    },
    {
        "template_id": "M_ST_MID",
        "entry_level": "M",
        "stage": "stacker",
        "task_class": "modular",
        "calendar_region": (30, 42),
        "full_duration_candidates": (3, 4, 5),
        "window_width_range": (11, 15),
    },
    {
        "template_id": "M_TU_LATE",
        "entry_level": "M",
        "stage": "train_unloader",
        "task_class": "major",
        "calendar_region": (48, 70),
        "full_duration_candidates": (3, 4, 5),
        "window_width_range": (19, 24),
    },
    {
        "template_id": "M_ST_X1",
        "entry_level": "M",
        "stage": "stacker",
        "task_class": "modular",
        "calendar_region": (44, 58),
        "full_duration_candidates": (3, 4, 5),
        "window_width_range": (12, 17),
    },
    {
        "template_id": "M_SL_SECOND",
        "entry_level": "M",
        "stage": "shiploader",
        "task_class": "major",
        "calendar_region": (70, 77),
        "full_duration_candidates": (2, 3),
        "window_width_range": (7, 10),
        "min_shiploader_count": 2,
    },
    {
        "template_id": "H_RC_EARLY",
        "entry_level": "H",
        "stage": "reclaimer",
        "task_class": "follow_up",
        "calendar_region": (8, 24),
        "full_duration_candidates": (3, 4, 5),
        "window_width_range": (14, 19),
    },
    {
        "template_id": "H_ST_MID",
        "entry_level": "H",
        "stage": "stacker",
        "task_class": "follow_up",
        "calendar_region": (22, 39),
        "full_duration_candidates": (2, 3, 4),
        "window_width_range": (15, 20),
    },
    {
        "template_id": "H_TU_MID",
        "entry_level": "H",
        "stage": "train_unloader",
        "task_class": "follow_up",
        "calendar_region": (34, 53),
        "full_duration_candidates": (3, 4, 5),
        "window_width_range": (17, 22),
    },
    {
        "template_id": "H_SL_LATE",
        "entry_level": "H",
        "stage": "shiploader",
        "task_class": "follow_up",
        "calendar_region": (54, 77),
        "full_duration_candidates": (2, 3),
        "window_width_range": (20, 25),
    },
)

F4_LEVEL_SETTINGS = {
    "L": {
        "event_1_incompatible": False,
        "event_2_incompatible": False,
        "event_2_capacity_factor": 0.90,
        "event_3_capacity_factor": 0.92,
        "buffer_days": 0,
        "include_event_3": False,
    },
    "M": {
        "event_1_incompatible": False,
        "event_2_incompatible": True,
        "event_2_capacity_factor": 0.86,
        "event_3_capacity_factor": 0.88,
        "buffer_days": 0,
        "include_event_3": True,
    },
    "H": {
        "event_1_incompatible": False,
        "event_2_incompatible": True,
        "event_2_capacity_factor": 0.82,
        "event_3_capacity_factor": 0.84,
        "buffer_days": 0,
        "include_event_3": True,
    },
}
F4_ACTIVITY_WINDOWS = [
    {
        "activity_id": "PA_EXPORT_WINDOW_1",
        "start_day_range": (28, 32),
        "duration_range": (6, 8),
        "asset_group": "DOWNSTREAM_EXPORT",
        "capacity_factor": 0.85,
        "maintenance_incompatible_from_level": "event_1_incompatible",
        "buffer_from_level_when_incompatible": True,
    },
    {
        "activity_id": "PA_X1_ROUTE_WINDOW",
        "start_day_range": (46, 50),
        "duration_range": (8, 10),
        "asset_group": "X1_ROUTE",
        "capacity_factor_from_level": "event_2_capacity_factor",
        "maintenance_incompatible_from_level": "event_2_incompatible",
        "buffer_from_level": True,
    },
    {
        "activity_id": "PA_EXPORT_WINDOW_3",
        "start_day_range": (66, 70),
        "duration_range": (6, 8),
        "asset_group": "DOWNSTREAM_EXPORT",
        "capacity_factor_from_level": "event_3_capacity_factor",
        "maintenance_incompatible": False,
        "include_from_level": "include_event_3",
    },
]
F4_MAINTENANCE_PRE_ACTIVITY_RANGE_DAYS = (10, 14)
F4_MAINTENANCE_POST_ACTIVITY_RANGE_DAYS = (12, 16)
F4_MAINTENANCE_TARGETS_BY_ACTIVITY = {
    "PA_EXPORT_WINDOW_1": ("downstream_export", 2),
    "PA_X1_ROUTE_WINDOW": ("x1_route_upstream", 2),
    "PA_EXPORT_WINDOW_3": ("downstream_export", 1),
}
STRUCTURAL_INFEASIBLE_MIN_THROUGHPUT_RATE_TPH = 12_000.0


# ============================================================
# 2. DATA CLASSES
# ============================================================


@dataclass(frozen=True)
class GeneratorConfig:
    """Controls the size and random variation of generated instances."""

    horizon_days: int = 90
    seed: int = 42
    base_supply_tpd: float = 178_000.0
    base_demand_tpd: float = 184_000.0
    noise_fraction: float = 0.035
    number_of_products: int = 4
    operating_hours_per_day: float = 24.0  # Hours of operation per day

    # Asset flow rates in tonnes per hour (t/hr)
    train_unloader_rate_tph: float = 2917.0
    stacker_rate_tph: float = 2292.0
    reclaimer_rate_tph: float = 2708.0
    shiploader_rate_tph: float = 4583.0

    # Arc flow rates in tonnes per hour (t/hr) - secondary connections
    arc_secondary_rate_tph: float = 1875.0  # For cross-connections between stages

    def validate(self) -> None:
        if self.horizon_days < 60:
            raise ValueError("horizon_days must be at least 60 for these scenarios")
        if self.number_of_products != 4:
            raise ValueError("the benchmark currently defines exactly four products")
        if not 0.0 <= self.noise_fraction <= 0.25:
            raise ValueError("noise_fraction must be between 0 and 0.25")
        if not 0.0 < self.operating_hours_per_day <= 24.0:
            raise ValueError("operating_hours_per_day must be in (0, 24]")
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

    # ============================================================
    # 7. VALIDATION
    # ============================================================

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
        expected_mode_order: tuple[str, ...] | None = None
        for task in self.maintenance_tasks:
            if task.task_id in task_ids:
                raise ValueError(f"duplicate task ID {task.task_id}")
            task_ids.add(task.task_id)
            task_by_id[task.task_id] = task
            if task.asset_id not in asset_ids:
                raise ValueError(f"task {task.task_id} references unknown asset")
            if not task.modes:
                raise ValueError(f"task {task.task_id} has no maintenance mode")
            mode_order = tuple(m.mode_id for m in task.modes)
            mode_ids = set(mode_order)
            if len(mode_ids) != len(mode_order):
                raise ValueError(f"task {task.task_id} has duplicate modes")
            if not mode_ids.issubset(PDF_MODE_SYMBOL):
                raise ValueError(f"task {task.task_id} has unsupported modes")
            if expected_mode_order is None:
                expected_mode_order = mode_order
            elif mode_order != expected_mode_order:
                raise ValueError(
                    f"task {task.task_id} uses a different mode set/order"
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

    def to_json_dict(
    self,
    include_reference_schedule: bool = False,
) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the scenario."""
        self.validate()

        payload = asdict(self)
        payload["model_ready_data"] = _json_safe(self.optimizer_objects())

        if not include_reference_schedule:
            # Construction/validation schedules must not be supplied to MILP or BD.
            payload.pop("manual_baseline", None)
            payload.pop("reference_schedule", None)
            payload.pop("design_evidence", None)

            model_ready = payload.get("model_ready_data")
            if isinstance(model_ready, dict):
                model_ready.pop("manual_baseline", None)
                model_ready.pop("reference_schedule", None)
                model_ready.pop("design_evidence", None)

        return payload

    # ============================================================
    # 6. MODEL EXPORT
    # ============================================================

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
        scenario_mode_order = get_scenario_mode_order(self.maintenance_tasks)
        modes = {
            task.task_id: [mode.mode_id for mode in task.modes]
            for task in self.maintenance_tasks
        }

        supply = build_aggregate_daily_quantity_map(
            self.inbound_supply,
            self.days,
        )

        demand = build_aggregate_daily_quantity_map(
            self.outbound_demand,
            self.days,
        )
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
        initial_inventory = build_aggregate_initial_inventory_map(
            self.initial_inventory
        )
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
        sigma, incompatible = build_operational_activity_profiles(self)
        predecessors, successors = build_network_adjacency_maps(self.arcs, nodes)
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
            "K": list(scenario_mode_order),
            "K_symbol": {
                mode_id: PDF_MODE_SYMBOL[mode_id] for mode_id in scenario_mode_order
            },
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
            # "manual_baseline": {
            #     decision.task_id: {
            #         "start_day": decision.start_day,
            #         "mode_id": decision.mode_id,
            #     }
            #     for decision in self.manual_baseline
            # },
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


# ============================================================
# 3. MAIN SCENARIO BUILDER
# ============================================================


def get_scenario_mode_order(tasks: Sequence[MaintenanceTask]) -> tuple[str, ...]:
    if not tasks:
        return MAINTENANCE_MODE_ORDER
    return tuple(mode.mode_id for mode in tasks[0].modes)


def build_scenario(
    scenario_name: str,
    config: GeneratorConfig | None = None,
) -> DissertationScenario:
    """Build one complete synthetic scenario."""
    config = config or GeneratorConfig()
    config.validate()
    if scenario_name not in SCENARIO_NAMES:
        raise ValueError(
            f"unknown scenario {scenario_name!r}; choose one of {', '.join(SCENARIO_NAMES)}"
        )

    rng_streams = make_rng_streams(config.seed)
    scenario = build_base_scenario(config, scenario_name, rng_streams)
    scenario = apply_scenario_family(scenario, scenario_name, config, rng_streams)
    update_realization_metadata(scenario, config)
    scenario.validate()

    baseline_metrics = evaluate_schedule(scenario, scenario.manual_baseline)
    reference_metrics = evaluate_schedule(scenario, scenario.reference_schedule)
    scenario.design_evidence = {
        "purpose": (
            "Construction check only. The reference schedule is a known witness, "
            "not an optimal solution. Final claims must compare the solver result "
            "against the manual baseline."
        ),
        "effective_network_capacity_tph": scenario.metadata[
            "effective_network_capacity_tph"
        ],
        "reference_effective_network_capacity_tph": scenario.metadata[
            "reference_effective_network_capacity_tph"
        ],
        "system_scale_factor": scenario.metadata["system_scale_factor"],
        "manual_baseline_metrics": baseline_metrics,
        "reference_schedule_metrics": reference_metrics,
        "reference_minus_baseline": _metric_delta(reference_metrics, baseline_metrics),
        "occasion_checks": validate_scenario_design(
            scenario_name, scenario, baseline_metrics, reference_metrics
        ),
    }
    return scenario


def get_scenario_level(scenario_name: str) -> str:
    return (
        scenario_name.rsplit("_", 1)[-1]
        if scenario_name.endswith(("_L", "_M", "_H"))
        else ""
    )


def get_system_scale_level(scenario_name: str) -> str:
    return (
        get_scenario_level(scenario_name)
        if scenario_name.startswith("F1_system_scale_")
        else "M"
    )


def get_system_scale_profile(scenario_name: str) -> dict[str, int]:
    return dict(SYSTEM_SCALE_PROFILES[get_system_scale_level(scenario_name)])


def get_maintenance_mode_order_and_duration_ratios(
    scenario_name: str,
) -> tuple[tuple[str, ...], dict[str, float]]:
    if scenario_name.startswith("F2_mode_tradeoff_"):
        return (
            MAINTENANCE_MODE_ORDER,
            F2_MODE_DURATION_RATIOS[get_scenario_level(scenario_name)],
        )
    return MAINTENANCE_MODE_ORDER, BASE_MODE_DURATION_RATIOS


def get_maintenance_mode_available_capacity_fraction(mode_id: str) -> float:
    return MAINTENANCE_MODE_AVAILABLE_CAPACITY[mode_id]


# ============================================================
# 4. COMMON BASE DATA
# ============================================================


def get_zone_capacities_tonnes(
    scenario_name: str,
    system_scale_factor: float = 1.0,
) -> dict[str, float]:
    if scenario_name.startswith("F3_yard_flow_"):
        return dict(F3_YARD_FLOW_LEVELS[get_scenario_level(scenario_name)]["zone_capacities_tonnes"])
    capacities = dict(BASE_ZONE_CAPACITIES_TONNES)
    if scenario_name.startswith("F1_system_scale_"):
        capacities = {
            zone_id: round(capacity * system_scale_factor, 3)
            for zone_id, capacity in capacities.items()
        }
    return capacities


def get_zone_minimum_inventory_ratios(scenario_name: str) -> dict[str, float]:
    if scenario_name.startswith("F3_yard_flow_"):
        return dict(F3_YARD_FLOW_LEVELS[get_scenario_level(scenario_name)]["minimum_inventory_ratios"])
    return dict(BASE_ZONE_MINIMUM_INVENTORY_RATIOS)


def get_zone_initial_inventory_ratios(scenario_name: str) -> dict[str, float]:
    if scenario_name.startswith("F3_yard_flow_"):
        return dict(F3_YARD_FLOW_LEVELS[get_scenario_level(scenario_name)]["initial_inventory_ratios"])
    return dict(BASE_INITIAL_INVENTORY_RATIOS)


def build_yard_zones(
    scenario_name: str,
    system_scale_factor: float = 1.0,
) -> list[YardZone]:
    capacities = get_zone_capacities_tonnes(scenario_name, system_scale_factor)
    minimum_ratios = get_zone_minimum_inventory_ratios(scenario_name)
    footprint_scale = system_scale_factor if scenario_name.startswith("F1_system_scale_") else 1.0
    return [
        YardZone(
            zone_id,
            capacities[zone_id],
            ZONE_ALLOWED_PRODUCTS[zone_id],
            round(ZONE_FOOTPRINT_CAPACITY_M2[zone_id] * footprint_scale, 3),
            round(capacities[zone_id] * minimum_ratios[zone_id], 3),
        )
        for zone_id in ("Z1", "Z2", "Z3", "Z4")
    ]


def get_minimum_throughput_peak_ratio(scenario_name: str) -> float:
    if scenario_name.startswith("F3_yard_flow_"):
        return float(F3_YARD_FLOW_LEVELS[get_scenario_level(scenario_name)]["throughput_peak_ratio"])
    return BASE_THROUGHPUT_PEAK_RATIO


def build_minimum_throughput_profile(
    *,
    days: Sequence[int],
    nominal_system_rate_tph: float,
    peak_ratio: float,
    rng: random.Random,
) -> dict[int, float]:
    horizon = len(days)
    boundaries = [
        max(index + 1, round(horizon * end_day / 90))
        for index, end_day in enumerate(THROUGHPUT_SEGMENT_END_DAY_FOR_90_DAY_HORIZON[:-1])
    ]
    boundaries.append(horizon)
    if peak_ratio <= 0.72:
        segment_ratios = THROUGHPUT_SEGMENT_RATIOS_BY_PEAK_CLASS["LOW_PEAK"]
    elif peak_ratio <= 0.82:
        segment_ratios = THROUGHPUT_SEGMENT_RATIOS_BY_PEAK_CLASS["MEDIUM_PEAK"]
    else:
        segment_ratios = THROUGHPUT_SEGMENT_RATIOS_BY_PEAK_CLASS["HIGH_PEAK"]

    profile: dict[int, float] = {}
    for day in days:
        segment = next(index for index, upper in enumerate(boundaries) if day < upper)
        base_ratio = segment_ratios[segment]
        segment_ratio = peak_ratio if base_ratio == "PEAK" else float(base_ratio)
        deterministic_variation = (
            0.008 * math.sin(2.0 * math.pi * day / 9.0)
            + 0.004 * math.sin(2.0 * math.pi * day / 17.0 + 0.7)
        )
        seeded_variation = rng.uniform(
            -THROUGHPUT_PROFILE_JITTER_FRACTION,
            THROUGHPUT_PROFILE_JITTER_FRACTION,
        )
        ratio = max(
            0.0,
            min(0.95, segment_ratio + deterministic_variation + seeded_variation),
        )
        profile[day] = round(nominal_system_rate_tph * ratio, 3)
    return profile


def build_products() -> list[Product]:
    normal_1_grades = (
        GradeSpec("FE", 57.5, 57.0, 58.0),
        GradeSpec("SIO2", 5.5, 5.0, 6.0),
        GradeSpec("AL2O3", 2.1, 1.8, 2.4),
        GradeSpec("MN", 0.75, 0.4, 1.1),
        GradeSpec("P", 0.063, 0.055, 0.070),
        GradeSpec("AS", 0.005, 0.002, 0.010),
        GradeSpec("LOI", 7.7, 7.2, 8.2),
    )
    normal_2_grades = (
        GradeSpec("FE", 56.8, 56.3, 57.3),
        GradeSpec("SIO2", 5.9, 5.5, 6.5),
        GradeSpec("AL2O3", 2.7, 2.4, 3.2),
        GradeSpec("MN", 0.80, 0.5, 1.3),
        GradeSpec("P", 0.061, 0.055, 0.068),
        GradeSpec("AS", 0.006, 0.002, 0.012),
        GradeSpec("LOI", 7.4, 7.0, 8.0),
    )
    normal_3_grades = (
        GradeSpec("FE", 60.3, 59.7, 60.8),
        GradeSpec("SIO2", 4.3, 3.8, 4.8),
        GradeSpec("AL2O3", 2.4, 2.1, 2.7),
        GradeSpec("MN", 0.06, 0.04, 0.09),
        GradeSpec("P", 0.133, 0.120, 0.145),
        GradeSpec("AS", 0.003, 0.002, 0.005),
        GradeSpec("LOI", 5.3, 5.0, 5.7),
    )
    special_x1_grades = (
        GradeSpec("FE", 57.2, 56.8, 57.6),
        GradeSpec("SIO2", 5.7, 5.2, 6.2),
        GradeSpec("AL2O3", 1.8, 1.6, 2.0),
        GradeSpec("MN", 0.025, 0.020, 0.035),
        GradeSpec("P", 0.072, 0.065, 0.080),
        GradeSpec("AS", 0.003, 0.002, 0.005),
        GradeSpec("LOI", 9.6, 9.0, 10.2),
    )
    return [
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


def build_assets(config: GeneratorConfig, scenario_name: str) -> list[Asset]:
    return build_assets_for_profile(config, get_system_scale_profile(scenario_name))


def build_assets_for_profile(
    config: GeneratorConfig,
    profile: Mapping[str, int],
) -> list[Asset]:
    asset_specs = (
        ("TU", "train_unloader", config.train_unloader_rate_tph, "Train unloader"),
        ("ST", "stacker", config.stacker_rate_tph, "Stacker"),
        ("RC", "reclaimer", config.reclaimer_rate_tph, "Reclaimer"),
        ("SL", "shiploader", config.shiploader_rate_tph, "Shiploader"),
    )
    assets: list[Asset] = []
    for prefix, stage, rate, label in asset_specs:
        for index in range(1, profile[stage] + 1):
            assets.append(
                Asset(
                    f"{prefix}{index}",
                    stage,
                    rate,
                    f"{label} {index}",
                )
            )
    return assets


def calculate_nominal_stage_rate_for_profile(
    config: GeneratorConfig,
    profile: Mapping[str, int],
) -> dict[str, float]:
    rate_by_stage = {
        "train_unloader": config.train_unloader_rate_tph,
        "stacker": config.stacker_rate_tph,
        "reclaimer": config.reclaimer_rate_tph,
        "shiploader": config.shiploader_rate_tph,
    }
    return {
        stage: profile[stage] * rate_by_stage[stage]
        for stage in STAGES
    }


def calculate_nominal_system_rate_for_profile(
    config: GeneratorConfig,
    profile: Mapping[str, int],
) -> float:
    return min(calculate_nominal_stage_rate_for_profile(config, profile).values())


def get_reference_nominal_system_rate_tph(config: GeneratorConfig) -> float:
    return calculate_nominal_system_rate_for_profile(
        config,
        SYSTEM_SCALE_PROFILES["M"],
    )


def get_reference_effective_network_capacity_tph(
    config: GeneratorConfig,
    products: Sequence[str] | None = None,
) -> float:
    product_ids = tuple(products) if products is not None else tuple(
        product.product_id for product in build_products()
    )
    assets = build_assets_for_profile(config, SYSTEM_SCALE_PROFILES["M"])
    arcs = build_network_arcs(product_ids, assets, config)
    return calculate_effective_network_capacity_tph(assets, arcs)


def get_system_scale_factor(
    config: GeneratorConfig,
    scenario_name: str,
    products: Sequence[str] | None = None,
) -> float:
    product_ids = tuple(products) if products is not None else tuple(
        product.product_id for product in build_products()
    )
    assets = build_assets(config, scenario_name)
    arcs = build_network_arcs(product_ids, assets, config)
    reference_capacity = get_reference_effective_network_capacity_tph(
        config,
        product_ids,
    )
    return calculate_effective_network_capacity_tph(assets, arcs) / reference_capacity


def build_base_scenario(
    config: GeneratorConfig,
    scenario_name: str,
    rng_streams: Mapping[str, random.Random],
) -> DissertationScenario:
    products = build_products()
    product_ids = tuple(p.product_id for p in products)

    assets = build_assets(config, scenario_name)
    arcs = build_network_arcs(product_ids, assets, config)
    effective_network_capacity = calculate_effective_network_capacity_tph(assets, arcs)
    reference_effective_network_capacity = get_reference_effective_network_capacity_tph(
        config,
        product_ids,
    )
    system_scale_factor = effective_network_capacity / reference_effective_network_capacity

    zones = build_yard_zones(scenario_name, system_scale_factor)
    special_requirements = build_special_product_requirements(
    config,
    assets,
    system_scale_factor=system_scale_factor,
    )
    days = list(range(config.horizon_days))

    supply = build_daily_product_profile(
        days=days,
        products=products,
        base_total=config.base_supply_tpd * system_scale_factor,
        noise_fraction=config.noise_fraction,
        rng=rng_streams["common/supply"],
        phase=0.4,
    )
    demand = build_daily_product_profile(
        days=days,
        products=products,
        base_total=config.base_demand_tpd * system_scale_factor,
        noise_fraction=config.noise_fraction,
        rng=rng_streams["common/demand"],
        phase=2.1,
    )

    initial_inventory = build_initial_inventory(
        zones,
        products,
        get_zone_initial_inventory_ratios(scenario_name),
        rng=rng_streams["common/initial_inventory"],
    )

    mode_order, mode_ratios = get_maintenance_mode_order_and_duration_ratios(
        scenario_name
    )
    tasks = build_maintenance_tasks(
        config.horizon_days,
        scenario_name=scenario_name,
        mode_order=mode_order,
        mode_duration_ratios=mode_ratios,
        assets=assets,
        pool_rng=rng_streams["common/maintenance_pool"],
        window_rng=rng_streams["common/maintenance_windows"],
    )
    spread_schedule = build_spread_baseline_schedule(tasks)

    nominal_stage_rate = calculate_nominal_stage_rate_tph(assets)
    nominal_system_rate = min(nominal_stage_rate.values())
    throughput_capacity_basis = (
        effective_network_capacity
        if scenario_name.startswith("F1_system_scale_")
        else nominal_system_rate
    )
    minimum_throughput_rate = build_minimum_throughput_profile(
        days=days,
        nominal_system_rate_tph=throughput_capacity_basis,
        peak_ratio=get_minimum_throughput_peak_ratio(scenario_name),
        rng=rng_streams["common/throughput"],
    )

    return DissertationScenario(
        metadata={
            "scenario_name": scenario_name,
            "scenario_family": "synthetic_dissertation_benchmark",
            "horizon_days": config.horizon_days,
            "seed": config.seed,
            "time_granularity": "day",
            "flow_model": "continuous_rate_based",
            "data_origin": "generated_from_first_principles",
            "nominal_stage_rate_tph": nominal_stage_rate,
            "nominal_system_rate_tph": nominal_system_rate,
            "reference_nominal_system_rate_tph": get_reference_nominal_system_rate_tph(config),
            "effective_network_capacity_tph": effective_network_capacity,
            "reference_effective_network_capacity_tph": reference_effective_network_capacity,
            "throughput_capacity_basis_tph": throughput_capacity_basis,
            "system_scale_factor": system_scale_factor,
            "system_scale_level": get_system_scale_level(scenario_name),
            "system_scale_profile": get_system_scale_profile(scenario_name),
            "operating_hours_per_day": config.operating_hours_per_day,
            "shiploader_count": get_system_scale_profile(scenario_name)["shiploader"],
            "mathematical_model_version": "MATH4003 Mathematical Model - current PDF",
            "maintenance_modes": {
                PDF_MODE_SYMBOL[mode_id]: {
                    "mode_id": mode_id,
                    "available_capacity": get_maintenance_mode_available_capacity_fraction(mode_id),
                    "duration_multiplier": mode_ratios[mode_id],
                }
                for mode_id in mode_order
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


def build_network_arcs(
    products: tuple[str, ...],
    assets: Sequence[Asset],
    config: GeneratorConfig,
) -> list[Arc]:
    """Build nested topology arcs from explicit L, M, and H layers."""
    arcs: list[Arc] = []
    arc_ids: set[str] = set()
    active_nodes = {asset.asset_id for asset in assets} | {
        "RAIL_SOURCE",
        "VESSEL_SINK",
        *ZONE_ALLOWED_PRODUCTS.keys(),
    }

    def add(
        from_node: str,
        to_node: str,
        rate_tph: float,
        allowed: Iterable[str],
    ) -> None:
        arc_id = f"{from_node}__{to_node}"
        if arc_id in arc_ids:
            return
        arc_ids.add(arc_id)
        arcs.append(
            Arc(
                arc_id=arc_id,
                from_node=from_node,
                to_node=to_node,
                capacity_tph=rate_tph,
                allowed_products=tuple(allowed),
            )
        )

    for layer in ("L", "M", "H"):
        for from_node, to_node, rate_key, allowed_token in NETWORK_ARC_LAYERS[layer]:
            if from_node not in active_nodes or to_node not in active_nodes:
                continue
            add(
                from_node,
                to_node,
                network_arc_rate_tph(rate_key, config),
                resolve_arc_allowed_products(allowed_token, products),
            )
    return arcs


def network_arc_rate_tph(rate_key: str, config: GeneratorConfig) -> float:
    rates = {
        "train_unloader": config.train_unloader_rate_tph,
        "stacker": config.stacker_rate_tph,
        "reclaimer": config.reclaimer_rate_tph,
        "shiploader": config.shiploader_rate_tph,
        "secondary": config.arc_secondary_rate_tph,
    }
    return rates[rate_key]


def resolve_arc_allowed_products(
    allowed_token: str | Sequence[str],
    products: Sequence[str],
) -> tuple[str, ...]:
    if allowed_token == ARC_ALLOWED_ALL_PRODUCTS:
        return tuple(products)
    allowed = tuple(allowed_token)
    return tuple(product for product in allowed if product in products)


def calculate_effective_network_capacity_tph(
    assets: Sequence[Asset],
    arcs: Sequence[Arc],
) -> float:
    """Return aggregate RAIL_SOURCE-to-VESSEL_SINK max flow with asset node capacities."""
    asset_by_id = {asset.asset_id: asset for asset in assets}
    residual: dict[str, dict[str, float]] = {}

    def add_edge(from_node: str, to_node: str, capacity: float) -> None:
        residual.setdefault(from_node, {})
        residual.setdefault(to_node, {})
        residual[from_node][to_node] = residual[from_node].get(to_node, 0.0) + capacity
        residual[to_node].setdefault(from_node, 0.0)

    def node_in(node: str) -> str:
        return f"{node}::in" if node in asset_by_id else node

    def node_out(node: str) -> str:
        return f"{node}::out" if node in asset_by_id else node

    for asset in assets:
        add_edge(node_in(asset.asset_id), node_out(asset.asset_id), asset.nominal_rate_tph)
    for arc in arcs:
        add_edge(node_out(arc.from_node), node_in(arc.to_node), arc.capacity_tph)

    source = "RAIL_SOURCE"
    sink = "VESSEL_SINK"
    max_flow = 0.0
    while True:
        parent: dict[str, str | None] = {source: None}
        queue = [source]
        for current in queue:
            for successor, capacity in residual.get(current, {}).items():
                if successor not in parent and capacity > 1e-9:
                    parent[successor] = current
                    if successor == sink:
                        queue = []
                        break
                    queue.append(successor)
        if sink not in parent:
            break

        path_capacity = math.inf
        current = sink
        while parent[current] is not None:
            previous = parent[current]
            path_capacity = min(path_capacity, residual[previous][current])
            current = previous

        current = sink
        while parent[current] is not None:
            previous = parent[current]
            residual[previous][current] -= path_capacity
            residual[current][previous] = residual[current].get(previous, 0.0) + path_capacity
            current = previous
        max_flow += path_capacity

    return round(max_flow, 6)


def build_special_product_requirements(
    config: GeneratorConfig,
    assets: Sequence[Asset],
    system_scale_factor: float = 1.0,
) -> list[SpecialProductRequirements]:
    """Return deterministic synthetic requirements for benchmark special-product handling."""
    tu, stacker, reclaimer, shiploader = select_x1_route_assets(assets)
    route_nodes = (
        "RAIL_SOURCE",
        tu,
        stacker,
        "Z4",
        reclaimer,
        shiploader,
        "VESSEL_SINK",
    )
    route_arcs = tuple(
        f"{from_node}__{to_node}"
        for from_node, to_node in zip(route_nodes, route_nodes[1:])
    )
    return [
        SpecialProductRequirements(
            product_id="X1",
            tonnes_per_square_metre=3.2,
            minimum_reserved_footprint_m2=90_000.0* system_scale_factor,
            segregation_buffer_m2=8_000.0,
            compatible_zones=("Z4",),
            required_assets=(tu, stacker, reclaimer, shiploader),
            compatible_arcs=route_arcs,
            minimum_campaign_tonnes=252_000.0* system_scale_factor,
            minimum_contiguous_days=9,
            earliest_campaign_start_day=25,
            latest_campaign_start_day=64,
            pre_campaign_buffer_days=2,
            post_campaign_buffer_days=2,
            cleanout_before_days=1,
            cleanout_after_days=1,
            maintenance_incompatible=True,
            blending_allowed=False,
            minimum_build_tonnes=168_000.0 * system_scale_factor,
            build_complete_before_reclaim=True,
        )
    ]


def assets_by_stage(assets: Sequence[Asset]) -> dict[str, list[str]]:
    return {
        stage: sorted(
            [asset.asset_id for asset in assets if asset.stage == stage],
            key=asset_ordinal,
        )
        for stage in STAGES
    }


def asset_ordinal(asset_id: str) -> int:
    return int("".join(ch for ch in asset_id if ch.isdigit()))


def select_x1_route_assets(assets: Sequence[Asset]) -> tuple[str, str, str, str]:
    by_stage = assets_by_stage(assets)
    route = ("TU2", "ST3", "RC2", "SL1")
    if not (
        route[0] in by_stage["train_unloader"]
        and route[1] in by_stage["stacker"]
        and route[2] in by_stage["reclaimer"]
        and route[3] in by_stage["shiploader"]
    ):
        raise ValueError("active topology does not contain the common X1 route assets")
    return route


def downstream_route_assets(
    requirement: SpecialProductRequirements,
) -> tuple[str, str]:
    reclaimer = next(asset_id for asset_id in requirement.required_assets if asset_id.startswith("RC"))
    shiploader = next(asset_id for asset_id in requirement.required_assets if asset_id.startswith("SL"))
    return reclaimer, shiploader


def build_daily_product_profile(
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


def build_initial_inventory(
    zones: Sequence[YardZone],
    products: Sequence[Product],
    fill_ratio: float | Mapping[str, float],
    *,
    rng: random.Random,
) -> dict[str, dict[str, float]]:
    product_share = {p.product_id: p.share_of_volume for p in products}
    inventory: dict[str, dict[str, float]] = {}
    for zone in zones:
        allowed = list(zone.allowed_products)
        zone_fill_ratio = (
            fill_ratio[zone.zone_id]
            if isinstance(fill_ratio, MappingABC)
            else fill_ratio
        )
        minimum_ratio = zone.minimum_inventory_tonnes / zone.capacity_tonnes
        jitter = rng.uniform(
            -INITIAL_INVENTORY_FILL_RATIO_JITTER,
            INITIAL_INVENTORY_FILL_RATIO_JITTER,
        )
        realized_fill_ratio = min(
            1.0 - 1e-6,
            max(minimum_ratio, zone_fill_ratio + jitter),
        )
        zone_total = round(zone.capacity_tonnes * realized_fill_ratio, 3)
        if not zone.minimum_inventory_tonnes <= zone_total < zone.capacity_tonnes:
            raise ValueError(
                f"initial inventory target for {zone.zone_id} violates inventory bounds"
            )
        raw_weights = {
            product: max(
                1e-9,
                product_share[product]
                * (
                    1.0
                    + rng.uniform(
                        -INITIAL_INVENTORY_PRODUCT_MIX_JITTER_FRACTION,
                        INITIAL_INVENTORY_PRODUCT_MIX_JITTER_FRACTION,
                    )
                ),
            )
            for product in allowed
        }
        total_weight = sum(raw_weights.values())
        inventory[zone.zone_id] = {
            product: round(zone_total * raw_weights[product] / total_weight, 3)
            for product in allowed
        }
    return inventory


def build_maintenance_tasks(
    horizon: int,
    *,
    scenario_name: str,
    mode_order: Sequence[str],
    mode_duration_ratios: Mapping[str, float],
    assets: Sequence[Asset],
    pool_rng: random.Random,
    window_rng: random.Random,
) -> list[MaintenanceTask]:
    candidate_pool = build_maintenance_candidate_pool(
        horizon=horizon,
        assets=assets,
        pool_rng=pool_rng,
        window_rng=window_rng,
    )
    specifications = select_f1_task_subset(candidate_pool, scenario_name)

    tasks: list[MaintenanceTask] = []
    for specification in specifications:
        modes = build_maintenance_modes(
            specification["full_duration_days"],
            mode_order=mode_order,
            duration_ratios=mode_duration_ratios,
        )
        max_duration = max(mode.duration_days for mode in modes)
        if specification["latest_start_day"] + max_duration > horizon:
            raise ValueError(
                f"maintenance window for {specification['task_id']} exceeds horizon"
            )
        if specification["lifecycle_due_day"] < (
            specification["latest_start_day"] + max_duration - 1
        ):
            raise ValueError(
                f"maintenance due day for {specification['task_id']} is too early"
            )
        tasks.append(
            MaintenanceTask(
                task_id=specification["task_id"],
                asset_id=specification["asset_id"],
                earliest_start_day=specification["earliest_start_day"],
                latest_start_day=specification["latest_start_day"],
                lifecycle_due_day=specification["lifecycle_due_day"],
                modes=modes,
                task_class=specification["task_class"],
            )
        )
    return tasks


def build_maintenance_candidate_pool(
    *,
    horizon: int,
    assets: Sequence[Asset],
    pool_rng: random.Random,
    window_rng: random.Random,
) -> list[dict[str, Any]]:
    assets_by_stage = {
        stage: sorted(
            [asset.asset_id for asset in assets if asset.stage == stage]
        )
        for stage in STAGES
    }
    stage_use_count = {
        stage: {asset_id: 0 for asset_id in asset_ids}
        for stage, asset_ids in assets_by_stage.items()
    }

    realized: list[dict[str, Any]] = []
    for template in MAINTENANCE_TEMPLATE_POOL:
        if len(assets_by_stage["shiploader"]) < template.get("min_shiploader_count", 1):
            continue
        stage = template["stage"]
        eligible_assets = eligible_maintenance_assets_for_template(
            assets_by_stage[stage],
            stage,
            template["entry_level"],
        )
        asset_id = choose_balanced_asset(
            eligible_assets,
            stage_use_count[stage],
            pool_rng,
        )
        stage_use_count[stage][asset_id] += 1
        full_duration = generate_maintenance_full_duration(
            template, horizon, pool_rng
        )
        earliest, latest, due = generate_maintenance_window(
            template=template,
            horizon=horizon,
            full_duration=full_duration,
            rng=window_rng,
        )
        realized.append(
            {
                "template_id": template["template_id"],
                "entry_level": template["entry_level"],
                "asset_id": asset_id,
                "stage": stage,
                "earliest_start_day": earliest,
                "latest_start_day": latest,
                "lifecycle_due_day": due,
                "full_duration_days": full_duration,
                "task_class": template["task_class"],
            }
        )

    occurrence_count: dict[str, int] = {}
    for specification in realized:
        asset_id = specification["asset_id"]
        occurrence_count[asset_id] = occurrence_count.get(asset_id, 0) + 1
        specification["task_id"] = f"M_{asset_id}_{occurrence_count[asset_id]:02d}"
    return realized


def eligible_maintenance_assets_for_template(
    stage_assets: Sequence[str],
    stage: str,
    entry_level: str,
) -> tuple[str, ...]:
    profile_count = SYSTEM_SCALE_PROFILES[entry_level][stage]
    return tuple(stage_assets[: min(profile_count, len(stage_assets))])


def choose_balanced_asset(
    eligible_assets: Sequence[str],
    use_count: Mapping[str, int],
    rng: random.Random,
) -> str:
    if not eligible_assets:
        raise ValueError("maintenance template has no eligible asset")
    minimum_count = min(use_count[asset_id] for asset_id in eligible_assets)
    candidates = [
        asset_id
        for asset_id in eligible_assets
        if use_count[asset_id] == minimum_count
    ]
    return rng.choice(candidates)


def generate_maintenance_full_duration(
    template: Mapping[str, Any],
    horizon: int,
    rng: random.Random,
) -> int:
    candidates = sorted(
        {
            max(2, round(duration * horizon / 90))
            for duration in template["full_duration_candidates"]
        }
    )
    return rng.choice(candidates)


def generate_maintenance_window(
    *,
    template: Mapping[str, Any],
    horizon: int,
    full_duration: int,
    rng: random.Random,
) -> tuple[int, int, int]:
    nominal_earliest = scale_maintenance_calendar_day(
        horizon, template["calendar_region"][0]
    )
    max_window_duration = max(
        1,
        math.ceil(full_duration * MAINTENANCE_WINDOW_SAFETY_DURATION_RATIO),
    )
    latest_allowed = horizon - max_window_duration
    if latest_allowed < 0:
        raise ValueError("maintenance duration cannot fit in horizon")

    earliest_low = max(0, nominal_earliest - MAINTENANCE_WINDOW_EM_JITTER_DAYS)
    earliest_high = min(
        latest_allowed,
        nominal_earliest + MAINTENANCE_WINDOW_EM_JITTER_DAYS,
    )
    if earliest_high < earliest_low:
        earliest_low = earliest_high = min(max(nominal_earliest, 0), latest_allowed)
    earliest = rng.randint(earliest_low, earliest_high)

    width_low, width_high = (
        max(1, round(width * horizon / 90))
        for width in template["window_width_range"]
    )
    window_width = rng.randint(min(width_low, width_high), max(width_low, width_high))
    latest = min(latest_allowed, max(earliest, earliest + window_width - 1))

    due_min = latest + max_window_duration - 1
    due_buffer = rng.randint(*MAINTENANCE_WINDOW_DUE_BUFFER_RANGE_DAYS)
    due = min(horizon - 1, due_min + due_buffer)
    return earliest, latest, due


def scale_maintenance_calendar_day(horizon: int, calibrated_day: int) -> int:
    return min(horizon - 1, max(0, round(calibrated_day * horizon / 90)))


def select_f1_task_subset(
    candidate_pool: Sequence[dict[str, Any]],
    scenario_name: str,
) -> list[dict[str, Any]]:
    level = (
        get_scenario_level(scenario_name)
        if scenario_name.startswith("F1_system_scale_")
        else "M"
    )
    maximum_rank = MAINTENANCE_LEVEL_ORDER[level]
    return [
        dict(specification)
        for specification in candidate_pool
        if MAINTENANCE_LEVEL_ORDER[specification["entry_level"]] <= maximum_rank
    ]


def expected_maintenance_operation_count(level: str) -> int:
    maximum_rank = MAINTENANCE_LEVEL_ORDER[level]
    return sum(
        1
        for template in MAINTENANCE_TEMPLATE_POOL
        if MAINTENANCE_LEVEL_ORDER[template["entry_level"]] <= maximum_rank
    )


def build_maintenance_modes(
    full_duration: int,
    *,
    mode_order: Sequence[str],
    duration_ratios: Mapping[str, float],
) -> tuple[MaintenanceMode, ...]:
    modes: list[MaintenanceMode] = []
    previous_duration = 0
    relative_cost = {
        FULL_SHUTDOWN_MODE: 1.00,
        PARTIAL_30_MODE: 1.15,
        PARTIAL_70_MODE: 1.35,
    }
    for mode_id in mode_order:
        ratio = duration_ratios[mode_id]
        duration = max(1, math.ceil(full_duration * ratio))
        if modes and duration <= previous_duration:
            duration = previous_duration + 1
        modes.append(
            MaintenanceMode(
                mode_id,
                duration,
                get_maintenance_mode_available_capacity_fraction(mode_id),
                relative_cost[mode_id],
            )
        )
        previous_duration = duration
    return tuple(modes)


def build_spread_baseline_schedule(
    tasks: Sequence[MaintenanceTask],
    priority_activities: Sequence[PriorityActivity] = (),
    *,
    avoid_activity_conflicts: bool = False,
) -> list[MaintenanceDecision]:
    """Create a deterministic greedy schedule inside generated EM/LM windows."""
    stage_offsets = {
        "TU": 0.10,
        "ST": 0.35,
        "RC": 0.60,
        "SL": 0.82,
    }
    schedule: list[MaintenanceDecision] = []
    occupied_by_asset: dict[str, set[int]] = {}
    for task in sorted(
        tasks,
        key=lambda item: (
            item.earliest_start_day,
            item.latest_start_day,
            item.asset_id,
            item.task_id,
        ),
    ):
        prefix = task.asset_id[:2]
        fraction = stage_offsets.get(prefix, 0.50)
        span = task.latest_start_day - task.earliest_start_day
        preferred_start = task.earliest_start_day + round(span * fraction)
        mode = next(mode for mode in task.modes if mode.mode_id == FULL_SHUTDOWN_MODE)
        feasible_starts = range(task.earliest_start_day, task.latest_start_day + 1)
        start = min(
            feasible_starts,
            key=lambda candidate: (
                activity_conflict_days(
                    task,
                    candidate,
                    mode.duration_days,
                    priority_activities,
                )
                if avoid_activity_conflicts
                else 0,
                asset_overlap_days(
                    task.asset_id,
                    candidate,
                    mode.duration_days,
                    occupied_by_asset,
                ),
                abs(candidate - preferred_start),
                candidate,
            ),
        )
        occupied_by_asset.setdefault(task.asset_id, set()).update(
            range(start, start + mode.duration_days)
        )
        schedule.append(MaintenanceDecision(task.task_id, start, FULL_SHUTDOWN_MODE))
    return schedule


def asset_overlap_days(
    asset_id: str,
    start_day: int,
    duration_days: int,
    occupied_by_asset: Mapping[str, set[int]],
) -> int:
    occupied = occupied_by_asset.get(asset_id, set())
    return sum(1 for day in range(start_day, start_day + duration_days) if day in occupied)


def activity_conflict_days(
    task: MaintenanceTask,
    start_day: int,
    duration_days: int,
    priority_activities: Sequence[PriorityActivity],
) -> int:
    conflict_days = 0
    for activity in priority_activities:
        if not activity.maintenance_incompatible:
            continue
        if task.asset_id not in activity.required_assets:
            continue
        protected_start = activity.start_day - activity.pre_campaign_buffer_days
        protected_end = activity.end_day + activity.post_campaign_buffer_days
        for day in range(start_day, start_day + duration_days):
            if protected_start <= day < protected_end:
                conflict_days += 1
    return conflict_days


# ============================================================
# 5. FAMILY-SPECIFIC DATA
# ============================================================


def apply_scenario_family(
    scenario: DissertationScenario,
    scenario_name: str,
    config: GeneratorConfig,
    rng_streams: Mapping[str, random.Random],
) -> DissertationScenario:
    handlers = {
        BASE_CONTROL_NAME: apply_base_control_metadata,
        "F1_system_scale_L": apply_system_scale_family,
        "F1_system_scale_M": apply_system_scale_family,
        "F1_system_scale_H": apply_system_scale_family,
        "F2_mode_tradeoff_L": apply_mode_tradeoff_family,
        "F2_mode_tradeoff_M": apply_mode_tradeoff_family,
        "F2_mode_tradeoff_H": apply_mode_tradeoff_family,
        "F3_yard_flow_L": apply_yard_flow_family,
        "F3_yard_flow_M": apply_yard_flow_family,
        "F3_yard_flow_H": apply_yard_flow_family,
        "F4_operational_conflict_L": apply_operational_conflict_family,
        "F4_operational_conflict_M": apply_operational_conflict_family,
        "F4_operational_conflict_H": apply_operational_conflict_family,
        "structurally_infeasible": apply_structural_infeasibility_validation,
    }
    return handlers[scenario_name](scenario, config, rng_streams)


def apply_base_control_metadata(
    scenario: DissertationScenario,
    config: GeneratorConfig,
    rng_streams: Mapping[str, random.Random],
) -> DissertationScenario:
    scenario.metadata.update(
        scenario_family="base_control",
        occasion_description=(
            "Calibrated normal case with heterogeneous yard capacities, low initial "
            "inventory, time-varying minimum throughput, and distributed maintenance "
            "windows."
        ),
        expected_model_value=(
            "Validate that the common data instance is feasible before testing the "
            "four experimental scenario families."
        ),
    )
    return scenario


def apply_system_scale_family(
    scenario: DissertationScenario,
    config: GeneratorConfig,
    rng_streams: Mapping[str, random.Random],
) -> DissertationScenario:
    level = get_scenario_level(scenario.metadata["scenario_name"])
    task_count = len(scenario.maintenance_tasks)
    mode_count = len(get_scenario_mode_order(scenario.maintenance_tasks))
    profile = scenario.metadata["system_scale_profile"]
    scenario.metadata.update(
        scenario_family="F1_system_scale",
        scenario_level=level,
        occasion_description=(
            f"System-scale {level} case with "
            f"{profile['train_unloader']} train unloaders, "
            f"{profile['stacker']} stackers, {profile['reclaimer']} reclaimers, "
            f"{profile['shiploader']} shiploaders, {task_count} maintenance "
            f"operations, and the common {mode_count}-mode F/P30/P70 set."
        ),
        expected_model_value=(
            "Measure how model size and scheduling complexity change as physical "
            "network size and maintenance-operation count grow at comparable "
            "relative utilisation."
        ),
    )
    return scenario


def apply_mode_tradeoff_family(
    scenario: DissertationScenario,
    config: GeneratorConfig,
    rng_streams: Mapping[str, random.Random],
) -> DissertationScenario:
    level = get_scenario_level(scenario.metadata["scenario_name"])
    first_task = scenario.maintenance_tasks[0]
    ratios = {
        mode.mode_id: round(mode.duration_days / first_task.modes[0].duration_days, 3)
        for mode in first_task.modes
    }
    scenario.metadata.update(
        scenario_family="F2_mode_tradeoff",
        scenario_level=level,
        occasion_description=(
            f"Maintenance-mode trade-off {level} case with fixed topology and "
            f"duration ratios {ratios}."
        ),
        expected_model_value=(
            "Test whether short severe shutdowns and longer partial-capacity work "
            "become attractive under different calendar and capacity conditions."
        ),
    )
    return scenario


def apply_yard_flow_family(
    scenario: DissertationScenario,
    config: GeneratorConfig,
    rng_streams: Mapping[str, random.Random],
) -> DissertationScenario:
    level = get_scenario_level(scenario.metadata["scenario_name"])
    initial_total = sum(sum(by_product.values()) for by_product in scenario.initial_inventory.values())
    capacity_total = sum(zone.capacity_tonnes for zone in scenario.zones)
    min_tpd = min(scenario.minimum_throughput_rate_tph.values()) * config.operating_hours_per_day
    max_tpd = max(scenario.minimum_throughput_rate_tph.values()) * config.operating_hours_per_day
    scenario.metadata.update(
        scenario_family="F3_yard_flow",
        scenario_level=level,
        occasion_description=(
            f"Yard-and-flow tightness {level} case with {initial_total / capacity_total:.1%} "
            f"initial utilisation and daily minimum throughput between "
            f"{min_tpd:,.0f} and {max_tpd:,.0f} tonnes."
        ),
        expected_model_value=(
            "Show the interaction between heterogeneous yard capacities, non-zero "
            "minimum inventories, low initial stock, and throughput pressure."
        ),
    )
    return scenario


def build_operational_activity_windows(
    level: str,
    requirement: SpecialProductRequirements,
    rng: random.Random,
    horizon: int,
) -> list[PriorityActivity]:
    settings = F4_LEVEL_SETTINGS[level]
    activities: list[PriorityActivity] = []
    for activity_spec in F4_ACTIVITY_WINDOWS:
        start_low, start_high = (
            scale_maintenance_calendar_day(horizon, day)
            for day in activity_spec["start_day_range"]
        )
        duration_low, duration_high = (
            max(1, round(days * horizon / 90))
            for days in activity_spec["duration_range"]
        )
        start_day = rng.randint(min(start_low, start_high), max(start_low, start_high))
        duration_days = rng.randint(
            min(duration_low, duration_high), max(duration_low, duration_high)
        )
        start_day = min(start_day, horizon - duration_days)
        end_day = start_day + duration_days

        include_key = activity_spec.get("include_from_level")
        if include_key and not settings[include_key]:
            continue

        if activity_spec["asset_group"] == "X1_ROUTE":
            required_assets = requirement.required_assets
        else:
            required_assets = downstream_route_assets(requirement)

        if "capacity_factor_from_level" in activity_spec:
            capacity_factor = settings[activity_spec["capacity_factor_from_level"]]
        else:
            capacity_factor = activity_spec["capacity_factor"]

        if "maintenance_incompatible_from_level" in activity_spec:
            maintenance_incompatible = settings[
                activity_spec["maintenance_incompatible_from_level"]
            ]
        else:
            maintenance_incompatible = activity_spec["maintenance_incompatible"]

        buffer_days = 0
        if activity_spec.get("buffer_from_level"):
            buffer_days = settings["buffer_days"]
        elif (
            activity_spec.get("buffer_from_level_when_incompatible")
            and maintenance_incompatible
        ):
            buffer_days = settings["buffer_days"]

        activities.append(
            PriorityActivity(
                activity_id=activity_spec["activity_id"],
                product_id="X1",
                start_day=start_day,
                end_day=end_day,
                minimum_tonnes_per_day=0.0,
                required_assets=required_assets,
                hard_requirement=True,
                earliest_start_day=start_day,
                latest_start_day=start_day,
                required_zone="Z4",
                minimum_total_tonnes=0.0,
                minimum_contiguous_days=1,
                pre_campaign_buffer_days=buffer_days,
                post_campaign_buffer_days=buffer_days,
                cleanout_before_days=0,
                cleanout_after_days=0,
                capacity_factor_during_activity=capacity_factor,
                maintenance_incompatible=maintenance_incompatible,
            )
        )
    return activities


def apply_operational_conflict_family(
    scenario: DissertationScenario,
    config: GeneratorConfig,
    rng_streams: Mapping[str, random.Random],
) -> DissertationScenario:
    level = get_scenario_level(scenario.metadata["scenario_name"])
    requirement = scenario.special_product_requirements[0]
    f4_rng = rng_streams["F4/activity_windows"]

    scenario.priority_activities = build_operational_activity_windows(
        level,
        requirement,
        f4_rng,
        config.horizon_days,
    )

    scenario.maintenance_tasks = retime_f4_maintenance_tasks(
        tasks=scenario.maintenance_tasks,
        activities=scenario.priority_activities,
        assets=scenario.assets,
        requirement=requirement,
        rng=f4_rng,
        horizon=config.horizon_days,
    )
    scenario.manual_baseline = build_spread_baseline_schedule(
        scenario.maintenance_tasks,
        scenario.priority_activities,
        avoid_activity_conflicts=False,
    )
    scenario.reference_schedule = build_spread_baseline_schedule(
        scenario.maintenance_tasks,
        scenario.priority_activities,
        avoid_activity_conflicts=True,
    )

    scenario.metadata.update(
        scenario_family="F4_operational_conflict",
        scenario_level=level,
        occasion_description=(
            f"Operational-conflict {level} case with {len(scenario.priority_activities)} protected "
            "activity windows around downstream export and the X1 route."
        ),
        expected_model_value=(
            "Use activity capacity reductions and, at the medium and high levels, "
            "maintenance incompatibility to create calendar regions where maintenance "
            "timing competes with protected operational work."
        ),
        stress_periods=[
            [activity.start_day, activity.end_day]
            for activity in scenario.priority_activities
        ],
    )
    return scenario


def retime_f4_maintenance_tasks(
    *,
    tasks: Sequence[MaintenanceTask],
    activities: Sequence[PriorityActivity],
    assets: Sequence[Asset],
    requirement: SpecialProductRequirements,
    rng: random.Random,
    horizon: int,
) -> list[MaintenanceTask]:
    stage_by_asset = {asset.asset_id: asset.stage for asset in assets}
    replacements: dict[str, MaintenanceTask] = {}
    selected_task_ids: set[str] = set()
    for activity in activities:
        group, target_count = F4_MAINTENANCE_TARGETS_BY_ACTIVITY.get(
            activity.activity_id,
            ("downstream_export", 1),
        )
        candidates = f4_candidate_tasks(
            tasks=tasks,
            activity=activity,
            stage_by_asset=stage_by_asset,
            requirement=requirement,
            group=group,
            excluded_task_ids=selected_task_ids,
        )
        chosen_tasks = choose_f4_target_tasks(candidates, target_count, rng)
        for task in chosen_tasks:
            selected_task_ids.add(task.task_id)
            replacements[task.task_id] = retime_task_around_activity(
                task=task,
                activity=activity,
                all_activities=activities,
                rng=rng,
                horizon=horizon,
            )
    return [replacements.get(task.task_id, task) for task in tasks]


def f4_candidate_tasks(
    *,
    tasks: Sequence[MaintenanceTask],
    activity: PriorityActivity,
    stage_by_asset: Mapping[str, str],
    requirement: SpecialProductRequirements,
    group: str,
    excluded_task_ids: set[str],
) -> list[MaintenanceTask]:
    unused = [task for task in tasks if task.task_id not in excluded_task_ids]
    if group == "x1_route_upstream":
        candidates = [
            task
            for task in unused
            if task.asset_id in requirement.required_assets
            and stage_by_asset[task.asset_id] in {"train_unloader", "stacker"}
        ]
        if candidates:
            return sorted(candidates, key=lambda task: task.task_id)
        return sorted(
            [
                task
                for task in unused
                if task.asset_id in requirement.required_assets
            ],
            key=lambda task: task.task_id,
        )

    direct_activity_assets = [
        task for task in unused if task.asset_id in activity.required_assets
    ]
    if direct_activity_assets:
        return sorted(direct_activity_assets, key=lambda task: task.task_id)
    return sorted(
        [
            task
            for task in unused
            if stage_by_asset[task.asset_id] in {"reclaimer", "shiploader"}
        ],
        key=lambda task: task.task_id,
    )


def choose_f4_target_tasks(
    candidates: Sequence[MaintenanceTask],
    target_count: int,
    rng: random.Random,
) -> list[MaintenanceTask]:
    if not candidates or target_count <= 0:
        return []
    count = min(target_count, len(candidates))
    return sorted(rng.sample(list(candidates), count), key=lambda task: task.task_id)


def retime_task_around_activity(
    *,
    task: MaintenanceTask,
    activity: PriorityActivity,
    all_activities: Sequence[PriorityActivity],
    rng: random.Random,
    horizon: int,
) -> MaintenanceTask:
    max_duration = max(mode.duration_days for mode in task.modes)
    latest_allowed = horizon - max_duration
    anchor_start = activity.start_day
    anchor_end = activity.end_day
    pre_span = rng.randint(*F4_MAINTENANCE_PRE_ACTIVITY_RANGE_DAYS)
    post_span = rng.randint(*F4_MAINTENANCE_POST_ACTIVITY_RANGE_DAYS)
    earliest = max(0, min(latest_allowed, anchor_start - pre_span))
    latest = min(
        latest_allowed,
        max(earliest, anchor_end + post_span - max_duration),
    )
    due = min(horizon - 1, latest + max_duration - 1)
    return _retime_task(task, earliest, latest, due)


def apply_structural_infeasibility_validation(
    scenario: DissertationScenario,
    config: GeneratorConfig,
    rng_streams: Mapping[str, random.Random],
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
        scenario.minimum_throughput_rate_tph[day] = STRUCTURAL_INFEASIBLE_MIN_THROUGHPUT_RATE_TPH
    scenario.metadata.update(
        occasion_description=(
            f"Daily minimum throughput is set above the physical system capacity "
            f"on days {start}-{end - 1}."
        ),
        expected_model_value=(
            "Correctly prove infeasibility instead of returning an apparently valid "
            "but physically impossible schedule. The infeasibility is created through "
            "the active minimum-throughput parameter."
        ),
        expected_feasibility="infeasible",
        stress_period=[start, end],
    )
    return scenario


# ============================================================
# 7. VALIDATION AND DESIGN EVIDENCE
# ============================================================


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
    supply = build_dense_product_quantity_map(
        scenario.inbound_supply, products, scenario.days
    )
    demand = build_dense_product_quantity_map(
        scenario.outbound_demand, products, scenario.days
    )
    capacity_factor = _effective_capacity_factors(scenario, schedule)
    _, maintenance_activity_incompatible = build_operational_activity_profiles(scenario)
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


def build_network_adjacency_maps(
    arcs: Sequence[Arc], nodes: Sequence[str]
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Return predecessor and successor node sets used by flow conservation."""
    predecessors = {node: [] for node in nodes}
    successors = {node: [] for node in nodes}
    for arc in arcs:
        predecessors[arc.to_node].append(arc.from_node)
        successors[arc.from_node].append(arc.to_node)
    return predecessors, successors


def build_operational_activity_profiles(
    scenario: DissertationScenario,
) -> tuple[dict[tuple[str, int], float], dict[tuple[str, int], int]]:
    """Build sigma[a,d] and h[a,d] from realized high-priority activities."""
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
    sigma, _ = build_operational_activity_profiles(scenario)
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


# ============================================================
# SHARED HELPERS
# ============================================================


def make_rng_streams(master_seed: int) -> dict[str, random.Random]:
    return {
        stream_name: make_rng(master_seed, *stream_name.split("/"))
        for stream_name in RNG_STREAM_NAMES
    }


def make_rng(master_seed: int, *stream_parts: str) -> random.Random:
    return random.Random(_stable_seed(master_seed, *stream_parts))


def _stable_seed(seed: int, *stream_parts: str) -> int:
    stream_name = "/".join(stream_parts)
    digest = hashlib.sha256(
        f"{GENERATOR_RANDOMIZATION_VERSION}:{seed}:{stream_name}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _day(horizon: int, fraction: float) -> int:
    return min(horizon - 1, max(0, round(fraction * (horizon - 1))))


def _period(
    horizon: int, start_fraction: float, end_fraction: float
) -> tuple[int, int]:
    start = _day(horizon, start_fraction)
    end = min(horizon, max(start + 1, _day(horizon, end_fraction)))
    return start, end


def calculate_nominal_stage_rate_tph(assets: Sequence[Asset]) -> dict[str, float]:
    """Calculate the total nominal rate (t/hr) per processing stage."""
    return {
        stage: sum(a.nominal_rate_tph for a in assets if a.stage == stage)
        for stage in STAGES
    }


def build_dense_product_quantity_map(
    records: Sequence[DailyProductQuantity],
    products: Sequence[str],
    days: Sequence[int],
) -> dict[tuple[str, int], float]:
    dense = {(p, day): 0.0 for p in products for day in days}
    for record in records:
        dense[(record.product_id, record.day)] += record.tonnes
    return dense

def build_aggregate_daily_quantity_map(
    records: Sequence[DailyProductQuantity],
    days: Sequence[int],
) -> dict[int, float]:
    """Aggregate product quantities into one commodity by day."""
    aggregate = {day: 0.0 for day in days}
    for record in records:
        aggregate[record.day] += record.tonnes
    return {
        day: round(tonnes, 3)
        for day, tonnes in aggregate.items()
    }


def build_aggregate_initial_inventory_map(
    initial_inventory: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    """Aggregate product inventories into one commodity by stockyard zone."""
    return {
        zone_id: round(sum(by_product.values()), 3)
        for zone_id, by_product in initial_inventory.items()
    }

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


def update_realization_metadata(
    scenario: DissertationScenario,
    config: GeneratorConfig,
) -> None:
    asset_stage = {asset.asset_id: asset.stage for asset in scenario.assets}
    maintenance_by_stage = {
        stage: sum(
            1
            for task in scenario.maintenance_tasks
            if asset_stage[task.asset_id] == stage
        )
        for stage in STAGES
    }
    initial_inventory_ratios = {
        zone.zone_id: round(
            sum(scenario.initial_inventory[zone.zone_id].values())
            / zone.capacity_tonnes,
            6,
        )
        for zone in scenario.zones
    }
    scenario.metadata["generation"] = {
        "master_seed": config.seed,
        "randomization_version": GENERATOR_RANDOMIZATION_VERSION,
        "rng_stream_version": RNG_STREAM_VERSION,
        "rng_stream_names": list(RNG_STREAM_NAMES),
        "flow": (
            "structural/config parameters -> family treatment -> "
            "seeded realization -> derived parameters -> validation"
        ),
    }
    scenario.metadata["realization"] = {
        "maintenance_operation_count": len(scenario.maintenance_tasks),
        "maintenance_by_stage_counts": maintenance_by_stage,
        "initial_inventory_fill_ratios": initial_inventory_ratios,
        "maintenance_tasks": {
            task.task_id: {
                "asset_id": task.asset_id,
                "stage": asset_stage[task.asset_id],
                "full_duration_days": next(
                    mode.duration_days
                    for mode in task.modes
                    if mode.mode_id == FULL_SHUTDOWN_MODE
                ),
                "earliest_start_day": task.earliest_start_day,
                "latest_start_day": task.latest_start_day,
                "lifecycle_due_day": task.lifecycle_due_day,
                "task_class": task.task_class,
            }
            for task in scenario.maintenance_tasks
        },
    }


def validate_scenario_design(
    scenario_name: str,
    scenario: DissertationScenario,
    baseline: Mapping[str, float | int],
    reference: Mapping[str, float | int],
) -> dict[str, Any]:
    product_ids = {product.product_id for product in scenario.products}
    zone_by_id = {zone.zone_id: zone for zone in scenario.zones}
    asset_ids = {asset.asset_id for asset in scenario.assets}
    arc_ids = {arc.arc_id for arc in scenario.arcs}
    task_ids = [task.task_id for task in scenario.maintenance_tasks]
    x1_requirement = next(
        req for req in scenario.special_product_requirements if req.product_id == "X1"
    )
    serialized_model_data = json.dumps(_json_safe(scenario.optimizer_objects()))
    checks: dict[str, bool] = {
        "initial_inventory_respects_zone_bounds": all(
            zone.minimum_inventory_tonnes
            <= sum(scenario.initial_inventory[zone.zone_id].values())
            < zone.capacity_tonnes
            for zone in scenario.zones
        ),
        "initial_inventory_uses_known_products": all(
            set(by_product).issubset(product_ids)
            for by_product in scenario.initial_inventory.values()
        ),
        "initial_inventory_respects_product_zone_compatibility": all(
            set(by_product).issubset(set(zone_by_id[zone_id].allowed_products))
            for zone_id, by_product in scenario.initial_inventory.items()
        ),
        "maintenance_assets_exist": all(
            task.asset_id in asset_ids for task in scenario.maintenance_tasks
        ),
        "maintenance_task_ids_are_unique": len(task_ids) == len(set(task_ids)),
        "maintenance_windows_valid_for_longest_mode": all(
            0
            <= task.earliest_start_day
            <= task.latest_start_day
            <= len(scenario.days) - max(mode.duration_days for mode in task.modes)
            and task.lifecycle_due_day
            >= task.latest_start_day + max(mode.duration_days for mode in task.modes) - 1
            for task in scenario.maintenance_tasks
        ),
        "no_inactive_asset_references": all(
            asset_id in asset_ids
            for task in scenario.maintenance_tasks
            for asset_id in (task.asset_id,)
        )
        and all(
            asset_id in asset_ids
            for activity in scenario.priority_activities
            for asset_id in activity.required_assets
        ),
        "generation_metadata_present": (
            scenario.metadata.get("generation", {}).get("master_seed")
            == scenario.metadata.get("seed")
            and "realization" in scenario.metadata
        ),
        "minimum_throughput_profile_varies": (
            max(scenario.minimum_throughput_rate_tph.values())
            > 1.2 * min(scenario.minimum_throughput_rate_tph.values())
        ),
        "mode_duration_increases_with_available_capacity": all(
            all(
                left.duration_days < right.duration_days
                for left, right in zip(task.modes, task.modes[1:])
            )
            for task in scenario.maintenance_tasks
        ),
        "three_active_maintenance_modes": (
            get_scenario_mode_order(scenario.maintenance_tasks)
            == MAINTENANCE_MODE_ORDER
        ),
        "no_p50_in_active_model_data": "partial_50" not in serialized_model_data,
        "effective_network_capacity_matches_topology": math.isclose(
            scenario.metadata["effective_network_capacity_tph"],
            calculate_effective_network_capacity_tph(scenario.assets, scenario.arcs),
            rel_tol=0.0,
            abs_tol=1e-6,
        ),
        "common_x1_route_assets": (
            x1_requirement.required_assets == ("TU2", "ST3", "RC2", "SL1")
        ),
        "common_x1_route_arcs_exist": set(x1_requirement.compatible_arcs).issubset(
            arc_ids
        ),
    }
    if scenario_name == BASE_CONTROL_NAME:
        checks["baseline_and_reference_are_identical"] = (
            scenario.manual_baseline == scenario.reference_schedule
        )
    elif scenario_name.startswith("F1_system_scale"):
        level = get_scenario_level(scenario_name)
        task_count = len(scenario.maintenance_tasks)
        mode_count = len(get_scenario_mode_order(scenario.maintenance_tasks))
        profile = SYSTEM_SCALE_PROFILES[level]
        stage_counts = {
            stage: sum(1 for asset in scenario.assets if asset.stage == stage)
            for stage in STAGES
        }
        expected_count = expected_maintenance_operation_count(level)
        checks["task_count_matches_scale_level"] = task_count == expected_count
        checks["asset_count_matches_scale_level"] = stage_counts == profile
        checks["mode_count_matches_scale_level"] = mode_count == 3
    elif scenario_name.startswith("F2_mode_tradeoff"):
        mode_order = get_scenario_mode_order(scenario.maintenance_tasks)
        checks["three_tradeoff_modes_present"] = mode_order == MAINTENANCE_MODE_ORDER
        checks["uses_reference_m_topology"] = (
            scenario.metadata.get("system_scale_profile") == SYSTEM_SCALE_PROFILES["M"]
        )
    elif scenario_name.startswith("F3_yard_flow"):
        min_inventory_ratios = [
            zone.minimum_inventory_tonnes / zone.capacity_tonnes
            for zone in scenario.zones
        ]
        checks["nonzero_minimum_inventory"] = min(min_inventory_ratios) >= 0.05
        checks["uses_reference_m_topology"] = (
            scenario.metadata.get("system_scale_profile") == SYSTEM_SCALE_PROFILES["M"]
        )
    elif scenario_name.startswith("F4_operational_conflict"):
        total_incompatible_days = sum(
            build_operational_activity_profiles(scenario)[1].values()
        )
        reduced_capacity_days = sum(
            1
            for value in build_operational_activity_profiles(scenario)[0].values()
            if value < 1.0
        )

        checks["multiple_activity_windows"] = (
            len(scenario.priority_activities) >= 2
        )

        checks["uses_reference_m_topology"] = (
            scenario.metadata.get("system_scale_profile")
            == SYSTEM_SCALE_PROFILES["M"]
        )

        expected_activity_count = (
            2 if get_scenario_level(scenario_name) == "L" else 3
        )

        checks["activity_count_matches_level"] = (
            len(scenario.priority_activities) == expected_activity_count
        )

        level = get_scenario_level(scenario_name)

        checks["has_capacity_reduction"] = (
            reduced_capacity_days > 0
        )

        if level == "L":
            checks["no_hard_maintenance_incompatibility_at_L"] = (
                total_incompatible_days == 0
            )

            checks["retimed_tasks_touch_activity_windows"] = any(
                task.asset_id in activity.required_assets
                and task.earliest_start_day < activity.end_day
                and task.latest_start_day >= activity.start_day
                for task in scenario.maintenance_tasks
                for activity in scenario.priority_activities
            )

        else:
            checks["has_hard_maintenance_incompatibility"] = (
                total_incompatible_days > 0
            )

            checks[
                "retimed_tasks_touch_incompatible_activity_assets"
            ] = any(
                task.asset_id in activity.required_assets
                and task.earliest_start_day < activity.end_day
                and task.latest_start_day >= activity.start_day
                for task in scenario.maintenance_tasks
                for activity in scenario.priority_activities
                if activity.maintenance_incompatible
            )

    elif scenario_name == "structurally_infeasible":
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


# ============================================================
# 8. OUTPUT / CLI
# ============================================================


def write_scenario(
    scenario: DissertationScenario,
    output_path: str | Path,
    *,
    include_reference_schedule: bool = False,
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
        newline="\n",
    )
    return path


def generate_suite(
    output_dir: str | Path,
    config: GeneratorConfig | None = None,
    scenario_names: Sequence[str] = SCENARIO_NAMES,
) -> dict[str, Path]:
    config = config or GeneratorConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    generated: dict[str, Path] = {}
    manifest_cases: list[dict[str, Any]] = []
    for scenario_name in scenario_names:
        scenario = build_scenario(scenario_name, config)
        scenario_modes = get_scenario_mode_order(scenario.maintenance_tasks)
        special_products = [product.product_id for product in scenario.products if product.is_special]
        special_requirement = next(
            req for req in scenario.special_product_requirements if req.product_id == "X1"
        )
        path = write_scenario(
            scenario,
            output / f"{scenario_name}.json",
            include_reference_schedule=False,
        )
        generated[scenario_name] = path
        manifest_cases.append(
            {
                "scenario_name": scenario_name,
                "file": path.name,
                "description": scenario.metadata["occasion_description"],
                "expected_model_value": scenario.metadata["expected_model_value"],
                "expected_feasibility": scenario.metadata["expected_feasibility"],
                "asset_count": len(scenario.assets),
                "system_scale_level": scenario.metadata["system_scale_level"],
                "system_scale_profile": scenario.metadata["system_scale_profile"],
                "system_scale_factor": scenario.metadata["system_scale_factor"],
                "effective_network_capacity_tph": scenario.metadata[
                    "effective_network_capacity_tph"
                ],
                "reference_effective_network_capacity_tph": scenario.metadata[
                    "reference_effective_network_capacity_tph"
                ],
                "throughput_capacity_basis_tph": scenario.metadata[
                    "throughput_capacity_basis_tph"
                ],
                "maintenance_operation_count": len(scenario.maintenance_tasks),
                "maintenance_modes": list(scenario_modes),
                "generation": scenario.metadata["generation"],
                "realization": {
                    "maintenance_operation_count": scenario.metadata["realization"][
                        "maintenance_operation_count"
                    ],
                    "maintenance_by_stage_counts": scenario.metadata["realization"][
                        "maintenance_by_stage_counts"
                    ],
                    "initial_inventory_fill_ratios": scenario.metadata[
                        "realization"
                    ]["initial_inventory_fill_ratios"],
                },
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
        "randomization_version": GENERATOR_RANDOMIZATION_VERSION,
        "rng_stream_version": RNG_STREAM_VERSION,
        "rng_stream_names": list(RNG_STREAM_NAMES),
        "system_scale_profiles": SYSTEM_SCALE_PROFILES,
        "reference_system_scale_level": "M",
        "default_maintenance_modes": list(MAINTENANCE_MODE_ORDER),
        "supported_maintenance_modes": list(MAINTENANCE_MODE_ORDER),
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
        newline="\n",
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
        choices=("all", *SCENARIO_NAMES),
        help="Named scenario to generate, or all benchmark scenarios.",
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
        shiploader_rate_tph=args.shiploader_rate_tph,
    )
    scenario_names = SCENARIO_NAMES if args.occasion == "all" else (args.occasion,)
    generated = generate_suite(args.output_dir, config, scenario_names)
    for name, path in generated.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

