"""Canonical JSON loader for the frozen port-yard MILP schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


CANONICAL_MODEL_READY_KEYS = (
    "D",
    "N",
    "E",
    "A",
    "A_NS",
    "P",
    "G",
    "M",
    "M_a",
    "K",
    "arc_from",
    "arc_to",
    "predecessors",
    "successors",
    "asset_capacity_tpd",
    "arc_capacity_tpd",
    "maintenance_duration_days",
    "maintenance_available_capacity_fraction",
    "maintenance_earliest_start",
    "maintenance_latest_start",
    "zone_min_inventory_tonnes",
    "zone_max_inventory_tonnes",
    "initial_inventory_tonnes",
    "overflow_penalty_per_tonne",
    "minimum_throughput_tpd",
    "special_activity_capacity_factor",
    "maintenance_activity_incompatible",
)

CANONICAL_MODEL_READY_KEY_SET = set(CANONICAL_MODEL_READY_KEYS)

LEGACY_OPTIMISATION_FIELDS = {
    "Z",
    "M_OPERATIONS",
    "supply_tonnes",
    "demand_tonnes",
    "maintenance_relative_cost",
    "SPECIAL_PRODUCTS",
    "product_is_special",
    "product_grade_specs",
    "product_blending_allowed",
    "product_volume_share",
    "product_arc_compatible",
    "arc_allowed_product",
    "zone_allowed_product",
    "product_zone_compatible",
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
}


def load_canonical_scenario(
    path: str | Path,
    *,
    require_single_commodity: bool = False,
) -> dict[str, Any]:
    """Load and validate one frozen-schema scenario JSON file."""
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if "model_ready_data" not in raw:
        raise ValueError("scenario JSON is missing model_ready_data")

    model_ready_data = raw["model_ready_data"]
    validate_canonical_model_ready_data(model_ready_data, require_single_commodity=require_single_commodity)

    D = [int(day) for day in model_ready_data["D"]]
    N = list(model_ready_data["N"])
    A = list(model_ready_data["A"])
    A_NS = list(model_ready_data["A_NS"])
    P = list(model_ready_data["P"])
    G = list(model_ready_data["G"])
    M = list(model_ready_data["M"])
    M_a = {asset: list(tasks) for asset, tasks in model_ready_data["M_a"].items()}
    K = list(model_ready_data["K"])

    arc_from = dict(model_ready_data["arc_from"])
    arc_to = dict(model_ready_data["arc_to"])
    arc_ids = list(model_ready_data["E"])
    E = [(arc_from[arc_id], arc_to[arc_id]) for arc_id in arc_ids]
    U = {(arc_from[arc_id], arc_to[arc_id]): float(model_ready_data["arc_capacity_tpd"][arc_id]) for arc_id in arc_ids}

    p = _parse_pair_float_mapping(model_ready_data["maintenance_duration_days"], int)
    eta = _parse_pair_float_mapping(model_ready_data["maintenance_available_capacity_fraction"], float)
    sigma = _parse_asset_day_mapping(model_ready_data["special_activity_capacity_factor"], float)
    h = _parse_asset_day_mapping(model_ready_data["maintenance_activity_incompatible"], int)
    z0_pg = _parse_pair_float_mapping(model_ready_data["initial_inventory_tonnes"], float)
    mu_min = {int(day): float(value) for day, value in model_ready_data["minimum_throughput_tpd"].items()}

    return {
        "raw": raw,
        "model_ready_data": model_ready_data,
        "D": D,
        "N": N,
        "A": A,
        "A_NS": A_NS,
        "P_": P,
        "G": G,
        "M": M,
        "M_a": M_a,
        "K": K,
        "E": E,
        "arc_ids": arc_ids,
        "arc_from": arc_from,
        "arc_to": arc_to,
        "U": U,
        "predecessors": _normalise_adjacency(model_ready_data["predecessors"], N),
        "successors": _normalise_adjacency(model_ready_data["successors"], N),
        "C": {asset: float(value) for asset, value in model_ready_data["asset_capacity_tpd"].items()},
        "p": p,
        "eta": eta,
        "e_m": {task: int(day) for task, day in model_ready_data["maintenance_earliest_start"].items()},
        "l_m": {task: int(day) for task, day in model_ready_data["maintenance_latest_start"].items()},
        "sigma": sigma,
        "h": h,
        "Z_min": {zone: float(value) for zone, value in model_ready_data["zone_min_inventory_tonnes"].items()},
        "Z_max": {zone: float(value) for zone, value in model_ready_data["zone_max_inventory_tonnes"].items()},
        "alpha": {zone: float(value) for zone, value in model_ready_data["overflow_penalty_per_tonne"].items()},
        "z0_pg": z0_pg,
        "mu_min": mu_min,
        "is_multicommodity": not require_single_commodity,
    }


def validate_canonical_model_ready_data(
    data: Mapping[str, Any],
    *,
    require_single_commodity: bool = False,
) -> None:
    keys = set(data)
    if keys != CANONICAL_MODEL_READY_KEY_SET:
        missing = sorted(CANONICAL_MODEL_READY_KEY_SET - keys)
        extra = sorted(keys - CANONICAL_MODEL_READY_KEY_SET)
        legacy = sorted(LEGACY_OPTIMISATION_FIELDS & keys)
        raise ValueError(
            f"model_ready_data must match the canonical schema; missing={missing}, extra={extra}, legacy={legacy}"
        )

    D = [int(day) for day in data["D"]]
    if not D or D != list(range(D[0], D[0] + len(D))):
        raise ValueError("D must be a non-empty consecutive list of integer days")
    day_set = set(D)

    N = set(data["N"])
    A = set(data["A"])
    A_NS = set(data["A_NS"])
    P = set(data["P"])
    G = set(data["G"])
    M = set(data["M"])
    K = set(data["K"])

    if not G:
        raise ValueError("G must be non-empty")
    if require_single_commodity and len(G) != 1:
        raise ValueError(f"single-commodity MILP requires exactly one commodity, received {len(G)}")
    if not A.issubset(N) or not A_NS.issubset(A) or not P.issubset(N):
        raise ValueError("A, A_NS, and P must reference valid nodes")
    if A & P:
        raise ValueError("asset and stockyard sets must be disjoint")

    arc_ids = set(data["E"])
    if set(data["arc_from"]) != arc_ids or set(data["arc_to"]) != arc_ids:
        raise ValueError("arc_from and arc_to keys must match E")
    if set(data["arc_capacity_tpd"]) != arc_ids:
        raise ValueError("arc_capacity_tpd keys must match E")
    for arc_id in arc_ids:
        if data["arc_from"][arc_id] not in N or data["arc_to"][arc_id] not in N:
            raise ValueError(f"arc {arc_id} references an unknown node")

    if set(data["asset_capacity_tpd"]) != A:
        raise ValueError("asset_capacity_tpd keys must match A")
    if set(data["zone_min_inventory_tonnes"]) != P or set(data["zone_max_inventory_tonnes"]) != P:
        raise ValueError("stockyard inventory bound keys must match P")
    if set(data["overflow_penalty_per_tonne"]) != P:
        raise ValueError("overflow penalty keys must match P")

    for asset, tasks in data["M_a"].items():
        if asset not in A:
            raise ValueError(f"M_a references unknown asset {asset}")
        unknown_tasks = set(tasks) - M
        if unknown_tasks:
            raise ValueError(f"M_a[{asset}] references unknown maintenance tasks {sorted(unknown_tasks)}")

    task_to_assets: dict[str, list[str]] = {task: [] for task in M}
    for asset, tasks in data["M_a"].items():
        for task in tasks:
            task_to_assets[task].append(asset)
    unassigned = sorted(task for task, assets in task_to_assets.items() if not assets)
    if unassigned:
        raise ValueError(f"maintenance tasks are not assigned to assets: {unassigned}")

    e_m = {task: int(day) for task, day in data["maintenance_earliest_start"].items()}
    l_m = {task: int(day) for task, day in data["maintenance_latest_start"].items()}
    if set(e_m) != M or set(l_m) != M:
        raise ValueError("maintenance window keys must match M")

    p = _parse_pair_float_mapping(data["maintenance_duration_days"], int)
    eta = _parse_pair_float_mapping(data["maintenance_available_capacity_fraction"], float)
    required_mode_pairs = {(task, mode) for task in M for mode in K}
    if set(p) != required_mode_pairs:
        raise ValueError("maintenance_duration_days must contain every (m,k) combination")
    if set(eta) != required_mode_pairs:
        raise ValueError("maintenance_available_capacity_fraction must contain every (m,k) combination")
    horizon_end = max(D)
    for task in M:
        if e_m[task] not in day_set or l_m[task] not in day_set or e_m[task] > l_m[task]:
            raise ValueError(f"maintenance task {task} has an invalid start window")
        for mode in K:
            duration = p[(task, mode)]
            if duration < 1:
                raise ValueError(f"maintenance duration must be positive for {(task, mode)}")
            if not 0.0 <= eta[(task, mode)] <= 1.0:
                raise ValueError(f"maintenance capacity fraction must be in [0,1] for {(task, mode)}")
            if l_m[task] + duration - 1 > horizon_end:
                raise ValueError(f"maintenance task {task} with mode {mode} cannot fit inside D")

    z0 = _parse_pair_float_mapping(data["initial_inventory_tonnes"], float)
    if set(z0) != {(zone, commodity) for zone in P for commodity in G}:
        raise ValueError("initial_inventory_tonnes must contain every (p,g) combination and no others")
    for zone in P:
        z_min = float(data["zone_min_inventory_tonnes"][zone])
        z_max = float(data["zone_max_inventory_tonnes"][zone])
        if z_min < 0.0 or z_max < z_min:
            raise ValueError(f"invalid stockyard bounds for {zone}")
        total = sum(z0[(zone, commodity)] for commodity in G)
        if not z_min <= total <= z_max:
            raise ValueError(f"aggregate initial inventory outside bounds for {zone}")

    mu_min = {int(day): float(value) for day, value in data["minimum_throughput_tpd"].items()}
    if set(mu_min) != day_set:
        raise ValueError("minimum_throughput_tpd keys must match D")

    sigma = _parse_asset_day_mapping(data["special_activity_capacity_factor"], float)
    for (asset, day), value in sigma.items():
        if asset not in A or day not in day_set or not 0.0 <= value <= 1.0:
            raise ValueError(f"invalid sigma entry {(asset, day)}")

    h = _parse_asset_day_mapping(data["maintenance_activity_incompatible"], int)
    for (asset, day), value in h.items():
        if asset not in A or day not in day_set or value not in (0, 1):
            raise ValueError(f"invalid h entry {(asset, day)}")


def _parse_pair_float_mapping(mapping: Mapping[Any, Any], value_type) -> dict[tuple[str, str], Any]:
    parsed = {}
    for key, value in mapping.items():
        left, right = _split_pair_key(key)
        parsed[(left, right)] = value_type(value)
    return parsed


def _parse_asset_day_mapping(mapping: Mapping[Any, Any], value_type) -> dict[tuple[str, int], Any]:
    parsed = {}
    for key, value in mapping.items():
        asset, day = _split_pair_key(key)
        parsed[(asset, int(day))] = value_type(value)
    return parsed


def _split_pair_key(key: Any) -> tuple[str, str]:
    if isinstance(key, (list, tuple)) and len(key) == 2:
        return str(key[0]), str(key[1])
    parts = str(key).split("::")
    if len(parts) != 2:
        raise ValueError(f"expected a two-part key separated by '::', received {key!r}")
    return parts[0], parts[1]


def _normalise_adjacency(mapping: Mapping[str, Any], nodes: list[str]) -> dict[str, list[str]]:
    normalised = {node: list(mapping.get(node, [])) for node in nodes}
    for node, neighbours in mapping.items():
        if node not in normalised:
            normalised[str(node)] = list(neighbours)
    return normalised
