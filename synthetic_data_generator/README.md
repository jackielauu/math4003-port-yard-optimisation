# Dissertation Synthetic Scenario Generator

This package creates standalone synthetic data for the **MATH4003 Integrated Mid-Term Port Yard Planning** model. It has no third-party Python dependency and does not require an operational database or another software project.

The generator supports formulation development, a monolithic MILP, Benders decomposition, feasibility testing, edge cases, baseline comparisons, and computational experiments.

## Current Model Alignment

The generator exports the main inputs required by the mathematical formulation:

- daily time set `D` over a default 90-day horizon;
- directed network nodes `N` and arcs `E`;
- capacity-constrained processing assets `A` and non-storage assets `A_NS`;
- stockyard nodes `P_STOCKYARD` with tonnage and footprint capacity;
- product set `G`, including a first-class special product;
- maintenance operations `M_OPERATIONS`, asset-specific operation sets `M_a`, and feasible start windows `W_m`;
- special activities `H`, activity start windows `W_h`, durations, fixed capacity profiles `sigma[a,d]`, and maintenance-incompatibility inputs `h[a,d]`;
- three maintenance modes `K = {F, P30, P70}`;
- nominal daily asset capacities `C_a` and daily arc capacities `U_i,j`;
- minimum daily throughput targets;
- product-specific X1 daily and campaign-volume commitments.

Legacy keys such as `T`, `P`, `Z`, `J`, and `M` are retained so earlier optimiser code does not need to be rewritten immediately.

## Benchmark Special Product: X1

The four-product benchmark is preserved. Product `X1` is now the benchmark special product:

```python
SPECIAL_PRODUCTS = ["X1"]

product_is_special = {
    "P1": 0,
    "P2": 0,
    "P3": 0,
    "X1": 1,
}
```

`X1` is named `Special Product X1`. The X1 numerical values are **synthetic planning assumptions** chosen to create a reproducible dissertation benchmark; they are not claimed to be real operational data.

Conceptually:

```text
X1 product definition
    -> spatial and routing requirements
    -> protected X1 campaign
    -> maintenance exclusions
    -> product-specific flow and volume constraints
```

A special product is not the same thing as a priority activity. The `Product` and `SpecialProductRequirements` records define X1's physical and operating rules. `PriorityActivity` then represents one protected X1 campaign that consumes those rules: route, zone, footprint, buffers, cleanout, and daily/total tonnes.

## Synthetic X1 Assumptions

The benchmark X1 requirement record uses:

- `product_id = "X1"`, `name = "Special Product X1"`, `is_special = True`;
- `blending_allowed = False`;
- dedicated compatible zone `Z4`;
- density `3.0 tonnes/m2`;
- minimum reserved footprint `90,000 m2`;
- segregation buffer `8,000 m2`;
- Z4 footprint capacity `185,000 m2`;
- route through `TU3 -> ST4 -> Z4 -> RC3 -> SL2` in the two-shiploader benchmark;
- route through `TU3 -> ST4 -> Z4 -> RC3 -> SL1` in the one-shiploader topology;
- minimum campaign volume `252,000 tonnes`;
- minimum contiguous campaign length `9 days`;
- campaign daily minimum `28,000 tonnes/day` in `priority_campaign_conflict`;
- two-day pre- and post-campaign buffers;
- one-day cleanout before and after the loading campaign;
- minimum build-before-reclaim inventory `168,000 tonnes`.

The footprint abstraction is:

```text
required footprint =
X1 inventory tonnes / tonnes_per_square_metre
+ segregation buffer
```

During a protected campaign the evaluator reserves at least `minimum_reserved_footprint_m2`, even if the instantaneous inventory footprint is smaller.

## Products, Grades, and Blending

The generator retains chemical grade specifications for Fe, SiO2, Al2O3, Mn, P, As, and LOI. It exports `product_grade_specs` and `product_blending_allowed`.

The lightweight evaluator does not solve detailed blending. The final MILP or Benders subproblem must enforce grade and non-blending constraints, including X1's `blending_allowed = False` rule.

## Stockyard Footprint and Segregation

Each `YardZone` has both:

- `capacity_tonnes`;
- `footprint_capacity_m2`.

The model-ready output includes:

```python
zone_footprint_capacity_m2
special_product_tonnes_per_square_metre
special_product_minimum_reserved_footprint_m2
special_product_segregation_buffer_m2
special_product_compatible_zones
product_zone_compatible
special_product_zone_compatible
zone_allowed_product
product_may_share_zone
```

X1 is compatible only with `Z4`. The segregation matrix prevents X1 from sharing protected stockyard space with ordinary products. This is an aggregate planning representation; detailed stockpile shape, coordinates, reclaim sequence, and parcel geometry remain future detailed MILP/subproblem responsibilities.

## Dedicated X1 Route

X1 has a specific compatible path through the existing terminal network. No unrelated second network is created.

Two shiploaders:

```text
RAIL_SOURCE -> TU3 -> ST4 -> Z4 -> RC3 -> SL2 -> VESSEL_SINK
```

One shiploader:

```text
RAIL_SOURCE -> TU3 -> ST4 -> Z4 -> RC3 -> SL1 -> VESSEL_SINK
```

The exporter includes:

```python
product_asset_compatible
product_arc_compatible
special_product_required_assets
special_product_compatible_arcs
```

Validation checks that the route is connected, every asset and arc exists, the dedicated zone is on the route, the route reaches a valid shiploader and vessel sink, and one-shiploader scenarios never reference `SL2`.

## Protected X1 Campaigns

`PriorityActivity` now represents a protected X1 campaign rather than serving as the product definition. Campaign-level exports include:

```python
H
W_h
special_activity_product
special_activity_duration_days
special_activity_minimum_tpd
special_activity_minimum_total_tonnes
special_activity_required_assets
special_activity_required_zone
special_activity_pre_buffer_days
special_activity_post_buffer_days
special_activity_cleanout_before_days
special_activity_cleanout_after_days
special_activity_capacity_factor
maintenance_activity_incompatible
special_product_minimum_tpd
special_product_campaign_minimum_tonnes
```

The maintenance-incompatibility calendar covers:

```text
pre-campaign buffer + campaign window + post-campaign buffer
```

Cleanout days are exported through the activity cleanout fields and represented in `sigma[a,d]` as capacity restrictions on the required route assets.

## Generated Occasions

All ten occasions remain available:

| Occasion | Planning condition | Intended model demonstration |
|---|---|---|
| `balanced_control` | Normal profiles and distributed maintenance | Control case with little scheduling value |
| `peak_export_conflict` | Maintenance during a strong export peak | Improve throughput and demand satisfaction |
| `overlapping_maintenance` | Downstream jobs start together | Recover minimum-throughput feasibility by staggering |
| `priority_campaign_conflict` | Maintenance blocks the protected X1 route, buffer, or cleanout interval | Respect special-product route, zone, footprint, and campaign requirements |
| `yard_overflow_risk` | High starting stock and inbound surge | Reduce overflow and inbound curtailment |
| `inventory_starvation_risk` | Low starting stock and demand peak | Coordinate upstream work with inventory buffering |
| `tight_lifecycle_windows` | Narrow, overlapping maintenance windows | Find a valid ordering inside due dates |
| `partial_shutdown_choice` | Required work during strong demand | Choose among `F`, `P30`, and `P70` |
| `fixed_schedule_control` | Earliest start equals latest start | Negative control with no timing flexibility |
| `structurally_infeasible` | Non-movable outages make the hard minimum impossible | Correctly prove infeasibility |

`priority_campaign_conflict` is the strongly active X1 case. Ordinary control scenarios still contain the X1 product definition and requirements, but they do not fail merely because X1 exists.

## Base, Manual Baseline, Reference Witness, Optimiser Result

These are separate concepts:

1. **Base spread schedule**: deterministic initial schedule created by `_spread_schedule()`.
2. **Manual baseline**: fixed non-optimised schedule after the occasion overlay.
3. **Reference witness**: hidden construction check showing that the intended mechanism exists.
4. **Optimiser result**: schedule produced by the MILP or Benders model.

The final research comparison is:

```text
fixed manual baseline vs integrated optimiser result
```

The reference witness is not supplied to the optimiser. `optimizer_objects()` deliberately excludes `reference_schedule` and `design_evidence`.

## Model-Ready Output Example

```python
from generate_dissertation_scenarios import GeneratorConfig, build_scenario

scenario = build_scenario(
    "priority_campaign_conflict",
    GeneratorConfig(horizon_days=90, seed=42),
)

data = scenario.optimizer_objects()
X1_route = data["special_product_compatible_arcs"]["X1"]
X1_zone = data["special_product_compatible_zones"]["X1"][0]
X1_daily_minimum = data["special_product_minimum_tpd"][("X1", 50)]
```

Generated scenario JSON files include a JSON-safe `model_ready_data` object. Tuple-key dictionaries are converted using `::`, for example `X1::50`.

## Lightweight Evaluator Boundary

`evaluate_schedule()` is a fast scenario-construction check. It:

- applies maintenance, activity, cleanout, and outage capacity factors;
- aggregates available rate by stage;
- updates aggregate product inventory;
- tracks X1 inventory in the dedicated compatible zone;
- measures total throughput, demand shortfall, overflow, minimum-throughput violations, X1 daily shortfall, X1 campaign-volume shortfall, footprint breaches, incompatible-zone use, and protected-window maintenance conflicts.

It does not solve a full continuous stockpile geometry model, detailed arc-level multicommodity flow, reclaim sequencing, or blending. The final MILP or Benders subproblem remains authoritative for those constraints.

## Generate Data

Default two-shiploader benchmark:

```bash
python generate_dissertation_scenarios.py \
  --occasion all \
  --output-dir generated_dissertation_scenarios \
  --horizon-days 90 \
  --seed 42 \
  --minimum-throughput-ratio 0.55
```

One-shiploader topology:

```bash
python generate_dissertation_scenarios.py \
  --occasion all \
  --output-dir generated_pdf_topology \
  --shiploader-count 1 \
  --shiploader-rate-tph 9166
```

The manifest records topology, maintenance modes, seed, horizon, X1 ID, compatible zone, required route assets, footprint assumptions, active protected campaign status, and design-check status.

## Tests

```bash
python -m unittest -v test_generate_dissertation_scenarios.py
```

The suite contains **24 tests** covering all ten occasions, deterministic generation, legacy keys, F/P30/P70 mode factors, one- and two-shiploader topologies, first-class X1 identity, requirements, route continuity, segregation, footprint checks, campaign metrics, JSON serialization, controls, and structural infeasibility.


