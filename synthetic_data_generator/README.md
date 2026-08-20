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
- special activities `H`, activity start windows `W_h`, durations, realized capacity profiles `sigma[a,d]`, and maintenance-incompatibility inputs `h[a,d]`;
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

A special product is not the same thing as an operational activity. The `Product` and `SpecialProductRequirements` records define X1's physical and operating rules. `PriorityActivity` records then represent protected operating windows that create the active `sigma[a,d]` and `h[a,d]` inputs used by the current MILP.

## Synthetic X1 Assumptions

The benchmark X1 requirement record uses:

- `product_id = "X1"`, `name = "Special Product X1"`, `is_special = True`;
- `blending_allowed = False`;
- dedicated compatible zone `Z4`;
- density `3.0 tonnes/m2`;
- minimum reserved footprint `90,000 m2`;
- segregation buffer `8,000 m2`;
- Z4 footprint capacity `185,000 m2`;
- common deterministic route through `TU2`, `ST3`, `Z4`, `RC2`, and `SL1`;
- benchmark campaign-volume metadata `252,000 tonnes`;
- benchmark contiguous-campaign metadata `9 days`;
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

X1 has one deterministic compatible path through the nested core terminal topology. No unrelated second network is created. The same route is used at every F1 scale and by the reference M topology:

```text
RAIL_SOURCE -> TU2 -> ST3 -> Z4 -> RC2 -> SL1 -> VESSEL_SINK
```

The required asset tuple is:

```text
("TU2", "ST3", "RC2", "SL1")
```

The exporter includes:

```python
product_asset_compatible
product_arc_compatible
special_product_required_assets
special_product_compatible_arcs
```

Validation checks that the route is connected, every asset and arc exists, the dedicated zone is on the route, the route reaches a valid shiploader and vessel sink, and no generated route refers to nonexistent assets.

## Protected Operational Activities

`PriorityActivity` records represent protected operating windows rather than serving as the product definition. Activity-level exports include:

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

For F4 scenarios, activity windows define either maintenance incompatibility, reduced activity capacity, or both. Other scenarios still contain the X1 product definition and requirements, but they do not add protected activity windows unless required by the scenario family.

## Generated Scenarios

The current benchmark suite is:

| Scenario group | Cases | Main data changed |
|---|---|---|
| `base_control` | 1 | Calibrated common network, inventory, throughput profile, and maintenance windows |
| `F1_system_scale_*` | L/M/H | Physical asset/network size, maintenance-operation count, and scaled operating quantities |
| `F2_mode_tradeoff_*` | L/M/H | Maintenance duration ratios for `p_mk` |
| `F3_yard_flow_*` | L/M/H | Yard capacities, minimum inventory, initial inventory, and `mu_min` |
| `F4_operational_conflict_*` | L/M/H | Protected activity windows, `sigma[a,d]`, `h[a,d]`, and selected maintenance windows |
| `structurally_infeasible` | validation | Minimum throughput above physical capacity for infeasibility validation |

## Seeded Realization Contract

Scenario families define the treatment; the seed defines the stochastic instance within that treatment. The generator uses independent named streams for common supply, demand, inventory, maintenance-pool, maintenance-window, throughput, and F4 activity-window realization. Common streams do not include the full scenario name, so paired scenarios share the same underlying stochastic data unless that family explicitly owns the parameter.

The following remain structural/configuration data rather than seeded data: product and grade definitions, compatibility rules, nominal asset and arc rates, maintenance capacity factors, and the F1/F2/F3/F4 level tables.

F1 system-scale cases use explicit nested topology profiles: `L = 2 TU / 3 ST / 2 RC / 1 SL`, `M = 3 / 4 / 3 / 2`, and `H = 4 / 5 / 4 / 3`. Their maintenance-operation counts are `8 / 12 / 16`, all with the common `F/P30/P70` mode set. F1 arcs are incremental (`E_L` retained inside `E_M`, and `E_M` retained inside `E_H`) with unchanged capacities and product compatibility. Supply, demand, yard capacities, footprint capacities, initial inventory magnitudes, and minimum throughput scale with effective source-to-sink network capacity from a deterministic max-flow calculation so relative utilisation remains comparable. F2 cases keep the same maintenance tasks and windows and change only mode-duration ratios. F3 cases keep common supply, demand, and maintenance data while changing yard capacities, minimum inventories, initial-inventory targets, and throughput pressure. F4 cases realize protected activity timing and select generated maintenance tasks by route/export properties rather than fixed task IDs.

## Base, Manual Baseline, Reference Witness, Optimiser Result

These are separate concepts:

1. **Base spread schedule**: deterministic initial schedule created by `build_spread_baseline_schedule()`.
2. **Manual baseline**: deterministic non-optimised schedule after the scenario-family overlay.
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
    "F4_operational_conflict_M",
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

Default benchmark:

```bash
python generate_dissertation_scenarios.py \
  --occasion all \
  --output-dir generated_dissertation_scenarios \
  --horizon-days 90 \
  --seed 42
```

The manifest records topology, maintenance modes, seed, randomization version, RNG stream names, horizon, X1 ID, compatible zone, required route assets, footprint assumptions, active protected campaign status, realized maintenance counts, realized initial-inventory ratios, and design-check status.

## Tests

```bash
python -m unittest -v test_generate_dissertation_scenarios.py
```

The suite covers the current scenario set, deterministic generation, seed-to-seed realization differences, paired common streams, F1 system-scale topology/scaling, active model-ready keys, maintenance-mode factors, dynamic network connectivity, X1 identity, route continuity, operational activity exports, JSON serialization, mode/yards-flow/conflict families, and structural infeasibility.


