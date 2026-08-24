# Dissertation Synthetic Scenario Generator

This directory generates benchmark instances for the frozen MATH4003 port-yard
MILP formulation. The generator now uses one canonical data model for both
single- and multi-commodity instances.

## Commodity Model

Single-commodity instances are represented as:

```python
G = ["G1"]
```

Multi-commodity instances are represented as:

```python
G = ["G1", "G2", "G3", "G4"]
```

The scenario family, seed, topology, capacities, maintenance operations,
maintenance windows, mode durations, yard bounds, minimum throughput, fixed
activity capacity factors, and maintenance/activity incompatibilities are
matched between the corresponding single- and multi-commodity instances. The
only structural difference is the cardinality of `G`. Multi-commodity initial
inventory is split reproducibly across `G1`-`G4` so that each zone total exactly
matches the single-commodity case.

## Canonical `model_ready_data`

The optimisation input schema is intentionally restricted to frozen-model data:

```text
D
N
E
A
A_NS
P
G
M
M_a
K
arc_from
arc_to
predecessors
successors
asset_capacity_tpd
arc_capacity_tpd
maintenance_duration_days
maintenance_available_capacity_fraction
maintenance_earliest_start
maintenance_latest_start
zone_min_inventory_tonnes
zone_max_inventory_tonnes
initial_inventory_tonnes
overflow_penalty_per_tonne
minimum_throughput_tpd
special_activity_capacity_factor
maintenance_activity_incompatible
```

Tuple-indexed dictionaries are serialized with `::` in JSON. For example:

```text
initial_inventory_tonnes["Z1::G1"]
maintenance_duration_days["M_ST1_01::P30"]
special_activity_capacity_factor["ST1::35"]
```

The benchmark no longer exports supply, demand, special products, X1
requirements, grades, blending flags, product compatibility, footprint,
segregation, campaign-volume, cleanout, maintenance-cost, or activity-start
data. Diagnostics that are useful for design review live outside
`model_ready_data`.

## Families

The experimental suite has four families, three levels, and ten seeds:

| Family | Levels | Main treatment |
|---|---:|---|
| `F1_system_scale` | L/M/H | Network and maintenance scale |
| `F2_mode_tradeoff` | L/M/H | Maintenance duration-vs-capacity ratios |
| `F3_yard_flow` | L/M/H | Yard bounds, initial inventory, throughput pressure, and upstream fixed restrictions |
| `F4_operational_conflict` | L/M/H | Fixed operational capacity/incompatibility windows |

Each commodity suite contains 120 instances.

## Scale And Hardness Targets

F1 uses approximately:

| Level | Assets | Maintenance operations |
|---|---:|---:|
| L | 12 | 30 |
| M | 20 | 60 |
| H | 32 | 100 |

F2 uses exactly three maintenance modes:

```text
K = {F, P30, P70}
eta = {0.00, 0.30, 0.70}
```

The central duration-ratio calibration is:

```text
L: F/P30/P70 = 1.00 / 1.35 / 2.85
M: F/P30/P70 = 1.00 / 1.45 / 3.30
H: F/P30/P70 = 1.00 / 1.60 / 3.80
```

Task-level seeded variation is applied to avoid identical ratios across all
maintenance operations.

## Generate Data

From the repository root:

```bash
python synthetic_data_generator/generate_dissertation_scenarios.py \
  --commodity-count 1 \
  --output-dir synthetic_data_generator/singlecommodity_scenarios \
  --seeds 1-10

python synthetic_data_generator/generate_dissertation_scenarios.py \
  --commodity-count 4 \
  --output-dir synthetic_data_generator/multicommodity_scenarios \
  --seeds 1-10
```

The legacy single-commodity filename is now only a wrapper:

```bash
python synthetic_data_generator/generate_dissertation_scenarios_for_single_commodity.py
```

## Tests

```bash
python -m unittest -v synthetic_data_generator/test_generate_dissertation_scenarios.py
python -m unittest -v test_milp_canonical_loaders.py
```

The tests enforce the canonical schema, reject forbidden legacy fields, verify
the F1 scale ladder, check mode trade-off calibration, validate yard and
throughput pressure, check F4 fixed-profile severity, and confirm matched
single/multi initial inventories. The MILP loader tests validate the shared
canonical loader and build representative single/multi monolithic models when a
Gurobi license is available.

## Solver Compatibility

The single- and multi-commodity monolithic MILP loaders now read the canonical
schema directly through `model_data_loader.py`. Deleted product-specific and
legacy alias fields are not required by those loaders. Benders implementations
were not modified as part of this generator/monolithic-loader integration.
