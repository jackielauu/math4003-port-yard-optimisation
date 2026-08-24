# math4003-port-yard-optimisation
MATH4003 Honours Dissertation: Integrated Mid-Term Port Yard Planning in Iron Ore Supply Chains: A Benders Decomposition Approach

## Run Current Models on Scenario Data

**Compatibility note:** the regenerated benchmark data in
`synthetic_data_generator/singlecommodity_scenarios` and
`synthetic_data_generator/multicommodity_scenarios` use the frozen canonical
schema. The single- and multi-commodity monolithic MILP loaders read that schema
directly through `model_data_loader.py`.

The repository currently has two runnable MILP models:

- `run_scenarios.py` for the single-commodity monolithic MILP in `monolithic_milp.py`;
- `run_scenarios_multicommodity.py` for the multi-commodity extension in `multicommodity_milp.py`.

Run all checked-in single-commodity scenarios:

```bash
python run_scenarios.py --excel
```

Run all checked-in multi-commodity scenarios:

```bash
python run_scenarios_multicommodity.py --excel
```

By default, these commands read from:

- `synthetic_data_generator/singlecommodity_scenarios`
- `synthetic_data_generator/multicommodity_scenarios`

and save Excel workbooks to:

- `results/singlecommodity_results.xlsx`
- `results/multicommodity_results.xlsx`

To test your own generated scenario folder:

```bash
python run_scenarios.py --scenario-dir path/to/your/singlecommodity_scenarios --excel results/my_single_results.xlsx
python run_scenarios_multicommodity.py --scenario-dir path/to/your/multicommodity_scenarios --excel results/my_multi_results.xlsx
```

To test one scenario file only:

```bash
python run_scenarios.py path/to/scenario.json --excel results/one_scenario.xlsx
```

For a notebook workflow, open `run_model_tests.ipynb`. Its first code cell contains the editable run parameters:

- `MODEL` selects `"single"`, `"multi"`, or `"both"`;
- `SCENARIO_INPUTS` selects a scenario folder, one JSON file, or several folders/files;
- `TIME_LIMIT_SECONDS` controls the per-scenario Gurobi time limit;
- `WORKERS` controls how many scenarios run at the same time;
- `THREADS_PER_SOLVE` controls Gurobi's thread count for each scenario solve;
- `EXCEL_PATH` and `CSV_PATH` control the table output file names;
- `SAVE_PAYLOADS` controls whether detailed solution JSON files are saved;
- `MAKE_PLOTS` controls whether PNG plots are created from those payloads;
- `PAYLOAD_DIR` and `PLOT_DIR` control the payload and plot output folders.

Set `MODEL = "both"` in the notebook to run the single-commodity and multi-commodity models in one pass. With `SCENARIO_INPUTS = None`, the notebook uses each model's matching default data folder and writes separate Excel files:

- `results/singlecommodity_results.xlsx`
- `results/multicommodity_results.xlsx`

To run scenarios in parallel from the command line:

```bash
python run_scenarios_multicommodity.py --workers 2 --threads-per-solve 1 --time-limit 60 --excel
```

For parallel runs, start conservatively. The total CPU pressure is roughly `WORKERS * THREADS_PER_SOLVE`, and your Gurobi license must allow simultaneous solves.

To save detailed solution payloads and PNG visualisations in the same run:

```bash
python run_scenarios_multicommodity.py --time-limit 60 --excel --payload-dir --plot-dir
```

This writes:

- the Excel summary to `results/multicommodity_results.xlsx`;
- per-scenario solution JSON payloads to `results/payloads/multi_commodity`;
- per-scenario PNG plots to `results/plots/multi_commodity`.

The payload JSON files contain the selected maintenance schedule, stockyard inventory, throughput, overflow, and run metadata. They are what the PNG plotter uses, so you can re-plot later without solving again:

```bash
python visualize_results.py results/payloads/multi_commodity --output-dir results/plots/multi_commodity
```

In `run_model_tests.ipynb`, set:

```python
SAVE_PAYLOADS = True
MAKE_PLOTS = True
PREVIEW_PLOTS = True
```

After saving outputs, the notebook's final preview cell displays the first few generated PNGs inline.

Useful options:

```bash
--time-limit 60      # per-scenario Gurobi time limit, in seconds
--max-scenarios 5    # quick smoke test on the first five JSON files
--workers 2          # run two scenario solves in parallel
--threads-per-solve 1 # use one Gurobi thread per scenario solve
--payload-dir        # save detailed JSON payloads for each solved scenario
--plot-dir           # save PNG plots from the payloads
--strict             # strict no-overflow capacity variant
--csv                # also save a CSV table
```
