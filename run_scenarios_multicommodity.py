"""Run multicommodity_milp's model against scenario JSON files."""

from multicommodity_milp import build_model, load_scenario
from scenario_runner import run_cli


def main():
    run_cli(
        model_label="multi_commodity",
        load_scenario=load_scenario,
        build_model=build_model,
        default_scenario_dir="synthetic_data_generator/multicommodity_scenarios",
        default_excel_path="results/multicommodity_results.xlsx",
        default_csv_path="results/multicommodity_results.csv",
        description=__doc__,
    )


if __name__ == "__main__":
    main()
