"""Run monolithic_milp's model against scenario JSON files."""

from monolithic_milp import build_model, load_scenario
from scenario_runner import run_cli


def main():
    run_cli(
        model_label="single_commodity",
        load_scenario=load_scenario,
        build_model=build_model,
        default_scenario_dir="synthetic_data_generator/singlecommodity_scenarios",
        default_excel_path="results/singlecommodity_results.xlsx",
        default_csv_path="results/singlecommodity_results.csv",
        description=__doc__,
    )


if __name__ == "__main__":
    main()
