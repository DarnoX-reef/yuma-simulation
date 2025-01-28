from dataclasses import replace
import bittensor as bt
import os
import json
import argparse

from yuma_simulation._internal.logger_setup import main_logger as logger
from yuma_simulation._internal.yumas import (
    SimulationHyperparameters,
    YumaParams,
    YumaSimulationNames,
)

from yuma_simulation.v1.api import generate_metagraph_based_dividends
from yuma_simulation._internal.experiment_setup import ExperimentSetup
from yuma_simulation._internal.metagraph_utils import (
    DownloadMetagraph,
    load_metas_from_directory,
)
from common_cli import _create_common_parser


def create_output_dir(output_dir, subnet_id):
    """
    Creates the output directory if it does not exist.
    """
    if not os.path.exists(f"./{output_dir}/subnet_{subnet_id}"):
        os.makedirs(f"./{output_dir}/subnet_{subnet_id}")
        logger.info(f"Created output directory: {output_dir}")
    else:
        logger.debug(f"Output directory already exists: {output_dir}")


def run_single_scenario(args):
    """
    Encapsulates the logic to run a single subnet scenario with the given arguments.
    """
    create_output_dir(args.output_dir, args.subnet_id)

    two_days_blocks = 14400
    current_block = bt.subtensor().get_current_block()
    start_block = current_block - two_days_blocks

    if args.download_new_metagraph:
        setup = ExperimentSetup(
            netuids=[args.subnet_id],
            start_block=start_block,
            tempo=args.tempo,
            data_points=args.epochs,
            metagraph_storage_path=f"./{args.metagraphs_dir}/subnet_{args.subnet_id}",
            result_path="./results",
            liquid_alpha=False,
        )

        downloader = DownloadMetagraph(setup)
        downloader.run()

    try:
        logger.info("Loading metagraphs.")
        metas = load_metas_from_directory(f"./{args.metagraphs_dir}/subnet_{args.subnet_id}")
    except Exception:
        logger.error("Error while loading metagraphs", exc_info=True)
        return

    if not metas:
        logger.error("No metagraphs loaded. Nothing to be generated.")
        return
    logger.debug(f"Loaded {len(metas)} metagraphs from {args.metagraphs_dir}.")

    try:
        for bond_penalty in args.bond_penalties:
            logger.info(f"Running simulation for bond_penalty={bond_penalty}")

            simulation_hyperparameters = SimulationHyperparameters(
                bond_penalty=bond_penalty,
            )

            if args.introduce_shift:
                file_name = f"./{args.output_dir}/subnet_{args.subnet_id}/metagraph_simulation_results_shifted_b{bond_penalty}.html"
                logger.debug(f"Output file: {file_name}")
            else:
                file_name = f"./{args.output_dir}/subnet_{args.subnet_id}/metagraph_simulation_results_b{bond_penalty}.html"
                logger.debug(f"Output file: {file_name}")

            yuma4_params = YumaParams(bond_alpha=0.025, alpha_high=0.99, alpha_low=0.9)
            yuma4_liquid_params = replace(yuma4_params, liquid_alpha=True)

            yumas = YumaSimulationNames()
            yuma_versions = [
                (yumas.YUMA4_LIQUID, yuma4_liquid_params),
            ]

            try:
                chart_table = generate_metagraph_based_dividends(
                    yuma_versions=yuma_versions,
                    yuma_hyperparameters=simulation_hyperparameters,
                    shift_validator_id=args.shift_validator_id,
                    metas=metas,
                    draggable_table=args.draggable_table,
                    introduce_shift=args.introduce_shift,
                    highlight_validator=args.highlight_validator,
                )

                with open(file_name, "w", encoding="utf-8") as f:
                    f.write(chart_table.data)
                    logger.info(f"Simulation results saved to {file_name}")

            except Exception as e:
                logger.error(f"Error generating the chart table: {e}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)


def main():
    parser = _create_common_parser()
    args = parser.parse_args()

    if args.use_json_config:
        # MULTI-RUN MODE
        with open(args.config_file, "r") as f:
            config_data = json.load(f)

        scenarios = config_data.get("scenarios", [])
        if not scenarios:
            logger.error("No scenarios found in the JSON file.")
            return

        for index, scenario_dict in enumerate(scenarios, start=1):
            logger.info(f"\n===== Running scenario {index} =====")

            scenario_args = argparse.Namespace(**vars(args))
            for key, value in scenario_dict.items():
                setattr(scenario_args, key, value)
            run_single_scenario(scenario_args)

    else:
        # SINGLE-RUN MODE
        run_single_scenario(args)


if __name__ == "__main__":
    main()
