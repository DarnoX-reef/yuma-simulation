import os
import bittensor as bt

from dataclasses import replace
from yuma_simulation._internal.logger_setup import main_logger as logger
from yuma_simulation._internal.simulation_utils import generate_total_dividends_table
from yuma_simulation._internal.yumas import (
    SimulationHyperparameters,
    YumaParams,
    YumaSimulationNames,
)
from yuma_simulation._internal.metagraph_utils import load_metas_from_directory
from yuma_simulation._internal.cases import MetagraphCase

from common_cli import _create_common_parser
from yuma_simulation._internal.experiment_setup import ExperimentSetup
from yuma_simulation._internal.metagraph_utils import (
    DownloadMetagraph,
)


def create_output_dir(output_dir):
    """
    Creates the output directory if it does not exist.
    """
    if not os.path.exists(f"./simulation_results/{output_dir}"):
        os.makedirs(f"./simulation_results/{output_dir}")
        logger.info(f"Created output directory: {output_dir}")
    else:
        logger.debug(f"Output directory already exists: {output_dir}")


def main():
    parser = _create_common_parser()
    args = parser.parse_args()
    create_output_dir(args.output_dir)

    two_days_blocks = 14400
    current_block = bt.subtensor().get_current_block()
    start_block = current_block - two_days_blocks

    if args.download_new_metagraph:
        setup = ExperimentSetup(
            netuids=[args.subnet_id],
            start_block=start_block,
            tempo=args.tempo,
            data_points=args.epochs,
            metagraph_storage_path=f"./metagraph_diagnostic/{args.metagraphs_dir}",
            result_path="./results",
            liquid_alpha=False,
        )

        downloader = DownloadMetagraph(setup)
        downloader.run()

    try:
        logger.info("Loading metagraphs.")
        metas = load_metas_from_directory(f"./metagraph_diagnostic/{args.metagraphs_dir}")
    except Exception:
        logger.error("Error while loading metagraphs", exc_info=True)
        return

    if not metas:
        logger.error("No metagraphs loaded. Nothing to be generated.")
        return
    logger.debug(f"Loaded {len(metas)} metagraphs from {args.metagraphs_dir}.")

    for bond_penalty in args.bond_penalties:
        logger.info(f"Calculating total dividends for bond_penalty={bond_penalty}")

        simulation_hyperparameters = SimulationHyperparameters(
            bond_penalty=bond_penalty,
        )

        if args.introduce_shift:
            file_name = f"./simulation_results/{args.output_dir}/metagraph_total_dividends_shifted_b{bond_penalty}.csv"
            logger.debug(f"Output file: {file_name}")
        else:
            file_name = f"./simulation_results/{args.output_dir}/metagraph_total_dividends_results_b{bond_penalty}.csv"
            logger.debug(f"Output file: {file_name}")

        base_yuma_params = YumaParams()
        liquid_alpha_on_yuma_params = replace(base_yuma_params, liquid_alpha=True)

        yuma4_params = YumaParams(
            bond_alpha=0.025,
            alpha_high=0.99,
            alpha_low=0.9,
        )
        yuma4_liquid_params = replace(yuma4_params, liquid_alpha=True)

        yumas = YumaSimulationNames()
        yuma_versions = [
            # (yumas.YUMA_RUST, base_yuma_params),
            # (yumas.YUMA, base_yuma_params),
            # (yumas.YUMA_LIQUID, liquid_alpha_on_yuma_params),
            # (yumas.YUMA2, base_yuma_params),
            # (yumas.YUMA3, base_yuma_params),
            # (yumas.YUMA31, base_yuma_params),
            # (yumas.YUMA32, base_yuma_params),
            # (yumas.YUMA4, base_yuma_params),
            (yumas.YUMA4_LIQUID, yuma4_liquid_params),
        ]

        try:
            logger.info("Creating MetagraphCase.")
            case = MetagraphCase(
                shift_validator_id=args.shift_validator_id,
                name="Metagraph simulation",
                metas=metas,
                num_epochs=len(metas),
                introduce_shift=args.introduce_shift,
            )
            logger.debug(f"MetagraphCase created successfully: {case.name}")
        except Exception:
            logger.error("Error while creating MetagraphCase.", exc_info=True)
            return

        logger.info(
            f"Starting generation of total dividends table for bond_penalty={bond_penalty}."
        )
        dividends_df = generate_total_dividends_table(
            cases=[case],
            yuma_versions=yuma_versions,
            simulation_hyperparameters=simulation_hyperparameters,
            is_metagraph=True,
        )

        dividends_df = dividends_df.applymap(
            lambda x: f"{x:.3e}" if isinstance(x, (float, int)) and abs(x) < 1e-6 else x
        )

        # Save the dataframe to CSV
        dividends_df.to_csv(file_name, index=False)
        logger.info(f"CSV file {file_name} has been created successfully.")


if __name__ == "__main__":
    main()
