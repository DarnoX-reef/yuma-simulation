"""
This module defines a common CLI parser for metagraph-based data simulation scripts.
It provides various arguments to configure the simulation, including parameters
for subnet id, bond penalties, epochs, tempo, and output settings.
"""

import argparse


def _create_common_parser():
    parser = argparse.ArgumentParser(
        description="Common CLI for Yuma simulation scripts."
    )

    parser.add_argument(
        "--subnet-id",
        type=int,
        default=0,
        help="The subnet id to run the diagnostics on.",
    )

    parser.add_argument(
        "--bond-penalties",
        nargs="+",
        type=float,
        default=[0, 0.5, 0.99, 1.0],
        help="List of bond penalties to simulate.",
    )
    parser.add_argument(
        "--epochs", type=int, default=40, help="Number of epochs to simulate."
    )
    parser.add_argument(
        "--tempo", type=int, default=360, help="Tempo value for the simulation."
    )
    parser.add_argument(
        "--start-block-offset",
        type=int,
        default=14400,
        help="Number of blocks to look back for the start block.",
    )
    parser.add_argument(
        "--shift-validator-id",
        type=int,
        default=0,
        help="The uid of the validator with the weights shifted by 1 epoch back.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="The dir to store simulation results",
    )
    parser.add_argument(
        "--metagraphs-dir",
        type=str,
        default="metagraphs",
        help="Directory which stores the downloaded metagraphs",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="metagraph_simulation_results",
        help="Prefix for output file names.",
    )
    parser.add_argument(
        "--draggable-table", action="store_true", help="Make the table draggable."
    )
    parser.add_argument(
        "--download-new-metagraph",
        action="store_true",
        help="Download a new metagraph.",
    )
    parser.add_argument(
        "--introduce-shift",
        action="store_true",
        help="Introduce shift in the simulation.",
    )
    parser.add_argument(
        "--use-json-config",
        action="store_true",
        help="Use a JSON config file to run multiple scenarios."
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default="metagraph_subnets_config.json",
        help="Path to JSON file with multiple scenario definitions."
    )
    parser.add_argument(
        "--highlight-validator",
        type=str,
        help="The validator that should be highlighted on a dividends plot legend."
    )

    return parser
