from dataclasses import replace

from yuma_simulation._internal.logger_setup import main_logger as logger
from yuma_simulation._internal.cases import cases
from yuma_simulation._internal.yumas import (
    SimulationHyperparameters,
    YumaParams,
    YumaSimulationNames,
)
from yuma_simulation.v1.api import generate_chart_table


def main():
    # List of bond_penalty values and corresponding file names
    bond_penalty_values = [0, 0.5, 0.99, 1.0]

    for bond_penalty in bond_penalty_values:
        # Setting global simulation parameters
        simulation_hyperparameters = SimulationHyperparameters(
            bond_penalty=bond_penalty,
        )

        file_name = f"simulation_results_b{bond_penalty}.html"

        # Setting individual yuma simulations parameters
        base_yuma_params = YumaParams()
        liquid_alpha_on_yuma_params = YumaParams(
            liquid_alpha=True,
        )
        yuma4_params = YumaParams(
            bond_alpha=0.025,
            alpha_high=0.99,
            alpha_low=0.9,
        )
        yuma4_liquid_params = replace(yuma4_params, liquid_alpha=True)

        yumas = YumaSimulationNames()
        yuma_versions = [
            (yumas.YUMA_RUST, base_yuma_params),
            (yumas.YUMA, base_yuma_params),
            (yumas.YUMA_LIQUID, liquid_alpha_on_yuma_params),
            (yumas.YUMA2, base_yuma_params),
            (yumas.YUMA3, base_yuma_params),
            (yumas.YUMA31, base_yuma_params),
            (yumas.YUMA32, base_yuma_params),
            (yumas.YUMA4, base_yuma_params),
            (yumas.YUMA4_LIQUID, yuma4_liquid_params),
        ]

        logger.info("Generating chart table...")
        chart_table = generate_chart_table(
            cases, yuma_versions, simulation_hyperparameters, draggable_table=True
        )

        # Save the HTML file
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(chart_table.data)

        logger.info(f"HTML saved to {file_name}")


if __name__ == "__main__":
    main()
