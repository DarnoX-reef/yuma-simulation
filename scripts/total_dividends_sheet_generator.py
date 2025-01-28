from dataclasses import replace

from yuma_simulation._internal.cases import cases
from yuma_simulation._internal.simulation_utils import generate_total_dividends_table
from yuma_simulation._internal.yumas import (
    SimulationHyperparameters,
    YumaParams,
    YumaSimulationNames,
)


def main():
    # List of bond_penalty values and corresponding file names
    bond_penalty_values = [0, 0.5, 0.99, 1.0]

    for bond_penalty in bond_penalty_values:
        # Define simulation hyperparameters
        simulation_hyperparameters = SimulationHyperparameters(
            bond_penalty=bond_penalty,
        )
        # Make sure the output file name matches the bond_penalty parameter
        file_name = f"total_dividends_b{bond_penalty}.csv"

        # Define Yuma parameter variations
        base_yuma_params = YumaParams()
        liquid_alpha_on_yuma_params = YumaParams(  # noqa: F841
            liquid_alpha=True,
        )

        yuma4_params = YumaParams(
            bond_alpha=0.025,
            alpha_high=0.99,
            alpha_low=0.9,
        )
        yuma4_liquid_params = replace(yuma4_params, liquid_alpha=True)  # noqa: F841

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

        print(
            f"Starting generation of total dividends table for bond_penalty={bond_penalty}."
        )
        dividends_df = generate_total_dividends_table(
            cases=cases,
            yuma_versions=yuma_versions,
            simulation_hyperparameters=simulation_hyperparameters,
        )

        # Check for missing values
        if dividends_df.isnull().values.any():
            print(
                f"CSV for bond_penalty={bond_penalty} contains missing values. Please check the simulation data."
            )
        else:
            print(
                f"No missing values detected in the CSV data for bond_penalty={bond_penalty}."
            )

        # Save the DataFrame to a CSV file
        dividends_df.to_csv(file_name, index=False, float_format="%.6f")
        print(f"CSV file {file_name} has been created successfully.")


if __name__ == "__main__":
    main()
