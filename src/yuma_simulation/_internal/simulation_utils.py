"""
This module provides functionalities to run Yuma simulations, generate charts, and produce tables of results.
It integrates various Yuma versions, handles different chart types, and organizes the outputs into HTML tables.
"""

import pandas as pd
import torch

from yuma_simulation._internal.cases import BaseCase
from yuma_simulation._internal.charts_utils import (
    _calculate_total_dividends,
)
from yuma_simulation._internal.yumas import (
    SimulationHyperparameters,
    Yuma,
    Yuma2,
    Yuma3,
    Yuma4,
    YumaConfig,
    YumaParams,
    YumaRust,
    YumaSimulationNames,
)


def run_simulation(
    case: BaseCase,
    yuma_version: str,
    yuma_config: YumaConfig,
) -> tuple[dict[str, list[float]], list[torch.Tensor], list[torch.Tensor]]:
    """Runs the Yuma simulation for a given case and Yuma version, returning dividends, bonds and incentive data."""

    dividends_per_validator: dict[str, list[float]] = {
        validator: [] for validator in case.validators
    }
    bonds_per_epoch: list[torch.Tensor] = []
    server_incentives_per_epoch: list[torch.Tensor] = []
    B_state: torch.Tensor | None = None
    W_prev: torch.Tensor | None = None
    server_consensus_weight: torch.Tensor | None = None

    simulation_names = YumaSimulationNames()

    for epoch in range(case.num_epochs):
        W: torch.Tensor = case.weights_epochs[epoch]
        S: torch.Tensor = case.stakes_epochs[epoch]

        stakes_tao: torch.Tensor = S * yuma_config.total_subnet_stake
        stakes_units: torch.Tensor = stakes_tao / 1000.0

        # Call the appropriate Yuma function
        if yuma_version in [simulation_names.YUMA, simulation_names.YUMA_LIQUID]:
            result = Yuma(W=W, S=S, B_old=B_state, config=yuma_config)
            B_state = result["validator_ema_bond"]
        elif yuma_version == simulation_names.YUMA2:
            result = Yuma2(W=W, W_prev=W_prev, S=S, B_old=B_state, config=yuma_config)
            B_state = result["validator_ema_bond"]
            W_prev = result["weight"]
        elif yuma_version == simulation_names.YUMA3:
            result = Yuma3(W, S, B_old=B_state, config=yuma_config)
            B_state = result["validator_bonds"]
        elif yuma_version == simulation_names.YUMA31:
            if B_state is not None and epoch == case.reset_bonds_epoch:
                B_state[:, case.reset_bonds_index] = 0.0
            result = Yuma3(W, S, B_old=B_state, config=yuma_config)
            B_state = result["validator_bonds"]
        elif yuma_version == simulation_names.YUMA32:
            if (
                B_state is not None
                and epoch == case.reset_bonds_epoch
                and server_consensus_weight is not None
                and server_consensus_weight[case.reset_bonds_index] == 0.0
            ):
                B_state[:, case.reset_bonds_index] = 0.0
            result = Yuma3(W, S, B_old=B_state, config=yuma_config)
            B_state = result["validator_bonds"]
            server_consensus_weight = result["server_consensus_weight"]
        elif yuma_version in [simulation_names.YUMA4, simulation_names.YUMA4_LIQUID]:
            if (
                B_state is not None
                and epoch == case.reset_bonds_epoch
                and server_consensus_weight is not None
                and server_consensus_weight[case.reset_bonds_index] == 0.0
            ):
                B_state[:, case.reset_bonds_index] = 0.0
            result = Yuma4(W, S, B_old=B_state, config=yuma_config)
            B_state = result["validator_bonds"]
            server_consensus_weight = result["server_consensus_weight"]
        elif yuma_version == "Yuma 0 (subtensor)":
            result = YumaRust(W, S, B_old=B_state, config=yuma_config)
            B_state = result["validator_ema_bond"]
        else:
            raise ValueError("Invalid Yuma function.")

        D_normalized: torch.Tensor = result["validator_reward_normalized"]

        E_i: torch.Tensor = yuma_config.validator_emission_ratio * D_normalized
        validator_emission: torch.Tensor = E_i * yuma_config.total_epoch_emission

        for i, validator in enumerate(case.validators):
            stake_unit = float(stakes_units[i].item())
            validator_emission_i = float(validator_emission[i].item())
            if stake_unit > 1e-6:
                dividend_per_1000_tao = validator_emission_i / stake_unit
            else:
                dividend_per_1000_tao = 0.0
            dividends_per_validator[validator].append(dividend_per_1000_tao)

        bonds_per_epoch.append(B_state.clone())
        server_incentives_per_epoch.append(result["server_incentive"])

    return dividends_per_validator, bonds_per_epoch, server_incentives_per_epoch


def _generate_draggable_html_table(
    table_data: dict[str, list[str]],
    summary_table: pd.DataFrame,
    case_row_ranges: list[tuple[int, int, int]],
) -> str:
    custom_css_js = """
    <style>
        body {
            margin: 0;
            padding: 0;
            overflow: hidden;
        }
        
        .scrollable-table-container {
            background-color: #FFFFFF; 
            width: 100%; 
            height: 100vh;
            overflow: auto;
            border: 1px solid #ccc;
            position: relative; 
            user-select: none;
            scrollbar-width: auto;
            -ms-overflow-style: auto;
            cursor: grab;
        }

        .scrollable-table-container:active {
            cursor: grabbing;
        }

        /* Remove any nth-child rules entirely */

        /* Classes for alternating case groups */
        .case-group-even td {
            background-color: #FFFFFF !important; 
        }
        .case-group-odd td {
            background-color: #F0F0F0 !important;
        }

        .scrollable-table-container img {
            user-select: none;
            -webkit-user-drag: none;
            pointer-events: none;
        }

        .scrollable-table-container::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }

        table {
            border-collapse: collapse;
            margin: 0;
            width: auto;
        }

        td, th {
            padding: 10px;
            vertical-align: top;
            text-align: center;
        }

    </style>

    <script>
        document.addEventListener('DOMContentLoaded', function () {
            const container = document.querySelector('.scrollable-table-container');
            let isDown = false;
            let startX, startY, scrollLeft, scrollTop;

            container.addEventListener('dragstart', function(e) {
                e.preventDefault();
            });

            container.addEventListener('mousedown', (e) => {
                e.preventDefault();
                isDown = true;
                startX = e.clientX;
                startY = e.clientY;
                scrollLeft = container.scrollLeft;
                scrollTop = container.scrollTop;
            });

            document.addEventListener('mouseup', () => {
                isDown = false;
            });

            document.addEventListener('mousemove', (e) => {
                if(!isDown) return;
                e.preventDefault();
                const x = e.clientX;
                const y = e.clientY;
                const walkX = x - startX;
                const walkY = y - startY;
                container.scrollLeft = scrollLeft - walkX;
                container.scrollTop = scrollTop - walkY;
            });
        });
    </script>
    """

    def get_case_index_for_row(row_idx):
        for start, end, c_idx in case_row_ranges:
            if start <= row_idx <= end:
                return c_idx
        return 0

    html_rows = []
    total_rows = len(next(iter(table_data.values())))
    for i in range(total_rows):
        case_idx = get_case_index_for_row(i)
        row_class = "case-group-even" if (case_idx % 2 == 0) else "case-group-odd"
        row_html = f"<tr class='{row_class}'>"
        for yuma_version in summary_table.columns:
            cell_content = summary_table[yuma_version][i]
            row_html += f"<td>{cell_content}</td>"
        row_html += "</tr>"
        html_rows.append(row_html)

    html_table = f"""
    <div class="scrollable-table-container">
        <table>
            <thead>
                <tr>{"".join(f"<th>{col}</th>" for col in summary_table.columns)}</tr>
            </thead>
            <tbody>
                {"".join(html_rows)}
            </tbody>
        </table>
    </div>
    """

    return custom_css_js + html_table


def _generate_ipynb_table(
    table_data: dict[str, list[str]],
    summary_table: pd.DataFrame,
    case_row_ranges: list[tuple[int, int, int]],
) -> str:
    custom_css = """
    <style>
        .scrollable-table-container {
            background-color: #FFFFFF;
            width: 100%; 
            overflow-x: auto;
            overflow-y: hidden;
            white-space: nowrap;
            border: 1px solid #ccc;
        }
        table {
            border-collapse: collapse;
            table-layout: auto;
            width: auto;
        }
        td, th {
            padding: 10px;
            vertical-align: top;
            text-align: center;
        }

        /* Classes for alternating case groups */
        .case-group-even td {
            background-color: #FFFFFF !important;
        }
        .case-group-odd td {
            background-color: #F8F8F8 !important;
        }
    </style>
    """

    def get_case_index_for_row(row_idx):
        for start, end, c_idx in case_row_ranges:
            if start <= row_idx <= end:
                return c_idx
        return 0

    html_rows = []
    num_rows = len(next(iter(table_data.values())))
    for i in range(num_rows):
        case_idx = get_case_index_for_row(i)
        row_class = "case-group-even" if (case_idx % 2 == 0) else "case-group-odd"
        row_html = f"<tr class='{row_class}'>"
        for yuma_version in summary_table.columns:
            cell_content = summary_table[yuma_version][i]
            row_html += f"<td>{cell_content}</td>"
        row_html += "</tr>"
        html_rows.append(row_html)

    html_table = f"""
    <div class="scrollable-table-container">
        <table>
            <thead>
                <tr>{"".join(f"<th>{col}</th>" for col in summary_table.columns)}</tr>
            </thead>
            <tbody>
                {"".join(html_rows)}
            </tbody>
        </table>
    </div>
    """
    return custom_css + html_table


def generate_total_dividends_table(
    cases: list[BaseCase],
    yuma_versions: list[tuple[str, YumaParams]],
    simulation_hyperparameters: SimulationHyperparameters,
    is_metagraph: bool = False,
) -> pd.DataFrame:
    """
    Generates a DataFrame of total dividends for validator names
    across multiple Yuma versions.

    If is_metagraph=True, it will not apply standardization (i.e.,
    use the names in case.validators directly).
    """

    all_column_names = set()
    rows = []

    for case in cases:
        # Decide how to name the validators
        if is_metagraph:
            # Use the validators as-is (e.g., hotkeys from MetagraphCase)
            final_validator_names = case.validators
        else:
            # Apply your original "Validator A/B/C" scheme
            final_validator_names = [
                f"Validator {chr(ord('A') + i)}" for i in range(len(case.validators))
            ]

        # Create a mapping from original name -> final display name
        validator_mapping = dict(zip(case.validators, final_validator_names))

        # Build the row for this case
        row = {"Case": case.name}

        # Run each Yuma version simulation
        for yuma_version, yuma_params in yuma_versions:
            yuma_config = YumaConfig(
                simulation=simulation_hyperparameters,
                yuma_params=yuma_params,
            )

            # Run the simulation
            dividends_per_validator, _, _ = run_simulation(
                case=case,
                yuma_version=yuma_version,
                yuma_config=yuma_config,
            )

            total_dividends, _ = _calculate_total_dividends(
                validators=case.validators,
                dividends_per_validator=dividends_per_validator,
                base_validator=case.base_validator,
                num_epochs=case.num_epochs,
            )

            # Map original validator names to our final (display) names
            final_dividends = {
                validator_mapping[original_val]: total_dividends.get(original_val, 0.0)
                for original_val in case.validators
            }

            # Store results in the row
            for val_name in final_validator_names:
                column_name = f"{val_name} - {yuma_version}"
                row[column_name] = final_dividends.get(val_name, 0.0)
                all_column_names.add(column_name)

        rows.append(row)

    # Build the DataFrame
    df = pd.DataFrame(rows)

    columns = ["Case"]
    for yuma_version, _ in yuma_versions:
        version_columns = sorted(
            col for col in all_column_names if col.endswith(f"- {yuma_version}")
        )
        columns.extend(version_columns)

    # Reindex DataFrame so all columns appear. Fill missing with 0.0
    df = df.reindex(columns=columns, fill_value=0.0)

    return df
