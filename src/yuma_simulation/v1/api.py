import pandas as pd
import torch
from IPython.display import HTML
from yuma_simulation._internal.logger_setup import main_logger as logger

from yuma_simulation._internal.cases import BaseCase
from yuma_simulation._internal.charts_utils import (
    _plot_bonds,
    _plot_dividends,
    _plot_incentives,
    _plot_validator_server_weights,
)
from yuma_simulation._internal.simulation_utils import (
    _generate_draggable_html_table,
    _generate_ipynb_table,
    run_simulation,
)
from yuma_simulation._internal.yumas import (
    SimulationHyperparameters,
    YumaConfig,
    YumaParams,
    YumaSimulationNames,
)

from yuma_simulation._internal.cases import MetagraphCase


def generate_chart_table(
    cases: list[BaseCase],
    yuma_versions: list[tuple[str, YumaParams]],
    yuma_hyperparameters: SimulationHyperparameters,
    draggable_table: bool = False,
    chart_types: list[str] = None,
    highlight_validator: str = None,
) -> HTML:
    table_data: dict[str, list[str]] = {
        yuma_version: [] for yuma_version, _ in yuma_versions
    }

    def process_chart(
        table_data: dict[str, list[str]], chart_base64_dict: dict[str, str]
    ) -> None:
        for yuma_version, chart_base64 in chart_base64_dict.items():
            table_data[yuma_version].append(chart_base64)

    case_row_ranges = []
    current_row_count = 0

    for idx, case in enumerate(cases):
        current_chart_types = chart_types or (
            ["weights", "dividends", "bonds", "normalized_bonds", "incentives"]
            if idx in [9, 10]
            else ["weights", "dividends", "bonds", "normalized_bonds"]
        )

        case_start = current_row_count
        for chart_type in current_chart_types:
            chart_base64_dict: dict[str, str] = {}
            for yuma_version, yuma_params in yuma_versions:
                yuma_config = YumaConfig(
                    simulation=yuma_hyperparameters, yuma_params=yuma_params
                )
                yuma_names = YumaSimulationNames()
                full_case_name = f"{case.name} - {yuma_version}"
                if yuma_version in [
                    yuma_names.YUMA,
                    yuma_names.YUMA_LIQUID,
                    yuma_names.YUMA2,
                ]:
                    full_case_name = (
                        f"{full_case_name} - beta={yuma_config.bond_penalty}"
                    )
                elif yuma_version == yuma_names.YUMA4_LIQUID:
                    full_case_name = f"{full_case_name} [{yuma_config.alpha_low}, {yuma_config.alpha_high}]"

                (
                    dividends_per_validator,
                    bonds_per_epoch,
                    server_incentives_per_epoch,
                ) = run_simulation(
                    case=case,
                    yuma_version=yuma_version,
                    yuma_config=yuma_config,
                )

                if chart_type == "weights":
                    chart_base64 = _plot_validator_server_weights(
                        validators=case.validators,
                        weights_epochs=case.weights_epochs,
                        servers=case.servers,
                        num_epochs=case.num_epochs,
                        case_name=full_case_name,
                        to_base64=True,
                    )
                elif chart_type == "dividends":
                    chart_base64 = _plot_dividends(
                        num_epochs=case.num_epochs,
                        validators=case.validators,
                        dividends_per_validator=dividends_per_validator,
                        case=full_case_name,
                        base_validator=case.base_validator,
                        to_base64=True,
                        highlight_validator=highlight_validator,
                    )
                elif chart_type == "bonds":
                    chart_base64 = _plot_bonds(
                        num_epochs=case.num_epochs,
                        validators=case.validators,
                        servers=case.servers,
                        bonds_per_epoch=bonds_per_epoch,
                        case_name=full_case_name,
                        to_base64=True,
                    )
                elif chart_type == "normalized_bonds":
                    chart_base64 = _plot_bonds(
                        num_epochs=case.num_epochs,
                        validators=case.validators,
                        servers=case.servers,
                        bonds_per_epoch=bonds_per_epoch,
                        case_name=full_case_name,
                        to_base64=True,
                        normalize=True,
                    )
                elif chart_type == "incentives":
                    chart_base64 = _plot_incentives(
                        servers=case.servers,
                        server_incentives_per_epoch=server_incentives_per_epoch,
                        num_epochs=case.num_epochs,
                        case_name=full_case_name,
                        to_base64=True,
                    )
                else:
                    raise ValueError("Invalid chart type.")

                chart_base64_dict[yuma_version] = chart_base64

            process_chart(table_data, chart_base64_dict)
            current_row_count += 1

        case_end = current_row_count - 1
        case_row_ranges.append((case_start, case_end, idx))

    summary_table = pd.DataFrame(table_data)

    if draggable_table:
        full_html = _generate_draggable_html_table(
            table_data, summary_table, case_row_ranges
        )
    else:
        full_html = _generate_ipynb_table(table_data, summary_table, case_row_ranges)

    return HTML(full_html)


def generate_metagraph_based_dividends(
    yuma_versions: list[tuple[str, YumaParams]],
    yuma_hyperparameters: SimulationHyperparameters,
    shift_validator_id: int,
    metas: list[torch.Tensor],
    draggable_table: bool = False,
    introduce_shift: bool = False,
    highlight_validator: str = None,
) -> HTML:
    if not metas:
        logger.error("No metagraphs loaded. Nothing to be generated")
        return
    logger.debug(f"Loaded {len(metas)} metagraphs.")

    try:
        logger.info("Creating MetagraphCase.")
        case = MetagraphCase(
            shift_validator_id=shift_validator_id,
            name="Simulation Example",
            metas=metas,
            num_epochs=len(metas),
            introduce_shift=introduce_shift,
        )
        logger.debug(f"MetagraphCase created successfully: {case.name}")
    except Exception:
        logger.error("Error while creating MetagraphCase.", exc_info=True)
        return
    logger.debug(f"Created MetagraphCase: {case.name}")

    try:
        logger.info("Generating chart table.")
        chart_table = generate_chart_table(
            [case],
            yuma_versions,
            yuma_hyperparameters,
            draggable_table=draggable_table,
            chart_types=["dividends"],
            highlight_validator=highlight_validator,
        )
    except Exception as e:
        logger.error(f"error generating the chart table {e}")
        return

    return chart_table
