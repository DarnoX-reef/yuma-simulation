from bs4 import BeautifulSoup

from src.yuma_simulation._internal.cases import cases
from src.yuma_simulation._internal.yumas import (
    SimulationHyperparameters,
    YumaParams,
    YumaSimulationNames,
)
from src.yuma_simulation.v1.api import generate_chart_table


def test_generate_chart_table_with_charts():
    base_yuma_params = YumaParams()
    yumas = YumaSimulationNames()
    yuma_versions = [
        (yumas.YUMA, base_yuma_params),
    ]
    simulation_hyperparameters = SimulationHyperparameters(bond_penalty=0.99)

    chart_table = generate_chart_table(cases, yuma_versions, simulation_hyperparameters)
    soup = BeautifulSoup(chart_table.data, "html.parser")

    # Check that at least one <img> tag is present, assuming charts are embedded images
    imgs = soup.find_all("img")
    assert len(imgs) > 0, "Should contain at least one chart image"

    # Check if images are base64 encoded
    for img in imgs:
        src = img.get("src", "")
        assert src.startswith("data:image/png;base64,"), (
            "Image should be base64-encoded"
        )
