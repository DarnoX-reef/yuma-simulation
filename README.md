# Yuma Consensus Simulation Package
&nbsp;[![Continuous Integration](https://github.com/DarnoX-reef/yuma-simulation/workflows/Continuous%20Integration/badge.svg)](https://github.com/DarnoX-reef/yuma-simulation/actions?query=workflow%3A%22Continuous+Integration%22)&nbsp;[![License](https://img.shields.io/pypi/l/yuma_simulation.svg?label=License)](https://pypi.python.org/pypi/yuma_simulation)&nbsp;[![python versions](https://img.shields.io/pypi/pyversions/yuma_simulation.svg?label=python%20versions)](https://pypi.python.org/pypi/yuma_simulation)&nbsp;[![PyPI version](https://img.shields.io/pypi/v/yuma_simulation.svg?label=PyPI%20version)](https://pypi.python.org/pypi/yuma_simulation)

## Overview
This package provides a suite of tools and simulation frameworks for exploring Bittensor Yuma Consensus mechanisms. Developed in Python, it supports multiple versions of Yuma simulations and includes features for both synthetic and real-world scenarios.

Simulations can be executed using predefined synthetic cases, designed to analyze and address challenges in various Yuma iterations. Alternatively, simulations can leverage archived real metagraph data from selected subnets, enabling realistic case studies. The output data can be visualized through interactive .html charts or exported as raw .csv files for further analysis.

In addition to core simulations, the package includes specialized scripts to generate data for both synthetic and real-world cases. A notable feature of the real-case simulations is the "shifted validator" mode, which introduces delayed weight commits by shifting the weight state of a specified validator to the subsequent epoch.

## Usage

> [!IMPORTANT]
> This package uses [ApiVer](#versioning), make sure to import `yuma_simulation.v1`.


## Versioning

This package uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
TL;DR you are safe to use [compatible release version specifier](https://packaging.python.org/en/latest/specifications/version-specifiers/#compatible-release) `~=MAJOR.MINOR` in your `pyproject.toml` or `requirements.txt`.

Additionally, this package uses [ApiVer](https://www.youtube.com/watch?v=FgcoAKchPjk) to further reduce the risk of breaking changes.
This means, the public API of this package is explicitly versioned, e.g. `yuma_simulation.v1`, and will not change in a backwards-incompatible way even when `yuma_simulation.v2` is released.

Internal packages, i.e. prefixed by `yuma_simulation._` do not share these guarantees and may change in a backwards-incompatible way at any time even in patch releases.


## Development


Pre-requisites:
- [pdm](https://pdm.fming.dev/)
- [nox](https://nox.thea.codes/en/stable/)
- [docker](https://www.docker.com/) and [docker compose plugin](https://docs.docker.com/compose/)


Ideally, you should run `nox -t format lint` before every commit to ensure that the code is properly formatted and linted.
Before submitting a PR, make sure that tests pass as well, you can do so using:
```
nox -t check # equivalent to `nox -t format lint test`
```

If you wish to install dependencies into `.venv` so your IDE can pick them up, you can do so using:
```
pdm install --dev
```

### Release process

Run `nox -s make_release -- X.Y.Z` where `X.Y.Z` is the version you're releasing and follow the printed instructions.

### Using the archived metagraph based cases
There are scripts tailored for generation of dividends charts and the total dividends data. The example usage:

python ./scripts/archived_metagraph_simulation.py \
  --subnet-id 21 \
  --bond-penalties 1.0, 0.99 \
  --epochs 40 \
  --tempo 360 \
  --start-block-offset 14400 \
  --shift-validator-id 0 \
  --draggable-table \
  --download-new-metagraph \
  --introduce-shift

This example will:
Simulate on subnet ID 21.
Use bond penalties 1.0 and 0.99.
Run for 40 epochs with a tempo of 360.
Offset the start block by 14400 blocks.
Shift validator ID 0 weights back by one epoch.
Enable draggable html table generation.
Force downloading a new metagraph.
Introduce a shift of the chosen validator weights in the simulation.

It is possible to run generation of multiple subnet data, the metagraph_subnets_config.json located at the root of the project is the configuration file used for that purpose. Flags applicable to individual subnet data generation are then defined in the configuration, while any other 'global' flags like '--download-new-metagraph' are required to provide as arguments for the script. The example usage:

python ./scripts/archived_metagraph_simulation.py \
  --use-json-config \
  --introduce-shift \
  --download-new-metagraph \
  --draggable-table

This example will:
Simulate on all the provided subnets in the json configuration file.
Use the bond penalties, tempo and other parameters as configured in the json config.
Introduce a shift of the chosen validator weights in the simulation.
Force downloading a new metagraph for each subnet simulation.
Enable draggable html table generation.
