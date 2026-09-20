"""Build and validate the M7 Pendulum FullExactHeat reference solution."""

from __future__ import annotations

import os

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

from lot_experiments.cli import configured_parser, resolve_cli
from lot_experiments.pendulum_reference import run_pendulum_reference


def main() -> None:
    parser = configured_parser(
        "Build the converged Pendulum-v1 FullExactHeat fitted-value reference"
    )
    _, config = resolve_cli(parser)
    run_pendulum_reference(config)


if __name__ == "__main__":
    main()
