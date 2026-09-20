"""Build and validate the M7 Pendulum FullExactHeat reference solution."""

from __future__ import annotations

from lot_experiments.cli import configured_parser, resolve_cli
from lot_experiments.pendulum_reference import run_pendulum_reference


def main() -> None:
    parser = configured_parser("Build the M7 Pendulum FullExactHeat reference")
    _, config = resolve_cli(parser)
    run_pendulum_reference(config)


if __name__ == "__main__":
    main()
