"""Shared command-line plumbing for thin scripts."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from lot_experiments.config import load_config


def configured_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a YAML setting using a dotted key (repeatable)",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def resolve_cli(parser: argparse.ArgumentParser) -> tuple[argparse.Namespace, dict[str, Any]]:
    arguments = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, str(arguments.log_level).upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return arguments, load_config(arguments.config, arguments.set)

