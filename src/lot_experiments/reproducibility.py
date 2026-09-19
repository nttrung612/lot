"""Seed derivation and run metadata capture."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from lot_experiments.config import config_json


TRACKED_PACKAGES = (
    "lot-experiments",
    "numpy",
    "scipy",
    "pandas",
    "PyYAML",
    "matplotlib",
    "seaborn",
    "gymnasium",
    "psutil",
    "pytest",
)


def derive_seed(run_seed: int, component: str) -> int:
    """Derive a stable uint64 seed without Python's randomized hash()."""

    payload = f"{int(run_seed)}:{component}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def rng_for(run_seed: int, component: str) -> np.random.Generator:
    return np.random.default_rng(derive_seed(run_seed, component))


def git_commit(repository: str | Path = ".") -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def package_versions(packages: Sequence[str] = TRACKED_PACKAGES) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def run_metadata(config: Mapping[str, Any], repository: str | Path = ".") -> dict[str, Any]:
    thread_variables = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    return {
        "git_commit": git_commit(repository),
        "config_json": config_json(config),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "thread_environment": {name: os.environ.get(name) for name in thread_variables},
        "package_versions": package_versions(),
    }

