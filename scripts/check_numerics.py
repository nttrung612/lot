"""Run a fast K <= 16 numerical diagnostic without launching experiments."""

from __future__ import annotations

import json

import numpy as np

from lot_experiments.backups import heat_backup
from lot_experiments.cli import configured_parser, resolve_cli
from lot_experiments.graphs import graph_from_config
from lot_experiments.heat_local import local_truncated_heat_column
from lot_experiments.kernels import (
    exact_heat_kernel,
    heat_approximation_certificate,
    poisson_tail,
    truncated_heat_kernel,
    uniformized_random_walk,
)


def main() -> None:
    parser = configured_parser("Validate the M0-M1 heat and backup formulas")
    _, config = resolve_cli(parser)
    diffusion_time = float(config["heat"]["diffusion_time"])
    radius = int(config["heat"]["radius"])
    T0 = float(config["backup"]["temperature"])
    graph_configs = config.get("validation_graphs", [config["graph"]])
    graph_summaries = []
    for graph_config in graph_configs:
        graph = graph_from_config(graph_config)
        if graph.K > 16:
            raise ValueError("the numerical diagnostic is intentionally limited to K <= 16")
        exact = exact_heat_kernel(graph.laplacian, diffusion_time)
        truncated = truncated_heat_kernel(graph.laplacian, diffusion_time, radius)
        local = np.column_stack(
            [
                local_truncated_heat_column(graph, diffusion_time, radius, anchor)
                for anchor in range(graph.K)
            ]
        )
        _, nu_u = uniformized_random_walk(graph.laplacian)
        beta_r = poisson_tail(nu_u * diffusion_time, radius)
        certificate = heat_approximation_certificate(
            exact, truncated, beta_r, tolerance=2e-13
        )
        np.testing.assert_allclose(local, truncated, atol=5e-14, rtol=0.0)
        assert certificate.holds
        support_is_local = all(
            set(np.flatnonzero(local[:, anchor])) <= graph.ball(anchor, radius)
            for anchor in range(graph.K)
        )
        assert support_is_local
        q = np.linspace(-0.75, 0.75, graph.K)
        mu = np.full(graph.K, 1.0 / graph.K)
        backup = heat_backup(q, exact, mu, T0)
        shifted = heat_backup(q + 2.5, exact, mu, T0)
        np.testing.assert_allclose(shifted.value, backup.value + 2.5, atol=2e-14)
        np.testing.assert_allclose(shifted.policy, backup.policy, atol=2e-14)
        graph_summaries.append(
            {
                "K": graph.K,
                "beta_r": beta_r,
                "exact_column_sum_error": float(
                    np.max(np.abs(exact.sum(axis=0) - 1.0))
                ),
                "family": graph.family,
                "l1_bound": certificate.l1_bound,
                "local_dense_max_error": float(np.max(np.abs(local - truncated))),
                "max_column_l1_error": certificate.max_column_l1_error,
                "policy_sum_error": float(abs(backup.policy.sum() - 1.0)),
                "support_is_local": support_is_local,
            }
        )
    summary = {"graphs": graph_summaries, "status": "ok"}
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
