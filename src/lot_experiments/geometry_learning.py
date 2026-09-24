"""M6 reward-free learned-geometry and reward-transfer experiment."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from lot_experiments.backups import kernel_from_cost
from lot_experiments.config import config_json
from lot_experiments.environments.ring_control import RingControlMDP
from lot_experiments.learning.transition_geometry import (
    empirical_transition_geometry,
    max_cost_error,
    population_transition_geometry,
    probe_distribution,
    raw_action_coordinate_cost,
    sample_reward_free_batch,
    theorem10_cost_certificate,
    transition_diffusion_cost,
)
from lot_experiments.planners.dense import PlanningResult, dense_value_iteration
from lot_experiments.reproducibility import derive_seed, rng_for, run_metadata
from lot_experiments.results import result_row, validate_results, write_results_atomic


PRIMARY_TARGET = "transition_diffusion_cost"
GEOMETRY_COLUMNS = (
    "experiment_part",
    "reference_target",
    "replicate",
    "samples_per_action",
    "reward_free_transition_samples",
    "amortized_transition_samples",
    "reward_goal",
    "goal_fraction",
    "anchor_reference_goal",
    "probe_distribution",
    "probe_uniform_mass",
    "geometry_id",
    "cost_max_error",
    "signature_max_l2_error",
    "laplacian_operator_error",
    "fixed_point_q_error",
    "propagation_envelope",
    "envelope_holds",
    "theoretical_cost_certificate",
    "certificate_holds",
    "downstream_transition_calls",
    "fixed_point_iterations",
    "fixed_point_residual",
    "metadata_json",
    "error_message",
)

SUMMARY_METRICS = (
    "cost_max_error",
    "signature_max_l2_error",
    "laplacian_operator_error",
    "fixed_point_q_error",
    "propagation_envelope",
    "absolute_value_error",
    "policy_l1_error",
    "geometry_preprocess_seconds",
    "online_seconds",
    "reward_free_transition_samples",
)


def _deep_defaults(destination: dict[str, Any], defaults: Mapping[str, Any]) -> None:
    for key, value in defaults.items():
        if key not in destination:
            destination[key] = copy.deepcopy(value)
        elif isinstance(value, Mapping) and isinstance(destination[key], dict):
            _deep_defaults(destination[key], value)


def resolve_geometry_learning_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve and validate the complete M6 configuration."""

    resolved = copy.deepcopy(dict(config))
    _deep_defaults(
        resolved,
        {
            "experiment": "geometry_learning",
            "seed": 0,
            "K": 16,
            "samples_per_action": [16, 32, 64, 128, 256, 512, 1024, 2048],
            "goal_fractions": [0.0, 0.2, 0.4, 0.6, 0.8],
            "paired_seeds": 30,
            "gamma": 0.95,
            "action_cost": 0.05,
            "anchor_uniform_mass": 0.05,
            "root_state_fraction": 0.5,
            "anchor_reference_fraction": 0.0,
            "probe": {
                "kind": "spike_uniform",
                "uniform_mass": 0.1,
                "center_fraction": 0.0,
            },
            "uniform_probe_ablation": {"enabled": True},
            "geometry": {"diffusion_time": 1.0, "kappa_C": 1.0},
            "regularization": {"tau": 0.1, "lambda_": 1.0},
            "reference": {"tolerance": 1e-9, "max_iterations": 5000},
            "delta": 0.05,
            "value_error_thresholds": [0.1, 0.05, 0.02],
            "bootstrap_repetitions": 2000,
            "raw_output": "outputs/raw/geometry_learning.parquet",
            "summary_output": "outputs/summaries/geometry_learning.csv",
            "table_output": "outputs/summaries/table1_geometry_learning.csv",
            "figure_png": "outputs/figures/geometry_learning.png",
        },
    )
    if resolved["experiment"] != "geometry_learning":
        raise ValueError("experiment must be 'geometry_learning'")
    K = resolved["K"]
    if isinstance(K, bool) or not isinstance(K, (int, np.integer)) or int(K) < 3:
        raise ValueError("K must be an integer at least three")
    samples = [int(value) for value in resolved["samples_per_action"]]
    if not samples or any(value < 1 for value in samples) or samples != sorted(set(samples)):
        raise ValueError("samples_per_action must be a strictly increasing positive grid")
    goals = [float(value) for value in resolved["goal_fractions"]]
    if not goals or any(not 0.0 <= value < 1.0 for value in goals):
        raise ValueError("goal_fractions must lie in [0, 1)")
    anchor_fraction = float(resolved["anchor_reference_fraction"])
    if not math.isfinite(anchor_fraction):
        raise ValueError("anchor_reference_fraction must be finite")
    if int(resolved["paired_seeds"]) < 1:
        raise ValueError("paired_seeds must be positive")
    if not 0.0 <= float(resolved["gamma"]) < 1.0:
        raise ValueError("gamma must lie in [0, 1)")
    if not 0.0 < float(resolved["delta"]) < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    if float(resolved["regularization"]["tau"]) <= 0.0 or float(
        resolved["regularization"]["lambda_"]
    ) <= 0.0:
        raise ValueError("regularization tau and lambda_ must be positive")
    if float(resolved["geometry"]["diffusion_time"]) < 0.0 or float(
        resolved["geometry"]["kappa_C"]
    ) <= 0.0:
        raise ValueError("geometry diffusion_time must be nonnegative and kappa_C positive")
    if int(resolved["bootstrap_repetitions"]) < 1:
        raise ValueError("bootstrap_repetitions must be positive")
    if any(float(value) <= 0.0 for value in resolved["value_error_thresholds"]):
        raise ValueError("value_error_thresholds must be positive")
    probe = resolved["probe"]
    center = int(round(float(probe["center_fraction"]) * int(K))) % int(K)
    probe_distribution(
        int(K),
        kind=str(probe["kind"]),
        uniform_mass=float(probe["uniform_mass"]),
        center=center,
    )
    return resolved


def _stable_run_id(parts: Sequence[Any], resolved_json: str) -> str:
    payload = json.dumps([*parts, resolved_json], separators=(",", ":"), default=str)
    return "geometry-" + hashlib.blake2b(payload.encode(), digest_size=10).hexdigest()


def _geometry_id(part: str, replicate: int, n: int, seed: int) -> str:
    payload = f"{part}:{replicate}:{n}:{seed}".encode()
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


def _solve_cost_target(
    mdp: RingControlMDP,
    costs: np.ndarray,
    *,
    gamma: float,
    tau: float,
    lambda_: float,
    method: str,
    target: str,
    tolerance: float,
    max_iterations: int,
    initial_values: np.ndarray | None = None,
    anchor_distributions: np.ndarray | None = None,
) -> PlanningResult:
    prior = np.full(mdp.K, 1.0 / mdp.K, dtype=np.float64)
    weights, offsets = kernel_from_cost(costs, prior, tau, lambda_)
    return dense_value_iteration(
        mdp,
        gamma=gamma,
        method=method,
        target=target,
        weights=weights,
        offsets=offsets,
        T0=tau * lambda_,
        tolerance=tolerance,
        max_iterations=max_iterations,
        initial_values=initial_values,
        anchor_distributions=anchor_distributions,
        circulant_kernel=False,
    )


def _blank_row(
    *,
    resolved: Mapping[str, Any],
    resolved_json: str,
    metadata_json: str,
    run_id: str,
    seed: int,
    method: str,
    target: str,
    part: str,
    replicate: int,
    n: int,
    goal: int,
    goal_fraction: float,
    probe_kind: str,
    geometry_id: str,
    status: str = "complete",
) -> dict[str, Any]:
    K = int(resolved["K"])
    row = result_row(
        experiment="geometry_learning",
        run_id=run_id,
        seed=seed,
        method=method,
        target=target,
        graph_family="transition_diffusion",
        K=K,
        diffusion_time=float(resolved["geometry"]["diffusion_time"]),
        temperature=float(resolved["regularization"]["tau"])
        * float(resolved["regularization"]["lambda_"]),
        gamma=float(resolved["gamma"]),
        delta=float(resolved["delta"]),
        root_state=int(round(float(resolved["root_state_fraction"]) * K)) % K,
        status=status,
        git_commit=json.loads(metadata_json)["git_commit"],
        config_json=resolved_json,
    )
    row.update({column: np.nan for column in GEOMETRY_COLUMNS})
    row.update(
        {
            "experiment_part": part,
            "reference_target": PRIMARY_TARGET,
            "replicate": replicate,
            "samples_per_action": n,
            "reward_free_transition_samples": K * n,
            "amortized_transition_samples": K * n / len(resolved["goal_fractions"]),
            "reward_goal": goal,
            "goal_fraction": goal_fraction,
            "anchor_reference_goal": int(
                round(float(resolved["anchor_reference_fraction"]) * K)
            )
            % K,
            "probe_distribution": probe_kind,
            "probe_uniform_mass": float(resolved["probe"]["uniform_mass"]),
            "geometry_id": geometry_id,
            "metadata_json": metadata_json,
            "error_message": "",
        }
    )
    return row


def _fill_planning_metrics(
    row: dict[str, Any],
    result: PlanningResult,
    oracle: PlanningResult,
    *,
    cost_error: float,
    tau: float,
    gamma: float,
) -> None:
    root = int(row["root_state"])
    q_error = float(np.max(np.abs(result.q_values - oracle.q_values)))
    envelope = gamma * tau * cost_error / (1.0 - gamma)
    numerical_slack = 10.0 * max(result.bellman_residual, oracle.bellman_residual, 1e-12)
    row.update(
        {
            "value_estimate": result.root_value(root),
            "reference_value": oracle.root_value(root),
            "absolute_value_error": abs(result.root_value(root) - oracle.root_value(root)),
            "policy_l1_error": float(np.abs(result.policy[root] - oracle.policy[root]).sum()),
            "cost_max_error": cost_error,
            "fixed_point_q_error": q_error,
            "propagation_envelope": envelope,
            "envelope_holds": q_error <= envelope + numerical_slack,
            "downstream_transition_calls": result.counters.transition_calls,
            "fixed_point_iterations": result.iterations,
            "fixed_point_residual": result.bellman_residual,
            "action_evaluations": result.counters.action_evaluations,
            "unique_actions_touched": result.counters.unique_actions_touched,
            "online_seconds": result.counters.online_seconds,
            "peak_memory_mb": result.counters.peak_memory_mb,
        }
    )
    if row["target"] == PRIMARY_TARGET:
        row["statistical_error"] = row["absolute_value_error"]


def _reference_rows(
    *,
    resolved: Mapping[str, Any],
    resolved_json: str,
    metadata_json: str,
    population_cost: np.ndarray,
    raw_cost: np.ndarray,
    population_preprocess_seconds: float,
    anchor_distributions: np.ndarray,
    completed: set[str],
) -> tuple[list[dict[str, Any]], dict[int, PlanningResult]]:
    K = int(resolved["K"])
    gamma = float(resolved["gamma"])
    tau = float(resolved["regularization"]["tau"])
    lambda_ = float(resolved["regularization"]["lambda_"])
    tolerance = float(resolved["reference"]["tolerance"])
    max_iterations = int(resolved["reference"]["max_iterations"])
    root_rows: list[dict[str, Any]] = []
    oracles: dict[int, PlanningResult] = {}
    zero_cost = np.zeros_like(population_cost)
    for fraction in map(float, resolved["goal_fractions"]):
        goal = int(round(fraction * K)) % K
        mdp = RingControlMDP(
            K,
            goal=goal,
            action_cost=float(resolved["action_cost"]),
            anchor_uniform_mass=float(resolved["anchor_uniform_mass"]),
        )
        oracle = _solve_cost_target(
            mdp,
            population_cost,
            gamma=gamma,
            tau=tau,
            lambda_=lambda_,
            method="oracle_population_cost",
            target=PRIMARY_TARGET,
            tolerance=tolerance,
            max_iterations=max_iterations,
            anchor_distributions=anchor_distributions,
        )
        if not oracle.converged:
            raise RuntimeError(f"oracle population solve failed for goal {goal}")
        oracles[goal] = oracle
        specs = (
            ("oracle_population_cost", PRIMARY_TARGET, population_cost, oracle),
            ("raw_action_coordinate_cost", "raw_action_coordinate_cost", raw_cost, None),
            ("uniform_maxent", "maxent", zero_cost, None),
        )
        for method, target, costs, known in specs:
            run_id = _stable_run_id(("reference", method, goal), resolved_json)
            if run_id in completed:
                continue
            result = known or _solve_cost_target(
                mdp,
                costs,
                gamma=gamma,
                tau=tau,
                lambda_=lambda_,
                method=method,
                target=target,
                tolerance=tolerance,
                max_iterations=max_iterations,
                initial_values=oracle.values,
                anchor_distributions=anchor_distributions,
            )
            row = _blank_row(
                resolved=resolved,
                resolved_json=resolved_json,
                metadata_json=metadata_json,
                run_id=run_id,
                seed=-1,
                method=method,
                target=target,
                part="reference",
                replicate=-1,
                n=0,
                goal=goal,
                goal_fraction=fraction,
                probe_kind=str(resolved["probe"]["kind"]),
                geometry_id="population",
            )
            row["reward_free_transition_samples"] = 0
            row["amortized_transition_samples"] = 0.0
            error = max_cost_error(costs, population_cost)
            _fill_planning_metrics(row, result, oracle, cost_error=error, tau=tau, gamma=gamma)
            row["transition_calls"] = 0
            if method == "oracle_population_cost":
                row["geometry_preprocess_seconds"] = population_preprocess_seconds
            root_rows.append(row)
    return root_rows, oracles


def _atomic_csv(frame: pd.DataFrame, destination: str | Path) -> Path:
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def run_geometry_learning(
    config: Mapping[str, Any], *, repository: str | Path = "."
) -> pd.DataFrame:
    """Run or resume the paired reward-free geometry experiment."""

    resolved = resolve_geometry_learning_config(config)
    resolved_json = config_json(resolved)
    metadata_json = json.dumps(
        run_metadata(resolved, repository), sort_keys=True, separators=(",", ":")
    )
    output = Path(resolved["raw_output"])
    if output.exists():
        existing = validate_results(pd.read_parquet(output))
        configurations = set(existing["config_json"].astype(str))
        if configurations and configurations != {resolved_json}:
            raise ValueError(
                "raw_output contains a different resolved configuration; choose a distinct output path"
            )
        rows = existing.to_dict(orient="records")
    else:
        rows = []
    for row in rows:
        row.setdefault("reference_target", PRIMARY_TARGET)
        if str(row.get("method")) != "learned_transition_geometry":
            row["theoretical_cost_certificate"] = np.nan
            row["certificate_holds"] = np.nan
        if str(row.get("target")) == PRIMARY_TARGET:
            row["statistical_error"] = row.get("absolute_value_error", np.nan)
    completed = {
        str(row["run_id"]) for row in rows if str(row["status"]) == "complete"
    }

    K = int(resolved["K"])
    anchor_goal = int(round(float(resolved["anchor_reference_fraction"]) * K)) % K
    base_mdp = RingControlMDP(
        K,
        goal=anchor_goal,
        action_cost=float(resolved["action_cost"]),
        anchor_uniform_mass=float(resolved["anchor_uniform_mass"]),
    )
    fixed_anchor_distributions = base_mdp.anchor_distributions
    probe = resolved["probe"]
    center = int(round(float(probe["center_fraction"]) * K)) % K
    nu = probe_distribution(
        K,
        kind=str(probe["kind"]),
        uniform_mass=float(probe["uniform_mass"]),
        center=center,
    )
    diffusion_time = float(resolved["geometry"]["diffusion_time"])
    kappa_C = float(resolved["geometry"]["kappa_C"])
    population_start = perf_counter()
    population = population_transition_geometry(base_mdp, nu)
    population_cost = transition_diffusion_cost(population, diffusion_time, kappa_C)
    population_seconds = perf_counter() - population_start
    if np.max(population_cost, initial=0.0) <= 1e-14:
        raise ValueError(
            "the primary probe distribution produces degenerate zero action geometry; "
            "use a reward-independent nonuniform full-support probe"
        )
    raw_cost = raw_action_coordinate_cost(K, kappa_C)

    reference_rows, oracles = _reference_rows(
        resolved=resolved,
        resolved_json=resolved_json,
        metadata_json=metadata_json,
        population_cost=population_cost,
        raw_cost=raw_cost,
        population_preprocess_seconds=population_seconds,
        anchor_distributions=fixed_anchor_distributions,
        completed=completed,
    )
    rows.extend(reference_rows)
    completed.update(str(row["run_id"]) for row in reference_rows)
    if reference_rows:
        write_results_atomic(rows, output)

    gamma = float(resolved["gamma"])
    tau = float(resolved["regularization"]["tau"])
    lambda_ = float(resolved["regularization"]["lambda_"])
    delta = float(resolved["delta"])
    tolerance = float(resolved["reference"]["tolerance"])
    max_iterations = int(resolved["reference"]["max_iterations"])
    samples_grid = list(map(int, resolved["samples_per_action"]))
    max_n = max(samples_grid)
    goal_specs = [
        (float(fraction), int(round(float(fraction) * K)) % K)
        for fraction in resolved["goal_fractions"]
    ]

    for replicate in range(int(resolved["paired_seeds"])):
        run_seed = derive_seed(int(resolved["seed"]), f"geometry:{replicate}") % (2**63 - 1)
        batch = sample_reward_free_batch(base_mdp, nu, max_n, rng_for(run_seed, "primary_probe"))
        for n in samples_grid:
            geometry_id = _geometry_id("primary", replicate, n, run_seed)
            geometry_start = perf_counter()
            empirical = empirical_transition_geometry(batch, n)
            learned_cost = transition_diffusion_cost(empirical, diffusion_time, kappa_C)
            no_diffusion_cost = transition_diffusion_cost(empirical, 0.0, kappa_C)
            permutation = rng_for(run_seed, f"permutation:{n}").permutation(K)
            permuted_cost = learned_cost[np.ix_(permutation, permutation)]
            geometry_seconds = perf_counter() - geometry_start
            cost_certificate = theorem10_cost_certificate(
                state_count=K,
                action_count=K,
                samples_per_action=n,
                delta=delta,
                diffusion_time=diffusion_time,
                kappa_C=kappa_C,
            )
            signature_error = float(
                np.max(
                    np.linalg.norm(empirical.signatures - population.signatures, axis=1),
                    initial=0.0,
                )
            )
            laplacian_error = float(
                np.linalg.norm(empirical.laplacian - population.laplacian, ord=2)
            )
            method_costs = (
                ("learned_transition_geometry", PRIMARY_TARGET, learned_cost),
                (
                    "learned_no_diffusion",
                    "transition_no_diffusion_cost",
                    no_diffusion_cost,
                ),
                (
                    "permuted_learned_geometry",
                    "permuted_transition_diffusion_cost",
                    permuted_cost,
                ),
            )
            for fraction, goal in goal_specs:
                mdp = RingControlMDP(
                    K,
                    goal=goal,
                    action_cost=float(resolved["action_cost"]),
                    anchor_uniform_mass=float(resolved["anchor_uniform_mass"]),
                )
                oracle = oracles[goal]
                for method, target, costs in method_costs:
                    run_id = _stable_run_id(
                        ("primary", replicate, n, goal, method), resolved_json
                    )
                    if run_id in completed:
                        continue
                    row = _blank_row(
                        resolved=resolved,
                        resolved_json=resolved_json,
                        metadata_json=metadata_json,
                        run_id=run_id,
                        seed=run_seed,
                        method=method,
                        target=target,
                        part="primary",
                        replicate=replicate,
                        n=n,
                        goal=goal,
                        goal_fraction=fraction,
                        probe_kind=str(probe["kind"]),
                        geometry_id=geometry_id,
                    )
                    try:
                        result = _solve_cost_target(
                            mdp,
                            costs,
                            gamma=gamma,
                            tau=tau,
                            lambda_=lambda_,
                            method=method,
                            target=target,
                            tolerance=tolerance,
                            max_iterations=max_iterations,
                            initial_values=oracle.values,
                            anchor_distributions=fixed_anchor_distributions,
                        )
                        if not result.converged:
                            raise RuntimeError(
                                f"value iteration did not converge; residual={result.bellman_residual:g}"
                            )
                        cost_error = max_cost_error(costs, population_cost)
                        _fill_planning_metrics(
                            row,
                            result,
                            oracle,
                            cost_error=cost_error,
                            tau=tau,
                            gamma=gamma,
                        )
                        row.update(
                            {
                                "transition_calls": K * n,
                                "geometry_preprocess_seconds": geometry_seconds,
                                "signature_max_l2_error": signature_error,
                                "laplacian_operator_error": laplacian_error,
                            }
                        )
                        if method == "learned_transition_geometry":
                            row.update(
                                {
                                    "theoretical_cost_certificate": cost_certificate,
                                    "certificate_holds": cost_error
                                    <= cost_certificate + 1e-12,
                                }
                            )
                    except Exception as error:
                        row["status"] = "failed"
                        row["error_message"] = f"{type(error).__name__}: {error}"
                    rows.append(row)
                    completed.add(run_id)
            write_results_atomic(rows, output)

        if bool(resolved["uniform_probe_ablation"]["enabled"]):
            uniform_nu = probe_distribution(K, kind="uniform")
            uniform_population = population_transition_geometry(base_mdp, uniform_nu)
            uniform_population_cost = transition_diffusion_cost(
                uniform_population, diffusion_time, kappa_C
            )
            uniform_batch = sample_reward_free_batch(
                base_mdp, uniform_nu, max_n, rng_for(run_seed, "uniform_probe")
            )
            for n in samples_grid:
                run_id = _stable_run_id(
                    ("uniform_probe_ablation", replicate, n), resolved_json
                )
                if run_id in completed:
                    continue
                start = perf_counter()
                empirical = empirical_transition_geometry(uniform_batch, n)
                costs = transition_diffusion_cost(empirical, diffusion_time, kappa_C)
                seconds = perf_counter() - start
                error = max_cost_error(costs, uniform_population_cost)
                certificate = theorem10_cost_certificate(
                    state_count=K,
                    action_count=K,
                    samples_per_action=n,
                    delta=delta,
                    diffusion_time=diffusion_time,
                    kappa_C=kappa_C,
                )
                row = _blank_row(
                    resolved=resolved,
                    resolved_json=resolved_json,
                    metadata_json=metadata_json,
                    run_id=run_id,
                    seed=run_seed,
                    method="learned_transition_geometry",
                    target="transition_diffusion_cost_uniform_probe",
                    part="uniform_probe_ablation",
                    replicate=replicate,
                    n=n,
                    goal=-1,
                    goal_fraction=np.nan,
                    probe_kind="uniform",
                    geometry_id=_geometry_id("uniform", replicate, n, run_seed),
                )
                row.update(
                    {
                        "transition_calls": K * n,
                        "geometry_preprocess_seconds": seconds,
                        "cost_max_error": error,
                        "signature_max_l2_error": float(
                            np.max(np.linalg.norm(empirical.signatures, axis=1), initial=0.0)
                        ),
                        "laplacian_operator_error": float(
                            np.linalg.norm(
                                empirical.laplacian - uniform_population.laplacian,
                                ord=2,
                            )
                        ),
                        "theoretical_cost_certificate": certificate,
                        "certificate_holds": error <= certificate + 1e-12,
                    }
                )
                rows.append(row)
                completed.add(run_id)
            write_results_atomic(rows, output)

    frame = validate_results(pd.DataFrame(rows))
    # Also persists schema-only additions when a fully completed run is resumed.
    write_results_atomic(frame, output)
    failures = frame.loc[frame["status"] != "complete"]
    if not failures.empty:
        examples = "; ".join(failures["error_message"].astype(str).head(3))
        raise RuntimeError(f"{len(failures)} geometry-learning runs failed: {examples}")
    return frame


def _bootstrap_interval(
    values: np.ndarray, repetitions: int, rng: np.random.Generator
) -> tuple[float, float]:
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    medians = np.median(values[indices], axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def summarize_geometry_learning(
    raw: pd.DataFrame, *, resolved_config: Mapping[str, Any] | None = None
) -> pd.DataFrame:
    """Summarize M6 by method, sample count, and reward task."""

    frame = validate_results(raw)
    frame = frame.loc[
        (frame["experiment"] == "geometry_learning") & (frame["status"] == "complete")
    ].copy()
    repetitions = 2000
    if resolved_config is not None:
        resolved = resolve_geometry_learning_config(resolved_config)
        frame = frame.loc[frame["config_json"] == config_json(resolved)]
        repetitions = int(resolved["bootstrap_repetitions"])
    identifiers = [
        "experiment_part",
        "method",
        "target",
        "K",
        "samples_per_action",
        "reward_goal",
        "goal_fraction",
        "probe_distribution",
        "diffusion_time",
        "temperature",
        "gamma",
        "delta",
        "config_json",
    ]
    records: list[dict[str, Any]] = []
    for keys, group in frame.groupby(identifiers, dropna=False, sort=True):
        record = dict(zip(identifiers, keys, strict=True))
        record["n_runs"] = len(group)
        rng = np.random.default_rng(
            derive_seed(0, json.dumps(list(map(str, keys)), separators=(",", ":")))
        )
        for metric in SUMMARY_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
            record[f"{metric}_median"] = float(np.median(values)) if len(values) else np.nan
            record[f"{metric}_q25"] = float(np.quantile(values, 0.25)) if len(values) else np.nan
            record[f"{metric}_q75"] = float(np.quantile(values, 0.75)) if len(values) else np.nan
            low, high = _bootstrap_interval(values, repetitions, rng)
            record[f"{metric}_ci_low"] = low
            record[f"{metric}_ci_high"] = high
        for field in ("envelope_holds", "certificate_holds"):
            values = group[field].dropna()
            record[f"{field}_rate"] = (
                float(values.astype(bool).mean()) if len(values) else np.nan
            )
        records.append(record)
    return pd.DataFrame(records)


def geometry_threshold_table(
    raw: pd.DataFrame, thresholds: Sequence[float]
) -> pd.DataFrame:
    """Samples needed for median and worst-task Q-error thresholds."""

    frame = validate_results(raw)
    frame = frame.loc[
        (frame["experiment"] == "geometry_learning")
        & (frame["status"] == "complete")
        & (frame["experiment_part"] == "primary")
        & (frame["method"] == "learned_transition_geometry")
        & (frame["target"] == PRIMARY_TARGET)
    ].copy()
    if frame.empty:
        raise ValueError("no primary learned-geometry rows available")
    task_curves = (
        frame.groupby(["samples_per_action", "reward_goal"], as_index=False)[
            "fixed_point_q_error"
        ]
        .median()
        .sort_values("samples_per_action")
    )
    aggregate = task_curves.groupby("samples_per_action")["fixed_point_q_error"].agg(
        median="median", worst_case="max"
    )
    K = int(frame["K"].iloc[0])
    records = []
    for threshold in map(float, thresholds):
        for aggregation in ("median", "worst_case"):
            eligible = aggregate.index[aggregate[aggregation] <= threshold]
            n = int(eligible.min()) if len(eligible) else np.nan
            records.append(
                {
                    "value_error_threshold": threshold,
                    "task_aggregation": aggregation,
                    "samples_per_action": n,
                    "total_reward_free_samples": K * n if np.isfinite(n) else np.nan,
                    "target": PRIMARY_TARGET,
                }
            )
    return pd.DataFrame(records)


def write_geometry_summary_atomic(summary: pd.DataFrame, destination: str | Path) -> Path:
    return _atomic_csv(summary, destination)


def write_geometry_table_atomic(table: pd.DataFrame, destination: str | Path) -> Path:
    return _atomic_csv(table, destination)
