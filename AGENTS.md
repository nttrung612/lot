# AGENTS.md

## Project mission

This repository implements experiments for the paper **What Action Geometry Buys: A Design Space for Optimal-Transport Bellman Backups**. The code must test the paper's claims faithfully, produce reproducible paper-ready artifacts, and keep computational comparisons scientifically fair.

Read `EXPERIMENT_PLAN.md` before making architectural or experimental changes. Treat it as the implementation specification. If code and the plan disagree, stop and resolve the scientific meaning before extending the code.

## Work order

Implement milestones in this order unless the user asks otherwise:

1. Numerical kernels and tests.
2. Kernel design map.
3. Sharp pruning and heat truncation.
4. Synthetic ring-control planning.
5. Reward-free geometry learning.
6. Pendulum-v1.
7. Effective-resistance visualization.

Keep each milestone runnable before starting the next one. Prefer a small complete vertical slice over partially implementing several experiments.

## Core correctness rules

1. Preserve target identity. Every result row must include a `target` field.
2. `FullExactHeat` is the reference for exact-heat value and policy errors.
3. Never present MaxEnt, hard max, diffusion-Gibbs LOT, or a permuted graph as estimators of the exact-heat target.
4. Keep these objects distinct everywhere:
   - `P`: MDP transition kernel over states.
   - `P_G`: random-walk kernel over actions.
   - `H_t`: exact heat kernel over actions.
   - `H_tr`: normalized finite-walk heat kernel over actions.
5. A local method must build columns lazily from adjacency access. Do not silently form the dense heat matrix.
6. Report heat approximation error relative to exact heat, rather than only error relative to the truncated problem.
7. Keep true downstream dynamics fixed in the geometry-learning experiment. That experiment measures geometry error, not model error.
8. For Pendulum action-grid refinement, distinguish index-scale heat from resolution-consistent physical heat.
9. Do not claim policy-return improvement, simple regret, or best-action identification from root-value experiments.
10. Do not claim that empirical randomized baselines are lower-bounded by deterministic pruning results.

## Mathematical conventions

- Candidate action is indexed by `i`; anchor action is indexed by `j`.
- Kernel columns are candidate distributions conditioned on anchors: `w[i, j]`.
- Columns, not rows, sum to one.
- Use `T0 = tau * lambda_` in code. Do not shadow Python's `lambda` keyword.
- Use stable log-sum-exp implementations.
- Pruned backup weights are not renormalized when computing the signed pruning error.
- Normalized retained weights may be used for sampling, with the removed mass restored through the correct offset.
- The truncated heat kernel is the normalized Poisson head. It is not exact heat cropped to a graph ball.
- `beta_r` is the omitted Poisson tail; it is distinct from an exact-kernel mass budget `alpha`.
- The action graph is separate from the environment transition graph.

## Required numerical checks

Before trusting an experiment, verify:

- Heat columns are nonnegative and column-stochastic.
- Truncated heat support lies inside the correct graph ball.
- The Poisson-tail L1 certificate holds on small graphs.
- Translation equivariance of every backup.
- Returned policies are nonnegative and sum to one.
- The zero-cost backup agrees with prior-weighted MaxEnt.
- Dense and local implementations agree when the local radius covers the graph.
- The sharp pruning construction attains the closed-form worst-case error.
- Target labels prevent invalid cross-target error plots.

Tests should use small deterministic cases and tight tolerances. Do not add broad slow tests to the default unit-test command.

## Code organization

- Put reusable logic under `src/lot_experiments/`.
- Keep scripts thin; scripts load configuration and call library functions.
- Keep plotting separate from simulation.
- Do not make plotting functions rerun experiments.
- Store raw run-level data under `outputs/raw/`.
- Store aggregated tables under `outputs/summaries/`.
- Store publication figures under `outputs/figures/`.
- Never edit raw result files by hand.
- Include the complete resolved configuration in every output.

## Dependencies and commands

Use Python 3.11+ and `uv`.

Expected commands:

```bash
uv sync
uv run pytest -q
uv run python scripts/run_kernel_map.py --config configs/kernel_map.yaml
uv run python scripts/run_pruning.py --config configs/pruning.yaml
uv run python scripts/run_ring_planning.py --config configs/ring_planning.yaml
uv run python scripts/run_geometry_learning.py --config configs/geometry_learning.yaml
uv run python scripts/run_pendulum.py --config configs/pendulum.yaml
uv run python scripts/run_resistance.py --config configs/resistance.yaml
uv run python scripts/make_all_figures.py
```

If the CLI differs, update both this file and the README in the same change.

Prefer NumPy, SciPy, pandas, PyYAML, Matplotlib, Seaborn, Gymnasium, psutil, and pytest. Avoid adding large frameworks without a concrete need.

## Reproducibility

- Use `numpy.random.Generator`; do not rely on module-global random state.
- Derive component seeds deterministically from the run seed.
- Use paired seeds across comparable stochastic methods.
- Save package versions, git commit, hardware, thread count, and resolved configuration.
- Write outputs atomically so interrupted jobs do not leave valid-looking partial files.
- Resume by stable run identifiers and skip only completed matching configurations.
- Record failures and timeouts explicitly.
- Default runtime benchmarks to one process and single-threaded BLAS.

## Cost accounting

All planners must use common counters for:

- Transition-oracle calls.
- Action-value evaluations.
- Unique actions touched.
- Graph-neighbor accesses.
- Geometry preprocessing time.
- Online execution time.
- Peak memory when measured.

Do not use wall-clock time as the only efficiency metric. Report cached and uncached costs explicitly. Do not charge one method for preprocessing while silently amortizing another method's preprocessing.

## Baseline policy

Implement same-target baselines first:

1. Full exact heat.
2. Exact-heat top-mass pruning.
3. Proposed truncated local heat.
4. Poisson endpoint Monte Carlo.
5. Uniform action Monte Carlo.
6. Equal-size random subset.

Then add different-target references:

1. Uniform MaxEnt.
2. Diffusion-distance Gibbs LOT.
3. Hard max.
4. Permuted action graph.

Every figure legend or caption comparing different targets must say so explicitly.

## Statistics and plotting

- Use 30 paired seeds for expensive stochastic planners and 100 for inexpensive kernel studies.
- Report median and interquartile range by default.
- Use bootstrap 95% confidence intervals for headline stochastic results.
- Use log axes only when values are strictly positive and scaling is the message.
- Include identity or theoretical-reference lines where relevant.
- Use a consistent color per method across all figures.
- Use solid lines for proposed/full methods and dashed lines for randomized or ablation baselines.
- Save PDF and PNG versions of paper figures.
- Keep text legible at single-column paper width.
- Include units and target names in labels or captions.

Do not encode expected conclusions into data filters, axis limits, or omitted runs.

## Pendulum rules

- Use the official `Pendulum-v1` dynamics for the primary benchmark.
- Keep raw environment reward for behavioral reporting; use the declared normalized reward for regularized planning.
- Use shared evaluation states and episode seeds across methods.
- Validate the fitted-value reference on a finer state grid.
- Report both index-scale and physical-resolution heat.
- Do not describe index-scale `K` scaling as fixed physical action resolution.
- Report exact-heat value error before policy return.

## Performance work

Start with a correct reference implementation. Optimize only after profiling.

- Prefer sparse matrix-vector products and adjacency-list local propagation.
- Avoid forming `K x K` matrices in local methods.
- Cache columns lazily and expose cache statistics.
- Keep dense implementations as correctness references.
- When adding vectorization or caching, compare against the reference on small instances.

## Scope control

The initial deliverable is the experiment suite in `EXPERIMENT_PLAN.md`. Avoid adding neural agents, MuJoCo tasks, dashboards, distributed training, or new theoretical variants unless the user asks.

Do not modify the paper PDF. Generate results, figures, tables, and concise text summaries that can later be inserted into the paper.

## Completion standard

A milestone is complete only when:

1. Its implementation is reusable and configured outside the code.
2. Its numerical invariants have meaningful tests.
3. It writes raw results with complete metadata.
4. It produces the planned summary and figure.
5. Its README command runs from a clean checkout.
6. Its findings are stated within the scope of the measured target.

