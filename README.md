# LOT experiments

Numerical experiments for **What Action Geometry Buys: A Design Space for
Optimal-Transport Bellman Backups**.

The repository currently implements milestones M0-M9: configuration loading,
result validation, graph construction, exact and normalized finite-walk heat
kernels, reusable Poisson-tail certificates, stable LOT backups, lazy local
heat columns, operation accounting, and deterministic validation on all planned
small graph families, the kernel design-map experiment with a Figure 1 draft,
the sharp-pruning/Poisson-truncation experiment with a Figure 2 draft, and the
stochastic ring-control environment with dense target references and baseline
planners, the end-to-end paired-seed ring study with Figure 3, and reward-free
transition-geometry learning with reward transfer, deterministic propagation
checks, a sample-threshold table, and a paper-ready figure, plus an exact
``Pendulum-v1`` discrete-torque wrapper and a development/validation-grid
FullExactHeat fitted-value reference, all planned Pendulum baselines, radius
ablations, behavioral evaluation, checkpointed execution, and Figure 4. The
appendix effective-resistance study computes the value-dependent LOT transport
co-occurrence graph and verifies policy-space curvature by balanced entropic-OT
finite differences.

## Setup and checks

Python 3.11 or newer and [`uv`](https://docs.astral.sh/uv/) are required.

```bash
uv sync
uv run pytest -q
uv run python scripts/check_numerics.py --config configs/numerics.yaml
uv run python scripts/run_kernel_map.py --config configs/kernel_map.yaml
uv run python scripts/run_pruning.py --config configs/pruning.yaml
uv run python scripts/run_ring_planning.py --config configs/ring_planning.yaml
uv run python scripts/run_geometry_learning.py --config configs/geometry_learning.yaml
uv run python scripts/run_pendulum_reference.py --config configs/pendulum_reference.yaml
uv run python scripts/run_pendulum.py --config configs/pendulum_smoke.yaml
uv run python scripts/run_pendulum.py --config configs/pendulum.yaml
uv run python scripts/run_resistance.py --config configs/resistance.yaml
uv run python scripts/make_all_figures.py
```

For a multi-core server, distribute independent ring-planning cases across
processes while keeping every worker on single-threaded BLAS:

```bash
uv run python scripts/run_ring_planning.py \
  --config configs/ring_planning.yaml \
  --set execution.workers=16 \
  --set raw_output=outputs/raw/ring_planning_parallel.parquet \
  --set summary_output=outputs/summaries/ring_planning_parallel.csv \
  --set figure_png=outputs/figures/figure3_ring_planning_parallel.png \
  --set figure_pdf=outputs/figures/figure3_ring_planning_parallel.pdf
```

Only the parent process writes checkpoints, and resumption still uses stable
run identifiers. Parallel rows are labeled with `timing_mode=concurrent`;
their accuracy and operation counts are valid, but the runtime panel is
intentionally withheld. Use `execution.workers=1` for paper-ready wall-clock
measurements.

M8 Pendulum uses the same parent-only checkpointing pattern. The full
configuration requests 40 workers; because the grid contains 30 independent
`(K, heat_scaling, temperature)` cases, at most 30 workers are active at once:

```bash
uv run python scripts/run_pendulum.py --config configs/pendulum.yaml
```

All methods for one case stay in the same worker and share its FullExactHeat
reference. Parallel Pendulum rows use `timing_mode=concurrent`, so Figure 4
withholds its runtime panel while retaining accuracy, policy, behavior, and
operation-count panels. For publishable timing, run with
`execution.workers=1` and distinct output paths.

The diagnostic is deliberately small (`K <= 16`) and does not write experiment
results. Experiment raw, summary, and figure artifacts belong in `outputs/raw/`,
`outputs/summaries/`, and `outputs/figures/`, respectively.

## Numerical conventions

- Kernel columns are candidate distributions conditioned on anchors.
- `P_G = I - L / nu_u` is an action-graph walk, never an MDP transition kernel.
- Direct heat uses the centered backup (zero anchor offsets).
- Cost-based LOT retains the prior-mass offsets from Equation (19) of the paper.
- Pruned backup weights are not renormalized in the value.
- In M2, truncated heat uses the smallest radius satisfying `beta_r <= alpha`;
  both the exact-kernel budget `alpha` and the distinct Poisson tail `beta_r`
  are stored in every truncated result row.
- M2 runtime measurements force one process and single-threaded BLAS.
- M3 fixed-set pruning keeps exact heat as the target and never renormalizes
  retained weights. Poisson truncation is labeled as a distinct truncated-heat
  target and reports kernel, backup, and fixed-point errors to FullExactHeat.
- M4 keeps the MDP kernel `P` inside `RingControlMDP` and passes the action graph
  separately. `FullExactHeat` is the dense reference; MaxEnt, hard max,
  diffusion-Gibbs, and permuted-graph solutions retain distinct target labels.
  Local finite-walk caches store sparse column supports rather than a hidden
  dense heat matrix.
- M5 freezes paired anchor/candidate samples within each empirical Bellman
  operator, runs it to a checked fixed point, and reports median/IQR plus
  bootstrap 95% confidence intervals. Coverage is explicitly empirical; the
  plug-in Monte Carlo baselines are not presented as certified estimators.
  Figure 3 keeps MaxEnt, hard max, diffusion-Gibbs, expander, and permuted-graph
  targets separate from the primary cycle exact-heat comparison.
- M6 implements the paper's reward-free signatures
  `z_a = E[e_{S'} - e_S]`, shared state Laplacian, and transition-diffusion
  cost. Learned and oracle-cost fixed points always use the same true ring MDP
  transition kernel, so their difference is geometry error, not model error.
  The primary probe distribution is fixed, reward-independent, full-support,
  and nonuniform. This is necessary because uniform `nu` is stationary under
  every translation action on the ring and would force all population
  signatures and costs to zero; the uniform-probe case is retained as an
  explicit cost-only ablation. One learned geometry is reused across every
  reward goal at a given paired seed and sample count, and the anchor priors
  are frozen to the configured reference goal rather than changing with the
  reward task.
- M7 mirrors the official deterministic ``Pendulum-v1`` equations, removes
  only the episode time limit for discounted fixed-point computation, and
  keeps raw rewards separate from rewards divided by ``16.2736044``. Its
  periodic-angle fitted value iteration compares the 129x129 development
  grid against 257x257, stores both converged value tables and the observed
  discretization error, and applies the exact path heat semigroup with a
  nonnegative, tail-stable dense reference backend. Both index-scale and
  physical-resolution heat are explicit; the latter scales path edge weights
  by ``1 / delta_u^2``.

The default M7 reference is strict about terminology: both fitted Bellman
fixed points must converge, while `reference_accepted` is set only when the
development-to-validation value and policy-statistic tolerances also pass.
The initial 129x129/257x257 run does not pass the declared `1e-3` normalized
value tolerance, so the refined artifact is retained with
`reference_accepted=false` as required by the experiment plan. Set
`reference.require_grid_convergence=true` to make that condition a hard CLI
failure when testing a finer reference grid.

- M8 uses one explicit state-dependent `nominal_pd` anchor for every method;
  this makes local torque neighborhoods meaningful without changing the true
  Pendulum dynamics. `truncated_heat_local` constructs sparse heat columns
  lazily from path adjacency. Exact-heat pruning and randomized estimators keep
  `reference_target=exact_heat`, while truncated heat, local uniform/RBF,
  MaxEnt, hard max, squared-torque LOT, and permuted geometry retain their own
  target labels. Index heat keeps a fixed graph radius; physical heat scales
  the primary radius with action resolution and therefore does not claim
  action-cardinality-independent cost. Results checkpoint atomically after
  each `(K, heat_scaling, temperature)` case and resume only matching configs.

- M9 keeps the weak-bridge input action graph distinct from the induced
  co-occurrence graph `A_Q`. It evaluates `Omega(pi)` with both transport
  marginals fixed, checks `A_Q 1 = pi_Q` and `rank(L_Q) = K - 1`, and verifies
  that central finite differences converge to `T0 R_Q(i,k)` for within-cluster,
  bridge-endpoint, and cross-cluster transfers across several `Q` vectors. This
  is an interpretation of an exact local identity, not evidence of an
  optimization-rate improvement.

The smoke configuration runs in seconds to a few minutes depending on the
machine. A measured single-core 129x129 benchmark at `K=51` took 18.4 seconds
for FullExactHeat and 138.9 seconds for radius-8 local heat before Anderson
acceleration. Extrapolating the complete 30-case paper grid with eleven methods
gives roughly 36-72 hours on one CPU, with physical-heat cases at `K=401/801`
dominating. The command is resumable; isolated single-thread timing should be
used for the paper runtime panels.
