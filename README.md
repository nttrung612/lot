# LOT experiments

Numerical experiments for **What Action Geometry Buys: A Design Space for
Optimal-Transport Bellman Backups**.

The repository currently implements milestones M0-M4: configuration loading,
result validation, graph construction, exact and normalized finite-walk heat
kernels, reusable Poisson-tail certificates, stable LOT backups, lazy local
heat columns, operation accounting, and deterministic validation on all planned
small graph families, the kernel design-map experiment with a Figure 1 draft,
the sharp-pruning/Poisson-truncation experiment with a Figure 2 draft, and the
stochastic ring-control environment with dense target references and baseline
planners. The end-to-end ring sweep and later learning studies are not yet
implemented.

## Setup and checks

Python 3.11 or newer and [`uv`](https://docs.astral.sh/uv/) are required.

```bash
uv sync
uv run pytest -q
uv run python scripts/check_numerics.py --config configs/numerics.yaml
uv run python scripts/run_kernel_map.py --config configs/kernel_map.yaml
uv run python scripts/run_pruning.py --config configs/pruning.yaml
uv run python scripts/make_all_figures.py
```

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
