# LOT experiments

Numerical experiments for **What Action Geometry Buys: A Design Space for
Optimal-Transport Bellman Backups**.

The repository currently implements milestone M0: configuration loading,
result validation, graph construction, exact and normalized finite-walk heat
kernels, stable LOT backups, lazy local heat columns, operation accounting,
and small deterministic numerical checks. Large experiment sweeps are not yet
implemented.

## Setup and checks

Python 3.11 or newer and [`uv`](https://docs.astral.sh/uv/) are required.

```bash
uv sync
uv run pytest -q
uv run python scripts/check_numerics.py --config configs/numerics.yaml
```

The diagnostic is deliberately small (`K <= 16`) and does not write experiment
results. Future raw, summary, and figure artifacts belong in `outputs/raw/`,
`outputs/summaries/`, and `outputs/figures/`, respectively.

## Numerical conventions

- Kernel columns are candidate distributions conditioned on anchors.
- `P_G = I - L / nu_u` is an action-graph walk, never an MDP transition kernel.
- Direct heat uses the centered backup (zero anchor offsets).
- Cost-based LOT retains the prior-mass offsets from Equation (19) of the paper.
- Pruned backup weights are not renormalized in the value.

