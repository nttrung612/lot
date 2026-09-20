# Laplacian OT Experiments: Implementation Plan

## 1. Goal

Build a small, reproducible experimental suite supporting the paper's central narrative:

$$\text{structured action geometry} \implies \text{concentrated action kernels} \implies \text{small local neighborhoods} \implies \text{accurate planning for the full heat-regularized target}.$$

A second set of experiments supports the learning narrative:

$$\text{reward-free transition samples} \implies \text{learned action geometry} \implies \text{controlled value error across reward tasks}.$$

The suite contains four synthetic studies and one practical benchmark, `Pendulum-v1`. The main paper should emphasize target accuracy, actions evaluated, graph accesses, oracle calls, runtime, and memory. Policy return is secondary because the main planning theorem concerns a scalar regularized value.

## 2. Claims and required evidence

| Paper claim | Required experiment | Success criterion |
|---|---|---|
| Kernel concentration controls deterministic local branching | Kernel design-map study | Heat kernels have small $d_{\mathrm{ker}}(\alpha)$ on slow-growth graphs; MaxEnt and dense Gibbs constructions scale with $K$ |
| The pruning certificate is sharp | One-backup pruning study | Constructed worst-case values attain the theoretical error up to numerical precision |
| Exact heat is dense but admits local finite-walk approximation | Poisson truncation study | Kernel, backup, and fixed-point errors decrease with radius and stay below their certificates |
| Local planning estimates the full exact-heat value | Stochastic ring MDP | Local method reaches the requested root-value accuracy while touching fewer actions than dense computation |
| The benefit depends on geometry | Path/grid/expander and permuted-graph controls | Gains appear on slow-growth aligned graphs and shrink on expanders or misaligned graphs |
| Geometry can be learned without rewards and reused | Reward-transfer study | Cost and fixed-point errors approach an $n^{-1/2}$ slope across several reward tasks |
| The mechanism applies to a recognizable control problem | `Pendulum-v1` | Local heat methods reproduce the dense heat value and policy with fewer action evaluations |
| Inverse policy curvature is effective resistance | Appendix visualization | Finite-difference regularizer curvature converges to $T_0 R_Q(i, k)$ |

## 3. Scientific invariants

Every experiment and plot must obey these rules:

1. Separate methods by target. A method that changes the regularizer cannot be ranked as an estimator of the exact-heat target.
2. Use `FullExactHeat` as the reference for exact-heat value and policy error.
3. Decompose local heat error whenever possible:

   $$    \vert{}\widehat V - V_t^*\vert{}    \le    \vert{}\widehat V - V_{t,r}^*\vert{}    +    \vert{}V_{t,r}^* - V_t^*\vert{}.    $$

4. Report transition calls, action evaluations, graph-neighbor accesses, runtime, and memory separately.
5. Report one-time geometry preprocessing separately from online query cost.
6. Use lazy heat-column construction. Do not precompute all columns for a method advertised as local.
7. Treat `Pendulum-v1` action-grid refinement carefully. Report both index-scale heat and resolution-consistent physical heat; do not claim $K$-independent complexity for the latter.
8. Keep the MDP transition kernel $P$, action-graph random walk $P_G$, and heat kernel $H_t$ as distinct objects in code, notation, logs, and plots.

## 4. Repository bootstrap

Create the following structure:

```text
.
├── AGENTS.md
├── EXPERIMENT_PLAN.md
├── README.md
├── pyproject.toml
├── configs/
│   ├── kernel_map.yaml
│   ├── pruning.yaml
│   ├── ring_planning.yaml
│   ├── geometry_learning.yaml
│   ├── pendulum.yaml
│   └── resistance.yaml
├── src/lot_experiments/
│   ├── __init__.py
│   ├── cli.py
│   ├── graphs.py
│   ├── kernels.py
│   ├── backups.py
│   ├── pruning.py
│   ├── heat_local.py
│   ├── metrics.py
│   ├── reproducibility.py
│   ├── environments/
│   │   ├── ring_control.py
│   │   └── pendulum_discrete.py
│   ├── planners/
│   │   ├── dense.py
│   │   ├── local_heat.py
│   │   ├── poisson_mc.py
│   │   └── random_subset.py
│   ├── learning/
│   │   └── transition_geometry.py
│   └── plotting/
│       ├── common.py
│       ├── kernel_map.py
│       ├── planning.py
│       ├── learning.py
│       └── pendulum.py
├── scripts/
│   ├── run_kernel_map.py
│   ├── run_pruning.py
│   ├── run_ring_planning.py
│   ├── run_geometry_learning.py
│   ├── run_pendulum.py
│   ├── run_resistance.py
│   └── make_all_figures.py
├── tests/
│   ├── test_kernels.py
│   ├── test_pruning_bound.py
│   ├── test_heat_truncation.py
│   ├── test_backups.py
│   └── test_target_labels.py
└── outputs/
    ├── raw/
    ├── summaries/
    └── figures/
```

Recommended stack:

- Python 3.11 or newer.
- `uv` for dependencies and command execution.
- NumPy, SciPy, pandas, PyYAML, Matplotlib, Seaborn, Gymnasium, psutil, pytest.
- Parquet for raw results and CSV for compact paper tables.
- Sparse SciPy matrices for action graphs and local heat operations.

Avoid adding PyTorch or JAX unless later experiments require function approximation.

## 5. Common result schema

Every run should write one or more rows with at least:

```text
experiment
run_id
seed
method
target
graph_family
K
alpha
radius
diffusion_time
poisson_mean
temperature
gamma
epsilon
delta
root_state
value_estimate
reference_value
absolute_value_error
statistical_error
heat_approximation_error
policy_l1_error
transition_calls
action_evaluations
unique_actions_touched
graph_neighbor_accesses
geometry_preprocess_seconds
online_seconds
peak_memory_mb
status
git_commit
config_json
```

Use explicit target labels such as:

- `exact_heat`
- `truncated_heat`
- `maxent`
- `diffusion_gibbs`
- `hard_max`

Plotting code must reject comparisons labeled as a common target when target labels differ.

## 6. Phase 0: numerical core and validation

### Deliverables

- Graph constructors for path, cycle, grid/torus, and random regular expander.
- Exact heat columns using `scipy.sparse.linalg.expm_multiply` or dense `expm` for small validation cases.
- Uniformized random-walk matrix:

  $$P_G = I - L / \nu_u, \qquad \nu_u \ge d_{\max}.$$

- Normalized finite-walk heat:

  $$H_{t,r} = \frac{1}{s_r}\sum_{k=0}^r e^{-\theta}\frac{\theta^k}{k!}P_G^k,   \qquad \theta = \nu_u t.$$

- Stable full and pruned log-sum-exp backups.
- Exact policy output for each backup.
- Operation counters shared across all planners.

### Required tests

- Every exact heat column is nonnegative and sums to one within tolerance.
- Every truncated heat column is nonnegative, sums to one, and has support inside the $r$-hop ball.
- $\Vert{}H_t(:, j) - H_{t,r}(:, j)\Vert{}_1 \le 2 \beta_r + \text{tolerance}$ on small graphs.
- The zero-cost LOT backup agrees with prior-weighted MaxEnt.
- Dense heat and local heat agree as $r$ becomes sufficiently large.
- Translation equivariance: $F(Q + c\mathbf{1}) = F(Q) + c$.
- Computed policies sum to one.

### Exit criterion

All numerical tests pass and a small diagnostic notebook or script verifies the formulas on $K \le 16$ graphs.

## 7. Phase 1: kernel design-map experiment

### Graphs and grid

- Path: $K \in \{64, 128, 256, 512, 1024, 2048, 4096\}$.
- Grid/torus: nearby square sizes up to approximately 4096 vertices.
- Random regular expander: degree 4 and the same $K$ values where possible.
- $\alpha \in \{10^{-1}, 10^{-2}, 10^{-3}, 10^{-4}\}$.
- $\theta = \nu_u t \in \{0.5, 2, 8\}$.

### Methods

- `UniformMaxEnt`.
- `DiffusionDistanceGibbs`.
- `ExactHeat`.
- `TruncatedHeat`.

### Measurements

- $d_{\mathrm{ker}}(\alpha)$.
- $d_{\mathrm{ker}}(\alpha) / K$.
- Minimum high-mass graph-ball radius.
- Exact support size.
- One-column construction time.
- Vertices and edges accessed.
- One-column and full-matrix memory estimates.

### Figures

1. $d_{\mathrm{ker}}(\alpha)$ versus $K$, log-log.
2. $d_{\mathrm{ker}}(\alpha)$ versus $\log(1/\alpha)$.
3. Path/grid/expander comparison at fixed $K$, $\alpha$, and $\theta$.
4. Local column-construction cost versus $K$.

### Expected evidence

- Uniform MaxEnt grows linearly in $K$.
- Exact heat concentrates on slow-growth graphs before saturation.
- Large diffusion time and expander graphs reduce locality.
- Truncated heat provides exact graph-ball support and local construction.

Do not hard-code expected slopes into assertions. The experiment should reveal them.

## 8. Phase 2: sharp pruning and Poisson truncation

### Worst-case pruning construction

For each retained top-weight set $I$, use

$$Q_i = 0 \text{ for } i \in I, \qquad Q_i = S \text{ for } i \notin I,$$

with $S / T_0 \in \{1, 2, 4, 8\}$.

Compare observed error with

$$T_0 \log\left(1 + e^{S/T_0}\frac{\alpha}{1-\alpha}\right).$$

### Typical-value construction

Use smooth sinusoidal values on cycles and smooth Gaussian random fields on grids. Report actual error and the worst-case certificate.

### Fixed-set baselines

At equal cardinality compare:

- Top kernel weights.
- Smallest graph ball.
- Uniform random subset.
- Farthest-action subset as a negative control.

### Poisson-truncation sweep

For $r = 0, 1, \dots, r_{\max}$, record:

- Poisson tail $\beta_r$.
- Kernel $L_1$ error.
- Backup error.
- Fixed-point error on a small tabular MDP.
- Ball size and graph accesses.

### Figures

1. Measured worst-case error versus sharp formula, with identity line.
2. Typical error and worst-case certificate versus omitted mass.
3. Kernel, backup, and fixed-point errors versus radius.
4. Error versus graph-ball size.

### Exit criterion

The worst-case construction attains the formula within `1e-8` on small deterministic cases, and all reported theoretical bounds hold with a declared numerical tolerance.

## 9. Phase 3: stochastic ring-control planning

### Environment

Use

$$\mathcal{S} = \mathcal{A} = \mathbb{Z}_K, \qquad s' = (s + a + \xi) \bmod K,$$

where

$$\Pr(\xi = 0) = 0.8, \qquad \Pr(\xi = 1) = \Pr(\xi = -1) = 0.1.$$

For goal $g$, define

$$r_g(s, a) = -\frac{d_{\mathrm{cyc}}(s, g)^2}{(K/2)^2} -c_a \frac{d_{\mathrm{cyc}}(a, 0)^2}{(K/2)^2}.$$

Default grid:

- $K \in \{64, 128, 256, 512, 1024\}$.
- $\gamma = 0.95$.
- $c_a = 0.05$.
- $T_0 \in \{0.05, 0.1, 0.2\}$.
- $\theta \in \{0.5, 2, 8\}$.
- $\epsilon \in \{0.2, 0.1, 0.05\}$.
- $\delta = 0.05$.

Use a cycle action graph. Use a full-support anchor distribution

$$\mu_s = (1 - \zeta)e_{j(s)} + \zeta \, \mathrm{Unif}(\mathcal{A}), \qquad \zeta = 0.05,$$

where $j(s)$ is a nominal action toward the goal. Add uniform $\mu$ as an appendix ablation.

### Same-target baselines

| Method ID | Description |
|---|---|
| `full_exact_heat` | Dense reference using exact heat and every action |
| `dense_second_order` | Full-action recursive estimator, equivalent to retaining all actions |
| `exact_heat_topmass` | Exact heat with deterministic top-mass pruning; isolates concentration |
| `truncated_heat_local` | Proposed locally computed finite-walk heat method |
| `poisson_endpoint_mc` | Samples exact-heat endpoints through Poisson random walks |
| `uniform_action_mc` | Uniform action sampling with appropriate importance weighting |
| `random_subset` | Equal-cardinality subset without geometry-aware selection |

### Different-target references

- `uniform_maxent`.
- `diffusion_distance_gibbs`.
- `hard_max`.
- `permuted_action_graph`.

Place these in separate plots or clearly label the target column.

### Primary measurements

- Root-value error to $V_{\mathrm{exact\_heat}}^*$.
- Error to $V_{\mathrm{truncated\_heat}}^*$.
- Heat approximation error.
- Empirical coverage of the requested $(\epsilon, \delta)$ guarantee.
- Transition-oracle calls.
- Action evaluations and unique actions touched.
- Graph-neighbor accesses.
- Online time and peak memory.

### Main plots

1. Root-value error versus transition calls.
2. Root-value error versus action evaluations.
3. Actions touched versus $K$ at fixed requested accuracy.
4. Runtime versus $K$.
5. Statistical and heat-bias error decomposition versus radius.
6. Positive control on cycle versus negative controls on expander and permuted graphs.

### Repetitions

- 30 paired seeds for recursive or Monte Carlo estimators.
- Report median, interquartile range, and bootstrap 95% confidence interval.
- Record failures and timeouts as rows; never silently drop them.

## 10. Phase 4: reward-free geometry learning and transfer

### Data

Use the ring-control transition system. Draw $n$ transitions per action from a
fixed, reward-independent, full-support probe distribution $\nu$. The primary
probe is a fixed-state/uniform mixture. A uniform-$\nu$ ablation is mandatory:
on the translation-invariant ring it is stationary under every action, hence
$z_a=P_a^\top\nu-\nu=0$ and $C_t=0$ for all actions. It cannot serve as the
primary geometry-learning task. This ablation records the degeneracy rather
than silently changing the action signature. Use:

```text
n in {16, 32, 64, 128, 256, 512, 1024, 2048}
```

Construct population and empirical:

- Action signatures $z_a$ and $\widehat{z}_a$.
- Shared Laplacians $L_\nu$ and $\widehat{L}_\nu$.
- Costs $C_t$ and $\widehat{C}_t$.

### Reward tasks

Reuse the same learned geometry for

```text
g / K in {0.0, 0.2, 0.4, 0.6, 0.8}
```

### Baselines

- Oracle population cost $C_t$.
- Proposed learned cost $\widehat{C}_t$.
- No-diffusion cost with $t = 0$.
- Raw action-coordinate cost.
- Permuted learned geometry.
- Uniform MaxEnt reference.

### Metrics

$$\Vert{}\widehat{C}_t - C_t\Vert{}_{\max}, \qquad \Vert{}Q_{\widehat{C}_t}^* - Q_{C_t}^*\Vert{}_\infty.$$

Also report the deterministic envelope

$$\frac{\gamma \tau}{1 - \gamma} \Vert{}\widehat{C}_t - C_t\Vert{}_{\max}.$$

Keep the true transition kernel in both downstream Bellman operators so that the measured error is geometry-learning error.

### Figures and table

1. Cost error versus $n$, log-log, with an $n^{-1/2}$ reference slope.
2. Fixed-point error versus $n$, with the propagation envelope.
3. Thin lines for individual reward tasks plus median and worst-case curves.
4. Samples required to reach fixed value-error thresholds $\{0.1, 0.05, 0.02\}$.

## 11. Phase 5: `Pendulum-v1`

### Purpose

Show that torque geometry provides useful local action neighborhoods on a standard control benchmark. Pendulum supports practical relevance; it is not used to prove the finite-state learning rate.

### Action grids

Discretize torque $[-2, 2]$ as

$$u_i = -2 + \frac{4(i - 1)}{K - 1}, \qquad K \in \{51, 101, 201, 401, 801\}.$$

Use a path graph over ordered torques.

### Two heat scalings

1. `index_heat`: unit edge weights and fixed diffusion time. This matches a growing index-graph family but shrinks the physical torque bandwidth as $K$ increases.
2. `physical_heat`: with $\Delta_u = 4 / (K - 1)$, use

   $$L_K^{\mathrm{phys}} = L_K / \Delta_u^2.$$

   This maintains a comparable continuous torque geometry. Do not claim $K$-independent complexity in this regime.

### Planning settings

- Primary environment: unmodified deterministic Pendulum dynamics.
- Remove the external episode `TimeLimit` only for discounted fixed-point computation.
- $\gamma = 0.99$.
- Divide planning rewards by $16.2736044$.
- $T_0 \in \{0.02, 0.05, 0.1\}$.
- Report raw unnormalized Pendulum returns for behavioral interpretation.

### Reference solution

Use fitted value iteration on periodic state grids:

- Development grid: $129 \times 129$ over angle and angular velocity.
- Validation grid: $257 \times 257$.
- Periodic interpolation in angle.
- Exact deterministic next-state computation.

Accept the reference when reported root values and policy statistics change by less than a predeclared tolerance, initially $10^{-3}$ in normalized value. If this tolerance is not achieved, refine the grid and report the observed discretization error.

### Evaluation states

Fixed diagnostics:

```text
(theta, theta_dot) in {
  (pi, 0),
  (pi/2, 0),
  (-pi/2, 0),
  (0, 0),
  (pi, 2)
}
```

Also use 100 initial states from the standard reset distribution, shared across all methods.

### Same-target baselines

- `full_exact_heat`.
- `exact_heat_topmass`.
- `truncated_heat_local`.
- `poisson_endpoint_mc`.
- `uniform_random_subset`.
- `local_uniform_neighborhood`.
- `local_rbf_neighborhood`.

The local-uniform and RBF baselines use the same neighborhood size as the proposed method and test whether heat weights add value beyond generic locality.

### Different-objective references

- Uniform MaxEnt.
- Hard-max discretized control.
- Squared-torque-cost LOT with $C_{ij} = (u_i - u_j)^2$.
- Randomly permuted path graph.

### Primary metrics

- Error to full exact-heat value on the evaluation set.
- Mean and maximum Bellman residual.
- Policy $L_1$ error to the full exact-heat policy.
- Actions evaluated per backup.
- Fraction of actions evaluated.
- Graph-neighbor accesses.
- Time per Bellman sweep.
- Total planning time to the common stopping tolerance.
- Peak memory.

### Secondary behavioral metrics

- Raw episodic return over 100 shared initial states.
- Regularized discounted value.
- Mean squared angle error.
- Fraction of time satisfying $|\theta| \le 0.2$ and $|\dot{\theta}| \le 1$.
- Mean absolute torque.

### Main Pendulum plots

1. Error to exact heat versus actions evaluated per backup.
2. Planning time versus $K$, with separate index-heat and physical-heat panels.
3. Policy $L_1$ error versus computation.
4. Raw return and regularized value, with the full exact-heat reference marked.
5. Radius ablation: heat bias and computation versus $r$.

### Required interpretation

The desired conclusion is:

> On Pendulum, ordered torque geometry concentrates the heat backup around related controls. Local heat computation reproduces the dense heat-regularized value and policy with fewer action evaluations.

Do not state action-cardinality-independent complexity for resolution-consistent action-grid refinement.

## 12. Phase 6: effective-resistance appendix visualization

Use a two-cluster action graph joined by a weak bridge. For several $Q$ vectors:

1. Compute the optimal coupling $\Gamma_Q$.
2. Form $A_Q = \Gamma_Q \operatorname{diag}(\mu)^{-1} \Gamma_Q^\top$.
3. Form $L_Q$ and its pseudoinverse.
4. Compare within-cluster, bridge, and cross-cluster pairs.
5. Numerically verify

   $$
   \frac{\Omega(\pi_Q + h(e_i - e_k)) - 2\Omega(\pi_Q) + \Omega(\pi_Q - h(e_i - e_k))}{h^2}
   \longrightarrow T_0 R_Q(i, k).
   $$

Plot the input graph, induced co-occurrence graph, resistance matrix, and finite-difference convergence. Present this as interpretation of an exact identity, not independent evidence for an optimization improvement.

## 13. Statistical protocol

- Use paired seeds across comparable stochastic methods.
- Use 30 seeds for expensive planners and 100 for inexpensive kernel/backup experiments.
- Default summary: median and interquartile range.
- Headline stochastic results: bootstrap 95% confidence intervals.
- Store every run, including failures and timeouts.
- Fix hyperparameters on one declared development configuration.
- Do not tune separately for every $K$ unless the tuning rule is itself part of the method.
- Record machine, CPU, thread count, BLAS backend, package versions, and git commit.
- Force single-threaded BLAS for clean runtime comparisons unless parallel scaling is a stated experiment.
- Independent cases may run in separate single-threaded processes to accelerate
  accuracy, coverage, and operation-count collection. Label their per-process
  wall times as concurrent diagnostics and exclude them from paper-ready runtime
  plots; obtain those plots from an otherwise identical one-worker run.

## 14. Fair cost accounting

Report these costs independently:

1. Transition-oracle queries.
2. Action-value evaluations.
3. Graph-neighbor accesses.
4. Dense linear-algebra operations.
5. Geometry preprocessing time.
6. Online query time.
7. Peak memory.

For cached methods, report cold-start and amortized results. Never hide full-kernel construction in an unreported preprocessing step.

## 15. Main-paper deliverables

Recommended main-paper package:

1. **Figure 1:** kernel concentration and graph-family negative controls.
2. **Figure 2:** sharp pruning and Poisson truncation.
3. **Figure 3:** synthetic end-to-end error/computation scaling.
4. **Figure 4:** Pendulum accuracy/computation frontier.
5. **Table 1:** learned-geometry rate and reward transfer.

Appendix:

- Additional $t$, $T_0$, $\alpha$, and $r$ sweeps.
- Full runtime and memory tables.
- Online Q-learning curves if implemented.
- Effective-resistance visualization.
- MaxEnt/MLMC comparison in the zero-cost special case, if implemented.
- Additional Pendulum states and grid-refinement checks.

## 16. Implementation milestones

- [ ] M0: Bootstrap package, configs, result schema, logging, and tests.
- [ ] M1: Exact and truncated heat kernels validated on small graphs.
- [ ] M2: Kernel design-map experiment complete with Figure 1 draft.
- [ ] M3: Sharp pruning and Poisson-truncation experiment complete.
- [x] M4: Ring environment, dense references, and baseline planners implemented.
- [x] M5: End-to-end synthetic planning results complete.
- [x] M6: Reward-free learned-geometry experiment complete.
- [x] M7: Pendulum wrapper and converged reference solution complete.
- [ ] M8: Pendulum baselines and final figures complete.
- [ ] M9: Effective-resistance appendix figure complete.
- [ ] M10: Reproduction command reruns all paper artifacts from clean outputs.

## 17. Final acceptance checklist

- [ ] Every main claim is linked to at least one figure or table.
- [ ] Same-target and different-target methods are never mixed in an error ranking.
- [ ] All theoretical certificates are numerically checked on small instances.
- [ ] Dense and local implementations agree when truncation is removed.
- [ ] Negative controls are present: expander, large diffusion time, random/permuted geometry.
- [ ] Raw results, summaries, and figures can be regenerated from configs.
- [ ] Pendulum results include physical-resolution scaling and reference-grid convergence.
- [ ] Runtime tables distinguish preprocessing, cold-start, and amortized cost.
- [ ] All plots have units, confidence summaries, and explicit target labels.
- [ ] The README contains one command for each experiment and one command for all paper figures.
