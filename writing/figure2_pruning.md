# Figure 2 — From Kernel Concentration to Certified Backup and Fixed-Point Accuracy

## 1. Role of this experiment

Figure 1 established that heat-kernel mass can concentrate on a small action neighborhood. Figure 2 asks the next question:

> If most kernel mass is local, how much error is introduced by evaluating only that local mass, and how does the error propagate from one Bellman backup to a planning fixed point?

The figure studies two related but mathematically distinct operations:

1. **Deterministic pruning of Exact Heat.** A subset of entries from an exact heat column is retained, and the omitted weights are not renormalized. This approximates the original Exact Heat backup.
2. **Poisson truncation of the heat kernel.** The infinite Poisson mixture defining Exact Heat is replaced by a normalized finite mixture. This produces the distinct Truncated Heat target, whose error is measured against Exact Heat.

This separation is essential. Panels A–B test the sharp pruning certificate for a fixed Exact Heat target. Panels C–D test how a certified kernel approximation propagates through a backup and a small tabular fixed point.

Together, the panels support the following chain:

$$
\text{small omitted kernel mass}
\Longrightarrow
\text{controlled one-backup error}
\Longrightarrow
\text{controlled fixed-point error}.
$$

The experiment also distinguishes a worst-case guarantee from typical behavior. Panel A constructs action values that attain the pruning formula exactly. Panel B uses smooth action-value fields and shows how conservative that same certificate is in structured cases.

## 2. Exact and pruned heat backups

For one anchor with normalized kernel weights

$$
w_i\ge0,
\qquad
\sum_i w_i=1,
$$

and action values $Q_i$, the full heat backup is

$$
F_w(Q)
=
T_0
\log
\left(
\sum_i
w_i
\exp\left(\frac{Q_i}{T_0}\right)
\right).
$$

Let $I$ be the retained action set. The pruned backup is

$$
F_{w,I}(Q)
=
T_0
\log
\left(
\sum_{i\in I}
w_i
\exp\left(\frac{Q_i}{T_0}\right)
\right).
$$

The weights inside $I$ are **not renormalized**. This convention preserves the interpretation of pruning as removing terms from the same Exact Heat log-partition function. Renormalizing the retained weights would define a different backup and invalidate the sharp pruning formula tested here.

Define the actual omitted mass

$$
\alpha
=
\sum_{i\notin I}w_i,
$$

and the action-value span

$$
S
=
\max_i Q_i-\min_i Q_i.
$$

The signed pruning error obeys

$$
0
\le
F_w(Q)-F_{w,I}(Q)
\le
T_0
\log
\left(
1+
\exp\left(\frac{S}{T_0}\right)
\frac{\alpha}{1-\alpha}
\right).
$$

The experiment refers to the right-hand side as the **sharp pruning formula**. The bound depends on two quantities:

- omitted kernel mass $\alpha$;
- dynamic range measured in temperature units, $S/T_0$.

This dependence explains why small omitted mass alone is insufficient. A small amount of omitted kernel mass can still matter if it is placed on actions whose values exceed the retained actions by many temperatures.

## 3. Data and experimental design

Figure 2 uses fully synthetic, controlled inputs. No learned model or external dataset is involved.

The resolved configuration is:

| Component | Configuration |
|---|---|
| Temperature | $T_0=0.1$ |
| Value-span ratios | $S/T_0\in\{1,2,4,8\}$ |
| Worst-case graph | Cycle with $K=41$ |
| Typical graphs | Cycle with $K=25$ and grid with $6\times6=36$ actions |
| Typical repetitions | 100 paired replicates |
| Poisson-truncation graphs | Path with 25 actions, cycle with 25 actions, and $5\times5$ grid |
| Poisson means | $\theta\in\{0.5,2,8\}$ |
| Truncation radii | $r=0,\ldots,20$ |
| Fixed-point discount | $\gamma=0.9$ |
| Fixed-point states | 5 |

The raw output contains 19,422 completed rows:

| Experiment part | Rows |
|---|---:|
| Adversarial worst case | 24 |
| Smooth typical values | 19,200 |
| Poisson truncation | 189 |
| Exact-heat fixed-point references | 9 |

The 19,200 typical rows arise from

$$
2\text{ graphs}
\times
100\text{ replicates}
\times
4\text{ span ratios}
\times
6\text{ subset radii}
\times
4\text{ selection methods}.
$$

All stochastic components use deterministically derived NumPy generator seeds. The same smooth value field is reused across the four subset-selection methods within a replicate, which makes their comparisons paired.

## 4. Part I: adversarial construction

### 4.1. Kernel and retained set

The adversarial experiment uses one Exact Heat column on a cycle with:

$$
K=41,
\qquad
\theta=2,
\qquad
j=0.
$$

For each nominal mass budget, the implementation sorts the exact heat weights and retains the smallest top-mass set whose total mass is at least $1-\alpha_{\mathrm{nominal}}$.

Because actions are discrete, the actual omitted mass is generally smaller than the nominal budget. For example:

| Nominal budget | Actual omitted mass | Retained actions |
|---:|---:|---:|
| 0.40 | 0.260953 | 3 |
| 0.20 | 0.167714 | 4 |
| 0.10 | 0.074475 | 5 |
| 0.05 | 0.045684 | 6 |
| 0.02 | 0.016893 | 7 |
| 0.01 | 0.003162 | 9 |

The certificate is evaluated using the actual omitted mass. This prevents the comparison from attributing unused nominal budget to the algorithm.

### 4.2. Worst-case action values

After selecting $I$, action values are constructed as

$$
Q_i=
\begin{cases}
0, & i\in I,\\
S, & i\notin I.
\end{cases}
$$

This assigns the smallest value to every retained action and the largest value to every omitted action. It is the arrangement that maximizes the damage caused by removing a fixed amount of kernel mass.

For this construction,

$$
F_w(Q)
=
T_0
\log
\left[
(1-\alpha)
+
\alpha
\exp\left(\frac{S}{T_0}\right)
\right],
$$

while

$$
F_{w,I}(Q)
=
T_0\log(1-\alpha).
$$

Their difference is exactly

$$
F_w(Q)-F_{w,I}(Q)
=
T_0
\log
\left(
1+
\exp\left(\frac{S}{T_0}\right)
\frac{\alpha}{1-\alpha}
\right).
$$

Thus, the upper bound is attainable rather than merely sufficient.

## 5. Part II: smooth action-value fields and equal-cardinality baselines

The typical-value experiment asks whether the worst-case construction reflects ordinary smooth action values.

### 5.1. Smooth value generation

On the cycle, each replicate uses a randomly phase-shifted smooth signal:

$$
\widetilde Q_i
=
\sin(\omega_i+\phi_1)
+
0.25\sin(2\omega_i+\phi_2),
\qquad
\omega_i=\frac{2\pi i}{K}.
$$

On the grid, the experiment samples white Gaussian noise $z$ and smooths it with the graph heat operator:

$$
\widetilde Q
=
\exp(-t_sL)z,
\qquad
t_s=1.
$$

Each field is affinely rescaled so that

$$
\max_i Q_i-\min_i Q_i=S.
$$

Figure 2B displays the case

$$
\frac{S}{T_0}=4,
\qquad
S=0.4.
$$

### 5.2. Equal-cardinality selection methods

For each graph radius $\rho\in\{0,1,2,3,4,5\}$, the graph ball around the anchor determines a cardinality $m$. Every selection method must retain exactly $m$ actions:

| Method | Retained set |
|---|---|
| Top heat weights | The $m$ actions with largest exact heat weights |
| Graph ball | All actions in the radius-$\rho$ ball |
| Uniform random subset | A uniformly sampled subset of size $m$ |
| Farthest subset | The $m$ actions farthest from the anchor |

The cycle cardinalities are

$$
m\in\{1,3,5,7,9,11\},
$$

and the centered $6\times6$ grid cardinalities are

$$
m\in\{1,5,13,23,31,35\}.
$$

Equal cardinality is the fairness constraint. It ensures that differences reflect which actions are retained rather than how many actions a method receives.

The x-axis in Panel B is the **actual omitted heat mass** of each selected set. Two methods with the same cardinality can appear at very different x-coordinates because one set may align with the heat kernel and the other may discard nearly all of its mass.

## 6. Part III: normalized Poisson truncation

Exact Heat has the uniformized representation

$$
H_t
=
\sum_{k=0}^{\infty}
p_kP_G^k,
\qquad
p_k=e^{-\theta}\frac{\theta^k}{k!}.
$$

The normalized finite-walk kernel is

$$
H_{t,r}
=
\frac{1}{1-\beta_r}
\sum_{k=0}^{r}
p_kP_G^k,
$$

where

$$
\beta_r
=
\Pr\left[N>r\right],
\qquad
N\sim\operatorname{Pois}(\theta).
$$

Unlike deterministic pruning, $H_{t,r}$ is normalized and defines a distinct kernel target. The experiment therefore labels it <code>truncated_heat</code> and compares it against the <code>exact_heat</code> reference.

### 6.1. Kernel certificate

For every anchor,

$$
\left\|
H_t(:,j)-H_{t,r}(:,j)
\right\|_1
\le
2\beta_r.
$$

The reported kernel error is the maximum column error:

$$
E_{\mathrm{ker}}(r)
=
\max_j
\left\|
H_t(:,j)-H_{t,r}(:,j)
\right\|_1.
$$

### 6.2. One-backup certificate

The exact kernel decomposes as

$$
H_t
=
(1-\beta_r)H_{t,r}
+
\beta_rR_r,
$$

where $R_r$ is the normalized omitted Poisson tail.

For action-value span $S$, the one-backup error is bounded by

$$
\delta_r(S)
=
T_0
\log
\left[
1+
\beta_r
\left(
\exp\left(\frac{S}{T_0}\right)-1
\right)
\right].
$$

This expression differs from the pruning formula. Deterministic pruning leaves a subnormalized retained measure and introduces the ratio $\alpha/(1-\alpha)$. Poisson truncation renormalizes the head, leading to the mixture form $\beta_r(\exp(S/T_0)-1)$.

### 6.3. Fixed-point certificate

The Bellman operator is a $\gamma$-contraction. Therefore,

$$
\left\|
V_{t,r}^*-V_t^*
\right\|_\infty
\le
\frac{\delta_r(S)}{1-\gamma}.
$$

The corresponding action-value error satisfies

$$
\left\|
Q_{t,r}^*-Q_t^*
\right\|_\infty
\le
\frac{\gamma\,\delta_r(S)}{1-\gamma}.
$$

Figure 2C displays the kernel error, one-backup error, and fixed-point value error together with their theoretical bounds.

## 7. Fixed-point propagation environment

The fixed-point component uses a small deterministic tabular construction designed only to test error propagation:

- five states;
- $K=25$ actions;
- one anchor action per state, evenly spaced over the action graph;
- discount factor $\gamma=0.9$;
- Bellman iteration tolerance $10^{-12}$.

Actions are divided into five groups through

$$
u(a)
=
\left\lfloor
\frac{5a}{K}
\right\rfloor.
$$

From state $s$, action $a$ has a nominal destination

$$
\bar s'
=
\bigl(s+u(a)\bigr)\bmod 5.
$$

The transition law assigns probability 0.8 to $\bar s'$ and probability 0.1 to each adjacent state.

Rewards combine a state term with distance from the state's anchor action:

$$
r(s,a)
=
-0.3\,d_{\mathrm{state}}(s,0)^2
-0.7
\left(
\frac{d_G(a,j_s)}
{\max_{a'}d_G(a',j_s)}
\right)^2,
$$

with normalized cyclic state distance. This construction produces smooth graph-aware rewards while keeping the true dynamics fixed across Exact and Truncated Heat.

For each graph and $\theta$, the experiment first solves the FullExactHeat fixed point. Each truncated radius is then evaluated against that same reference.

## 8. Reading Figure 2

### 8.1. Panel A — Adversarial construction

Panel A plots one-backup pruning error against the actual omitted heat mass. A separate color and marker represent each value-span ratio; solid curves show the sharp formula and hollow markers show measured errors for:

$$
6\text{ mass budgets}
\times
4\text{ span ratios}
=24\text{ cases}.
$$

The markers lie on their corresponding formula curves. The panel annotates the maximum absolute formula gap in the raw output:

$$
1.11\times10^{-16}.
$$

Representative results include:

| Actual omitted mass | $S/T_0$ | Measured error |
|---:|---:|---:|
| 0.003162 | 1 | 0.000859 |
| 0.003162 | 8 | 0.234710 |
| 0.074475 | 4 | 0.168518 |
| 0.260953 | 8 | 0.695993 |

The panel validates two claims.

First, the formula is numerically sharp: there are admissible action values that attain it.

Second, the value span matters strongly. Holding omitted mass near 0.003 fixed while increasing $S/T_0$ from 1 to 8 increases the error from approximately $8.6\times10^{-4}$ to 0.235. A pruning rule based only on kernel mass cannot guarantee small error without controlling the action-value span.

### 8.2. Panel B — Smooth values

Panel B reports median one-backup error over 100 paired smooth-value replicates at $S/T_0=4$. Both axes use logarithmic scales.

The top-weight and graph-ball curves overlap on both displayed graphs. For the symmetric cycle and the centered regular grid at this diffusion scale, the largest heat weights are ordered by graph distance, so the corresponding equal-cardinality graph ball selects the same actions.

On the cycle:

| Retained actions | Graph-ball omitted mass | Graph-ball median error | Random-subset median error |
|---:|---:|---:|---:|
| 3 | 0.260953 | 0.038422 | 0.223850 |
| 7 | 0.016893 | 0.003144 | 0.140687 |
| 11 | 0.000502 | 0.000134 | 0.084336 |

On the $6\times6$ grid:

| Retained actions | Graph-ball omitted mass | Graph-ball median error | Random-subset median error |
|---:|---:|---:|---:|
| 5 | 0.394592 | 0.058439 | 0.212919 |
| 23 | 0.021397 | 0.003209 | 0.040739 |
| 35 | 0.000084 | 0.000006 | 0.000691 |

The main result is that selection quality is explained by retained kernel mass, not cardinality alone. A local graph ball performs well because the heat kernel is aligned with graph distance. A random subset of the same size omits much more heat mass, while the farthest subset deliberately retains the least relevant region and gives the largest error.

All typical errors remain below the worst-case curve. The gap is expected: smooth fields do not systematically place their maximum values on every omitted action and their minimum values on every retained action.

The summary stores medians and interquartile ranges. The current panel displays medians only.

### 8.3. Panel C — Error propagation

Panel C uses the 25-action cycle with $\theta=2$ and varies $r$ from 0 to 20.

| Radius | Ball size | Poisson tail | Kernel L1 error | Backup error | Fixed-point value error |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 0.8647 | 1.3830 | 0.1237 | 0.2864 |
| 2 | 5 | 0.3233 | 0.2100 | 0.00842 | 0.0471 |
| 4 | 9 | 0.05265 | 0.03384 | 0.00146 | 0.00886 |
| 6 | 13 | 0.004534 | 0.003230 | 0.000149 | 0.000945 |
| 8 | 17 | 0.0002374 | 0.0001936 | $8.97\times10^{-6}$ | $5.86\times10^{-5}$ |
| 10 | 21 | $8.31\times10^{-6}$ | $7.51\times10^{-6}$ | $3.47\times10^{-7}$ | $2.33\times10^{-6}$ |
| 12 | 25 | $2.07\times10^{-7}$ | $2.02\times10^{-7}$ | $9.36\times10^{-9}$ | $6.41\times10^{-8}$ |

All three observed errors decay rapidly as the Poisson radius increases. At $r=6$, the local kernel uses at most 13 of 25 actions and already reduces fixed-point value error below $10^{-3}$.

The observed one-backup error need not decrease monotonically at every adjacent radius. For example, it increases slightly from $r=1$ to $r=2$. The theorem bounds the magnitude through a monotone Poisson-tail envelope; it does not require realized errors for a particular value field to be monotone.

The kernel bound is relatively informative. At $r=6$:

$$
E_{\mathrm{ker}}=0.003230,
\qquad
2\beta_r=0.009068.
$$

The backup and fixed-point envelopes are much looser for this MDP because they use only the global exact action-value span,

$$
S\approx1.056991,
$$

and then apply the contraction factor $1/(1-\gamma)=10$. The bounds remain valid, but their slack shows the cost of a distribution-free worst-case guarantee.

This does not conflict with Panel A. Panel A deliberately realizes the worst-case alignment for deterministic pruning; Panel C uses smooth MDP-generated values and a different normalized-kernel approximation.

### 8.4. Panel D — Accuracy versus local action count

Panel D replaces radius on the x-axis with maximum graph-ball size. It expresses the result in the resource most directly relevant to local planning: how many actions may be touched by a local column.

For a 25-cycle:

$$
|B(j,r)|
=
\min(2r+1,25).
$$

The panel shows a steep accuracy improvement as the local action count grows from 1 to 25. In particular, 13 local actions at $r=6$ give:

$$
E_{\mathrm{ker}}\approx3.23\times10^{-3},
$$

$$
E_{\mathrm{backup}}\approx1.49\times10^{-4},
$$

$$
E_{\mathrm{fixed}}\approx9.45\times10^{-4}.
$$

At $r=12$, the graph ball covers all 25 actions. Errors remain nonzero because normalized Poisson truncation still omits walk lengths greater than 12. Increasing $r$ beyond the graph diameter no longer expands support, but it continues to improve the Poisson mixture. This produces the vertical sequence of points at action count 25.

This distinction matters operationally:

- graph-ball size controls spatial support and local action access;
- Poisson radius controls how many walk orders are retained;
- after support saturates, further accuracy comes from additional propagation work rather than fewer actions.

## 9. Numerical validation

The complete run satisfies all configured checks:

- all 24 adversarial cases match the sharp formula within the numerical tolerance;
- all 19,200 smooth-value pruning cases satisfy their pruning certificate;
- all 189 Poisson-truncation cases satisfy the kernel L1 certificate;
- all 189 cases satisfy the one-backup certificate;
- all 189 cases satisfy both fixed-point value and action-value certificates;
- the largest recorded fixed-point residual is below $10^{-12}$.

Target labels remain explicit:

- Panels A–B use <code>target=exact_heat</code>;
- Panels C–D use <code>target=truncated_heat</code> and <code>reference_target=exact_heat</code>;
- FullExactHeat reference rows are stored separately.

The run used one process and single-threaded numerical libraries. Its metadata records seed, resolved configuration, package versions, git commit, hardware, and thread environment.

## 10. Scientific interpretation

### 10.1. The certificate is sharp but typical behavior is milder

Panel A proves that the pruning expression cannot be improved without adding assumptions about $Q$. Panel B shows why the bound can nevertheless be conservative in practice. Smooth graph-aligned value fields rarely realize the adversarial arrangement in which all omitted actions are maximizers and all retained actions are minimizers.

The two panels answer different questions:

- Panel A asks whether the theorem is mathematically tight.
- Panel B asks how the same guarantee compares with structured values.

Both are needed. Typical results alone would not establish a worst-case guarantee, while the adversarial result alone would overstate the errors likely under smooth structure.

### 10.2. Kernel mass is a better selection criterion than set size

Equal-cardinality comparisons show that evaluating the same number of actions does not imply comparable accuracy. What matters is how much relevant kernel mass the subset captures.

Top-weight pruning is optimal for retained mass at fixed cardinality. A graph ball matches it when heat weights decrease with graph distance. This connection is what converts a global top-weight rule into a locally discoverable rule on aligned, slow-growth graphs.

### 10.3. Kernel error propagates through progressively more problem structure

Panels C–D track three levels:

$$
H_{t,r}\approx H_t,
$$

$$
\mathcal T_{t,r}Q\approx\mathcal T_tQ,
$$

$$
V_{t,r}^*\approx V_t^*.
$$

The kernel L1 error is purely geometric. Backup error also depends on the action-value landscape and temperature. Fixed-point error additionally depends on the discount factor and repeated Bellman application.

This hierarchy explains why a kernel-level certificate is necessary but does not by itself describe planning accuracy. Figure 2 supplies the missing propagation evidence.

### 10.4. Spatial locality and series truncation are separate resources

Panel D reveals a useful distinction that radius-only plots can hide. On a finite graph, the support may already cover every action while the finite Poisson series has not converged numerically to Exact Heat.

Before the graph diameter, increasing $r$ expands both support and walk depth. After the diameter, support cost is saturated but propagation depth continues to grow. A practical planner should therefore report graph accesses, actions touched, and walk-propagation work separately.

## 11. Claims supported by Figure 2

Figure 2 supports the following statements:

- the deterministic pruning certificate is sharp;
- unnormalized top-mass pruning preserves the Exact Heat target interpretation;
- smooth graph-aligned values produce substantially smaller errors than the adversarial envelope;
- equal-size local or top-weight sets capture more useful heat mass than random or farthest sets;
- normalized finite-walk heat converges to Exact Heat as radius increases;
- kernel, one-backup, and fixed-point errors remain below their respective certificates;
- the local action-count versus accuracy tradeoff can be measured explicitly.

## 12. Claims not established by Figure 2

The figure does not establish:

- policy-return improvement;
- simple-regret or best-action guarantees;
- statistical lower bounds for random-subset baselines;
- that observed fixed-point error must decrease at every successive radius;
- that the fixed-point certificates are tight on typical MDPs;
- that full support implies Exact Heat has been recovered;
- end-to-end savings on a large planning problem.

The end-to-end action-evaluation and oracle-call claims are tested by the synthetic ring-planning experiment.

## 13. Threats to validity and possible refinements

1. **Small synthetic graphs.** The certificate checks deliberately use small deterministic graphs so exact references remain reliable. Scaling behavior belongs to the kernel-map and planning experiments.

2. **Only two typical graph families.** Smooth pruning uses a cycle and grid. Adding a misaligned or expander control would test how much the graph-ball advantage depends on geometric alignment.

3. **Median-only Panel B.** The summary contains the 25th and 75th percentiles, but the current figure does not display them. Bands or error bars would expose variability across smooth fields and random subsets.

4. **One visible truncation slice.** Panels C–D show only the cycle at $\theta=2$, although raw data also contain path/grid and $\theta\in\{0.5,2,8\}$. Supplementary panels could demonstrate how diffusion scale and volume growth change the tradeoff.

5. **Loose backup and fixed-point bounds.** The span-only envelope is valid but conservative. A tighter result would require additional information about where high values occur relative to omitted heat mass.

6. **Full kernels in the diagnostic fixed point.** The experiment materializes every truncated column to compute exact small-graph errors. This is an offline validation choice; it should not be reported as the memory behavior of the lazy planning algorithm.

## 14. Paper-ready Results paragraph

> **Kernel-mass control yields certified backup and fixed-point accuracy.** Figure 2 first verifies that the deterministic Exact Heat pruning certificate is sharp. Across 24 adversarial constructions on a 41-action cycle, measured one-backup errors agree with the closed-form expression to within $1.11\times10^{-16}$. Under 100 paired smooth-value replicates, errors are substantially smaller than this worst-case envelope, and equal-cardinality graph balls match top-heat-weight selection on the cycle and centered grid while outperforming random and farthest subsets by retaining more heat mass. We then replace Exact Heat with the normalized Poisson head and measure error relative to FullExactHeat on a 25-action tabular control problem. At $\theta=2$ and radius $r=6$, the local kernel touches at most 13 actions while attaining kernel L1 error $3.23\times10^{-3}$, one-backup error $1.49\times10^{-4}$, and fixed-point value error $9.45\times10^{-4}$. All 189 graph–diffusion–radius cases satisfy the kernel, backup, and fixed-point certificates. These results connect kernel concentration to controlled Bellman error while keeping deterministic pruning mass $\alpha$ distinct from the Poisson tail $\beta_r$.

## 15. Proposed Figure 2 caption

> **Figure 2: Sharp pruning and propagation of local heat approximation error.** (A) The adversarial construction places value 0 on retained actions and value $S$ on omitted actions. Measured Exact Heat pruning errors (hollow markers) follow the sharp formula (solid curves) as actual omitted mass changes across six budgets and $S/T_0\in\{1,2,4,8\}$. The maximum absolute gap is $1.11\times10^{-16}$. (B) For smooth values at $S/T_0=4$, top-heat-weight and graph-ball subsets have much smaller median error than equal-cardinality random and farthest subsets; all observed errors remain below the worst-case certificate. (C) On a 25-action cycle with $\theta=2$, normalized Poisson-head kernel error, one-backup error, and fixed-point value error decay with truncation radius and remain below their respective bounds. (D) The same errors plotted against maximum graph-ball size expose the accuracy–local-action tradeoff. Panels A–B prune unnormalized Exact Heat weights and approximate the same Exact Heat target. Panels C–D compare the distinct Truncated Heat target with FullExactHeat; the Poisson tail $\beta_r$ is distinct from pruning mass $\alpha$.

## 16. Reproducing the experiment

Run the complete pruning and truncation experiment:

~~~bash
uv run python scripts/run_pruning.py --config configs/pruning.yaml
~~~

Regenerate Figure 2 from the existing summary:

~~~bash
uv run python scripts/make_all_figures.py --pruning-config configs/pruning.yaml
~~~

Relevant artifacts:

- configuration: [configs/pruning.yaml](../configs/pruning.yaml);
- raw run-level data: [outputs/raw/pruning.parquet](../outputs/raw/pruning.parquet);
- aggregated summary: [outputs/summaries/pruning.csv](../outputs/summaries/pruning.csv);
- PNG figure: [outputs/figures/figure2_pruning.png](../outputs/figures/figure2_pruning.png);
- PDF figure: [outputs/figures/figure2_pruning.pdf](../outputs/figures/figure2_pruning.pdf);
- experiment implementation: [src/lot_experiments/pruning_experiment.py](../src/lot_experiments/pruning_experiment.py);
- pruning certificates: [src/lot_experiments/pruning.py](../src/lot_experiments/pruning.py);
- plotting code: [src/lot_experiments/plotting/pruning.py](../src/lot_experiments/plotting/pruning.py).
