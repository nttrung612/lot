# Figure 1 — Kernel Design Map: When Does Action Geometry Yield a Local Bellman Backup?

## 1. Role of this experiment

Figure 1 tests the first link in the paper's central empirical argument:

$$
\text{structured action geometry}
\Longrightarrow
\text{concentrated action kernels}
\Longrightarrow
\text{small local action neighborhoods}.
$$

The relevant question is not simply whether the heat kernel is sparse. On a connected graph and at positive diffusion time, the exact heat kernel is generally dense: every action can receive positive mass. The more useful question is:

> How many actions carry almost all of the mass of a kernel column, how does that number scale with the total number of actions $K$, and can that high-mass region be constructed using only local graph access?

This question must be answered before evaluating an end-to-end planner. If a kernel does not concentrate, a geometry-aware Bellman backup still needs to evaluate almost all $K$ actions. If the kernel concentrates but locating its important mass requires constructing a dense $K\times K$ matrix, the concentration does not translate into a local algorithm.

Figure 1 therefore isolates the kernel from rewards, environment transitions, and planning depth. This controlled design prevents properties of a particular MDP from obscuring the mechanism under study.

The experiment addresses three questions:

1. **Scaling:** Does the effective support remain nearly constant as $K$ grows?
2. **Geometric dependence:** At the same $K$, how does concentration change across paths, grids, tori, and expanders?
3. **Algorithmic realizability:** Can a concentrated column be constructed from local adjacency access rather than global dense computation?

Figure 1 is a mechanism study at the kernel level. It does not measure value error, policy error, simple regret, or policy return. Those questions belong to the pruning and planning experiments that follow.

## 2. Experimental object and notation

Let $G=(\mathcal A,E)$ be an undirected action graph with $K=|\mathcal A|$ actions and graph Laplacian $L$. The action graph is distinct from the state-transition graph of an MDP.

Throughout the implementation:

- $j$ indexes an anchor action;
- $i$ indexes a candidate action;
- $w[i,j]$ is the probability assigned to candidate $i$ conditional on anchor $j$;
- columns, rather than rows, are probability distributions:

$$
w[i,j]=\Pr(i\mid j),
\qquad
\sum_i w[i,j]=1.
$$

### 2.1. Primary metric: kernel effective count

For a permitted omitted-mass budget $\alpha\in(0,1)$, define

$$
d_{\mathrm{ker}}(\alpha)
=
\max_j
\min_{I\subseteq\mathcal A}
\left\{
|I|:
\sum_{i\in I}w[i,j]\ge 1-\alpha
\right\}.
$$

For each anchor $j$, the implementation sorts the column weights in decreasing order and finds the smallest prefix whose cumulative mass is at least $1-\alpha$. It then takes the maximum count over all anchors. The metric is therefore a worst-anchor statistic rather than a result for a favorable interior anchor.

Two complementary measurements are also reported:

- **Effective fraction:** $d_{\mathrm{ker}}(\alpha)/K$.
- **High-mass graph-ball radius:** the smallest graph radius whose ball contains at least $1-\alpha$ of the column mass, maximized over anchors.

These measurements answer different questions. Effective count measures **how many actions** are needed. High-mass radius measures whether those actions can be **discovered by a contiguous local graph traversal**.

A small radius alone does not imply a small action set. On an expander, a graph ball can contain a large fraction of all vertices after only a few hops. The operational quantity is therefore the combination of diffusion radius and graph-ball volume growth.

### 2.2. Diffusion-scale normalization

The graph Laplacian is converted to a random-walk kernel by uniformization:

$$
P_G=I-\frac{L}{\nu_u},
\qquad
\nu_u\ge d_{\max}.
$$

The experiment sweeps

$$
\theta=\nu_u t
$$

rather than raw diffusion time $t$. Under the Poisson representation of the heat kernel, $\theta$ is the expected random-walk length. This normalization makes diffusion scales more comparable across graph families with different degrees.

## 3. Data: controlled synthetic action graphs

This experiment does not use trajectories, rewards, state transitions, or a training dataset. Its data consist of synthetic action graphs chosen to control neighborhood volume growth.

| Graph family | Structure | Scientific role |
|---|---|---|
| Path | One-dimensional graph with boundaries | Canonical slow-growth geometry; ball size grows linearly with radius |
| Grid | Two-dimensional lattice with boundaries | Tests the effect of higher geometric dimension |
| Torus | Periodic two-dimensional lattice | Separates dimensionality from boundary effects |
| Random 4-regular graph | Degree-four expander-like graph | Negative control with rapid neighborhood growth |

The requested action-space sizes are

$$
K\in\{64,128,256,512,1024,2048,4096\}.
$$

Paths and random regular graphs use exactly the requested size. Grids and tori use a square side length obtained by rounding $\sqrt K$, so the realized size can differ slightly from the requested size. For example, requested sizes 128, 512, and 2048 produce square graphs with 121, 529, and 2025 vertices. Both <code>requested_K</code> and realized <code>K</code> are recorded.

The parameter grid is

$$
\alpha\in\{10^{-1},10^{-2},10^{-3},10^{-4}\},
\qquad
\theta\in\{0.5,2,8\}.
$$

With four graph families, seven requested sizes, three diffusion scales, four mass tolerances, and four methods, the current output contains

$$
4\times7\times3\times4\times4=1344
$$

completed result rows. The experiment seed is 0. The seed affects random regular graph generation; the other graph families are deterministic.

## 4. Compared kernels

The four methods in Figure 1 do **not** all define the same target. They are placed in the same design map to compare kernel concentration and construction cost, not to treat all methods as estimators of Exact Heat.

| Method | Kernel construction | Target label | Role |
|---|---|---|---|
| Uniform MaxEnt | Uniform probability over all actions | <code>maxent</code> | Geometry-free reference |
| Exact Heat | Matrix exponential of the graph Laplacian | <code>exact_heat</code> | Exact dense heat reference |
| Truncated Local Heat | Normalized finite Poisson mixture of local random walks | <code>truncated_heat</code> | Certified graph-local heat approximation |
| Diffusion-distance Gibbs | Gibbs weights derived from diffusion distance | <code>diffusion_gibbs</code> | Different geometry-aware LOT target |

### 4.1. Uniform MaxEnt

Uniform MaxEnt ignores the action graph:

$$
w[i,j]=\frac{1}{K}.
$$

Its effective count is available analytically:

$$
d_{\mathrm{ker}}(\alpha)
=
\left\lceil(1-\alpha)K\right\rceil.
$$

It represents a regularizer that does not prefer candidates near the anchor.

### 4.2. Exact Heat

The exact heat kernel is

$$
H_t=\exp(-tL).
$$

The implementation computes columns using batched sparse <code>expm_multiply</code>. This avoids applying a dense matrix exponential directly, but the analysis still materializes a dense $K\times K$ output in order to compute worst-anchor diagnostics.

Exact Heat has global mathematical support on the connected graphs used here. Its role is to determine whether a dense kernel can nevertheless have a small high-mass core.

### 4.3. Truncated Local Heat

Uniformization gives the exact expansion

$$
H_t
=
\sum_{k=0}^{\infty}
e^{-\theta}\frac{\theta^k}{k!}P_G^k.
$$

Truncated Local Heat retains walk lengths from zero through $r$ and renormalizes the retained Poisson head:

$$
H_{t,r}
=
\frac{1}{s_r}
\sum_{k=0}^{r}
e^{-\theta}\frac{\theta^k}{k!}P_G^k,
\qquad
s_r=\Pr(N\le r),
\quad
N\sim\operatorname{Pois}(\theta).
$$

The radius is the smallest integer satisfying

$$
\beta_r
=
\Pr(N>r)
\le
\alpha.
$$

The resulting approximation obeys the column-wise certificate

$$
\left\|
H_t(:,j)-H_{t,r}(:,j)
\right\|_1
\le
2\beta_r.
$$

The local implementation never forms the dense heat matrix. Starting at anchor $j$, it explores the graph only out to radius $r$, constructs the required random-walk powers locally, and combines them using normalized Poisson weights. Its support is contained exactly within the $r$-hop graph ball.

The current configuration uses the same numerical value $\alpha$ for two purposes:

1. the omitted-mass tolerance in $d_{\mathrm{ker}}(\alpha)$;
2. the upper budget used to select a Poisson radius.

The concepts remain distinct. The first controls how much kernel mass may be omitted when measuring concentration. The actual omitted Poisson mass is $\beta_r$.

### 4.4. Diffusion-distance Gibbs

For a symmetric heat kernel, squared diffusion distance is computed as

$$
d_t^2(i,j)
=
H_{2t}(i,i)+H_{2t}(j,j)-2H_{2t}(i,j).
$$

The Gibbs kernel is then

$$
w[i,j]
=
\frac{
\exp\left[-d_t^2(i,j)/\lambda\right]
}{
\sum_{i'}
\exp\left[-d_t^2(i',j)/\lambda\right]
},
\qquad
\lambda=1.
$$

This method is a useful control because it uses graph geometry without using a heat column directly as the prior. At the current cost scale and $\lambda=1$, its weights are close to uniform. This does not establish a universal property of diffusion-distance Gibbs kernels. It shows that geometry-aware regularization does not automatically produce locality; concentration also depends on how geometry enters the objective and on the relative temperature or cost scale.

## 5. Measurement protocol and fairness

For every graph, size, diffusion scale, and method, kernel statistics are computed over **all anchors**. Effective count, high-mass radius, and maximum exact-versus-truncated error are therefore worst-anchor diagnostics.

One-column construction time is measured at representative anchor

$$
j=\left\lfloor\frac{K}{2}\right\rfloor.
$$

The timed operations are:

- **Exact Heat:** construct one column with sparse <code>expm_multiply</code>;
- **Truncated Local Heat:** cold local construction using adjacency access;
- **Diffusion-distance Gibbs:** construct the required heat column at $2t$, convert it to diffusion costs, and normalize Gibbs weights;
- **Uniform MaxEnt:** allocate a uniform vector of length $K$.

Each timing is the median of three repetitions. The runner forces BLAS libraries to one thread, including OMP, OpenBLAS, MKL, VECLIB, and NUMEXPR. This prevents hidden dense linear-algebra parallelism from making the comparison hardware-dependent in an uncontrolled way.

Geometry preprocessing time and cached-column time are recorded separately. Panel D plots only uncached one-column construction time.

The memory values are representation-level estimates:

$$
\text{dense column}=8K\text{ bytes},
$$

$$
\text{dense full matrix}=8K^2\text{ bytes},
$$

and a sparse local column is estimated at approximately 16 bytes per nonzero, accounting for one index and one floating-point value. These estimates do not include Python object or allocator overhead.

The raw output records target labels, resolved configuration, git commit, package versions, hardware, and thread settings. The current data snapshot was produced at commit <code>78fb2606b8af234eb31d91933390c1b28b00a742</code> with Python 3.13.12, NumPy 2.5.3, and SciPy 1.18.1 on a macOS ARM machine reporting 10 logical CPUs. Absolute timing values are specific to that machine; scaling trends and graph-access counters are more portable evidence.

## 6. Reading Figure 1

Unless stated otherwise, the displayed panels fix

$$
\alpha=10^{-2},
\qquad
\theta=2.
$$

### 6.1. Panel A — Scaling on a path

Panel A plots $d_{\mathrm{ker}}(\alpha)$ against $K$ on log-log axes. It directly tests whether fixed-scale diffusion on a slow-growth graph reaches a neighborhood whose size is independent of the total action-space size.

| $K$ | Exact Heat | Truncated Heat | Uniform MaxEnt | Diffusion Gibbs |
|---:|---:|---:|---:|---:|
| 64 | 9 | 8 | 64 | 64 |
| 1024 | 9 | 8 | 1014 | 1014 |
| 4096 | 9 | 8 | 4056 | 4055 |

Exact Heat has effective count 9 across the entire range from $K=64$ to $K=4096$. Truncated Heat remains at 8. Uniform MaxEnt and Diffusion Gibbs grow almost linearly with $K$.

At $K=4096$, Exact Heat needs only

$$
\frac{9}{4096}\approx0.22\%
$$

of the action set to retain 99% of a column's mass. Uniform MaxEnt and the current Diffusion Gibbs target require approximately 99% of all actions.

This result does not mean Exact Heat is algebraically sparse. Its exact support size remains $K$. Instead, it shows that Exact Heat is **effectively local** on a path at the chosen diffusion scale. This distinction between algebraic density and computationally relevant mass is central to the paper.

Truncated Heat occasionally needs one fewer action than Exact Heat. This should not be interpreted as being “more accurate.” The normalized Poisson head removes the tail and renormalizes the retained mass, which can make the truncated target slightly more concentrated. Concentration is compared through effective counts; approximation to Exact Heat is evaluated separately through the L1 certificate.

### 6.2. Panel B — Mass tolerance at $K=4096$

Panel B increases required mass coverage from 90% to 99.99%. The word “Accuracy” in the current plot title refers to retained-mass accuracy, not value-estimation accuracy. A less ambiguous camera-ready title would be “Mass tolerance at $K=4096$.”

| $\alpha$ | Retained mass | Exact Heat | Truncated Heat | Uniform MaxEnt | Diffusion Gibbs |
|---:|---:|---:|---:|---:|---:|
| $10^{-1}$ | 90% | 5 | 5 | 3687 | 3686 |
| $10^{-2}$ | 99% | 9 | 8 | 4056 | 4055 |
| $10^{-3}$ | 99.9% | 11 | 11 | 4092 | 4092 |
| $10^{-4}$ | 99.99% | 13 | 13 | 4096 | 4096 |

Reducing $\alpha$ by a factor of 1000 increases the Exact Heat effective count only from 5 to 13. Truncated Heat closely follows the same pattern. The two diffuse reference targets rapidly approach the full set of 4096 actions.

The certified truncation radius increases moderately:

- $r=4$ for $\alpha=10^{-1}$;
- $r=6$ for $\alpha=10^{-2}$;
- $r=8$ for $\alpha=10^{-3}$;
- $r=9$ for $\alpha=10^{-4}$.

At $\alpha=10^{-2}$,

$$
\beta_r=0.0045338,
\qquad
2\beta_r=0.0090676.
$$

The maximum observed column L1 error on the $K=4096$ path is 0.0032296, below the certified upper bound.

The scientific point is that demanding substantially greater mass coverage enlarges the heat neighborhood gradually on a path rather than immediately destroying locality.

### 6.3. Panel C — Geometry control near $K=1024$

Panel C fixes the action-space size, diffusion scale, and mass tolerance, then changes only the action geometry. It is the main control for distinguishing a geometric mechanism from an artifact of $K$ or the implementation.

| Graph | Exact Heat fraction | Truncated Heat fraction | Uniform fraction | Diffusion Gibbs fraction |
|---|---:|---:|---:|---:|
| Path | 0.0088 (9 actions) | 0.0078 (8 actions) | 0.9902 | 0.9902 |
| Grid | 0.0352 (36 actions) | 0.0352 (36 actions) | 0.9902 | 0.9893 |
| Torus | 0.0352 (36 actions) | 0.0352 (36 actions) | 0.9902 | 0.9902 |
| Random 4-regular | 0.2461 (252 actions) | 0.2090 (214 actions) | 0.9902 | 0.9902 |

Three observations drive the interpretation.

First, moving from a one-dimensional path to a two-dimensional grid or torus increases the Exact Heat effective count from 9 to 36. The diffusion radius remains small, but a two-dimensional graph ball contains more vertices.

Second, grid and torus results are nearly identical at $K=1024$. This indicates that geometric dimension and local volume growth, rather than boundary effects alone, drive the difference from the path.

Third, Exact Heat requires 252 actions on the random regular graph, and Truncated Heat requires 214. These counts are about 28 and 27 times their path counterparts. The Exact Heat high-mass radius on the random regular graph is only five hops, yet a five-hop expander neighborhood already contains many vertices.

This is the key negative control:

> A small diffusion radius produces a small computational branching factor only when graph balls themselves grow slowly.

The expander result prevents an unjustified universal claim that local heat planning has cost independent of $K$ on every action graph.

### 6.4. Panel D — One-column construction

Panel D measures cold, uncached construction time for one kernel column on a path.

| $K$ | Exact Heat | Truncated Heat | Uniform | Diffusion Gibbs |
|---:|---:|---:|---:|---:|
| 64 | 0.399 ms | 0.113 ms | 0.00075 ms | 0.349 ms |
| 1024 | 0.497 ms | 0.118 ms | 0.00096 ms | 0.512 ms |
| 4096 | 0.985 ms | 0.124 ms | 0.00117 ms | 1.099 ms |

Truncated local construction is nearly flat as $K$ grows. At $\theta=2$ and $\alpha=10^{-2}$, the certified radius is always 6, and an interior path column has only 13 support entries.

At $K=4096$, Truncated Heat is approximately:

$$
\frac{0.985}{0.124}\approx7.9
$$

times faster than Exact Heat column construction, and

$$
\frac{1.099}{0.124}\approx8.8
$$

times faster than Diffusion Gibbs column construction on the benchmark machine.

Uniform construction is the fastest because filling a vector with a constant is cheap. This does not imply that a Uniform MaxEnt planning backup is cheapest. Retaining 99% of its prior mass still requires 4056 action values. Panel D measures kernel construction, whereas Panels A and C characterize the number of downstream action entries carrying the relevant mass. The panels must therefore be interpreted together.

At $K=4096$, the estimated dense full-matrix storage is

$$
134{,}217{,}728\text{ bytes}=128\text{ MiB}.
$$

The corresponding sparse Truncated Heat estimate is 851,968 bytes, approximately 0.81 MiB, or about 158 times smaller. The actual local planner does not materialize even this full sparse matrix; it constructs and caches columns lazily.

### 6.5. Additional diagnostic: sensitivity to diffusion scale

Although the displayed panels fix $\theta=2$, the raw experiment sweeps $\theta\in\{0.5,2,8\}$. On a path with $K=1024$ and $\alpha=10^{-2}$:

| $\theta$ | Certified radius $r$ | Exact Heat count | Truncated Heat count |
|---:|---:|---:|---:|
| 0.5 | 3 | 5 | 5 |
| 2 | 6 | 9 | 8 |
| 8 | 15 | 15 | 15 |

As diffusion time increases, the kernel spreads farther and the required local budget increases. Even at $\theta=8$, however, 99% of the path-kernel mass requires only 15 of 1024 actions.

This diagnostic clarifies the claim's scope: locality is jointly determined by graph geometry, diffusion scale, and the requested mass tolerance.

## 7. Numerical checks supporting the result

The conclusion is not based only on the visual shape of the curves. The pipeline checks the relevant numerical invariants:

- every kernel column is nonnegative and column-stochastic;
- Exact Heat is the reference for heat approximation error;
- Truncated Heat support lies within the correct $r$-hop graph ball;
- local construction uses adjacency access and does not form a dense heat matrix;
- all 336 Truncated Heat rows in the sweep satisfy

$$
\left\|
H_t(:,j)-H_{t,r}(:,j)
\right\|_1
\le
2\beta_r;
$$

- the largest observed ratio between L1 error and its certificate over the full sweep is approximately 0.873;
- every output row carries an explicit target label.

At $K=1024$, $\alpha=10^{-2}$, and $\theta=2$, the representative local path column accesses 13 vertices and 22 neighbor entries, compared with dense diagnostics that cover all 1024 vertices.

On the random regular graph, the same local construction accesses 783 vertices and 1484 neighbor entries at the representative anchor. These counters confirm the proposed mechanism: the algorithm adapts to local graph volume and consequently loses much of its advantage on an expander.

## 8. What action geometry buys

### 8.1. Geometry changes effective dimension, not nominal dimension

The action space still contains $K$ actions. Geometry changes how kernel mass is distributed across those actions. With finite diffusion scale and slow graph-volume growth, almost all mass lies in a small neighborhood. A planner can then replace the nominal branching factor $K$ with an effective local branching factor $d_{\mathrm{ker}}(\alpha)$.

Panel A gives the clearest evidence. Increasing the action-space size by a factor of 64, from 64 to 4096, does not increase the Exact Heat effective count on a path. Consequently, the potential computational advantage grows as the action discretization becomes finer.

### 8.2. Algebraically dense does not mean computationally global

Counting exact nonzeros would incorrectly classify Exact Heat as entirely nonlocal. Effective count reveals that a dense kernel can have a very small high-mass core.

The two heat constructions play complementary roles:

- Exact Heat establishes that mass concentration exists.
- Truncated Local Heat makes that concentration computationally accessible.
- The Poisson-tail certificate quantifies the cost of replacing exact dense heat with a local finite-walk approximation.

This distinction is essential. Concentration is a property of the target distribution; local construction is an algorithmic property. Figure 1 measures both.

### 8.3. Volume growth is the hidden geometric condition

The path-grid-expander comparison shows that diffusion radius alone does not determine cost. Local work is governed approximately by

$$
\text{local work}
\approx
|B(j,r)|
+
\text{adjacency accesses within }B(j,r).
$$

On a path, $|B(j,r)|=O(r)$. On a two-dimensional grid, it is $O(r^2)$. On an expander, graph-ball volume can grow nearly exponentially until it saturates at $K$.

Any theoretical or empirical claim of low-cost local planning must therefore state a volume-growth or geometric-regularity condition. Finite diffusion time alone is insufficient.

### 8.4. “Geometry-aware” is not a binary property

Diffusion-distance Gibbs uses graph geometry, but at $\lambda=1$ and the current cost scale its kernel is nearly uniform. Geometry can enter a regularizer through a direct heat prior, a transport cost, or another transformation, and those choices define different objectives and computational profiles.

Figure 1 must therefore not be read as ranking Diffusion Gibbs below Exact Heat under a common objective. It demonstrates that a distinct geometry-aware target does not necessarily translate geometry into local branching at the tested scale.

### 8.5. Figure 1 establishes a necessary mechanism, not the final planning claim

Kernel concentration shows that a local planner has the opportunity to reduce action evaluations. It does not prove that the exact-heat Bellman value remains accurate after repeated backups.

The end-to-end claim requires two additional steps:

1. control pruning and heat-truncation error at the level of one Bellman backup;
2. propagate that error through finite-horizon or fixed-point planning and compare against <code>FullExactHeat</code>.

Figure 1 should therefore be presented as identifying the mechanism and its operating regime. Later experiments establish backup-level sharpness and planning-level target accuracy.

## 9. Claims that Figure 1 does not support

Figure 1 should not be used to claim:

- improved policy return, simple regret, or best-action identification;
- that Truncated Heat and Exact Heat are identical targets;
- that Uniform MaxEnt or Diffusion Gibbs estimates the Exact Heat target;
- that one-column runtime equals full-planner runtime;
- that local heat complexity is independent of $K$ on every graph;
- that absolute timings transfer directly to other hardware or thread settings;
- that three timing repetitions constitute statistical replication of graph behavior;
- that one random graph seed characterizes all random regular graphs;
- that the effective-mass budget $\alpha$ and actual Poisson tail $\beta_r$ are the same quantity.

## 10. Threats to validity and recommended refinements

1. **One random graph realization.** Seed 0 provides reproducibility but not uncertainty across random regular graphs. A headline camera-ready result should use multiple graph seeds and report median and interquartile range.

2. **Only three timing repetitions.** This is adequate for a diagnostic run, but sub-millisecond measurements are sensitive to operating-system noise. A final benchmark should include warm-up, more repetitions, and timing dispersion.

3. **Ambiguous Panel B title.** “Accuracy at K=4096” can be mistaken for value-estimation accuracy. “Mass tolerance at K=4096” or “Concentration versus retained mass” would be clearer.

4. **Only one Gibbs scale.** The near-uniform Diffusion Gibbs result is specific to $\lambda=1$ and the current cost scale. A supplementary sweep over $\lambda$ would reveal the transition between concentrated and diffuse regimes.

5. **Kernel timing excludes downstream value access.** Planning experiments must separately report action-value evaluations, transition-oracle calls, graph-neighbor accesses, online time, and preprocessing time.

6. **Rounded grid sizes.** Captions should say “near $K=1024$” for general cross-family comparisons, even though the displayed 1024-point happens to have exactly 1024 vertices in every family.

## 11. Paper-ready Results paragraph

> **Kernel geometry determines the effective action dimension.** Figure 1 isolates kernel concentration from downstream MDP effects. On the path graph at fixed diffusion scale $\theta=2$, the number of actions required to retain 99% of a heat column remains constant as the action space grows: Exact Heat uses 9 actions and Truncated Heat uses 8 across $K=64$–$4096$, whereas Uniform MaxEnt and the $\lambda=1$ diffusion-distance Gibbs target require approximately $0.99K$ actions. The effect depends on graph volume growth. At $K=1024$, Exact Heat retains 99% mass with 9 actions on the path, 36 on the grid and torus, but 252 on a random 4-regular graph. Thus, a small diffusion radius yields small computational branching only when graph balls themselves grow slowly. The normalized Poisson-head construction turns this concentration into a local algorithm: at $K=4096$ and $\alpha=10^{-2}$, it uses a 13-action support, constructs an uncached column in 0.124 ms, and satisfies the exact-heat certificate $\|H_t(:,j)-H_{t,r}(:,j)\|_1\le2\beta_r$. These comparisons characterize concentration and construction cost across distinct regularization targets; they do not treat MaxEnt or diffusion-distance Gibbs as estimators of Exact Heat.

## 12. Proposed Figure 1 caption

> **Figure 1: Action geometry controls kernel concentration and the cost of local construction.** (A) On a path with omitted-mass tolerance $\alpha=10^{-2}$ and Poisson mean $\theta=2$, the 99%-mass effective counts of Exact Heat and Truncated Heat remain constant as $K$ grows, whereas Uniform MaxEnt and diffusion-distance Gibbs with $\lambda=1$ scale linearly with $K$. (B) At $K=4096$, increasingly strict mass tolerances enlarge the heat neighborhood only gradually. (C) At $K=1024$, heat remains strongly concentrated on slow-growth path and grid geometries but becomes substantially less local on a random 4-regular graph, demonstrating the role of graph-ball volume growth. (D) Lazy Poisson-head construction has nearly $K$-independent one-column time on the path at fixed radius. The methods define different kernel targets; the panels compare concentration and construction cost, not common-target value error. Truncated Heat is related to Exact Heat through the certified bound $\|H_t(:,j)-H_{t,r}(:,j)\|_1\le2\beta_r$.

## 13. Reproducing the experiment

Run the kernel-map experiment:

~~~bash
uv run python scripts/run_kernel_map.py --config configs/kernel_map.yaml
~~~

Regenerate figures from existing summaries without rerunning the experiment:

~~~bash
uv run python scripts/make_all_figures.py --kernel-map-config configs/kernel_map.yaml
~~~

Relevant artifacts:

- configuration: [configs/kernel_map.yaml](../configs/kernel_map.yaml);
- raw run-level data: [outputs/raw/kernel_map.parquet](../outputs/raw/kernel_map.parquet);
- compact summary: [outputs/summaries/kernel_map.csv](../outputs/summaries/kernel_map.csv);
- PNG figure: [outputs/figures/figure1_kernel_map.png](../outputs/figures/figure1_kernel_map.png);
- PDF figure: [outputs/figures/figure1_kernel_map.pdf](../outputs/figures/figure1_kernel_map.pdf);
- experiment runner: [src/lot_experiments/kernel_map.py](../src/lot_experiments/kernel_map.py);
- plotting code: [src/lot_experiments/plotting/kernel_map.py](../src/lot_experiments/plotting/kernel_map.py).
