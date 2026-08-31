# Reproduction Report

## Bayesian Reinforcement Learning for Navigation in Unknown Environments

**Target paper:** Mohammad Alali and Mahdi Imani, *Bayesian reinforcement learning for navigation planning in unknown environments*, Frontiers in Artificial Intelligence, 2024.  
**DOI:** https://doi.org/10.3389/frai.2024.1308031

---

## 1. Scope and goals

This project is an independent reproduction and diagnostic analysis of the **injury-location experiments in Figures 8 and 9** of Alali and Imani (2024).

The original paper proposes a Bayesian planning formulation for navigation in an unknown environment. Instead of committing to a single possible environment model, the method augments the agent state with a posterior distribution over all candidate models and trains a DQN directly in this belief space.

The initial goal of this project was straightforward:

1. reconstruct the Figure 8 and Figure 9 maze environments;
2. implement the Bayesian belief-state transition and reward;
3. train the proposed belief-space DQN;
4. implement the three reported comparators:
   - known-model Baseline,
   - MAP,
   - Active Learning;
5. reproduce the reported 1,000-trial performance curves.

During reproduction, the proposed policy and known-model Baseline were reproduced closely, but the MAP and Active-Learning curves initially differed substantially from the paper. The project therefore expanded from a direct implementation into a **controlled reproducibility investigation**.

The two main questions became:

- **MAP:** can the reported MAP behavior be explained by implementation choices that are not uniquely specified by Equation (21)?
- **Active Learning:** can the near-zero Active-Learning performance in Figures 8 and 9 be recovered under reasonable implementations of Equation (22)?

The final conclusions are:

- the proposed Bayesian policy is reproduced closely;
- the known-model Baseline is reproduced;
- MAP is strongly dependent on model-level and action-level tie handling, and the reported curves can be closely recovered after explicitly resolving those ambiguities;
- the reported near-zero Active-Learning behavior could not be recovered, even after testing DQN, independently seeded DQN, tabular Q-learning, and exact model-specific optimal Q-functions.

This report documents both the successful reproduction and the diagnostic path that led to these conclusions.

---

## 2. Original formulation

### 2.1 Unknown environment models

Each unknown cell can take one of three types:

- `W`: wall,
- `E`: empty,
- `I`: injury/victim.

With $m$ unknown cells, this creates $3^m$ possible environment models. Figures 8 and 9 both contain three unknown cells, so the uncertainty set contains


$$
3^3 = 27
$$


possible maze models.

Let the posterior over these models at time $k$ be


$$
\vartheta_k =
\left[
P(\theta^\ast=\theta_1\mid\mathcal{D}_k),
\ldots,
P(\theta^\ast=\theta_{27}\mid\mathcal{D}_k)
\right],
$$


where $\mathcal{D}_k$ represents the states and actions observed up to time $k$.

The paper defines the belief state as


$$
b_k = [s_k,\vartheta_k]^T.
$$


For the injury-location task, the physical state additionally contains auxiliary binary variables that track whether each potential injury has already been found.

---

### 2.2 Bayesian posterior update

After taking action $a$, transitioning from state $s$, and observing a successor state $s'$, the model posterior is updated using Bayes' rule:


$$
\vartheta'(j)
=
\frac{
P(s' \mid s,a,\theta_j)\vartheta(j)
}{
\sum_l P(s' \mid s,a,\theta_l)\vartheta(l)
}.
$$


This posterior is part of the next belief state, so action selection can depend not only on the robot location but also on current uncertainty about the environment.

---

### 2.3 Motion model

The paper uses four discrete actions:

- up,
- down,
- left,
- right.

Motion is stochastic. The intended direction occurs with probability $0.8$, while each perpendicular direction occurs with probability $0.1$.

The reproduction uses the same $0.8/0.1/0.1$ motion model.

---

### 2.4 Injury reward

For the injury-location experiments, the model-specific reward is


$$
R_\theta(s,a,s') =
\begin{cases}
1, & \text{if a previously unlocated injury is found},\\
0, & \text{otherwise}.
\end{cases}
$$


Auxiliary variables ensure that the same injury is not rewarded repeatedly.

The evaluation metric is the **average cumulative number of located injuries** as a function of the number of environment steps.

---

## 3. Policies reproduced

### 3.1 Proposed Bayesian planning policy

The proposed method learns a Q-function over the belief space:


$$
Q^\ast(b,a)
=
\mathbb{E}
\left[
\sum_{t=0}^{\infty}
\gamma^t \tilde{R}(b_t,a_t)
\right].
$$


A DQN approximates this function and selects actions greedily at test time.

The reported network/training hyperparameters were reproduced:

| Hyperparameter | Value |
|---|---:|
| Hidden layers | 3 |
| Units per hidden layer | 128 |
| Activation | ReLU |
| Learning rate | $5\times10^{-4}$ |
| Replay capacity | $10^5$ |
| Batch size | 64 |
| Discount $\gamma$ | 0.95 |
| Epsilon | 0.1 |
| Target soft-update $\tau$ | $10^{-3}$ |
| Q-network update frequency | 4 |
| Training episodes | 5,000 |
| Training horizon | 250 |
| Evaluation horizon | 50 |
| Evaluation trials | 1,000 |

The reproduction uses a Q-network and target network with the same three-layer $128$-unit architecture.

---

### 3.2 Known-model Baseline

The Baseline assumes the true environment model is known. It therefore follows a model-specific policy for the actual test model.

This is an optimistic comparator: it represents navigation without uncertainty about which of the 27 environments is real.

---

### 3.3 MAP policy

The paper defines the MAP model by


$$
\theta_k^{MAP}
=
\arg\max_{\theta_i}
\vartheta_k(i),
$$


then selects


$$
a_k
=
\arg\max_a
q^\ast_{\theta_k^{MAP}}(s_k,a).
$$


At first sight this appears fully specified. However, Figures 8 and 9 use a uniform prior over all three cell types. Consequently, at the initial state,


$$
\vartheta_0(i)=\frac{1}{27}
\qquad \forall i.
$$


Therefore the first `argmax` is a **27-way tie**.

A second ambiguity appears when the selected model has multiple equally optimal actions. This becomes especially important for models containing no injuries, because all injury-reward Q-values can be exactly equal.

These ambiguities motivated the MAP diagnostic analysis in Section 8.

---

### 3.4 Active-Learning policy

The paper defines the one-step Active-Learning comparator as


$$
a_k
=
\arg\max_a
\mathbb{E}_{\vartheta_k}
[q^\ast_\theta(s_k,a)]
=
\arg\max_a
\sum_i
\vartheta_k(i)
q^\ast_{\theta_i}(s_k,a).
$$


Unlike MAP, this policy uses all model-specific Q-functions rather than selecting a single model.

The paper argues that this approach can be weak when the posterior remains broad, because incorrect models continue contributing to the weighted action values.

The reproduction implements Equation (22) directly.

---

## 4. Target experiments

## 4.1 Figure 8: 4×4 maze

Figure 8 uses:

- a 4×4 maze;
- three unknown cells;
- 27 possible environment models;
- uniform prior for every unknown cell:


$$
p_0^i = [1/3,1/3,1/3];
$$


- true test environment:


$$
\theta^\ast = [I,W,I].
$$


There are two injuries to locate.

The paper reports:

- Baseline reaches both injuries quickly;
- the proposed Bayesian policy first finds one injury rapidly and then gradually approaches approximately $1.5$ located injuries by step 50;
- MAP rises gradually to approximately $1.0$;
- Active Learning remains essentially at zero throughout the 50-step horizon.

---

## 4.2 Figure 9: 6×6 maze

Figure 9 uses:

- a 6×6 maze;
- seven known walls;
- 29 possible robot locations;
- three unknown cells;
- 27 possible environment models;
- uniform prior:


$$
p_0^i = [1/3,1/3,1/3];
$$


- true test environment:


$$
\theta^\ast = [W,I,I].
$$


Again, there are two injuries.

The paper reports:

- Baseline reaches two injuries rapidly;
- the proposed Bayesian policy gradually approaches the Baseline and is close to two injuries by step 50;
- MAP remains weak, reaching only roughly $0.4$;
- Active Learning remains essentially at zero.

---

## 5. Implementation

The public implementation is contained in:

```text
final_bayesian_navigation.py
```

The code includes:

- exact reconstruction of the two target maze geometries;
- deterministic enumeration of all 27 models;
- stochastic transitions;
- explicit observation likelihoods;
- Bayesian model-posterior updates;
- injury-tracking state variables;
- belief encoding;
- belief-space DQN training;
- model-specific comparator training;
- exact model-specific value iteration for diagnostic/MAP use;
- evaluation over 1,000 Monte Carlo trials;
- 95% confidence intervals;
- plotting and internal consistency checks.

The implementation intentionally keeps the reproduction and the diagnostics in the same mathematical environment. This is important: discrepancies are investigated by changing one policy assumption at a time rather than changing maze dynamics, rewards, or evaluation logic.

---

## 6. Implementation assumptions not fully determined by the paper

A reproduction inevitably requires choices for details that are not completely specified in the article.

The most important explicit assumptions in this implementation are:

### 6.1 Environment-model ordering

Models are deterministically enumerated over `W`, `E`, and `I`.

The ordering matters for any MAP implementation that resolves a posterior tie by selecting the first maximum.

---

### 6.2 Wall observations

The reproduction treats a collision with an unknown wall as an informative observation: the robot remains in place and the posterior is updated using the observed wall outcome.

The paper explicitly describes identification of empty and injury cells when visited, while wall-contact observation semantics are less explicit.

A sensitivity check that removed the special wall observation did not explain the large Active-Learning discrepancy.

---

### 6.3 MAP tie handling

Two different ties are treated separately:

1. **model posterior tie**, in
   $\arg\max_i\vartheta_k(i)$;
2. **greedy action tie**, in
   $\arg\max_a q_\theta(s,a)$.

The final MAP reproduction uses:

- model tie: first maximizer in the deterministic ordering;
- Q source: exact value iteration;
- action tie: uniform random choice among exactly tied greedy actions.

The same convention is used for both Figures 8 and 9.

---

### 6.4 Neural state representation

The neural belief-state representation uses an explicit encoding of robot state, injury-tracking variables, and the environment-model posterior.

This is an implementation choice because the paper specifies the mathematical belief state but does not provide source code for the exact numeric neural input encoding.

---

## 7. Main reproduction results

The final curated plots are stored in:

```text
results/main_results/figure8/
results/main_results/figure9/
```

### 7.1 Figure 8

![Figure 8 reproduction](../results/main_results/figure8/figure8c_all_policies.png)

At step 50, the reproduction is approximately:

| Policy | Paper | Reproduction |
|---|---:|---:|
| Baseline | ~2.0 | ~2.0 |
| Proposed Bayesian | ~1.5–1.6 | ~1.4 |
| MAP | ~1.0 | ~0.9 |
| Active Learning | ~0.0 | ~1.8 |

The proposed method reproduces the characteristic behavior of the paper: the first injury is located quickly, followed by a slower increase toward the second.

The Baseline also matches the expected rapid convergence to both injuries.

After the MAP diagnostic described below, MAP follows the same qualitative scale and endpoint as the paper.

The major unresolved difference is Active Learning.

---

### 7.2 Figure 9

![Figure 9 reproduction](../results/main_results/figure9/figure9c_all_policies.png)

At step 50, the reproduction is approximately:

| Policy | Paper | Reproduction |
|---|---:|---:|
| Baseline | ~2.0 | ~2.0 |
| Proposed Bayesian | ~1.9–2.0 | ~2.0 |
| MAP | ~0.4 | ~0.4 |
| Active Learning | ~0.0 | ~2.0 |

The proposed method closely reproduces the Figure 9 trajectory and final performance.

The final MAP implementation also closely matches the weak MAP curve reported in the paper.

Again, Active Learning is the substantial outlier.

---

## 8. MAP diagnostic analysis

The initial reproduction produced MAP curves that changed drastically when apparently minor implementation choices were changed.

This was investigated systematically rather than tuning a single run to visually match the paper.

Diagnostic scripts include:

```text
experiments/diagnostics/map_tie_sweep.py
experiments/diagnostics/map_fixed_initial_model_sweep.py
experiments/diagnostics/map_exact_q_diagnostic.py
```

---

### 8.1 Why the initial MAP action is undefined without a convention

Both target experiments use a uniform prior. At time zero,


$$
P(\theta^\ast=\theta_i)=1/27
$$


for all 27 models.

Thus Equation (21) does not uniquely select a MAP model.

If a programming language's standard `argmax` is used, the result depends on the arbitrary ordering of models in memory.

This is not a small numerical issue. Different tied models imply different walls and different injuries, so their optimal policies can point in completely different directions.

---

### 8.2 Fixed-initial-model sweep

A diagnostic forced each of the 27 tied models to be selected as the initial MAP model while holding the rest of the environment and evaluation procedure fixed.

For Figure 9, the step-50 performance spanned almost the full task range: some fixed choices produced essentially zero injuries, while others produced approximately two.

![Fixed-model sensitivity](../results/diagnostics/map/map_fixed_initial_model_sensitivity.png)

This demonstrates that the initial posterior tie alone is sufficient to produce major differences in the reported MAP curve.

---

### 8.3 Action-level ties

The model tie is not the only ambiguity.

For example, under an all-wall model (`WWW`), there are no injuries and the exact injury-reward Q-function can satisfy


$$
Q(s,\mathrm{UP})
=
Q(s,\mathrm{DOWN})
=
Q(s,\mathrm{LEFT})
=
Q(s,\mathrm{RIGHT})
=
0
$$


at relevant states.

A conventional `np.argmax` then silently converts an exact four-way tie into a preference for whichever action has the lowest array index.

Changing action ordering alone produced large changes in MAP performance.

---

### 8.4 Removing DQN approximation from the diagnosis

To determine whether this sensitivity was caused by neural approximation, exact model-specific Q-functions were computed with value iteration.

The resulting diagnostic envelope is shown below.

![Exact-Q MAP envelope](../results/diagnostics/map/map_exact_q_envelope.png)

The large sensitivity remains with exact Q-values. Therefore it is not primarily a DQN-training failure.

The diagnostic isolates the cause as the semantics of the nested `argmax` operations in MAP.

---

### 8.5 Final MAP convention

A single convention was required to work for **both** Figures 8 and 9; separate figure-specific tuning was rejected.

The final convention is:

1. choose the first model among posterior maximizers;
2. use exact model-specific $Q^\ast_\theta$ from value iteration;
3. if several actions are exactly optimal, sample uniformly from the tied actions.

Under this common convention:

- Figure 8 MAP rises toward roughly the same endpoint as the paper;
- Figure 9 MAP reaches roughly $0.4$ at step 50, closely matching the paper.

This is treated as a **conditional reproduction**: the curve is reproducible after making an implementation convention explicit that is not uniquely determined by Equation (21).

---

## 9. Active-Learning diagnostic analysis

Active Learning required a different investigation.

The initial implementation of Equation (22) was much stronger than the paper's curve. In some runs it was almost indistinguishable from the known-model Baseline.

Rather than deliberately degrading the policy, several hypotheses were tested.

The primary final diagnostic is:

```text
experiments/diagnostics/active_learning_final_diagnostic.py
```

with additional comparison code in:

```text
experiments/diagnostics/diagnose_active_vs_baseline.py
```

---

### 9.1 Hypothesis 1: correlated DQN seeds

The first comparator bank trained the model-specific DQNs using common seeds.

Because Equation (22) averages raw Q-values from 27 policies, correlated approximation errors could conceivably make their weighted vote unusually coherent.

A second Figure 9 comparator bank used distinct training seeds for the model-specific DQNs.

**Result:** the hypothesis was rejected.

Active Learning remained extremely strong. In the tested 1,000-trial Figure 9 evaluation, the Active and Baseline outcome arrays were effectively identical.

---

### 9.2 Hypothesis 2: DQN function approximation generalizes too well

DQN assigns values to states even when those exact state-action pairs were rarely observed in training. It was therefore possible that neural generalization was making Equation (22) unrealistically effective.

To test this, Active Learning was evaluated using model-specific **tabular Q-learning**.

**Result:** the hypothesis was not sufficient.

For Figure 8, tabular Active Learning was weaker than the neural implementation but still reached approximately $1.8$ injuries by step 50—far above the near-zero curve in the paper.

---

### 9.3 Hypothesis 3: learned model-specific Q-functions are inaccurate

The strongest possible control was then performed: learned model-specific Q-functions were removed entirely.

For each of the 27 models, $Q^\ast_\theta(s,a)$ was computed with exact value iteration. Equation (22) was then evaluated directly with these exact Q-functions.

**Result:** Active Learning became stronger, not weaker.

In the Figure 8 diagnostic, exact-Q Active Learning reached approximately two injuries by step 50.

This is an important observation: the discrepancy cannot be explained simply by the reproduced Q-functions being insufficiently converged.

---

### 9.4 Learned-vs-exact trajectory diagnostic

The final Figure 8 diagnostic compared tabular learned Q-values against exact optimal Q-values along the states actually visited by Active Learning.

![Learned vs exact Active Learning](../results/diagnostics/active_learning/active_learned_vs_exact.png)

The learned policy and exact policy remain broadly aligned.

Measured diagnostics included approximately:

- learned Active at step 50: **1.815** injuries;
- exact-Q Active at step 50: **2.0** injuries;
- action agreement with exact greedy actions along learned trajectories: **about 82%**;
- mean weighted-Q RMSE along trajectories: **about 0.011**;
- posterior mass placed on completely unvisited selected Q entries: **approximately 0**.

The posterior frequently remains unchanged—consistent with the qualitative weakness discussed in the paper—but this does not force the policy into the near-zero behavior of the published curve.

![Action agreement](../results/diagnostics/active_learning/action_agreement_by_step.png)

![Weighted-Q error](../results/diagnostics/active_learning/weighted_q_error_by_step.png)

---

### 9.5 Active-Learning conclusion

The near-zero Active-Learning result in Figures 8 and 9 was **not reproduced from the reported mathematical specification**.

The discrepancy survived:

- common-seed DQN comparators;
- independently seeded DQN comparators;
- tabular Q-learning comparators;
- exact model-specific value iteration;
- wall-observation sensitivity checks.

The exact-Q experiment is the strongest evidence: if Equation (22) is evaluated with optimal model-specific Q-functions under the reproduced environment, Active Learning is highly effective.

This report therefore does **not** claim that the published Active-Learning result is erroneous. The supported conclusion is narrower:

> The reported near-zero Active-Learning curves appear to depend on implementation or training details that are not sufficiently specified in the paper to recover them from Equation (22) and the reported experimental setup alone.

No hyperparameter search was performed for the purpose of artificially forcing the Active-Learning curve toward zero.

---

## 10. What was successfully reproduced

A useful distinction is between **algorithm reproduction** and **curve reproduction**.

### Proposed Bayesian policy

The core belief-space formulation, posterior update, stochastic transition model, injury reward, DQN architecture, and training protocol were reconstructed from the paper.

The resulting Figure 8 and Figure 9 curves closely match the reported scale and qualitative behavior.

**Status: reproduced closely.**

---

### Known-model Baseline

The known-environment comparator reaches both injuries rapidly, as expected and as reported.

**Status: reproduced.**

---

### MAP

The original specification leaves behavior under tied posterior models and tied optimal actions underdetermined.

Once these details are made explicit and neural approximation is removed with exact value iteration, one common convention closely recovers the Figure 8 and Figure 9 MAP curves.

**Status: conditionally reproduced; strongly implementation-sensitive.**

---

### Active Learning

Direct implementations of Equation (22) remain much stronger than reported under all tested Q-function backends.

**Status: unresolved reproduction discrepancy.**

---

## 11. Reproducibility lessons

This reproduction exposed several issues that are easy to miss when evaluating RL/planning papers only by comparing final curves.

### 11.1 `argmax` is an algorithmic choice when ties are common

Tie handling can be scientifically consequential rather than a harmless coding detail.

This is especially true when:

- the prior is deliberately uniform;
- the method explicitly chooses a single MAP model;
- some models yield degenerate reward structures.

---

### 11.2 Comparator implementation deserves the same scrutiny as the proposed method

The proposed Bayesian policy was comparatively well specified through its DQN architecture and hyperparameters.

The model-specific comparator pipeline had more degrees of freedom:

- how the 27 policies are trained;
- initialization and random seeds;
- state-action coverage;
- greedy tie behavior;
- whether dynamic programming or learned Q-values are used;
- numerical treatment of near-ties.

These choices can dominate comparator performance.

---

### 11.3 Exact small-state solutions are valuable diagnostic controls

Figures 8 and 9 are small enough that model-specific optimal Q-functions can be computed exactly.

Value iteration was therefore used not as a replacement for the paper's main Bayesian DQN, but as a **diagnostic instrument**.

This made it possible to distinguish:

- policy-definition ambiguity,
- training error,
- function-approximation error.

For MAP, the sensitivity remained under exact Q-values.

For Active Learning, exact Q-values made performance stronger.

These two controls materially changed the interpretation of the initial discrepancies.

---

## 12. Limitations of this reproduction

This project does not attempt to reproduce every experiment in the paper.

The present scope is limited to:

- Figure 8 injury-location experiment;
- Figure 9 injury-location experiment;
- the associated Proposed, Baseline, MAP, and Active-Learning policies.

The entropy-reduction experiments and other maze experiments are outside the current reproduction scope.

Additional limitations include:

- the original source code was not available as part of this reproduction;
- some low-level observation and tie-breaking semantics must therefore be inferred;
- exact paper curves are compared visually because the original raw Figure 8/9 evaluation arrays are not provided;
- DQN training is stochastic and can vary with seed;
- final diagnostic choices are documented explicitly, but they may differ from the authors' unpublished implementation.

---

## 13. Repository artifacts

### Main code

```text
final_bayesian_navigation.py
```

### Main results

```text
results/main_results/figure8/
results/main_results/figure9/
```

### Initial reproduction

```text
results/initial_reproduction/
```

This is intentionally retained to show the state of the reproduction before diagnostic corrections.

### MAP diagnostics

```text
experiments/diagnostics/map_tie_sweep.py
experiments/diagnostics/map_fixed_initial_model_sweep.py
experiments/diagnostics/map_exact_q_diagnostic.py

results/diagnostics/map/
```

### Active-Learning diagnostics

```text
experiments/diagnostics/active_learning_final_diagnostic.py
experiments/diagnostics/diagnose_active_vs_baseline.py

results/diagnostics/active_learning/
```

---

## 14. Reproduction commands

The main script contains several CLI modes. Exact paths for trained checkpoints depend on where the user stores them.

### Internal consistency checks

```bash
python final_bayesian_navigation.py \
  --figure 8 \
  --mode self-test
```

### Proposed policy training

```bash
python final_bayesian_navigation.py \
  --figure 8 \
  --mode train-proposed \
  --episodes 5000 \
  --horizon 250
```

and

```bash
python final_bayesian_navigation.py \
  --figure 9 \
  --mode train-proposed \
  --episodes 5000 \
  --horizon 250
```

### Neural model-specific comparators

```bash
python final_bayesian_navigation.py \
  --figure 8 \
  --mode train-neural-comparators \
  --comparator-episodes 5000 \
  --comparator-horizon 250
```

### Final evaluation convention

For MAP, the final reproduction uses:

```text
--map-q-backend exact
--map-tie-rule first
--map-action-tie-rule random
```

The same MAP convention is used for both figures.

---

## 15. Conclusion

The reproduction successfully reconstructs the main Bayesian navigation mechanism and closely reproduces the proposed-policy behavior in Figures 8 and 9.

More importantly, the diagnostic analysis shows that the two comparator discrepancies have different causes.

For **MAP**, the disagreement can be traced to a concrete reproducibility issue: under the uniform prior, Equation (21) contains an unresolved model tie, followed in some models by unresolved action ties. Exact value iteration confirms that the resulting performance sensitivity is intrinsic to the tie convention rather than neural approximation. Once a single explicit convention is fixed, the reported Figure 8 and Figure 9 MAP behavior can be closely recovered.

For **Active Learning**, no equivalent resolution was found. The near-zero performance reported in the paper is not produced by direct evaluation of Equation (22) under DQN, independent DQN seeds, tabular Q-learning, or exact model-specific optimal Q-functions. In particular, exact Q-functions make Active Learning more effective rather than less effective.

The outcome of this project is therefore not only a reproduction of two experimental figures. It is also a documented case study in how seemingly minor implementation details—especially comparator construction and tie semantics—can materially affect conclusions in reinforcement-learning experiments.

Future work in this repository will be separated from the reproduction study and will investigate additional robustness and methodological extensions of Bayesian navigation under model uncertainty.

---

## Reference

Alali, M., & Imani, M. (2024). Bayesian reinforcement learning for navigation planning in unknown environments. *Frontiers in Artificial Intelligence, 7*, 1308031. https://doi.org/10.3389/frai.2024.1308031
