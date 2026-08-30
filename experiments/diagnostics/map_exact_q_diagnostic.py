"""
Exact-Q MAP diagnostic for Figures 8/9.

Purpose
-------
Remove DQN/comparator-training approximation from MAP and isolate the two
argmax ambiguities that the paper does not specify:

1) model tie: argmax_theta posterior(theta)
2) action tie: argmax_a q*_theta(s,a)

The script:
- computes exact model-specific q* tables by discounted value iteration;
- sweeps MAP model-tie rules;
- sweeps deterministic action-order tie conventions and random action ties;
- evaluates every one of the 27 initially tied models;
- writes per-step traces for representative rollouts;
- plots an envelope across fixed-initial MAP choices.

It does not train or overwrite any DQN checkpoint.

Example:
    python map_exact_q_diagnostic.py \
      --figure 9 \
      --project-file ../../src/final_bayesian_navigation.py \
      --trials 1000 --horizon 50 \
      --output-dir ../../results/diagnostics/map_exact_q/figure9
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


REPORT_STEPS = (10, 20, 30, 40, 50)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--figure", type=int, choices=(8, 9), required=True)
    p.add_argument("--project-file", type=Path, required=True)
    p.add_argument("--trials", type=int, default=1000)
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--base-eval-seed", type=int, default=10000)
    p.add_argument("--map-seed", type=int, default=30001)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--vi-tol", type=float, default=1e-12)
    p.add_argument("--vi-max-iter", type=int, default=10000)
    p.add_argument("--tie-atol", type=float, default=1e-10)
    p.add_argument("--trace-seed", type=int, default=10000)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def load_project_module(path: Path) -> Any:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("bayesian_navigation_project", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def model_label(model: Iterable[Any]) -> str:
    return "".join(cell.value for cell in model)


def exact_q_bank(nav: Any, scenario: Any, gamma: float, tol: float, max_iter: int):
    env = nav.MazeEnvironment(scenario.config)
    all_q = []
    iterations = []
    residuals = []
    states = None

    for model_index, model in enumerate(scenario.models):
        mdp = nav.ModelSpecificMDP(env, model, scenario.config.unknown_cells)
        if states is None:
            states = mdp.states
        elif mdp.states != states:
            raise RuntimeError("Model-specific MDP state order differs across models.")

        kernel = nav.build_model_kernel(mdp)
        n_states = len(mdp.states)
        n_actions = len(nav.Action)
        v = np.zeros(n_states, dtype=np.float64)
        q = np.zeros((n_states, n_actions), dtype=np.float64)

        residual = np.inf
        for iteration in range(1, max_iter + 1):
            for s_idx in range(n_states):
                for a_idx in range(n_actions):
                    total = 0.0
                    for next_idx, probability, reward in kernel[s_idx][a_idx]:
                        total += probability * (reward + gamma * v[next_idx])
                    q[s_idx, a_idx] = total
            v_new = np.max(q, axis=1)
            residual = float(np.max(np.abs(v_new - v)))
            v = v_new
            if residual <= tol:
                break
        else:
            raise RuntimeError(
                f"Value iteration did not converge for model {model_index}; residual={residual}"
            )

        # Recompute Q once from the converged V, so Q and V are synchronized.
        for s_idx in range(n_states):
            for a_idx in range(n_actions):
                q[s_idx, a_idx] = sum(
                    probability * (reward + gamma * v[next_idx])
                    for next_idx, probability, reward in kernel[s_idx][a_idx]
                )

        all_q.append(q.copy())
        iterations.append(iteration)
        residuals.append(residual)
        print(
            f"VI model={model_index:02d} {model_label(model)} "
            f"iterations={iteration} residual={residual:.3e}"
        )

    assert states is not None
    q_values = np.stack(all_q, axis=0)
    config = nav.ComparatorConfig(episodes=0, horizon=0, alpha=0.0, gamma=gamma, epsilon=0.0)
    bank = nav.ComparatorBank(
        models=scenario.models,
        states=states,
        q_values=q_values,
        visit_counts=np.zeros_like(q_values, dtype=np.int64),
        model_seeds=np.full(len(scenario.models), -1, dtype=np.int64),
        config=config,
    )
    return bank, np.asarray(iterations), np.asarray(residuals)


def tied_indices(values: np.ndarray, atol: float) -> np.ndarray:
    maximum = float(np.max(values))
    return np.flatnonzero(np.isclose(values, maximum, rtol=0.0, atol=atol))


class DiagnosticMAPPolicy:
    def __init__(
        self,
        nav: Any,
        bank: Any,
        model_tie_rule: str,
        action_tie_rule: str,
        base_seed: int,
        tie_atol: float,
        action_order: tuple[int, ...] | None = None,
        fixed_initial_model: int | None = None,
    ) -> None:
        valid_model = {"first", "sticky-random", "random-each-step", "fixed-initial"}
        valid_action = {"first", "random", "ordered"}
        if model_tie_rule not in valid_model:
            raise ValueError(model_tie_rule)
        if action_tie_rule not in valid_action:
            raise ValueError(action_tie_rule)
        if model_tie_rule == "fixed-initial" and fixed_initial_model is None:
            raise ValueError("fixed-initial requires fixed_initial_model")
        if action_tie_rule == "ordered" and action_order is None:
            raise ValueError("ordered requires action_order")

        self.nav = nav
        self.bank = bank
        self.model_tie_rule = model_tie_rule
        self.action_tie_rule = action_tie_rule
        self.base_seed = int(base_seed)
        self.tie_atol = float(tie_atol)
        self.action_order = action_order
        self.fixed_initial_model = fixed_initial_model
        self.rng = np.random.default_rng(base_seed)
        self.selected_index: int | None = None
        self._fixed_was_used = False

    def reset(self, trial_seed: int) -> None:
        self.rng = np.random.default_rng(self.base_seed + int(trial_seed))
        self.selected_index = None
        self._fixed_was_used = False

    def choose_model(self, posterior: np.ndarray) -> tuple[int, np.ndarray]:
        candidates = tied_indices(np.asarray(posterior), self.tie_atol)

        if self.model_tie_rule == "first":
            idx = int(candidates[0])
        elif self.model_tie_rule == "random-each-step":
            idx = int(self.rng.choice(candidates))
        elif self.model_tie_rule == "sticky-random":
            if self.selected_index is None or self.selected_index not in candidates:
                self.selected_index = int(self.rng.choice(candidates))
            idx = int(self.selected_index)
        else:  # fixed-initial
            fixed = int(self.fixed_initial_model)
            if fixed in candidates and (not self._fixed_was_used or self.selected_index == fixed):
                self.selected_index = fixed
                self._fixed_was_used = True
            elif self.selected_index is not None and self.selected_index in candidates:
                pass
            else:
                # deterministic fallback: lowest-index surviving maximizer.
                self.selected_index = int(candidates[0])
            idx = int(self.selected_index)

        return idx, candidates

    def choose_action(self, q_values: np.ndarray) -> tuple[int, np.ndarray]:
        candidates = tied_indices(np.asarray(q_values), self.tie_atol)
        if self.action_tie_rule == "first":
            return int(candidates[0]), candidates
        if self.action_tie_rule == "random":
            return int(self.rng.choice(candidates)), candidates

        assert self.action_order is not None
        candidate_set = set(int(x) for x in candidates)
        for action in self.action_order:
            if int(action) in candidate_set:
                return int(action), candidates
        raise RuntimeError("No ordered action candidate found")

    def decision(self, state: Any):
        model_index, model_candidates = self.choose_model(state.model_probabilities)
        q_values = self.bank.values_for_model(model_index, state)
        action, action_candidates = self.choose_action(q_values)
        return action, model_index, model_candidates, np.asarray(q_values), action_candidates

    def select_action(self, state: Any) -> int:
        action, *_ = self.decision(state)
        return int(action)


def evaluate(nav: Any, scenario: Any, policy: Any, trials: int, horizon: int, base_seed: int):
    result = nav.evaluate_policy(scenario, policy, trials, horizon, base_seed)
    return np.asarray(result.mean), np.asarray(result.ci95), np.asarray(result.cumulative_injuries)


def report_points(mean: np.ndarray) -> str:
    return ", ".join(
        f"{step}:{mean[step-1]:.3f}" for step in REPORT_STEPS if step <= len(mean)
    )


def write_trace(nav: Any, scenario: Any, policy: DiagnosticMAPPolicy, horizon: int, seed: int, path: Path):
    simulator = nav.TrueEnvironmentSimulator(
        nav.MazeEnvironment(scenario.config),
        scenario.models,
        scenario.true_model,
        scenario.config.unknown_cells,
    )
    simulator.reseed(seed)
    policy.reset(seed)
    state = simulator.initial_state(scenario.config.start, scenario.model_prior)

    rows = []
    for step in range(horizon):
        action, model_index, model_candidates, q_values, action_candidates = policy.decision(state)
        posterior = np.asarray(state.model_probabilities)
        outcome = simulator.step(state, action)
        obs = outcome.observation
        rows.append(
            {
                "step": step + 1,
                "position_before": str(state.position),
                "eta_before": "".join(str(int(x)) for x in state.eta),
                "posterior_max": float(np.max(posterior)),
                "model_tie_count": int(len(model_candidates)),
                "model_index": int(model_index),
                "model_label": model_label(scenario.models[model_index]),
                "q_up": float(q_values[int(nav.Action.UP)]),
                "q_down": float(q_values[int(nav.Action.DOWN)]),
                "q_left": float(q_values[int(nav.Action.LEFT)]),
                "q_right": float(q_values[int(nav.Action.RIGHT)]),
                "action_tie_count": int(len(action_candidates)),
                "action": nav.Action(action).name,
                "observation_next_position": str(obs.next_position),
                "observed_cell": "" if obs.observed_cell is None else str(obs.observed_cell),
                "observed_type": "" if obs.observed_type is None else obs.observed_type.value,
                "reward": float(outcome.reward),
                "located_after": int(outcome.next_state.located_injury_count),
            }
        )
        state = outcome.next_state

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.trials <= 0 or args.horizon <= 0:
        raise ValueError("trials/horizon must be positive")

    nav = load_project_module(args.project_file)
    scenario = nav.create_scenario(args.figure)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print(f"Figure {args.figure}: exact-Q MAP diagnostic")
    print("=" * 90)
    bank, vi_iterations, vi_residuals = exact_q_bank(
        nav, scenario, args.gamma, args.vi_tol, args.vi_max_iter
    )

    np.savez_compressed(
        args.output_dir / "exact_q_bank.npz",
        q_values=bank.q_values,
        model_labels=np.asarray([model_label(m) for m in scenario.models]),
        vi_iterations=vi_iterations,
        vi_residuals=vi_residuals,
        gamma=np.asarray(args.gamma),
    )

    # 1) High-level model tie rules under literal action argmax.
    summary_rows = []
    curves = {}
    for model_rule in ("first", "sticky-random", "random-each-step"):
        policy = DiagnosticMAPPolicy(
            nav, bank, model_rule, "first", args.map_seed, args.tie_atol
        )
        mean, ci, cumulative = evaluate(
            nav, scenario, policy, args.trials, args.horizon, args.base_eval_seed
        )
        key = f"model={model_rule}|action=first"
        curves[key] = mean
        summary_rows.append({
            "family": "model_tie_rule",
            "label": key,
            "final_mean": float(mean[-1]),
            "final_ci95": float(ci[-1]),
            **{f"step_{s}": float(mean[s-1]) for s in REPORT_STEPS if s <= args.horizon},
        })
        print(key, "|", report_points(mean))

    # 2) Fixed initial model sweep (27 initial MAP maximizers), deterministic fallback/action tie.
    fixed_curves = []
    fixed_rows = []
    for idx, model in enumerate(scenario.models):
        policy = DiagnosticMAPPolicy(
            nav, bank, "fixed-initial", "first", args.map_seed, args.tie_atol,
            fixed_initial_model=idx,
        )
        mean, ci, cumulative = evaluate(
            nav, scenario, policy, args.trials, args.horizon, args.base_eval_seed
        )
        fixed_curves.append(mean)
        fixed_rows.append({
            "model_index": idx,
            "model_label": model_label(model),
            "final_mean": float(mean[-1]),
            "final_ci95": float(ci[-1]),
            **{f"step_{s}": float(mean[s-1]) for s in REPORT_STEPS if s <= args.horizon},
        })
        print(f"fixed={idx:02d} {model_label(model)} | {report_points(mean)}")

    fixed_curves_arr = np.stack(fixed_curves)
    with (args.output_dir / "fixed_initial_model_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fixed_rows[0].keys()))
        writer.writeheader(); writer.writerows(fixed_rows)

    # 3) Action-tie sensitivity using several deterministic priority permutations + random ties.
    # Keep model tie literal-first here so this sweep isolates action tie semantics.
    action_orders = [
        (0, 1, 2, 3),  # U D L R (project enum order)
        (3, 2, 1, 0),  # R L D U
        (2, 3, 0, 1),  # L R U D
        (1, 0, 3, 2),  # D U R L
        (0, 2, 1, 3),  # U L D R
        (2, 0, 3, 1),  # L U R D
    ]
    action_rows = []
    for order in action_orders:
        policy = DiagnosticMAPPolicy(
            nav, bank, "first", "ordered", args.map_seed, args.tie_atol,
            action_order=order,
        )
        mean, ci, cumulative = evaluate(
            nav, scenario, policy, args.trials, args.horizon, args.base_eval_seed
        )
        label = "-".join(nav.Action(a).name for a in order)
        action_rows.append({
            "action_tie_rule": "ordered",
            "priority": label,
            "final_mean": float(mean[-1]),
            "final_ci95": float(ci[-1]),
            **{f"step_{s}": float(mean[s-1]) for s in REPORT_STEPS if s <= args.horizon},
        })
        curves[f"model=first|action={label}"] = mean
        print(f"action-order={label} | {report_points(mean)}")

    policy = DiagnosticMAPPolicy(nav, bank, "first", "random", args.map_seed, args.tie_atol)
    mean, ci, cumulative = evaluate(nav, scenario, policy, args.trials, args.horizon, args.base_eval_seed)
    action_rows.append({
        "action_tie_rule": "random",
        "priority": "random among tied actions",
        "final_mean": float(mean[-1]),
        "final_ci95": float(ci[-1]),
        **{f"step_{s}": float(mean[s-1]) for s in REPORT_STEPS if s <= args.horizon},
    })
    curves["model=first|action=random"] = mean
    print(f"action=random | {report_points(mean)}")

    with (args.output_dir / "action_tie_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(action_rows[0].keys()))
        writer.writeheader(); writer.writerows(action_rows)

    with (args.output_dir / "model_tie_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader(); writer.writerows(summary_rows)

    # 4) MAP envelope across all 27 deterministic initial maximizers.
    steps = np.arange(1, args.horizon + 1)
    envelope_min = fixed_curves_arr.min(axis=0)
    envelope_max = fixed_curves_arr.max(axis=0)
    envelope_mean = fixed_curves_arr.mean(axis=0)

    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    ax.fill_between(steps, envelope_min, envelope_max, alpha=0.22, label="Fixed-initial MAP envelope")
    ax.plot(steps, envelope_mean, linewidth=2.0, label="Mean over 27 fixed initial models")
    for key in ("model=first|action=first", "model=sticky-random|action=first", "model=random-each-step|action=first"):
        ax.plot(steps, curves[key], linewidth=1.6, label=key)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Average Located Injuries")
    ax.set_xlim(1, args.horizon)
    ax.set_ylim(0.0, 2.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_title(f"Figure {args.figure}: Exact-Q MAP tie sensitivity")
    fig.tight_layout()
    fig.savefig(args.output_dir / "map_exact_q_envelope.png", dpi=180)
    plt.close(fig)

    np.savez_compressed(
        args.output_dir / "map_exact_q_results.npz",
        fixed_initial_curves=fixed_curves_arr,
        envelope_min=envelope_min,
        envelope_max=envelope_max,
        envelope_mean=envelope_mean,
        model_labels=np.asarray([model_label(m) for m in scenario.models]),
        report_steps=np.asarray([s for s in REPORT_STEPS if s <= args.horizon]),
    )

    # 5) Representative traces.
    trace_specs = [
        ("first_first", DiagnosticMAPPolicy(nav, bank, "first", "first", args.map_seed, args.tie_atol)),
        ("sticky_first", DiagnosticMAPPolicy(nav, bank, "sticky-random", "first", args.map_seed, args.tie_atol)),
        ("random_model_random_action", DiagnosticMAPPolicy(nav, bank, "random-each-step", "random", args.map_seed, args.tie_atol)),
    ]
    for name, policy in trace_specs:
        write_trace(nav, scenario, policy, args.horizon, args.trace_seed, args.output_dir / f"trace_{name}.csv")

    best_idx = int(np.argmax(fixed_curves_arr[:, -1]))
    worst_idx = int(np.argmin(fixed_curves_arr[:, -1]))
    for name, idx in (("best_fixed", best_idx), ("worst_fixed", worst_idx)):
        policy = DiagnosticMAPPolicy(
            nav, bank, "fixed-initial", "first", args.map_seed, args.tie_atol,
            fixed_initial_model=idx,
        )
        write_trace(nav, scenario, policy, args.horizon, args.trace_seed, args.output_dir / f"trace_{name}_{idx:02d}_{model_label(scenario.models[idx])}.csv")

    metadata = {
        "figure": args.figure,
        "true_model": model_label(scenario.true_model),
        "gamma": args.gamma,
        "vi_tol": args.vi_tol,
        "tie_atol": args.tie_atol,
        "trials": args.trials,
        "horizon": args.horizon,
        "base_eval_seed": args.base_eval_seed,
        "best_fixed_initial_model": {"index": best_idx, "label": model_label(scenario.models[best_idx]), "final": float(fixed_curves_arr[best_idx, -1])},
        "worst_fixed_initial_model": {"index": worst_idx, "label": model_label(scenario.models[worst_idx]), "final": float(fixed_curves_arr[worst_idx, -1])},
        "envelope_final": {"min": float(envelope_min[-1]), "max": float(envelope_max[-1]), "mean": float(envelope_mean[-1])},
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nDONE")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
