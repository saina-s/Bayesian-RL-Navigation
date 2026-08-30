"""Final Active Learning reproduction diagnostic for Figures 8/9.

This script does NOT modify or retrain the supplied comparator checkpoint.
It compares the supplied tabular model-specific Q bank used in paper Eq. (22)
against an exact value-iteration Q* bank under the same simulator semantics.

Outputs:
  summary.json
  initial_state_model_contributions.csv
  trajectory_steps.csv
  by_step_summary.csv
  active_learned_vs_exact.png
  action_agreement_by_step.png
  zero_visit_mass_by_step.png
  weighted_q_error_by_step.png

Run from the project root, e.g.:
  python active_learning_final_diagnostic.py \
    --figure 8 \
    --project-file final_bayesian_navigation_MAP_exact.py \
    --comparator-bank old_outputs/outputs/planning/figure8_comparator_qlearning_standalone.npz \
    --trials 1000 --horizon 50 \
    --output-dir results/diagnostics/active_final/figure8_tabular
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def load_project_module(path: Path):
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Project file not found: {path}")
    name = "bayes_nav_project_for_active_diag"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import project module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    nz = p[p > 0.0]
    return float(-np.sum(nz * np.log(nz)))


def margin(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return float("nan")
    order = np.sort(values)
    return float(order[-1] - order[-2])


def optimal_actions(values: np.ndarray, atol: float = 1e-10) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    m = float(np.max(values))
    return np.flatnonzero(np.isclose(values, m, rtol=0.0, atol=atol))


def action_name(mod, action_index: int) -> str:
    return mod.Action(int(action_index)).name


def model_label(model) -> str:
    return "".join(cell.value for cell in model)


def state_index(bank, state) -> int:
    model_state = bank.model_state(state)
    return int(bank._state_to_index[model_state])


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", type=int, choices=(8, 9), required=True)
    parser.add_argument(
        "--project-file", type=Path, default=Path("final_bayesian_navigation_MAP_exact.py")
    )
    parser.add_argument("--comparator-bank", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--base-eval-seed", type=int, default=10000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exact-gamma", type=float, default=None)
    args = parser.parse_args()

    if args.trials <= 0 or args.horizon <= 0:
        raise ValueError("trials and horizon must be positive")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    mod = load_project_module(args.project_file)
    scenario = mod.create_scenario(args.figure)
    learned = mod.load_comparator_bank(args.comparator_bank)

    if tuple(learned.models) != tuple(scenario.models):
        raise ValueError("Comparator model ordering does not match selected scenario.")

    gamma = (
        float(args.exact_gamma)
        if args.exact_gamma is not None
        else float(getattr(learned.config, "gamma", 0.95))
    )

    print("Building exact Q* bank with value iteration (diagnostic only; no training/checkpoint changes)...")
    exact = mod.build_exact_comparator_bank(scenario, gamma=gamma)

    if tuple(learned.states) != tuple(exact.states):
        raise ValueError("Learned and exact banks use different state ordering.")

    # Global bank quality
    lq = np.asarray(learned.q_values, dtype=np.float64)
    eq = np.asarray(exact.q_values, dtype=np.float64)
    visits = np.asarray(learned.visit_counts, dtype=np.int64)

    global_rmse = float(np.sqrt(np.mean((lq - eq) ** 2)))
    visited_mask = visits > 0
    unvisited_mask = ~visited_mask
    visited_rmse = (
        float(np.sqrt(np.mean((lq[visited_mask] - eq[visited_mask]) ** 2)))
        if np.any(visited_mask)
        else float("nan")
    )
    unvisited_exact_abs_mean = (
        float(np.mean(np.abs(eq[unvisited_mask]))) if np.any(unvisited_mask) else 0.0
    )
    pair_coverage = float(np.mean(visited_mask))

    total_model_states = lq.shape[0] * lq.shape[1]
    learned_in_exact_optimal = 0
    exact_tie_states = 0
    for mi in range(lq.shape[0]):
        for si in range(lq.shape[1]):
            la = int(np.argmax(lq[mi, si]))
            opts = optimal_actions(eq[mi, si])
            learned_in_exact_optimal += int(np.any(opts == la))
            exact_tie_states += int(len(opts) > 1)
    greedy_action_agreement = learned_in_exact_optimal / total_model_states
    exact_tie_state_fraction = exact_tie_states / total_model_states

    # Initial state decomposition
    initial_state = mod.InjurySearchState(
        position=scenario.config.start,
        model_probabilities=scenario.model_prior,
        eta=np.ones(len(scenario.config.unknown_cells), dtype=np.int8),
    )
    initial_si = state_index(learned, initial_state)
    initial_lall = learned.values_for_all_models(initial_state)
    initial_eall = exact.values_for_all_models(initial_state)
    initial_p = np.asarray(initial_state.model_probabilities, dtype=np.float64)
    initial_lw = initial_p @ initial_lall
    initial_ew = initial_p @ initial_eall
    initial_la = int(np.argmax(initial_lw))
    initial_eopts = optimal_actions(initial_ew)

    contribution_rows: list[dict[str, Any]] = []
    for mi, model in enumerate(scenario.models):
        row: dict[str, Any] = {
            "model_index": mi,
            "model": model_label(model),
            "prior": float(initial_p[mi]),
            "learned_greedy": action_name(mod, int(np.argmax(initial_lall[mi]))),
            "exact_greedy_first": action_name(mod, int(np.argmax(initial_eall[mi]))),
        }
        for ai, action in enumerate(mod.Action):
            row[f"learned_Q_{action.name}"] = float(initial_lall[mi, ai])
            row[f"exact_Q_{action.name}"] = float(initial_eall[mi, ai])
            row[f"error_{action.name}"] = float(initial_lall[mi, ai] - initial_eall[mi, ai])
            row[f"visits_{action.name}"] = int(visits[mi, initial_si, ai])
            row[f"weighted_contribution_{action.name}"] = float(initial_p[mi] * initial_lall[mi, ai])
        contribution_rows.append(row)

    contribution_fields = list(contribution_rows[0].keys())
    write_csv(out / "initial_state_model_contributions.csv", contribution_fields, contribution_rows)

    # Learned Active trajectories, with exact-Q counterfactual at same states
    simulator = mod.TrueEnvironmentSimulator(
        mod.MazeEnvironment(scenario.config),
        scenario.models,
        scenario.true_model,
        scenario.config.unknown_cells,
    )

    all_rows: list[dict[str, Any]] = []
    learned_cumulative = np.zeros((args.trials, args.horizon), dtype=np.float64)

    for trial in range(args.trials):
        trial_seed = args.base_eval_seed + trial
        simulator.reseed(trial_seed)
        state = simulator.initial_state(scenario.config.start, scenario.model_prior)

        for step in range(args.horizon):
            si = state_index(learned, state)
            p = np.asarray(state.model_probabilities, dtype=np.float64)
            lall = learned.values_for_all_models(state)
            eall = exact.values_for_all_models(state)
            lw = p @ lall
            ew = p @ eall

            learned_action = int(np.argmax(lw))
            exact_opts = optimal_actions(ew)
            exact_first = int(exact_opts[0])
            agreement = bool(np.any(exact_opts == learned_action))

            v = visits[:, si, :]
            zero_visit_mass_selected = float(np.sum(p[v[:, learned_action] == 0]))
            zero_visit_mass_all_actions = [
                float(np.sum(p[v[:, ai] == 0])) for ai in range(len(mod.Action))
            ]
            posterior_weighted_visits_selected = float(p @ v[:, learned_action])

            outcome = simulator.step(state, learned_action)
            next_state = outcome.next_state
            posterior_l1_change = float(
                np.sum(np.abs(next_state.model_probabilities - state.model_probabilities))
            )

            row: dict[str, Any] = {
                "trial": trial,
                "trial_seed": trial_seed,
                "step": step + 1,
                "row": state.position[0],
                "col": state.position[1],
                "eta": "".join(str(int(x)) for x in state.eta),
                "injuries_before": state.located_injury_count,
                "posterior_entropy": entropy(p),
                "posterior_max": float(np.max(p)),
                "posterior_support": int(np.count_nonzero(p > 1e-12)),
                "learned_action": action_name(mod, learned_action),
                "exact_action_first": action_name(mod, exact_first),
                "learned_action_exact_optimal": int(agreement),
                "learned_margin": margin(lw),
                "exact_margin": margin(ew),
                "weighted_Q_rmse": float(np.sqrt(np.mean((lw - ew) ** 2))),
                "all_model_Q_rmse_at_state": float(np.sqrt(np.mean((lall - eall) ** 2))),
                "zero_visit_posterior_mass_selected_action": zero_visit_mass_selected,
                "posterior_weighted_visit_count_selected_action": posterior_weighted_visits_selected,
                "observation_next_row": outcome.observation.next_position[0],
                "observation_next_col": outcome.observation.next_position[1],
                "observed_cell": "" if outcome.observation.observed_cell is None else str(outcome.observation.observed_cell),
                "observed_type": "" if outcome.observation.observed_type is None else outcome.observation.observed_type.value,
                "reward": float(outcome.reward),
                "posterior_l1_change": posterior_l1_change,
                "position_changed": int(next_state.position != state.position),
                "injuries_after": next_state.located_injury_count,
            }
            for ai, action in enumerate(mod.Action):
                row[f"learned_weighted_Q_{action.name}"] = float(lw[ai])
                row[f"exact_weighted_Q_{action.name}"] = float(ew[ai])
                row[f"zero_visit_mass_{action.name}"] = zero_visit_mass_all_actions[ai]
            all_rows.append(row)

            state = next_state
            learned_cumulative[trial, step] = state.located_injury_count

    trajectory_fields = list(all_rows[0].keys())
    write_csv(out / "trajectory_steps.csv", trajectory_fields, all_rows)

    # Exact Active performance under identical simulator/seeds
    exact_result = mod.evaluate_policy(
        scenario,
        mod.ActiveLearningPolicy(exact),
        args.trials,
        args.horizon,
        args.base_eval_seed,
    )
    exact_cumulative = np.asarray(exact_result.cumulative_injuries, dtype=np.float64)

    # Per-step aggregate diagnostic
    by_step_rows: list[dict[str, Any]] = []
    for step in range(1, args.horizon + 1):
        rows = [r for r in all_rows if int(r["step"]) == step]
        agreements = np.asarray([r["learned_action_exact_optimal"] for r in rows], dtype=float)
        zero_mass = np.asarray([r["zero_visit_posterior_mass_selected_action"] for r in rows], dtype=float)
        qerr = np.asarray([r["weighted_Q_rmse"] for r in rows], dtype=float)
        pchange = np.asarray([r["posterior_l1_change"] for r in rows], dtype=float)
        ent = np.asarray([r["posterior_entropy"] for r in rows], dtype=float)
        by_step_rows.append(
            {
                "step": step,
                "learned_active_mean_injuries": float(learned_cumulative[:, step - 1].mean()),
                "exact_active_mean_injuries": float(exact_cumulative[:, step - 1].mean()),
                "exact_action_agreement_rate": float(agreements.mean()),
                "mean_zero_visit_posterior_mass_selected_action": float(zero_mass.mean()),
                "mean_weighted_Q_rmse": float(qerr.mean()),
                "posterior_unchanged_rate": float(np.mean(pchange <= 1e-12)),
                "mean_posterior_entropy": float(ent.mean()),
            }
        )
    write_csv(out / "by_step_summary.csv", list(by_step_rows[0].keys()), by_step_rows)

    # Overall trajectory metrics
    agreements_all = np.asarray(
        [r["learned_action_exact_optimal"] for r in all_rows], dtype=float
    )
    zero_mass_all = np.asarray(
        [r["zero_visit_posterior_mass_selected_action"] for r in all_rows], dtype=float
    )
    qerr_all = np.asarray([r["weighted_Q_rmse"] for r in all_rows], dtype=float)
    pchange_all = np.asarray([r["posterior_l1_change"] for r in all_rows], dtype=float)
    stuck_all = np.asarray(
        [
            int((not bool(r["position_changed"])) and float(r["posterior_l1_change"]) <= 1e-12)
            for r in all_rows
        ],
        dtype=float,
    )

    summary = {
        "figure": args.figure,
        "project_file": str(args.project_file),
        "comparator_bank": str(args.comparator_bank),
        "trials": args.trials,
        "horizon": args.horizon,
        "base_eval_seed": args.base_eval_seed,
        "qlearning_config": {
            "episodes": int(learned.config.episodes),
            "horizon": int(learned.config.horizon),
            "alpha": float(learned.config.alpha),
            "gamma": float(learned.config.gamma),
            "epsilon": float(learned.config.epsilon),
        },
        "global_q_bank": {
            "rmse_all_q_entries": global_rmse,
            "rmse_visited_q_entries": visited_rmse,
            "state_action_pair_coverage": pair_coverage,
            "mean_abs_exact_q_on_unvisited_entries": unvisited_exact_abs_mean,
            "learned_greedy_action_is_exact_optimal_rate_all_model_states": greedy_action_agreement,
            "exact_q_action_tie_fraction_all_model_states": exact_tie_state_fraction,
        },
        "initial_state": {
            "position": list(initial_state.position),
            "posterior_entropy": entropy(initial_p),
            "learned_weighted_q": {
                action.name: float(initial_lw[ai]) for ai, action in enumerate(mod.Action)
            },
            "exact_weighted_q": {
                action.name: float(initial_ew[ai]) for ai, action in enumerate(mod.Action)
            },
            "learned_action": action_name(mod, initial_la),
            "exact_optimal_actions": [action_name(mod, int(ai)) for ai in initial_eopts],
            "learned_margin": margin(initial_lw),
            "exact_margin": margin(initial_ew),
            "learned_action_exact_optimal": bool(np.any(initial_eopts == initial_la)),
            "zero_visit_posterior_mass_by_action": {
                action.name: float(np.sum(initial_p[visits[:, initial_si, ai] == 0]))
                for ai, action in enumerate(mod.Action)
            },
        },
        "trajectory": {
            "learned_active_final_mean_injuries": float(learned_cumulative[:, -1].mean()),
            "exact_active_final_mean_injuries": float(exact_cumulative[:, -1].mean()),
            "learned_vs_exact_action_agreement_rate": float(agreements_all.mean()),
            "mean_zero_visit_posterior_mass_selected_action": float(zero_mass_all.mean()),
            "fraction_steps_selected_action_has_any_zero_visit_posterior_mass": float(np.mean(zero_mass_all > 1e-12)),
            "mean_weighted_q_rmse": float(qerr_all.mean()),
            "posterior_unchanged_rate": float(np.mean(pchange_all <= 1e-12)),
            "same_position_and_posterior_unchanged_rate": float(stuck_all.mean()),
        },
    }

    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Plots
    x = np.arange(1, args.horizon + 1)
    learned_mean = learned_cumulative.mean(axis=0)
    exact_mean = exact_cumulative.mean(axis=0)

    plt.figure(figsize=(9, 5))
    plt.plot(x, learned_mean, label="Active with supplied Q-learning bank")
    plt.plot(x, exact_mean, label="Active with exact model-specific Q*")
    plt.xlabel("Steps")
    plt.ylabel("Average Located Injuries")
    plt.title(f"Figure {args.figure}: Active Learning — learned vs exact Q")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "active_learned_vs_exact.png", dpi=180)
    plt.close()

    agreement_by_step = np.asarray([r["exact_action_agreement_rate"] for r in by_step_rows])
    plt.figure(figsize=(9, 5))
    plt.plot(x, agreement_by_step)
    plt.ylim(-0.02, 1.02)
    plt.xlabel("Steps")
    plt.ylabel("Agreement rate")
    plt.title("Learned Active action is exact-optimal at same visited state")
    plt.tight_layout()
    plt.savefig(out / "action_agreement_by_step.png", dpi=180)
    plt.close()

    zero_by_step = np.asarray(
        [r["mean_zero_visit_posterior_mass_selected_action"] for r in by_step_rows]
    )
    plt.figure(figsize=(9, 5))
    plt.plot(x, zero_by_step)
    plt.ylim(-0.02, 1.02)
    plt.xlabel("Steps")
    plt.ylabel("Posterior mass")
    plt.title("Posterior mass on models where selected Q(s,a) was never visited")
    plt.tight_layout()
    plt.savefig(out / "zero_visit_mass_by_step.png", dpi=180)
    plt.close()

    err_by_step = np.asarray([r["mean_weighted_Q_rmse"] for r in by_step_rows])
    plt.figure(figsize=(9, 5))
    plt.plot(x, err_by_step)
    plt.xlabel("Steps")
    plt.ylabel("RMSE")
    plt.title("Posterior-weighted Q error: learned bank vs exact Q*")
    plt.tight_layout()
    plt.savefig(out / "weighted_q_error_by_step.png", dpi=180)
    plt.close()

    # Human-readable console summary.
    print("\n" + "=" * 78)
    print(f"FINAL ACTIVE DIAGNOSTIC — Figure {args.figure}")
    print("=" * 78)
    print(f"Learned Active @ {args.horizon}: {summary['trajectory']['learned_active_final_mean_injuries']:.4f}")
    print(f"Exact-Q Active @ {args.horizon}: {summary['trajectory']['exact_active_final_mean_injuries']:.4f}")
    print(
        "Initial learned weighted Q: "
        + ", ".join(f"{k}={v:.6f}" for k, v in summary["initial_state"]["learned_weighted_q"].items())
    )
    print(
        "Initial exact weighted Q:   "
        + ", ".join(f"{k}={v:.6f}" for k, v in summary["initial_state"]["exact_weighted_q"].items())
    )
    print(f"Initial learned action: {summary['initial_state']['learned_action']}")
    print(f"Initial exact optimal: {summary['initial_state']['exact_optimal_actions']}")
    print(
        "Action agreement on learned trajectories: "
        f"{summary['trajectory']['learned_vs_exact_action_agreement_rate']:.3f}"
    )
    print(
        "Mean posterior mass on unvisited selected Q entries: "
        f"{summary['trajectory']['mean_zero_visit_posterior_mass_selected_action']:.3f}"
    )
    print(f"Global Q-bank RMSE vs exact: {global_rmse:.6f}")
    print(f"Visited state-action coverage: {pair_coverage:.3f}")
    print(f"Outputs: {out.resolve()}")
    print("=" * 78)


if __name__ == "__main__":
    main()
