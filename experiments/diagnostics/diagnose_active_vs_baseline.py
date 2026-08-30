"""
Diagnose why Figure 9 Active Learning overlaps the Baseline.

Run from the folder containing final_bayesian_navigation.py.

Example:
    .\.venv\Scripts\python.exe .\diagnose_active_vs_baseline.py `
      --figure 9 `
      --comparator-bank .\outputs\planning\figure9_comparator_dqn_distinct.pt `
      --trials 1000 `
      --horizon 50 `
      --base-eval-seed 10000 `
      --output-dir .\outputs\standalone\figure9_distinct\active_baseline_diagnostic

This script does not train or overwrite checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", type=int, choices=(8, 9), default=9)
    parser.add_argument(
        "--project-file",
        type=Path,
        default=Path("final_bayesian_navigation.py"),
    )
    parser.add_argument(
        "--comparator-bank",
        type=Path,
        default=Path("outputs/planning/figure9_comparator_dqn_distinct.pt"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--base-eval-seed", type=int, default=10000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/standalone/figure9_distinct/"
            "active_baseline_diagnostic"
        ),
    )
    return parser.parse_args()


def load_project_module(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(
            f"Project file not found: {path}. "
            "Run from the project directory."
        )

    spec = importlib.util.spec_from_file_location(
        "bayesian_navigation_project", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def entropy(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0.0]
    return float(-np.sum(positive * np.log(positive)))


def maximizer_count(probabilities: np.ndarray) -> int:
    maximum = float(np.max(probabilities))
    return int(
        np.count_nonzero(
            np.isclose(probabilities, maximum, rtol=0.0, atol=1e-12)
        )
    )


def first_crossing(values: np.ndarray, threshold: float) -> np.ndarray:
    crossed = values >= threshold
    result = np.full(values.shape[0], values.shape[1] + 1, dtype=np.int64)
    valid = crossed.any(axis=1)
    result[valid] = crossed[valid].argmax(axis=1) + 1
    return result


def main() -> None:
    args = parse_args()
    nav = load_project_module(args.project_file)

    scenario = nav.create_scenario(args.figure)
    device = nav.get_device(args.device)
    bank = nav.load_neural_comparator_bank(
        args.comparator_bank,
        scenario,
        device,
    )

    baseline = nav.BaselinePolicy(bank, scenario.true_model)
    active = nav.ActiveLearningPolicy(bank)
    true_index = bank.model_index(scenario.true_model)

    # Normal evaluations: checks whether the stored reward arrays are
    # exactly equal, not merely whether their means overlap.
    baseline_result = nav.evaluate_policy(
        scenario,
        baseline,
        args.trials,
        args.horizon,
        args.base_eval_seed,
    )
    active_result = nav.evaluate_policy(
        scenario,
        active,
        args.trials,
        args.horizon,
        args.base_eval_seed,
    )

    reward_arrays_equal = bool(
        np.array_equal(
            baseline_result.cumulative_injuries,
            active_result.cumulative_injuries,
        )
    )
    reward_agreement = float(
        np.mean(
            baseline_result.cumulative_injuries
            == active_result.cumulative_injuries
        )
    )

    # Follow Active Learning trajectories. At each exact same visited
    # belief state, also ask which action Baseline would select.
    simulator = nav.TrueEnvironmentSimulator(
        nav.MazeEnvironment(scenario.config),
        scenario.models,
        scenario.true_model,
        scenario.config.unknown_cells,
    )

    shape = (args.trials, args.horizon)
    action_agreement = np.zeros(shape, dtype=np.bool_)
    active_actions = np.zeros(shape, dtype=np.int8)
    baseline_actions = np.zeros(shape, dtype=np.int8)
    p_true_before = np.zeros(shape, dtype=np.float64)
    p_true_after = np.zeros(shape, dtype=np.float64)
    entropy_before = np.zeros(shape, dtype=np.float64)
    entropy_after = np.zeros(shape, dtype=np.float64)
    maximizers_before = np.zeros(shape, dtype=np.int16)
    maximizers_after = np.zeros(shape, dtype=np.int16)
    explicit_wall = np.zeros(shape, dtype=np.bool_)
    unknown_observation = np.zeros(shape, dtype=np.bool_)
    injuries = np.zeros(shape, dtype=np.float64)

    trial_seeds = (
        args.base_eval_seed
        + np.arange(args.trials, dtype=np.int64)
    )

    for trial_index, trial_seed in enumerate(trial_seeds):
        simulator.reseed(int(trial_seed))
        state = simulator.initial_state(
            scenario.config.start,
            scenario.model_prior,
        )

        for step_index in range(args.horizon):
            probabilities = state.model_probabilities
            p_true_before[trial_index, step_index] = probabilities[true_index]
            entropy_before[trial_index, step_index] = entropy(probabilities)
            maximizers_before[trial_index, step_index] = maximizer_count(
                probabilities
            )

            active_action = int(active.select_action(state))
            baseline_action = int(baseline.select_action(state))
            active_actions[trial_index, step_index] = active_action
            baseline_actions[trial_index, step_index] = baseline_action
            action_agreement[trial_index, step_index] = (
                active_action == baseline_action
            )

            outcome = simulator.step(state, active_action)
            state = outcome.next_state

            next_probabilities = state.model_probabilities
            p_true_after[trial_index, step_index] = next_probabilities[true_index]
            entropy_after[trial_index, step_index] = entropy(
                next_probabilities
            )
            maximizers_after[trial_index, step_index] = maximizer_count(
                next_probabilities
            )

            explicit_wall[trial_index, step_index] = (
                outcome.observation.observed_type is nav.CellType.WALL
            )
            unknown_observation[trial_index, step_index] = (
                outcome.observation.observed_type is not None
            )
            injuries[trial_index, step_index] = (
                state.located_injury_count
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = args.output_dir / "active_vs_baseline_trace.npz"
    np.savez_compressed(
        npz_path,
        figure=np.asarray(args.figure),
        comparator_bank=np.asarray(str(args.comparator_bank)),
        trial_seeds=trial_seeds,
        action_agreement=action_agreement,
        active_actions=active_actions,
        baseline_actions=baseline_actions,
        true_probability_before=p_true_before,
        true_probability_after=p_true_after,
        entropy_before=entropy_before,
        entropy_after=entropy_after,
        maximizers_before=maximizers_before,
        maximizers_after=maximizers_after,
        explicit_wall_observation=explicit_wall,
        unknown_observation=unknown_observation,
        cumulative_injuries=injuries,
        baseline_cumulative=baseline_result.cumulative_injuries,
        active_cumulative=active_result.cumulative_injuries,
    )

    csv_path = args.output_dir / "step_summary.csv"
    fields = [
        "step",
        "action_agreement_rate",
        "mean_true_model_probability_after",
        "fraction_true_probability_after_ge_0_5",
        "fraction_true_probability_after_ge_0_9",
        "mean_entropy_after",
        "mean_number_of_maximizers_after",
        "explicit_wall_observation_rate",
        "mean_cumulative_injuries",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(args.horizon):
            writer.writerow(
                {
                    "step": index + 1,
                    "action_agreement_rate": float(
                        action_agreement[:, index].mean()
                    ),
                    "mean_true_model_probability_after": float(
                        p_true_after[:, index].mean()
                    ),
                    "fraction_true_probability_after_ge_0_5": float(
                        np.mean(p_true_after[:, index] >= 0.5)
                    ),
                    "fraction_true_probability_after_ge_0_9": float(
                        np.mean(p_true_after[:, index] >= 0.9)
                    ),
                    "mean_entropy_after": float(
                        entropy_after[:, index].mean()
                    ),
                    "mean_number_of_maximizers_after": float(
                        maximizers_after[:, index].mean()
                    ),
                    "explicit_wall_observation_rate": float(
                        explicit_wall[:, index].mean()
                    ),
                    "mean_cumulative_injuries": float(
                        injuries[:, index].mean()
                    ),
                }
            )

    cross_05 = first_crossing(p_true_after, 0.5)
    cross_09 = first_crossing(p_true_after, 0.9)

    def crossing_text(values: np.ndarray) -> str:
        reached = values <= args.horizon
        if not np.any(reached):
            return "never reached"
        return (
            f"reached_fraction={reached.mean():.3f}, "
            f"median_step={np.median(values[reached]):.1f}"
        )

    print("=" * 82)
    print("Active Learning vs Baseline diagnostic")
    print("=" * 82)
    print(f"comparator_bank={args.comparator_bank}")
    print(
        f"seeds={args.base_eval_seed}.."
        f"{args.base_eval_seed + args.trials - 1}"
    )
    print(f"reward_arrays_exactly_equal={reward_arrays_equal}")
    print(f"per_trial_step_reward_agreement={reward_agreement:.4f}")
    print(f"overall_action_agreement={action_agreement.mean():.4f}")
    print(
        "fraction_of_trials_with_explicit_WALL="
        f"{explicit_wall.any(axis=1).mean():.4f}"
    )
    print(f"P(true)>=0.5: {crossing_text(cross_05)}")
    print(f"P(true)>=0.9: {crossing_text(cross_09)}")
    print()
    print(
        " step | agree | P(true) | entropy | maximizers | WALL | injuries"
    )
    print("-" * 82)
    for step in (1, 5, 10, 20, 30, 40, 50):
        if step > args.horizon:
            continue
        index = step - 1
        print(
            f"{step:>5} | "
            f"{action_agreement[:, index].mean():>5.3f} | "
            f"{p_true_after[:, index].mean():>7.3f} | "
            f"{entropy_after[:, index].mean():>7.3f} | "
            f"{maximizers_after[:, index].mean():>10.2f} | "
            f"{explicit_wall[:, index].mean():>4.3f} | "
            f"{injuries[:, index].mean():>8.3f}"
        )

    print()
    print(f"saved_npz={npz_path}")
    print(f"saved_csv={csv_path}")


if __name__ == "__main__":
    main()
