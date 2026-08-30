"""
Evaluate Figure 9 MAP with each of the 27 initially tied models fixed
across all evaluation trials.

Purpose
-------
At the uniform prior, all 27 models are MAP maximizers. The current
sticky-random implementation draws a new initial maximizer in every trial,
so the reported curve is a mixture over many initial models. A deterministic
argmax implementation instead reuses the same first model in every trial.

This diagnostic evaluates all 27 deterministic initial choices separately.
It does not train or overwrite checkpoints.

Example
-------
.\.venv\Scripts\python.exe .\map_fixed_initial_model_sweep.py `
  --figure 9 `
  --comparator-bank .\outputs\planning\figure9_comparator_dqn_standalone.pt `
  --trials 1000 `
  --horizon 50 `
  --base-eval-seed 10000 `
  --output-dir .\outputs\standalone\figure9_map_fixed_initial_common
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


REPORT_STEPS = (10, 20, 30, 40, 50)


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
        default=Path(
            "outputs/planning/figure9_comparator_dqn_standalone.pt"
        ),
    )
    parser.add_argument(
        "--comparator-backend",
        choices=("neural", "tabular"),
        default="neural",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--base-eval-seed", type=int, default=10000)
    parser.add_argument("--fallback-tie-seed", type=int, default=30001)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/standalone/figure9_map_fixed_initial_common"
        ),
    )
    return parser.parse_args()


def load_project_module(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(
            f"Project file not found: {path}. "
            "Run this script from the project directory."
        )

    spec = importlib.util.spec_from_file_location(
        "bayesian_navigation_project", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


def model_label(model: tuple[Any, ...]) -> str:
    return "".join(cell.value for cell in model)


def injury_count(model: tuple[Any, ...], cell_type: Any) -> int:
    return sum(cell is cell_type.INJURY for cell in model)


def wall_count(model: tuple[Any, ...], cell_type: Any) -> int:
    return sum(cell is cell_type.WALL for cell in model)


def main() -> None:
    args = parse_args()
    if args.trials <= 0 or args.horizon <= 0:
        raise ValueError("Trials and horizon must be positive.")

    nav = load_project_module(args.project_file)
    scenario = nav.create_scenario(args.figure)
    device = nav.get_device(args.device)

    if args.comparator_backend == "neural":
        bank = nav.load_neural_comparator_bank(
            args.comparator_bank,
            scenario,
            device,
        )
    else:
        bank = nav.load_comparator_bank(args.comparator_bank)

    class FixedInitialStickyMAP(nav.MAPPolicy):
        """Use one fixed initial maximizer in every evaluation trial."""

        def __init__(self, initial_model_index: int) -> None:
            super().__init__(
                bank,
                args.fallback_tie_seed,
                tie_rule="sticky-random",
            )
            self.initial_model_index = int(initial_model_index)

        def reset(self, trial_seed: int) -> None:
            # The same initial MAP model is reused in every trial.
            # If later evidence eliminates it, fallback sticky-random
            # selection is reproducible for that trial.
            self.rng = np.random.default_rng(
                self.base_seed + int(trial_seed)
            )
            self.selected_index = self.initial_model_index

    args.output_dir.mkdir(parents=True, exist_ok=True)

    curves: list[np.ndarray] = []
    ci_curves: list[np.ndarray] = []
    cumulative_arrays: list[np.ndarray] = []
    rows: list[dict[str, float | int | str]] = []

    true_index = bank.model_index(scenario.true_model)

    print("=" * 94)
    print("MAP fixed-initial-model sweep")
    print("=" * 94)
    print(f"figure={args.figure}")
    print(f"comparator_bank={args.comparator_bank}")
    print(
        f"evaluation_seeds={args.base_eval_seed}.."
        f"{args.base_eval_seed + args.trials - 1}"
    )
    print(f"true_model_index={true_index}")
    print()

    for model_index, model in enumerate(scenario.models):
        policy = FixedInitialStickyMAP(model_index)
        result = nav.evaluate_policy(
            scenario,
            policy,
            args.trials,
            args.horizon,
            args.base_eval_seed,
        )

        mean = np.asarray(result.mean, dtype=np.float64)
        ci95 = np.asarray(result.ci95, dtype=np.float64)
        cumulative = np.asarray(
            result.cumulative_injuries,
            dtype=np.float64,
        )

        curves.append(mean)
        ci_curves.append(ci95)
        cumulative_arrays.append(cumulative)

        row: dict[str, float | int | str] = {
            "model_index": model_index,
            "model": model_label(model),
            "is_true_model": int(model_index == true_index),
            "injury_count": injury_count(model, nav.CellType),
            "wall_count": wall_count(model, nav.CellType),
            "final_mean": float(mean[-1]),
            "final_ci95": float(ci95[-1]),
        }
        for step in REPORT_STEPS:
            if step <= args.horizon:
                row[f"step_{step}"] = float(mean[step - 1])
        rows.append(row)

        points = ", ".join(
            f"{step}:{mean[step - 1]:.3f}"
            for step in REPORT_STEPS
            if step <= args.horizon
        )
        true_marker = " TRUE" if model_index == true_index else ""
        print(
            f"model={model_index:02d} {model_label(model)}"
            f"{true_marker} | {points}"
        )

    curve_array = np.stack(curves, axis=0)
    ci_array = np.stack(ci_curves, axis=0)
    cumulative_array = np.stack(cumulative_arrays, axis=0)

    csv_path = args.output_dir / "map_fixed_initial_models.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    npz_path = args.output_dir / "map_fixed_initial_models.npz"
    np.savez_compressed(
        npz_path,
        figure=np.asarray(args.figure),
        comparator_bank=np.asarray(str(args.comparator_bank)),
        model_indices=np.arange(len(scenario.models), dtype=np.int64),
        model_labels=np.asarray(
            [model_label(model) for model in scenario.models]
        ),
        true_model_index=np.asarray(true_index),
        trial_seeds=(
            args.base_eval_seed
            + np.arange(args.trials, dtype=np.int64)
        ),
        mean_curves=curve_array,
        ci95_curves=ci_array,
        cumulative_injuries=cumulative_array,
    )

    steps = np.arange(1, args.horizon + 1)
    figure, axis = plt.subplots(figsize=(9.0, 5.8))

    for model_index, curve in enumerate(curve_array):
        if model_index == true_index:
            axis.plot(
                steps,
                curve,
                linewidth=2.6,
                label=(
                    f"True initial model "
                    f"{model_label(scenario.models[model_index])}"
                ),
            )
        else:
            axis.plot(steps, curve, linewidth=0.9, alpha=0.42)

    axis.axhline(
        0.5,
        linestyle="--",
        linewidth=1.2,
        label="Approximate paper MAP level at step 50",
    )
    axis.set_xlabel("Steps")
    axis.set_ylabel("Average Located Injuries")
    axis.set_xlim(1, args.horizon)
    axis.set_ylim(0.0, 2.05)
    axis.grid(alpha=0.25)
    axis.legend()
    axis.set_title(
        f"Figure {args.figure} MAP: Fixed Initial Maximizer Sensitivity"
    )
    figure.tight_layout()

    figure_path = args.output_dir / "map_fixed_initial_models.png"
    figure.savefig(figure_path, dpi=240, bbox_inches="tight")
    plt.close(figure)

    paper_like = [
        row
        for row in rows
        if 0.0 < float(row["final_mean"]) <= 0.55
    ]
    zero_models = [
        row for row in rows if float(row["final_mean"]) == 0.0
    ]

    print()
    print("Summary by initial model injury count:")
    for count in range(4):
        selected = [
            float(row["final_mean"])
            for row in rows
            if int(row["injury_count"]) == count
        ]
        if selected:
            values = np.asarray(selected, dtype=np.float64)
            print(
                f"  injuries={count}: models={len(values)}, "
                f"step50 mean={values.mean():.3f}, "
                f"range={values.min():.3f}..{values.max():.3f}"
            )

    print()
    print(
        "Models with 0 < step50 <= 0.55 "
        f"(paper-like range, diagnostic only): {len(paper_like)}"
    )
    for row in paper_like:
        print(
            f"  index={row['model_index']:02d} "
            f"model={row['model']} "
            f"step50={float(row['final_mean']):.3f}"
        )

    print(f"Zero-performance fixed models: {len(zero_models)}")
    print()
    print(f"saved_csv={csv_path}")
    print(f"saved_npz={npz_path}")
    print(f"saved_figure={figure_path}")
    print()
    print(
        "Do not select one fixed model only because it matches the paper. "
        "This sweep tests whether an unpublished deterministic model ordering "
        "could explain the discrepancy."
    )


if __name__ == "__main__":
    main()
