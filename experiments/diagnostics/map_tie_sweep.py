"""
Evaluate Figure 9 MAP under multiple sticky-random tie seeds.

This is a sensitivity analysis. It does not train or overwrite any checkpoint.

Recommended main run:
    .\.venv\Scripts\python.exe .\map_tie_sweep.py `
      --figure 9 `
      --comparator-bank .\outputs\planning\figure9_comparator_dqn_standalone.pt `
      --tie-seeds 30001-30020 `
      --trials 1000 `
      --horizon 50 `
      --base-eval-seed 10000 `
      --output-dir .\outputs\standalone\figure9_map_tie_sweep_common

Optional repeat with the distinct comparator bank:
    .\.venv\Scripts\python.exe .\map_tie_sweep.py `
      --figure 9 `
      --comparator-bank .\outputs\planning\figure9_comparator_dqn_distinct.pt `
      --tie-seeds 30001-30020 `
      --trials 1000 `
      --horizon 50 `
      --base-eval-seed 10000 `
      --output-dir .\outputs\standalone\figure9_map_tie_sweep_distinct
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


REPORT_STEPS = (10, 20, 30, 40, 50)


def parse_seed_spec(spec: str) -> list[int]:
    """Parse comma-separated integers and inclusive ranges such as 1-5."""
    seeds: list[int] = []
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue

        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            if end < start:
                raise ValueError(f"Invalid descending seed range: {part}")
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(part))

    unique: list[int] = []
    seen: set[int] = set()
    for seed in seeds:
        if seed not in seen:
            seen.add(seed)
            unique.append(seed)

    if not unique:
        raise ValueError("At least one MAP tie seed is required.")
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Average sticky-random MAP over multiple tie seeds and separately "
            "evaluate first/random-each-step as sensitivity policies."
        )
    )
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
    parser.add_argument(
        "--tie-seeds",
        default="30001-30020",
        help="Comma-separated seeds and inclusive ranges.",
    )
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--base-eval-seed", type=int, default=10000)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/standalone/figure9_map_tie_sweep_common"
        ),
    )
    parser.add_argument(
        "--skip-sensitivity",
        action="store_true",
        help="Do not evaluate first and random-each-step.",
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


def ci95_from_cumulative(cumulative: np.ndarray) -> np.ndarray:
    if cumulative.shape[0] <= 1:
        return np.zeros(cumulative.shape[1], dtype=np.float64)
    standard_error = cumulative.std(axis=0, ddof=1) / np.sqrt(
        cumulative.shape[0]
    )
    return 1.96 * standard_error


def step_values(curve: np.ndarray, horizon: int) -> dict[str, float]:
    return {
        f"step_{step}": float(curve[step - 1])
        for step in REPORT_STEPS
        if step <= horizon
    }


def main() -> None:
    args = parse_args()
    tie_seeds = parse_seed_spec(args.tie_seeds)

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

    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_seed_cumulative: list[np.ndarray] = []
    per_seed_means: list[np.ndarray] = []
    per_seed_ci95: list[np.ndarray] = []
    seed_rows: list[dict[str, float | int]] = []

    print("=" * 88)
    print("Sticky-random MAP tie-seed sweep")
    print("=" * 88)
    print(f"figure={args.figure}")
    print(f"comparator_bank={args.comparator_bank}")
    print(
        f"evaluation_seeds={args.base_eval_seed}.."
        f"{args.base_eval_seed + args.trials - 1}"
    )
    print(f"map_tie_seeds={tie_seeds}")
    print()

    for index, tie_seed in enumerate(tie_seeds, start=1):
        policy = nav.MAPPolicy(
            bank,
            tie_seed,
            tie_rule="sticky-random",
        )
        result = nav.evaluate_policy(
            scenario,
            policy,
            args.trials,
            args.horizon,
            args.base_eval_seed,
        )

        cumulative = np.asarray(
            result.cumulative_injuries,
            dtype=np.float64,
        )
        mean = np.asarray(result.mean, dtype=np.float64)
        ci95 = np.asarray(result.ci95, dtype=np.float64)

        per_seed_cumulative.append(cumulative)
        per_seed_means.append(mean)
        per_seed_ci95.append(ci95)

        row: dict[str, float | int] = {
            "map_tie_seed": tie_seed,
            "final_mean": float(mean[-1]),
            "final_ci95": float(ci95[-1]),
        }
        row.update(step_values(mean, args.horizon))
        seed_rows.append(row)

        displayed = ", ".join(
            f"{step}:{mean[step - 1]:.3f}"
            for step in REPORT_STEPS
            if step <= args.horizon
        )
        print(
            f"[{index:>2}/{len(tie_seeds)}] "
            f"seed={tie_seed} | {displayed}"
        )

    cumulative_array = np.stack(per_seed_cumulative, axis=0)
    mean_array = np.stack(per_seed_means, axis=0)
    ci95_array = np.stack(per_seed_ci95, axis=0)

    across_seed_mean = mean_array.mean(axis=0)
    across_seed_sd = (
        mean_array.std(axis=0, ddof=1)
        if len(tie_seeds) > 1
        else np.zeros(args.horizon, dtype=np.float64)
    )
    across_seed_min = mean_array.min(axis=0)
    across_seed_max = mean_array.max(axis=0)

    # These are separate policies, not part of the sticky-random average.
    sensitivity: dict[str, dict[str, np.ndarray | int | str]] = {}
    if not args.skip_sensitivity:
        first_seed = tie_seeds[0]
        for tie_rule in ("first", "random-each-step"):
            policy = nav.MAPPolicy(
                bank,
                first_seed,
                tie_rule=tie_rule,
            )
            result = nav.evaluate_policy(
                scenario,
                policy,
                args.trials,
                args.horizon,
                args.base_eval_seed,
            )
            sensitivity[tie_rule] = {
                "tie_seed": first_seed,
                "mean": np.asarray(result.mean, dtype=np.float64),
                "ci95": np.asarray(result.ci95, dtype=np.float64),
                "cumulative": np.asarray(
                    result.cumulative_injuries,
                    dtype=np.float64,
                ),
            }

    npz_payload: dict[str, np.ndarray] = {
        "figure": np.asarray(args.figure),
        "comparator_bank": np.asarray(str(args.comparator_bank)),
        "comparator_backend": np.asarray(args.comparator_backend),
        "tie_rule": np.asarray("sticky-random"),
        "map_tie_seeds": np.asarray(tie_seeds, dtype=np.int64),
        "evaluation_trial_seeds": (
            args.base_eval_seed
            + np.arange(args.trials, dtype=np.int64)
        ),
        "per_seed_cumulative": cumulative_array,
        "per_seed_mean": mean_array,
        "per_seed_ci95": ci95_array,
        "across_tie_seed_mean": across_seed_mean,
        "across_tie_seed_sd": across_seed_sd,
        "across_tie_seed_min": across_seed_min,
        "across_tie_seed_max": across_seed_max,
    }

    for tie_rule, values in sensitivity.items():
        key = tie_rule.replace("-", "_")
        npz_payload[f"{key}_tie_seed"] = np.asarray(
            int(values["tie_seed"])
        )
        npz_payload[f"{key}_mean"] = np.asarray(values["mean"])
        npz_payload[f"{key}_ci95"] = np.asarray(values["ci95"])
        npz_payload[f"{key}_cumulative"] = np.asarray(
            values["cumulative"]
        )

    npz_path = args.output_dir / "map_tie_sweep_results.npz"
    np.savez_compressed(npz_path, **npz_payload)

    per_seed_csv = args.output_dir / "sticky_random_per_seed.csv"
    with per_seed_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(seed_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(seed_rows)

    aggregate_csv = args.output_dir / "sticky_random_aggregate.csv"
    aggregate_fields = [
        "step",
        "mean_across_tie_seeds",
        "sd_across_tie_seed_means",
        "min_across_tie_seeds",
        "max_across_tie_seeds",
    ]
    with aggregate_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader()
        for step_index in range(args.horizon):
            writer.writerow(
                {
                    "step": step_index + 1,
                    "mean_across_tie_seeds": float(
                        across_seed_mean[step_index]
                    ),
                    "sd_across_tie_seed_means": float(
                        across_seed_sd[step_index]
                    ),
                    "min_across_tie_seeds": float(
                        across_seed_min[step_index]
                    ),
                    "max_across_tie_seeds": float(
                        across_seed_max[step_index]
                    ),
                }
            )

    metadata_path = args.output_dir / "map_tie_sweep_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "figure": args.figure,
                "project_file": str(args.project_file),
                "comparator_bank": str(args.comparator_bank),
                "comparator_backend": args.comparator_backend,
                "sticky_random_tie_seeds": tie_seeds,
                "evaluation_base_seed": args.base_eval_seed,
                "trials": args.trials,
                "horizon": args.horizon,
                "sensitivity_policies_are_not_averaged_with_sticky_random": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    steps = np.arange(1, args.horizon + 1)
    figure, axis = plt.subplots(figsize=(8.4, 5.4))
    axis.errorbar(
        steps,
        across_seed_mean,
        yerr=across_seed_sd,
        label="MAP sticky-random: mean ± SD across tie seeds",
    )
    for tie_rule, values in sensitivity.items():
        axis.plot(
            steps,
            np.asarray(values["mean"]),
            label=f"MAP sensitivity: {tie_rule}",
        )
    axis.set_xlabel("Steps")
    axis.set_ylabel("Average Located Injuries")
    axis.set_xlim(1, args.horizon)
    axis.set_ylim(0.0, 2.05)
    axis.grid(alpha=0.25)
    axis.legend()
    axis.set_title(
        f"Figure {args.figure} MAP Tie-Breaking Sensitivity"
    )
    figure.tight_layout()

    figure_path = args.output_dir / "map_tie_sensitivity.png"
    figure.savefig(figure_path, dpi=240, bbox_inches="tight")
    plt.close(figure)

    print()
    print("Sticky-random aggregate across tie seeds:")
    for step in REPORT_STEPS:
        if step > args.horizon:
            continue
        index = step - 1
        print(
            f"  step {step:>2}: "
            f"{across_seed_mean[index]:.4f} "
            f"+/- {across_seed_sd[index]:.4f} SD across tie-seed means "
            f"(range {across_seed_min[index]:.4f}.."
            f"{across_seed_max[index]:.4f})"
        )

    if sensitivity:
        print()
        print("Separate sensitivity policies:")
        for tie_rule, values in sensitivity.items():
            curve = np.asarray(values["mean"])
            displayed = ", ".join(
                f"{step}:{curve[step - 1]:.3f}"
                for step in REPORT_STEPS
                if step <= args.horizon
            )
            print(f"  {tie_rule}: {displayed}")

    print()
    print(f"saved_npz={npz_path}")
    print(f"saved_per_seed_csv={per_seed_csv}")
    print(f"saved_aggregate_csv={aggregate_csv}")
    print(f"saved_figure={figure_path}")
    print(f"saved_metadata={metadata_path}")


if __name__ == "__main__":
    main()
