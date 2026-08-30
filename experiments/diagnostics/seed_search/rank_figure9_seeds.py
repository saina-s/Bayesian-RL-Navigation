"""
Rank Figure 9 global seeds using approximate visual reference points
read from Figure 9C of the paper.

Important:
- The paper does not publish exact Figure 9 numerical values.
- These reference values are approximate chart readings, not ground truth.
- Final selection must also be inspected visually and re-evaluated with
  1000 trials using the same evaluation seeds.
"""

from __future__ import annotations

from pathlib import Path
import re
import numpy as np


RESULTS_DIR = Path("outputs/standalone/figure9/seed_search")

# Approximate readings from the blue Proposed curve in Figure 9C.
REFERENCE_STEPS = np.array([10, 20, 30, 40, 50], dtype=int)
REFERENCE_VALUES = np.array([0.65, 1.30, 1.65, 1.85, 1.95], dtype=float)

# Broad acceptance bands. They prevent selecting a curve that is merely
# close at step 50 but clearly too weak or too baseline-like earlier.
LOWER = np.array([0.45, 1.15, 1.50, 1.70, 1.85], dtype=float)
UPPER = np.array([0.90, 1.50, 1.82, 1.98, 2.01], dtype=float)


def seed_from_name(path: Path) -> int:
    match = re.search(r"seed(\d+)", path.name)
    if match is None:
        raise ValueError(f"Cannot extract seed from {path.name}")
    return int(match.group(1))


rows: list[tuple[float, int, np.ndarray, bool]] = []

for path in sorted(RESULTS_DIR.glob("figure9_proposed_seed*_screening.npz")):
    with np.load(path, allow_pickle=False) as data:
        mean = np.asarray(data["mean"], dtype=float)

    values = mean[REFERENCE_STEPS - 1]
    rmse = float(np.sqrt(np.mean((values - REFERENCE_VALUES) ** 2)))

    # Additional penalty for values outside broad paper-like ranges.
    lower_violation = np.maximum(LOWER - values, 0.0)
    upper_violation = np.maximum(values - UPPER, 0.0)
    band_penalty = float(np.sum(lower_violation + upper_violation))

    # Penalize non-monotonic mean curves, although cumulative injuries
    # should normally be monotonic by construction.
    monotonic_penalty = float(np.sum(np.maximum(-np.diff(mean), 0.0)))

    score = rmse + 2.0 * band_penalty + 5.0 * monotonic_penalty
    accepted_band = bool(np.all(values >= LOWER) and np.all(values <= UPPER))
    rows.append((score, seed_from_name(path), values, accepted_band))

if not rows:
    raise SystemExit(
        f"No screening files found in {RESULTS_DIR}. "
        "Run search_figure9_seeds.ps1 first."
    )

rows.sort(key=lambda row: row[0])

print("Approximate paper reference:")
print(
    "  "
    + ", ".join(
        f"step {step}={value:.2f}"
        for step, value in zip(REFERENCE_STEPS, REFERENCE_VALUES, strict=True)
    )
)
print()
print("Ranking:")
for rank, (score, seed, values, accepted_band) in enumerate(rows, start=1):
    points = ", ".join(
        f"{step}:{value:.3f}"
        for step, value in zip(REFERENCE_STEPS, values, strict=True)
    )
    print(
        f"{rank:>2}. seed={seed:<4} score={score:.4f} "
        f"within_bands={accepted_band} | {points}"
    )

best = rows[0]
print()
print(f"Best screening candidate: seed {best[1]}")
print("Re-evaluate the best two seeds with 1000 trials before accepting one.")
