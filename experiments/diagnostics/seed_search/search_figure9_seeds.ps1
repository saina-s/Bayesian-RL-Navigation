# Figure 9 controlled global-seed search
# Run from: C:\Users\ASUS\bayesian-navigation
#
# This script:
# 1) changes ONLY --global-seed
# 2) keeps transition/replay/action seeds fixed
# 3) trains every candidate for the full paper setting: 5000 x 250
# 4) screens each candidate on the same 200 evaluation seeds
# 5) preserves every checkpoint, log, and evaluation file separately

$ErrorActionPreference = "Stop"

Set-Location "C:\Users\ASUS\bayesian-navigation"

$seeds = 0, 1, 2, 3, 4

$resultsDir = ".\outputs\standalone\figure9\seed_search"
New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null

foreach ($s in $seeds) {
    Write-Host ""
    Write-Host "=================================================="
    Write-Host "FIGURE 9 - TRAINING GLOBAL SEED $s"
    Write-Host "=================================================="

    .\.venv\Scripts\python.exe .\final_bayesian_navigation.py `
      --figure 9 `
      --mode train-proposed `
      --episodes 5000 `
      --horizon 250 `
      --global-seed $s `
      --transition-seed 1001 `
      --replay-seed 1002 `
      --action-seed 1003 `
      --progress-every 100 `
      2>&1 | Tee-Object "$resultsDir\figure9_seed${s}_training.txt"

    if ($LASTEXITCODE -ne 0) {
        throw "Training failed for seed $s"
    }

    Write-Host ""
    Write-Host "FIGURE 9 - SCREENING GLOBAL SEED $s"

    .\.venv\Scripts\python.exe .\final_bayesian_navigation.py `
      --figure 9 `
      --mode evaluate-proposed `
      --trials 200 `
      --eval-horizon 50 `
      --base-eval-seed 10000 `
      --global-seed $s `
      2>&1 | Tee-Object "$resultsDir\figure9_seed${s}_screening.txt"

    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation failed for seed $s"
    }

    Copy-Item `
      ".\outputs\standalone\figure9\figure9_proposed_evaluation.npz" `
      "$resultsDir\figure9_proposed_seed${s}_screening.npz" `
      -Force
}

Write-Host ""
Write-Host "Seed search batch completed."
Write-Host "Now run:"
Write-Host ".\.venv\Scripts\python.exe .\rank_figure9_seeds.py"
