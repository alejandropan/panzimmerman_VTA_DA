"""
Precompute the PSIS-LOO comparison table. Run once; the figure scripts read the CSVs.

This is the only place that touches the `*_loo.pkl` files, the
`*_loo_i_reordered.npy` corrections, or ArviZ. It writes three small CSVs:

    loo_comparison_5models.csv   Extended Data Fig. 2a  (no Conditioned Reinforcer)
    loo_comparison_6models.csv   Extended Data Fig. 11  (all six)
    loo_pointwise_correlations.csv   the alignment check, for the methods

Re-run this if a model is refitted, if a `*_loo_i_reordered.npy` changes, or if a
model is added. Nothing else needs to change.

Usage:
    python compute_loo_comparison.py
"""

# %%
# ============================================================================
# Imports and configuration
# ============================================================================
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import arviz as az

ON_CLUSTER = os.name != "nt"
WITTEN = Path("/jukebox/witten") if ON_CLUSTER else Path(r"\\cup.princeton.edu\witten")
LOO_DIR = WITTEN / "Alex" / "Data" / "ephys_bandit" / "laser_stan_fits" / "loo_results"

# Pickle stem in LOO_DIR -> manuscript model number.
LOO_MODELS = {
    "Standard_Forgetting": 1,
    "Standard_Forgetting_RPE_laser": 2,
    "QFS_Variable_Alpha": 3,
    "REINFORCE": 4,
    "actor_critic": 5,
    "sequential_actor_critic": 6,
}

# Which comparison goes in which CSV. Dropping a model does not change any other
# model's elpd_loo, se or p_loo (they are per-model), nor elpd_diff and dse (they
# are computed against the best model, which stays in either way).
VARIANTS = {
    "loo_comparison_6models.csv": (),      # Extended Data Fig. 11
    "loo_comparison_5models.csv": (6,),    # Extended Data Fig. 2a
}

MIN_POINTWISE_RHO = 0.80


# %%
# ============================================================================
# Load, correct, check
# ============================================================================
def load_loos(loo_dir=LOO_DIR, models=LOO_MODELS, verbose=True):
    """
    Load one ELPDData per model, applying any corrected session ordering.

    A `<stem>_loo_i_reordered.npy` overwrites the pickle's `loo_i`. Those files
    come from fix_rpe_order.py, for models whose fitting script walked the data
    tree without `sorted()` and so permuted session blocks in the pointwise
    output. `elpd_loo`, `se` and `p_loo` are sums and were always right; `dse`
    and `elpd_diff` are pairwise and were not.

    Args:
        loo_dir: Directory holding `<stem>_loo.pkl` files.
        models: Mapping of pickle stem -> manuscript model number.
        verbose: Print which models had a reordering applied.

    Returns:
        `(loos, pointwise)`, both keyed by model number: the ELPDData objects and
        their pointwise `loo_i` arrays.
    """
    loos, pointwise = {}, {}
    for stem, number in models.items():
        with open(Path(loo_dir) / f"{stem}_loo.pkl", "rb") as fh:
            loo = pickle.load(fh)["loo"]

        reordered = Path(loo_dir) / f"{stem}_loo_i_reordered.npy"
        if reordered.exists():
            fixed, raw = np.load(reordered), np.asarray(loo.loo_i.values)
            if not np.isclose(fixed.sum(), raw.sum()):
                raise ValueError(
                    f"{stem}: reordered elpd {fixed.sum():.2f} != pickle "
                    f"{raw.sum():.2f}. A reordering only permutes values, so the "
                    "sums must match - the .npy is stale or wrong.")
            loo.loo_i.values[:] = fixed
            if verbose:
                print(f"  model {number} ({stem}): applied reordered loo_i")
        elif verbose:
            print(f"  model {number} ({stem}): no reordering needed")

        loos[number] = loo
        pointwise[number] = np.asarray(loo.loo_i.values)
    return loos, pointwise


def check_alignment(pointwise, min_rho=MIN_POINTWISE_RHO):
    """
    Guard against the session-ordering bug, and report the correlation matrix.

    Every model describes the same choices, so their pointwise `loo_i` are
    strongly correlated; a permuted model collapses to rho ~ 0 while leaving
    `elpd_loo` and `se` untouched. The threshold detects permutation, not model
    similarity - aligned values sit at 0.80-0.999, and a model with poor Pareto-k
    lands at the low end because its `loo_i` are noisier.

    Args:
        pointwise: Mapping of model number -> pointwise loo_i array.
        min_rho: Fail below this correlation.

    Returns:
        `(R, worst)`: the correlation matrix and the smallest off-diagonal value.
    """
    lengths = {len(v) for v in pointwise.values()}
    if len(lengths) != 1:
        raise ValueError(f"models have different numbers of trials: {lengths}")

    numbers = list(pointwise)
    R = pd.DataFrame(
        [[float(np.corrcoef(pointwise[a], pointwise[b])[0, 1]) for b in numbers]
         for a in numbers],
        index=numbers, columns=numbers)
    off_diagonal = R.where(~np.eye(len(numbers), dtype=bool))
    worst = float(off_diagonal.min().min())
    if worst < min_rho:
        raise ValueError(
            f"pointwise rho = {worst:.3f} involving model "
            f"{off_diagonal.min().idxmin()} (expected > {min_rho}). "
            "Trial ordering differs between models.")
    return R, worst


def compare_loos(loos, exclude=()):
    """
    Run `az.compare` and return a tidy frame, best model first.

    Args:
        loos: Mapping of model number -> ELPDData.
        exclude: Model numbers to leave out of the comparison.

    Returns:
        DataFrame indexed by model number with `elpd`, `se` and `dse` columns.
    """
    keep = {n: loo for n, loo in loos.items() if n not in exclude}
    cmp = az.compare(keep)

    # Column names moved around between arviz versions.
    elpd_col = next(c for c in ("elpd_loo", "loo", "elpd_waic", "waic")
                    if c in cmp.columns)
    out = pd.DataFrame({
        "elpd": cmp[elpd_col].astype(float),
        "se": cmp["se"].astype(float),
        "dse": cmp["dse"].astype(float),
    })
    out.index = pd.Index([int(i) for i in cmp.index], name="model")
    return out


# %%
# ============================================================================
# Run
# ============================================================================
print("loading LOO pickles:")
loos, pointwise = load_loos()

R, worst_rho = check_alignment(pointwise)
pd.set_option("display.width", 170)
print("\npointwise correlation of loo_i (alignment check):")
print(R.round(3).to_string())
print(f"\nmin pairwise rho = {worst_rho:.4f}")
print(f'  -> methods: "all pairwise rho > {np.floor(worst_rho * 100) / 100:.2f}"')
R.to_csv(LOO_DIR / "loo_pointwise_correlations.csv")

for filename, exclude in VARIANTS.items():
    comparison = compare_loos(loos, exclude)
    comparison.to_csv(LOO_DIR / filename)
    print(f"\n=== {filename} (excluded: {exclude or 'none'}) ===")
    print(comparison.round(2).to_string())

print(f"\nwrote 3 CSVs to {LOO_DIR}")
