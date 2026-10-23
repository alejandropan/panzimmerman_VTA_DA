"""
PSIS-LOO for each model, with the session ordering fixed at source.

Replaces PSISLOO.py, PSISLOO_W_SEQUENTIAL.py, PSISLOO_W_SEQUENTIAL_fixed.py and
fix_rpe_order.py. Writes one pickle per model plus a summary CSV; the figure
reads those through compute_loo_comparison.py and never touches a posterior.

Usage (interactive, cell by cell), or:
    python compute_loo.py
"""

import os
import pickle
from datetime import datetime, timezone
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

# ── configuration ───────────────────────────────────────────────────────────
STAN_ROOT = Path("/jukebox/witten/Alex/Data/ephys_bandit/laser_stan_fits")
SUBJECTS = Path("/jukebox/witten/Alex/Data/Subjects")
if os.name == "nt":
    STAN_ROOT = Path(r"\\cup.princeton.edu\witten\Alex\Data"
                     r"\ephys_bandit\laser_stan_fits")
    SUBJECTS = Path(r"\\cup.princeton.edu\witten\Alex\Data\Subjects")

LOO_DIR = STAN_ROOT / "loo_results"
CODE_DIR = (Path(__file__).resolve().parent if "__file__" in globals()
            else Path.cwd())
METADATA_DIRS = [CODE_DIR / "metadata", LOO_DIR]
REQUIRES_REORDER = {"Standard_Forgetting_RPE_laser"} # this model was fit on 
#the same sessions but ordered differently so we need a small reordering before proceding

from ephys_alf_summary import LASER_ONLY_ABB                    
LASER_SESSIONS = LASER_ONLY_ABB
N_SESSIONS = 41

K_BAD = 0.7                          
MIN_RHO = 0.80     # This is just to check that there no trial ordering problems                     


# Sampler diagnostics, never part of the posterior.
SAMPLER_COLS = {"lp__", "accept_stat__", "stepsize__", "treedepth__",
                "n_leapfrog__", "divergent__", "energy__", "trial_idx"}


MODELS = {
    "Standard_Forgetting":           (1, "manuscript", "standard_w_forgetting", "log_lik"),
    "Standard_Forgetting_RPE_laser": (2, "manuscript", "standard_w_forgetting_RPE_laser", "log_lik"),
    "QFS_Variable_Alpha":            (3, "manuscript", "QFS_variable_alpha", "log_lik"),
    "REINFORCE":                     (4, "manuscript", "REINFORCE_new", "log_lik"),
    "actor_critic":                  (5, "manuscript", "actor_critic", "log_lik"),
    "sequential_actor_critic":       (6, "manuscript",
                                      "sequential_actor_critic_no_awin_m_v_choice",
                                      "log_lik"),
}


# ── the padded session grid ─────────────────────────────────────────────────
def load_nt_all(sessions=LASER_SESSIONS, subjects=SUBJECTS):
    """
    Stan's per-slot trial counts.

    The grid is mice x session slots with most slots empty; NT_all[i] == 0 is
    the only authoritative marker of a padding slot. Everything about layout and
    ordering below is checked against this rather than against a hardcoded trial
    count.
    """
    import model_comparison_accu as mc
    standata = mc.make_stan_data_reduced(mc.load_data_sessions_stan(
        sessions, ROOT_FOLDER=subjects, trial_start=0, trial_end=None,
        expect_sessions=N_SESSIONS))
    nt = np.asarray(standata["NT_all"], dtype=int)
    return nt, standata


def chain_files(output_dir):
    """
    The per-chain CSVs, ordered by chain number.

    Sorted numerically rather than lexically: cmdstanpy names them
    `<model>-<timestamp>_<chain>.csv`, so a plain sort would put chain 10 before
    chain 2 as soon as a fit uses ten or more chains. Chain order does not
    affect ELPD - ESS is symmetric across chains - but a stable, correct order
    keeps the saved n_chains and any per-chain debugging honest.

    The glob is not recursive and excludes anything named "summary", because
    some fits leave a cmdstan summary in `output/` and others in
    `output/summary/`.
    """

    def chain_number(path):
        tail = path.stem.rsplit("_", 1)[-1]
        return int(tail) if tail.isdigit() else -1

    files = sorted((p for p in Path(output_dir).glob("*.csv")
                    if "summary" not in p.name),
                   key=lambda p: (chain_number(p), p.name))
    if not files:
        raise FileNotFoundError(f"no chain CSVs in {output_dir}")
    return files


def split_columns(csv_path, ll_stem):
    """
    Column names for the log-likelihood and for the posterior, kept separate.

    Matching is on the exact stem: a column is log_lik only if the part before
    its first index separator equals `ll_stem`. `startswith` would also take
    log_lik_c and log_lik_rep, and concatenating two log-likelihood variables
    produces a vector that passes every length check while being meaningless.
    """
    def stem_of(column):
        return column.split(".")[0].split("[")[0]

    header = pd.read_csv(csv_path, comment="#", nrows=0).columns.tolist()
    log_lik_like = {stem_of(c) for c in header if "log_lik" in stem_of(c)}

    ll = [c for c in header if stem_of(c) == ll_stem]
    if not ll:
        raise ValueError(f"{csv_path.name}: no column with stem '{ll_stem}'. "
                         f"log_lik-like stems present: "
                         f"{sorted(log_lik_like) or 'none'}")
    ignored = sorted(log_lik_like - {ll_stem})
    if ignored:
        print(f"    note: ignoring other log-likelihood variables {ignored}")

    posterior = [c for c in header if c not in SAMPLER_COLS
                 and stem_of(c) not in log_lik_like]
    return ll, posterior


def detect_layout(finite, nt):
    """
    Which of the three known layouts the log_lik vector has, and the index that
    puts its real entries in canonical session-major order.

    COMPACT       length == sum(NT_all), no padding. Already in fit order.
    TAIL          oversized, every real entry at the front. The Stan code filled
                  it with a running counter and the declared size is an upper
                  bound. Trimming the tail preserves fit order.
    SESSION_GRID  the full slots x max-trials grid, entry (m-1)*TMAX + (t-1)
                  real iff t <= NT_all[m]. Compacting in column order gives
                  session-major directly.
    """
    n_slots, tmax, total = len(nt), int(nt.max()), int(nt.sum())
    n = len(finite)

    if n == total and finite.all():
        return "COMPACT", np.arange(n)

    if int(finite.sum()) != total:
        raise ValueError(
            f"{int(finite.sum())} finite entries but NT_all sums to {total}. "
            "Either padding is written as a real number rather than NaN, or "
            "this is not the session grid - do not compact this blindly.")

    tail = np.zeros(n, dtype=bool)
    tail[:total] = True
    if np.array_equal(finite, tail):
        return "TAIL", np.arange(total)

    if n == n_slots * tmax:
        grid = np.zeros(n, dtype=bool)
        for m in range(1, n_slots + 1):
            grid[(m - 1) * tmax:(m - 1) * tmax + nt[m - 1]] = True
        if np.array_equal(finite, grid):
            return "SESSION_GRID", np.flatnonzero(finite)

    runs = 1 + int((np.diff(finite.astype(int)) != 0).sum())
    raise ValueError(
        f"unrecognised log_lik layout: {n} columns, {total} finite in ~{runs} "
        f"runs, grid would be {n_slots} x {tmax}. Inspect the Stan file's "
        "log_lik indexing before trusting this model's pointwise output.")


def session_order_index(name, nt, metadata_dirs=METADATA_DIRS):
    """
    Index that reorders a fit's own session order into canonical slot order.

    Returns None when there is no order file, meaning the fit already assembled
    its sessions in canonical order and nothing needs moving.
    """
    path = next((Path(directory) / f"{name}_fit_session_order.csv"
                 for directory in metadata_dirs
                 if (Path(directory) / f"{name}_fit_session_order.csv").exists()),
                None)
    if path is None:
        return None

    order = pd.read_csv(path).sort_values("fit_position")
    walk = order["canonical_slot"].tolist()
    slots = [i + 1 for i in range(len(nt)) if nt[i] > 0]
    if sorted(walk) != slots:
        raise ValueError(f"{path.name}: covers slots {sorted(walk)[:5]}... but "
                         f"NT_all says the real slots are {slots[:5]}...")
    declared = order["n_trials"].tolist()
    actual = [int(nt[s - 1]) for s in walk]
    if declared != actual:
        raise ValueError(f"{path.name}: n_trials disagrees with NT_all - the "
                         "order file was frozen against different data.")

    offset, start = {}, 0
    for slot in walk:
        offset[slot] = start
        start += int(nt[slot - 1])
    index = np.concatenate([np.arange(offset[s], offset[s] + int(nt[s - 1]))
                            for s in slots])
    print(f"    session order: applying {path.name} "
          f"({int((index != np.arange(len(index))).sum())} of {len(index)} "
          "positions move)")
    return index


# ── reading one model ───────────────────────────────────────────────────────
def read_chains(files, ll_cols, posterior_cols):
    """log_lik as (chain, draw, column) and the posterior as (chain, draw, param)."""
    ll_chunks, post_chunks = [], []
    for path in files:
        frame = pd.read_csv(path, comment="#", usecols=ll_cols + posterior_cols)
        ll_chunks.append(frame[ll_cols].to_numpy(dtype=np.float64))
        post_chunks.append(frame[posterior_cols].to_numpy(dtype=np.float64))
        del frame
    return np.stack(ll_chunks), np.stack(post_chunks)


def run_model(name, nt, loo_dir=LOO_DIR):
    """Compute LOO for one model and write its pickle. Returns a summary row."""
    number, group, folder, ll_stem = MODELS[name]
    print(f"\n[{number}] {name}")
    if folder is None:
        raise FileNotFoundError(
            f"{name}: no cmdstan output directory configured - fill it into "
            "MODELS before running this model.")

    files = chain_files(STAN_ROOT / folder / "output")
    ll_cols, posterior_cols = split_columns(files[0], ll_stem)

    ll, posterior = read_chains(files, ll_cols, posterior_cols)

    finite = np.isfinite(ll).all(axis=(0, 1))
    layout, keep = detect_layout(finite, nt)
    ll = ll[:, :, keep]
    print(f"    layout {layout} -> {ll.shape[2]} observations")

    reorder = session_order_index(name, nt)
    if reorder is None and name in REQUIRES_REORDER:
        raise FileNotFoundError(
            f"{name} needs {name}_fit_session_order.csv - its sessions are "
            "stored in a permuted order and without the file the reordering "
            "would be skipped silently, giving a wrong ELPD. Searched:\n  "
            + "\n  ".join(str(Path(d)) for d in METADATA_DIRS)
            + "\nCopy the file from the LOO results folder; do not re-run "
              "freeze_session_order.py, which needs artifacts that may be gone.")
    if reorder is not None:
        ll = ll[:, :, reorder]

    idata = az.from_dict(
        log_likelihood={"log_lik": xr.DataArray(
            ll, dims=["chain", "draw", "log_lik_dim_0"])},
        posterior={c: xr.DataArray(posterior[:, :, i], dims=["chain", "draw"])
                   for i, c in enumerate(posterior_cols)})
    n_chains, n_draws = ll.shape[0], ll.shape[1]
    del posterior

    loo = az.loo(idata, var_name="log_lik", pointwise=True)
    pointwise = np.asarray(loo.loo_i.values, dtype=float)
    k = np.asarray(loo.pareto_k.values, dtype=float)
    n_bad = int((k > K_BAD).sum())

    print(f"    ELPD = {loo.elpd_loo:.2f} +/- {loo.se:.2f}   "
          f"p_loo = {loo.p_loo:.2f}")
    if n_bad:
        worst = np.argsort(k)[::-1][:n_bad]
        print(f"    *** {n_bad} observation(s) with Pareto-k > {K_BAD} ")
    else:
        print(f"    no Pareto-k > {K_BAD} (max {k.max():.3f})")

    out_path = Path(loo_dir) / f"{name}_loo.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump({
            "loo": loo,
            "model": name,
            "number": number,
            "group": group,
            "ll_stem": ll_stem,
            "layout": layout,
            "n_obs": int(len(pointwise)),
            "n_chains": int(n_chains),
            "n_draws": int(n_draws),
            "reordered": reorder is not None,
            "order_source": (f"{name}_fit_session_order.csv"
                             if reorder is not None else "canonical"),
            "posterior_columns": len(posterior_cols),
            "data_path": str(SUBJECTS),
            "data_sessions": len(LASER_SESSIONS),
            "output_dir": str(STAN_ROOT / folder / "output"),
            "computed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, fh)

    return {"model": name, "number": number, "group": group, "status": "ok",
            "elpd": float(loo.elpd_loo), "se": float(loo.se),
            "p_loo": float(loo.p_loo), "bad_k": n_bad,
            "ok_k": int(((k > 0.5) & (k <= K_BAD)).sum()),
            "k_max": float(k.max()), "n_obs": int(len(pointwise)),
            "layout": layout, "reordered": reorder is not None,
            "out_path": str(out_path)}


def check_alignment(pointwise, min_rho=MIN_RHO):
    """
    Every model describes the same choices, so their pointwise loo_i must be
    strongly correlated.

    A model whose columns ended up in a different order collapses to rho ~ 0
    while its elpd_loo and se look perfectly normal, because those are
    order-invariant sums. dse and elpd_diff are pairwise and are what a
    permutation corrupts.

    Blame is assigned by counting how many others each model disagrees with: by
    each model's worst correlation alone, every model looks guilty, because an
    aligned model's worst partner is the permuted one.
    """
    numbers = sorted(pointwise)
    R = pd.DataFrame([[float(np.corrcoef(pointwise[a], pointwise[b])[0, 1])
                       for b in numbers] for a in numbers],
                     index=numbers, columns=numbers)
    off = R.where(~np.eye(len(numbers), dtype=bool))
    counts = {n: int((off.loc[n] < min_rho).sum()) for n in numbers}
    culprits = [n for n, c in counts.items() if c == len(numbers) - 1]
    return R, culprits


# %% ── run ───────────────────────────────────────────────────────────────────
NT, STANDATA = load_nt_all()
print(f"{int((NT > 0).sum())} real sessions of {len(NT)} slots, "
      f"{int(NT.sum())} trials")

rows, pointwise = [], {}
for model_name in MODELS:
    try:
        rows.append(run_model(model_name, NT))
        with open(rows[-1]["out_path"], "rb") as handle:
            pointwise[rows[-1]["number"]] = np.asarray(
                pickle.load(handle)["loo"].loo_i.values, dtype=float)
    except Exception as error:                                    # noqa: BLE001
        print(f"    FAILED: {error}")
        number, group = MODELS[model_name][0], MODELS[model_name][1]
        rows.append({"model": model_name, "number": number, "group": group,
                     "status": "failed", "error": str(error)})

summary = pd.DataFrame(rows)
summary.to_csv(LOO_DIR / "loo_summary.csv", index=False)

print("\n── summary ─────────────────────────────────────────────────────────")
pd.set_option("display.width", 200)
succeeded = summary[summary["status"] == "ok"].sort_values(
    ["group", "elpd"], ascending=[True, False])
if len(succeeded):
    # Printed separately from the failures so the integer columns stay integers:
    # a failed row has no n_obs or bad_k, and one NaN upcasts the whole column.
    counts = ["bad_k", "ok_k", "n_obs", "number"]
    print(succeeded.astype({c: int for c in counts}).drop(
        columns=["status", "out_path", "error"], errors="ignore"
    ).to_string(index=False))
failed = summary[summary["status"] == "failed"]
if len(failed):
    print(f"\n{len(failed)} model(s) failed:")
    for _, row in failed.iterrows():
        print(f"  {row['error']}")
print(f"\nwrote {LOO_DIR / 'loo_summary.csv'}")

# %% ── alignment guard ───────────────────────────────────────────────────────
# Models in different groups are fitted to different trials, so only compare
# within a group.
for group_name in sorted({r["group"] for r in rows if r["status"] == "ok"}):
    members = {r["number"]: pointwise[r["number"]] for r in rows
               if r["status"] == "ok" and r["group"] == group_name}
    if len(members) < 2:
        continue
    if len({len(v) for v in members.values()}) > 1:
        raise ValueError(f"group '{group_name}': models have different "
                         f"observation counts {[len(v) for v in members.values()]}")
    R, culprits = check_alignment(members)
    print(f"\npointwise correlation of loo_i, group '{group_name}':")
    print(R.round(3).to_string())
    if culprits:
        raise ValueError(
            f"model(s) {culprits} are in a different trial order from the rest. "
            "Their elpd_loo and se are still valid, but every dse and "
            "elpd_diff involving them is meaningless. Check the layout and the "
            "session order file before using these results.")
    print(f"  all aligned, min pairwise rho = "
          f"{float(R.where(~np.eye(len(R), dtype=bool)).min().min()):.4f}")
