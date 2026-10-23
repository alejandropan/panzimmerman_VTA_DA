"""
Extended Data Fig. 2c - the model confusion matrix.

Simulate choices from each of the three fitted models, refit all three models to
each simulated dataset, and ask how well each recovered model predicts the
choices it was fitted to. Nine cells: rows are the generative model, columns the
recovered one.

The nine Stan fits already exist (they are the expensive part - 8 to 200 hours
each). This script does the final step: for every posterior draw of every fit, a
forward pass over the simulated choices, giving log p of each choice; averaged
over draws and over each session's trials, then reported as the mean and s.e.m.
across the 41 sessions. The numbers are written to a CSV that
`extended_data_figure2.py` reads.

Usage:
    python confusion_matrix.py
"""

import hashlib
import os
import re
import time
import warnings
from collections import defaultdict
from pathlib import Path

# One BLAS thread per process, set before numpy is imported because the
# libraries read these at load time. Without it every worker starts its own
# thread pool sized to the whole node, and N_WORKERS x that many threads
# contend for the cores actually allocated - on a cluster node this is the
# difference between a speed-up and a slowdown. Only set if the environment has
# not already chosen, so an explicit `export OMP_NUM_THREADS=2` still wins.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np
import pandas as pd

# `model_comparison_accu` looks parameters up with
# `float(saved_params.loc[saved_params['name'] == ..., 'Mean'])`, which pandas
# deprecates and pandas 3 raises on. Suppressed by message rather than by
# blanket-ignoring FutureWarning, so a different deprecation still gets through.
warnings.filterwarnings(
    "ignore", category=FutureWarning,
    message=r".*Calling float on a single element Series is deprecated.*")

import model_comparison_accu as mc
from ephys_alf_summary import LASER_ONLY_ABB

# Named rather than imported, so the name can be handed to a worker process and
# resolved there. Checked at import so a rename fails here rather than four
# hours into a run.
for _name in ("q_learning_model_reduced_stay_forgetting",
              "reinforce_model_reduced_new", "actor_critic"):
    if not callable(getattr(mc, _name, None)):
        raise ImportError(f"model_comparison_accu has no forward pass {_name}")

# The session-list loader. Added to `model_comparison_accu` during curation so
# the Stan grids could be built from Subjects plus a manifest instead of from a
# second copy of the data; checked here for the same reason as the forward
# passes, and because its absence would otherwise surface as an AttributeError
# after the imports have already cost a minute.
if not callable(getattr(mc, "load_data_sessions_stan", None)):
    raise ImportError(
        "model_comparison_accu has no load_data_sessions_stan - this script "
        "builds its grid from Subjects + LASER_ONLY_ABB rather than from a "
        "copy of the data, and needs that entry point")

# %%
# ============================================================================
# Configuration
# ============================================================================
STAN_ROOT = Path("/jukebox/witten/Alex/Data/ephys_bandit/laser_stan_fits")
SUBJECTS = Path("/jukebox/witten/Alex/Data/Subjects")
if os.name == "nt":
    STAN_ROOT = Path(r"\\cup.princeton.edu\witten\Alex\Data"
                     r"\ephys_bandit\laser_stan_fits")
    SUBJECTS = Path(r"\\cup.princeton.edu\witten\Alex\Data\Subjects")
RECOVERY = STAN_ROOT / "recovery_models"

# The behavioural grid the fits were built on: the 41 sessions of
# `LASER_ONLY_ABB`, read from Subjects.
#
# The original read `ephys_bandit/data_laser_only`, a partial COPY of the
# Subjects tree made so a folder walk would pick up exactly these sessions.
# Two copies is room for drift - `check_data_roots.py` compared them and found
# every modelling field identical across all 41 sessions, with only timing and
# wheel fields differing, and Subjects is the corrected one there. So the
# manifest is now the record of WHICH sessions, and Subjects is the single
# record of what is in them.
LASER_SESSIONS = LASER_ONLY_ABB

# What the nine fits were conditioned on. The recovery fits index their
# parameters by grid slot - `alpha_ses[i]` means "session in slot i" - so a grid
# built from a different session set would silently pair each session with
# another session's parameters. Nothing downstream would notice.
EXPECTED_SESSIONS = 41
EXPECTED_TRIALS = 22406
OUT_CSV = RECOVERY / "confusion_matrix.csv"

# Draws to average log p over. None uses EVERY draw in the run.
N_DRAWS = None

def available_cpus():
    """
    Cores this process may actually use.

    `os.cpu_count()` reports the whole node, which on a cluster is not what the
    job was given: asking SLURM for 8 cores on a 64-core node and then starting
    40 workers oversubscribes by 5x and runs slower than serial.
    `sched_getaffinity` respects the cpuset the scheduler pins the job to, and
    SLURM_CPUS_PER_TASK is checked first because it is the number actually
    requested.
    """
    allocated = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocated and allocated.isdigit():
        return int(allocated)
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1



# The forward pass is the whole cost: 6000 draws x 9 cells = 54,000 passes over
# 22,406 trials, about six hours in one process. Each draw is independent, so
# they parallelise cleanly, therefore we parallelize.
#
# Workers return one number per session rather than one per trial - 41 floats
# instead of 22,406 - which keeps the inter-process traffic negligible. That is
# exact, not an approximation: averaging over draws and over a session's trials
# are both linear, so they commute.
N_WORKERS = available_cpus()

POINT_ESTIMATE = False       # True: one pass per cell at the posterior mean
                             # from summary.csv. About a minute for all nine,
                             # and agrees with the full average to ~0.001.

SUMMARY_TOLERANCE_S = 180    # summary.csv is attributed to the run whose last
                             # chain finished within this of it (In case
                             # several runs were saved)
EXPECTED_CHAINS = 4

QFS, REINFORCE, AC = 1, 4, 5
NAMES = {QFS: "Q-learning", REINFORCE: "REINFORCE", AC: "Actor-Critic"}
ORDER = [QFS, REINFORCE, AC]

# The forward pass for each RECOVERED model. These are the models the rest of
# the figure is about: model 1 is Q-learning with forgetting.
FORWARD = {QFS: "q_learning_model_reduced_stay_forgetting",
           REINFORCE: "reinforce_model_reduced_new",
           AC: "actor_critic"}

# (generative, recovered) -> the fit folder. `confusion_recover_<X>_w_<Y>` has X
# generative and Y recovered; the three diagonal cells are the plain recovery
# folders, which also own the simulated datasets.
PAIRS = {
    (QFS, QFS):             RECOVERY / "standard_w_forgetting_recovery",
    (REINFORCE, REINFORCE): RECOVERY / "REINFORCE_recovery",
    (AC, AC):               RECOVERY / "actor_critic_recovery",
    (AC, QFS):              RECOVERY / "confusion/confusion_recover_AC_w_standard",
    (REINFORCE, QFS):       RECOVERY / "confusion/confusion_recover_REINFORCE_w_standard",
    (AC, REINFORCE):        RECOVERY / "confusion/confusion_recover_AC_w_REINFORCE",
    (QFS, REINFORCE):       RECOVERY / "confusion/confusion_recover_standard_w_REINFORCE",
    (QFS, AC):              RECOVERY / "confusion/confusion_recover_standard_w_AC",
    (REINFORCE, AC):        RECOVERY / "confusion/confusion_recover_REINFORCE_w_AC",
}

# A confusion analysis simulates once per GENERATIVE model and fits all three
# recovered models to that same dataset, so there are three datasets.
# Only the diagonal folders carry `o_sim`/`c_sim`; the six off-diagonal folders
# borrow from here.
SIM_SOURCE = {QFS:       RECOVERY / "standard_w_forgetting_recovery",
              REINFORCE: RECOVERY / "REINFORCE_recovery",
              AC:        RECOVERY / "actor_critic_recovery"}


# %%
# ============================================================================
# Picking one run
# ============================================================================
def runs_in(output_dir):
    """Chain CSVs grouped by the run timestamp cmdstanpy puts in the filename."""
    runs = defaultdict(list)
    for path in output_dir.glob("*.csv"):
        if "summary" in path.name:
            continue
        match = re.search(r"-(\d{8,})_(\d+)$", path.stem)
        if match:
            runs[match.group(1)].append((int(match.group(2)), path))
    for chains in runs.values():
        chains.sort()
    return runs


def summary_of(output_dir):
    for name in ("summary.csv", "summary/summary.csv"):
        candidate = output_dir / name
        if candidate.exists():
            return candidate
    return None


def chains_of_summary_run(directory):
    """
    The chains of the run that `summary.csv` belongs to.

    Returns (run_id, [chain paths], summary path). Raises if the run cannot be
    identified or is incomplete - a partial run must not be silently used.
    """
    output = Path(directory) / "output"
    runs = runs_in(output)
    if not runs:
        raise FileNotFoundError(f"no chain CSVs in {output}")
    summary = summary_of(output)
    if summary is None:
        raise FileNotFoundError(
            f"no summary.csv in {output}, so no run can be attributed - "
            "refusing to pick one by filesystem ordering")

    target = summary.stat().st_mtime
    ends = {run: max(path.stat().st_mtime for _, path in chains)
            for run, chains in runs.items()}
    run, gap = min(((r, abs(target - t)) for r, t in ends.items()),
                   key=lambda pair: pair[1])
    if gap > SUMMARY_TOLERANCE_S:
        raise ValueError(
            f"{output}: summary.csv matches no run's completion time; the "
            f"closest is {run}, {gap / 3600:.1f} h away. Resolve by hand.")

    chains = [path for _, path in runs[run]]
    if len(chains) != EXPECTED_CHAINS:
        raise ValueError(f"{output}: run {run} has {len(chains)} chains, "
                         f"expected {EXPECTED_CHAINS}")
    return run, chains, summary


# %%
# ============================================================================
# The simulated data
# ============================================================================
def load_sim_arrays(directory):
    """
    The simulated dataset one generative model produced.

    Outcomes and choices are required; `l_sim` exists only for the models that
    use a laser and is None otherwise.
    """
    directory = Path(directory)
    l_path = directory / "l_sim.npy"
    return (np.load(directory / "o_sim.npy"),
            np.load(directory / "c_sim.npy"),
            np.load(l_path) if l_path.exists() else None)


def fingerprint(array):
    """Content hash of a per-trial array: raw bytes, plus dtype and shape."""
    array = np.asarray(array)
    if array.ndim < 2:
        return None
    digest = hashlib.sha256()
    digest.update(str((array.dtype.str, array.shape)).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()[:16]


# Outcomes, choices and laser are simulated. Everything else in the grid - the
# block structure `p_left`, the session and mouse indices, `NT_all` - is REAL in
# a simulated fit by design: the simulators replay the real task and only the
# animal's side of it is generated.
OUTCOME_NAMES, CHOICE_NAMES, LASER_NAMES = ("o", "r"), ("c",), ("l",)
BEHAVIOURAL_NAMES = OUTCOME_NAMES + CHOICE_NAMES + LASER_NAMES


def make_grids(sessions=LASER_SESSIONS, subjects=SUBJECTS):
    """
    The behavioural grid, and one grid per generative model with its own choices.
    """
    psy = mc.load_data_sessions_stan(sessions, ROOT_FOLDER=subjects,
                                     trial_start=0, trial_end=None,
                                     expect_sessions=EXPECTED_SESSIONS)
    real = dict(mc.make_stan_data_reduced(psy))

    nt_check = np.asarray(real["NT_all"], dtype=int)
    n_sessions, trials = int(np.sum(nt_check > 0)), int(nt_check.sum())
    if (n_sessions, trials) != (EXPECTED_SESSIONS, EXPECTED_TRIALS):
        raise ValueError(
            f"the grid from {subjects} has {n_sessions} sessions and {trials} "
            f"trials, but the fits were built on {EXPECTED_SESSIONS} and "
            f"{EXPECTED_TRIALS}. Their parameters are indexed by grid slot, so "
            "a different session set would pair each session with another "
            "session's parameters without any error.")

    real_fingerprints = {}
    for key in BEHAVIOURAL_NAMES:
        code = (fingerprint(real[key]) if isinstance(real.get(key), np.ndarray)
                else None)
        if code is not None:
            real_fingerprints[code] = key

    nt_all = np.asarray(real["NT_all"], dtype=int)
    sub = np.asarray(real["sub_idx"]) - 1
    ses = np.asarray(real["sess_idx"]) - 1

    grids = {}
    for generative, source in SIM_SOURCE.items():
        o_sim, c_sim, l_sim = load_sim_arrays(source)
        o_sim, c_sim = np.asarray(o_sim), np.asarray(c_sim)

        expected = (real["NS"], real["NSESS"])
        if o_sim.shape[:2] != expected or c_sim.shape != o_sim.shape:
            raise ValueError(
                f"{NAMES[generative]}: sim arrays are {o_sim.shape} / "
                f"{c_sim.shape} against a grid of {expected} x trials")
        n_trials = int(o_sim.shape[2])
        if n_trials < int(nt_all.max()):
            raise ValueError(
                f"{NAMES[generative]}: sim arrays hold {n_trials} trials but "
                f"the longest session has {int(nt_all.max())} - these were "
                "simulated from different data")


        stray = 0
        for slot in range(real["NSxNSESS"]):
            used = int(nt_all[slot])
            if used == 0 or used >= n_trials:
                continue
            stray += int(np.count_nonzero(c_sim[sub[slot], ses[slot], used:]))
        if stray:
            raise ValueError(
                f"{NAMES[generative]}: {stray} non-zero simulated choice(s) "
                "beyond NT_all within real sessions - the arrays were simulated "
                "with different session lengths than the grid says")

        local = {k: v for k, v in real.items() if k not in BEHAVIOURAL_NAMES}
        local["NT"] = n_trials
        outcomes, choices = o_sim.astype(int), c_sim.astype(int)
        for key in OUTCOME_NAMES:
            local[key] = outcomes
        for key in CHOICE_NAMES:
            local[key] = choices
        if l_sim is not None:
            for key in LASER_NAMES:
                local[key] = np.asarray(l_sim).astype(int)

        for key, value in local.items():
            if (isinstance(value, np.ndarray)
                    and fingerprint(value) in real_fingerprints):
                raise ValueError(
                    f"{NAMES[generative]}: grid key '{key}' is byte-identical "
                    f"to the real '{real_fingerprints[fingerprint(value)]}' "
                    "array, so real behaviour is reaching a simulated fit")
        grids[generative] = local

    return real, grids


# %%
# ============================================================================
# One cell
# ============================================================================
def bracketize(columns):
    """'param.1' -> 'param[1]', the naming the forward passes look up."""
    return [c.replace(".", "[") + "]" if re.search(r"\.\d+", c) else c
            for c in columns]


def chain_draws(directory, n_draws):
    """
    (run id, parameter names, [draws x parameters] array) for the selected run.

    Read with `comment="#"`
    """
    run, chains, _ = chains_of_summary_run(directory)
    output = pd.concat([pd.read_csv(path, comment="#") for path in chains],
                       ignore_index=True)
    output.columns = bracketize(list(output.columns))
    values = output.to_numpy(dtype=float)
    if n_draws is not None and n_draws < len(values):
        # Evenly spaced over the concatenation, so each chain contributes the
        # same number of draws rather than the subsample landing in one chain.
        values = values[np.linspace(0, len(values) - 1, n_draws).astype(int)]
    return run, list(output.columns), values


def posterior_mean(directory):
    """(run id, names, one-row array) - the posterior mean from summary.csv."""
    run, _, summary = chains_of_summary_run(directory)
    frame = pd.read_csv(summary)
    frame = frame.rename(columns={frame.columns[0]: "name"})
    return (run, list(frame["name"].astype(str)),
            pd.to_numeric(frame["Mean"], errors="coerce")
            .to_numpy(dtype=float)[None, :])


def choice_probability(model, grid, names, values):
    """p(chosen) per trial for one draw, and the pass's own frame."""
    params = pd.DataFrame({"name": names, "Mean": values})
    sim = model(grid, saved_params=params)
    raw = sim["predicted_choice"].to_numpy(dtype=float)
    # `predicted_choice` is the probability of one side; flip it where the
    # choice was the other side. The same two lines as `lppd_from_chain`.
    return np.where(sim["choices"].to_numpy() == 0, 1 - raw, raw), sim


def session_index(sim):
    """
    Row -> session slot, and the number of sessions.
    """
    if "id" in sim.columns:
        labels = sim["id"].to_numpy()
    else:
        labels = (sim["mouse"].astype(int) * 1000
                  + sim["ses"].astype(int)).to_numpy()
    codes, _ = pd.factorize(labels, sort=True)
    return codes.astype(np.intp), int(codes.max()) + 1

def session_means(logp, codes, n_sessions):
    """Mean of `logp` within each session, in session-slot order."""
    means = pd.Series(logp).groupby(codes).mean().reindex(range(n_sessions))
    if means.isna().any():
        raise ValueError("a session slot has no trials")
    return means.to_numpy()

def draw_block(task):
    """
    One contiguous block of draws: the running total of per-session mean log p,
    and how many draws went into it.

    """
    model_name, grid, names, values_block, codes, n_sessions = task
    model = getattr(mc, model_name)
    total = np.zeros(n_sessions)
    for row in values_block:
        p, _ = choice_probability(model, grid, names, row)
        if len(p) != len(codes):
            raise RuntimeError(
                f"a draw produced {len(p)} rows, expected {len(codes)} - the "
                "row order is not stable across draws, so averaging "
                "positionally would mix trials")
        with np.errstate(divide="ignore"):
            total += session_means(np.log(p), codes, n_sessions)
    return total, len(values_block)


def blocks_of(values, n_blocks):
    """`values` split into at most `n_blocks` contiguous slices."""
    if len(values) == 0:
        return []
    return [part for part in np.array_split(values, min(n_blocks, len(values)))
            if len(part)]


def progress(iterable, total, label):
    """tqdm if it is installed, a plain periodic line if not."""
    try:
        from tqdm import tqdm
    except ImportError:
        pass
    else:
        return tqdm(iterable, total=total, desc=label, leave=False)

    def ticker():
        step = max(1, total // 20)
        for index, item in enumerate(iterable):
            if index % step == 0:
                print(f"\r   {label}  {index}/{total}", end="", flush=True)
            yield item
        print(f"\r   {label}  {total}/{total}", flush=True)

    return ticker()


def one_cell(generative, recovered, grids):
    """
    Mean per-trial log likelihood per session, for one (generative, recovered)
    cell.

    log p is averaged over draws and over the trials of a session, matching
    Figure 1c, whose LL is `np.log(p_choice)` averaged within a session. The two
    averages commute, so each draw is reduced to one number per session in the
    worker and only those are combined here - exact, and it keeps 22,406 floats
    per draw out of the inter-process queue.

    Returns (mean, s.e.m., per-session values, run id, draws used).
    """
    directory = PAIRS[(generative, recovered)]
    model_name = FORWARD[recovered]
    grid = grids[generative]
    label = f"{NAMES[generative]} -> {NAMES[recovered]}"

    if POINT_ESTIMATE:
        run, names, values = posterior_mean(directory)
    else:
        run, names, values = chain_draws(directory, N_DRAWS)

    # The first draw runs here: it fixes the session layout every worker needs,
    # and it fails in the parent - with a readable traceback - if the forward
    # pass is going to fail at all.
    model = getattr(mc, model_name)
    first_p, first_sim = choice_probability(model, grid, names, values[0])
    codes, n_sessions = session_index(first_sim)
    with np.errstate(divide="ignore"):
        total = session_means(np.log(first_p), codes, n_sessions)

    rest = values[1:]
    if len(rest):
        # A few blocks per worker: the grid is re-sent once per block, which is
        # trivial next to a block's compute, and it keeps the progress bar
        # moving rather than jumping in whole-worker steps.
        parts = blocks_of(rest, max(1, N_WORKERS) * 4)
        tasks = [(model_name, grid, names, part, codes, n_sessions)
                 for part in parts]

        pool = None
        if N_WORKERS > 1 and len(tasks) > 1:
            try:
                import multiprocess as mp
            except ImportError:
                print(f"   {label}: `multiprocess` not installed, running "
                      "serially")
            else:
                pool = mp.Pool(processes=min(N_WORKERS, len(tasks)))
        if pool is None:
            for task in progress(tasks, len(tasks), label):
                block_total, _ = draw_block(task)
                total += block_total
        else:
            with pool:
                for block_total, _ in progress(
                        pool.imap_unordered(draw_block, tasks),
                        len(tasks), label):
                    total += block_total

    per_session = total / len(values)
    if not np.all(np.isfinite(per_session)):
        raise ValueError(f"{label}: non-finite log likelihood in "
                         f"{int(np.sum(~np.isfinite(per_session)))} session(s)")
    return (float(np.mean(per_session)),
            float(np.std(per_session, ddof=1) / np.sqrt(len(per_session))),
            per_session, run, len(values))


# %%
# ============================================================================
# Compute
# ============================================================================
print(f"looking under {RECOVERY}")
print(f"{'posterior mean' if POINT_ESTIMATE else (str(N_DRAWS) + ' draws') if N_DRAWS else 'every draw'} "
      f"per cell, {N_WORKERS} worker(s)")
real_standata, GRIDS = make_grids()
print(f"grid: {int(np.sum(np.asarray(real_standata['NT_all']) > 0))} sessions, "
      f"{int(np.sum(np.asarray(real_standata['NT_all'])))} trials")
print(f"simulated datasets: {[NAMES[g] for g in GRIDS]}\n")

# The three datasets must differ, or the off-diagonal cells compare nothing.
codes = {NAMES[g]: fingerprint(GRIDS[g]["c"]) for g in GRIDS}
if len(set(codes.values())) != len(codes):
    raise SystemExit(f"two generative models share simulated choices: {codes}")
print(f"simulated choice datasets, all distinct: {codes}\n")

rows = []
started = time.time()
for generative in ORDER:
    for recovered in ORDER:
        mean, sem, _, run, used = one_cell(generative, recovered, GRIDS)
        rows.append({"generative": generative, "recovered": recovered,
                     "generative_name": NAMES[generative],
                     "recovered_name": NAMES[recovered],
                     "mean_loglik": mean, "sem_loglik": sem,
                     "run": run, "n_draws": used})
        print(f"  {NAMES[generative]:12s} -> {NAMES[recovered]:12s} "
              f"{mean:+.4f} +/- {sem:.4f}   run {run}, {used} draw(s)")
print(f"\n{time.time() - started:.0f}s")

table = pd.DataFrame(rows)
table.to_csv(OUT_CSV, index=False)
print(f"wrote {OUT_CSV}")