"""
Posterior decision variables for all six models, written per session.

Outputs land in each session's alf folder as `<prefix>_mean_<name>.npy` and
`<prefix>_median_<name>.npy`. Extended Data Fig. 2b reads QFS_median_QR/QL and
REINFORCE/AC_median_reward; Fig. 1c,d read `<prefix>_median_choice_prediction`.

Usage (interactive), or:
    python extract_decision_variables.py
"""

import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ephys_alf_summary import LASER_ONLY_ABB
from model_comparison_accu import (
    load_data_sessions_stan, make_stan_data_reduced,
    q_learning_model_reduced_stay_forgetting,
    q_learning_model_reduced_stay_forgetting_RPE_laser,
    q_learning_model_reduced_stay_forgetting_variable_alpha,
    reinforce_model_reduced_new, actor_critic,
    sequential_actor_critic_forward
)

warnings.filterwarnings("ignore")

# ── configuration ───────────────────────────────────────────────────────────
STAN_ROOT = Path("/jukebox/witten/Alex/Data/ephys_bandit/laser_stan_fits")
SUBJECTS = Path("/jukebox/witten/Alex/Data/Subjects")
if os.name == "nt":
    STAN_ROOT = Path(r"\\cup.princeton.edu\witten\Alex\Data"
                     r"\ephys_bandit\laser_stan_fits")
    SUBJECTS = Path(r"\\cup.princeton.edu\witten\Alex\Data\Subjects")

N_SESSIONS = 41

N_WORKERS = min(40, os.cpu_count() or 1)

# Which models to run. Subset this when working interactively.
RUN = ["QFS", "QFS_RPE_laser", "QFS_variable_alpha", "REINFORCE", "AC",
       "SeqAC_noawin_m_avchoice"]

# Identity columns, constant across draws, taken from one template draw.
REQUIRED_ID = ["mouse", "ses", "id", "choices"]
OPTIONAL_ID = ["trial", "outcomes", "licks"]


# ── the six models ──────────────────────────────────────────────────────────
# `outputs` maps the saved file suffix onto the forward pass's column name, so
# `QFS` writes QFS_median_QR.npy from the column `QR`.

Q_OUTPUTS = {
    "V": "V", "choice_prediction": "predicted_choice",
    "QL": "QL", "QR": "QR",
    "QLstay": "QLstay", "QRstay": "QRstay",
    "QLreward": "QLreward", "QRreward": "QRreward",
}
PROPENSITY_OUTPUTS = {
    "choice_prediction": "predicted_choice",
    "reward": "deltaQ",
}

MODELS = {
    "QFS": dict(
        number=1, folder="standard_w_forgetting",
        forward=q_learning_model_reduced_stay_forgetting,
        params_as="frame", outputs=dict(Q_OUTPUTS)),
    "QFS_RPE_laser": dict(
        number=2, folder="standard_w_forgetting_RPE_laser",
        forward=q_learning_model_reduced_stay_forgetting_RPE_laser,
        params_as="frame", outputs=dict(Q_OUTPUTS)),
    "QFS_variable_alpha": dict(
        number=3, folder="QFS_variable_alpha",
        forward=q_learning_model_reduced_stay_forgetting_variable_alpha,
        params_as="frame",
        outputs={**Q_OUTPUTS, "var_alpha": "var_alpha",
                 "effective_alpha": "effective_alpha", "fixed_alpha": "alpha"}),
    "REINFORCE": dict(
        number=4, folder="REINFORCE_new",
        forward=reinforce_model_reduced_new,
        params_as="frame", outputs=dict(PROPENSITY_OUTPUTS)),
    "AC": dict(
        number=5, folder="actor_critic",
        forward=actor_critic,
        params_as="frame", outputs={**PROPENSITY_OUTPUTS, "V": "V"}),
    "SeqAC_noawin_m_avchoice": dict(
        number=6, folder="sequential_actor_critic_no_awin_m_v_choice",
        forward=sequential_actor_critic_forward,                      # set below, once it is defined
        params_as="dict", standata_alias={"o": "r"},
        outputs={"V_cs": "V_cs", "V_choice": "V_choice",
                 "pi_choice": "pi_choice", "pi_lick": "pi_lick",
                 "delta_cs": "delta_cs", "delta_laser": "delta_laser",
                 "p_chosen": "p_chosen", "p_lick": "p_lick",
                 "choice_prediction": "predicted_choice",
                 "lick_prediction": "predicted_lick",
                 "side_bias": "side_bias"}),
}


# ── loading the posterior ───────────────────────────────────────────────────
def chain_files(folder, n_expected=4):
    """
    The per-chain CSVs for the single run in a fit folder.

    cmdstanpy writes `<model>-<timestamp>_<chain>.csv`. Grouped by timestamp and
    sorted by chain number, so the selection is explicit rather than resting on
    directory order, and a second run in the folder is an error instead of a
    silent mixture.
    """
    out = Path(folder) / "output"
    csvs = [p for p in out.glob("*.csv") if "summary" not in p.name]
    if not csvs:
        raise FileNotFoundError(f"no chain CSVs in {out}")

    runs = {}
    for path in csvs:
        match = re.search(r"-(\d{8,})_(\d+)$", path.stem)
        runs.setdefault(match.group(1) if match else "unnamed", []).append(path)
    if len(runs) > 1:
        raise ValueError(
            f"{out} holds {len(runs)} runs {sorted(runs)}. Pick one explicitly - "
            "the original code resolved each chain with its own glob and could "
            "mix chains across runs.")

    files = sorted(next(iter(runs.values())),
                   key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
    if n_expected is not None and len(files) != n_expected:
        raise ValueError(f"{out}: {len(files)} chains, expected {n_expected}")
    return files


def bracketize(columns):
    """
    'param.1' -> 'param[1]', the naming the forward passes look up.

    Replicates the originals' transform exactly: any name containing '.<digits>'
    has every '.' replaced by '[' and a ']' appended. That is only correct for
    one index - 'x.1.2' would become 'x[1[2]' - so a two-index name is reported
    rather than mangled quietly.
    """
    out, suspicious = [], []
    for column in columns:
        if re.search(r"\.\d+", column):
            if len(re.findall(r"\.\d+", column)) > 1:
                suspicious.append(column)
            out.append(column.replace(".", "[") + "]")
        else:
            out.append(column)
    if suspicious:
        raise ValueError(
            f"{len(suspicious)} parameter name(s) carry more than one index, "
            f"e.g. {suspicious[:3]}. The originals' bracketizing mangles these "
            "('x.1.2' -> 'x[1[2]'), so the forward pass would look up a name "
            "that does not exist. Handle these explicitly before proceeding.")
    return out


def load_posterior(folder):
    """All chains as one frame, with parameter names already bracketized."""
    files = chain_files(folder)
    frames = []
    for index, path in enumerate(files):
        chain = pd.read_csv(path, comment="#")
        chain["chain"] = index + 1
        frames.append(chain)
        print(f"      {path.name}: {len(chain)} draws")
    output = pd.concat(frames, ignore_index=True)

    objects = output.dtypes[output.dtypes == object]
    if len(objects):
        raise RuntimeError(f"non-numeric columns (parse failure): "
                           f"{objects.index.tolist()[:10]}")
    output.columns = bracketize(list(output.columns))
    return output


# ── one draw ────────────────────────────────────────────────────────────────
_STATE = {}          # set once per worker by the Pool initializer


def init_worker(standata, forward, params_as, names, value_cols):
    _STATE.update(standata=standata, forward=forward, params_as=params_as,
                  names=names, value_cols=value_cols)


def simulate_draw(values):
    """
    One posterior draw's decision variables, as a plain (rows, cols) array.

    Row order is identical for every draw because it is driven entirely by
    standata, which is what lets the caller stack draws positionally instead of
    concatenating and grouping.
    """
    if _STATE["params_as"] == "dict":
        params = dict(zip(_STATE["names"], values))
    else:
        params = pd.DataFrame({"name": _STATE["names"], "Mean": values})

    sim = _STATE["forward"](_STATE["standata"], saved_params=params)
    left = sim["choices"] == 0
    sim.loc[left, "predicted_choice"] = 1 - sim.loc[left, "predicted_choice"]
    return sim[_STATE["value_cols"]].to_numpy(dtype=np.float64)


def validate_columns(sim, outputs, name):
    """
    Check the declared outputs against what the forward pass actually returns.

    Fails before the expensive loop and names the real columns, so a wrong guess
    in MODELS costs seconds. Also reports numeric columns that exist but are not
    declared, because those are decision variables being silently dropped.
    """
    missing = [c for c in outputs.values() if c not in sim.columns]
    if missing:
        raise ValueError(
            f"{name}: declared output column(s) {missing} are not returned by "
            f"the forward pass. It returned: {sorted(sim.columns)}. Fix "
            f"MODELS['{name}']['outputs'].")

    known = set(outputs.values()) | set(REQUIRED_ID) | set(OPTIONAL_ID) | {"acc"}
    extra = sorted(c for c in sim.select_dtypes("number").columns
                   if c not in known)
    if extra:
        print(f"      note: not saved: {extra}")


# ── saving ──────────────────────────────────────────────────────────────────
def save_per_session(psy, data, prefix, outputs, subjects=SUBJECTS):
    """
    Write one .npy per declared output per session, into that session's alf.

    The session walk and its assertions are: iterate mouse, date,
    session in sorted order, keep a running `counter` that indexes `ses`, and
    check the stored choices against `_ibl_trials.choice.npy` before writing
    anything. That check is what guarantees a  row block really belongs to the
    session it is about to be written into.
    """
    written = 0
    for ns, mouse in enumerate(sorted(psy["mouse"].unique())):
        animal = psy.loc[psy["mouse"] == mouse]
        counter = 0
        for day in sorted(animal["date"].unique()):
            day_s = animal.loc[animal["date"] == day]
            for ses in sorted(day_s["ses"].unique()):
                session = day_s.loc[day_s["ses"] == ses]
                alf = subjects / mouse / day / ses / "alf"

                ses_data = data.loc[(data["mouse"] == ns)
                                    & (data["ses"] == counter)]
                counter += 1
                if len(ses_data) != len(session):
                    raise ValueError(f"{alf}: {len(ses_data)} rows vs "
                                     f"{len(session)} trials")

                choice = 1 * (np.load(alf / "_ibl_trials.choice.npy") == -1)
                if len(choice) != len(ses_data):
                    raise ValueError(f"{alf}: {len(choice)} choices vs "
                                     f"{len(ses_data)} rows")
                if not np.array_equal(choice, ses_data["choices"].to_numpy(),
                                      equal_nan=True):
                    raise ValueError(f"{alf}: choices do not match the session")

                for suffix, column in outputs.items():
                    np.save(alf / f"{prefix}_{suffix}.npy",
                            ses_data[column].to_numpy())
                written += 1
    return written


# %% ── run ───────────────────────────────────────────────────────────────────
from multiprocess import Pool
from tqdm import tqdm

psy = load_data_sessions_stan(LASER_ONLY_ABB, ROOT_FOLDER=SUBJECTS,
                              trial_start=0, trial_end=None,
                              expect_sessions=N_SESSIONS)
base_standata = make_stan_data_reduced(psy)
print(f"{len(psy)} trials, "
      f"{int((np.asarray(base_standata['NT_all']) > 0).sum())} sessions")

for name in RUN:
    spec = MODELS[name]
    print(f"\n[{spec['number']}] {name}  ({spec['folder']})")

    standata = dict(base_standata)
    for target, source in spec.get("standata_alias", {}).items():
        # SeqAC was written with "outcome" terminology, the others with "reward".
        standata[target] = standata[source]

    output = load_posterior(STAN_ROOT / spec["folder"])
    names = list(output.columns)
    n_draws = len(output)
    print(f"      {n_draws} draws total")

    outputs = spec["outputs"]
    value_cols = sorted(set(outputs.values()))

    # Template draw: fixes the row order, the identity columns, and validates
    # the declared outputs - all before the expensive loop starts.
    first = output.iloc[0].to_numpy(dtype=np.float64)
    params0 = (dict(zip(names, first)) if spec["params_as"] == "dict"
               else pd.DataFrame({"name": names, "Mean": first}))
    template_sim = spec["forward"](standata, saved_params=params0)
    validate_columns(template_sim, outputs, name)

    id_cols = REQUIRED_ID + [c for c in OPTIONAL_ID if c in template_sim.columns]
    template = template_sim[id_cols].reset_index(drop=True)
    n_rows, n_cols = len(template), len(value_cols)
    if n_rows != len(psy):
        raise ValueError(f"{name}: forward pass returned {n_rows} rows, "
                         f"the behaviour has {len(psy)}")

    # Every draw's decision variables, stacked along a new first axis. float32
    # halves the footprint and is well below the posterior's own uncertainty.
    # np.nanmedian peaks at about three times the stack, so ask the scheduler
    # for roughly 3x the figure printed here.
    stack = np.empty((n_draws, n_rows, n_cols), dtype=np.float32)
    print(f"      {n_rows} rows x {n_cols} columns x {n_draws} draws "
          f"= {stack.nbytes / 1e9:.1f} GB float32")

    rows_iter = (output.iloc[i].to_numpy(dtype=np.float64) for i in range(n_draws))
    with Pool(N_WORKERS, initializer=init_worker,
              initargs=(standata, spec["forward"], spec["params_as"],
                        names, value_cols)) as pool:
        for k, array in enumerate(tqdm(
                pool.imap(simulate_draw, rows_iter, chunksize=50),
                total=n_draws, desc=f"      {name}")):
            if array.shape != (n_rows, n_cols):
                raise RuntimeError(
                    f"draw {k} produced {array.shape}, expected "
                    f"{(n_rows, n_cols)} - row alignment is not holding, so "
                    "stacking positionally would mix trials between draws.")
            stack[k] = array

    # Collapse the draws. nan-aware because the SeqAC forward pass leaves
    # predicted_lick, p_lick and delta_laser as NaN on unrewarded trials, and

    mean_array = np.nanmean(stack, axis=0, dtype=np.float64)
    median_array = np.nanmedian(stack, axis=0)
    del stack

    for statistic, array in (("mean", mean_array), ("median", median_array)):
        frame = pd.concat([template.copy(),
                           pd.DataFrame(array, columns=value_cols)], axis=1)
        for column in ("choices", "outcomes", "licks"):
            if column in frame.columns:
                frame[column] = frame[column].astype(int)
        written = save_per_session(psy, frame, f"{name}_{statistic}", outputs)
        print(f"      wrote {len(outputs)} files x {written} sessions "
              f"as {name}_{statistic}_*.npy")

print("\nDone.")