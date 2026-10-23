"""
Turn each decoder's per-recording `.npy` files into the two summary CSVs the
figure scripts read.

    python compile_decoders.py                        # verify all, write none
    python compile_decoders.py --write                # regenerate all
    python compile_decoders.py --write relative_value # regenerate just one

Replaces `compile_nulls_and_real_slim.py` and `compile_vcs_residuals.py`,
which held byte-identical copies of `load_decoders` and differed only in a
list of paths, most of them commented out.

Per spec it writes:

    summary_102030405060.csv    the real fits, all six neuron counts
    nsummary_<null_n>.csv       the 200 nulls, at this spec's neuron count

Unlike everything upstream, this step is **fully checkable**: the `.npy`
files still exist, so the CSVs can be regenerated and compared against the
ones that were shipped. That is what the default does, and it is worth
running before `--write` ever is.
"""

import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from decoder_specs import REAL_NEURON_COMBOS, SPECS

# Which decoders to compile. Every spec unless some are named on the command
# line. Naming them matters for `--write`: regenerating a CSV that is already
# correct is harmless, but a CSV deliberately kept as it was shipped must not
# be caught up in a blanket rewrite.
RUN = [argument for argument in sys.argv[1:] if not argument.startswith('--')]
unknown = [name for name in RUN if name not in SPECS]
if unknown:
    raise SystemExit(f"unknown spec(s): {', '.join(unknown)}. "
                     f"Known: {', '.join(SPECS)}")
RUN = RUN or list(SPECS)

N_NULLS_EXPECTED = 200

# `real_<region>_<mouse>_<date>_<hem>_p_summary.npy`, and for a null the same
# with `<iter>_null` in front. The original sliced this out by fixed character
# offsets counted from the end of the string, which works only because a
# mouse name is always six characters and a date always ten. Same fields, but
# it says what it expects and an unrecognised name raises instead of being
# silently mis-parsed.
SUMMARY_NAME = re.compile(
    r'^(?:(?P<iteration>\d+)_null|real)'
    r'_(?P<region>.+?)'
    r'_(?P<mouse>[A-Za-z]+_\d+)'
    r'_(?P<date>\d{4}-\d{2}-\d{2})'
    r'_(?P<hemisphere>\d+)'
    r'_p_summary\.npy$')

# `variable`, `epoch` and `x_type` were written as the placeholder strings
# 'x', 'x' and 'raw' into every row of every CSV. Nothing reads them, and the
# spec now knows the real values, so they are filled in. This is the one way
# the regenerated CSVs differ from the shipped ones, and the check below
# treats it as expected.
PLACEHOLDER_COLUMNS = ['variable', 'epoch', 'x_type']


def describe(spec_name):
    """
    The three annotation columns for one spec.

    Args:
        spec_name: Key into SPECS.

    Returns:
        (variable, epoch, x_type).
    """
    spec = SPECS[spec_name]
    return (spec_name,
            spec['alignment'].replace('_time', ''),
            'residual' if spec['filetype'].startswith('residual') else 'raw')


def load_decoders(folder, n_neuron_combos, null, variable, epoch, x_type):
    """
    Every `.npy` in one decoder's folder, as one long table.

    Each file holds a (time bin x neuron count) grid of scores for one
    recording; this unstacks it to one row per (time bin, neuron count) and
    labels it with the recording it came from.

    Args:
        folder: The spec's output folder.
        n_neuron_combos: Neuron counts the file's columns correspond to, in
            order.
        null: Read the null files rather than the real ones.
        variable, epoch, x_type: Annotation columns.

    Returns:
        (summary, errors): the table, and any files that could not be read.
    """
    pattern = '/*null*p_summary.npy' if null else '/*real*p_summary.npy'
    # Sorted, where the original took glob's filesystem order: regeneration
    # is then reproducible. It changes the ROW ORDER of the CSV relative to
    # the shipped one and nothing else - the figure scripts group and merge,
    # they never index by position - and `compare_to_shipped` sorts before
    # comparing so the check is unaffected.
    paths = sorted(glob.glob(str(folder) + pattern))
    if not paths:
        raise FileNotFoundError(
            f"no {'null' if null else 'real'} .npy files in {folder}. "
            "Compiling would write an empty CSV, which the figure scripts "
            "would read as a panel with no data.")

    errors, blocks = [], []
    for path in paths:
        match = SUMMARY_NAME.match(Path(path).name)
        if match is None:
            errors.append((path, 'filename not recognised'))
            continue
        try:
            r_summary = np.load(path)
            mse_summary = np.load(path[:-13] + 'mse_summary.npy')
        except (OSError, ValueError) as problem:
            errors.append((path, str(problem)))
            continue
        if r_summary.shape != mse_summary.shape:
            errors.append((path, f"r {r_summary.shape} vs mse "
                                 f"{mse_summary.shape}"))
            continue
        # A file can hold FEWER neuron counts than the sweep asked for, and
        # usually does: `run_decoder` keeps only the counts that fit the
        # neurons available (`neuron_samples_n <= n_neurons_available`), and
        # since the sweep ascends that leaves a prefix. A region with 15
        # neurons yields one column, meaning 10; with 25, two columns,
        # meaning 10 and 20. So column `c` is `n_neuron_combos[c]`, and the
        # count of columns is not something to check against - only an
        # excess would be wrong.
        if r_summary.shape[1] > len(n_neuron_combos):
            errors.append((path, f"{r_summary.shape[1]} neuron counts in the "
                                 f"file, at most {len(n_neuron_combos)} "
                                 "expected"))
            continue

        n_bins, n_combos = r_summary.shape
        combos_here = list(n_neuron_combos)[:n_combos]
        fields = match.groupdict()
        block = pd.DataFrame({
            'r': r_summary.reshape(-1, order='F'),
            'mse': mse_summary.reshape(-1, order='F'),
            'time_bin': np.tile(np.arange(n_bins), n_combos),
            'n_neurons': np.repeat(combos_here, n_bins),
            'region': fields['region'],
            'mouse': fields['mouse'],
            'hemisphere': fields['hemisphere'],
            'date': fields['date'],
        })
        # The original built this file's rows as one small frame per neuron
        # count, each with its own 0..n_bins-1 index, then concatenated. The
        # `index` column that `reset_index` produces therefore repeats per
        # block - which makes it equal to `time_bin`. Reproduced so the
        # regenerated CSV carries the same column.
        block.index = np.tile(np.arange(n_bins), n_combos)
        # `type` appears twice in the id. Inherited, and load-bearing: the
        # figure scripts match on this string.
        if null:
            block['iter'] = fields['iteration']
            block['type'] = 'null'
            block['id'] = (block['iter'] + block['region'] + block['type']
                           + block['date'] + block['mouse']
                           + block['hemisphere'] + block['type'])
        else:
            block['type'] = 'real'
            block['id'] = (block['region'] + block['type'] + block['date']
                           + block['mouse'] + block['hemisphere']
                           + block['type'])
        blocks.append(block)

    summary = pd.concat(blocks).reset_index()
    summary['variable'] = variable
    summary['epoch'] = epoch
    summary['x_type'] = x_type
    return summary, errors


def report_null_completeness(nulls):
    """
    Whether every recording got all 200 null draws.

    A leftover `missing_nulls.pkl` on the share is evidence that some null
    jobs once failed. If any never completed, the affected cells have a null
    distribution built from fewer draws and nothing downstream notices -
    `null_correct` records `n_draws` but never checks it.

    Args:
        nulls: The null table from `load_decoders`.
    """
    per_cell = (nulls.groupby(['region', 'mouse', 'date', 'hemisphere'])['iter']
                .nunique())
    short = per_cell[per_cell != N_NULLS_EXPECTED]
    if len(short) == 0:
        print(f"        all {len(per_cell)} recordings have "
              f"{N_NULLS_EXPECTED} nulls")
        return
    print(f"        {len(short)} of {len(per_cell)} recordings have fewer "
          f"than {N_NULLS_EXPECTED} nulls:")
    for key, count in short.items():
        print(f"          {'/'.join(map(str, key))}: {count}")


# Below this, a difference is the CSV round-trip rather than different data.
# Either measure passing is enough: a value near zero has a large relative
# error for an absolutely negligible gap.
FLOAT_NOISE = 1e-12


def column_difference(old_column, new_column):
    """
    How two versions of one column differ.

    Args:
        old_column, new_column: Aligned Series.

    Returns:
        None if identical, else `(description, max_absolute, max_relative)`.
        The two magnitudes are None for a non-numeric column, which is what
        stops a text mismatch ever being written off as float noise.
    """
    numeric = old_column.dtype.kind in 'fiu' and new_column.dtype.kind in 'fiu'
    if numeric:
        a = old_column.astype(float).to_numpy()
        b = new_column.astype(float).to_numpy()
        differing = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        if not differing.any():
            return None
        gap = np.abs(a[differing] - b[differing])
        scale = np.maximum(np.abs(a[differing]), np.abs(b[differing]))
        relative = gap / np.where(scale > 0, scale, 1)
        # `to_csv` does not round-trip a float64 exactly, and perturbs most
        # values in their last bits. So the count of values that differ AT
        # ALL is dominated by the round trip and says nothing - quoting it
        # next to a max difference of 1e-2 implies a hundred thousand rows
        # are wrong when a couple of thousand are. Both counts are reported,
        # and the magnitudes describe the values above the floor alone.
        real = (gap > FLOAT_NOISE) & (relative > FLOAT_NOISE)
        if not real.any():
            return (f"{differing.sum()}/{len(a)} values differ in the last "
                    f"bits only (max |diff| {gap.max():.3e})",
                    gap.max(), relative.max())
        return (f"{real.sum()}/{len(a)} values differ beyond the CSV round "
                f"trip, max |diff| {gap[real].max():.3e}, max relative "
                f"{relative[real].max():.3e}; "
                f"{differing.sum() - real.sum()} more in the last bits only",
                gap[real].max(), relative[real].max())
    a, b = old_column.astype(str), new_column.astype(str)
    differing = (a != b).to_numpy()
    if not differing.any():
        return None
    examples = [f"{x!r} vs {y!r}" for x, y in
                zip(a[differing].head(2), b[differing].head(2))]
    return (f"{differing.sum()}/{len(a)} values, e.g. {'; '.join(examples)}",
            None, None)


def read_shipped(path):
    """
    Read a summary CSV without pandas turning the word 'null' into NaN.

    `type` holds the literal string 'null' on every row of an nsummary file,
    and 'null' is one of pandas' default NA sentinels - so the column comes
    back as all-NaN float64 unless the defaults are switched off. `to_csv`
    writes a real NaN as an empty field, so that is the only sentinel needed.

    Args:
        path: The CSV.

    Returns:
        The table.
    """
    return pd.read_csv(path, keep_default_na=False, na_values=[''])


def compare_to_shipped(new, path):
    """
    Whether a regenerated CSV matches the one already there.

    The three annotation columns are excluded: they were placeholders and are
    now filled in deliberately.

    Args:
        new: Regenerated table.
        path: The CSV on disk.

    Returns:
        (verdict, message).
    """
    if not Path(path).exists():
        return 'absent', f"nothing on disk yet ({len(new)} rows)"
    old = read_shipped(path)
    shared = [c for c in new.columns
              if c in old.columns and c not in PLACEHOLDER_COLUMNS]
    if len(old) != len(new):
        return 'differs', f"{len(old)} rows on disk, {len(new)} regenerated"

    key = ['id', 'time_bin', 'n_neurons']
    duplicated = old.duplicated(subset=key).sum()
    if duplicated:
        return 'differs', (f"{duplicated} rows share an (id, time_bin, "
                           "n_neurons) key, so the two cannot be aligned")
    old_sorted = old.sort_values(key).reset_index(drop=True)
    new_sorted = new.sort_values(key).reset_index(drop=True)

    differences, magnitudes = {}, []
    for column in shared:
        report = column_difference(old_sorted[column], new_sorted[column])
        if report is not None:
            differences[column] = report[0]
            magnitudes.append(report[1:])
    if not differences:
        return 'same', f"{len(new)} rows, {len(shared)} columns identical"

    # Writing and re-reading a float is not a real difference. A column that
    # is not numeric at all has no magnitude, and never counts as noise.
    noise_only = all(
        absolute is not None
        and (absolute < FLOAT_NOISE or relative < FLOAT_NOISE)
        for absolute, relative in magnitudes)
    verdict = 'noise' if noise_only else 'differs'
    lines = '; '.join(f"{c}: {r}" for c, r in differences.items())
    return verdict, lines


# ============================================================================
# Run
# ============================================================================
WRITE = '--write' in sys.argv

print("REGENERATING the summary CSVs" if WRITE else
      "VERIFYING against the CSVs on disk - nothing will be written")
print(f"{len(RUN)} spec(s)\n")

problems = []
for name in RUN:
    spec = SPECS[name]
    folder = Path(spec['folder'])
    variable, epoch, x_type = describe(name)
    print(f"      {name:17s} {folder}")

    if not folder.exists():
        print("        folder does not exist - skipped\n")
        problems.append(name)
        continue

    for null, combos, filename in [
            (False, REAL_NEURON_COMBOS, 'summary_102030405060.csv'),
            (True, [spec['null_n']], f"nsummary_{spec['null_n']}.csv")]:
        try:
            table, errors = load_decoders(folder, combos, null, variable,
                                          epoch, x_type)
        except FileNotFoundError as empty:
            print(f"        {filename}: {empty}")
            problems.append(name)
            continue

        verdict, message = compare_to_shipped(table, folder / filename)
        flag = {'same': 'OK  ', 'absent': 'NEW ', 'noise': 'OK~ ',
                'differs': 'DIFF'}[verdict]
        print(f"        {flag} {filename:28s} {message}")
        if errors:
            print(f"             {len(errors)} unreadable file(s), first: "
                  f"{errors[0][0]} - {errors[0][1]}")
        if null:
            report_null_completeness(table)
        if verdict == 'differs':
            problems.append(name)

        if WRITE:
            table.to_csv(folder / filename)
            print(f"             wrote {filename}")
    print()

if problems:
    print(f"{len(sorted(set(problems)))} spec(s) need attention: "
          f"{', '.join(sorted(set(problems)))}")
elif not WRITE:
    print("Every regenerated CSV matches what is on disk. "
          "Nothing was written.")
