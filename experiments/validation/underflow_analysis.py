"""Exact characterization of the archived temperature-underflow failure.

The manuscript previously stated that the archived schedule "reaches zero at
epoch 14,527". That is true of the closed form tau_0 * 0.95^t, but the
archived code implements the *iterative* recurrence

    self.tau *= self.hp.tau_decay

which behaves differently in the subnormal range: repeated rounding to nearest
makes the iteration stall at a fixed subnormal value and it never reaches
exactly zero. The operative failure is therefore not a division by zero but an
overflow of the tuple log-weight (1/tau) * sum_i log p_i(t_i) to +/-inf, after
which the max-subtraction step computes inf - inf = nan and every tuple weight
becomes non-finite.

This script determines, from the arithmetic alone:

  * the epoch at which the closed form underflows to exactly zero;
  * the value and epoch at which the iterative recurrence stalls;
  * the first epoch at which a tuple log-weight of unit magnitude overflows;
  * the first epoch at which the projector returns a non-finite weight vector;

and then counts, from the raw sweep logs, how many archived runs reached each
of those epochs.

Usage:
    PYTHONPATH=. python -m experiments.validation.underflow_analysis \
        --raw results/raw --tables paper/ijoc/tables
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

TAU0 = 2.0
DECAY = 0.95
DBL_MAX = sys.float_info.max


def closed_form_zero_epoch() -> int:
    t = 1
    while TAU0 * DECAY ** t != 0.0:
        t += 1
        if t > 10 ** 6:
            raise RuntimeError("closed form never reached zero")
    return t


def iterative_profile(max_epochs: int = 200000):
    """Trace the iterative recurrence; return the stall epoch and value."""
    tau = TAU0
    prev = None
    stall_epoch, stall_value = None, None
    first_overflow_unit = None      # |log w| = 1 overflows
    first_nonfinite_inv = None      # 1/tau itself becomes inf
    for t in range(1, max_epochs + 1):
        tau = tau * DECAY
        if first_overflow_unit is None and tau > 0.0 and 1.0 / tau > DBL_MAX:
            first_overflow_unit = t
        if first_overflow_unit is None and tau > 0.0:
            # explicit check on the quantity the projector actually forms
            if not np.isfinite(np.float64(-1.0) / np.float64(tau)):
                first_overflow_unit = t
        if first_nonfinite_inv is None and tau == 0.0:
            first_nonfinite_inv = t
        if prev is not None and tau == prev and tau > 0.0:
            stall_epoch, stall_value = t, tau
            break
        prev = tau
    return stall_epoch, stall_value, first_overflow_unit, first_nonfinite_inv


def first_nonfinite_projection_epoch(logw_magnitude: float = 1.0) -> int:
    """First epoch at which the tuple log-weight overflows for a given |log w|."""
    tau = TAU0
    for t in range(1, 200000):
        tau = tau * DECAY
        if tau <= 0.0:
            return t
        val = np.float64(-logw_magnitude) / np.float64(tau)
        if not np.isfinite(val):
            return t
    raise RuntimeError("no overflow found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw")
    ap.add_argument("--tables", default="paper/ijoc/tables")
    args = ap.parse_args()

    print("== arithmetic ==")
    zero_epoch = closed_form_zero_epoch()
    stall_e, stall_v, ovf_unit, zero_it = iterative_profile()
    print(f"  closed form tau_0*0.95^t first equals 0.0 at epoch {zero_epoch}")
    print(f"  iterative recurrence stalls at epoch {stall_e} "
          f"with tau = {stall_v!r}")
    print(f"  iterative recurrence reaches exactly 0.0: "
          f"{zero_it if zero_it else 'never'}")

    thresholds = {}
    for mag, label in [(1.0, "unit"), (10.0, "ten"), (100.0, "hundred")]:
        thresholds[label] = first_nonfinite_projection_epoch(mag)
        print(f"  first epoch where (1/tau)*log w overflows for |log w|={mag:g}:"
              f" {thresholds[label]}")

    first_fail = min(thresholds.values())

    # ------------------------------------------------------------------
    print("== affected archived runs ==")
    def canon(n): return n[:-5] if n.endswith(".json") else n
    rows = []
    for f in sorted(glob.glob(f"{args.raw}/revision_sweep/*.jsonl")):
        for line in open(f):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    mir = [r for r in rows if r["solver"] in ("mirage", "mirage_regions")]
    have_epochs = [r for r in mir if isinstance(r.get("epochs"), int)
                   and r["epochs"] >= 0]
    print(f"  MIRAGE-family runs examined: {len(have_epochs)}")

    def count_beyond(e):
        aff = [r for r in have_epochs if r["epochs"] >= e]
        sat = [r for r in aff if r["status"] == "SAT_VERIFIED"]
        return len(aff), len(sat)

    n_zero, sat_zero = count_beyond(zero_epoch)
    n_fail, sat_fail = count_beyond(first_fail)
    print(f"  runs reaching the closed-form zero epoch ({zero_epoch}): "
          f"{n_zero} (verified witnesses among them: {sat_zero})")
    print(f"  runs reaching the true first-failure epoch ({first_fail}): "
          f"{n_fail} (verified witnesses among them: {sat_fail})")

    maxep = max(r["epochs"] for r in have_epochs)
    print(f"  maximum epoch count in the archive: {maxep}")

    # instances involved
    aff_inst = sorted({canon(r["instance"]) for r in have_epochs
                       if r["epochs"] >= first_fail})
    fam = collections.Counter(i.split("-")[0] for i in aff_inst)
    print(f"  distinct entries affected: {len(aff_inst)} {dict(fam)}")

    # ------------------------------------------------------------------
    macros = {
        "RevTauZeroEpoch": f"{zero_epoch:,}".replace(",", "{,}"),
        "RevTauStallEpoch": f"{stall_e:,}".replace(",", "{,}"),
        # Math content without delimiters; used inside $\\tau\\approx\\ldots$.
        "RevTauStallValue": (lambda v: "%s\\times10^{%d}" % (
            f"{v:.2e}".split("e")[0], int(f"{v:.2e}".split("e")[1])))(stall_v),
        "RevTauFailEpoch": f"{first_fail:,}".replace(",", "{,}"),
        "RevUnderflowTotal": f"{len(have_epochs):,}".replace(",", "{,}"),
        "RevUnderflowRuns": n_zero,
        "RevUnderflowRunsTrue": n_fail,
        "RevUnderflowSat": sat_fail,
        "RevUnderflowEntries": len(aff_inst),
        "RevUnderflowMaxEpoch": f"{maxep:,}".replace(",", "{,}"),
    }
    out = os.path.join(args.tables, "underflow_numbers.tex")
    with open(out, "w") as fh:
        fh.write("% AUTO-GENERATED by "
                 "experiments/validation/underflow_analysis.py -- do not edit\n")
        for k, v in sorted(macros.items()):
            fh.write(f"\\providecommand{{\\{k}}}{{}}\n")
            fh.write(f"\\renewcommand{{\\{k}}}{{{v}}}\n")
    print(f"  wrote {out}")

    with open(os.path.join(args.tables, "underflow_entries.txt"), "w") as fh:
        fh.write("Benchmark entries with at least one archived MIRAGE-family run\n")
        fh.write(f"reaching epoch {first_fail} (first tuple-weight overflow)\n\n")
        for i in aff_inst:
            fh.write(i + "\n")

    assert sat_fail == 0, (
        "a run past the first-failure epoch returned a verified witness; "
        "the claim that no witness depends on post-underflow behavior fails")
    print("\nASSERTION PASSED: no verified witness comes from a run past the "
          "first-failure epoch")


if __name__ == "__main__":
    main()
