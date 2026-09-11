"""Recompute, from the retained raw logs, every number the paper states.

Until now a block of `paper/ijoc/tables/numbers.tex` was labelled "manually
curated, verified against results/raw/". The manuscript, whose subject is data
integrity, should not contain hand-transcribed numbers, so this script derives
them instead and asserts each one. It writes `curated_numbers.tex`, which
`numbers.tex` no longer needs to carry.

Two conventions matter and are the ones the archive audit already uses:

  * a *verified witness* is a run whose terminal status is `SAT_VERIFIED`.
    Only the Choco adapter also populates a separate `witness_verified` field;
    the MIRAGE-family adapters leave it null and encode the verifier's verdict
    in the status itself, so keying on the field alone silently reports zero
    witnesses for MIRAGE-R;
  * instance keys must be canonicalized, because some adapters logged the
    `.json` suffix and others did not. Without this a solver appears to have
    737 instances rather than 736.

Every emitted macro is compared against the value the manuscript was written
with, which is recorded in EXPECTED below. A mismatch is an error, not a
silent update: it means either the logs or the paper moved.

Usage:
    PYTHONPATH=. python -m experiments.validation.paper_numbers \
        --raw results/raw --tables paper/ijoc/tables
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import statistics
import sys

N_INSTANCES = 736

# The values the manuscript was written with. Asserted, not trusted.
EXPECTED = {
    "RevWitChoco": "406", "RevWitRateChoco": "55.2",
    "RevWitCpSat": "318", "RevWitRateCpSat": "43.2",
    "RevWitHybrid": "303", "RevWitRateHybrid": "41.2",
    "RevWitRateRunCsp": "1.9",
    "RevSolvedMirage": "62", "RevWitRateMirage": "8.4",
    "RevSolvedMirageRegions": "64", "RevWitRateMirageRegions": "8.7",
    "RevAnySeedMirage": "65", "RevAnySeedRateMirage": "8.8",
    "RevUnkMirage": "665", "RevUnkWithSatMirage": "3",
    "RevHybSubsetN": "160", "RevHybSubsetSeeds": "3",
    "RevAblExecInstances": "161", "RevAblSkipped": "575", "RevAblSeeds": "3",
    "RevTrajN": "40", "RevTrajSat": "13", "RevTrajTimeout": "27",
    "RevTrajSlopeN": "27",
}

SOLVER_ORDER = ["choco", "ortools", "hybrid", "mirage_regions", "mirage",
                "runcsp"]
SOLVER_LABEL = {
    "choco": "Choco",
    "ortools": "OR-Tools CP-SAT",
    "hybrid": r"MIRAGE-R$\to$CP-SAT",
    "mirage_regions": "MIRAGE-R+regions",
    "mirage": "MIRAGE-R",
    "runcsp": "RUN-CSP",
}

# The main comparison, cell by cell, as printed in the manuscript. Asserted for
# the same reason as EXPECTED: the table used to be typeset by hand.
EXPECTED_MAIN = {
    "choco":          (406, 253,  67,   0, 10, 0.55),
    "ortools":        (318, 179, 230,   0,  9, 16.27),
    "hybrid":         (303, 173, 251,   0,  9, 65.09),
    "mirage_regions": ( 64,   0, 663,   0,  9, 4.31),
    "mirage":         ( 62,   0, 665,   0,  9, 3.91),
    "runcsp":         ( 14,   0,  34, 675, 13, 32.84),
}
TIME_LIMIT = 300.0

MAIN_CAPTION = (r"\TABLE{Main comparison, majority-of-five aggregation."
                r"\label{tab:main}}")

MAIN_NOTE = (r"{\emph{wit.\%} is $\text{SAT}/736$ and \emph{dec.\%} is"
             r" $(\text{SAT}+\text{UNSAT})/736$; they coincide for the"
             r" incomplete methods and must not be mixed. \emph{unsup.} and"
             r" \emph{error} count entries on which every seed returned that"
             r" status, and \emph{unk.} is the residual. Median time is over"
             r" solved entries at unequal thread allocations (CP-SAT four,"
             r" others one) under concurrent shards, so it is descriptive"
             r" only.}")

SOLVER_MACRO = {
    "choco": ("RevWitChoco", "RevWitRateChoco"),
    "ortools": ("RevWitCpSat", "RevWitRateCpSat"),
    "hybrid": ("RevWitHybrid", "RevWitRateHybrid"),
    "mirage": ("RevSolvedMirage", "RevWitRateMirage"),
    "mirage_regions": ("RevSolvedMirageRegions", "RevWitRateMirageRegions"),
    "runcsp": (None, "RevWitRateRunCsp"),
}


def canon(name: str) -> str:
    name = name.strip()
    return name[:-5] if name.endswith(".json") else name


def load(pattern: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def group(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["solver"], canon(r["instance"]))].append(r)
    return by


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw")
    ap.add_argument("--tables", default="paper/ijoc/tables")
    args = ap.parse_args()
    os.makedirs(args.tables, exist_ok=True)

    out: dict[str, str] = {}

    # ---- main sweep: verified-witness coverage ---------------------------
    sweep = load(os.path.join(args.raw, "revision_sweep", "*.jsonl"))
    by = group(sweep)
    solvers = sorted({s for s, _ in by})
    for s in solvers:
        insts = sorted({i for (sv, i) in by if sv == s})
        assert len(insts) == N_INSTANCES, \
            f"{s}: {len(insts)} instances, expected {N_INSTANCES}"
        maj = sum(1 for i in insts
                  if sum(1 for r in by[(s, i)] if r["status"] == "SAT_VERIFIED")
                  > len(by[(s, i)]) / 2)
        cnt_macro, rate_macro = SOLVER_MACRO[s]
        if cnt_macro:
            out[cnt_macro] = str(maj)
        out[rate_macro] = f"{100 * maj / N_INSTANCES:.1f}"

    m_insts = sorted({i for (sv, i) in by if sv == "mirage"})

    def sat_seeds(inst):
        return sum(1 for r in by[("mirage", inst)] if r["status"] == "SAT_VERIFIED")

    any_seed = sum(1 for i in m_insts if sat_seeds(i) > 0)
    out["RevAnySeedMirage"] = str(any_seed)
    out["RevAnySeedRateMirage"] = f"{100 * any_seed / N_INSTANCES:.1f}"
    out["RevUnkWithSatMirage"] = str(sum(
        1 for i in m_insts
        if sat_seeds(i) > 0 and sat_seeds(i) <= len(by[("mirage", i)]) / 2))
    # the residual "unk." column excludes entries whose every seed errored
    all_error = sum(1 for i in m_insts
                    if all(r["status"] == "ERROR" for r in by[("mirage", i)]))
    out["RevUnkMirage"] = str(
        sum(1 for i in m_insts if sat_seeds(i) <= len(by[("mirage", i)]) / 2)
        - all_error)

    # ---- the main comparison, generated rather than typeset --------------
    main_rows, main_bad = [], []
    for s_ in SOLVER_ORDER:
        insts = sorted({i for (sv, i) in by if sv == s_})

        def majority(inst, status, s_=s_):
            rs = by[(s_, inst)]
            return sum(1 for r in rs if r["status"] == status) > len(rs) / 2

        def unanimous(inst, status, s_=s_):
            return all(r["status"] == status for r in by[(s_, inst)])

        sat = [i for i in insts if majority(i, "SAT_VERIFIED")]
        uns = [i for i in insts if majority(i, "UNSAT")]
        unsup = [i for i in insts if unanimous(i, "UNSUPPORTED_FRAGMENT")]
        err = [i for i in insts if unanimous(i, "ERROR")]
        unk = len(insts) - len(sat) - len(uns) - len(unsup) - len(err)
        times = []
        for i in sat + uns:
            ts = [float(r["time"]) for r in by[(s_, i)]
                  if r.get("time") not in (None, "")]
            if ts:
                times.append(min(statistics.median(ts), TIME_LIMIT))
        med = statistics.median(times) if times else float("nan")
        cells = (len(sat), len(uns), unk, len(unsup), len(err), round(med, 2))
        if cells != EXPECTED_MAIN[s_]:
            main_bad.append((s_, cells, EXPECTED_MAIN[s_]))
        wit = 100 * len(sat) / N_INSTANCES
        dec = 100 * (len(sat) + len(uns)) / N_INSTANCES
        main_rows.append(
            f"{SOLVER_LABEL[s_]} & {N_INSTANCES} & {len(sat)} & {len(uns)} & "
            f"{unk} & {len(unsup)} & {len(err)} & {wit:.2f} & {dec:.2f} & "
            f"{med:.2f} " + r"\\")
    for s_, got, want in main_bad:
        print(f"  MISMATCH main comparison {s_}: computed {got}, manuscript {want}")

    # ---- warm-start grid and schedule ablation ---------------------------
    grid = load(os.path.join(args.raw, "revision_hybrid_grid", "*.jsonl"))
    out["RevHybSubsetN"] = str(len({canon(r["instance"]) for r in grid}))
    out["RevHybSubsetSeeds"] = str(len({r["seed"] for r in grid}))

    # The grid was partly re-executed on resumption, so it must be keyed and
    # deduplicated before any rate is taken from it; the archived aggregation
    # did not do this, and its cell rates are inflated by the duplicates.
    gkeys: dict = {}
    for r in grid:
        gkeys.setdefault((r.get("warmup_epochs"), r.get("hint_mode"),
                          canon(r["instance"]), r["seed"]), r)
    gcells: dict = {}
    for (w, h, _i, _s), r in gkeys.items():
        gcells.setdefault((w, h), []).append(r)

    def _dec(r):
        return r["status"] in ("SAT_VERIFIED", "UNSAT", "UNSAT_PROVED")

    stats = {}
    for c, v in gcells.items():
        d = [r for r in v if _dec(r)]
        stats[c] = (len(d), sum(1 for r in v if r["status"] == "SAT_VERIFIED"),
                    len(v), statistics.median(r["time"] for r in d))
    ctrl = (0, "none")
    hinted = {c: t for c, t in stats.items() if c[1] != "none"}
    best = max(hinted, key=lambda c: (hinted[c][0], -hinted[c][3]))
    out["RevHybGridRuns"] = str(stats[ctrl][2])
    out["RevHybCtrlDec"] = f"{100 * stats[ctrl][0] / stats[ctrl][2]:.1f}"
    out["RevHybCtrlWit"] = f"{100 * stats[ctrl][1] / stats[ctrl][2]:.1f}"
    out["RevHybCtrlMed"] = f"{stats[ctrl][3]:.1f}"
    out["RevHybBestDec"] = f"{100 * stats[best][0] / stats[best][2]:.1f}"
    out["RevHybBestMed"] = f"{stats[best][3]:.1f}"
    out["RevHybBestCellDedup"] = f"warm-up {best[0]}, {best[1]}"
    out["RevHybNoCellBeatsCtrl"] = (
        "yes" if max(t[0] for t in hinted.values()) <= stats[ctrl][0] else "no")

    # Budget-matched controls. The grid ran hint_mode "none" at every positive
    # warm-up, and the solver computes the marginals there and then discards
    # them, so those cells pay the warm-up cost without receiving the hints.
    # Comparing a hinted cell with the "none" cell at the same warm-up
    # separates the cost of the warm-up from the effect of the hints.
    warmups = sorted({w for w, _h in stats if w > 0})
    base = {w: stats[(w, "none")][0] for w in warmups}
    deltas = {(w, m): stats[(w, m)][0] - base[w]
              for w in warmups for m in ("value", "order", "both")}
    out["RevHybMatchedBase"] = str(sorted(set(base.values()))[0])
    out["RevHybMatchedSame"] = "yes" if len(set(base.values())) == 1 else "no"
    out["RevHybMatchedBest"] = f"{max(deltas.values()):+d}"
    out["RevHybMatchedWorst"] = f"{min(deltas.values()):+d}"
    out["RevHybMatchedWorstCell"] = "warm-up {}, {}".format(
        *min(deltas, key=deltas.get))
    out["RevHybCtrlSlowMed"] = f"{stats[(max(warmups), 'none')][3]:.1f}"

    abl = load(os.path.join(args.raw, "revision_ablation", "*.jsonl"))
    executed = {canon(r["instance"]) for r in abl if r["status"] != "SKIPPED_SIZE"}
    out["RevAblExecInstances"] = str(len(executed))
    out["RevAblSkipped"] = str(N_INSTANCES - len(executed))
    out["RevAblSeeds"] = str(len({r["seed"] for r in abl}))

    # ---- trajectory sub-experiment ---------------------------------------
    traj = load(os.path.join(args.raw, "revision_theory", "trajectories.jsonl"))
    out["RevTrajN"] = str(len(traj))
    out["RevTrajSat"] = str(sum(1 for r in traj if r["status"] == "SAT_VERIFIED"))
    out["RevTrajTimeout"] = str(sum(1 for r in traj
                                    if r["status"] == "UNKNOWN_TIMEOUT"))
    # slope eligibility: more than five checkpoints survive the analysis filter
    eligible = 0
    for r in traj:
        y = r.get("running_min_trajectory") or []
        if len(y) < 3:
            continue
        y0 = y[0] if y[0] > 0 else 1.0
        n = sum(1 for j, v in enumerate(y)
                if v / y0 > 1e-3 and (j + 1) * r["decode_freq"] > 1)
        eligible += n > 5
    out["RevTrajSlopeN"] = str(eligible)

    # ---- assert against what the manuscript states -----------------------
    bad = [(k, v, EXPECTED[k]) for k, v in sorted(out.items())
           if k in EXPECTED and v != EXPECTED[k]]
    for k, got, want in bad:
        print(f"  MISMATCH {k}: computed {got}, manuscript {want}")
    missing = sorted(set(EXPECTED) - set(out))
    for k in missing:
        print(f"  NOT COMPUTED {k}")
    if bad or missing or main_bad:
        sys.exit("paper_numbers: the logs and the manuscript disagree")

    main_path = os.path.join(args.tables, "main_comparison.tex")
    with open(main_path, "w") as fh:
        fh.write("\n".join([
            r"\begin{table}[htbp]", MAIN_CAPTION,
            r"{\small\begin{tabular}{lrrrrrrrrr}", r"\toprule",
            r"solver & \#inst & SAT & UNSAT & unk. & unsup. & error & wit.\% "
            r"& dec.\% & med.\ solved-case time (s) \\",
            r"\midrule", *main_rows, r"\bottomrule",
            r"\end{tabular}}", MAIN_NOTE, r"\end{table}", ""]))
    print(f"[paper_numbers] wrote {main_path}")

    path = os.path.join(args.tables, "curated_numbers.tex")
    with open(path, "w") as fh:
        fh.write("% AUTO-GENERATED by experiments/validation/paper_numbers.py"
                 " -- do not edit\n")
        for k in sorted(out):
            fh.write(f"\\providecommand{{\\{k}}}{{}}\n"
                     f"\\renewcommand{{\\{k}}}{{{out[k]}}}\n")
    print(f"[paper_numbers] {len(out)} macros recomputed from the raw logs and "
          f"matched against the manuscript")
    print(f"[paper_numbers] wrote {path}")


if __name__ == "__main__":
    main()
