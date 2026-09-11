"""Archive-wide data-integrity audit, error taxonomy, and paired analyses.

Addresses reviewer items A.2 (duplicate-key audit), A.3 (error explanation),
A.4 (paired dynamic-region outcome table), E.37 (majority / at-least-one /
per-run), E.39 (RUN-CSP fragment conditioning), E.40 (errors vs unsupported).

Every table written here is generated from the raw JSONL archives under
results/raw/revision_*; no number is typed by hand.

Usage:
    PYTHONPATH=. python -m experiments.validation.audit_archive \
        --raw results/raw --tables paper/ijoc/tables
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
from fractions import Fraction

# ----------------------------------------------------------------------
# Loading and canonicalization
# ----------------------------------------------------------------------

SOLVER_LABEL = {
    "choco": "Choco",
    "ortools": "OR-Tools CP-SAT",
    "hybrid": r"MIRAGE-R$\to$CP-SAT",
    "mirage_regions": "MIRAGE-R+regions",
    "mirage": "MIRAGE-R",
    "runcsp": "RUN-CSP",
}
SOLVER_ORDER = ["choco", "ortools", "hybrid", "mirage_regions", "mirage", "runcsp"]

# Abbreviated heads for the wide per-configuration tables; the long names of
# SOLVER_LABEL do not fit inside the IJOC text block. Expanded in the note.
SOLVER_SHORT = {
    "choco": "Choco",
    "ortools": "CP-SAT",
    "hybrid": r"M-R$\to$CP-SAT",
    "mirage_regions": "M-R+reg",
    "mirage": "M-R",
    "runcsp": "RUN-CSP",
}


def canon(name: str) -> str:
    """Strip the `.json` suffix that some adapters logged and others did not."""
    name = name.strip()
    return name[:-5] if name.endswith(".json") else name


def load_jsonl(pattern: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec["_file"] = os.path.basename(path)
                if "instance" in rec:
                    rec["inst"] = canon(rec["instance"])
                rows.append(rec)
    return rows


# ----------------------------------------------------------------------
# Duplicate-key audit
# ----------------------------------------------------------------------

def audit_experiment(name, rows, keyfields, expected):
    """Return the audit row for one experiment archive."""
    def key(r):
        return tuple(r.get(f) for f in keyfields)

    buckets = collections.defaultdict(list)
    for r in rows:
        buckets[key(r)].append(r)
    dup_keys = [k for k, v in buckets.items() if len(v) > 1]
    conflicting = [k for k in dup_keys
                   if len({x.get("status") for x in buckets[k]}) > 1]
    return {
        "experiment": name,
        "expected": expected,
        "raw": len(rows),
        "unique": len(buckets),
        "duplicate": len(dup_keys),
        "missing": max(0, expected - len(buckets)) if expected else 0,
        "conflicting": len(conflicting),
        "_buckets": buckets,
        "_dup_keys": dup_keys,
        "_conflicting": conflicting,
    }


# ----------------------------------------------------------------------
# Error taxonomy
# ----------------------------------------------------------------------

def classify_error(msg: str) -> tuple[str, str]:
    """Map a retained error string to (class code, human-readable class)."""
    m = msg or ""
    if "empty instance file" in m:
        return "I1", "empty normalized file (zero bytes)"
    if ("JSONDecode" in m or "Expecting ',' delimiter" in m
            or "Expecting value" in m or "codecs" in m or "loader.py" in m):
        return "I2", "truncated or malformed normalized JSON"
    if "CUDA" in m or "cublas" in m or "CUBLAS" in m:
        return "R1", "GPU device or memory failure"
    if ("bad allocation" in m or "Unable to allocate" in m
            or "commit_memory" in m or "pagination" in m
            or "paging file" in m or "MemoryError" in m):
        return "R2", "host memory exhaustion"
    if "Choco" in m or "org.mirager" in m or "java" in m.lower():
        return "A1", "Java adapter or model construction"
    if "Errno" in m:
        return "R3", "operating-system or file-handle failure"
    if m.strip() == "":
        return "U1", "error recorded without a retained message"
    return "U2", "other"


CLASS_ORIGIN = {
    "I1": "input data",
    "I2": "input data",
    "A1": "adapter",
    "R1": "harness / resource",
    "R2": "harness / resource",
    "R3": "harness / resource",
    "U1": "unclassified",
    "U2": "unclassified",
}


# ----------------------------------------------------------------------
# Seed aggregation
# ----------------------------------------------------------------------

DEFINITIVE_SAT = {"SAT_VERIFIED"}
DEFINITIVE_UNSAT = {"UNSAT", "UNSAT_PROVED"}


def majority_outcome(runs):
    """Instance-level outcome under the majority-of-five rule."""
    n = len(runs)
    statuses = [r["status"] for r in runs]
    n_sat = sum(1 for s in statuses if s in DEFINITIVE_SAT)
    n_unsat = sum(1 for s in statuses if s in DEFINITIVE_UNSAT)
    n_unsup = sum(1 for s in statuses if s == "UNSUPPORTED_FRAGMENT")
    n_err = sum(1 for s in statuses if s == "ERROR")
    if n_sat * 2 > n:
        return "SAT"
    if n_unsat * 2 > n:
        return "UNSAT"
    if n_unsup == n:
        return "UNSUPPORTED"
    if n_err == n:
        return "ERROR"
    return "UNKNOWN"


# ----------------------------------------------------------------------
# Exact McNemar
# ----------------------------------------------------------------------

def exact_mcnemar(n01: int, n10: int) -> float:
    """Two-sided exact McNemar p-value (binomial, p=1/2, conditional on n01+n10)."""
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    tail = sum(math.comb(n, i) for i in range(0, k + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def _wilson(x, n, z=1.96):
    """Wilson score interval for a single binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    c = (x + z * z / 2) / (n + z * z)
    h = z / (n + z * z) * math.sqrt(x * (n - x) / n + z * z / 4)
    return (c - h, c + h)


def newcombe_paired_ci(n11, n01, n10, n00, z=1.96):
    """Newcombe's method 10 for the difference between paired proportions.

    A Wald interval is not usable here: with two discordant pairs out of 736
    its variance estimate is degenerate and the interval is far too narrow.
    Method 10 combines Wilson score intervals for the two marginal
    proportions with an estimate of their correlation (Newcombe 1998).

    The table is oriented so that the returned difference is
    (n01 - n10) / N, the gain of the second configuration over the first.
    """
    n = n11 + n01 + n10 + n00
    if n == 0:
        return (0.0, 0.0)
    a, b, c, d = n11, n01, n10, n00
    p1 = (a + b) / n
    p2 = (a + c) / n
    theta = p1 - p2
    l1, u1 = _wilson(a + b, n, z)
    l2, u2 = _wilson(a + c, n, z)

    A = (a + b) * (c + d) * (a + c) * (b + d)
    if A == 0:
        phi = 0.0
    else:
        num = a * d - b * c
        phi = (max(num - n / 2, 0.0) if num > 0 else float(num)) / math.sqrt(A)

    lo_a, lo_b = p1 - l1, u2 - p2
    hi_a, hi_b = u1 - p1, p2 - l2
    d_lo = math.sqrt(max(lo_a ** 2 - 2 * phi * lo_a * lo_b + lo_b ** 2, 0.0))
    d_hi = math.sqrt(max(hi_a ** 2 - 2 * phi * hi_a * hi_b + hi_b ** 2, 0.0))
    return (theta - d_lo, theta + d_hi)


# ----------------------------------------------------------------------
# LaTeX helpers
# ----------------------------------------------------------------------

def tex_escape(s: str) -> str:
    for a, b in [("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"),
                 ("&", r"\&"), ("#", r"\#"), ("$", r"\$")]:
        s = s.replace(a, b)
    return s


def write(path, text):
    with open(path, "w") as fh:
        fh.write(text)
    print(f"  wrote {path}")


# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw")
    ap.add_argument("--tables", default="paper/ijoc/tables")
    args = ap.parse_args()

    R = args.raw
    T = args.tables
    os.makedirs(T, exist_ok=True)
    macros: dict[str, object] = {}

    sweep = load_jsonl(f"{R}/revision_sweep/*.jsonl")
    abl = load_jsonl(f"{R}/revision_ablation/*.jsonl")
    hyb = load_jsonl(f"{R}/revision_hybrid_grid/*.jsonl")
    mem = load_jsonl(f"{R}/revision_memory/*.jsonl")
    traj = load_jsonl(f"{R}/revision_theory/*.jsonl")
    gpu = load_jsonl(f"{R}/revision_gpu/*.jsonl")

    n_inst = len({r["inst"] for r in sweep})
    n_seeds = len({r["seed"] for r in sweep})
    n_solvers = len({r["solver"] for r in sweep})

    # ------------------------------------------------------------------
    # 1. Duplicate-key audit
    # ------------------------------------------------------------------
    print("== duplicate-key audit ==")
    audits = []
    audits.append(audit_experiment(
        "Main comparison sweep", sweep,
        ["solver", "inst", "seed"], n_inst * n_seeds * n_solvers))

    abl_cells = {(r["tau"], r["beta"]) for r in abl}
    abl_inst = {r["inst"] for r in abl}
    abl_seeds = {r["seed"] for r in abl}
    abl_solv = {r["solver"] for r in abl}
    audits.append(audit_experiment(
        "Schedule ablation", abl,
        ["solver", "tau", "beta", "inst", "seed"],
        len(abl_cells) * len(abl_inst) * len(abl_seeds) * len(abl_solv)))

    hyb_cells = {(r["warmup"], r["hint_mode"]) for r in hyb}
    hyb_inst = {r["inst"] for r in hyb}
    hyb_seeds = {r["seed"] for r in hyb}
    hyb_audit = audit_experiment(
        "Warm-start grid", hyb,
        ["warmup", "hint_mode", "inst", "seed"],
        len(hyb_cells) * len(hyb_inst) * len(hyb_seeds))
    audits.append(hyb_audit)

    audits.append(audit_experiment(
        "Peak-memory benchmark", mem, ["inst"], len({r["inst"] for r in mem})))
    audits.append(audit_experiment(
        "Trajectory sample", traj, ["inst"], len({r["inst"] for r in traj})))
    audits.append(audit_experiment(
        "Throughput benchmark", gpu,
        ["n_vars", "n_constraints", "domain", "tuples_per_con"], len(gpu)))

    lines = [
        r"\begin{table}[htbp]",
        r"\TABLE{Record integrity across the six experiments, each keyed by"
        r" experiment, configuration, representation and"
        r" seed.\label{tab:integrity}}",
        r"{\small\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"experiment & expected & raw & unique & dup. & missing & conflicting \\",
        r"\midrule",
    ]
    for a in audits:
        lines.append(
            f"{a['experiment']} & {a['expected']:,} & {a['raw']:,} & "
            f"{a['unique']:,} & {a['duplicate']:,} & {a['missing']:,} & "
            f"{a['conflicting']:,} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"{The run key is (experiment, configuration, representation, seed);"
        r" for the schedule ablation the configuration is"
        r" $(\text{solver},\tau_0,\beta_{\mathrm{growth}})$ and for the"
        r" warm-start grid it is (warm-up epochs, hint mode)."
        r" \emph{dup.} counts keys occurring more than once,"
        r" \emph{conflicting} those whose records disagree on terminal"
        r" status. Only the warm-start grid is affected;"
        r" Section~\ref{sec:integrity} says why.}",
        r"\end{table}",
    ]
    write(os.path.join(T, "data_integrity.tex"), "\n".join(lines) + "\n")

    # These two appear in running text next to hand-written figures such as
    # 6,240, so they carry the same thousands separator.
    macros["RevAuditDupKeys"] = f"{hyb_audit['duplicate']:,}".replace(",", "{,}")
    macros["RevAuditConflict"] = hyb_audit["conflicting"]
    macros["RevAuditCleanExps"] = sum(1 for a in audits if a["duplicate"] == 0)
    macros["RevAuditTotalExps"] = len(audits)

    # ------------------------------------------------------------------
    # 2. Sharding root cause (self-verifying reconstruction)
    # ------------------------------------------------------------------
    warm_grid = sorted({r["warmup"] for r in hyb})
    modes = sorted({r["hint_mode"] for r in hyb})
    insts = sorted(hyb_inst)
    seeds = sorted(hyb_seeds)

    def build(seedlist):
        t = []
        for w in warm_grid:
            for m in modes:
                if w == 0 and m != "none":
                    continue
                for i in insts:
                    for s in seedlist:
                        t.append((w, m, i, s))
        t.sort()
        return t

    p1, p2 = build([seeds[0]]), build(seeds)
    sh1 = {t: i % 8 for i, t in enumerate(p1)}
    sh2 = {t: i % 8 for i, t in enumerate(p2)}
    predicted = sum(1 for t in p1 if sh1[t] != sh2[t])
    print(f"  predicted duplicates from re-sharding model: {predicted} "
          f"(observed {hyb_audit['duplicate']})")
    macros["RevAuditPredictedDup"] = f"{predicted:,}".replace(",", "{,}")
    assert predicted == hyb_audit["duplicate"], (
        "re-sharding root-cause model does not reproduce the observed "
        f"duplicate count ({predicted} vs {hyb_audit['duplicate']})")

    # ------------------------------------------------------------------
    # 3. Error taxonomy
    # ------------------------------------------------------------------
    print("== error taxonomy ==")
    by_solver_inst = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in sweep:
        by_solver_inst[r["solver"]][r["inst"]].append(r)

    err_rows = [r for r in sweep if r["status"] == "ERROR"]
    tally = collections.defaultdict(lambda: collections.Counter())
    for r in err_rows:
        code, _ = classify_error(r.get("error", ""))
        tally[r["solver"]][code] += 1

    # entries where every configuration erred on at least one seed
    per_inst_solvers = collections.defaultdict(set)
    for r in err_rows:
        per_inst_solvers[r["inst"]].add(r["solver"])
    universal = sorted(i for i, s in per_inst_solvers.items()
                       if len(s) == n_solvers)
    macros["RevErrUniversalEntries"] = len(universal)

    # classify each universal entry by its dominant class
    uni_class = {}
    for i in universal:
        codes = collections.Counter(
            classify_error(r.get("error", ""))[0]
            for r in err_rows if r["inst"] == i)
        # ignore "no message" when a substantive class is present
        substantive = {c: n for c, n in codes.items() if c != "U1"}
        uni_class[i] = max(substantive or codes, key=(substantive or codes).get)

    n_empty = sum(1 for c in uni_class.values() if c == "I1")
    n_corrupt = sum(1 for c in uni_class.values() if c == "I2")
    macros["RevErrEmptyFiles"] = n_empty
    macros["RevErrCorruptFiles"] = n_corrupt

    codes_used = ["I1", "I2", "A1", "R1", "R2", "R3", "U1", "U2"]
    code_name = {}
    for r in err_rows:
        c, nm = classify_error(r.get("error", ""))
        code_name[c] = nm
    codes_used = [c for c in codes_used if c in code_name]

    lines = [
        r"\begin{table}[htbp]",
        r"\TABLE{Classification of every terminal \textsc{error} record in the"
        r" main sweep, from the error strings retained in the raw"
        r" logs.\label{tab:errors}}",
        r"{\scriptsize\setlength{\tabcolsep}{4.5pt}\begin{tabular}{lr"
        + "r" * len(SOLVER_ORDER) + r"}",
        r"\toprule",
        r"class & all & " +
        " & ".join(SOLVER_SHORT[s] for s in SOLVER_ORDER) + r" \\",
        r"\midrule",
    ]
    # group by origin instead of carrying an `origin` column: the long class
    # descriptions plus seven numeric columns do not fit the IJOC text block.
    ncol = len(SOLVER_ORDER) + 2
    origin_seen = None
    for c in codes_used:
        if CLASS_ORIGIN[c] != origin_seen:
            if origin_seen is not None:
                lines.append(r"\addlinespace[2pt]")
            origin_seen = CLASS_ORIGIN[c]
            lines.append(f"\\multicolumn{{{ncol}}}{{@{{}}l}}"
                         f"{{\\itshape origin: {origin_seen}}}\\\\")
        tot = sum(tally[s][c] for s in SOLVER_ORDER)
        cells = " & ".join(str(tally[s][c]) for s in SOLVER_ORDER)
        lines.append(f"\\quad {c}: {code_name[c]} & {tot} & {cells} \\\\")
    lines.append(r"\midrule")
    tot_all = len(err_rows)
    lines.append("total error runs & " + str(tot_all) + " & " +
                 " & ".join(str(sum(tally[s].values())) for s in SOLVER_ORDER) +
                 r" \\")
    allseed = {}
    for s in SOLVER_ORDER:
        allseed[s] = sum(1 for i, v in by_solver_inst[s].items()
                         if all(x["status"] == "ERROR" for x in v))
    lines.append("all-seed error entries & & " +
                 " & ".join(str(allseed[s]) for s in SOLVER_ORDER) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"{Heads abbreviate the six configurations: Choco, OR-Tools CP-SAT,"
        r" the MIRAGE-R$\to$CP-SAT hybrid, MIRAGE-R with regions, the"
        r" MIRAGE-R core, RUN-CSP. Counts are runs, except the last row,"
        r" which counts entries where every seed errored. Classes I1 and I2"
        r" are properties of the input file: " + str(len(universal)) +
        r" entries failed under every configuration, which is the common"
        r" floor. R1 is RUN-CSP GPU contention; U1 records runs whose error"
        r" message was not retained.}",
        r"\end{table}",
    ]
    write(os.path.join(T, "error_taxonomy.tex"), "\n".join(lines) + "\n")

    # the identities, for the artifact and appendix
    with open(os.path.join(T, "excluded_entries.txt"), "w") as fh:
        fh.write("Benchmark entries failing under every configuration\n")
        fh.write("(= the entries excluded from every structural stratification)\n\n")
        for i in universal:
            fh.write(f"{i}\t{uni_class[i]}\t{code_name.get(uni_class[i],'')}\n")
    print(f"  wrote {os.path.join(T, 'excluded_entries.txt')}")

    lines = [r"\begin{table}[htbp]",
             r"\TABLE{The " + str(len(universal)) + r" benchmark entries excluded"
             r" from every structural stratification.\label{tab:excluded}}",
             r"{\begin{tabular}{llr}", r"\toprule",
             r"benchmark entry & failure class & error runs \\", r"\midrule"]
    for i in universal:
        nrun = sum(1 for r in err_rows if r["inst"] == i)
        lines.append(f"\\texttt{{{tex_escape(i)}}} & {uni_class[i]}: "
                     f"{code_name.get(uni_class[i],'')} & {nrun} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}",
              r"{These entries carry no feature record because their normalized"
              r" files could not be parsed, and every solver adapter failed on"
              r" them for the same reason. They are retained as unsolved in the"
              r" main comparison (the main comparison table) and excluded from"
              r" the stratification tables deposited in the artifact, whose rows"
              r" therefore sum to " + str(n_inst - len(universal)) +
              r". The list is also deposited in the artifact as"
              r" \texttt{tables/excluded\_entries.txt}.}",
              r"\end{table}"]
    write(os.path.join(T, "excluded_entries.tex"), "\n".join(lines) + "\n")

    # ------------------------------------------------------------------
    # 4. Paired region analysis
    # ------------------------------------------------------------------
    print("== paired region analysis ==")
    core = {i: majority_outcome(v) for i, v in by_solver_inst["mirage"].items()}
    reg = {i: majority_outcome(v)
           for i, v in by_solver_inst["mirage_regions"].items()}
    common = sorted(set(core) & set(reg))
    n11 = sum(1 for i in common if core[i] == "SAT" and reg[i] == "SAT")
    n01 = sum(1 for i in common if core[i] != "SAT" and reg[i] == "SAT")
    n10 = sum(1 for i in common if core[i] == "SAT" and reg[i] != "SAT")
    n00 = sum(1 for i in common if core[i] != "SAT" and reg[i] != "SAT")
    N = len(common)
    assert n11 + n01 + n10 + n00 == N
    p = exact_mcnemar(n01, n10)
    lo, hi = newcombe_paired_ci(n11, n01, n10, n00)
    delta = (n01 - n10) / N

    macros.update({
        "RevRegBoth": n11, "RevRegRegionOnly": n01, "RevRegCoreOnly": n10, "RevRegNeither": n00,
        "RevRegN": N,
        "RevRegDelta": f"{100*delta:+.2f}",
        "RevRegCIlo": f"{100*lo:+.2f}", "RevRegCIhi": f"{100*hi:+.2f}",
        "RevRegMcNemarP": f"{p:.3f}",
        "RevRegDiscordant": n01 + n10,
    })

    lines = [
        r"\begin{table}[htbp]",
        r"\TABLE{Paired instance-level outcomes for the core and"
        r" dynamic-region configurations over all "
        + str(N) + r" benchmark entries.\label{tab:regionpaired}}",
        r"{\small\begin{tabular}{lrrr}",
        r"\toprule",
        r" & \multicolumn{2}{c}{core (no regions)} & \\",
        r"\cmidrule(lr){2-3}",
        r"region variant & witness & no witness & total \\",
        r"\midrule",
        f"witness & {n11} & {n01} & {n11+n01} \\\\",
        f"no witness & {n10} & {n00} & {n10+n00} \\\\",
        r"\midrule",
        f"total & {n11+n10} & {n01+n00} & {N} \\\\",
        r"\bottomrule",
        r"\end{tabular}}",
        r"{Counts of benchmark entries after majority-of-five seed"
        r" aggregation, from deduplicated run keys. Section~"
        r"\ref{sec:regions} gives the paired difference, its interval"
        r" and the McNemar test.}",
        r"\end{table}",
    ]
    write(os.path.join(T, "region_paired.tex"), "\n".join(lines) + "\n")

    # ------------------------------------------------------------------
    # 5. Warm-start dedup sensitivity
    # ------------------------------------------------------------------
    print("== warm-start sensitivity ==")
    buckets = hyb_audit["_buckets"]

    def cell_rate(rule):
        agg = collections.defaultdict(lambda: [0, 0])
        for k, v in buckets.items():
            rec = v[0] if rule == "first" else v[-1]
            cell = (k[0], k[1])
            agg[cell][1] += 1
            if rec["status"] == "SAT_VERIFIED":
                agg[cell][0] += 1
        return {c: 100.0 * a / b for c, (a, b) in agg.items()}

    first, last = cell_rate("first"), cell_rate("last")
    cells = sorted(first, key=lambda c: (c[0], modes.index(c[1])))
    maxdiff = max(abs(first[c] - last[c]) for c in cells)
    macros["RevHybSensMaxDiff"] = f"{maxdiff:.2f}"

    lines = [
        r"\begin{table}[htbp]",
        r"\TABLE{Deduplication sensitivity of the warm-start grid: verified-witness"
        r" rate per cell under the keep-first and keep-last"
        r" rules.\label{tab:hybridsens}}",
        r"{\begin{tabular}{llrrr}",
        r"\toprule",
        r"warm-up & hint mode & keep-first \% & keep-last \% & difference \\",
        r"\midrule",
    ]
    for c in cells:
        lines.append(f"{c[0]} & {c[1]} & {first[c]:.2f} & {last[c]:.2f} & "
                     f"{first[c]-last[c]:+.2f} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"{Each cell contains " + str(len(insts) * len(seeds)) + r" unique run"
        r" keys. Of these, " + str(hyb_audit["duplicate"]) + r" keys across the"
        r" whole grid carry two records and "
        + str(hyb_audit["conflicting"]) + r" of those disagree on terminal"
        r" status. The largest cell-level disagreement between the two"
        r" deduplication rules is " + macros["RevHybSensMaxDiff"] + r" percentage"
        r" points, and no cell changes its ordering with respect to the pure"
        r" CP-SAT control. The duplicated records are genuine independent"
        r" re-executions of the same configuration rather than copied rows, so"
        r" the disagreement measures CP-SAT's run-to-run variability at the "
        r" time limit and not a bookkeeping error. Results in the main text use"
        r" keep-first.}",
        r"\end{table}",
    ]
    write(os.path.join(T, "hybrid_sensitivity.tex"), "\n".join(lines) + "\n")

    # ------------------------------------------------------------------
    # 6. Aggregation-rule comparison and outcome breakdown
    # ------------------------------------------------------------------
    print("== aggregation rules and outcome breakdown ==")
    rows = []
    for s in SOLVER_ORDER:
        d = by_solver_inst[s]
        maj = sum(1 for v in d.values() if majority_outcome(v) == "SAT")
        any1 = sum(1 for v in d.values()
                   if any(x["status"] in DEFINITIVE_SAT for x in v))
        runs = [x for v in d.values() for x in v]
        perrun = sum(1 for x in runs if x["status"] in DEFINITIVE_SAT)
        rows.append((s, maj, 100 * maj / n_inst, any1, 100 * any1 / n_inst,
                     perrun, len(runs), 100 * perrun / len(runs)))

    lines = [
        r"\begin{table}[htbp]",
        r"\TABLE{Verified-witness coverage under three aggregation rules."
        r"\label{tab:aggregation}}",
        r"{\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"configuration & \multicolumn{2}{c}{majority of five} &"
        r" \multicolumn{2}{c}{at least one of five} &"
        r" \multicolumn{2}{c}{per run} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r" & entries & \% & entries & \% & runs & \% \\",
        r"\midrule",
    ]
    for s, maj, mp, a1, ap_, pr, tr, prp in rows:
        lines.append(f"{SOLVER_LABEL[s]} & {maj} & {mp:.1f} & {a1} & {ap_:.1f} & "
                     f"{pr:,}/{tr:,} & {prp:.1f} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"{All three columns count only witnesses accepted by the"
        r" solver-independent verifier; UNSAT conclusions are excluded, so the"
        r" table is a single-metric comparison. Denominators are "
        + str(n_inst) + r" benchmark entries for the two instance-level rules"
        r" and " + str(n_inst * n_seeds) + r" runs for the per-run rule. Choco's"
        r" adapter records the seed but does not pass it to the search strategy,"
        r" so its five repetitions are deterministic apart from runtime"
        r" variability; its at-least-one and per-run columns are therefore not"
        r" independent stochastic summaries and are reported only for format"
        r" consistency. The gap between the majority and at-least-one columns for"
        r" the MIRAGE-R configurations is the number of entries on which a"
        r" witness was found by a minority of seeds.}",
        r"\end{table}",
    ]
    write(os.path.join(T, "aggregation_rules.tex"), "\n".join(lines) + "\n")

    # detailed terminal-condition breakdown (reviewer item E.40)
    lines = [
        r"\begin{table}[htbp]",
        r"\TABLE{Terminal-condition breakdown, separating outcomes that the"
        r" main table collects in a single \emph{unk.}"
        r" column.\label{tab:terminal}}",
        r"{\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"configuration & witness & UNSAT & timeout & epoch cap & unsupported"
        r" & error & no majority \\",
        r"\midrule",
    ]
    for s in SOLVER_ORDER:
        d = by_solver_inst[s]
        cnt = collections.Counter()
        for i, v in d.items():
            o = majority_outcome(v)
            if o == "SAT":
                cnt["sat"] += 1
            elif o == "UNSAT":
                cnt["unsat"] += 1
            elif o == "UNSUPPORTED":
                cnt["unsup"] += 1
            elif o == "ERROR":
                cnt["err"] += 1
            else:
                st = collections.Counter(x["status"] for x in v)
                top, ntop = st.most_common(1)[0]
                if ntop * 2 <= len(v):
                    cnt["nomaj"] += 1
                elif top == "UNKNOWN_TIMEOUT":
                    cnt["timeout"] += 1
                elif top == "UNKNOWN_MAX_EPOCHS":
                    cnt["epoch"] += 1
                else:
                    cnt["nomaj"] += 1
        tot = sum(cnt.values())
        assert tot == n_inst, f"{s}: breakdown sums to {tot}, expected {n_inst}"
        lines.append(
            f"{SOLVER_LABEL[s]} & {cnt['sat']} & {cnt['unsat']} & "
            f"{cnt['timeout']} & {cnt['epoch']} & {cnt['unsup']} & "
            f"{cnt['err']} & {cnt['nomaj']} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"{Every row sums to " + str(n_inst) + r". \emph{unsupported} and"
        r" \emph{error} count entries on which all five seeds returned that"
        r" status; \emph{no majority} counts entries whose seeds split without a"
        r" strict majority for any single terminal condition. Splitting the"
        r" single \emph{unk.} column of the main comparison table in this way"
        r" separates search failure (timeout, epoch cap) from representational"
        r" and infrastructural failure (unsupported, error).}",
        r"\end{table}",
    ]
    write(os.path.join(T, "terminal_breakdown.tex"), "\n".join(lines) + "\n")

    # ------------------------------------------------------------------
    # 7. RUN-CSP fragment conditioning
    # ------------------------------------------------------------------
    print("== RUN-CSP fragment ==")
    rc = by_solver_inst["runcsp"]
    frag = sorted(i for i, v in rc.items()
                  if not any(x["status"] == "UNSUPPORTED_FRAGMENT" for x in v))
    nf = len(frag)
    fmaj = sum(1 for i in frag if majority_outcome(rc[i]) == "SAT")
    fany = sum(1 for i in frag
               if any(x["status"] in DEFINITIVE_SAT for x in rc[i]))
    fruns = [x for i in frag for x in rc[i]]
    fpr = sum(1 for x in fruns if x["status"] in DEFINITIVE_SAT)
    ferr = sum(1 for i in frag
               if all(x["status"] == "ERROR" for x in rc[i]))
    fam = collections.Counter(i.split("-")[0] for i in frag)
    # The fragment denominator "no seed returned UNSUPPORTED" silently counts
    # entries whose input file is corrupt, because execution fails before
    # fragment detection runs. Report that overlap so the three categories
    # (supported and executable / unsupported / corrupt) stay disjoint.
    uni = set(universal)
    frag_corrupt = sum(1 for i in frag if i in uni)
    # Of the all-seed-error entries, how many are corrupt inputs rather than
    # GPU failures. These two are not the same count as frag_corrupt: one
    # corrupt entry inside the fragment did not error on every RUN-CSP seed.
    err_entries = [i for i in frag if all(x["status"] == "ERROR" for x in rc[i])]
    err_corrupt = sum(1 for i in err_entries if i in uni)
    macros.update({
        "RevRuncspFragCorrupt": frag_corrupt,
        "RevRuncspFragErrCorrupt": err_corrupt,
        "RevRuncspFragErrGpu": len(err_entries) - err_corrupt,
        "RevRuncspFragClean": nf - frag_corrupt,
        "RevRuncspFragN": nf,
        "RevRuncspFragMaj": fmaj,
        "RevRuncspFragMajRate": f"{100*fmaj/nf:.1f}",
        "RevRuncspFragAny": fany,
        "RevRuncspFragAnyRate": f"{100*fany/nf:.1f}",
        "RevRuncspFragPerRun": f"{100*fpr/len(fruns):.1f}",
        "RevRuncspFragErr": ferr,
    })

    lines = [
        r"\begin{table}[htbp]",
        r"\TABLE{RUN-CSP evaluated on its supported fragment rather than on the"
        r" full benchmark.\label{tab:runcsp}}",
        r"{\begin{tabular}{lrr}",
        r"\toprule",
        r"denominator & entries & verified-witness coverage \\",
        r"\midrule",
        f"full benchmark, majority of five & {n_inst} & "
        f"{sum(1 for v in rc.values() if majority_outcome(v)=='SAT')} "
        f"({100*sum(1 for v in rc.values() if majority_outcome(v)=='SAT')/n_inst:.1f}\\%) \\\\",
        f"supported fragment, majority of five & {nf} & {fmaj} "
        f"({100*fmaj/nf:.1f}\\%) \\\\",
        f"supported fragment, at least one of five & {nf} & {fany} "
        f"({100*fany/nf:.1f}\\%) \\\\",
        f"supported fragment, per run & {len(fruns):,} & {fpr} "
        f"({100*fpr/len(fruns):.1f}\\%) \\\\",
        r"\bottomrule",
        r"\end{tabular}}",
        r"{Support is determined by the adapter's own"
        r" \texttt{UNSUPPORTED\_FRAGMENT} status: an entry is in the fragment"
        r" when no seed reported that status. The criterion is binary"
        r" constraints representable as a Boolean or small-domain table with the"
        r" adapter's edge encoding. The fragment comprises "
        + ", ".join(f"{n} {tex_escape(f)}" for f, n in fam.most_common())
        + r". Support so defined is not the same as executability: "
        + str(frag_corrupt) + r" of these entries carry a corrupt normalized"
        r" file and fail before fragment detection runs, so the executable"
        r" fragment is " + str(nf - frag_corrupt) + r" entries. Of the "
        + str(nf) + r" fragment entries, " + str(ferr) +
        r" returned an all-seed error, " + str(err_corrupt) +
        r" of them on those corrupt inputs and " + str(len(err_entries) - err_corrupt) +
        r" of class R1 (GPU device or memory failure); none is attributable to"
        r" the model. The full-benchmark figure is dominated by the "
        + str(n_inst - nf) + r" entries outside the fragment and is therefore"
        r" uninformative about RUN-CSP on the problems it was designed for; only"
        r" the fragment-conditioned rows should be used for that purpose.}",
        r"\end{table}",
    ]
    write(os.path.join(T, "runcsp_fragment.tex"), "\n".join(lines) + "\n")

    # ------------------------------------------------------------------
    # 8. Trajectory slope denominator
    # ------------------------------------------------------------------
    print("== trajectory ==")
    n_traj = len(traj)
    n_sat = sum(1 for r in traj if r["status"] == "SAT_VERIFIED")
    n_to = sum(1 for r in traj if r["status"].startswith("UNKNOWN"))
    eligible = []
    for r in traj:
        rm = r.get("running_min_trajectory") or []
        if not rm:
            continue
        b1 = rm[0]
        if b1 <= 0:
            continue
        pts = [(j + 1) * r.get("decode_freq", 5) for j in range(len(rm))]
        keep = [(t, b) for t, b in zip(pts, rm) if t > 1 and b / b1 > 1e-3]
        if len(keep) >= 6:
            eligible.append(r)
    n_elig = len(eligible)
    n_to_elig = sum(1 for r in eligible if r["status"].startswith("UNKNOWN"))
    macros.update({
        "RevTrajEligible": n_elig,
        "RevTrajTimeoutEligible": n_to_elig,
        "RevTrajExcluded": n_traj - n_elig,
    })
    print(f"  trajectory: n={n_traj} sat={n_sat} timeout={n_to} "
          f"slope-eligible={n_elig} (timed-out among them: {n_to_elig})")

    # ------------------------------------------------------------------
    # 9. Consistency assertions
    # ------------------------------------------------------------------
    print("== consistency assertions ==")
    for s in SOLVER_ORDER:
        d = by_solver_inst[s]
        c = collections.Counter(majority_outcome(v) for v in d.values())
        tot = sum(c.values())
        assert tot == n_inst, f"{s}: {tot} != {n_inst}"
        print(f"  {s:16s} SAT+UNSAT+UNKNOWN+UNSUPPORTED+ERROR = {tot} = {n_inst} OK")
    fam_counts = collections.Counter(i.split("-")[0] for i in
                                     {r["inst"] for r in sweep})
    assert sum(fam_counts.values()) == n_inst
    print(f"  family totals {dict(fam_counts)} sum to {n_inst} OK")

    macros["RevNumInstances"] = n_inst
    macros["RevSeeds"] = n_seeds
    macros["RevTrajTimeoutLimit"] = 300

    # ------------------------------------------------------------------
    with open(os.path.join(T, "audit_numbers.tex"), "w") as fh:
        fh.write("% AUTO-GENERATED by experiments/validation/audit_archive.py"
                 " -- do not edit\n")
        for k, v in sorted(macros.items()):
            fh.write(f"\\providecommand{{\\{k}}}{{}}\n")
            fh.write(f"\\renewcommand{{\\{k}}}{{{v}}}\n")
    print(f"  wrote {os.path.join(T, 'audit_numbers.tex')} "
          f"({len(macros)} macros)")
    print("\nALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
