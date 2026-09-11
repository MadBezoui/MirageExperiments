"""Aggregate all revision-sweep shards into the paper's tables, figures, and
number macros.

Reads sharded JSONL outputs of run_sweep / run_ablation_full / run_memory_full
/ run_theory / run_hybrid_grid / gpu_throughput plus the instance-feature CSV
and produces (revision plan P0.2, P3.1-P3.5):

  tables/  main_comparison.tex      Table 2 incl. supported-fragment column
           strata_*.tex             per-stratum solve rates (family, arity,
                                    density, domain size) -- MIRAGE's niche
           hybrid_deepdive.tex      Table 1: full warmup x hint-mode grid
           ablation_full.tex        full-set tau x beta pivot (PRIMARY)
           region_summary.tex       real region firing statistics
           memory_summary.tex       fitted log-log slopes, not adjectives
           protocol.tex             the single protocol appendix table
           wilcoxon.tex             Holm-corrected signed-rank tests
           numbers.tex              newcommand macros: every number the
                                    manuscript's prose cites
           soundness.txt            zero-false-positive + adapter-mismatch audit
  figures/ solve_rate_density.pdf   binned solve-rate with Wilson 95% CIs
           ablation_tau_beta.pdf, memory_scaling.pdf, cactus_plot.pdf,
           performance_profile.pdf, convergence.pdf, gpu_throughput.pdf

Every macro gets a safe default so the manuscript compiles even before all
phases finish; re-run this after each phase.

Usage:
  PYTHONPATH=. python -m experiments.validation.aggregate \
      --raw-dir results/raw --timeout 300 \
      --features data/frozen/instance_features.csv \
      --tables-dir paper/ijoc/tables --figures-dir paper/ijoc/figures
"""
from __future__ import annotations
import argparse
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 13})

SOLVED = {"SAT_VERIFIED", "UNSAT"}
PRETTY = {
    "mirage": "MIRAGE-R", "mirage_regions": "MIRAGE-R+regions",
    "hybrid": "MIRAGE-R$\\to$CP-SAT", "ortools": "OR-Tools CP-SAT",
    "choco": "Choco", "gecode": "Gecode", "runcsp": "RUN-CSP",
}
MACRO_NAME = {
    "mirage": "Mirage", "mirage_regions": "MirageRegions", "hybrid": "Hybrid",
    "ortools": "CpSat", "choco": "Choco", "gecode": "Gecode",
    "runcsp": "RunCsp",
}

MACROS = {}


def set_macro(name, value):
    if isinstance(value, float):
        value = f"{value:.1f}" if abs(value) >= 0.1 else f"{value:.3f}"
    MACROS[name] = str(value)


def _default_macros():
    for s in MACRO_NAME.values():
        for stem in ("Solved", "SolveRate", "MedianTime", "ParTwo"):
            MACROS.setdefault(f"\\Rev{stem}{s}", "--")
    for k in ("\\RevNumInstances", "\\RevStrataN", "\\RevStrataExcluded",
              "\\RevRuncspFragment", "\\RevSeeds",
              "\\RevTimeout", "\\RevAdapterMismatch", "\\RevFalsePositives",
              "\\RevBestStratumName", "\\RevBestStratumMirage",
              "\\RevBestStratumCpsat", "\\RevBestStratumN",
              "\\RevAblBestTau", "\\RevAblBestBeta", "\\RevAblBestRate",
              "\\RevAblWorstRate", "\\RevAblSpread", "\\RevAblRegionsDelta",
              "\\RevMemSlopeMirage", "\\RevMemSlopeCpsat",
              "\\RevMemMedianRatio", "\\RevMemN",
              "\\RevHybBestCell", "\\RevHybBestRate", "\\RevHybCpsatRate",
              "\\RevHybBestMedTime", "\\RevHybCpsatMedTime",
              "\\RevGpuLargestN", "\\RevGpuSpeedupCuda", "\\RevGpuSpeedupVec",
              "\\RevRegionsFiredPct", "\\RevRegionsMeanAdded",
              "\\RevRegionsFiredSolveRate", "\\RevRegionsFiredBaseRate",
              "\\RevRegionsWilcoxonP",
              "\\RevConvergenceSlope", "\\RevChocoOnlyN",
              "\\RevChocoOnlyTopFam", "\\RevChocoOnlyTopFamN",
              "\\RevChocoOnlyMedTime"):
        MACROS.setdefault(k, "--")


# --------------------------------------------------------------------------- #
def canon_instance(name):
    """Canonical instance key: some adapters logged 'X.json', others 'X'.
    Without this, per-solver instance sets do not align (the Choco 737-row
    bug), cross-solver soundness comparisons silently miss, and nunique()
    double-counts."""
    if isinstance(name, str) and name.endswith(".json"):
        return name[:-5]
    return name


def read_jsonl(patterns):
    rows = []
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if "instance" in r:
                        r["instance"] = canon_instance(r["instance"])
                    rows.append(r)
    return rows


def family(inst):
    base = inst.replace(".json", "")
    return base.split("-")[0] if "-" in base else base


def save_fig(fig, figdir, name):
    os.makedirs(figdir, exist_ok=True)
    fig.savefig(os.path.join(figdir, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(figdir, name + ".png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def save_fig_alias(figures_dir, src, dst):
    import shutil
    for ext in (".pdf", ".png"):
        s = os.path.join(figures_dir, src + ext)
        if os.path.exists(s):
            shutil.copyfile(s, os.path.join(figures_dir, dst + ext))


def _esc_header(s):
    """Escape % and _ for LaTeX, but only if not already escaped, so column
    names that already contain \\% or \\_ are not double-escaped into \\\\%
    (which would render as a line break plus a comment and break the header)."""
    import re
    s = str(s)
    s = re.sub(r"(?<!\\)%", r"\\%", s)
    s = re.sub(r"(?<!\\)_", r"\\_", s)
    return s


def latex_table(df, caption, label, floatfmt="%.2f", note=""):
    cols = list(df.columns)
    out = ["\\begin{table}[htbp]", f"\\TABLE{{{caption}\\label{{{label}}}}}",
           "{\\begin{tabular}{l" + "r" * (len(cols) - 1) + "}", "\\toprule",
           " & ".join(_esc_header(c) for c in cols) + " \\\\", "\\midrule"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("--" if math.isnan(v) else (floatfmt % v))
            else:
                s = str(v)
                if "$" not in s and "\\" not in s:
                    s = s.replace("_", "\\_")
                cells.append(s)
        out.append(" & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}}",
            "{" + note + "}", "\\end{table}", ""]
    return "\n".join(out)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(centre - half, 0.0), min(centre + half, 1.0)


# --------------------------------------------------------------------------- #
def aggregate_sweep(rows, timeout):
    """Per (solver, instance): majority-of-seeds solved flag, median time,
    PAR2 (unsolved seed counted as 2*timeout)."""
    by = defaultdict(list)
    for r in rows:
        if "solver" in r and "instance" in r:
            by[(r["solver"], r["instance"])].append(r)
    recs = []
    for (solver, inst), rs in by.items():
        statuses = [x.get("status") for x in rs]
        times = [x.get("time") if isinstance(x.get("time"), (int, float))
                 else timeout for x in rs]
        solved = [s in SOLVED for s in statuses]
        is_solved = sum(solved) > len(solved) / 2      # majority rule
        recs.append({
            "solver": solver, "instance": inst, "family": family(inst),
            "n_seeds": len(rs),
            "solved": is_solved,
            # SAT/UNSAT split of the *solved* instances so that
            # SAT + UNSAT == solved in the main table (a solver never returns
            # both on the same instance -- the soundness audit checks this).
            "sat": is_solved and any(s == "SAT_VERIFIED" for s in statuses),
            "unsat": is_solved and not any(s == "SAT_VERIFIED"
                                           for s in statuses),
            "unsupported": all(s == "UNSUPPORTED_FRAGMENT" for s in statuses),
            "mismatch": any(s == "ADAPTER_MISMATCH" for s in statuses),
            "error": all(s == "ERROR" for s in statuses),
            "median_time": float(np.median(times)),
            "par2": float(np.mean([t if sv else 2 * timeout
                                   for t, sv in zip(times, solved)])),
            "regions_added": int(np.median([x.get("regions_added", 0) or 0
                                            for x in rs])),
        })
    return pd.DataFrame(recs)


def runcsp_fragment_from_logs(df):
    """Operational fragment: instances the RUN-CSP adapter accepted (i.e. not
    every seed returned UNSUPPORTED_FRAGMENT).  This is the adapter's own
    binary-table check, so it needs no external feature file."""
    g = df[df["solver"] == "runcsp"]
    if g.empty:
        return set()
    return set(g.loc[~g["unsupported"], "instance"])


def main_table(df, feats, timeout, tables_dir):
    os.makedirs(tables_dir, exist_ok=True)
    frag = set()
    if feats is not None and "runcsp_fragment" in feats.columns:
        frag = set(feats.loc[feats["runcsp_fragment"] == 1, "instance"])
    if not frag:
        frag = runcsp_fragment_from_logs(df)
    rows = []
    for solver, g in df.groupby("solver"):
        nsolved = int(g["solved"].sum())
        in_frag = g[g["instance"].isin(frag)] if frag else g.iloc[0:0]
        rows.append({
            "solver": PRETTY.get(solver, solver),
            "\\#inst": len(g),
            "solved": nsolved,
            "SAT": int(g["sat"].sum()),
            "UNSAT": int(g["unsat"].sum()),
            "unsup.": int(g["unsupported"].sum()),
            "error": int(g["error"].sum()),
            "cov.\\%": 100.0 * nsolved / max(len(g), 1),
            "cov.frag\\%": (100.0 * in_frag["solved"].sum()
                            / max(len(in_frag), 1)) if len(in_frag) else float("nan"),
            "med.time(s)": float(g.loc[g["solved"], "median_time"].median())
            if nsolved else float("nan"),
            "PAR2(s)": float(g["par2"].mean()),
        })
        m = MACRO_NAME.get(solver)
        if m:
            set_macro(f"\\RevSolved{m}", nsolved)
            set_macro(f"\\RevSolveRate{m}", 100.0 * nsolved / max(len(g), 1))
            if nsolved:
                set_macro(f"\\RevMedianTime{m}",
                          float(g.loc[g["solved"], "median_time"].median()))
            set_macro(f"\\RevParTwo{m}", float(g["par2"].mean()))
    # locate the Choco-vs-CP-SAT gap (macro-bound so the prose can explain it)
    solved_sets = {s: set(g.loc[g["solved"], "instance"])
                   for s, g in df.groupby("solver")}
    if "choco" in solved_sets and "ortools" in solved_sets:
        gap = solved_sets["choco"] - solved_sets["ortools"]
        set_macro("\\RevChocoOnlyN", len(gap))
        if gap:
            fams = pd.Series([family(i) for i in gap])
            top = fams.value_counts()
            set_macro("\\RevChocoOnlyTopFam", str(top.index[0]))
            set_macro("\\RevChocoOnlyTopFamN", int(top.iloc[0]))
            gm = df[(df["solver"] == "choco") & (df["instance"].isin(gap))]
            set_macro("\\RevChocoOnlyMedTime",
                      float(gm["median_time"].median()))

    tab = pd.DataFrame(rows).sort_values("cov.\\%", ascending=False)
    tab.to_csv(os.path.join(tables_dir, "main_comparison.csv"), index=False)
    note = ("Majority-of-seeds solve rule. PAR2 counts an unsolved run as twice "
            "the time limit. `unsup.' = instances outside a method's supported "
            "fragment (RUN-CSP restricted to binary table constraints), and "
            "`cov.frag' is coverage restricted to that fragment.")
    with open(os.path.join(tables_dir, "main_comparison.tex"), "w") as f:
        f.write(latex_table(tab, "Main comparison on the frozen benchmark set.",
                            "tab:main", note=note))
    set_macro("\\RevNumInstances", int(df["instance"].nunique()))
    if frag:
        set_macro("\\RevRuncspFragment", len(frag))
    return tab


def strata_tables(df, feats, tables_dir):
    """P3.1: per-stratum solve rates; find MIRAGE's niche quantitatively."""
    # per-family table needs no external features -- always regenerate it
    piv = (df.groupby(["family", "solver"])["solved"].mean().mul(100)
           .round(1).unstack("solver").fillna(0.0).reset_index())
    fam_counts = df.groupby("family")["instance"].nunique()
    piv.insert(1, "\\#inst", piv["family"].map(fam_counts).astype(int).values)
    piv.columns = [PRETTY.get(c, c) for c in piv.columns]
    piv.to_csv(os.path.join(tables_dir, "per_family_coverage.csv"), index=False)
    with open(os.path.join(tables_dir, "per_family_coverage.tex"), "w") as f:
        f.write(latex_table(piv, "Per-family solve rate (\\%).",
                            "tab:perfamily", floatfmt="%.1f"))
    if feats is None:
        return
    d = df.merge(feats, on="instance", how="left", suffixes=("", "_f"))
    # An instance enters a stratum only if its stratifying features are defined.
    # Report the covered count and the excluded remainder so the per-stratum
    # counts (which sum to RevStrataN, not RevNumInstances) are reconciled.
    n_total = df["instance"].nunique()
    feat_cols = [c for c in ("max_arity", "density", "max_domain",
                             "dominant_type") if c in d.columns]
    covered = (d.dropna(subset=feat_cols)["instance"].nunique()
               if feat_cols else n_total)
    set_macro("\\RevStrataN", covered)
    set_macro("\\RevStrataExcluded", n_total - covered)
    d["arity_bucket"] = pd.cut(d["max_arity"], [0, 2, 4, 8, np.inf],
                               labels=["$\\le$2", "3--4", "5--8", "$>$8"])
    d["density_bucket"] = pd.cut(d["density"], [0, 0.5, 1, 2, 4, np.inf],
                                 labels=["$\\le$0.5", "0.5--1", "1--2",
                                         "2--4", "$>$4"])
    d["domain_bucket"] = pd.cut(d["max_domain"], [0, 2, 8, 32, np.inf],
                                labels=["2", "3--8", "9--32", "$>$32"])
    strat_defs = {
        "dominant_type": ("strata_type", "dominant constraint type"),
        "arity_bucket": ("strata_arity", "max constraint arity"),
        "density_bucket": ("strata_density", "constraint density $|C|/|X|$"),
        "domain_bucket": ("strata_domain", "max domain size"),
    }
    best = None
    for col, (fname, nice) in strat_defs.items():
        piv = (d.groupby([col, "solver"], observed=True)["solved"].mean()
               .mul(100).round(1).unstack("solver"))
        counts = d.groupby(col, observed=True)["instance"].nunique()
        piv.insert(0, "\\#inst", counts)
        piv = piv.reset_index().rename(columns={col: nice})
        piv.columns = [PRETTY.get(c, c) for c in piv.columns]
        piv.to_csv(os.path.join(tables_dir, fname + ".csv"), index=False)
        with open(os.path.join(tables_dir, fname + ".tex"), "w") as f:
            f.write(latex_table(
                piv, f"Solve rate (\\%) stratified by {nice}.",
                "tab:" + fname, floatfmt="%.1f"))
        for _, r in piv.iterrows():
            n = r.get("\\#inst", 0)
            try:
                n = int(n)
            except Exception:
                continue
            if n < 10:
                continue
            vals = []
            for key in (PRETTY["mirage"], PRETTY["mirage_regions"]):
                try:
                    vals.append(float(r[key]))
                except Exception:
                    pass
            if not vals:
                continue
            mir = max(vals)
            try:
                cps = float(r[PRETTY["ortools"]])
            except Exception:
                continue
            if math.isnan(mir) or math.isnan(cps):
                continue
            delta = mir - cps
            if best is None or delta > best[0]:
                best = (delta, f"{nice} = {r.iloc[0]}", mir, cps, n)
    if best:
        set_macro("\\RevBestStratumName", best[1])
        set_macro("\\RevBestStratumMirage", best[2])
        set_macro("\\RevBestStratumCpsat", best[3])
        set_macro("\\RevBestStratumN", best[4])

def solve_rate_density_fig(df, feats, figures_dir):
    """P3.2: replace the binary scatter with binned solve rate + Wilson CIs."""
    if feats is None:
        return
    d = df[df["solver"].isin(["mirage", "mirage_regions", "ortools"])].merge(
        feats[["instance", "density"]], on="instance", how="left").dropna(
        subset=["density"])
    if d.empty:
        return
    edges = np.unique(np.quantile(d["density"], np.linspace(0, 1, 9)))
    if len(edges) < 3:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for solver, marker in (("mirage", "o"), ("mirage_regions", "s"),
                           ("ortools", "^")):
        g = d[d["solver"] == solver]
        if g.empty:
            continue
        xs, ys, lo, hi = [], [], [], []
        for i in range(len(edges) - 1):
            sel = g[(g["density"] >= edges[i]) & (g["density"] <= edges[i + 1])]
            if len(sel) < 5:
                continue
            p, l, h = wilson_ci(int(sel["solved"].sum()), len(sel))
            xs.append(0.5 * (edges[i] + edges[i + 1]))
            ys.append(p)
            lo.append(p - l)
            hi.append(h - p)
        ax.errorbar(xs, ys, yerr=[lo, hi], marker=marker, capsize=3,
                    label=PRETTY.get(solver, solver))
    ax.set_xlabel("constraint density $|C|/|X|$ (quantile bins)")
    ax.set_ylabel("solve rate")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    save_fig(fig, figures_dir, "solve_rate_density")
    save_fig_alias(figures_dir, "solve_rate_density", "failure_density")


def cactus_and_profile(df, timeout, figures_dir):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for solver, g in df.groupby("solver"):
        t = np.sort(g.loc[g["solved"], "median_time"].values)
        if len(t) == 0:
            continue
        ax.plot(np.arange(1, len(t) + 1), t, label=PRETTY.get(solver, solver))
    ax.set_yscale("log")
    ax.set_xlabel("instances solved")
    ax.set_ylabel("time (s)")
    ax.legend(fontsize=9)
    save_fig(fig, figures_dir, "cactus_plot")

    piv = df.pivot_table(index="instance", columns="solver", values="par2")
    best = piv.min(axis=1)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    taus = np.logspace(0, np.log10(max(2 * timeout, 10)), 200)
    for solver in piv.columns:
        ratios = (piv[solver] / best).dropna()
        ys = [(ratios <= t).mean() for t in taus]
        ax.plot(taus, ys, label=PRETTY.get(solver, solver))
    ax.set_xscale("log")
    ax.set_xlabel(r"performance ratio $\tau$")
    ax.set_ylabel(r"fraction of instances within $\tau$ of best")
    ax.legend(fontsize=9)
    save_fig(fig, figures_dir, "performance_profile")


def soundness(rows, tables_dir):
    unsat_by_complete, sat_claims = set(), set()
    mismatches = []
    for r in rows:
        st = r.get("status")
        if r.get("solver") in ("ortools", "choco", "gecode", "hybrid") \
                and st == "UNSAT":
            unsat_by_complete.add(r["instance"])
        if r.get("solver") in ("mirage", "mirage_regions") \
                and st == "SAT_VERIFIED":
            sat_claims.add(r["instance"])
        if st == "ADAPTER_MISMATCH":
            mismatches.append((r.get("solver"), r.get("instance")))
    conflicts = sorted(unsat_by_complete & sat_claims)
    with open(os.path.join(tables_dir, "soundness.txt"), "w") as f:
        f.write("SOUNDNESS + ADAPTER AUDIT\n")
        f.write(f"UNSAT proved by a complete solver: {len(unsat_by_complete)}\n")
        f.write(f"MIRAGE-family SAT_VERIFIED claims:  {len(sat_claims)}\n")
        f.write(f"false positives (conflicts):        {len(conflicts)}\n")
        for c in conflicts:
            f.write(f"  CONFLICT {c}\n")
        f.write(f"adapter mismatches:                 {len(mismatches)}\n")
        for s, i in mismatches[:50]:
            f.write(f"  MISMATCH {s} {i}\n")
        f.write("RESULT: " + ("PASS" if not conflicts and not mismatches
                              else "FAIL") + "\n")
    set_macro("\\RevFalsePositives", len(conflicts))
    set_macro("\\RevAdapterMismatch", len(mismatches))


# --------------------------------------------------------------------------- #
def ablation(rows, tables_dir, figures_dir):
    if not rows:
        return
    df = pd.DataFrame([r for r in rows if "tau" in r and "beta" in r])
    if df.empty:
        return
    df["solved"] = df["status"].isin(SOLVED)
    pv = (df[df["solver"] == "mirage"]
          .groupby(["tau", "beta"])["solved"].mean().mul(100).round(1)
          .unstack("beta"))
    if pv.empty:
        return
    pv.to_csv(os.path.join(tables_dir, "ablation_full.csv"))
    tex = pv.reset_index().rename(columns={"tau": "$\\tau_0$"})
    tex.columns = [("$\\beta_g=" + str(c) + "$") if isinstance(c, float)
                   else c for c in tex.columns]
    with open(os.path.join(tables_dir, "ablation_full.tex"), "w") as f:
        f.write(latex_table(
            tex, "Full-set ablation: solve rate (\\%) of the "
                 "projector-consensus core over "
                 "$\\tau_0\\times\\beta_{\\mathrm{growth}}$ "
                 "(all manifest instances, all seeds).",
            "tab:ablation", floatfmt="%.1f"))

    stacked = pv.stack()
    if len(stacked):
        best_idx = stacked.idxmax()
        # Pass beta as a string so set_macro's .1f rounding does not collapse
        # grid values like 1.02 -> "1.0" (it must match the ablation table).
        set_macro("\\RevAblBestTau", float(best_idx[0]))
        set_macro("\\RevAblBestBeta", f"{best_idx[1]:g}")
        set_macro("\\RevAblBestRate", float(stacked.max()))
        set_macro("\\RevAblWorstRate", float(stacked.min()))
        set_macro("\\RevAblSpread", float(stacked.max() - stacked.min()))

    both = (df.groupby(["solver", "tau", "beta"])["solved"].mean().mul(100)
            .unstack("solver"))
    if {"mirage", "mirage_regions"} <= set(both.columns):
        delta = (both["mirage_regions"] - both["mirage"]).mean()
        set_macro("\\RevAblRegionsDelta", float(delta))
        both.round(1).to_csv(os.path.join(tables_dir, "ablation_regions.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    im = axes[0].imshow(pv.values, aspect="auto", cmap="viridis")
    axes[0].set_xticks(range(len(pv.columns)), [str(c) for c in pv.columns])
    axes[0].set_yticks(range(len(pv.index)), [str(i) for i in pv.index])
    axes[0].set_xlabel(r"$\beta_{\mathrm{growth}}$")
    axes[0].set_ylabel(r"$\tau_0$")
    axes[0].set_title("solve rate (%)")
    fig.colorbar(im, ax=axes[0])
    for beta in pv.columns:
        axes[1].plot(pv.index, pv[beta], marker="o", label=rf"$\beta_g={beta}$")
    axes[1].set_xlabel(r"$\tau_0$")
    axes[1].set_ylabel("solve rate (%)")
    axes[1].legend(fontsize=9)
    save_fig(fig, figures_dir, "ablation_tau_beta")
    save_fig_alias(figures_dir, "ablation_tau_beta", "ablation_heatmap")


# --------------------------------------------------------------------------- #
def regions(df, tables_dir, figures_dir):
    """Honest region-mechanism study from REAL regions_added counters."""
    g = df[df["solver"] == "mirage_regions"]
    if g.empty:
        return
    fired = g[g["regions_added"] > 0]
    base = df[df["solver"] == "mirage"].set_index("instance")["solved"]
    rows = [{
        "quantity": "instances where regions fired",
        "value": f"{len(fired)} / {len(g)} "
                 f"({100.0 * len(fired) / max(len(g), 1):.1f}\\%)"},
        {"quantity": "mean regions added when fired",
         "value": f"{fired['regions_added'].mean():.2f}" if len(fired) else "--"},
        {"quantity": "solve rate on fired subset (with regions)",
         "value": f"{100.0 * fired['solved'].mean():.1f}\\%" if len(fired) else "--"},
        {"quantity": "solve rate on fired subset (core, no regions)",
         "value": (f"{100.0 * base.reindex(fired['instance']).fillna(False).mean():.1f}\\%"
                   if len(fired) else "--")},
    ]
    with open(os.path.join(tables_dir, "region_summary.tex"), "w") as f:
        f.write(latex_table(pd.DataFrame(rows),
                            "Region materialization on the full sweep "
                            "(true firing counters).", "tab:regions"))
    set_macro("\\RevRegionsFiredPct", 100.0 * len(fired) / max(len(g), 1))
    if len(fired):
        set_macro("\\RevRegionsMeanAdded", float(fired["regions_added"].mean()))
        set_macro("\\RevRegionsFiredSolveRate",
                  100.0 * float(fired["solved"].mean()))
        set_macro("\\RevRegionsFiredBaseRate", 100.0 * float(
            base.reindex(fired["instance"]).fillna(False).mean()))
    fig, ax = plt.subplots(figsize=(6, 3.8))
    upper = int(g["regions_added"].max()) + 2 if len(g) else 2
    ax.hist(g["regions_added"], bins=range(0, upper))
    ax.set_xlabel("regions materialized per instance")
    ax.set_ylabel("instances")
    save_fig(fig, figures_dir, "region_analysis")


# --------------------------------------------------------------------------- #
def memory(rows, tables_dir, figures_dir):
    if not rows:
        return
    df = pd.DataFrame([r for r in rows if r.get("status") == "OK"
                       and r.get("mirage_peak_mb", -1) > 0
                       and r.get("ortools_peak_mb", -1) > 0])
    if len(df) < 5:
        return
    x = np.log10(df["size_mb"].clip(lower=1e-4))
    slopes = {}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for col, label, marker in (("mirage_peak_mb", "MIRAGE-R", "o"),
                               ("ortools_peak_mb", "OR-Tools CP-SAT", "^")):
        y = np.log10(df[col])
        b, a = np.polyfit(x, y, 1)
        resid = y - (a + b * x)
        se = float(np.sqrt(np.sum(resid ** 2) / max(len(x) - 2, 1)
                           / np.sum((x - x.mean()) ** 2)))
        slopes[col] = (b, se)
        ax.scatter(df["size_mb"], df[col], s=14, marker=marker,
                   label=f"{label} (slope {b:.2f}$\\pm${se:.2f})")
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(10 ** xs, 10 ** (a + b * xs), lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("instance JSON size (MB)")
    ax.set_ylabel("peak RSS (MB)")
    ax.legend(fontsize=9)
    save_fig(fig, figures_dir, "memory_scaling")
    save_fig_alias(figures_dir, "memory_scaling", "memory_profile")

    ratio = (df["ortools_peak_mb"] / df["mirage_peak_mb"]).median()
    set_macro("\\RevMemSlopeMirage", slopes["mirage_peak_mb"][0])
    set_macro("\\RevMemSlopeCpsat", slopes["ortools_peak_mb"][0])
    set_macro("\\RevMemMedianRatio", float(ratio))
    set_macro("\\RevMemN", len(df))
    summary = pd.DataFrame([
        {"solver": "MIRAGE-R",
         "median peak (MB)": df["mirage_peak_mb"].median(),
         "fitted log-log slope": slopes["mirage_peak_mb"][0]},
        {"solver": "OR-Tools CP-SAT",
         "median peak (MB)": df["ortools_peak_mb"].median(),
         "fitted log-log slope": slopes["ortools_peak_mb"][0]},
    ])
    note = ("Peak resident set of the whole process tree, sampled at 25\\,ms, "
            f"over n={len(df)} instances. Both measurements include a Python "
            "driver, and CP-SAT's core is C++ so absolute values are not "
            "language-neutral, only trends are interpreted.")
    with open(os.path.join(tables_dir, "memory_summary.tex"), "w") as f:
        f.write(latex_table(summary, "Peak-memory scaling.",
                            "tab:memory", note=note))


# --------------------------------------------------------------------------- #
def hybrid_grid(rows, tables_dir):
    if not rows:
        return
    df = pd.DataFrame([r for r in rows if "warmup" in r and "hint_mode" in r])
    if df.empty:
        return
    df["solved"] = df["status"].isin(SOLVED)
    agg = (df.groupby(["warmup", "hint_mode"])
           .agg(solve_rate=("solved", lambda s: 100.0 * s.mean()),
                med_time=("time", "median"), n=("solved", "size"))
           .reset_index())
    pretty = agg.rename(columns={"warmup": "warm-up epochs",
                                 "hint_mode": "hint mode",
                                 "solve_rate": "solve rate (\\%)",
                                 "med_time": "median time (s)",
                                 "n": "runs"})
    with open(os.path.join(tables_dir, "hybrid_deepdive.tex"), "w") as f:
        f.write(latex_table(
            pretty.round(2),
            "Hybrid grid: warm-up epochs $\\times$ hint mode on the stratified "
            "subset. (0, none) is the pure CP-SAT control through the same "
            "harness.", "tab:hybrid", floatfmt="%.1f"))
    base = agg[(agg["warmup"] == 0) & (agg["hint_mode"] == "none")]
    non = agg[(agg["warmup"] > 0)]
    if len(non):
        b = non.sort_values(["solve_rate", "med_time"],
                            ascending=[False, True]).iloc[0]
        set_macro("\\RevHybBestCell",
                  f"warm-up {int(b['warmup'])}, {b['hint_mode']}")
        set_macro("\\RevHybBestRate", float(b["solve_rate"]))
        set_macro("\\RevHybBestMedTime", float(b["med_time"]))
    if len(base):
        set_macro("\\RevHybCpsatRate", float(base.iloc[0]["solve_rate"]))
        set_macro("\\RevHybCpsatMedTime", float(base.iloc[0]["med_time"]))


# --------------------------------------------------------------------------- #
def theory(rows, figures_dir):
    trajs = [r for r in rows if r.get("running_min_trajectory")]
    if not trajs:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    slopes = []
    for r in trajs[:40]:
        y = np.asarray(r["running_min_trajectory"], dtype=float)
        if len(y) < 3:
            continue
        y0 = y[0] if y[0] > 0 else 1.0
        t = np.arange(1, len(y) + 1) * r.get("decode_freq", 5)
        ax.plot(t, np.maximum(y / y0, 1e-3), alpha=0.35, lw=0.8)
        mask = (y / y0 > 1e-3) & (t > 1)
        if mask.sum() > 5:
            b, _ = np.polyfit(np.log(t[mask]), np.log(y[mask] / y0), 1)
            slopes.append(b)
    tmax = max(len(r["running_min_trajectory"]) for r in trajs) * 5
    tref = np.logspace(0.3, np.log10(max(tmax, 10)), 40)
    ax.plot(tref, tref ** (-0.5), "k--", lw=2,
            label=r"$T^{-1/2}$ reference (Thm.~1)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("epoch $T$")
    ax.set_ylabel("normalized best violation count")
    ax.legend()
    save_fig(fig, figures_dir, "convergence")
    if slopes:
        set_macro("\\RevConvergenceSlope", float(np.median(slopes)))


# --------------------------------------------------------------------------- #
def gpu(rows, figures_dir):
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    backends = sorted({b for r in rows for b in r["sec_per_epoch"]})
    lbl = {"numpy_loop": "per-projector loop (numpy)",
           "torch_cpu": "batched (torch, CPU)",
           "torch_cuda": "batched (torch, CUDA fp32)"}
    for b in backends:
        xs = [r["n_vars"] for r in rows if b in r["sec_per_epoch"]]
        ys = [r["sec_per_epoch"][b] for r in rows if b in r["sec_per_epoch"]]
        ax.plot(xs, ys, marker="o", label=lbl.get(b, b))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("variables $n$ (constraints $=2n$)")
    ax.set_ylabel("wall-clock per epoch (s)")
    ax.legend(fontsize=9)
    save_fig(fig, figures_dir, "gpu_throughput")
    largest = max(rows, key=lambda r: r["n_vars"])
    spe = largest["sec_per_epoch"]
    set_macro("\\RevGpuLargestN", largest["n_vars"])
    if "torch_cuda" in spe and "torch_cpu" in spe:
        set_macro("\\RevGpuSpeedupCuda", spe["torch_cpu"] / spe["torch_cuda"])
    loops = [r for r in rows if "numpy_loop" in r["sec_per_epoch"]
             and "torch_cpu" in r["sec_per_epoch"]]
    if loops:
        big = max(loops, key=lambda r: r["n_vars"])
        set_macro("\\RevGpuSpeedupVec",
                  big["sec_per_epoch"]["numpy_loop"]
                  / big["sec_per_epoch"]["torch_cpu"])


# --------------------------------------------------------------------------- #
def wilcoxon(df, tables_dir):
    try:
        from scipy.stats import wilcoxon as _w
    except ImportError:
        return
    piv = df.pivot_table(index="instance", columns="solver", values="par2")
    if "mirage" not in piv.columns:
        return
    rows, pvals = [], []
    for other in [c for c in piv.columns if c != "mirage"]:
        pair = piv[["mirage", other]].dropna()
        if len(pair) < 10 or (pair["mirage"] == pair[other]).all():
            continue
        stat, p = _w(pair["mirage"], pair[other])
        rows.append({"comparison": f"MIRAGE-R vs {PRETTY.get(other, other)}",
                     "n": len(pair), "W": stat, "p": p})
        pvals.append(p)
        if other == "mirage_regions":
            set_macro("\\RevRegionsWilcoxonP", f"{p:.2f}")
    if not rows:
        return
    order = np.argsort(pvals)
    m = len(pvals)
    for rank, idx in enumerate(order):
        rows[idx]["p (Holm)"] = min(pvals[idx] * (m - rank), 1.0)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(tables_dir, "wilcoxon.csv"), index=False)
    with open(os.path.join(tables_dir, "wilcoxon.tex"), "w") as f:
        f.write(latex_table(tab.round(4),
                            "Wilcoxon signed-rank tests on per-instance "
                            "PAR2 (Holm-corrected).", "tab:wilcoxon",
                            floatfmt="%.4f"))


# --------------------------------------------------------------------------- #
def protocol_table(args, tables_dir):
    """P3.4: the single protocol appendix table."""
    import platform
    versions = {}
    for mod in ("numpy", "pandas", "ortools", "cpmpy", "torch", "psutil"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = "n/a"
    rows = [
        ("benchmark instances", str(args.expected_instances)),
        ("seeds / majority rule", f"{args.seeds} seeds, an instance counts as "
                                  "solved iff $>$50\\% of seeds solve it"),
        ("per-instance time limit", f"{args.timeout:.0f}\\,s wall-clock "
                                    "(subprocess hard-kill at limit + grace)"),
        ("PAR2", "unsolved run scored $2\\times$ limit"),
        ("MIRAGE defaults", "$\\tau_0=2.0$, $\\tau$-decay $0.95$, "
                            "$\\beta_0=1$, $\\beta_{\\mathrm{growth}}=1.05$, "
                            "$\\beta_{\\max}=5$"),
        ("annealing closed form", "$\\tau_t=\\tau_0\\cdot0.95^{t}$,\\ \\ "
                                  "$\\beta_t=\\min(5,\\,1.05^{t})$"),
        ("$\\varepsilon$-clip", "$p_i(a)\\ge\\varepsilon=10^{-12}$ "
                                "(consensus and polarization)"),
        ("stall window $k$", "10 decode checks without strict improvement"),
        ("tuple cap (regions)", "200{,}000"),
        ("decode frequency", "every 5 epochs"),
        ("hybrid warm-up budget", "$\\le20\\%$ of the total limit"),
        ("RUN-CSP protocol", "transductive per-instance unsupervised training, "
                             "binary-table fragment only, restarts to deadline"),
        ("hardware", platform.platform().replace("_", "\\_")
                     + f", {os.cpu_count()} logical CPUs"),
        ("software", ", ".join(f"{k} {v}" for k, v in versions.items())),
        ("parallelism", "8 shard jobs, per-task solver threads = 1 "
                        "(4 for the main-sweep CP-SAT rows)"),
    ]
    tab = pd.DataFrame(rows, columns=["item", "value"])
    with open(os.path.join(tables_dir, "protocol.tex"), "w") as f:
        f.write(latex_table(tab, "Complete experimental protocol.",
                            "tab:protocol"))
    set_macro("\\RevSeeds", args.seeds)
    set_macro("\\RevTimeout", int(args.timeout))


def write_macros(tables_dir):
    _default_macros()
    with open(os.path.join(tables_dir, "numbers.tex"), "w") as f:
        f.write("% AUTO-GENERATED by experiments/validation/aggregate.py -- "
                "do not edit\n")
        for k in sorted(MACROS):
            name = k.lstrip("\\")
            f.write(f"\\providecommand{{\\{name}}}{{{MACROS[k]}}}\n")
            f.write(f"\\renewcommand{{\\{name}}}{{{MACROS[k]}}}\n")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="results/raw")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--expected-instances", type=int, default=736)
    ap.add_argument("--features", default="data/frozen/instance_features.csv")
    ap.add_argument("--tables-dir", default="paper/ijoc/tables")
    ap.add_argument("--figures-dir", default="paper/ijoc/figures")
    args = ap.parse_args()

    os.makedirs(args.tables_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    feats = None
    if os.path.exists(args.features):
        feats = pd.read_csv(args.features)
        if "instance" in feats.columns:
            feats["instance"] = feats["instance"].map(canon_instance)

    raw = args.raw_dir
    # REVISION LOGS ONLY: the pre-revision "weekend" logs contained a
    # fabricated regions_added field and broken baseline adapters; they are
    # never mixed into the paper's numbers.
    sweep_rows = read_jsonl([os.path.join(raw, "revision_sweep", "*.jsonl")])
    seen, dedup = set(), []
    for r in sweep_rows:
        k = (r.get("solver"), r.get("instance"), r.get("seed"))
        if k in seen:
            continue
        seen.add(k)
        dedup.append(r)
    sweep_rows = dedup
    print(f"[aggregate] sweep rows: {len(sweep_rows)}")

    if sweep_rows:
        df = aggregate_sweep(sweep_rows, args.timeout)
        main_table(df, feats, args.timeout, args.tables_dir)
        strata_tables(df, feats, args.tables_dir)
        solve_rate_density_fig(df, feats, args.figures_dir)
        cactus_and_profile(df, args.timeout, args.figures_dir)
        soundness(sweep_rows, args.tables_dir)
        regions(df, args.tables_dir, args.figures_dir)
        wilcoxon(df, args.tables_dir)

    ablation(read_jsonl([os.path.join(raw, "revision_ablation", "*.jsonl")]),
             args.tables_dir, args.figures_dir)
    memory(read_jsonl([os.path.join(raw, "revision_memory", "*.jsonl")]),
           args.tables_dir, args.figures_dir)
    hybrid_grid(read_jsonl([os.path.join(raw, "revision_hybrid_grid",
                                         "*.jsonl")]), args.tables_dir)
    theory(read_jsonl([os.path.join(raw, "revision_theory", "*.jsonl")]),
           args.figures_dir)
    gpu(read_jsonl([os.path.join(raw, "revision_gpu", "*.jsonl")]),
        args.figures_dir)
    protocol_table(args, args.tables_dir)
    write_macros(args.tables_dir)
    print(f"[aggregate] DONE -> {args.tables_dir}, {args.figures_dir}")


if __name__ == "__main__":
    main()
