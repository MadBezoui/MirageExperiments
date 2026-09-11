"""Regenerate every manuscript figure with a single, publication-quality style.

This is a self-contained re-styling / re-plotting pass over the figures used in
paper/ijoc/ms.tex. It reads the same revision raw logs that experiments.
validation.aggregate consumes (so the underlying numbers are unchanged) and the
paper's own strata_density table, and writes both .pdf (for LaTeX) and .png
(for quick inspection) into paper/ijoc/figures/.

Design goals (uniform across all panels):
  * Computer-Modern-like serif + cm mathtext, so figures match the LaTeX body.
  * One shared solver palette and marker set.
  * Light dotted grids, de-emphasised spines, no in-figure titles (all
    descriptive text lives in the LaTeX caption).
  * Fixed axis scales chosen for readability (log where the data span decades).

Usage:
  PYTHONPATH=. python experiments/make_paper_figures.py \
      --raw-dir results/raw --tables-dir paper/ijoc/tables \
      --figures-dir paper/ijoc/figures --timeout 300
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

# --------------------------------------------------------------------------- #
# Global style
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.serif": ["CMU Serif", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.0,
    "axes.edgecolor": "#444444",
    "axes.grid": True,
    "grid.color": "#B9BFC9",
    "grid.linewidth": 0.8,
    "grid.linestyle": ":",
    "grid.alpha": 0.7,
    "axes.axisbelow": True,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#CCCCCC",
    "legend.fancybox": False,
    "lines.linewidth": 2.5,
    "lines.markersize": 7.0,
    "figure.autolayout": False,
})

# A single, colour-blind-aware palette shared by every figure.
C = {
    "mirage":         "#1F6FB2",   # blue
    "mirage_regions": "#12A19A",   # teal
    "hybrid":         "#E8833A",   # orange
    "ortools":        "#7A4FA3",   # purple
    "choco":          "#C0392B",   # red
    "runcsp":         "#7F8C8D",   # grey
    "ref":            "#2B2B2B",   # near-black reference lines
    "accent":         "#D81B60",   # highlight
}
MARK = {
    "mirage": "o", "mirage_regions": "s", "hybrid": "D",
    "ortools": "^", "choco": "v", "runcsp": "P",
}
PRETTY = {
    "mirage": "MIRAGE-R", "mirage_regions": "MIRAGE-R+regions",
    "hybrid": r"MIRAGE-R$\to$CP-SAT", "ortools": "OR-Tools CP-SAT",
    "choco": "Choco", "runcsp": "RUN-CSP",
}
SOLVED = {"SAT_VERIFIED", "UNSAT"}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def canon(name):
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
                        r["instance"] = canon(r["instance"])
                    rows.append(r)
    return rows


def family(inst):
    base = str(inst).replace(".json", "")
    return base.split("-")[0] if "-" in base else base


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(centre - half, 0.0), min(centre + half, 1.0)


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3.5, width=0.8, colors="#444444")
    return ax


def save(fig, figdir, name, aliases=()):
    os.makedirs(figdir, exist_ok=True)
    for stem in (name, *aliases):
        fig.savefig(os.path.join(figdir, stem + ".pdf"), bbox_inches="tight",
                    pad_inches=0.02)
        fig.savefig(os.path.join(figdir, stem + ".png"), bbox_inches="tight",
                    pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote {name}" + (f" (+{','.join(aliases)})" if aliases else ""))


def aggregate_sweep(rows, timeout):
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
        is_solved = sum(solved) > len(solved) / 2
        recs.append({
            "solver": solver, "instance": inst,
            "solved": is_solved,
            "median_time": float(np.median(times)),
            "par2": float(np.mean([t if sv else 2 * timeout
                                   for t, sv in zip(times, solved)])),
            "regions_added": int(np.median([x.get("regions_added", 0) or 0
                                            for x in rs])),
            "sat": is_solved and any(s == "SAT_VERIFIED" for s in statuses),
        })
    import pandas as pd
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------- #
# 1. Loss landscape (analytical illustration of fractional stationarity)
# --------------------------------------------------------------------------- #
def fig_loss_landscape(figdir):
    g = np.linspace(1e-3, 1 - 1e-3, 400)
    PX, PY = np.meshgrid(g, g)
    # XOR / (x != y): satisfying tuples (0,1),(1,0). Soft loss = prob of a
    # violating joint assignment under the product marginal.
    p_viol = PX * PY + (1 - PX) * (1 - PY)          # Pr[x=y]
    soft = p_viol                                    # panel (a): raw soft loss
    tau = 0.35
    # panel (b): temperature-regularised free energy F_tau = loss - tau*H
    H = -(PX * np.log(PX) + (1 - PX) * np.log(1 - PX)
          + PY * np.log(PY) + (1 - PY) * np.log(1 - PY))
    free = p_viol - tau * H

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))

    ax = axes[0]
    cf = ax.contourf(PX, PY, soft, levels=18, cmap="viridis")
    ax.contour(PX, PY, soft, levels=8, colors="white", linewidths=0.35,
               alpha=0.55)
    # gradient-zero saddle at (1/2,1/2)
    ax.plot([0.5], [0.5], "o", ms=11, color=C["accent"], mec="white",
            mew=1.3, zorder=5, label=r"fractional saddle ($\nabla=0$)")
    # satisfying integral minima at (0,1) and (1,0)
    ax.plot([0.03, 0.97], [0.97, 0.03], "*", ms=18, color="white",
            mec="#222222", mew=1.1, zorder=6,
            label="satisfying integral minima")
    ax.set_xlabel(r"$p_x=\Pr[x=1]$")
    ax.set_ylabel(r"$p_y=\Pr[y=1]$")
    ax.set_title(r"(a) soft loss of $x\neq y$")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
                    ncol=1, fontsize=8.5, handletextpad=0.4)
    leg.get_frame().set_linewidth(0.6)
    cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("soft loss", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    ax = axes[1]
    cf = ax.contourf(PX, PY, free, levels=18, cmap="magma")
    ax.contour(PX, PY, free, levels=8, colors="white", linewidths=0.35,
               alpha=0.4)
    ax.plot([0.5], [0.5], "o", ms=11, color="#39D0C6", mec="#0b3d3a",
            mew=1.2, zorder=5,
            label=r"minimizer at finite $\tau$: $(\frac{1}{2},\frac{1}{2})$")
    # annealing arrow: optimum migrates toward a vertex as tau -> 0
    ax.annotate("", xy=(0.94, 0.06), xytext=(0.54, 0.46),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=2.0,
                                mutation_scale=16))
    ax.text(0.72, 0.30, r"anneal $\tau\!\to\!0$", color="white", fontsize=8.5,
            rotation=-45, ha="center", va="center")
    ax.set_xlabel(r"$p_x$")
    ax.set_ylabel(r"$p_y$")
    ax.set_title(r"(b) free energy $F_\tau$")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
                    fontsize=8.5, handletextpad=0.4)
    leg.get_frame().set_linewidth(0.6)
    cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$F_\tau$", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    fig.subplots_adjust(wspace=0.42, bottom=0.24)
    save(fig, figdir, "loss_landscape")


# --------------------------------------------------------------------------- #
# 2. Solve rate vs density (from the paper's own strata_density table)
# --------------------------------------------------------------------------- #
def _parse_strata_density(tables_dir):
    path = os.path.join(tables_dir, "strata_density.tex")
    if not os.path.exists(path):
        return None
    header, rows = None, []
    for line in open(path, encoding="utf-8"):
        if "&" not in line or "\\midrule" in line:
            continue
        cells = [c.strip() for c in line.split("\\\\")[0].split("&")]
        if any("constraint density" in c for c in cells):
            header = cells
            continue
        if header and len(cells) == len(header):
            rows.append(cells)
    if not header or not rows:
        return None
    return header, rows


def fig_solve_rate_density(figdir, tables_dir):
    parsed = _parse_strata_density(tables_dir)
    if parsed is None:
        print("  [skip] solve_rate_density: strata_density.tex not found")
        return
    header, rows = parsed
    # locate columns
    def col(substr):
        for j, h in enumerate(header):
            if substr in h:
                return j
        return None
    j_n = col("\\#inst")
    cols = {
        "mirage": col("MIRAGE-R &") if col("MIRAGE-R &") is not None
        else [j for j, h in enumerate(header) if h.strip() == "MIRAGE-R"][0],
        "mirage_regions": [j for j, h in enumerate(header)
                           if "regions" in h][0],
        "ortools": [j for j, h in enumerate(header) if "OR-Tools" in h][0],
    }
    labels = [r[0].replace("$\\le$", "≤").replace("$>$", ">")
              .replace("--", "–") for r in rows]
    ns = [int(r[j_n]) for r in rows]
    x = np.arange(len(rows))
    off = {"mirage": -0.12, "mirage_regions": 0.0, "ortools": 0.12}

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    style_ax(ax)
    for s in ("ortools", "mirage_regions", "mirage"):
        rates = np.array([float(rows[i][cols[s]]) / 100.0
                          for i in range(len(rows))])
        lo, hi = [], []
        for i in range(len(rows)):
            k = int(round(rates[i] * ns[i]))
            _, l, h = wilson_ci(k, ns[i])
            lo.append(max(rates[i] - l, 0.0)); hi.append(max(h - rates[i], 0.0))
        ax.errorbar(x + off[s], rates, yerr=[lo, hi], marker=MARK[s],
                    color=C[s], capsize=3, capthick=1.0, elinewidth=1.0,
                    lw=1.6, label=PRETTY[s], markeredgecolor="white",
                    markeredgewidth=0.6)
    for xi, n in zip(x, ns):
        ax.annotate(f"n={n}", (xi, 1.045), ha="center", va="bottom",
                    fontsize=8, color="#666666")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel(r"constraint density $|C|/|X|$")
    ax.set_ylabel("solve rate")
    ax.set_ylim(-0.03, 1.12)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.55))
    save(fig, figdir, "solve_rate_density", aliases=("failure_density",))


# --------------------------------------------------------------------------- #
# 3. Ablation heatmap + per-beta profiles
# --------------------------------------------------------------------------- #
def fig_ablation(figdir, rows):
    import pandas as pd
    df = pd.DataFrame([r for r in rows if "tau" in r and "beta" in r and r.get("status") != "SKIPPED_SIZE"])
    if df.empty:
        print("  [skip] ablation: no rows")
        return
    df["solved"] = df["status"].isin(SOLVED)
    pv = (df[df["solver"] == "mirage"]
          .groupby(["tau", "beta"])["solved"].mean().mul(100).round(1)
          .unstack("beta"))
    if pv.empty:
        print("  [skip] ablation: empty pivot")
        return
    taus = list(pv.index)
    betas = list(pv.columns)
    vals = pv.values

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9),
                             gridspec_kw={"width_ratios": [1.05, 1.0]})
    ax = axes[0]
    im = ax.imshow(vals, aspect="auto", cmap="viridis", origin="upper")
    ax.set_xticks(range(len(betas)), [f"{b:g}" for b in betas])
    ax.set_yticks(range(len(taus)), [f"{t:g}" for t in taus])
    ax.set_xlabel(r"$\beta_{\mathrm{growth}}$")
    ax.set_ylabel(r"$\tau_0$")
    # annotate every cell; highlight the best
    bi, bj = np.unravel_index(np.nanargmax(vals), vals.shape)
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    for i in range(len(taus)):
        for j in range(len(betas)):
            v = vals[i, j]
            if np.isnan(v):
                continue
            txt = "#FFFFFF" if (v - vmin) / (vmax - vmin + 1e-9) < 0.5 else "#111111"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", color=txt,
                    fontsize=9.5, fontweight="bold" if (i, j) == (bi, bj)
                    else "normal")
    ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False,
                               edgecolor=C["accent"], lw=2.2))
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("solve rate (%)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax.set_title("(a) solve-rate grid")

    ax = axes[1]
    style_ax(ax)
    cmap = plt.cm.plasma(np.linspace(0.1, 0.8, len(betas)))
    for k, b in enumerate(betas):
        ax.plot(taus, pv[b].values, marker="o", color=cmap[k],
                markeredgecolor="white", markeredgewidth=0.6,
                label=rf"$\beta_g={b:g}$")
    ax.set_xlabel(r"$\tau_0$")
    ax.set_ylabel("solve rate (%)")
    ax.set_title(r"(b) profiles over $\tau_0$")
    ax.legend(title=None)
    fig.subplots_adjust(wspace=0.32)
    save(fig, figdir, "ablation_tau_beta", aliases=("ablation_heatmap",))


# --------------------------------------------------------------------------- #
# 4. Region materialization histogram
# --------------------------------------------------------------------------- #
def fig_regions(figdir, df):
    g = df[df["solver"] == "mirage_regions"]
    if g.empty:
        print("  [skip] regions: no mirage_regions rows")
        return
    fired = g[g["regions_added"] > 0]["regions_added"].astype(int)
    frac = 100.0 * len(fired) / max(len(g), 1)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    style_ax(ax)
    if len(fired):
        cap = int(np.percentile(fired, 90))
        cap = int(np.clip(cap, 20, 40))
        bins = np.arange(0.5, cap + 1.5, 1)
        n_over = int((fired > cap).sum())
        ax.hist(fired.clip(upper=cap), bins=bins, color=C["mirage_regions"],
                edgecolor="white", linewidth=0.5, alpha=0.9)
        ax.set_xlim(0.5, cap + 0.5)
        med = int(np.median(fired))
        ax.axvline(med, color=C["accent"], lw=1.6, ls="--",
                   label=f"median = {med}")
        if n_over:
            ax.annotate(f"$\\geq{cap}$: {n_over} instances\n(tail to "
                        f"{int(fired.max())})",
                        xy=(cap, n_over), xytext=(cap * 0.62, n_over * 0.9),
                        fontsize=8.5, color="#555555", ha="right", va="center",
                        arrowprops=dict(arrowstyle="->", color="#888888",
                                        lw=0.9))
        ax.legend(loc="center right")
    ax.set_yscale("log")
    ax.set_xlabel("regions materialized per instance (when regions fire)")
    ax.set_ylabel("instances (log scale)")
    ax.annotate(f"regions fired on {frac:.1f}% of instances\n"
                f"({len(fired)} of {len(g)})",
                xy=(0.97, 0.93), xycoords="axes fraction", ha="right",
                va="top", fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.35", fc="#F4F6F9",
                          ec="#C9CDD3", lw=0.8))
    save(fig, figdir, "region_analysis")


# --------------------------------------------------------------------------- #
# 5. Peak-memory scaling
# --------------------------------------------------------------------------- #
def fig_memory(figdir, rows):
    import pandas as pd
    df = pd.DataFrame([r for r in rows if r.get("status") == "OK"
                       and r.get("mirage_peak_mb", -1) > 0
                       and r.get("ortools_peak_mb", -1) > 0])
    if len(df) < 5:
        print("  [skip] memory: <5 rows")
        return
    x = np.log10(df["size_mb"].clip(lower=1e-4))
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    style_ax(ax)
    slopes = {}
    for col, key in (("mirage_peak_mb", "mirage"),
                     ("ortools_peak_mb", "ortools")):
        y = np.log10(df[col])
        b, a = np.polyfit(x, y, 1)
        resid = y - (a + b * x)
        se = float(np.sqrt(np.sum(resid ** 2) / max(len(x) - 2, 1)
                           / np.sum((x - x.mean()) ** 2)))
        slopes[key] = b
        ax.scatter(df["size_mb"], df[col], s=20, marker=MARK[key],
                   color=C[key], alpha=0.65, edgecolor="white",
                   linewidth=0.3, zorder=3,
                   label=f"{PRETTY[key]}  (slope {b:.2f}$\\pm${se:.2f})")
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(10 ** xs, 10 ** (a + b * xs), lw=1.8, color=C[key], zorder=4)
    ratio = (df["ortools_peak_mb"] / df["mirage_peak_mb"]).median()
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("instance JSON size (MB)")
    ax.set_ylabel("peak RSS (MB)")
    ax.annotate(f"median peak ratio\n{ratio:.1f}$\\times$ in MIRAGE-R's favour",
                xy=(0.03, 0.95), xycoords="axes fraction", va="top",
                fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.35", fc="#F4F6F9",
                          ec="#C9CDD3", lw=0.8))
    ax.legend(loc="lower right")
    save(fig, figdir, "memory_scaling", aliases=("memory_profile",))


# --------------------------------------------------------------------------- #
# 6. GPU / batched throughput
# --------------------------------------------------------------------------- #
def fig_gpu(figdir, rows):
    if not rows:
        print("  [skip] gpu: no rows")
        return
    lbl = {"numpy_loop": "per-projector loop (NumPy)",
           "torch_cpu": "batched (Torch, CPU)",
           "torch_cuda": "batched (Torch, CUDA fp32)"}
    col = {"numpy_loop": C["choco"], "torch_cpu": C["hybrid"],
           "torch_cuda": C["mirage_regions"]}
    mk = {"numpy_loop": "v", "torch_cpu": "D", "torch_cuda": "o"}
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    style_ax(ax)
    backends = ["numpy_loop", "torch_cpu", "torch_cuda"]
    for b in backends:
        pts = [(r["n_vars"], r["sec_per_epoch"][b]) for r in rows
               if b in r.get("sec_per_epoch", {})]
        if not pts:
            continue
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker=mk[b], color=col[b], markeredgecolor="white",
                markeredgewidth=0.6, label=lbl.get(b, b))
    # speed-up annotation at the largest common size
    big = max(rows, key=lambda r: r["n_vars"])
    spe = big["sec_per_epoch"]
    ann = []
    loops = [r for r in rows if "numpy_loop" in r["sec_per_epoch"]
             and "torch_cpu" in r["sec_per_epoch"]]
    if loops:
        bl = max(loops, key=lambda r: r["n_vars"])
        ann.append(f"vectorization at $n={bl['n_vars']}$: "
                   f"{bl['sec_per_epoch']['numpy_loop']/bl['sec_per_epoch']['torch_cpu']:.0f}$\\times$")
    if "torch_cuda" in spe and "torch_cpu" in spe:
        ann.append(f"CUDA over CPU at $n={big['n_vars']}$: "
                   f"{spe['torch_cpu']/spe['torch_cuda']:.0f}$\\times$")
        ann.append("(different sizes; do not multiply)")
    if ann:
        ax.annotate("\n".join(ann),
                    xy=(0.03, 0.05), xycoords="axes fraction", va="bottom",
                    fontsize=9.5,
                    bbox=dict(boxstyle="round,pad=0.35", fc="#F4F6F9",
                              ec="#C9CDD3", lw=0.8))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"variables $n$ (constraints $=2n$)")
    ax.set_ylabel("wall-clock per epoch (s)")
    ax.legend(loc="upper left")
    save(fig, figdir, "gpu_throughput")


# --------------------------------------------------------------------------- #
# 7. REMOVED -- the former fig_failure_entropy() drew entropy values from
#    rng.normal(0.10, 0.05) and rng.normal(0.60, 0.11); only the two sample
#    counts came from the sweep. The solver does not log marginal entropy,
#    so the figure was synthetic and has been withdrawn from the paper along
#    with the terminal-regime claims that cited it. Do not reinstate it
#    without instrumenting the solver and re-running the sweep.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# 8. Convergence trajectories (running-min violation), coloured by outcome.
#    No theoretical T^{-1/2} envelope: the fixed-temperature core has no such
#    global rate; most runs plateau at a positive level (non-solution fixed
#    point).
# --------------------------------------------------------------------------- #
def fig_convergence(figdir, rows):
    trajs = [r for r in rows if r.get("running_min_trajectory")]
    if not trajs:
        print("  [skip] convergence: no trajectories")
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    style_ax(ax)
    slopes = []
    for r in trajs[:60]:
        y = np.asarray(r["running_min_trajectory"], dtype=float)
        if len(y) < 3:
            continue
        y0 = y[0] if y[0] > 0 else 1.0
        t = np.arange(1, len(y) + 1) * r.get("decode_freq", 5)
        solved = r.get("status") in ("SAT_VERIFIED", "UNSAT") \
            or r.get("final_violations", 1) == 0
        color = C["mirage"] if solved else C["choco"]
        ax.plot(t, np.maximum(y / y0, 1e-3), color=color, alpha=0.28, lw=0.9,
                zorder=2)
        mask = (y / y0 > 1e-3) & (t > 1)
        if mask.sum() > 5:
            b, _ = np.polyfit(np.log(t[mask]), np.log(y[mask] / y0), 1)
            slopes.append(b)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=C["mirage"], lw=1.6, alpha=0.7,
               label="run reaching feasibility (violations $\\to 0$)"),
        Line2D([0], [0], color=C["choco"], lw=1.6, alpha=0.7,
               label="timed-out run without decoded improvement"),
    ]
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"epoch $T$")
    ax.set_ylabel("normalized running-min violation count")
    if slopes:
        ax.annotate(f"median fitted log-log slope: {np.median(slopes):.3f}\n"
                    "(essentially flat, no global decay)",
                    xy=(0.03, 0.06), xycoords="axes fraction", va="bottom",
                    fontsize=9.5,
                    bbox=dict(boxstyle="round,pad=0.35", fc="#F4F6F9",
                              ec="#C9CDD3", lw=0.8))
    ax.legend(handles=handles, loc="upper right")
    save(fig, figdir, "convergence")


# --------------------------------------------------------------------------- #
# 9 + 10. Cactus and Dolan--More performance profile
# --------------------------------------------------------------------------- #
def fig_cactus_and_profile(figdir, df, timeout):
    order = ["choco", "ortools", "hybrid", "mirage_regions", "mirage", "runcsp"]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    style_ax(ax)
    for s in order:
        g = df[df["solver"] == s]
        t = np.sort(g.loc[g["solved"], "median_time"].values)
        if len(t) == 0:
            continue
        ax.plot(np.arange(1, len(t) + 1), np.maximum(t, 1e-3), color=C[s],
                lw=2.0, label=PRETTY[s])
    ax.set_yscale("log")
    ax.set_xlabel("instances solved")
    ax.set_ylabel("time (s)")
    ax.legend(loc="upper left", ncol=2)
    save(fig, figdir, "cactus_plot")

    piv = df.pivot_table(index="instance", columns="solver", values="par2")
    best = piv.min(axis=1)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    style_ax(ax)
    taus = np.logspace(0, np.log10(max(2 * timeout, 10)), 200)
    for s in order:
        if s not in piv.columns:
            continue
        ratios = (piv[s] / best).dropna()
        ys = [(ratios <= t).mean() for t in taus]
        ax.plot(taus, ys, color=C[s], lw=2.0, label=PRETTY[s])
    ax.set_xscale("log")
    ax.set_xlabel(r"performance ratio $\tau$")
    ax.set_ylabel(r"fraction of instances within $\tau$ of best")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", ncol=2)
    save(fig, figdir, "performance_profile")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="results/raw")
    ap.add_argument("--tables-dir", default="paper/ijoc/tables")
    ap.add_argument("--figures-dir", default="paper/ijoc/figures")
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()
    raw, fd = args.raw_dir, args.figures_dir
    os.makedirs(fd, exist_ok=True)

    print("[figures] loss landscape")
    fig_loss_landscape(fd)
    print("[figures] solve rate vs density")
    fig_solve_rate_density(fd, args.tables_dir)

    sweep = read_jsonl([os.path.join(raw, "revision_sweep", "*.jsonl")])
    # dedup (solver, instance, seed)
    seen, dd = set(), []
    for r in sweep:
        k = (r.get("solver"), r.get("instance"), r.get("seed"))
        if k in seen:
            continue
        seen.add(k); dd.append(r)
    sweep = dd
    print(f"[figures] sweep rows: {len(sweep)}")
    df = aggregate_sweep(sweep, args.timeout)

    print("[figures] ablation")
    fig_ablation(fd, read_jsonl([os.path.join(raw, "revision_ablation",
                                              "*.jsonl")]))
    print("[figures] regions")
    fig_regions(fd, df)
    print("[figures] memory")
    fig_memory(fd, read_jsonl([os.path.join(raw, "revision_memory",
                                            "*.jsonl")]))
    print("[figures] gpu")
    fig_gpu(fd, read_jsonl([os.path.join(raw, "revision_gpu", "*.jsonl")]))
    print("[figures] convergence")
    fig_convergence(fd, read_jsonl([os.path.join(raw, "revision_theory",
                                                 "*.jsonl")]))
    print("[figures] cactus + profile")
    fig_cactus_and_profile(fd, df, args.timeout)
    print("[figures] DONE ->", fd)


if __name__ == "__main__":
    main()
