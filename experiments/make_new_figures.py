"""Figures for the new theoretical and diagnostic sections.

  fig:critical    critical ratio gamma* = 1/mu_max(L) across disequality graphs
  fig:fpdiag      decoded plateau versus continuous-state diagnostics

Usage:
    PYTHONPATH=. python -m experiments.make_new_figures
"""
from __future__ import annotations

import argparse
import collections
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.theory_multifactor import (cycle_graph, is_bipartite,
                                            laplacian)

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 200, "savefig.bbox": "tight",
})


def fig_critical(figdir):
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))

    # (a) odd vs even cycles
    ns = np.arange(3, 40)
    gam = []
    for n in ns:
        L, _, _ = laplacian(*cycle_graph(int(n)))
        gam.append(1.0 / float(np.max(np.linalg.eigvals(L).real)))
    gam = np.array(gam)
    odd, even = ns % 2 == 1, ns % 2 == 0
    ax[0].plot(ns[even], gam[even], "o-", ms=3, lw=1,
               label="even cycle (bipartite)", color="#1f77b4")
    ax[0].plot(ns[odd], gam[odd], "s-", ms=3, lw=1,
               label="odd cycle (non-bipartite)", color="#d62728")
    ax[0].axhline(0.5, ls="--", lw=.8, color="k")
    ax[0].annotate(r"$\gamma=1/2$ (default $\tau_0=2,\beta_0=1$)",
                   xy=(24, 0.5), xytext=(15, 0.545), fontsize=7,
                   arrowprops=dict(arrowstyle="->", lw=.6))
    ax[0].set_xlabel("cycle length $n$")
    ax[0].set_ylabel(r"critical ratio $\gamma^\star=1/\mu_{\max}(L)$")
    ax[0].set_title("(a) critical ratio by cycle parity")
    ax[0].legend(frameon=False, loc="upper right")

    # (b) contraction factor at the default ratio
    fac = 0.5 * np.array([1.0 / g for g in gam])   # gamma * mu_max at gamma=1/2
    ax[1].plot(ns[even], fac[even], "o-", ms=3, lw=1, color="#1f77b4",
               label="even cycle")
    ax[1].plot(ns[odd], fac[odd], "s-", ms=3, lw=1, color="#d62728",
               label="odd cycle")
    ax[1].axhline(1.0, ls="--", lw=.8, color="k")
    ax[1].set_xlabel("cycle length $n$")
    ax[1].set_ylabel(r"one-epoch factor $\gamma\,\mu_{\max}$ at $\gamma=1/2$")
    ax[1].set_title("(b) behavior at the default ratio")
    ax[1].set_ylim(0.7, 1.05)
    ax[1].legend(frameon=False, loc="lower right")
    ax[1].text(20, 0.80,
               "below 1: strict contraction\nto the uniform fractional point",
               fontsize=7, ha="center",
               bbox=dict(fc="white", ec="0.7", lw=.5, boxstyle="round,pad=0.3"))

    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"critical_ratio.{ext}"))
    plt.close(fig)
    print("  wrote critical_ratio.pdf")


def fig_diagnostics(jsonl, figdir):
    recs = [json.loads(l) for l in open(jsonl) if l.strip()]
    order = []
    for r in recs:
        if r["config"] not in order:
            order.append(r["config"])
    short = {
        "bipartite, annealed default": "bipartite, annealed",
        "non-bipartite, annealed default": "non-bipartite, annealed",
        "random tables, annealed default": "tables, annealed",
        r"non-bipartite, frozen $\gamma=1/2$": r"non-bipartite, $\gamma=1/2$",
        r"non-bipartite, frozen $\gamma=1/4$": r"non-bipartite, $\gamma=1/4$",
        r"bipartite, frozen $\gamma=1/2$": r"bipartite, $\gamma=1/2$",
        r"random tables, frozen $\gamma=1/4$": r"tables, $\gamma=1/4$",
    }
    labels = [short.get(c, c) for c in order]
    x = np.arange(len(order))

    def col(key):
        return [np.median([r[key] for r in recs if r["config"] == c])
                for c in order]

    fig, ax = plt.subplots(1, 3, figsize=(7.4, 3.4))

    ax[0].bar(x, col("running_min_flat_fraction"), color="#7f7f7f", width=.65)
    ax[0].set_ylim(0, 1.05)
    ax[0].set_ylabel("flat fraction of running minimum")
    ax[0].set_title("(a) what the solver observes")

    ent = col("entropy")
    gap = col("gap_mean")
    ax[1].bar(x - .18, ent, width=.36, label=r"$\bar H$", color="#1f77b4")
    ax[1].bar(x + .18, gap, width=.36, label=r"$g_{\mathrm{mean}}$",
              color="#ff7f0e")
    ax[1].set_ylim(0, 1.05)
    ax[1].set_ylabel("terminal-state statistic")
    ax[1].set_title("(b) what the state actually is")
    ax[1].legend(frameon=False, ncol=2, loc="upper center")

    pl = col("plateau_onset_epoch")
    st = [np.median([r["stationary_onset_epoch"] for r in recs
                     if r["config"] == c
                     and r["stationary_onset_epoch"] is not None])
          for c in order]
    ax[2].plot(x, pl, "o-", ms=4, lw=1, color="#7f7f7f",
               label="stall trigger fires")
    ax[2].plot(x, st, "s-", ms=4, lw=1, color="#2ca02c",
               label="displacement below threshold")
    ax[2].set_yscale("log")
    ax[2].set_ylim(8, 700)
    ax[2].set_ylabel("epoch")
    ax[2].set_title("(c) when each event happens")
    ax[2].legend(frameon=False, loc="upper center", fontsize=7)

    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(labels, fontsize=7, rotation=40, ha="right")
        a.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(wspace=0.38)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"fp_diagnostics.{ext}"))
    plt.close(fig)
    print("  wrote fp_diagnostics.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figdir", default="paper/ijoc/figures")
    ap.add_argument("--diag", default="results/raw/diagnostics/diagnostics.jsonl")
    args = ap.parse_args()
    os.makedirs(args.figdir, exist_ok=True)
    fig_critical(args.figdir)
    if os.path.exists(args.diag):
        fig_diagnostics(args.diag, args.figdir)


if __name__ == "__main__":
    main()
