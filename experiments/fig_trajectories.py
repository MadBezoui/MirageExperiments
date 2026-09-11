"""Decoded-progress trajectory figure and its dispersion statistics.

Regenerates `figures/convergence.pdf` from
`results/raw/revision_theory/trajectories.jsonl`.

The figure has to carry one finding: decoded progress is a coarse staircase.
A run either stalls above about a tenth of its initial violation count or drops
straight to zero, and nothing lands in between -- which is why fitting a decay
exponent to it says nothing.

Three design decisions follow, and each replaces something an earlier version
got wrong:

  * the ordinate is **linear**, not logarithmic. B_j reaches exactly zero when a
    witness is found, and a logarithmic axis cannot draw zero; the earlier
    version invented a floor at 1e-3, drew dotted "drops" down to it and marked
    them with stars, spending three of its four decades on that scaffolding.
    On a linear axis zero is just zero and the whole apparatus disappears;
  * the trajectories are drawn as **steps**, because B_j is a running minimum
    sampled every `decode_freq` epochs and is constant in between. Interpolating
    would draw a smooth decay the data does not contain;
  * the band no timed-out run ever entered is shaded, and the terminal values
    are drawn as a marginal sharing that same ordinate, so the hole in the
    distribution lines up with the empty band.

Slope eligibility reproduces the archived analysis exactly: an unweighted
regression of log(B_j/B_1) on log t_j over checkpoints with t_j > 1 and
B_j/B_1 > 1e-3, fitted only when more than five checkpoints survive.

Usage:
    PYTHONPATH=. python -m experiments.fig_trajectories \
        --raw results/raw/revision_theory/trajectories.jsonl \
        --figdir paper/ijoc/figures --out paper/ijoc/tables
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

C_TIMEOUT = "#C0392B"   # red, matching `choco` in make_paper_figures
C_WITNESS = "#1F6FB2"   # blue, matching `mirage`
C_BAND = "#8c9199"
FLOOR = 1e-3            # the analysis filter on B_j/B_1, not a plotting device


def load(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def prepare(rows):
    """One record per run with the quantities the panels need."""
    out = []
    for r in rows:
        y = np.asarray(r["running_min_trajectory"], dtype=float)
        solved = (r.get("status") == "SAT_VERIFIED"
                  or r.get("final_violations", 1) == 0)
        if len(y) < 3:
            out.append(dict(short=True, solved=solved,
                            terminal=0.0 if solved else None, slope=None))
            continue
        y0 = y[0] if y[0] > 0 else 1.0
        t = np.arange(1, len(y) + 1) * r["decode_freq"]
        ratio = y / y0
        mask = (ratio > FLOOR) & (t > 1)
        slope = None
        if mask.sum() > 5:
            slope = float(np.polyfit(np.log(t[mask]), np.log(ratio[mask]), 1)[0])
        flat = float(np.mean(np.diff(y) == 0)) if len(y) > 1 else 1.0
        out.append(dict(short=False, solved=solved, t=t, ratio=ratio,
                        terminal=float(ratio[-1]), slope=slope, flat=flat,
                        steps=int(np.count_nonzero(np.diff(y) < 0))))
    return out


# --------------------------------------------------------------------------- #

def panel_staircase(ax, recs, floor_line, t_witness_max, handles):
    tmax = 1.0
    for r in recs:
        if r["short"]:
            continue
        color = C_WITNESS if r["solved"] else C_TIMEOUT
        ax.step(r["t"], r["ratio"], where="post", color=color, alpha=0.42,
                lw=1.1, zorder=3)
        tmax = max(tmax, float(r["t"][-1]))
        if r["solved"]:
            ax.plot([r["t"][-1]], [0.0], "o", ms=3.4, color=C_WITNESS,
                    mec="white", mew=0.5, zorder=5)

    ax.axhspan(0.0, floor_line, color=C_BAND, alpha=0.14, zorder=1, lw=0)
    ax.axhline(floor_line, color=C_BAND, lw=0.9, ls=(0, (4, 2)), zorder=2)

    ax.set_xscale("log")
    ax.set_xlim(3.5, tmax * 1.5)
    ax.set_ylim(-0.045, 1.06)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel(r"epoch $t_j$ (log scale)", fontsize=9)
    ax.set_ylabel(r"running minimum $B_j/B_1$", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.annotate(f"no timed-out run ever entered this band "
                f"($B_j/B_1<{floor_line:.2f}$)",
                xy=(0.985, floor_line * 0.5), xycoords=("axes fraction", "data"),
                ha="right", va="center", fontsize=7.2, color="#4a4f57",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none",
                          alpha=0.82))
    ax.annotate(f"every witness found by epoch {t_witness_max:g}",
                xy=(t_witness_max, 0.0), xytext=(60, 0.36),
                fontsize=7.2, color=C_WITNESS, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="->", color=C_WITNESS, lw=0.8,
                                shrinkA=2, shrinkB=4,
                                connectionstyle="arc3,rad=0.25"))
    ax.legend(handles=handles, loc="upper right", fontsize=7.6,
              frameon=True, framealpha=0.94, edgecolor="#CCCCCC",
              borderpad=0.5, labelspacing=0.4, handlelength=1.5)
    ax.set_title("(a) every decoded trajectory, as the step function it is",
                 fontsize=9.5, pad=5)


def panel_marginal(ax, recs, floor_line):
    """Terminal B_j/B_1 for all runs, on the ordinate of panel (a)."""
    rng = np.random.default_rng(0)
    for solved, color in ((False, C_TIMEOUT), (True, C_WITNESS)):
        vals = [r["terminal"] for r in recs
                if r["solved"] is solved and r["terminal"] is not None]
        x = 0.5 + rng.uniform(-0.2, 0.2, size=len(vals))
        ax.plot(x, vals, "o", ms=4.4, color=color, alpha=0.8, mec="white",
                mew=0.6, zorder=3, linestyle="none")
    ax.axhspan(0.0, floor_line, color=C_BAND, alpha=0.14, zorder=1, lw=0)
    ax.axhline(floor_line, color=C_BAND, lw=0.9, ls=(0, (4, 2)), zorder=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.045, 1.06)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelsize=8)
    ax.grid(False)
    # no in-panel labels: the colours and the shaded band already say which
    # points are stalled and which solved, and the panel is too narrow to
    # carry text without colliding with the data.
    ax.set_title("(b) terminal value", fontsize=9.5, pad=5)


def panel_steps(ax, recs):
    """How many strict decreases a run ever makes: the staircase is short."""
    timed = [r["steps"] for r in recs if not r["short"] and not r["solved"]]
    won = [r["steps"] for r in recs if not r["short"] and r["solved"]]
    bins = np.arange(-0.5, max(timed + won) + 1.5)
    ax.hist([timed, won], bins=bins, stacked=True,
            color=[C_TIMEOUT, C_WITNESS], alpha=0.85,
            edgecolor="white", linewidth=0.7, zorder=3)
    ax.set_xlabel("strict decreases per run", fontsize=9)
    ax.set_ylabel("runs", fontsize=9)
    ax.set_xticks(np.arange(0, int(bins[-1]) + 1))
    ax.tick_params(labelsize=8)
    ax.set_title("(c) the staircase is short", fontsize=9.5, pad=5)


# --------------------------------------------------------------------------- #

def macro(name, value):
    return (f"\\providecommand{{\\{name}}}{{}}\n"
            f"\\renewcommand{{\\{name}}}{{{value}}}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw",
                    default="results/raw/revision_theory/trajectories.jsonl")
    ap.add_argument("--figdir", default="paper/ijoc/figures")
    ap.add_argument("--out", default="paper/ijoc/tables")
    args = ap.parse_args()
    os.makedirs(args.figdir, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    rows = load(args.raw)
    recs = prepare(rows)
    eligible = [r for r in recs if not r["short"] and r["slope"] is not None]
    solved = [r for r in recs if r["solved"]]
    slopes = np.asarray([r["slope"] for r in eligible])
    flat_median = float(np.median([r["flat"] for r in eligible]))
    to_min = [float(r["ratio"].min()) for r in eligible]
    min_ratio, med_ratio = float(min(to_min)), float(np.median(to_min))
    below_tenth = sum(1 for v in to_min if v < 0.1)
    steps_median = float(np.median([r["steps"] for r in eligible]))

    w_epochs = []
    for r, raw in zip(recs, rows):
        if r["solved"]:
            y = np.asarray(raw["running_min_trajectory"], dtype=float)
            z = np.flatnonzero(y == 0)
            if z.size:
                w_epochs.append(float((z[0] + 1) * raw["decode_freq"]))
    t_witness_max = max(w_epochs)

    # The paper states that the slope-eligible runs are exactly the runs that
    # timed out. Assert it rather than assume it.
    assert not any(r["solved"] for r in eligible), \
        "a run that found a witness entered the slope fit"
    assert len(recs) == len(eligible) + len(solved), \
        "runs are not partitioned into slope-eligible and witness-finding"
    assert min_ratio > 0, "a timed-out run reached zero violations"

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "axes.grid": True, "grid.color": "#B9BFC9",
                         "grid.linewidth": 0.6, "grid.linestyle": ":",
                         "grid.alpha": 0.65, "axes.axisbelow": True})
    fig, axes = plt.subplots(
        1, 3, figsize=(9.6, 2.9),
        gridspec_kw=dict(width_ratios=[3.1, 0.72, 1.45], wspace=0.30))
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#666666")
        ax.spines["bottom"].set_color("#666666")

    handles = [
        Line2D([0], [0], color=C_TIMEOUT, lw=2.0, alpha=0.85,
               label=f"timed out ({len(eligible)})"),
        Line2D([0], [0], color=C_WITNESS, lw=2.0, alpha=0.85,
               label=f"witness found ({len(solved)})"),
        Patch(facecolor=C_BAND, alpha=0.2, label="never occupied"),
    ]
    panel_staircase(axes[0], recs, min_ratio, t_witness_max, handles)
    panel_marginal(axes[1], recs, min_ratio)
    for side in ("left", "bottom"):
        axes[1].spines[side].set_visible(False)
    panel_steps(axes[2], recs)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.figdir, f"convergence.{ext}"),
                    dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    med = float(np.median(slopes))
    q1, q3 = (float(v) for v in np.percentile(slopes, [25, 75]))
    lines = [
        "% AUTO-GENERATED by experiments/fig_trajectories.py -- do not edit",
        macro("RevTrajSlopeMedian", f"{med:.3f}"),
        macro("RevTrajSlopeQOne", f"{q1:.3f}"),
        macro("RevTrajSlopeQThree", f"{q3:.3f}"),
        macro("RevTrajSlopeMin", f"{float(slopes.min()):.3f}"),
        macro("RevTrajSlopeMax", f"{float(slopes.max()):.3f}"),
        macro("RevTrajFlatPct", f"{flat_median * 100:.0f}"),
        macro("RevTrajEligibleN", len(eligible)),
        macro("RevTrajWitnessN", len(solved)),
        macro("RevTrajMinRatio", f"{min_ratio:.3f}"),
        macro("RevTrajMedRatio", f"{med_ratio:.3f}"),
        macro("RevTrajBelowTenth", below_tenth),
        macro("RevTrajWitnessMaxEpoch", int(t_witness_max)),
        macro("RevTrajStepsMedian", f"{steps_median:.0f}"),
    ]
    path = os.path.join(args.out, "traj_numbers.tex")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"[traj] wrote {path}")
    print(f"[traj] eligible={len(eligible)} witness={len(solved)} "
          f"median={med:.4f} IQR=[{q1:.4f},{q3:.4f}] flat={flat_median:.3f}")
    print(f"[traj] never below {min_ratio:.3f} of the initial count "
          f"(median {med_ratio:.3f}); {below_tenth} runs below 0.1")
    print(f"[traj] all {len(solved)} witnesses by epoch {int(t_witness_max)}; "
          f"median strict decreases per eligible run {steps_median:.0f}")


if __name__ == "__main__":
    main()
