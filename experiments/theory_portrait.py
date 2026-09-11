"""Exact trajectories up to the implementation floor (analytical toy example).

This script produces the figure that replaces the earlier purely schematic
"fixed point versus plateau" drawing. Every trajectory shown is computed by
the solver's own projector, consensus and polarization routines through
`experiments.theory_multifactor.frozen_step`; nothing is drawn by hand.

Four panels:

  (a)-(c) phase portraits of the single Boolean disequality x != y on the open
          square (0,1)^2 at gamma = beta/tau below, at, and above the critical
          ratio gamma* = 1/2. They display, respectively, global contraction to
          the uniform fractional point, the one-dimensional continuum of fixed
          points t = 1-s, and escape to the two satisfying vertices. Panel (c)
          also carries the symmetry failure case: an exactly symmetric start
          s = t stays pinned at the unstable uniform point forever, because the
          map preserves the diagonal exactly.

  (d)     the odd cycle C_5 (non-bipartite, hence unsatisfiable) under the same
          three ratios. The continuous state, measured by ||u||_inf, spans many
          orders of magnitude across the three regimes while the decoded
          violation count is flat and identical in all three. This is the
          paper's central methodological claim, exhibited on a family where the
          dynamics are known in closed form.

Usage:
    PYTHONPATH=. python -m experiments.theory_portrait \
        --figdir paper/ijoc/figures --out paper/ijoc/tables
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.theory_multifactor import (  # noqa: E402
    cycle_graph,
    frozen_step,
    laplacian,
    make_projectors,
)

# The three regimes, relative to the single-edge critical ratio gamma* = 1/2.
GAMMAS = [0.35, 0.50, 0.75]
GAMMA_LABELS = [
    r"(a) $\gamma=0.35<\gamma^\star$",
    r"(b) $\gamma=0.50=\gamma^\star$",
    r"(c) $\gamma=0.75>\gamma^\star$",
]
COLORS = ["#1f6fb4", "#b8860b", "#b1283a"]
STEPS_PORTRAIT = 60
STEPS_CYCLE = 120
LOGIT_CLIP = 30.0  # keep sigmoid/logit inside the exactly representable range


def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-u))


def logit(s):
    return np.log(s) - np.log(1.0 - s)


def trajectory(u0, projs, n, gamma, steps):
    """Iterate the frozen map at ratio gamma, holding tau = 1 and beta = gamma.

    The map depends on (tau, beta) only through gamma = beta/tau, so fixing
    tau = 1 loses nothing and keeps every intermediate quantity well scaled.
    """
    u = np.asarray(u0, dtype=float)
    out = [u.copy()]
    for _ in range(steps):
        u = frozen_step(u, projs, n, tau=1.0, beta=gamma)
        u = np.clip(u, -LOGIT_CLIP, LOGIT_CLIP)
        out.append(u.copy())
    return np.array(out)


def decode_violations(u, edges):
    """Decoded violation count: a_i = argmax_a p_i(a), ties broken to value 0."""
    bits = (u > 0.0).astype(int)  # u == 0 decodes to value 0, the min index
    return int(sum(1 for i, j in edges if bits[i] == bits[j]))


# --------------------------------------------------------------------------- #
# Panels (a)-(c): single-edge phase portraits
# --------------------------------------------------------------------------- #

def panel_portrait(ax, gamma, label, seed=0):
    n, edges = 2, [(0, 1)]
    projs = make_projectors(n, edges)

    # A deterministic grid of starts, plus the exactly symmetric diagonal one.
    grid = [0.08, 0.22, 0.36, 0.5, 0.64, 0.78, 0.92]
    starts = [(a, b) for a in grid for b in grid if not (a == 0.5 and b == 0.5)]

    for s0, t0 in starts:
        traj = trajectory(logit(np.array([s0, t0])), projs, n, gamma,
                          STEPS_PORTRAIT)
        st = sigmoid(traj)
        ax.plot(st[:, 0], st[:, 1], lw=0.7, color="#8c9199", alpha=0.75,
                zorder=1)
        ax.plot(st[0, 0], st[0, 1], ".", ms=2.6, color="#5b6068", zorder=2)

    # The critical continuum, drawn only where it exists.
    if abs(gamma - 0.5) < 1e-12:
        ax.plot([0, 1], [1, 0], lw=2.6, color="#b8860b", alpha=0.9, zorder=3,
                solid_capstyle="round")

    # Symmetry failure case: s = t is preserved exactly, so a symmetric start
    # never leaves the diagonal and terminates at the uniform point even when
    # that point is unstable.
    if gamma > 0.5:
        traj = trajectory(logit(np.array([0.85, 0.85])), projs, n, gamma,
                          STEPS_PORTRAIT)
        st = sigmoid(traj)
        ax.plot(st[:, 0], st[:, 1], lw=2.0, color="#b1283a", zorder=4)
        ax.plot(st[0, 0], st[0, 1], "o", ms=4.5, color="#b1283a", zorder=5)
        ax.annotate("symmetric start:\nnever escapes", xy=(0.70, 0.70),
                    xytext=(0.06, 0.99), fontsize=6.4, color="#b1283a",
                    ha="left", va="top", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.22", fc="white",
                              ec="none", alpha=0.88),
                    arrowprops=dict(arrowstyle="->", color="#b1283a", lw=0.9))

    # Satisfying vertices and the uniform point.
    ax.plot([1, 0], [0, 1], "*", ms=10, color="#1a7f37", zorder=6,
            clip_on=False)
    ax.plot([0.5], [0.5], "o", ms=5, mfc="white", mec="#222", mew=1.1, zorder=7,
            clip_on=False)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$s=p_x(1)$", fontsize=7.5)
    ax.set_ylabel(r"$t=p_y(1)$", fontsize=7.5)
    ax.set_title(label, fontsize=8, pad=3)
    ax.tick_params(labelsize=6.5)
    for sp in ax.spines.values():
        sp.set_color("#C9CDD3")


# --------------------------------------------------------------------------- #
# Panel (d): odd cycle, continuous state versus decoded violations
# --------------------------------------------------------------------------- #

def panel_cycle(ax, seed=0):
    n, edges = cycle_graph(5)
    projs = make_projectors(n, edges)
    L, _, _ = laplacian(n, edges)
    mu_max = float(np.max(np.linalg.eigvals(L).real))

    rng = np.random.default_rng(seed)
    u0 = rng.normal(scale=1.0, size=n)

    ax2 = ax.twinx()
    for gamma, color in zip(GAMMAS, COLORS):
        traj = trajectory(u0, projs, n, gamma, STEPS_CYCLE)
        norms = np.max(np.abs(traj), axis=1)
        viol = [decode_violations(u, edges) for u in traj]
        ax.semilogy(norms, lw=1.7, color=color,
                    label=rf"$\gamma={gamma:.2f}$")
        ax2.plot(viol, lw=1.6, ls=(0, (4, 2)), color=color, alpha=0.85)

    ax.set_xlabel("frozen epoch $t$", fontsize=7.5)
    ax.set_ylabel(r"$\|u^t\|_\infty$ (log)", fontsize=7.5)
    ax2.set_ylabel("decoded violations", fontsize=7.5)
    ax2.set_ylim(-0.4, max(2.6, len(edges) * 0.6))
    ax2.set_yticks([0, 1, 2])
    ax.set_title(rf"(d) $C_5$: $\mu_{{\max}}={mu_max:.3f}$, "
                 rf"$\gamma^\star={1/mu_max:.3f}$", fontsize=8, pad=3)
    ax.tick_params(labelsize=6.5)
    ax2.tick_params(labelsize=6.5)
    ax.legend(fontsize=6.8, loc="lower left", frameon=False)
    for sp in list(ax.spines.values()) + list(ax2.spines.values()):
        sp.set_color("#C9CDD3")
    return mu_max, u0, projs, n, edges


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figdir", default="paper/ijoc/figures")
    ap.add_argument("--out", default="paper/ijoc/tables")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.figdir, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    # One row: the paper is under a hard page limit, and a 2x2 grid of square
    # phase portraits costs most of a page. Panels stay legible because the
    # figure is drawn at close to its printed width.
    fig, axes = plt.subplots(1, 4, figsize=(9.8, 2.55),
                             gridspec_kw=dict(wspace=0.52))
    for ax, gamma, label in zip(axes[:3], GAMMAS, GAMMA_LABELS):
        panel_portrait(ax, gamma, label, seed=args.seed)
    mu_max, u0, projs, n, edges = panel_cycle(axes[3], seed=args.seed)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.figdir, f"phase_portrait.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------------
    # Macros. Every number quoted in the caption or the text is emitted
    # here, so none is typed by hand into the manuscript.
    # --------------------------------------------------------------
    # Terminal state of the exactly symmetric single-edge start at gamma=0.75.
    n2, e2 = 2, [(0, 1)]
    p2 = make_projectors(n2, e2)
    sym = trajectory(logit(np.array([0.85, 0.85])), p2, n2, 0.75,
                     STEPS_PORTRAIT)
    sym_final = float(np.max(np.abs(sigmoid(sym[-1]) - 0.5)))

    # Decoded violation count along the C_5 runs: constant in every regime.
    viol_sets = {}
    span = {}
    for gamma in GAMMAS:
        traj = trajectory(u0, projs, n, gamma, STEPS_CYCLE)
        viol_sets[gamma] = sorted({decode_violations(u, edges) for u in traj})
        span[gamma] = float(np.max(np.abs(traj[-1])))
    all_flat = all(len(v) == 1 for v in viol_sets.values())
    common = {v[0] for v in viol_sets.values()}
    ratio = max(span.values()) / max(min(span.values()), 1e-300)

    lines = [
        "% AUTO-GENERATED by experiments/theory_portrait.py -- do not edit",
        macro("RevPortraitSteps", STEPS_PORTRAIT),
        macro("RevPortraitCycleSteps", STEPS_CYCLE),
        macro("RevPortraitSymDev", "0" if sym_final == 0.0
              else fmt_sci(sym_final)),
        macro("RevPortraitViol", sorted(common)[0] if len(common) == 1 else "--"),
        macro("RevPortraitFlat", "yes" if all_flat else "no"),
        macro("RevPortraitSpanRatio", fmt_sci(ratio)),
        macro("RevPortraitMuMax", f"{mu_max:.3f}"),
        macro("RevPortraitGammaStar", f"{1.0/mu_max:.3f}"),
    ]
    path = os.path.join(args.out, "portrait_numbers.tex")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"[portrait] wrote {path}")
    print(f"[portrait] symmetric-start deviation from uniform: {sym_final:.3e}")
    for gamma in GAMMAS:
        print(f"[portrait] gamma={gamma}: decoded violations "
              f"{viol_sets[gamma]}, terminal |u|_inf={span[gamma]:.3e}")
    assert all_flat, "decoded violation count was not constant in some regime"
    assert len(common) == 1, "regimes disagree on the decoded violation level"


def fmt_sci(x):
    """Math content, no delimiters: 9.0e16 -> 9.0\\times10^{16}.

    The use site supplies the $...$, so the same macro composes inside an
    existing math group as well as on its own.
    """
    exp = int(np.floor(np.log10(abs(x))))
    mant = x / (10.0 ** exp)
    return f"{mant:.1f}\\times10^{{{exp}}}"


def macro(name, value):
    return (f"\\providecommand{{\\{name}}}{{}}\n"
            f"\\renewcommand{{\\{name}}}{{{value}}}")


if __name__ == "__main__":
    main()
