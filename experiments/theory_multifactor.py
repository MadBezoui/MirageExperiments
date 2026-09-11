"""Numerical verification of the multi-factor fixed-point theorem.

Claim (Theorem 1 of the manuscript). Let G = (V, E) be a graph with no
isolated vertex, and let the CSP have one Boolean variable per vertex and one
disequality constraint x_i != x_j per edge. Write p_i = (s_i, 1-s_i) and
u_i = logit(s_i). Then for the frozen unclipped MIRAGE-R map with inverse-degree
consensus weights and gamma = beta/tau,

    u^+ = gamma * L u,        L = I - D^{-1} A                      (*)

where A is the adjacency matrix and D the degree matrix, i.e. L is the
random-walk normalized graph Laplacian. Consequently:

  (i)   the frozen map is conjugate, through the coordinatewise logit, to a
        LINEAR map, globally on the open box (0,1)^V -- not merely to first
        order at the uniform state;
  (ii)  u = 0 (the uniform state) is always a fixed point, and a non-uniform
        interior fixed point exists iff 1/gamma is an eigenvalue of L, in
        which case the interior fixed-point set is the corresponding
        eigenspace;
  (iii) the uniform state is globally attracting on the box iff
        gamma * mu_max(L) < 1, and the convergence is geometric with rate
        gamma * mu_max;
  (iv)  mu_max(L) = 2 iff G has a bipartite connected component, so the
        critical ratio is gamma* = 1/mu_max >= 1/2, with equality exactly in
        the bipartite (2-colorable, i.e. satisfiable) case.

This script verifies (*) and each consequence directly against the solver's
own projector and consensus code, on a library of graphs.

Usage:
    PYTHONPATH=. python -m experiments.theory_multifactor \
        --out paper/ijoc/tables --figdir paper/ijoc/figures
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np

from src.mirage.consensus import apply_polarization, geometric_consensus
from src.mirage.csp_core import TableConstraint, Variable
from src.mirage.local_projectors import BooleanTableProjector

# Representability budget. One epoch maps u to gamma*L*u with |L u| <= 2|u|,
# so with |u| <= LOGIT_BOX and gamma <= GAMMA_MAX the output logit magnitude is
# at most 2*GAMMA_MAX*LOGIT_BOX, which must stay well inside the range where
# exp/log are exact to double precision (|logit| < 700).
LOGIT_BOX = 8.0
GAMMA_MAX = 8.0


# ----------------------------------------------------------------------
# Graph library
# ----------------------------------------------------------------------

def path_graph(n):
    return n, [(i, i + 1) for i in range(n - 1)]


def cycle_graph(n):
    return n, [(i, (i + 1) % n) for i in range(n)]


def complete_graph(n):
    return n, list(itertools.combinations(range(n), 2))


def complete_bipartite(a, b):
    return a + b, [(i, a + j) for i in range(a) for j in range(b)]


def petersen():
    outer = [(i, (i + 1) % 5) for i in range(5)]
    spokes = [(i, 5 + i) for i in range(5)]
    inner = [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
    return 10, outer + spokes + inner


def star(n):
    return n + 1, [(0, i + 1) for i in range(n)]


GRAPHS = {
    "single edge $K_2$": complete_graph(2),
    "path $P_4$": path_graph(4),
    "star $K_{1,5}$": star(5),
    "even cycle $C_4$": cycle_graph(4),
    "even cycle $C_6$": cycle_graph(6),
    "odd cycle $C_3$": cycle_graph(3),
    "odd cycle $C_5$": cycle_graph(5),
    "odd cycle $C_7$": cycle_graph(7),
    "odd cycle $C_9$": cycle_graph(9),
    "complete $K_4$": complete_graph(4),
    "complete $K_5$": complete_graph(5),
    "bipartite $K_{3,3}$": complete_bipartite(3, 3),
    "Petersen": petersen(),
}


# ----------------------------------------------------------------------
# Spectral quantities
# ----------------------------------------------------------------------

def laplacian(n, edges):
    A = np.zeros((n, n))
    for i, j in edges:
        A[i, j] = A[j, i] = 1.0
    deg = A.sum(1)
    assert np.all(deg > 0), "graph has an isolated vertex"
    return np.eye(n) - (A / deg[:, None]), A, deg


def is_bipartite(n, edges):
    adj = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    color = [-1] * n
    for s in range(n):
        if color[s] != -1:
            continue
        color[s] = 0
        stack = [s]
        while stack:
            v = stack.pop()
            for w in adj[v]:
                if color[w] == -1:
                    color[w] = 1 - color[v]
                    stack.append(w)
                elif color[w] == color[v]:
                    return False
    return True


# ----------------------------------------------------------------------
# The frozen map, evaluated through the solver's own components
# ----------------------------------------------------------------------



def make_projectors(n, edges):
    projs = []
    for k, (i, j) in enumerate(edges):
        c = TableConstraint(id=f"c{k}", scope=[f"x{i}", f"x{j}"], positive=True,
                            tuples=[(0, 1), (1, 0)])
        doms = {f"x{i}": [0, 1], f"x{j}": [0, 1]}
        projs.append(BooleanTableProjector(c, doms))
    return projs


def frozen_step(u, projs, n, tau, beta, epsilon=0.0):
    """One exact frozen MIRAGE-R epoch, evaluated on the real solver code.

    Input and output are logit vectors. p_i = (1-sigmoid(u_i), sigmoid(u_i))
    with index 1 carrying value 1, matching the projector's bit convention.
    """
    s = 1.0 / (1.0 + np.exp(-u))
    marg = {f"x{i}": np.array([1.0 - s[i], s[i]]) for i in range(n)}
    props = {f"x{i}": [] for i in range(n)}
    for pr in projs:
        out = pr.project({v: marg[v] for v in pr.scope}, tau)
        for v in pr.scope:
            props[v].append(out[v])
    u_new = np.empty(n)
    for i in range(n):
        pl = props[f"x{i}"]
        w = [1.0 / len(pl)] * len(pl)
        cons = geometric_consensus(pl, w, epsilon)
        pol = apply_polarization(cons, beta, epsilon)
        u_new[i] = np.log(pol[1]) - np.log(pol[0])
    return u_new


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/ijoc/tables")
    ap.add_argument("--figdir", default="paper/ijoc/figures")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.figdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rows = []
    max_lin_err = 0.0
    max_fp_err = 0.0

    print("== verifying u+ = gamma * L u on the solver's own code ==")
    for name, (n, edges) in GRAPHS.items():
        L, A, deg = laplacian(n, edges)
        eig = np.linalg.eigvals(L).real
        mu_max = float(np.max(eig))
        mu_min_pos = float(np.min(eig[eig > 1e-9])) if np.any(eig > 1e-9) else 0.0
        bip = is_bipartite(n, edges)
        gamma_star = 1.0 / mu_max

        # ---- (i) exact linearity, over random logit vectors and random gamma
        errs = []
        for _ in range(200):
            u = rng.uniform(-LOGIT_BOX, LOGIT_BOX, size=n)
            tau = float(rng.uniform(0.5, 4.0))
            beta = float(rng.uniform(0.5, 4.0))
            g = beta / tau
            assert g <= GAMMA_MAX
            pred = g * (L @ u)
            got = frozen_step(u, make_projectors(n, edges), n, tau, beta)
            scale = max(1.0, float(np.max(np.abs(pred))))
            errs.append(float(np.max(np.abs(pred - got))) / scale)
        lin_err = max(errs)
        max_lin_err = max(max_lin_err, lin_err)

        # ---- (ii) eigenvector of L for eigenvalue mu is fixed at gamma = 1/mu
        w, V = np.linalg.eig(L)
        k = int(np.argmax(w.real))
        v = V[:, k].real
        v = v / np.max(np.abs(v)) * 3.0          # keep inside the box
        tau, beta = 2.0, 2.0 / mu_max            # gamma = 1/mu_max
        got = frozen_step(v, make_projectors(n, edges), n, tau, beta)
        fp_err = float(np.max(np.abs(got - v))) / max(1.0, float(np.max(np.abs(v))))
        max_fp_err = max(max_fp_err, fp_err)

        # ---- (iii) the asymptotic contraction rate equals gamma*mu_max
        def observed_rate(gamma_factor, steps=120):
            """Geometric rate of ||u^t|| under the frozen map at gamma."""
            tau, beta = 2.0, 2.0 * gamma_factor * gamma_star
            projs = make_projectors(n, edges)
            u = rng.uniform(-0.5, 0.5, size=n)
            ratios = []
            for t in range(steps):
                nu = float(np.linalg.norm(u))
                if nu < 1e-250 or nu > 1e6:
                    break
                u2 = frozen_step(u, projs, n, tau, beta)
                nu2 = float(np.linalg.norm(u2))
                if nu2 <= 0.0:
                    break
                ratios.append(nu2 / nu)
                # renormalize to stay inside the representable box while
                # measuring the rate: the map is linear, so this is exact.
                u = u2 / max(1.0, nu2 / 0.5)
            tail = ratios[len(ratios) // 2:] or ratios
            return float(np.median(tail))

        rate_sub = observed_rate(0.9)
        rate_sup = observed_rate(1.1)
        pred_sub = 0.9 * gamma_star * mu_max      # = 0.9
        pred_sup = 1.1 * gamma_star * mu_max      # = 1.1
        rate_err = max(abs(rate_sub - pred_sub), abs(rate_sup - pred_sup))
        contracts = rate_sub < 1.0
        escapes = rate_sup > 1.0

        rows.append(dict(name=name, n=n, m=len(edges), bipartite=bip,
                         mu_max=mu_max, gamma_star=gamma_star,
                         lin_err=lin_err, fp_err=fp_err,
                         contracts=contracts, escapes=escapes,
                         rate_sub=rate_sub, rate_sup=rate_sup,
                         rate_err=rate_err))
        print(f"  {name:22s} n={n:2d} m={len({tuple(sorted(e)) for e in edges}):2d} "
              f"bip={str(bip):5s} mu_max={mu_max:.6f} gamma*={gamma_star:.6f} "
              f"lin_err={lin_err:.2e} fp_err={fp_err:.2e} "
              f"rate(0.9)={rate_sub:.4f} rate(1.1)={rate_sup:.4f}")

    assert max_lin_err < 1e-12, f"linearity failed: {max_lin_err}"
    assert max_fp_err < 1e-9, f"eigenvector fixed point failed: {max_fp_err}"
    max_rate_err = max(r["rate_err"] for r in rows)
    assert max_rate_err < 1e-6, f"rate prediction failed: {max_rate_err}"
    for r in rows:
        assert r["contracts"], f"no contraction below gamma* for {r['name']}"
        assert r["escapes"], f"no escape above gamma* for {r['name']}"
        if r["bipartite"]:
            assert abs(r["mu_max"] - 2.0) < 1e-9, r["name"]
            assert abs(r["gamma_star"] - 0.5) < 1e-9, r["name"]
        else:
            assert r["mu_max"] < 2.0 - 1e-9, r["name"]
            assert r["gamma_star"] > 0.5 + 1e-9, r["name"]
    print(f"\nALL ASSERTIONS PASSED "
          f"(max linearity error {max_lin_err:.2e}, "
          f"max fixed-point error {max_fp_err:.2e}, "
          f"max rate error {max_rate_err:.2e})")

    # ---- closed form for odd cycles: mu_max = 1 + cos(pi/n)
    print("\n== closed form for cycles: mu_max = 1 + cos(pi/n) for odd n ==")
    for n in (3, 5, 7, 9, 11, 21, 101):
        L, _, _ = laplacian(*cycle_graph(n))
        mu = float(np.max(np.linalg.eigvals(L).real))
        pred = 1.0 + np.cos(np.pi / n)
        print(f"  C_{n:<4d} numerical {mu:.10f}  closed form {pred:.10f}  "
              f"diff {abs(mu-pred):.2e}")
        assert abs(mu - pred) < 1e-9

    # ------------------------------------------------------------------
    # LaTeX table
    # ------------------------------------------------------------------
    lines = [
        r"\begin{table}[htbp]",
        r"\TABLE{Numerical verification of Theorem~1 on the"
        r" solver's own projector and consensus code.\label{tab:multifactor}}",
        r"{\footnotesize\begin{tabular}{lrrcrrrr}",
        r"\toprule",
        r"disequality graph & $|V|$ & $|E|$ & bipartite & $\mu_{\max}(L)$ &"
        r" $\gamma^\star$ & lin.\ err. & f.p.\ err. \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['name']} & {r['n']} & {r['m']} & "
            f"{'yes' if r['bipartite'] else 'no'} & {r['mu_max']:.4f} & "
            f"{r['gamma_star']:.4f} & {r['lin_err']:.1e} & {r['fp_err']:.1e} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"{\emph{lin.\ err.} is the largest relative deviation between $\gamma Lu$ and one implemented frozen epoch, over 200 random states and random $(\tau,\beta)$; \emph{f.p.\ err.} the displacement of a leading eigenvector at $\gamma=\gamma^\star$. Both are at double-precision round-off.}",
        r"\end{table}",
    ]
    with open(os.path.join(args.out, "multifactor.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote {os.path.join(args.out,'multifactor.tex')}")

    def fmt_sci(x):
        """Math content without delimiters: these three macros appear in
        running text, where 9.2e-14 would clash with the surrounding math."""
        mant, exp = f"{x:.1e}".split("e")
        return f"{mant}\\times10^{{{int(exp)}}}"

    macros = {
        "RevThmMaxLinErr": fmt_sci(max_lin_err),
        "RevThmMaxFpErr": fmt_sci(max_fp_err),
        "RevThmMaxRateErr": fmt_sci(max_rate_err),
        "RevThmGraphs": len(rows),
        "RevThmCThreeGamma": f"{1.0/(1.0+np.cos(np.pi/3)):.4f}",
        "RevThmCFiveGamma": f"{1.0/(1.0+np.cos(np.pi/5)):.4f}",
        "RevThmCNineGamma": f"{1.0/(1.0+np.cos(np.pi/9)):.4f}",
    }
    with open(os.path.join(args.out, "theory_numbers.tex"), "w") as fh:
        fh.write("% AUTO-GENERATED by experiments/theory_multifactor.py\n")
        for k, v in sorted(macros.items()):
            fh.write(f"\\providecommand{{\\{k}}}{{}}\n")
            fh.write(f"\\renewcommand{{\\{k}}}{{{v}}}\n")

    with open(os.path.join(args.out, "multifactor.json"), "w") as fh:
        json.dump(rows, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
