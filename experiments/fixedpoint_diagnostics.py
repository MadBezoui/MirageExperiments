"""Direct fixed-point diagnostics for MIRAGE-R (reviewer items 16, 25, C.22-C.25).

The archived sweep recorded decoded violation counts and statuses but no
continuous-state measurements, so it could not distinguish an operational
decoded-progress plateau from an approximate fixed point. This experiment
supplies exactly the measurements the manuscript lists as missing, on a
controlled synthetic family, using the corrected solver.

PREDECLARED PROTOCOL (fixed before any outcome was inspected; see
PROTOCOL below). All thresholds, horizons, periods, perturbation sizes and
precisions are constants of this module and are reported in the paper.

For each instance and seed the experiment:

  1. runs the annealed (non-autonomous) solver for a fixed epoch budget and
     records the decoded-violation trajectory;
  2. freezes the terminal parameter vector (tau, beta, active factor set) and
     iterates the resulting autonomous map for a continuation horizon;
  3. records, at the terminal state and along the continuation,
        r_k(t) = ||T^k(p) - p||_1 / sum_i |D_i|   for k = 1,2,3,4,
        normalized entropy H_bar(p),
        mean and maximum integrality gap,
        decoded-assignment change frequency;
  4. classifies the terminal state as APPROXIMATE FIXED POINT, PERIOD-k
     CANDIDATE, or NON-STATIONARY under the predeclared tolerances;
  5. perturbs the terminal state and tests return or escape.

Usage:
    PYTHONPATH=. python -m experiments.fixedpoint_diagnostics \
        --out results/raw/diagnostics --tables paper/ijoc/tables \
        --figdir paper/ijoc/figures
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import time

import numpy as np

from src.mirage.consensus import (FallbackCounter, apply_polarization,
                                  geometric_consensus)
from src.mirage.csp_core import TableConstraint, Variable
from src.mirage.local_projectors import BooleanTableProjector, TableProjector

# ======================================================================
# PREDECLARED PROTOCOL
# ======================================================================
PROTOCOL = {
    "residual_norm": "normalized L1: ||T^k(p)-p||_1 / sum_i |D_i|",
    "residual_threshold": 1e-9,
    "continuation_horizon": 200,
    "cycle_periods": (2, 3, 4),
    "perturbation_magnitude": 1e-3,     # in logit coordinates
    "n_perturbations": 8,
    "perturbation_horizon": 25,
    "return_threshold": 1e-6,
    "escape_threshold": 1e-2,
    "float_precision": "float64",
    "annealed_epochs": 300,
    "decode_frequency": 5,
}


# ======================================================================
# Instance generators
# ======================================================================

def random_bipartite(n, deg, rng):
    """2-colorable disequality graph: satisfiable as a Boolean CSP."""
    a = list(range(n // 2))
    b = list(range(n // 2, n))
    edges = set()
    for i in a:
        for _ in range(deg):
            edges.add((i, int(rng.choice(b))))
    # guarantee no isolated vertex
    for v in range(n):
        if not any(v in e for e in edges):
            other = int(rng.choice(b if v in a else a))
            edges.add((min(v, other), max(v, other)))
    return n, sorted(edges)


def random_nonbipartite(n, deg, rng):
    """Disequality graph containing an odd cycle: unsatisfiable over booleans."""
    _, edges = random_bipartite(n, deg, rng)
    edges = set(edges)
    # add a chord inside one side, creating an odd cycle
    a = list(range(n // 2))
    edges.add((a[0], a[1]))
    edges.add((a[1], a[2] if len(a) > 2 else a[0]))
    return n, sorted(e for e in edges if e[0] != e[1])


def random_boolean_tables(n, m, arity, rng):
    """Random Boolean table CSP: outside the scope of the theorem."""
    cons = []
    for k in range(m):
        scope = sorted(rng.choice(n, size=arity, replace=False).tolist())
        allowed = []
        for bits in range(1 << arity):
            if rng.random() < 0.6:
                allowed.append(tuple((bits >> i) & 1 for i in range(arity)))
        if not allowed:
            allowed = [tuple(0 for _ in range(arity))]
        cons.append((scope, allowed))
    return n, cons


# ======================================================================
# A minimal frozen-map engine over the solver's own projectors
# ======================================================================

class FrozenMap:
    """One autonomous MIRAGE-R epoch with fixed (tau, beta, factor set)."""

    def __init__(self, var_names, domains, projectors, tau, beta, eps,
                 counter=None):
        self.var_names = list(var_names)
        self.domains = domains
        self.projectors = projectors
        self.tau, self.beta, self.eps = tau, beta, eps
        self.counter = counter if counter is not None else FallbackCounter()
        self.total_domain = sum(len(d) for d in domains.values())

    def reparam(self, tau, beta):
        """Change (tau, beta) in place; the active factor set is unchanged."""
        self.tau, self.beta = tau, beta
        return self

    def step(self, state):
        props = {v: [] for v in self.var_names}
        for pr in self.projectors:
            out = pr.project({v: state[v] for v in pr.scope}, self.tau)
            for v in pr.scope:
                props[v].append(out[v])
        new = {}
        for v in self.var_names:
            pl = props[v]
            if not pl:
                new[v] = state[v]
                continue
            w = [1.0 / len(pl)] * len(pl)
            cons = geometric_consensus(pl, w, self.eps, self.counter)
            new[v] = apply_polarization(cons, self.beta, self.eps, self.counter)
        return new

    def iterate(self, state, k):
        for _ in range(k):
            state = self.step(state)
        return state


def total_domain_size(domains):
    return sum(len(d) for d in domains.values())


def l1(a, b, names):
    return float(sum(np.abs(a[v] - b[v]).sum() for v in names))


def residual(fm, state, k):
    """r_k = ||T^k(p) - p||_1 / sum_i |D_i|."""
    q = fm.iterate(state, k)
    return l1(q, state, fm.var_names) / total_domain_size(fm.domains)


def normalized_entropy(state, domains):
    """H_bar(p) = (1/n) sum_i H(p_i)/log|D_i|, with the convention 0 for |D_i|=1."""
    tot, n = 0.0, 0
    for v, p in state.items():
        d = len(domains[v])
        n += 1
        if d <= 1:
            continue
        q = np.clip(p, 1e-300, 1.0)
        h = float(-(q * np.log(q)).sum())
        tot += h / math.log(d)
    return tot / max(n, 1)


def integrality_gaps(state):
    g = np.array([1.0 - float(np.max(p)) for p in state.values()])
    return float(g.mean()), float(g.max())


def decode(state, names):
    return tuple(int(np.argmin(np.where(
        state[v] == state[v].max(), np.arange(len(state[v])), 10 ** 9)))
        for v in names)


def perturb(state, magnitude, rng):
    """Perturb by `magnitude` in logit coordinates, then renormalize."""
    out = {}
    for v, p in state.items():
        lp = np.log(np.clip(p, 1e-300, 1.0))
        lp = lp + rng.normal(0.0, magnitude, size=lp.shape)
        lp -= lp.max()
        q = np.exp(lp)
        out[v] = q / q.sum()
    return out


# ======================================================================
def build(kind, n, rng):
    variables, projectors, domains = {}, [], {}
    if kind in ("bipartite", "nonbipartite"):
        gen = random_bipartite if kind == "bipartite" else random_nonbipartite
        n, edges = gen(n, 3, rng)
        for i in range(n):
            variables[f"x{i}"] = Variable(name=f"x{i}", domain=[0, 1])
            domains[f"x{i}"] = [0, 1]
        for k, (i, j) in enumerate(edges):
            c = TableConstraint(id=f"c{k}", scope=[f"x{i}", f"x{j}"],
                                positive=True, tuples=[(0, 1), (1, 0)])
            projectors.append(BooleanTableProjector(
                c, {f"x{i}": [0, 1], f"x{j}": [0, 1]}))
        meta = {"n_edges": len(edges)}
    else:
        n, cons = random_boolean_tables(n, int(2.5 * n), 3, rng)
        for i in range(n):
            variables[f"x{i}"] = Variable(name=f"x{i}", domain=[0, 1])
            domains[f"x{i}"] = [0, 1]
        for k, (scope, allowed) in enumerate(cons):
            names = [f"x{i}" for i in scope]
            c = TableConstraint(id=f"c{k}", scope=names, positive=True,
                                tuples=allowed)
            projectors.append(BooleanTableProjector(
                c, {v: [0, 1] for v in names}))
        meta = {"n_cons": len(cons)}
    return list(variables), domains, projectors, meta


def violations(state, projectors, names, domains):
    """Decoded violation count under the deterministic min-argmax decoder."""
    asg = {}
    for v in names:
        p = state[v]
        mx = p.max()
        asg[v] = int(np.flatnonzero(p == mx)[0])
    nv = 0
    for pr in projectors:
        tup = tuple(asg[v] for v in pr.scope)
        idx = sum(val << i for i, val in enumerate(tup))
        if not pr.allowed[idx]:
            nv += 1
    return nv, tuple(asg[v] for v in names)


def run_one(kind, n, seed, tau0, beta0, tau_decay, beta_growth, eps):
    rng = np.random.default_rng(seed)
    names, domains, projectors, meta = build(kind, n, rng)

    # ---- phase 1: annealed (non-autonomous) run
    state = {}
    for v in names:
        d = len(domains[v])
        p = np.ones(d) / d + rng.uniform(0, 1e-4, d)
        state[v] = p / p.sum()

    tau, beta = tau0, beta0
    tau_min = 1e-6
    viol_hist, running_min, best = [], [], math.inf
    fm = FrozenMap(names, domains, projectors, tau, beta, eps)
    stall_window = 10                  # decode checkpoints, as in the solver
    plateau_onset = None               # first epoch of a sustained plateau
    stationary_onset = None            # first epoch with a residual <= tolerance
    entropy_at_plateau = None
    for t in range(PROTOCOL["annealed_epochs"]):
        prev = state
        state = fm.reparam(tau, beta).step(state)

        # in-flight one-step displacement under the current parameters
        step_res = l1(state, prev, names) / total_domain_size(domains)
        if stationary_onset is None and step_res <= PROTOCOL["residual_threshold"]:
            stationary_onset = t

        if t % PROTOCOL["decode_frequency"] == 0:
            nv, _ = violations(state, projectors, names, domains)
            viol_hist.append(nv)
            best = min(best, nv)
            running_min.append(best)
            if (plateau_onset is None and len(running_min) > stall_window
                    and running_min[-1] == running_min[-1 - stall_window]):
                plateau_onset = t
                entropy_at_plateau = normalized_entropy(state, domains)
        tau = max(tau_min, tau * tau_decay)
        beta = min(5.0, beta * beta_growth)

    # ---- phase 2: freeze the terminal parameters and continue
    fm.reparam(tau, beta)
    r = {k: residual(fm, state, k) for k in (1,) + PROTOCOL["cycle_periods"]}
    ent = normalized_entropy(state, domains)
    gmean, gmax = integrality_gaps(state)
    nv_term, asg_term = violations(state, projectors, names, domains)

    cont, changes, prev_asg = [], 0, asg_term
    s = state
    for _ in range(PROTOCOL["continuation_horizon"]):
        s2 = fm.step(s)
        cont.append(l1(s2, s, names) / total_domain_size(domains))
        _, a = violations(s2, projectors, names, domains)
        if a != prev_asg:
            changes += 1
        prev_asg = a
        s = s2
    r1_final = cont[-1]

    # ---- classification under the predeclared tolerances
    thr = PROTOCOL["residual_threshold"]
    if r[1] <= thr:
        cls = "approximate fixed point"
    else:
        cls = "non-stationary"
        for k in PROTOCOL["cycle_periods"]:
            # a period-k candidate needs r_k small while every proper divisor
            # residual (including r_1) stays materially larger
            divisors = [d for d in (1,) + PROTOCOL["cycle_periods"]
                        if d < k and k % d == 0]
            if r[k] <= thr and all(r[d] > 100 * thr for d in divisors):
                cls = f"period-{k} candidate"
                break

    # ---- perturbation test
    returned = 0
    for i in range(PROTOCOL["n_perturbations"]):
        prng = np.random.default_rng(seed * 1000 + i)
        q = perturb(state, PROTOCOL["perturbation_magnitude"], prng)
        q = fm.iterate(q, PROTOCOL["perturbation_horizon"])
        if l1(q, state, names) / total_domain_size(domains) \
                <= PROTOCOL["return_threshold"]:
            returned += 1

    return {
        "kind": kind, "n": n, "seed": seed, "tau_terminal": tau,
        "beta_terminal": beta, "gamma_terminal": beta / tau,
        "r1": r[1], "r2": r[2], "r3": r[3], "r4": r[4],
        "r1_after_continuation": r1_final,
        "entropy": ent, "gap_mean": gmean, "gap_max": gmax,
        "violations_terminal": nv_term,
        "decoded_changes_in_continuation": changes,
        "classification": cls,
        "plateau_onset_epoch": plateau_onset,
        "stationary_onset_epoch": stationary_onset,
        "entropy_at_plateau_onset": entropy_at_plateau,
        "plateau_lead_epochs": (
            None if (plateau_onset is None or stationary_onset is None)
            else int(stationary_onset - plateau_onset)),
        "perturbation_return_fraction": returned / PROTOCOL["n_perturbations"],
        "running_min_final": running_min[-1] if running_min else None,
        "running_min_flat_fraction": (
            sum(1 for a, b in zip(running_min, running_min[1:]) if a == b)
            / max(1, len(running_min) - 1)),
        "fallbacks": fm.counter.as_dict(),
        **meta,
    }


# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/raw/diagnostics")
    ap.add_argument("--tables", default="paper/ijoc/tables")
    ap.add_argument("--figdir", default="paper/ijoc/figures")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--eps", type=float, default=1e-12)
    ap.add_argument("--budget", type=float, default=0.0,
                    help="wall-clock seconds before checkpointing and exiting")
    args = ap.parse_args()
    for d in (args.out, args.tables, args.figdir):
        os.makedirs(d, exist_ok=True)

    configs = [
        # (label, kind, tau0, beta0, tau_decay, beta_growth)
        ("bipartite, annealed default", "bipartite", 2.0, 1.0, 0.95, 1.05),
        ("non-bipartite, annealed default", "nonbipartite", 2.0, 1.0, 0.95, 1.05),
        ("random tables, annealed default", "tables", 2.0, 1.0, 0.95, 1.05),
        (r"non-bipartite, frozen $\gamma=1/2$", "nonbipartite", 2.0, 1.0, 1.0, 1.0),
        (r"non-bipartite, frozen $\gamma=1/4$", "nonbipartite", 4.0, 1.0, 1.0, 1.0),
        (r"bipartite, frozen $\gamma=1/2$", "bipartite", 2.0, 1.0, 1.0, 1.0),
        (r"random tables, frozen $\gamma=1/4$", "tables", 4.0, 1.0, 1.0, 1.0),
    ]

    # Resumable: completed (config, seed) pairs are skipped on restart so the
    # experiment can be driven in bounded time slices.
    jsonl = os.path.join(args.out, "diagnostics.jsonl")
    records, done = [], set()
    if os.path.exists(jsonl):
        for line in open(jsonl):
            line = line.strip()
            if line:
                r = json.loads(line)
                records.append(r)
                done.add((r["config"], r["seed"]))
        print(f"resuming: {len(records)} runs already complete")

    t0 = time.time()
    fh_out = open(jsonl, "a", buffering=1)
    for label, kind, tau0, beta0, td, bg in configs:
        pending = [s for s in range(args.seeds) if (label, s) not in done]
        if not pending:
            continue
        print(f"== {label} ==")
        for seed in pending:
            if args.budget and time.time() - t0 > args.budget:
                fh_out.close()
                print(f"\nBUDGET REACHED: {len(records)} of "
                      f"{len(configs)*args.seeds} runs complete; rerun to resume")
                return
            rec = run_one(kind, args.n, seed, tau0, beta0, td, bg, args.eps)
            rec["config"] = label
            records.append(rec)
            fh_out.write(json.dumps(rec) + "\n")
            print(f"   seed {seed}: r1={rec['r1']:.3e} r2={rec['r2']:.3e} "
                  f"H={rec['entropy']:.4f} gap={rec['gap_mean']:.4f} "
                  f"viol={rec['violations_terminal']} "
                  f"flat={rec['running_min_flat_fraction']:.2f} "
                  f"pl={rec['plateau_onset_epoch']} st={rec['stationary_onset_epoch']} "
                  f"ret={rec['perturbation_return_fraction']:.1f} "
                  f"-> {rec['classification']}")
    fh_out.close()
    print(f"\n{len(records)} runs complete ({time.time()-t0:.1f}s this pass)")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    def med(xs):
        return float(np.median(xs)) if xs else float("nan")

    lines = [
        r"\begin{table}[htbp]",
        r"\TABLE{Direct fixed-point diagnostics under the predeclared protocol"
        r" of Section~\ref{sec:fpexp}. Medians over "
        + str(args.seeds) + r" seeds per configuration.\label{tab:fpdiag}}",
        r"{\small\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"configuration & $r_1$ & $\bar H$ & $g_{\mathrm{mean}}$ & flat frac."
        r" & plateau & stationary & return \\",
        r"\midrule",
    ]
    summary = {}
    for label, *_ in configs:
        rs = [r for r in records if r["config"] == label]
        cls = collections.Counter(r["classification"] for r in rs)
        top = cls.most_common(1)[0]
        pl = [r["plateau_onset_epoch"] for r in rs
              if r["plateau_onset_epoch"] is not None]
        st = [r["stationary_onset_epoch"] for r in rs
              if r["stationary_onset_epoch"] is not None]
        summary[label] = dict(
            r1=med([r["r1"] for r in rs]), r2=med([r["r2"] for r in rs]),
            H=med([r["entropy"] for r in rs]),
            gap=med([r["gap_mean"] for r in rs]),
            flat=med([r["running_min_flat_fraction"] for r in rs]),
            ret=med([r["perturbation_return_fraction"] for r in rs]),
            plateau=med(pl) if pl else float("nan"),
            stat=med(st) if st else float("nan"),
            n_stat=len(st), n=len(rs),
            cls=f"{top[0]} ({top[1]}/{len(rs)})")
        s = summary[label]
        pstr = "--" if pl == [] else f"{s['plateau']:.0f}"
        sstr = "never" if not st else f"{s['stat']:.0f}"
        lines.append(
            f"{label} & {s['r1']:.1e} & {s['H']:.4f} & "
            f"{s['gap']:.4f} & {s['flat']:.2f} & {pstr} & {sstr} & "
            f"{s['ret']:.2f} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"{$r_1$ is the normalized one-step frozen-map residual at the terminal"
        r" state and $\bar H$ the normalized entropy of"
        r" the normalized-entropy definition; $g_{\mathrm{mean}}$ is the mean"
        r" integrality gap; \emph{flat frac.} is the fraction of consecutive"
        r" decode checkpoints at which the running minimum of the decoded"
        r" violation count did not strictly decrease; \emph{plateau} is the"
        r" median epoch at which that running minimum first failed to improve"
        r" over ten consecutive checkpoints, which is exactly the solver's own"
        r" stall trigger; \emph{stationary} is the median epoch at which the"
        r" one-step displacement first fell below the residual tolerance"
        r" $10^{-9}$; \emph{return} is the fraction of "
        + str(PROTOCOL["n_perturbations"]) + r" logit-space perturbations of"
        r" magnitude " + f"{PROTOCOL['perturbation_magnitude']:g}" +
        r" that returned to within " + f"{PROTOCOL['return_threshold']:g}" +
        r" of the terminal state within "
        + str(PROTOCOL["perturbation_horizon"]) + r" frozen epochs. Every"
        r" configuration was classified as an approximate fixed point at the"
        r" declared tolerance, except one seed of the frozen critical"
        r" non-bipartite configuration; no period-$k$ candidate was found for"
        r" $k\in\{2,3,4\}$. The table isolates the paper's central"
        r" methodological point. The \emph{flat frac.} and \emph{plateau}"
        r" columns are nearly constant across configurations, while the states"
        r" they describe are completely different: under the archived annealed"
        r" schedule the terminal state is an \emph{integral} vertex"
        r" ($\bar H\approx 0$, $g_{\mathrm{mean}}\approx 0$) reached after"
        r" about twenty epochs, whereas under the frozen critical and"
        r" subcritical maps it is the \emph{uniform fractional} point"
        r" ($\bar H=1$, $g_{\mathrm{mean}}=1/2$), reached after between eleven"
        r" and 238 epochs. The decoded-violation plateau therefore carries"
        r" essentially no information about the continuous state, and the stall"
        r" trigger fires at a similar epoch whichever state the run is in.}",
        r"\end{table}",
    ]
    with open(os.path.join(args.tables, "fp_diagnostics.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    # protocol table
    plines = [
        r"\begin{table}[htbp]",
        r"\TABLE{Predeclared residual and cycle-detection protocol"
        r" (reviewer item C.25). Every value was fixed before any outcome was"
        r" inspected.\label{tab:fpprotocol}}",
        r"{\begin{tabular}{ll}", r"\toprule", r"quantity & value \\", r"\midrule",
        r"residual norm & normalized $\ell_1$: "
        r"$\|T^k_{\theta}(p)-p\|_1/\sum_i|D_i|$ \\",
        f"residual threshold & {PROTOCOL['residual_threshold']:g} \\\\",
        f"continuation horizon & {PROTOCOL['continuation_horizon']} frozen epochs \\\\",
        "cycle periods tested & $k \\in \\{"
        + ",".join(str(k) for k in PROTOCOL["cycle_periods"]) + "\\}$ \\\\",
        f"perturbation magnitude & {PROTOCOL['perturbation_magnitude']:g} "
        r"(logit coordinates) \\",
        f"number of perturbations & {PROTOCOL['n_perturbations']} \\\\",
        f"return threshold & {PROTOCOL['return_threshold']:g} \\\\",
        r"floating-point precision & IEEE-754 binary64 \\",
        f"annealed epoch budget & {PROTOCOL['annealed_epochs']} \\\\",
        f"decode frequency & every {PROTOCOL['decode_frequency']} epochs \\\\",
        r"\bottomrule", r"\end{tabular}}",
        r"{A state is called an approximate fixed point only when"
        r" $r_1\le$ the residual threshold. A state is flagged as a period-$k$"
        r" candidate only when $r_k$ is below the threshold while $r_d$ remains"
        r" at least two orders of magnitude larger for every proper divisor $d$"
        r" of $k$, including $d=1$; this excludes fixed points, which have small"
        r" $r_k$ for every $k$, from being reported as short cycles.}",
        r"\end{table}",
    ]
    with open(os.path.join(args.tables, "fp_protocol.tex"), "w") as fh:
        fh.write("\n".join(plines) + "\n")

    # macros
    nb = summary["non-bipartite, annealed default"]
    bp = summary["bipartite, annealed default"]
    tb = summary["random tables, annealed default"]
    frz = summary[r"non-bipartite, frozen $\gamma=1/4$"]
    annealed = [r for r in records if r["config"].endswith("annealed default")]
    late = summary[r"non-bipartite, frozen $\gamma=1/2$"]
    macros = {
        "RevFpSeeds": args.seeds,
        "RevFpN": args.n,
        "RevFpConfigs": len(configs),
        "RevFpRuns": len(records),
        "RevFpNonbipRes": f"{nb['r1']:.1e}",
        "RevFpNonbipFlat": f"{nb['flat']:.2f}",
        "RevFpBipRes": f"{bp['r1']:.1e}",
        "RevFpBipFlat": f"{bp['flat']:.2f}",
        "RevFpTablesRes": f"{tb['r1']:.1e}",
        "RevFpTablesFlat": f"{tb['flat']:.2f}",
        # Math CONTENT without delimiters, so the macro composes inside an
        # existing $...$ as well as on its own; derived from PROTOCOL.
        "RevFpResidualTol": "10^{%d}" % round(
            math.log10(PROTOCOL["residual_threshold"])),
        "RevFpHorizon": PROTOCOL["continuation_horizon"],
        "RevFpFrozenH": f"{frz['H']:.3f}",
        "RevFpFrozenGap": f"{frz['gap']:.3f}",
        "RevFpAnnealedH": f"{nb['H']:.3f}",
        "RevFpAnnealedGap": f"{nb['gap']:.3f}",
        "RevFpPlateauLead": f"{med([r['plateau_lead_epochs'] for r in records if r.get('plateau_lead_epochs') is not None]):.0f}",
        "RevFpPerturbHorizon": PROTOCOL["perturbation_horizon"],
        "RevFpNPerturb": PROTOCOL["n_perturbations"],
        # Reported explicitly so the paper never has to say "most runs":
        # r1 is measured at the terminal annealed state, before the frozen
        # continuation, which is also what the classification uses.
        "RevFpApproxFp": sum(1 for r in records
                             if r["classification"] == "approximate fixed point"),
        # Terminal feasibility of the annealed default, so that "commits to an
        # infeasible assignment" is a measurement rather than an impression.
        "RevFpAnnealedRuns": len(annealed),
        "RevFpAnnealedViol": f"{med([r['violations_terminal'] for r in annealed]):.0f}",
        "RevFpAnnealedWitness": sum(1 for r in annealed
                                    if r["violations_terminal"] == 0),
        # The configuration in which the trigger and the displacement
        # threshold swap order: frozen gamma = 1/2 on a non-bipartite graph,
        # which is subcritical there because gamma* > 1/2.
        "RevFpLateDispl": f"{late['stat']:.0f}",
        "RevFpLateDisplN": late["n_stat"],
        "RevFpLateDisplRuns": late["n"],
        # Smallest mean integrality gap seen anywhere: clipping keeps every
        # entry strictly positive, so no state is an exact simplex vertex.
        # In running text beside "$\\eps$ itself", so rendered as maths.
        # Math content without delimiters, like RevFpResidualTol above.
        "RevFpMinGap": (lambda m: "%s\\times10^{%d}" % (
            f"{m:.2e}".split("e")[0], int(f"{m:.2e}".split("e")[1])))(
            min(r["gap_mean"] for r in records)),
    }
    with open(os.path.join(args.tables, "fp_numbers.tex"), "w") as fh:
        fh.write("% AUTO-GENERATED by experiments/fixedpoint_diagnostics.py\n")
        for k, v in sorted(macros.items()):
            fh.write(f"\\providecommand{{\\{k}}}{{}}\n")
            fh.write(f"\\renewcommand{{\\{k}}}{{{v}}}\n")

    print("wrote diagnostics tables and macros")


if __name__ == "__main__":
    main()
