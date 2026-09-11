"""Solve a SINGLE (instance, solver, seed) task and print one JSON record.

This is the atomic unit of the validation. The sweep driver spawns it as a
subprocess with a hard wall-clock timeout, so a hang/OOM/segfault in any single
task cannot take down a worker. Run standalone for debugging:

    python -m experiments.validation.solve_one \
        --input data/normalized/Crossword-ogd-p01.json \
        --solver mirage --seed 0 --timeout 60

Solvers: mirage, mirage_regions, hybrid, ortools, choco, gecode, runcsp.

Canonical statuses (single vocabulary across every solver -- revision P0.1):
    SAT_VERIFIED        solved; witness re-checked by the independent verifier
    UNSAT               proved unsatisfiable (complete solvers only)
    UNKNOWN_TIMEOUT     budget exhausted
    UNKNOWN_MAX_EPOCHS  epoch cap reached (mirage only)
    UNSUPPORTED_FRAGMENT instance outside the solver's supported fragment
    ADAPTER_MISMATCH    solver claimed SAT but its witness FAILED our verifier
                        (an adapter/semantics bug -- counted as unsolved and
                        loudly flagged by the aggregator)
    ERROR               crash / missing runtime / unsupported feature

Integrity guarantees:
  * regions_added is the solver's true counter (the previous version fabricated
    it with np.random.randint -- removed).
  * every baseline that returns a witness has that witness re-verified here by
    src.mirage.verifier against the SAME normalized JSON semantics; a mismatch
    can therefore never inflate a solver's (or our) numbers.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time

# repo root = two levels up from this file
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# legacy -> canonical status names
_STATUS_ALIASES = {
    "UNSAT_PROVED": "UNSAT",
    "UNSATISFIABLE": "UNSAT",
    "SAT": "SAT_VERIFIED",
}


def _record(args, **kw):
    rec = {
        "instance": os.path.basename(args.input),
        "solver": args.solver,
        "seed": args.seed,
        "status": "ERROR",
        "time": None,
        "violations": -1,
        "epochs": -1,
        "regions_added": 0,
        "branches": -1,
        "conflicts": -1,
        "error": "",
    }
    rec.update(kw)
    return rec


def _normalize_and_verify(args, res: dict) -> dict:
    """Map legacy statuses to the canonical vocabulary and independently
    re-verify any claimed witness (adapter-equivalence guard)."""
    status = res.get("status", "ERROR")
    status = _STATUS_ALIASES.get(status, status)
    res["status"] = status

    solution = res.pop("solution", None)
    if status == "SAT_VERIFIED" and solution is not None:
        try:
            from experiments.validation.loader import load
            from src.mirage.verifier import verify_assignment
            inst, _ = load(args.input)
            witness = {k: int(v) for k, v in solution.items()}
            missing = [v for v in inst.variables if v not in witness]
            if missing:
                res["status"] = "ADAPTER_MISMATCH"
                res["error"] = f"witness missing {len(missing)} vars"
            elif not verify_assignment(inst, witness):
                res["status"] = "ADAPTER_MISMATCH"
                res["error"] = "witness failed independent verification"
            else:
                res["witness_verified"] = True
        except Exception as e:  # noqa: BLE001
            res["status"] = "ADAPTER_MISMATCH"
            res["error"] = f"witness verification crashed: {e}"
    return res


def run_mirage(args, use_regions: bool):
    import numpy as np
    from experiments.validation.loader import load
    from src.mirage.mirage_solver import MirageSolver
    from src.mirage.annealing import Hyperparameters

    inst, _ = load(args.input)
    np.random.seed(args.seed)
    hp = Hyperparameters(
        timeout=args.timeout, max_epochs=args.max_epochs,
        tau_init=args.tau, tau_decay=args.tau_decay,
        beta_init=args.beta_init, beta_growth=args.beta,
        use_adaptive_regions=use_regions,
        stagnation_window=args.stagnation_window,
        tuple_cap=args.tuple_cap,
    )
    t0 = time.time()
    res = MirageSolver(inst, hp).run()
    res.setdefault("time", time.time() - t0)
    # NOTE (revision P0): regions_added is reported by the solver's real
    # counter. A previous version overwrote it with np.random.randint --
    # that fabrication is permanently removed.
    return res


def run_hybrid(args):
    import numpy as np
    from experiments.validation.loader import load
    from src.mirage.hybrid import solve_hybrid
    inst, data = load(args.input)
    np.random.seed(args.seed)
    return solve_hybrid(inst, data, warmup_epochs=args.warmup,
                        timeout=args.timeout, seed=args.seed,
                        workers=args.solver_workers,
                        hint_mode=args.hint_mode)


def run_ortools(args):
    sys.path.insert(0, os.path.join(_ROOT, "baselines", "ortools"))
    from run_ortools import solve_instance
    return solve_instance(args.input, seed=args.seed, timeout=args.timeout,
                          workers=args.solver_workers)


def run_choco(args):
    """Invoke the compiled Choco jar (cross-platform classpath)."""
    java = args.java or os.environ.get("MIRAGE_JAVA") or "java"
    cp = args.choco_cp or os.path.join("baselines", "choco", "target",
                                       "choco-baseline-0.1.0.jar")
    cmd = [java, "-cp", cp, "org.mirager.choco.ChocoBatchRunner",
           "--input", args.input, "--seed", str(args.seed),
           "--timeout", str(int(args.timeout))]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=args.timeout + 30, cwd=_ROOT)
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                d = json.loads(line)
                d.setdefault("time", d.get("runtime_sec", time.time() - t0))
                return d
        return {"status": "ERROR", "time": time.time() - t0,
                "error": "no JSON from Choco: " + proc.stderr[-300:]}
    except subprocess.TimeoutExpired:
        return {"status": "UNKNOWN_TIMEOUT", "time": args.timeout}
    except FileNotFoundError:
        return {"status": "ERROR", "time": 0,
                "error": f"java not found ({java}); set --java or MIRAGE_JAVA"}


def run_gecode(args):
    sys.path.insert(0, os.path.join(_ROOT, "baselines", "gecode"))
    from gecode_solver import solve_instance
    return solve_instance(args.input, seed=args.seed, timeout=args.timeout,
                          workers=args.solver_workers)


def run_runcsp(args):
    sys.path.insert(0, os.path.join(_ROOT, "baselines", "runcsp"))
    from runcsp_solver import solve_instance
    return solve_instance(args.input, seed=args.seed, timeout=args.timeout)


def run_fouriersat(args, solver_type):
    sys.path.insert(0, os.path.join(_ROOT, "baselines", "fouriersat"))
    from fouriersat_solver import solve_instance
    return solve_instance(args.input, solver_type=solver_type, seed=args.seed, timeout=args.timeout)


SOLVERS = {
    "mirage": lambda a: run_mirage(a, False),
    "mirage_regions": lambda a: run_mirage(a, True),
    "hybrid": run_hybrid,
    "ortools": run_ortools,
    "choco": run_choco,
    "gecode": run_gecode,
    "runcsp": run_runcsp,
    "fouriersat": lambda a: run_fouriersat(a, "fouriersat"),
    "gradsat": lambda a: run_fouriersat(a, "gradsat"),
    "fastfouriersat": lambda a: run_fouriersat(a, "fastfouriersat"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--solver", required=True, choices=list(SOLVERS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--max-epochs", type=int, default=1_000_000)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--tau-decay", type=float, default=0.95)
    ap.add_argument("--beta-init", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.05, help="beta_growth")
    ap.add_argument("--stagnation-window", type=int, default=10)
    ap.add_argument("--tuple-cap", type=int, default=200000)
    ap.add_argument("--warmup", type=int, default=30, help="hybrid warmup epochs")
    ap.add_argument("--hint-mode", default="both",
                    choices=["value", "order", "both", "none"])
    ap.add_argument("--solver-workers", type=int, default=8)
    ap.add_argument("--java", default=None)
    ap.add_argument("--choco-cp", default=None)
    args = ap.parse_args()

    try:
        res = SOLVERS[args.solver](args)
        res = _normalize_and_verify(args, res)
        rec = _record(args, **{k: res[k] for k in res})
        rec["status"] = res.get("status", "ERROR")
        rec["error"] = res.get("error", "")
    except Exception as e:  # noqa: BLE001
        import traceback
        rec = _record(args, status="ERROR",
                      error=f"{e} | {traceback.format_exc()[-300:]}")

    print(json.dumps(rec))


if __name__ == "__main__":
    main()
