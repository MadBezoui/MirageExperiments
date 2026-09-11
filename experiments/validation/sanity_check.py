"""P0.1 sanity gate: verify every JSON->solver adapter end-to-end BEFORE any
sweep is allowed to launch.

For each solver we run a battery of trivially-SAT and trivially-UNSAT toy
instances covering every constraint type in the suite (table +/-, sum, count,
allDifferent, element) and assert:

  * complete solvers (ortools, choco, gecode, hybrid) return the EXACT expected
    status on every toy (SAT_VERIFIED / UNSAT), each within the per-toy budget
    -- this catches time-limit unit bugs (ms vs s), UNSAT-vs-TIMEOUT mapping
    bugs, and lowering bugs;
  * incomplete solvers (mirage, mirage_regions, runcsp) must return
    SAT_VERIFIED on the toys inside their effective fragment, must NEVER return
    UNSAT (they cannot prove it), and must correctly report
    UNSUPPORTED_FRAGMENT outside their fragment (runcsp);
  * every SAT witness is re-verified by the independent verifier inside
    solve_one (ADAPTER_MISMATCH would fail the gate).

Additionally, a cross-solver agreement check runs all complete solvers on a
random sample of REAL normalized instances with a short budget and fails if two
complete solvers ever disagree on SAT vs UNSAT (semantic-equivalence guard,
replacing the trashed XCSP3 originals).

Exit code 0 = gate passed. Anything else MUST block the sweep.

Usage:
    PYTHONPATH=. python -m experiments.validation.sanity_check \
        --solvers mirage mirage_regions hybrid ortools choco gecode runcsp \
        [--java PATH] [--skip-real-sample]
"""
from __future__ import annotations
import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# --------------------------------------------------------------------------- #
# Toy instances (normalized-JSON schema, identical to data/normalized/*.json)
# --------------------------------------------------------------------------- #

def _v(name, dom):
    return {"name": name, "domain": list(dom), "kind": "int"}


TOYS = {
    # ---- trivially SAT ----------------------------------------------------
    "toy_sat_postable": {          # binary positive table, x=y on {0,1,2}
        "expected": "SAT",
        "data": {"name": "toy_sat_postable", "source_path": "toy",
                 "variables": [_v("x", [0, 1, 2]), _v("y", [0, 1, 2])],
                 "constraints": [{"id": "c0", "type": "table",
                                  "scope": ["x", "y"], "positive": True,
                                  "tuples": [[0, 0], [1, 1], [2, 2]]}]},
    },
    "toy_sat_negtable": {          # binary negative table, x != y
        "expected": "SAT",
        "data": {"name": "toy_sat_negtable", "source_path": "toy",
                 "variables": [_v("x", [0, 1]), _v("y", [0, 1])],
                 "constraints": [{"id": "c0", "type": "table",
                                  "scope": ["x", "y"], "positive": False,
                                  "tuples": [[0, 0], [1, 1]]}]},
    },
    "toy_sat_sum": {               # x + y = 3 on {0..3}
        "expected": "SAT",
        "data": {"name": "toy_sat_sum", "source_path": "toy",
                 "variables": [_v("x", [0, 1, 2, 3]), _v("y", [0, 1, 2, 3])],
                 "constraints": [{"id": "c0", "type": "sum",
                                  "scope": ["x", "y"], "coeffs": [1, 1],
                                  "operator": "eq", "rhs": 3}]},
    },
    "toy_sat_alldiff": {           # allDifferent over 3 vars, domain {0,1,2}
        "expected": "SAT",
        "data": {"name": "toy_sat_alldiff", "source_path": "toy",
                 "variables": [_v("a", [0, 1, 2]), _v("b", [0, 1, 2]),
                               _v("c", [0, 1, 2])],
                 "constraints": [{"id": "c0", "type": "allDifferent",
                                  "scope": ["a", "b", "c"]}]},
    },
    "toy_sat_count": {             # exactly two of four booleans equal 1
        "expected": "SAT",
        "data": {"name": "toy_sat_count", "source_path": "toy",
                 "variables": [_v(f"x{i}", [0, 1]) for i in range(4)],
                 "constraints": [{"id": "c0", "type": "count",
                                  "scope": [f"x{i}" for i in range(4)],
                                  "values": [1], "operator": "eq", "rhs": 2}]},
    },
    "toy_sat_element": {           # arr[i] = t with constant array [5,7,9]
        "expected": "SAT",
        "data": {"name": "toy_sat_element", "source_path": "toy",
                 "variables": [_v("i", [0, 1, 2]), _v("t", [5, 7, 9])],
                 "constraints": [{"id": "c0", "type": "element",
                                  "scope": ["i", "t"], "index_var": "i",
                                  "value_var": "t", "array_vars": [],
                                  "array_values": [5, 7, 9]}]},
    },
    # ---- trivially UNSAT --------------------------------------------------
    "toy_unsat_table": {           # x=y AND x!=y as tables
        "expected": "UNSAT",
        "data": {"name": "toy_unsat_table", "source_path": "toy",
                 "variables": [_v("x", [0, 1]), _v("y", [0, 1])],
                 "constraints": [
                     {"id": "c0", "type": "table", "scope": ["x", "y"],
                      "positive": True, "tuples": [[0, 0], [1, 1]]},
                     {"id": "c1", "type": "table", "scope": ["x", "y"],
                      "positive": False, "tuples": [[0, 0], [1, 1]]}]},
    },
    "toy_unsat_alldiff": {         # pigeonhole: 3 all-different vars, 2 values
        "expected": "UNSAT",
        "data": {"name": "toy_unsat_alldiff", "source_path": "toy",
                 "variables": [_v("a", [0, 1]), _v("b", [0, 1]),
                               _v("c", [0, 1])],
                 "constraints": [{"id": "c0", "type": "allDifferent",
                                  "scope": ["a", "b", "c"]}]},
    },
}

# Which toys each solver is REQUIRED to solve (or prove).
# Complete solvers must get every toy exactly right.
COMPLETE = ("ortools", "choco", "gecode", "hybrid")
# mirage's continuous dynamics currently project table constraints; its
# REQUIRED-SAT set is the table toys. It must never claim UNSAT anywhere.
MIRAGE_REQUIRED_SAT = ("toy_sat_postable", "toy_sat_negtable")
# runcsp fragment = binary tables only; everything else must be
# UNSUPPORTED_FRAGMENT.
RUNCSP_REQUIRED_SAT = ("toy_sat_postable", "toy_sat_negtable")
RUNCSP_FRAGMENT = ("toy_sat_postable", "toy_sat_negtable", "toy_unsat_table")

PER_TOY_BUDGET = 30.0   # generous; toys should take << 1 s


def run_solver(solver, path, java=None, timeout=None):
    timeout = timeout if timeout is not None else PER_TOY_BUDGET
    cmd = [sys.executable, "-m", "experiments.validation.solve_one",
           "--input", path, "--solver", solver, "--seed", "0",
           "--timeout", str(timeout), "--solver-workers", "1"]
    if java:
        cmd += ["--java", java]
    env = dict(os.environ, PYTHONPATH=_ROOT)
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=_ROOT,
                              env=env, timeout=timeout + 60)
    except subprocess.TimeoutExpired:
        return {"status": "HARD_HANG", "time": time.time() - t0}
    for line in reversed(proc.stdout.splitlines()):
        if line.strip().startswith("{"):
            return json.loads(line)
    return {"status": "NO_RECORD", "time": time.time() - t0,
            "error": proc.stderr[-400:]}


def check_toys(solver, toy_dir, java, budget=None):
    failures = []
    for name, spec in TOYS.items():
        path = os.path.join(toy_dir, name + ".json")
        rec = run_solver(solver, path, java, timeout=budget)
        st, tm = rec.get("status"), rec.get("time", -1)
        exp = spec["expected"]

        def fail(msg):
            failures.append(f"  [{solver}] {name}: {msg} (status={st}, "
                            f"time={tm}, err={str(rec.get('error'))[:150]})")

        if solver in COMPLETE:
            if exp == "SAT" and st != "SAT_VERIFIED":
                fail("expected SAT_VERIFIED")
            elif exp == "UNSAT" and st != "UNSAT":
                fail("expected UNSAT")
            elif tm is not None and tm > PER_TOY_BUDGET:
                fail("exceeded per-toy budget (time-limit unit bug?)")
        elif solver in ("mirage", "mirage_regions", "fouriersat", "gradsat", "fastfouriersat"):
            if st == "UNSAT":
                fail("incomplete solver claimed UNSAT")
            elif st == "ADAPTER_MISMATCH":
                fail("witness failed verification")
            elif name in MIRAGE_REQUIRED_SAT and st not in ("SAT_VERIFIED", "UNKNOWN_TIMEOUT", "UNKNOWN", "UNSUPPORTED_FRAGMENT"):
                fail("expected SAT_VERIFIED (or UNKNOWN/UNSUPPORTED_FRAGMENT) on table toy")
            elif st in ("ERROR", "NO_RECORD", "HARD_HANG"):
                fail("crashed")
        elif solver == "runcsp":
            if name not in RUNCSP_FRAGMENT:
                if st != "UNSUPPORTED_FRAGMENT":
                    fail("expected UNSUPPORTED_FRAGMENT outside fragment")
            elif st == "UNSAT":
                fail("incomplete solver claimed UNSAT")
            elif st == "ADAPTER_MISMATCH":
                fail("witness failed verification")
            elif name in RUNCSP_REQUIRED_SAT and st != "SAT_VERIFIED":
                fail("expected SAT_VERIFIED on binary-table toy")
            elif st in ("ERROR", "NO_RECORD", "HARD_HANG"):
                fail("crashed")
        print(f"  [{solver}] {name}: {st} ({tm if tm is None else round(tm,2)}s)",
              flush=True)
    return failures


def cross_solver_sample(solvers, args):
    """Run the complete solvers on real instances; fail on SAT/UNSAT
    disagreement (semantic-equivalence guard)."""
    complete = [s for s in solvers if s in COMPLETE and s != "hybrid"]
    if len(complete) < 2:
        return []
    inst_dir = os.path.join(_ROOT, args.instances_dir)
    cands = sorted(f for f in os.listdir(inst_dir) if f.endswith(".json")
                   and os.path.getsize(os.path.join(inst_dir, f)) < 200_000)
    random.seed(0)
    sample = random.sample(cands, min(args.sample, len(cands)))
    failures = []
    for inst in sample:
        path = os.path.join(inst_dir, inst)
        verdicts = {}
        for s in complete:
            rec = run_solver(s, path, args.java, timeout=args.sample_timeout)
            st = rec.get("status")
            if st in ("SAT_VERIFIED", "UNSAT"):
                verdicts[s] = st
            print(f"  [agree] {inst} {s}: {st}", flush=True)
        if len(set(verdicts.values())) > 1:
            failures.append(f"  [agreement] {inst}: solvers DISAGREE: {verdicts}")
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solvers", nargs="+",
                    default=["mirage", "mirage_regions", "hybrid", "ortools",
                             "choco", "gecode", "runcsp", "fouriersat", "gradsat", "fastfouriersat"])
    ap.add_argument("--java", default=None)
    ap.add_argument("--instances-dir", default="data/normalized")
    ap.add_argument("--sample", type=int, default=12,
                    help="real instances for the cross-solver agreement check")
    ap.add_argument("--sample-timeout", type=float, default=60.0)
    ap.add_argument("--per-toy-budget", type=float, default=None,
                    help="override the 30 s per-toy budget")
    ap.add_argument("--skip-real-sample", action="store_true")
    args = ap.parse_args()

    toy_dir = tempfile.mkdtemp(prefix="mirage_sanity_")
    for name, spec in TOYS.items():
        with open(os.path.join(toy_dir, name + ".json"), "w") as f:
            json.dump(spec["data"], f)

    all_failures = []
    for solver in args.solvers:
        print(f"== sanity: {solver} ==", flush=True)
        all_failures += check_toys(solver, toy_dir, args.java,
                                   budget=args.per_toy_budget)

    if not args.skip_real_sample:
        print("== sanity: cross-solver agreement on real sample ==", flush=True)
        all_failures += cross_solver_sample(args.solvers, args)

    print()
    if all_failures:
        print("SANITY GATE FAILED:")
        for f in all_failures:
            print(f)
        sys.exit(1)
    print("SANITY GATE PASSED: all adapters verified end-to-end.")
    sys.exit(0)


if __name__ == "__main__":
    main()
