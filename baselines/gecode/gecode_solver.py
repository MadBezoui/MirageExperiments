"""Gecode baseline runner via CPMpy/MiniZinc for MIRAGE-R normalized JSON.

Requires the MiniZinc toolchain (which bundles Gecode) on PATH.

INTEGRITY RULES (revision P0.1):
  * NO silent fallback to another solver. If the Gecode backend is unavailable
    or fails, the record is an explicit ERROR attributed to gecode -- results
    must never be misattributed across solvers.
  * Domain holes are enforced exactly (membership table), never relaxed.
  * On SAT the solution is extracted and returned so the harness can re-check
    it with the independent verifier (adapter-equivalence guard).
  * Statuses use the canonical vocabulary:
      SAT_VERIFIED | UNSAT | UNKNOWN_TIMEOUT | ERROR
"""
import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import cpmpy as cp


def gecode_available() -> bool:
    """True iff the minizinc:gecode backend can actually run."""
    try:
        from cpmpy import SolverLookup
        return any(s.startswith("minizinc") for s in SolverLookup.solvernames())
    except Exception:
        return False


def build_model(data: Dict[str, Any]):
    var_map = {}
    constraints = []
    unsupported = []

    for v in data["variables"]:
        vals = sorted(int(x) for x in v["domain"])
        var_map[v["name"]] = cp.intvar(vals[0], vals[-1], name=v["name"])
        holes = set(range(vals[0], vals[-1] + 1)) - set(vals)
        if holes:
            # enforce exact domain membership (P0.1: no silent relaxation)
            constraints.append(cp.Table([var_map[v["name"]]], [[x] for x in vals]))

    for c in data["constraints"]:
        ctype = c["type"]
        scope = c.get("scope", [])

        if ctype == "table":
            svars = [var_map[s] for s in scope]
            tuples = [[int(x) for x in t] for t in c["tuples"]]
            if c.get("positive", True):
                constraints.append(cp.Table(svars, tuples))
            else:
                for t in tuples:
                    constraints.append(
                        ~cp.all([svars[i] == t[i] for i in range(len(svars))]))

        elif ctype == "sum":
            expr = sum(int(co) * var_map[s] for co, s in zip(c["coeffs"], scope))
            constraints.append(_relop(expr, c["operator"], int(c["rhs"])))

        elif ctype == "allDifferent":
            constraints.append(cp.AllDifferent([var_map[s] for s in scope]))

        elif ctype == "element":
            idx = var_map[c["index_var"]]
            target = var_map[c["value_var"]]
            if c.get("array_vars"):
                arr = [var_map[a] for a in c["array_vars"]]
                constraints.append(cp.Element(arr, idx) == target)
            else:
                consts = [int(x) for x in c["array_values"]]
                constraints.append(cp.Element(consts, idx) == target)

        elif ctype == "count":
            values = set(int(x) for x in c.get("values", [c.get("value")]))
            expr = sum(cp.any([var_map[s] == v for v in values])
                       if len(values) > 1 else (var_map[s] == next(iter(values)))
                       for s in scope)
            op = c.get("operator", c.get("op", "")).lower()
            constraints.append(_relop(expr, op, int(c["rhs"])))
        else:
            unsupported.append(ctype)

    return cp.Model(constraints), var_map, unsupported


def _relop(expr, op: str, rhs: int):
    op = op.lower()
    if op in ("=", "eq"):
        return expr == rhs
    if op in ("!=", "ne"):
        return expr != rhs
    if op in ("<", "lt"):
        return expr < rhs
    if op in ("<=", "le"):
        return expr <= rhs
    if op in (">", "gt"):
        return expr > rhs
    if op in (">=", "ge"):
        return expr >= rhs
    raise ValueError("Unknown operator " + str(op))


def solve_instance(json_path: str, seed: int = 0, timeout: float = 60.0,
                   workers: int = 1) -> Dict[str, Any]:
    name = os.path.basename(json_path)
    t0 = time.time()
    base = {"instance": name, "solver": "gecode", "seed": seed,
            "violations": -1, "branches": 0, "conflicts": 0, "error": ""}
    try:
        if not gecode_available():
            return {**base, "status": "ERROR", "time": time.time() - t0,
                    "error": "minizinc backend unavailable (install MiniZinc "
                             "and put it on PATH); NOT falling back"}

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        model, var_map, unsupported = build_model(data)
        if unsupported:
            return {**base, "status": "ERROR", "time": time.time() - t0,
                    "error": "unsupported constraints: "
                             + str(sorted(set(unsupported)))}

        # NO fallback: gecode or an explicit error (P0.1)
        sat = model.solve(solver="minizinc:gecode", time_limit=timeout)
        elapsed = time.time() - t0

        from cpmpy.solvers.solver_interface import ExitStatus
        exit_status = model.status().exitstatus

        if sat:
            solution = {n: int(v.value()) for n, v in var_map.items()}
            return {**base, "status": "SAT_VERIFIED", "time": elapsed,
                    "violations": 0, "solution": solution}
        if exit_status == ExitStatus.UNSATISFIABLE:
            return {**base, "status": "UNSAT", "time": elapsed, "violations": 0}
        return {**base, "status": "UNKNOWN_TIMEOUT", "time": elapsed}

    except Exception as e:  # noqa: BLE001
        return {**base, "status": "ERROR", "time": time.time() - t0,
                "error": str(e)[:300]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    rec = solve_instance(args.input, args.seed, args.timeout, args.workers)
    rec.pop("solution", None)
    print(json.dumps(rec))
    return 0 if rec["status"] != "ERROR" else 1


if __name__ == "__main__":
    sys.exit(main())
