"""OR-Tools CP-SAT baseline runner for MIRAGE-R normalized JSON instances.

Reads a normalized instance (the same data/normalized/*.json files consumed by
the MIRAGE-R solver), reconstructs the model in CP-SAT, and reports a record
schema-compatible with the validation harness.

Supported constraint types (matching src/mirage/csp_core.py):
    table (positive/negative), sum, count, allDifferent, element.

On SAT the witness is extracted and returned so the harness can re-check it
with the independent verifier (adapter-equivalence guard, revision P0.1).

Usage:
    python baselines/ortools/run_ortools.py --input data/normalized/Foo.json \
        --seed 0 --timeout 60 [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

from ortools.sat.python import cp_model


def build_model(data: Dict[str, Any]):
    """Translate a normalized CSP instance into a CP-SAT model.

    Returns (model, var_map, unsupported).
    """
    model = cp_model.CpModel()

    var_map: Dict[str, Any] = {}
    var_domain: Dict[str, List[int]] = {}
    for v in data["variables"]:
        vals = [int(x) for x in v["domain"]]
        var_domain[v["name"]] = vals
        dom = cp_model.Domain.FromValues(vals)
        var_map[v["name"]] = model.NewIntVarFromDomain(dom, v["name"])

    unsupported: List[str] = []

    for c in data["constraints"]:
        ctype = c["type"]
        scope = c.get("scope", [])

        if ctype == "table":
            tuples = [tuple(int(x) for x in t) for t in c["tuples"]]
            svars = [var_map[s] for s in scope]
            if c.get("positive", True):
                model.AddAllowedAssignments(svars, tuples)
            else:
                model.AddForbiddenAssignments(svars, tuples)

        elif ctype == "sum":
            expr = sum(int(co) * var_map[s] for co, s in zip(c["coeffs"], scope))
            _add_relop(model, expr, c["operator"], int(c["rhs"]))

        elif ctype == "count":
            values = set(int(x) for x in c["values"])
            bools = []
            for s in scope:
                dom_vals = var_domain[s]
                in_vals = sorted(v for v in dom_vals if v in values)
                out_vals = sorted(v for v in dom_vals if v not in values)
                b = model.NewBoolVar("cnt_" + str(c["id"]) + "_" + str(s))
                if not in_vals:
                    model.Add(b == 0)
                elif not out_vals:
                    model.Add(b == 1)
                else:
                    model.AddLinearExpressionInDomain(
                        var_map[s], cp_model.Domain.FromValues(in_vals)
                    ).OnlyEnforceIf(b)
                    model.AddLinearExpressionInDomain(
                        var_map[s], cp_model.Domain.FromValues(out_vals)
                    ).OnlyEnforceIf(b.Not())
                bools.append(b)
            _add_relop(model, sum(bools), c["operator"], int(c["rhs"]))

        elif ctype == "allDifferent":
            model.AddAllDifferent([var_map[s] for s in scope])

        elif ctype == "element":
            idx = var_map[c["index_var"]]
            target = var_map[c["value_var"]]
            if c.get("array_vars"):
                model.AddElement(idx, [var_map[a] for a in c["array_vars"]], target)
            else:
                consts = [int(x) for x in c["array_values"]]
                model.AddElement(idx, consts, target)

        else:
            unsupported.append(ctype)

    return model, var_map, unsupported


def _add_relop(model, expr, op: str, rhs: int):
    op = op.lower()
    if op in ("=", "eq"):
        model.Add(expr == rhs)
    elif op in ("!=", "ne"):
        model.Add(expr != rhs)
    elif op in ("<", "lt"):
        model.Add(expr < rhs)
    elif op in ("<=", "le"):
        model.Add(expr <= rhs)
    elif op in (">", "gt"):
        model.Add(expr > rhs)
    elif op in (">=", "ge"):
        model.Add(expr >= rhs)
    else:
        raise ValueError("Unknown operator " + str(op))


def solve_instance(json_path: str, seed: int = 0, timeout: float = 60.0,
                   workers: int = 8) -> Dict[str, Any]:
    name = os.path.basename(json_path)
    t0 = time.time()
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        model, var_map, unsupported = build_model(data)
        if unsupported:
            return {
                "instance": name, "solver": "ortools", "seed": seed,
                "status": "ERROR", "time": time.time() - t0, "violations": -1,
                "error": "unsupported constraints: " + str(sorted(set(unsupported))),
            }

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(timeout)
        solver.parameters.random_seed = int(seed)
        solver.parameters.num_search_workers = int(workers)

        status = solver.Solve(model)
        elapsed = time.time() - t0

        solution = None
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            out_status, violations = "SAT_VERIFIED", 0
            # extract the witness so the harness can re-check it with the
            # independent verifier (adapter-equivalence guard, revision P0.1)
            solution = {n: int(solver.Value(v)) for n, v in var_map.items()}
        elif status == cp_model.INFEASIBLE:
            out_status, violations = "UNSAT", 0
        else:
            out_status, violations = "UNKNOWN_TIMEOUT", -1

        rec = {
            "instance": name, "solver": "ortools", "seed": seed,
            "status": out_status, "time": elapsed, "violations": violations,
            "error": "", "branches": solver.NumBranches(),
            "conflicts": solver.NumConflicts(),
        }
        if solution is not None:
            rec["solution"] = solution
        return rec
    except Exception as e:  # noqa: BLE001
        return {
            "instance": name, "solver": "ortools", "seed": seed,
            "status": "ERROR", "time": time.time() - t0, "violations": -1,
            "error": str(e),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    rec = solve_instance(args.input, args.seed, args.timeout, args.workers)
    rec.pop("solution", None)
    print(json.dumps(rec))
    return 0 if rec["status"] != "ERROR" else 1


if __name__ == "__main__":
    sys.exit(main())
