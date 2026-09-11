"""Reconstruct a full CSPInstance (ALL constraint types) from normalized JSON.

The verifier checks every constraint type, so MIRAGE-R must be built with the
complete constraint set (not only table constraints) to remain sound. This module
is shared by every validation runner.
"""
from __future__ import annotations
import json
from typing import Any, Dict, Tuple

from src.mirage.csp_core import (
    CSPInstance, Variable, TableConstraint, SumConstraint, CountConstraint,
    ElementConstraint, AllDifferentConstraint, CrosswordOverlapConstraint,
)


def reconstruct(data: Dict[str, Any]) -> CSPInstance:
    variables = {v["name"]: Variable(v["name"], v["domain"], v.get("kind", "int"))
                 for v in data["variables"]}
    cons = []
    for c in data["constraints"]:
        t = c["type"]
        if t == "table":
            cons.append(TableConstraint(c["id"], c["scope"], c["positive"],
                                        [tuple(x) for x in c["tuples"]]))
        elif t == "sum":
            cons.append(SumConstraint(c["id"], c["scope"], list(c["coeffs"]),
                                      c["operator"], c["rhs"]))
        elif t == "count":
            cons.append(CountConstraint(c["id"], c["scope"], list(c["values"]),
                                        c["operator"], c["rhs"]))
        elif t == "allDifferent":
            cons.append(AllDifferentConstraint(c["id"], c["scope"]))
        elif t == "element":
            cons.append(ElementConstraint(
                c["id"], c["scope"], c["index_var"], c["value_var"],
                c.get("array_vars", []), c.get("array_values", [])))
        elif t == "crosswordOverlap":
            cons.append(CrosswordOverlapConstraint(
                c["id"], c["scope"], c["pos_i"], c["pos_j"], c.get("alphabet", [])))
        else:
            raise ValueError(f"unknown constraint type: {t}")
    return CSPInstance(data["name"], data["source_path"], variables, cons,
                       data.get("metadata", {}))


def load(path: str) -> Tuple[CSPInstance, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return reconstruct(data), data
