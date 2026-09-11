"""Compute per-instance structural features for the stratified analysis (P3.1).

For every normalized instance, extract: family, #variables, #constraints,
constraint-type mix, max/mean arity, max/mean domain size, constraint density
(constraints per variable), table-tuple mass, file size. Output a single CSV
consumed by the aggregator to stratify Table 2 by constraint family, arity,
density and domain size.

Usage:
    PYTHONPATH=. python -m experiments.validation.compute_features \
        --instances-dir data/normalized --out data/frozen/instance_features.csv
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import os

TYPES = ("table", "sum", "count", "allDifferent", "element", "crosswordOverlap")


def family(inst: str) -> str:
    base = inst.replace(".json", "")
    return base.split("-")[0] if "-" in base else base


def features(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    name = os.path.basename(path)
    n_vars = len(data["variables"])
    dom_sizes = [len(v["domain"]) for v in data["variables"]] or [0]
    cons = data["constraints"]
    arities = [len(c.get("scope", [])) for c in cons] or [0]
    type_counts = {t: 0 for t in TYPES}
    n_tuples = 0
    for c in cons:
        t = c.get("type", "?")
        if t in type_counts:
            type_counts[t] += 1
        if t == "table":
            n_tuples += len(c.get("tuples", []))
    dominant = max(type_counts, key=type_counts.get) if cons else "none"
    binary_only = all(a <= 2 for a in arities) and all(
        c.get("type") == "table" for c in cons)
    return {
        "instance": name,
        "family": family(name),
        "n_vars": n_vars,
        "n_constraints": len(cons),
        "density": round(len(cons) / max(n_vars, 1), 4),
        "max_arity": max(arities),
        "mean_arity": round(sum(arities) / max(len(arities), 1), 3),
        "max_domain": max(dom_sizes),
        "mean_domain": round(sum(dom_sizes) / max(len(dom_sizes), 1), 3),
        "n_table_tuples": n_tuples,
        "dominant_type": dominant,
        "runcsp_fragment": int(binary_only),
        "size_mb": round(os.path.getsize(path) / 1e6, 4),
        **{f"n_{t}": type_counts[t] for t in TYPES},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances-dir", default="data/normalized")
    ap.add_argument("--out", default="data/frozen/instance_features.csv")
    args = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(os.path.join(args.instances_dir, "*.json"))):
        try:
            if os.path.getsize(p) == 0:
                continue
            rows.append(features(p))
        except Exception as e:  # noqa: BLE001
            print(f"[features] SKIP {p}: {e}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[features] wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
