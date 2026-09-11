"""Convergence-rate validation (TODO_AIJ.md sec 3).

Runs MIRAGE-R on a size-stratified set of instances and records the per-decode
running-minimum violation trajectory, which the aggregator fits against the
O(sqrt(log m / T)) bound of Theorem 1 and renders as a log-log convergence plot.

This phase is light; run it once (e.g. in job 0) after the heavy sweep:
  PYTHONPATH=. python -m experiments.validation.run_theory \
      --manifest data/frozen/instances_clean.csv --sample 30 --timeout 120 \
      --out-dir results/raw/weekend_theory
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np

from experiments.validation.loader import load
from experiments.validation.run_sweep import discover_instances
from src.mirage.mirage_solver import MirageSolver
from src.mirage.annealing import Hyperparameters


def pick(args):
    insts = discover_instances(args)
    paths = []
    for n in insts:
        p = os.path.join(args.instances_dir, n)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            paths.append((os.path.getsize(p), n))
    paths = [pn for pn in paths if pn[0] / 1e6 <= args.max_mb]
    paths.sort()
    if len(paths) <= args.sample:
        return [n for _, n in paths]
    step = len(paths) / args.sample
    return [paths[int(i * step)][1] for i in range(args.sample)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--instances-dir", default="data/normalized")
    ap.add_argument("--sample", type=int, default=30)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--max-mb", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/raw/weekend_theory")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "trajectories.jsonl")
    names = pick(args)
    print(f"[theory] {len(names)} instances", flush=True)

    with open(out_path, "w", buffering=1) as fh:
        for k, n in enumerate(names, 1):
            try:
                inst, _ = load(os.path.join(args.instances_dir, n))
                m = max(len(v.domain) for v in inst.variables.values())
                np.random.seed(args.seed)
                hp = Hyperparameters(timeout=args.timeout, max_epochs=10**9,
                                     tau_init=2.0, beta_growth=1.05)
                solver = MirageSolver(inst, hp)
                res = solver.run()
                hist = list(solver.violation_history)
                running_min, best = [], float("inf")
                for v in hist:
                    best = min(best, v)
                    running_min.append(best)
                rec = {
                    "instance": n, "status": res["status"], "m": int(m),
                    "n_vars": len(inst.variables), "decode_freq": hp.decoding_frequency,
                    "violations_trajectory": hist,
                    "running_min_trajectory": running_min,
                    "final_violations": res.get("violations", -1),
                }
                fh.write(json.dumps(rec) + "\n")
                print(f"[theory] {k}/{len(names)} {n} m={m} -> {res['status']} "
                      f"(len {len(hist)})", flush=True)
            except Exception as e:  # noqa: BLE001
                fh.write(json.dumps({"instance": n, "status": "ERROR", "error": str(e)}) + "\n")
    print(f"[theory] DONE -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
