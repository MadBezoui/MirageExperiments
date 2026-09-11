"""Full hybrid grid (revision P3.5 / Table 1): warm-up epochs x hint modes.

Grid = WARMUP_GRID x {value, order, both, none} on a size- and family-
stratified subset of the manifest. The (warmup=0, mode=none) cell is exactly
the CP-SAT baseline measured through the same harness, so the table is
internally controlled. Sharded/resumable like the sweep.

Example (shard 0 of 8):
  PYTHONPATH=. python -m experiments.validation.run_hybrid_grid \
      --manifest data/frozen/instances_clean.csv --seeds 0 1 2 \
      --timeout 120 --num-shards 8 --shard-id 0 --concurrency 6 \
      --subset-size 160 --out-dir results/raw/revision_hybrid_grid
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from experiments.validation.run_sweep import discover_instances

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

WARMUP_GRID = [0, 10, 30, 100]
MODES = ["value", "order", "both", "none"]


def stratified_subset(args):
    """Deterministic family-stratified, size-spread subset of the manifest."""
    insts = discover_instances(args)
    by_family = defaultdict(list)
    for n in insts:
        p = os.path.join(args.instances_dir, n)
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            continue
        if args.max_mb > 0 and os.path.getsize(p) / 1e6 > args.max_mb:
            continue
        fam = n.split("-")[0] if "-" in n else n.split(".")[0]
        by_family[fam].append((os.path.getsize(p), n))
    total = sum(len(v) for v in by_family.values())
    chosen = []
    for fam, items in sorted(by_family.items()):
        items.sort()
        k = max(1, round(args.subset_size * len(items) / max(total, 1)))
        step = len(items) / k
        chosen += [items[min(int(i * step), len(items) - 1)][1] for i in range(k)]
    # de-dup, deterministic order
    seen, out = set(), []
    for n in chosen:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out[:args.subset_size]


def build_tasks(args):
    insts = stratified_subset(args)
    tasks = []
    for warmup in WARMUP_GRID:
        for mode in MODES:
            if warmup == 0 and mode != "none":
                continue  # without warm-up there are no marginals to hint with
            for inst in insts:
                for seed in args.seeds:
                    tasks.append((warmup, mode, inst, seed))
    tasks.sort()
    return [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard_id]


def key(rec):
    return (int(rec["warmup"]), rec["hint_mode"], rec["instance"],
            int(rec["seed"]))


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    done.add(key(json.loads(line)))
                except Exception:
                    continue
    return done


def run_task(args, warmup, mode, inst, seed):
    json_path = os.path.join(args.instances_dir, inst)
    base = {"warmup": warmup, "hint_mode": mode, "instance": inst,
            "seed": seed, "solver": "hybrid"}
    cmd = [sys.executable, "-m", "experiments.validation.solve_one",
           "--input", json_path, "--solver", "hybrid", "--seed", str(seed),
           "--timeout", str(args.timeout),
           "--solver-workers", str(args.solver_workers),
           "--warmup", str(warmup), "--hint-mode", mode]
    env = dict(os.environ, PYTHONPATH=_ROOT)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=_ROOT,
                              env=env, timeout=args.timeout + args.grace)
        for line in reversed(proc.stdout.splitlines()):
            if line.strip().startswith("{"):
                r = json.loads(line)
                r.update(base)
                return r
        return {**base, "status": "ERROR", "time": -1,
                "error": proc.stderr[-200:]}
    except subprocess.TimeoutExpired:
        return {**base, "status": "UNKNOWN_TIMEOUT", "time": args.timeout}
    except Exception as e:  # noqa: BLE001
        return {**base, "status": "ERROR", "time": -1, "error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--instances-dir", default="data/normalized")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--shard-id", type=int, required=True)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--solver-workers", type=int, default=1)
    ap.add_argument("--grace", type=float, default=45.0)
    ap.add_argument("--max-mb", type=float, default=10.0)
    ap.add_argument("--subset-size", type=int, default=160)
    ap.add_argument("--out-dir", default="results/raw/revision_hybrid_grid")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"hybrid_grid_shard{args.shard_id}.jsonl")
    tasks = build_tasks(args)
    done = load_done(out_path)
    todo = [t for t in tasks if (t[0], t[1], t[2], int(t[3])) not in done]
    print(f"[hybrid-grid shard {args.shard_id}/{args.num_shards}] "
          f"total={len(tasks)} done={len(done)} todo={len(todo)}", flush=True)

    lock = threading.Lock()
    fh = open(out_path, "a", buffering=1)
    start, completed = time.time(), 0
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(run_task, args, *t): t for t in todo}
            for fut in as_completed(futs):
                rec = fut.result()
                with lock:
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    completed += 1
                    if completed % 20 == 0 or completed == len(todo):
                        el = time.time() - start
                        eta = (len(todo) - completed) / (completed / el) if completed else 0
                        print(f"[hybrid-grid {args.shard_id}] {completed}/{len(todo)} "
                              f"ETA {eta/3600:.2f}h", flush=True)
    finally:
        fh.close()
    print(f"[hybrid-grid {args.shard_id}] DONE -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
