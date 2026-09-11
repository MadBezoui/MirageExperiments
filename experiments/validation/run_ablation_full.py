"""Full, shardable ablation over temperature (tau), polarization growth (beta),
and region topology (regions off/on) -- TODO_AIJ.md sec 2.

Tasks = TAU_GRID x BETA_GRID x {mirage, mirage_regions} x instances x seeds,
split across shards exactly like run_sweep, resumable, each task isolated in a
solve_one subprocess.

Example (shard 0 of 5):
  PYTHONPATH=. python -m experiments.validation.run_ablation_full \
      --manifest data/frozen/instances_clean.csv --seeds 0 1 2 \
      --timeout 120 --num-shards 5 --shard-id 0 --concurrency 8 \
      --out-dir results/raw/weekend_ablation
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from experiments.validation.run_sweep import discover_instances, load_done

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

TAU_GRID = [0.5, 1.0, 2.0, 4.0]
BETA_GRID = [1.02, 1.05, 1.10]
SOLVERS = ["mirage", "mirage_regions"]


def build_tasks(args):
    insts = discover_instances(args)
    tasks = []
    for tau in TAU_GRID:
        for beta in BETA_GRID:
            for solver in SOLVERS:
                for inst in insts:
                    for seed in args.seeds:
                        tasks.append((tau, beta, solver, inst, seed))
    tasks.sort()
    return [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard_id]


def key(rec):
    return (rec["tau"], rec["beta"], rec["solver"], rec["instance"], int(rec["seed"]))


def load_done_ab(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    done.add(key(json.loads(line)))
                except Exception:
                    continue
    return done


def run_task(args, tau, beta, solver, inst, seed):
    json_path = os.path.join(args.instances_dir, inst)
    base = {"tau": tau, "beta": beta, "solver": solver, "instance": inst, "seed": seed}
    if not os.path.exists(json_path) or os.path.getsize(json_path) == 0:
        return {**base, "status": "ERROR", "time": 0, "violations": -1, "error": "bad file"}
    if args.max_mb > 0 and os.path.getsize(json_path) / 1e6 > args.max_mb:
        return {**base, "status": "SKIPPED_SIZE", "time": 0, "violations": -1, "error": ""}
    cmd = [sys.executable, "-m", "experiments.validation.solve_one",
           "--input", json_path, "--solver", solver, "--seed", str(seed),
           "--timeout", str(args.timeout), "--solver-workers", "1",
           "--tau", str(tau), "--beta", str(beta),
           "--stagnation-window", str(args.stagnation_window),
           "--tuple-cap", str(args.tuple_cap)]
    env = dict(os.environ, PYTHONPATH=_ROOT)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=_ROOT,
                              env=env, timeout=args.timeout + args.grace)
        for line in reversed(proc.stdout.splitlines()):
            if line.strip().startswith("{"):
                r = json.loads(line)
                r.update(base)  # ensure config fields present
                return r
        return {**base, "status": "ERROR", "time": -1, "violations": -1,
                "error": proc.stderr[-200:]}
    except subprocess.TimeoutExpired:
        return {**base, "status": "UNKNOWN_TIMEOUT", "time": args.timeout, "violations": -1}
    except Exception as e:  # noqa: BLE001
        return {**base, "status": "ERROR", "time": -1, "violations": -1, "error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--instances-dir", default="data/normalized")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--num-shards", type=int, default=5)
    ap.add_argument("--shard-id", type=int, required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--grace", type=float, default=45.0)
    ap.add_argument("--max-mb", type=float, default=5.0,
                    help="ablation is best on tractable instances; default 5MB cap")
    ap.add_argument("--stagnation-window", type=int, default=10)
    ap.add_argument("--tuple-cap", type=int, default=200000)
    ap.add_argument("--out-dir", default="results/raw/weekend_ablation")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"ablation_shard{args.shard_id}.jsonl")
    tasks = build_tasks(args)
    done = load_done_ab(out_path)
    todo = [t for t in tasks if (t[0], t[1], t[2], t[3], int(t[4])) not in done]
    print(f"[ablation shard {args.shard_id}/{args.num_shards}] total={len(tasks)} "
          f"done={len(done)} todo={len(todo)}", flush=True)

    lock = threading.Lock()
    fh = open(out_path, "a", buffering=1)
    start, completed = time.time(), 0
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(run_task, args, *t): t for t in todo}
            for fut in as_completed(futs):
                rec = fut.result()
                with lock:
                    fh.write(json.dumps(rec) + "\n"); fh.flush()
                    completed += 1
                    if completed % 20 == 0 or completed == len(todo):
                        el = time.time() - start
                        eta = (len(todo) - completed) / (completed / el) if completed else 0
                        print(f"[ablation {args.shard_id}] {completed}/{len(todo)} "
                              f"ETA {eta/3600:.2f}h", flush=True)
    finally:
        fh.close()
    print(f"[ablation {args.shard_id}] DONE -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
