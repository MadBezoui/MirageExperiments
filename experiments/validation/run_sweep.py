"""Sharded, resumable benchmark sweep (revision protocol: 8 parallel jobs).

Each of the 8 jobs runs this with a distinct --shard-id (0..7) and the same
--num-shards 8. Tasks (instance x solver x seed) are deterministically split
across shards, so the jobs cover disjoint work and can run fully in parallel
on the same machine or on different nodes.

Robustness:
  * every task runs in its own subprocess (experiments.validation.solve_one) with
    a hard wall-clock timeout, so a hang/OOM/segfault never kills the shard;
  * results are appended to a per-shard JSONL and the run is RESUMABLE -- restart
    the same command and it skips tasks already recorded;
  * a configurable local thread pool runs several tasks concurrently.

Example (one of five jobs):
  PYTHONPATH=. python -m experiments.validation.run_sweep \
      --manifest data/frozen/instances_clean.csv \
      --solvers mirage mirage_regions hybrid ortools choco \
      --seeds 0 1 2 3 4 --timeout 300 \
      --num-shards 5 --shard-id 0 --concurrency 8 --solver-workers 4 \
      --out-dir results/raw/weekend
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def discover_instances(args):
    names = []
    if args.manifest:
        with open(args.manifest, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status", "clean") in ("clean", "relaxed"):
                    names.append(row["instance"].replace(".xml", ".json"))
    else:
        for p in sorted(glob.glob(os.path.join(args.instances_dir, "*.json"))):
            names.append(os.path.basename(p))
    # de-dup, stable order
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n); out.append(n)
    return out


def build_tasks(args):
    insts = discover_instances(args)
    tasks = []
    for inst in insts:
        for solver in args.solvers:
            for seed in args.seeds:
                tasks.append((inst, solver, seed))
    tasks.sort()  # deterministic
    # assign task i to shard i % num_shards
    return [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard_id]


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["instance"], r["solver"], int(r["seed"])))
                except Exception:
                    continue
    return done


def run_task(args, inst, solver, seed):
    json_path = os.path.join(args.instances_dir, inst)
    base = {"instance": inst, "solver": solver, "seed": seed}
    if not os.path.exists(json_path):
        return {**base, "status": "ERROR", "time": 0, "violations": -1,
                "error": "instance file missing"}
    if os.path.getsize(json_path) == 0:
        return {**base, "status": "ERROR", "time": 0, "violations": -1,
                "error": "empty instance file"}
    if args.max_mb > 0 and os.path.getsize(json_path) / 1e6 > args.max_mb:
        return {**base, "status": "SKIPPED_SIZE", "time": 0, "violations": -1,
                "error": f">{args.max_mb}MB"}

    cmd = [sys.executable, "-m", "experiments.validation.solve_one",
           "--input", json_path, "--solver", solver, "--seed", str(seed),
           "--timeout", str(args.timeout), "--solver-workers", str(args.solver_workers),
           "--tau", str(args.tau), "--beta", str(args.beta),
           "--stagnation-window", str(args.stagnation_window),
           "--tuple-cap", str(args.tuple_cap), "--warmup", str(args.warmup)]
    if args.java:
        cmd += ["--java", args.java]
    if args.choco_cp:
        cmd += ["--choco-cp", args.choco_cp]

    env = dict(os.environ, PYTHONPATH=_ROOT)
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=_ROOT,
                              env=env, timeout=args.timeout + args.grace)
        for line in reversed(proc.stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        return {**base, "status": "ERROR", "time": time.time() - t0,
                "violations": -1, "error": "no record: " + proc.stderr[-300:]}
    except subprocess.TimeoutExpired:
        return {**base, "status": "UNKNOWN_TIMEOUT", "time": args.timeout,
                "violations": -1, "error": "hard kill (subprocess timeout)"}
    except Exception as e:  # noqa: BLE001
        return {**base, "status": "ERROR", "time": time.time() - t0,
                "violations": -1, "error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--manifest", help="CSV with an 'instance' column")
    ap.add_argument("--instances-dir", default="data/normalized")
    ap.add_argument("--solvers", nargs="+",
                    default=["mirage", "mirage_regions", "hybrid", "ortools",
                             "choco", "gecode", "runcsp"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--shard-id", type=int, required=True)
    ap.add_argument("--concurrency", type=int, default=8,
                    help="tasks run concurrently within this shard")
    ap.add_argument("--solver-workers", type=int, default=4,
                    help="threads CP-SAT/hybrid may use per task")
    ap.add_argument("--grace", type=float, default=60.0,
                    help="extra seconds before hard-killing a task subprocess")
    ap.add_argument("--max-mb", type=float, default=0.0,
                    help="skip instances larger than this many MB (0 = no limit)")
    ap.add_argument("--out-dir", default="results/raw/weekend")
    # mirage hyperparameters (defaults match the ablation winner)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--beta", type=float, default=1.05)
    ap.add_argument("--stagnation-window", type=int, default=10)
    ap.add_argument("--tuple-cap", type=int, default=200000)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--java", default=None)
    ap.add_argument("--choco-cp", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"sweep_shard{args.shard_id}.jsonl")

    tasks = build_tasks(args)
    done = load_done(out_path)
    todo = [t for t in tasks if (t[0], t[1], int(t[2])) not in done]

    print(f"[shard {args.shard_id}/{args.num_shards}] total={len(tasks)} "
          f"done={len(done)} todo={len(todo)} concurrency={args.concurrency} "
          f"timeout={args.timeout}s", flush=True)

    lock = threading.Lock()
    fh = open(out_path, "a", buffering=1)
    start = time.time()
    completed = 0

    def work(t):
        return run_task(args, *t)

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(work, t): t for t in todo}
            for fut in as_completed(futs):
                rec = fut.result()
                with lock:
                    fh.write(json.dumps(rec) + "\n"); fh.flush()
                    completed += 1
                    if completed % 10 == 0 or completed == len(todo):
                        el = time.time() - start
                        rate = completed / el if el > 0 else 0
                        eta = (len(todo) - completed) / rate if rate > 0 else 0
                        print(f"[shard {args.shard_id}] {completed}/{len(todo)} "
                              f"({rate:.2f}/s, ETA {eta/3600:.2f}h) last={rec['solver']}"
                              f":{rec['instance']}={rec['status']}", flush=True)
    finally:
        fh.close()
    print(f"[shard {args.shard_id}] DONE in {(time.time()-start)/3600:.2f}h -> {out_path}",
          flush=True)


if __name__ == "__main__":
    main()
