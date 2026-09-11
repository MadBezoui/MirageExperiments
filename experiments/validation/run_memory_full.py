"""Shardable peak-memory profiling: MIRAGE-R vs OR-Tools CP-SAT.

Revision fixes (P3.3):
  * CROSS-PLATFORM: peak RSS is measured by the parent polling the child's
    ``psutil.Process().memory_info().rss`` at 25 ms (the old version used the
    Unix-only ``resource`` module and could never run on Windows). On Windows
    we additionally read ``memory_info().peak_wset`` when available, which is
    the OS-maintained true peak.
  * Coverage: run on every manifest instance under --max-mb (>= 30 instances
    spanning the claimed order of magnitude); the aggregator fits log-log
    slopes instead of claiming "grows exponentially".
  * Same-language caveat: both processes are Python drivers; CP-SAT's core is
    C++ but its clause database lives in the measured process. The paper's
    caption must carry this disclaimer (the aggregator emits it).

Example (shard 0 of 8):
  PYTHONPATH=. python -m experiments.validation.run_memory_full \
      --manifest data/frozen/instances_clean.csv --timeout 120 \
      --num-shards 8 --shard-id 0 --concurrency 2 --max-mb 20 \
      --out-dir results/raw/revision_memory
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

from experiments.validation.run_sweep import discover_instances

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

CHILD = r'''
import sys, json, time
path, solver, timeout = sys.argv[1], sys.argv[2], float(sys.argv[3])
sys.path.insert(0, ".")
t0 = time.time()
try:
    if solver == "mirage":
        from experiments.validation.loader import load
        from src.mirage.mirage_solver import MirageSolver
        from src.mirage.annealing import Hyperparameters
        import numpy as np
        inst, _ = load(path); np.random.seed(0)
        res = MirageSolver(inst, Hyperparameters(timeout=timeout, max_epochs=10**9,
                            tau_init=2.0, beta_growth=1.05)).run()
        status = res["status"]
    else:
        sys.path.insert(0, "baselines/ortools")
        from run_ortools import solve_instance
        status = solve_instance(path, seed=0, timeout=timeout, workers=1)["status"]
    print(json.dumps({"status": status, "time": time.time()-t0}))
except Exception as e:
    print(json.dumps({"status": "ERROR", "time": time.time()-t0, "error": str(e)[:200]}))
'''


def _peak_of_tree(proc) -> float:
    """Current RSS of the process tree, in MB. On Windows also consults the
    OS-maintained peak working set."""
    import psutil
    total, peak_ws = 0, 0
    try:
        procs = [proc] + proc.children(recursive=True)
    except psutil.Error:
        return 0.0
    for p in procs:
        try:
            mi = p.memory_info()
            total += mi.rss
            peak_ws += getattr(mi, "peak_wset", 0)
        except psutil.Error:
            continue
    return max(total, peak_ws) / 1e6


def measure(args, inst_path: str, solver: str):
    """Spawn the child and poll its RSS to obtain peak memory (MB)."""
    import psutil
    try:
        child = subprocess.Popen(
            [sys.executable, "-c", CHILD, inst_path, solver, str(args.timeout)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            cwd=_ROOT, env=dict(os.environ, PYTHONPATH=_ROOT))
        ps = psutil.Process(child.pid)
        peak = 0.0
        deadline = time.time() + args.timeout + args.grace
        while child.poll() is None:
            peak = max(peak, _peak_of_tree(ps))
            if time.time() > deadline:
                for p in ([ps] + ps.children(recursive=True)):
                    try:
                        p.kill()
                    except psutil.Error:
                        pass
                return {"status": "UNKNOWN_TIMEOUT", "peak_mb": peak,
                        "time": args.timeout}
            time.sleep(0.025)
        out, _ = child.communicate(timeout=10)
        for line in reversed((out or "").splitlines()):
            if line.strip().startswith("{"):
                rec = json.loads(line)
                rec["peak_mb"] = round(peak, 2)
                return rec
        return {"status": "ERROR", "peak_mb": peak, "time": -1,
                "error": "no record from child"}
    except Exception as e:  # noqa: BLE001
        return {"status": "ERROR", "peak_mb": -1, "time": -1,
                "error": str(e)[:200]}


def build_tasks(args):
    insts = discover_instances(args)
    insts.sort()
    return [n for i, n in enumerate(insts) if i % args.num_shards == args.shard_id]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--instances-dir", default="data/normalized")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--shard-id", type=int, required=True)
    ap.add_argument("--concurrency", type=int, default=2,
                    help="keep LOW: concurrent tasks share RAM and would bias "
                         "peak-RSS readings upward under memory pressure")
    ap.add_argument("--grace", type=float, default=45.0)
    ap.add_argument("--max-mb", type=float, default=20.0)
    ap.add_argument("--out-dir", default="results/raw/revision_memory")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"memory_shard{args.shard_id}.jsonl")
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                done.add(json.loads(line)["instance"])
            except Exception:
                pass
    tasks = [t for t in build_tasks(args) if t not in done]
    print(f"[memory shard {args.shard_id}] todo={len(tasks)}", flush=True)

    lock = threading.Lock()
    fh = open(out_path, "a", buffering=1)

    def work(inst):
        path = os.path.join(args.instances_dir, inst)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return {"instance": inst, "status": "ERROR", "error": "bad file"}
        if args.max_mb > 0 and os.path.getsize(path) / 1e6 > args.max_mb:
            return {"instance": inst, "status": "SKIPPED_SIZE"}
        m = measure(args, path, "mirage")
        o = measure(args, path, "ortools")
        return {"instance": inst, "status": "OK",
                "size_mb": round(os.path.getsize(path) / 1e6, 4),
                "mirage_peak_mb": m.get("peak_mb", -1),
                "ortools_peak_mb": o.get("peak_mb", -1),
                "mirage_status": m.get("status"),
                "ortools_status": o.get("status")}

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(work, t): t for t in tasks}
            for i, fut in enumerate(as_completed(futs), 1):
                rec = fut.result()
                with lock:
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                if i % 10 == 0 or i == len(tasks):
                    print(f"[memory {args.shard_id}] {i}/{len(tasks)}", flush=True)
    finally:
        fh.close()
    print(f"[memory {args.shard_id}] DONE -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
