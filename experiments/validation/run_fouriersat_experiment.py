import os
import glob
import pandas as pd


def main():
    # 1. Define instances
    instance_dir = "data/normalized"
    boolean_instances = sorted(glob.glob(os.path.join(instance_dir, "bool_n50_*.json")))
    if not boolean_instances:
        print("No boolean instances found. Please run generate_boolean_csps.py first.")
        return
        
    print(f"Found {len(boolean_instances)} boolean instances.")
    
    # 2. Solvers to compare
    solvers = ["mirage", "mirage_regions", "fouriersat"]
    seeds = [0, 1, 2, 3, 4]
    
    # 3. Create tasks list
    tasks = []
    for inst in boolean_instances:
        # Extract features from filename to keep track
        # bool_n50_k3_d2.0_s42.json
        basename = os.path.basename(inst)
        parts = basename.replace(".json", "").split("_")
        k = int(parts[2][1:])
        d = float(parts[3][1:])
        
        for s in solvers:
            for seed in seeds:
                tasks.append({
                    "input": inst,
                    "solver": s,
                    "seed": seed,
                    "timeout": 300, # 5 min budget as per P0 protocol
                    # extract other properties
                    "k": k,
                    "d": d
                })
                
    # 4. Run tasks via the existing harness parallel runner
    print(f"Running {len(tasks)} tasks...")
    # Actually we can just spawn them, but it's better to use `run_all_tasks` if it's available.
    # `run_all_tasks` expects a list of task dicts and an output shard prefix.
    # We will write a custom minimal runner for this specific experiment.
    
    out_dir = "results/raw/fouriersat_sweep"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "fouriersat_results.jsonl")
    
    # We will reuse the same logic as the main sweep, but just call solve_one directly
    # Wait, solve_one can be called as a subprocess to be safe against segfaults.
    import json
    import subprocess
    import sys
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # Load already done
    done = set()
    if os.path.exists(out_file):
        with open(out_file, "r") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done.add((rec["instance"], rec["solver"], rec["seed"]))
                    
    def worker(task):
        inst_name = os.path.basename(task["input"])
        if (inst_name, task["solver"], task["seed"]) in done:
            return None
            
        cmd = [
            sys.executable, "-m", "experiments.validation.solve_one",
            "--input", task["input"],
            "--solver", task["solver"],
            "--seed", str(task["seed"]),
            "--timeout", str(task["timeout"]),
            "--solver-workers", "1"
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=task["timeout"] + 30)
            for line in reversed(proc.stdout.splitlines()):
                if line.strip().startswith("{"):
                    rec = json.loads(line)
                    rec["k"] = task["k"]
                    rec["d"] = task["d"]
                    return rec
        except Exception as e:
            return {
                "instance": inst_name,
                "solver": task["solver"],
                "seed": task["seed"],
                "status": "ERROR",
                "error": str(e),
                "k": task["k"],
                "d": task["d"]
            }
        return None

    with open(out_file, "a") as f, ThreadPoolExecutor(max_workers=26) as pool:
        futures = {pool.submit(worker, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures)):
            res = fut.result()
            if res is not None:
                f.write(json.dumps(res) + "\n")
                f.flush()
            if (i+1) % 50 == 0:
                print(f"Progress: {i+1} / {len(tasks)}")
                
    print("Experiment finished.")

if __name__ == "__main__":
    main()
