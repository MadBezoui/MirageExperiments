import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_choco_solver(json_path: str, seed: int, timeout: float):
    java_bin = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".local", "java", "jdk-17.0.11+9", "bin", "java.exe"))
    if not os.path.exists(java_bin):
        java_bin = "java"
    cmd = [java_bin, "-cp", "baselines/choco/target/classes;baselines/choco/target/dependency/*", 
           "org.mirager.choco.ChocoBatchRunner", "--input", json_path, "--seed", str(seed), "--timeout", str(int(timeout))]
    start = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 5)
        elapsed = time.time() - start
        for line in proc.stdout.splitlines():
            if line.startswith("{"):
                return json.loads(line)
        return {"status": "ERROR", "error": "No JSON output from Choco", "time": elapsed}
    except subprocess.TimeoutExpired:
        return {"status": "UNKNOWN_TIMEOUT", "time": timeout}
    except Exception as e:
        return {"status": "ERROR", "error": str(e), "time": timeout}

def worker(rec):
    json_path = os.path.join("data/normalized", rec["instance"])
    res = run_choco_solver(json_path, rec["seed"], 300)
    rec["time"] = res.get("runtime_sec", res.get("time", 300.0))
    return rec

def main():
    records = []
    choco_to_rerun = []
    
    with open("results/raw/main_sweep.jsonl", "r") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            if rec["solver"] == "choco" and rec["status"] in ("SAT_VERIFIED", "UNSAT_PROVED"):
                choco_to_rerun.append(rec)
            else:
                records.append(rec)
                
    print(f"Rerunning {len(choco_to_rerun)} Choco jobs to recover runtimes...")
    
    rerun_records = []
    max_workers = os.cpu_count() or 4
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, r) for r in choco_to_rerun]
        for idx, future in enumerate(as_completed(futures)):
            res = future.result()
            rerun_records.append(res)
            if (idx + 1) % 50 == 0:
                print(f"[{idx+1}/{len(choco_to_rerun)}] Finished")
                
    records.extend(rerun_records)
    
    with open("results/raw/main_sweep.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
            
if __name__ == "__main__":
    main()
