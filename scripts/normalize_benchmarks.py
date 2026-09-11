import argparse
import os
import glob
import json
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.mirage.xcsp3_to_core import parse_and_convert

import subprocess
import tempfile

def process_file(f, output_dir):
    basename = os.path.basename(f)
    json_name = basename.replace(".xml", ".json")
    out_path = os.path.join(output_dir, json_name)
    
    if os.path.exists(out_path):
        return {
            "instance": basename,
            "family": "unknown_cached",
            "status": "clean",
            "relaxed": False,
            "reason": "cached"
        }
        
    try:
        # We use a subprocess to strictly kill it if it hangs (e.g. combinatorial explosion in parsing)
        cmd = [sys.executable, "-c", f"""
import sys, json, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.mirage.xcsp3_to_core import parse_and_convert
try:
    core_inst = parse_and_convert(r"{f}")
    if core_inst.metadata["parser_status"] in ("clean", "relaxed"):
        with open(r"{out_path}", "w") as outf:
            json.dump(core_inst.to_normalized_json(), outf, indent=2)
    print(json.dumps(core_inst.metadata))
except Exception as e:
    print(json.dumps({{"parser_status": "error", "unsupported_reason": str(e), "family": "error", "relaxed": False}}))
"""]
        
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if proc.returncode != 0:
            return {
                "instance": basename,
                "family": "error",
                "status": "error",
                "relaxed": False,
                "reason": f"Crash: {proc.stderr[:100]}"
            }
            
        meta = json.loads(proc.stdout.strip() or "{}")
        status = meta.get("parser_status", "error")
        reason = meta.get("unsupported_reason", "")
        
        return {
            "instance": basename,
            "family": meta.get("family", "unknown"),
            "status": status,
            "relaxed": meta.get("relaxed", False),
            "reason": reason
        }
        
    except subprocess.TimeoutExpired:
        return {
            "instance": basename,
            "family": "error",
            "status": "timeout",
            "relaxed": False,
            "reason": "Strict 10s parsing timeout exceeded"
        }
    except Exception as e:
        return {
            "instance": basename,
            "family": "error",
            "status": "error",
            "relaxed": False,
            "reason": str(e)
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--jobs", type=int, default=15)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)

    files = glob.glob(os.path.join(args.input, "**", "*.xml"), recursive=True)
    
    records = []
    
    print(f"Starting normalization of {len(files)} files with {args.jobs} jobs...")
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        future_to_f = {executor.submit(process_file, f, args.output): f for f in files}
        for i, future in enumerate(as_completed(future_to_f)):
            f = future_to_f[future]
            try:
                record = future.result()
                records.append(record)
                if i % 10 == 0 or record["status"] == "error":
                    print(f"[{i+1}/{len(files)}] Processed {record['instance']} -> {record['status']}")
            except Exception as e:
                print(f"Failed to process {f}: {e}")
        
    with open(args.report, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instance", "family", "status", "relaxed", "reason"])
        writer.writeheader()
        writer.writerows(records)
        
    print(f"Normalization complete. Census written to {args.report}")

if __name__ == "__main__":
    main()
