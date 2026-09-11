import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_memory_scaling():
    records = []
    raw_dir = "results/raw/weekend_memory"
    if os.path.exists(raw_dir):
        for fname in os.listdir(raw_dir):
            if fname.endswith(".jsonl") and "shard" in fname:
                with open(os.path.join(raw_dir, fname), "r") as f:
                    for line in f:
                        records.append(json.loads(line))
                        
    if not records:
        print("No memory records found.")
        return

    df = pd.DataFrame(records)
    out_dir = "paper/ijoc/figures"
    os.makedirs(out_dir, exist_ok=True)
    
    # The file has format: {"instance": "...", "size_mb": ..., "mirage_peak_mb": ..., "ortools_peak_mb": ...}
    # We will plot mirage and ortools scaling.
    plt.figure()
    
    if "size_mb" in df.columns and "mirage_peak_mb" in df.columns:
        valid_mirage = df[df["mirage_peak_mb"] > 0]
        if len(valid_mirage) > 0:
            plt.scatter(valid_mirage["size_mb"], valid_mirage["mirage_peak_mb"], label="mirage", alpha=0.6)
            
    if "size_mb" in df.columns and "ortools_peak_mb" in df.columns:
        valid_ortools = df[df["ortools_peak_mb"] > 0]
        if len(valid_ortools) > 0:
            plt.scatter(valid_ortools["size_mb"], valid_ortools["ortools_peak_mb"], label="ortools", alpha=0.6)
            
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Instance Size (MB)")
    plt.ylabel("Peak Resident Memory (MB)")
    plt.title("Memory Scaling Study")
    plt.legend()
    
    plt.savefig(os.path.join(out_dir, "memory_scaling.pdf"))
    print("Generated memory scaling plot.")

if __name__ == "__main__":
    generate_memory_scaling()
