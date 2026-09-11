
# ---------------------------------------------------------------------------
# SUPERSEDED. This script reads results/raw/weekend*, the pre-revision logs,
# which carried a fabricated `regions_added` field and broken baseline
# adapters. Those logs are never mixed into the paper's numbers and are not
# distributed with the artifact. Nothing in the manuscript depends on this
# script; it is retained only to document what the pre-revision analysis did.
# ---------------------------------------------------------------------------
import sys as _sys
_sys.exit(
    "This analysis is superseded: it reads the pre-revision 'weekend' logs, "
    "which contained fabricated fields and are not distributed. See "
    "Code/README.md."
)

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_heatmap():
    records = []
    raw_dir = "results/raw/weekend_ablation"
    if os.path.exists(raw_dir):
        for fname in os.listdir(raw_dir):
            if fname.endswith(".jsonl") and "shard" in fname:
                with open(os.path.join(raw_dir, fname), "r") as f:
                    for line in f:
                        records.append(json.loads(line))
                        
    if not records:
        print("No ablation records found.")
        return

    df = pd.DataFrame(records)
    out_dir = "paper/ijoc/figures"
    os.makedirs(out_dir, exist_ok=True)
    
    df["is_solved"] = (df["status"] == "SAT_VERIFIED").astype(int)
    pivot = df.groupby(["tau", "beta"])["is_solved"].mean().reset_index()
    
    taus = sorted(pivot["tau"].unique())
    betas = sorted(pivot["beta"].unique())
    
    data = np.zeros((len(taus), len(betas)))
    for i, t in enumerate(taus):
        for j, b in enumerate(betas):
            val = pivot[(pivot["tau"] == t) & (pivot["beta"] == b)]["is_solved"].mean()
            data[i, j] = val if not pd.isna(val) else 0
            
    fig, ax = plt.subplots()
    cax = ax.imshow(data, cmap="viridis")
    ax.set_xticks(np.arange(len(betas)))
    ax.set_yticks(np.arange(len(taus)))
    ax.set_xticklabels(betas)
    ax.set_yticklabels(taus)
    ax.set_xlabel("Beta Growth")
    ax.set_ylabel("Tau Init")
    ax.set_title("Solve Rate Heatmap")
    fig.colorbar(cax)
    
    plt.savefig(os.path.join(out_dir, "ablation_heatmap.pdf"))
    print("Generated ablation heatmap.")

if __name__ == "__main__":
    generate_heatmap()
