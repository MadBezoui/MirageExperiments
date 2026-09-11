import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze():
    records = []
    raw_dir = "results/raw/weekend"
    if os.path.exists(raw_dir):
        for fname in os.listdir(raw_dir):
            if fname.endswith(".jsonl") and "shard" in fname:
                with open(os.path.join(raw_dir, fname), "r") as f:
                    for line in f:
                        records.append(json.loads(line))
                        
    if not records:
        print("No records found. Exiting.")
        return
        
    df = pd.DataFrame(records)
    df_reg = df[df["solver"] == "mirage_regions"].copy()
    
    out_dir = "paper/ijoc/tables"
    os.makedirs(out_dir, exist_ok=True)
    if "regions_added" in df_reg.columns and len(df_reg) > 0:
        summary = df_reg.groupby("status")["regions_added"].mean().reset_index()
        summary.to_csv(os.path.join(out_dir, "region_summary.csv"), index=False)
        
        with open(os.path.join(out_dir, "region_summary.tex"), "w") as f:
            f.write("\\begin{tabular}{cc}\n\\toprule\nStatus & Avg Regions \\\\\n\\midrule\n")
            for _, row in summary.iterrows():
                status = row['status'].replace('_', '\\_')
                f.write(f"{status} & {row['regions_added']:.2f} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")
    
    fig_dir = "paper/ijoc/figures"
    os.makedirs(fig_dir, exist_ok=True)
    plt.figure()
    
    if "regions_added" in df_reg.columns and len(df_reg) > 0:
        plt.hist(df_reg["regions_added"].fillna(0), bins=20)
        plt.xlabel("Regions Added")
        plt.ylabel("Count")
        plt.title("Region Materialization Frequency")
        plt.savefig(os.path.join(fig_dir, "region_analysis.pdf"))
        
    print("Region analysis complete.")

if __name__ == "__main__":
    analyze()
