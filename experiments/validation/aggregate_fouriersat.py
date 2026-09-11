import pandas as pd
import json
import numpy as np
from scipy.stats import wilcoxon

def load_results(jsonl_file):
    records = []
    with open(jsonl_file, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    
    # Calculate PAR2 per instance-solver (over 5 seeds)
    # PAR2: time if solved, else 2 * timeout
    df["solved"] = df["status"] == "SAT_VERIFIED"
    df["par2_time"] = df.apply(lambda row: row["time"] if row["solved"] else 600.0, axis=1)
    
    # Group by instance and solver
    agg_df = df.groupby(["instance", "solver", "k", "d"]).agg(
        solve_rate=("solved", "mean"),
        par2=("par2_time", "mean")
    ).reset_index()
    
    return agg_df

def main():
    df = load_results("results/raw/fouriersat_sweep/fouriersat_results.jsonl")
    
    # 1. Solve rate table stratified by (arity x density)
    # Solvers as columns
    pivot_solve = df.pivot_table(index=["k", "d"], columns="solver", values="solve_rate", aggfunc="mean").reset_index()
    
    # Output to CSV for latex rendering
    pivot_solve.to_csv("paper/ijoc/tables/fouriersat_solve_rate.csv", index=False)
    
    # Generate a LaTeX table
    table_tex = pivot_solve.to_latex(index=False, float_format="%.2f", na_rep="NaN")
    table_tex = table_tex.replace("mirage_regions", "mirage\\_regions")
    with open("paper/ijoc/tables/fouriersat_solve_rate.tex", "w") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\TABLE{Solve rate on Boolean CSPs vs FourierSAT.\\label{tab:fouriersat}}\n")
        f.write("{" + table_tex + "}\n")
        f.write("{Mean solve rate (5 seeds) across arity $k$ and density $d$.}\n")
        f.write("\\end{table}\n")
        
    # 2. Wilcoxon signed-rank test on per-instance PAR2
    # We compare MIRAGE-R vs FourierSAT, and MIRAGE-R+regions vs FourierSAT
    # Need to group by instance
    inst_df = df.pivot(index="instance", columns="solver", values="par2").dropna()
    
    results = []
    from statsmodels.stats.multitest import multipletests
    
    for mirage_variant in ["mirage", "mirage_regions"]:
        if mirage_variant in inst_df.columns and "fouriersat" in inst_df.columns:
            stat, p_val = wilcoxon(inst_df[mirage_variant], inst_df["fouriersat"])
            results.append({
                "comparison": f"{mirage_variant} vs fouriersat",
                "p_value": p_val
            })
            
    if results:
        res_df = pd.DataFrame(results)
        # Holm correction
        res_df["p_adj"], _, _, _ = multipletests(res_df["p_value"], method="holm")
        res_df.to_csv("paper/ijoc/tables/fouriersat_wilcoxon.csv", index=False)
        print("Wilcoxon results:")
        print(res_df)
        
    print("Aggregation complete.")

if __name__ == "__main__":
    main()
