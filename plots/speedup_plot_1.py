import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("speedups.csv", header=0)
df.columns = df.columns.str.strip() 
df = df[df["dim"] == 8]

df = df[[
    "th", "num_points",
    "par_single_mean", "par_single_soa_mean", "par_single_mix_mean",
    "par_single_std", "par_single_soa_std", "par_single_mix_std"
]]

df_mean = df.melt(id_vars=["th", "num_points"],
                  value_vars=["par_single_mean", "par_single_soa_mean", "par_single_mix_mean"],
                  var_name="Comparison", value_name="Speedup")

df_std = df.melt(id_vars=["th", "num_points"],
                 value_vars=["par_single_std", "par_single_soa_std", "par_single_mix_std"],
                 var_name="Comparison", value_name="Std")

df_std["Comparison"] = df_std["Comparison"].str.replace("_std", "_mean")

df_combined = pd.merge(df_mean, df_std, on=["th", "num_points", "Comparison"])
label_map = {
    "par_single_mean": "seq vs par",
    "par_single_soa_mean": "seq + soa vs par + soa",
    "par_single_mix_mean": "seq vs par + soa"
}
df_combined["Comparison"] = df_combined["Comparison"].map(label_map)

y_min = df_combined["Speedup"].min() - df_combined["Std"].max()
y_max = df_combined["Speedup"].max() + df_combined["Std"].max()
y_limits = (max(0, y_min), y_max * 1.05)

output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

for np_value in sorted(df_combined["num_points"].unique()):
    df_subset = df_combined[df_combined["num_points"] == np_value]
    filename = f"speedup_{np_value}.pdf"
    plt.figure(figsize=(8, 5))
    for name, group in df_subset.groupby("Comparison"):
        plt.errorbar(group["th"], group["Speedup"], yerr=group["Std"],
                     label=name, marker="o", capsize=3, linestyle='-')
    
    plt.xlabel("Threads")
    plt.ylabel("Speedup")
    plt.ylim(y_limits)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()