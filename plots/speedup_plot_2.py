import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv("speedups.csv", header=0)
df.columns = df.columns.str.strip()
df = df[df["dim"] == 8]
df = df[df["th"] == 8]

df_mean = df.melt(id_vars=["num_points"],
                  value_vars=["par_single_mean", "par_single_soa_mean", "par_single_mix_mean"],
                  var_name="Comparison", value_name="Speedup")

df_std = df.melt(id_vars=["num_points"],
                 value_vars=["par_single_std", "par_single_soa_std", "par_single_mix_std"],
                 var_name="Comparison", value_name="Std")

df_std["Comparison"] = df_std["Comparison"].str.replace("_std", "_mean")
df_combined = pd.merge(df_mean, df_std, on=["num_points", "Comparison"])

label_map = {
    "par_single_mean": "seq vs par",
    "par_single_soa_mean": "seq + soa vs par + soa",
    "par_single_mix_mean": "seq vs par + soa"
}

df_combined["Comparison"] = df_combined["Comparison"].map(label_map)

plt.figure(figsize=(8, 6))
sns.set_theme(style="whitegrid")

for label, group in df_combined.groupby("Comparison"):
    plt.errorbar(group["num_points"], group["Speedup"], yerr=group["Std"],
                 label=label, marker="o", capsize=3, linestyle='-')

plt.xlabel("Dataset Size")
plt.ylabel("Speedup (8 Threads)")
plt.xscale("log")
plt.legend()
plt.tight_layout()
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "best_speedups.pdf"))
