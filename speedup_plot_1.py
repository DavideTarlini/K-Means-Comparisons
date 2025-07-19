import os
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("speedups.csv", header=None)
df = df.iloc[:, :8]
df.columns = ["threads", "k", "d", "num", "seqSimd", "par_atomic", "par", "parSimd"]

fixed_num = 1000
filtered_df = df[df["num"] == fixed_num]

filtered_df = filtered_df.sort_values("threads")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(filtered_df["threads"], filtered_df["seqSimd"], label="seqSimd", marker='o')
plt.plot(filtered_df["threads"], filtered_df["par_atomic"], label="par_atomic", marker='o')
plt.plot(filtered_df["threads"], filtered_df["par"], label="par", marker='o')
plt.plot(filtered_df["threads"], filtered_df["parSimd"], label="parSimd", marker='o')

plt.xlabel("Threads", fontsize=20)
plt.ylabel("Speedup", fontsize=20)
plt.xscale("log", base=2)
plt.xticks(fontsize=18) 
plt.yticks(fontsize=18)
plt.grid(True)
plt.tight_layout()
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "speedups_1000.pdf"))
