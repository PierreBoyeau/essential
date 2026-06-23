import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as gg

# Load Fitness Data
fitness_path = "/workspace/data/calvo2020_dcas9fitness/Supp_data2_log2FC.csv"
fitness_df = pd.read_csv(fitness_path).rename(columns={"Unnamed: 0": "spacer"}).set_index("spacer")

# Filter for essential genes (T4 < -3 approx, based on previous analysis)
# Let's look at a broader range first, T4 < -2
essential = fitness_df[fitness_df["T4"] < -2].copy()

# Compute metrics
# User proposal: (T2 - T4) / T4
essential["user_score"] = (essential["T2"] - essential["T4"]) / essential["T4"]

# Alternative: Simple difference (slope)
essential["diff"] = essential["T2"] - essential["T4"]

# Alternative: Ratio T2/T4
essential["ratio"] = essential["T2"] / essential["T4"]

print(f"Number of analyzed sgRNAs: {len(essential)}")

# Summary stats
print("\n--- User Score: (T2 - T4) / T4 ---")
print(essential["user_score"].describe())

print("\n--- Correlation with T4 ---")
print(f"User Score vs T4: {essential['user_score'].corr(essential['T4']):.4f}")
print(f"Diff (T2-T4) vs T4: {essential['diff'].corr(essential['T4']):.4f}")
print(f"Ratio (T2/T4) vs T4: {essential['ratio'].corr(essential['T4']):.4f}")

# Check for edge cases
print("\n--- Edge Cases ---")
print(
    "User Score > 0 (T2 < T4, steeper early? or noise):",
    len(essential[essential["user_score"] > 0]),
)
print("User Score < -1 (T2 > 0, growth initially?):", len(essential[essential["user_score"] < -1]))

# Sample some genes
print("\n--- Sample Genes (Top 5 Late - score near -1) ---")
print(essential.sort_values("user_score").head(5)[["gene", "T2", "T4", "user_score"]])

print("\n--- Sample Genes (Top 5 Early - score near 0) ---")
print(
    essential.sort_values("user_score", ascending=False).head(5)[["gene", "T2", "T4", "user_score"]]
)

# Plotting
# We want to see if the score is just a proxy for T4 (stronger genes deplete faster?)
p = (
    gg.ggplot(essential, gg.aes(x="T4", y="user_score"))
    + gg.geom_point(alpha=0.3)
    + gg.geom_smooth(method="lm", color="red")
    + gg.labs(title="Dependence of Temporality Score on Fitness Strength", y="(T2 - T4) / T4")
)
p.save("/workspace/experiments/01012026_branching/temporality_vs_fitness.png")

# Plot T2 vs T4
p2 = (
    gg.ggplot(essential, gg.aes(x="T4", y="T2"))
    + gg.geom_point(alpha=0.3)
    + gg.geom_abline(intercept=0, slope=1, color="blue", linetype="dashed")
    + gg.labs(title="T2 vs T4 Fitness")
)
p2.save("/workspace/experiments/01012026_branching/t2_vs_t4.png")
