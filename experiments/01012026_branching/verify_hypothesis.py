# ---------------------------------------------------------
# Verification of Dosage Insensitive vs Sensitive Essential Genes
# ---------------------------------------------------------

# Filter for essential genes (T4 < -3)
essential_df = merged_df[merged_df["T4"] < -3].copy()
print(f"Number of essential sgRNAs (T4 < -3) with dosage data: {len(essential_df)}")

# Define sensitive vs insensitive based on dosage difference
# Sensitive: 0MM causes defect, 2MM rescues (large negative diff)
sensitive = essential_df[essential_df["diff"] < -0.2]

# Insensitive: 0MM and 2MM are similar (diff near 0)
insensitive = essential_df[(essential_df["diff"] > -0.1) & (essential_df["diff"] < 0.1)]

print(f"\nSensitive (diff < -0.2): {len(sensitive)}")
print(f"Mean value_partial (2MM growth) for sensitive: {sensitive['value_partial'].mean():.4f}")
print(f"Mean value_full (0MM growth) for sensitive: {sensitive['value_full'].mean():.4f}")

print(f"\nInsensitive (|diff| < 0.1): {len(insensitive)}")
print(f"Mean value_partial (2MM growth) for insensitive: {insensitive['value_partial'].mean():.4f}")
print(f"Mean value_full (0MM growth) for insensitive: {insensitive['value_full'].mean():.4f}")

# Check distribution of growth rates within the insensitive group
print("\nDistribution of value_full (0MM Growth) for Insensitive group:")
print(insensitive["value_full"].describe())

# Check for bimodal distribution (Low Growth vs High Growth)
# Low Growth (< 0.6): Likely hypersensitive (even 2MM is lethal)
# High Growth (>= 0.6): Likely escapers/ineffective (even 0MM is healthy)
insensitive_low = insensitive[insensitive["value_full"] < 0.6]
insensitive_high = insensitive[insensitive["value_full"] >= 0.6]

print(f"\nInsensitive Low Growth (Hypersensitive?): {len(insensitive_low)}")
if len(insensitive_low) > 0:
    print(f"Mean value_full: {insensitive_low['value_full'].mean():.4f}")

print(f"Insensitive High Growth (Escapers/Ineffective?): {len(insensitive_high)}")
if len(insensitive_high) > 0:
    print(f"Mean value_full: {insensitive_high['value_full'].mean():.4f}")

# Visualize the bimodal distribution
(
    gg.ggplot(insensitive, gg.aes(x="value_full"))
    + gg.geom_histogram(binwidth=0.05, fill="steelblue", color="white")
    + gg.labs(
        title="Distribution of Growth Rates (0MM) for Dosage-Insensitive Essential Genes",
        x="Instantaneous Growth Rate (0MM)",
    )
)
