import pandas as pd

INPUT_CSV = "HI_Small.csv"
OUTPUT_CSV = "HI_Small_Trans.csv"

NEGATIVE_RATIO = 25
RANDOM_SEED = 42

# ==========================
# LOAD DATA
# ==========================

print("Loading dataset...")

df = pd.read_csv(INPUT_CSV)

# ==========================
# SPLIT POSITIVE / NEGATIVE
# ==========================

positive_df = df[df["Is Laundering"] == 1]

negative_df = df[df["Is Laundering"] == 0]

num_positive = len(positive_df)

target_negative_count = num_positive * NEGATIVE_RATIO

print(f"Positive transactions : {num_positive:,}")
print(f"Negative transactions : {len(negative_df):,}")
print(f"Target negatives      : {target_negative_count:,}")

# ==========================
# SAMPLE NEGATIVES
# ==========================

if target_negative_count > len(negative_df):
    print(
        "WARNING: Not enough negative samples available. "
        "Keeping all negatives."
    )
    sampled_negative_df = negative_df
else:
    sampled_negative_df = negative_df.sample(
        n=target_negative_count,
        random_state=RANDOM_SEED,
    )

# ==========================
# COMBINE
# ==========================

balanced_df = pd.concat(
    [positive_df, sampled_negative_df],
    ignore_index=True,
)

# Shuffle rows
balanced_df = balanced_df.sample(
    frac=1,
    random_state=RANDOM_SEED,
).reset_index(drop=True)

# ==========================
# SAVE
# ==========================

balanced_df.to_csv(
    OUTPUT_CSV,
    index=False,
)

# ==========================
# REPORT
# ==========================

final_positive = (balanced_df["Is Laundering"] == 1).sum()
final_negative = (balanced_df["Is Laundering"] == 0).sum()

print("\nDone!")
print(f"Saved to: {OUTPUT_CSV}")

print("\nFinal distribution:")
print(f"Positive : {final_positive:,}")
print(f"Negative : {final_negative:,}")

print(
    f"Ratio    : 1:{final_negative/final_positive:.2f}"
)