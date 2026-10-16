import pandas as pd

df = pd.read_csv("data/Reviews.csv", usecols=["Score", "Text"])
df = df.dropna(subset=["Score", "Text"])

# Keep 1–2 stars as negative and 4–5 stars as positive
df = df[df["Score"].isin([1, 2, 4, 5])].copy()
df["sentiment"] = df["Score"].apply(
    lambda score: "positive" if score >= 4 else "negative"
)

# Take 20,000 reviews from each sentiment
balanced = (
    df.groupby("sentiment", group_keys=False)
    .sample(n=20_000, random_state=42)
    .reset_index(drop=True)
)

balanced[["Text", "sentiment"]].to_csv(
    "data/reviews_prepared.csv", index=False
)

print("Prepared dataset saved.")
print(balanced["sentiment"].value_counts())
print(balanced[["Text", "sentiment"]].head())