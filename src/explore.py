import pandas as pd
from pathlib import Path

DATA_FILE = Path("data/raw/twcs.csv")

print("Loading dataset...")
df = pd.read_csv(DATA_FILE)

print(f"Total tweets: {len(df):,}")

# Brand/support messages are outbound tweets
brand_tweets = df[df["inbound"] == False]

print(f"Outbound/brand tweets: {len(brand_tweets):,}")

# Count tweets by author
brand_counts = (
    brand_tweets["author_id"]
    .value_counts()
    .head(30)
)

print("\nTop 30 support accounts:")
print("-" * 50)

for i, (brand, count) in enumerate(brand_counts.items(), start=1):
    print(f"{i:2}. {brand:<30} {count:>10,} tweets")