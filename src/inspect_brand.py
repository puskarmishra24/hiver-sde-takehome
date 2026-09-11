import pandas as pd
from pathlib import Path

DATA_FILE = Path("data/raw/twcs.csv")

BRAND = "AppleSupport"

print("Loading dataset...")
df = pd.read_csv(DATA_FILE)

# Get tweets involving AppleSupport.
# A tweet belongs to the brand's conversations if either:
# - AppleSupport wrote it
# - the tweet is a response to AppleSupport
apple = df[
    (df["author_id"] == BRAND) |
    (df["text"].str.contains("@AppleSupport", case=False, na=False))
].copy()

print(f"\nTweets involving {BRAND}: {len(apple):,}")

print("\nInbound vs outbound:")
print(apple["inbound"].value_counts())

print("\nSample conversations:")
print("=" * 80)

# Show 20 customer messages
customer_messages = apple[apple["inbound"] == True].head(20)

for _, row in customer_messages.iterrows():
    print("\nTweet ID:", row["tweet_id"])
    print("Customer:", row["author_id"])
    print("Message:", row["text"])
    print("Response tweet:", row["response_tweet_id"])
    print("In response to:", row["in_response_to_tweet_id"])
    print("-" * 80)