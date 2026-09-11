import pandas as pd
from pathlib import Path
import json

DATA_FILE = Path("data/raw/twcs.csv")
OUTPUT_FILE = Path("data/processed/apple_conversations.jsonl")

BRAND = "AppleSupport"

print("Loading dataset...")
df = pd.read_csv(DATA_FILE)

print(f"Total tweets: {len(df):,}")

# ---------------------------------------------------------
# 1. Get tweets involving AppleSupport
# ---------------------------------------------------------

brand_df = df[
    (df["author_id"] == BRAND) |
    (df["text"].str.contains("@AppleSupport", case=False, na=False))
].copy()

print(f"AppleSupport tweets: {len(brand_df):,}")

# ---------------------------------------------------------
# 2. Create lookup by tweet ID
# ---------------------------------------------------------

tweet_lookup = df.set_index("tweet_id").to_dict("index")

# ---------------------------------------------------------
# 3. Find customer tweets
# ---------------------------------------------------------

customer_df = brand_df[brand_df["inbound"] == True].copy()

print(f"Customer tweets: {len(customer_df):,}")

# ---------------------------------------------------------
# 4. Build conversation examples
# ---------------------------------------------------------

conversations = []

for _, row in customer_df.iterrows():

    tweet_id = int(row["tweet_id"])

    # Find the brand's response(s)
    response_ids = row["response_tweet_id"]

    if pd.isna(response_ids):
        continue

    response_ids = str(response_ids).split(",")

    responses = []

    for response_id in response_ids:

        response_id = response_id.strip()

        try:
            response_id = int(float(response_id))
        except ValueError:
            continue

        if response_id not in tweet_lookup:
            continue

        response = tweet_lookup[response_id]

        # Only keep AppleSupport responses
        if response["author_id"] != BRAND:
            continue

        responses.append({
            "tweet_id": response_id,
            "text": response["text"],
            "created_at": response["created_at"]
        })

    if not responses:
        continue

    conversations.append({
        "customer_tweet_id": tweet_id,
        "customer_id": str(row["author_id"]),
        "customer_message": row["text"],
        "customer_created_at": row["created_at"],
        "brand_responses": responses
    })


# ---------------------------------------------------------
# 5. Save
# ---------------------------------------------------------

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    for conversation in conversations:
        f.write(json.dumps(conversation, ensure_ascii=False) + "\n")

print("\nFinished!")
print(f"Conversation examples: {len(conversations):,}")
print(f"Saved to: {OUTPUT_FILE}")