import json
import random
from pathlib import Path
import pandas as pd

INPUT_FILE = Path("data/processed/apple_conversations.jsonl")
OUTPUT_FILE = Path("data/processed/apple_sample_100.csv")

SAMPLE_SIZE = 100

print("Loading conversations...")

conversations = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        conversations.append(json.loads(line))

print(f"Total conversations: {len(conversations):,}")

# Use a fixed seed so we get the same sample every time
random.seed(42)

sample = random.sample(
    conversations,
    min(SAMPLE_SIZE, len(conversations))
)

rows = []

for conversation in sample:

    responses = conversation.get("brand_responses", [])

    brand_response = ""

    if responses:
        brand_response = responses[0]["text"]

    rows.append({
        "customer_tweet_id": conversation["customer_tweet_id"],
        "customer_id": conversation["customer_id"],
        "customer_message": conversation["customer_message"],
        "brand_response": brand_response,
        "customer_created_at": conversation["customer_created_at"]
    })

df = pd.DataFrame(rows)

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

print("\nDone!")
print(f"Sample size: {len(df)}")
print(f"Saved to: {OUTPUT_FILE}")