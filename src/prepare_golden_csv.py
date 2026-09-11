import json
import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/golden/golden_set.jsonl")
OUTPUT_FILE = Path("data/golden/golden_set.csv")

rows = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        conversation = json.loads(line)

        responses = conversation.get("brand_responses", [])

        brand_response = ""

        if responses:
            brand_response = responses[0]["text"]

        rows.append({
            "customer_tweet_id": conversation["customer_tweet_id"],
            "customer_id": conversation["customer_id"],
            "customer_message": conversation["customer_message"],
            "brand_response": brand_response,
            "customer_created_at": conversation["customer_created_at"],
            "intent": "",
            "expected_escalation": "",
            "difficulty": "",
            "notes": ""
        })

df = pd.DataFrame(rows)

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

print(f"Created {len(df)} golden examples.")
print(f"Saved to: {OUTPUT_FILE}")