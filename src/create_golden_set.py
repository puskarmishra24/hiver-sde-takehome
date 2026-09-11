import json
import random
from pathlib import Path

INPUT_FILE = Path("data/processed/apple_conversations.jsonl")
OUTPUT_FILE = Path("data/golden/golden_set.jsonl")

GOLDEN_SIZE = 200

print("Loading AppleSupport conversations...")

conversations = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        conversations.append(json.loads(line))

print(f"Total conversations: {len(conversations):,}")

# Fixed random seed so the sample is reproducible
random.seed(123)

golden = random.sample(
    conversations,
    GOLDEN_SIZE
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for conversation in golden:
        f.write(
            json.dumps(
                conversation,
                ensure_ascii=False
            ) + "\n"
        )

print("\nGolden set created!")
print(f"Examples: {len(golden)}")
print(f"Saved to: {OUTPUT_FILE}")