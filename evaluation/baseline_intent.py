import csv
from collections import Counter
from pathlib import Path

GOLDEN_PATH = Path("data/golden/golden_set.csv")


def main():
    with GOLDEN_PATH.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    intents = [row["intent"] for row in rows]
    counts = Counter(intents)

    majority_intent, majority_count = counts.most_common(1)[0]

    correct = sum(
        1
        for row in rows
        if row["intent"] == majority_intent
    )

    accuracy = correct / len(rows)

    print("Trivial Intent Baseline")
    print("========================")
    print(f"Examples: {len(rows)}")
    print(f"Majority intent: {majority_intent}")
    print(f"Majority count: {majority_count}")
    print(f"Accuracy: {accuracy:.4f}")
    print()
    print("Intent distribution:")
    
    for intent, count in counts.most_common():
        print(f"  {intent}: {count}")


if __name__ == "__main__":
    main()