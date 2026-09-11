import csv
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


GOLDEN_PATH = Path("data/golden/golden_set.csv")
MODEL_PATH = Path("results/intent_classifier.joblib")
RESULTS_PATH = Path("results/intent_predictions.csv")


def main():
    print("Loading model...")
    model = joblib.load(MODEL_PATH)

    print("Loading golden set...")
    with GOLDEN_PATH.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    texts = [row["customer_message"] for row in rows]
    true_labels = [row["intent"] for row in rows]

    print("Making predictions...")
    predictions = model.predict(texts)

    accuracy = accuracy_score(true_labels, predictions)

    print()
    print("Simple Intent Classifier")
    print("========================")
    print(f"Examples: {len(rows)}")
    print(f"Accuracy: {accuracy:.4f}")
    print()
    print("Classification Report")
    print("---------------------")

    print(
        classification_report(
            true_labels,
            predictions,
            zero_division=0,
        )
    )

    print("Confusion Matrix")
    print("----------------")

    labels = sorted(set(true_labels))

    matrix = confusion_matrix(
        true_labels,
        predictions,
        labels=labels,
    )

    print("Labels:")
    print(labels)
    print(matrix)

    # Save predictions for later analysis and report generation.
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with RESULTS_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "customer_tweet_id",
                "customer_message",
                "true_intent",
                "predicted_intent",
                "correct",
            ],
        )

        writer.writeheader()

        for row, prediction in zip(rows, predictions):
            writer.writerow(
                {
                    "customer_tweet_id": row["customer_tweet_id"],
                    "customer_message": row["customer_message"],
                    "true_intent": row["intent"],
                    "predicted_intent": prediction,
                    "correct": row["intent"] == prediction,
                }
            )

    print()
    print(f"Predictions saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()