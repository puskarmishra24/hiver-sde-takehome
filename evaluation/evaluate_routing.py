import csv
import joblib
import sys
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

sys.path.insert(0, "src")

from retrieve_resolutions import ResolutionRetriever
from route_ticket import decide_route


GOLDEN_PATH = "data/golden/golden_set.csv"
MODEL_PATH = "results/intent_classifier.joblib"
OUTPUT_PATH = "results/routing_predictions.csv"


def main():
    print("Loading golden set...")
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Golden examples: {len(rows)}")

    print("Loading intent classifier...")
    clf = joblib.load(MODEL_PATH)

    print("Loading historical conversations...")
    retriever = ResolutionRetriever()

    predictions = []

    for i, row in enumerate(rows, start=1):
        message = row["customer_message"]

        # Intent prediction and confidence.
        probabilities = clf.predict_proba([message])[0]
        predicted_index = probabilities.argmax()

        predicted_intent = clf.classes_[predicted_index]
        intent_confidence = float(probabilities[predicted_index])

        # Historical resolution retrieval.
        results = retriever.search(message, top_k=3)

        if results:
            retrieval_similarity = float(results[0]["similarity"])
        else:
            retrieval_similarity = 0.0

        # Routing decision.
        route = decide_route(
            predicted_intent,
            retrieval_similarity,
            message,
            intent_confidence,
        )

        expected = row["expected_escalation"].strip().lower() == "true"
        predicted_escalation = route["decision"] == "escalate"

        predictions.append({
            "customer_tweet_id": row["customer_tweet_id"],
            "customer_message": message,
            "expected_escalation": expected,
            "predicted_escalation": predicted_escalation,
            "predicted_intent": predicted_intent,
            "intent_confidence": intent_confidence,
            "retrieval_similarity": retrieval_similarity,
            "decision": route["decision"],
            "reason": route["reason"],
        })

        if i % 25 == 0:
            print(f"Processed {i}/{len(rows)}")

    # Save detailed predictions.
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=predictions[0].keys(),
        )
        writer.writeheader()
        writer.writerows(predictions)

    y_true = [x["expected_escalation"] for x in predictions]
    y_pred = [x["predicted_escalation"] for x in predictions]

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        pos_label=True,
        average="binary",
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[False, True],
    ).ravel()

    print("\n=== Routing Evaluation ===")
    print(f"Examples: {len(predictions)}")
    print(f"Routing accuracy: {accuracy:.4f}")
    print(f"Escalation precision: {precision:.4f}")
    print(f"Escalation recall: {recall:.4f}")
    print(f"Escalation F1: {f1:.4f}")

    print("\nConfusion matrix:")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TP: {tp}")

    print(f"\nSaved predictions to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()