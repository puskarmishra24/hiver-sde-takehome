import csv
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib


CONVERSATIONS_PATH = Path("data/processed/apple_conversations.jsonl")
GOLDEN_PATH = Path("data/golden/golden_set.csv")
MODEL_PATH = Path("results/intent_classifier.joblib")


# Simple keyword-based labeling rules used ONLY to create
# a development/training set from the historical conversations.
INTENT_KEYWORDS = {
    "software_update_issue": [
        "update", "ios", "install ios", "upgrade", "latest ios",
        "software update", "downloading update"
    ],
    "device_performance_issue": [
        "slow", "freezing", "freeze", "crash", "crashing", "restart",
        "restarting", "battery", "overheating", "overheat", "performance"
    ],
    "keyboard_input_issue": [
        "keyboard", "autocorrect", "auto correct", "typing", "key",
        "keypress", "emoji", "predictive text"
    ],
    "connectivity_issue": [
        "wifi", "wi-fi", "bluetooth", "hotspot", "internet",
        "connection", "connect", "connecting", "network"
    ],
    "app_or_media_issue": [
        "app", "application", "itunes", "music", "apple music",
        "podcast", "video", "photo", "photos", "notification"
    ],
    "apple_id_icloud_issue": [
        "apple id", "icloud", "i-cloud", "icloud account",
        "icloud storage", "contacts", "sync", "password"
    ],
    "payment_billing_issue": [
        "charge", "charged", "billing", "bill", "payment",
        "refund", "subscription", "apple pay", "purchase"
    ],
    "order_sales_issue": [
        "order", "ordered", "delivery", "deliver", "shipping",
        "reservation", "buy", "bought", "purchase"
    ],
    "hardware_accessory_issue": [
        "iphone", "ipad", "macbook", "airpods", "airpod",
        "apple watch", "charger", "cable", "headphones",
        "screen", "button", "hardware"
    ],
    "security_phishing_issue": [
        "phishing", "scam", "scammer", "fraud", "hack",
        "hacked", "security", "suspicious", "fake email"
    ],
    "support_contact_issue": [
        "support", "customer service", "contact apple",
        "call apple", "speak to someone", "representative"
    ],
    "warranty_repair_recycling": [
        "repair", "warranty", "replace", "replacement",
        "recycle", "recycling", "genius bar"
    ],
    "product_feature_question": [
        "how do i", "how can i", "can i", "does iphone",
        "does ipad", "feature", "setting", "enable", "disable"
    ],
}


def load_golden_ids():
    with GOLDEN_PATH.open("r", encoding="utf-8") as f:
        return {
            row["customer_tweet_id"]
            for row in csv.DictReader(f)
        }


def label_message(text):
    """
    Assign an intent using simple keyword rules.

    If several intents match, use the intent with the
    largest number of keyword matches.
    """
    text = text.lower()

    scores = {}

    for intent, keywords in INTENT_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in text)
        if score > 0:
            scores[intent] = score

    if not scores:
        return "general_or_unclear"

    return max(scores, key=scores.get)


def load_training_data(golden_ids):
    texts = []
    labels = []

    with CONVERSATIONS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            tweet_id = str(row["customer_tweet_id"])

            # Golden examples MUST remain held out.
            if tweet_id in golden_ids:
                continue

            text = row["customer_message"].strip()

            if not text:
                continue

            intent = label_message(text)

            texts.append(text)
            labels.append(intent)

    return texts, labels


def main():
    print("Loading golden IDs...")
    golden_ids = load_golden_ids()
    print(f"Golden examples held out: {len(golden_ids):,}")

    print("Creating development training labels...")
    texts, labels = load_training_data(golden_ids)

    print(f"Training examples: {len(texts):,}")
    print(f"Number of intents: {len(set(labels))}")

    print()
    print("Training intent classifier...")

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_features=100_000,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
            ),
        ),
    ])

    model.fit(texts, labels)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print()
    print("Training complete!")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()