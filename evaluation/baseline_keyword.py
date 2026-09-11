import csv
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report


GOLDEN_PATH = Path("data/golden/golden_set.csv")


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


def label_message(text):
    text = text.lower()

    scores = {}

    for intent, keywords in INTENT_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in text)
        if score > 0:
            scores[intent] = score

    if not scores:
        return "general_or_unclear"

    return max(scores, key=scores.get)


def main():
    with GOLDEN_PATH.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    true_labels = [row["intent"] for row in rows]
    predictions = [label_message(row["customer_message"]) for row in rows]

    accuracy = accuracy_score(true_labels, predictions)

    print()
    print("Keyword Intent Baseline")
    print("=======================")
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


if __name__ == "__main__":
    main()