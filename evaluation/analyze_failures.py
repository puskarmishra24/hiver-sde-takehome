import pandas as pd
from pathlib import Path


INPUT_PATH = Path("results/top_failure_examples.csv")
OUTPUT_PATH = Path("results/failure_analysis.txt")


def main():
    df = pd.read_csv(INPUT_PATH)

    groups = [
        ("general_or_unclear", "app_or_media_issue"),
        ("product_feature_question", "app_or_media_issue"),
        ("keyboard_input_issue", "app_or_media_issue"),
        ("device_performance_issue", "software_update_issue"),
        ("apple_id_icloud_issue", "app_or_media_issue"),
    ]

    lines = []

    lines.append("TOP 5 INTENT CLASSIFICATION FAILURE MODES")
    lines.append("=" * 50)
    lines.append("")

    for i, (true_intent, predicted_intent) in enumerate(groups, 1):
        subset = df[
            (df["true_intent"] == true_intent)
            & (df["predicted_intent"] == predicted_intent)
        ]

        lines.append(
            f"{i}. {true_intent} -> {predicted_intent}"
        )
        lines.append(f"   Examples: {len(subset)}")
        lines.append("")

        # Include up to 5 real examples for the report.
        for _, row in subset.head(5).iterrows():
            message = str(row["customer_message"]).replace("\n", " ")
            lines.append(f"   Tweet ID: {row['customer_tweet_id']}")
            lines.append(f"   Message: {message}")
            lines.append("")

        lines.append("-" * 50)
        lines.append("")

    OUTPUT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"Failure analysis saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()