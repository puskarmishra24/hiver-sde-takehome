import os
import pandas as pd

INPUT = "data/golden/golden_set.csv"
OUTPUT = "data/golden/golden_set_reviewed.csv"

INTENTS = [
    "software_update_issue",
    "device_performance_issue",
    "keyboard_input_issue",
    "connectivity_issue",
    "app_or_media_issue",
    "apple_id_icloud_issue",
    "payment_billing_issue",
    "order_sales_issue",
    "hardware_accessory_issue",
    "security_phishing_issue",
    "support_contact_issue",
    "warranty_repair_recycling",
    "product_feature_question",
    "general_or_unclear",
]


def main():
    original = pd.read_csv(INPUT)

    if os.path.exists(OUTPUT):
        df = pd.read_csv(OUTPUT)
        print(f"Resuming existing review: {OUTPUT}")
    else:
        df = original.copy()
        print("Starting new review.")

    reviewed = 0

    for i in range(len(df)):
        row = df.iloc[i]

        # Already reviewed rows are marked in notes.
        if str(row.get("review_status", "")).lower() == "reviewed":
            continue

        print("\n" + "=" * 80)
        print(f"Example {i + 1}/{len(df)}")
        print(f"Tweet ID: {row['customer_tweet_id']}")
        print("-" * 80)
        print(row["customer_message"])
        print("-" * 80)
        print(f"PROPOSED: {row['intent']}")
        print(f"ESCALATION: {row['expected_escalation']}")
        print()
        print("1-14 = change intent")
        print("ENTER = keep")
        print("s = save and stop")

        choice = input("Choice: ").strip().lower()

        if choice == "s":
            df.to_csv(OUTPUT, index=False)
            print(f"\nProgress saved: {OUTPUT}")
            print(f"Reviewed this session: {reviewed}")
            return

        if choice == "":
            pass
        elif choice.isdigit() and 1 <= int(choice) <= 14:
            df.at[i, "intent"] = INTENTS[int(choice) - 1]
        else:
            print("Invalid choice. Example not marked reviewed.")
            continue

        df.at[i, "review_status"] = "reviewed"
        reviewed += 1

        # Save after EVERY example.
        df.to_csv(OUTPUT, index=False)

    print("\n" + "=" * 80)
    print("REVIEW COMPLETE")
    print("=" * 80)
    print(f"Reviewed: {len(df)}")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()