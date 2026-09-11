import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score


REVIEW_PATH = "results/human_reply_review.csv"
JUDGE_PATH = "results/llm_judge_results.csv"


def main():
    review = pd.read_csv(REVIEW_PATH)
    judge = pd.read_csv(JUDGE_PATH)

    merged = review.merge(
        judge,
        on="customer_tweet_id",
        suffixes=("_review", "_judge"),
    )

    if len(merged) == 0:
        raise ValueError("No matching tweet IDs between review and judge files.")

    print(f"Matched examples: {len(merged)}")

    # Overall score correlation.
    rho, p_value = spearmanr(
        merged["human_overall"],
        merged["overall_score"],
    )

    # Pass/fail agreement.
    review_pass = (
        merged["human_pass"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    judge_pass = (
        merged["pass"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    agreement = (review_pass == judge_pass).mean()
    kappa = cohen_kappa_score(review_pass, judge_pass)

    print(f"Spearman correlation (overall): {rho:.4f}")
    print(f"Spearman p-value: {p_value:.4f}")
    print(f"Pass/fail agreement: {agreement:.2%}")
    print(f"Cohen's kappa: {kappa:.4f}")

    print("\nMean scores:")
    print(f"Review overall: {merged['human_overall'].mean():.2f}")
    print(f"Judge overall:  {merged['overall_score'].mean():.2f}")

    merged.to_csv(
        "results/judge_review_comparison.csv",
        index=False,
    )

    print("\nSaved to results/judge_review_comparison.csv")


if __name__ == "__main__":
    main()