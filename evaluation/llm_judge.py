import csv
import json
import os
import sys

from openai import OpenAI


INPUT_PATH = "results/human_reply_review.csv"
OUTPUT_PATH = "results/llm_judge_results.csv"

MODEL = "gpt-5.6-luna"


JUDGE_PROMPT = """
You are evaluating an AI customer-support reply for Apple Support.

Evaluate the draft reply against the customer's message and the retrieved
historical support evidence.

Score each dimension from 1 to 5:

1. correctness:
   Does the reply provide advice that is appropriate for the customer's issue?

2. helpfulness:
   Would this reply meaningfully help the customer move toward resolution?

3. relevance:
   Does the reply directly address the customer's message without unnecessary
   or unrelated content?

4. groundedness:
   Is the reply supported by the retrieved historical support evidence?
   Penalize claims or troubleshooting steps that are not supported by the evidence.

Also provide:
- overall_score: 1 to 5
- pass: true if overall_score >= 4, otherwise false
- short_reason: concise explanation of the rating

Return ONLY valid JSON with exactly these fields:

{
  "correctness": <integer>,
  "helpfulness": <integer>,
  "relevance": <integer>,
  "groundedness": <integer>,
  "overall_score": <integer>,
  "pass": <boolean>,
  "short_reason": "<string>"
}
"""


def build_prompt(customer_message, draft_reply, sources):
    evidence = []

    for i, source in enumerate(sources, 1):
        evidence.append(
            f"Source {i}\n"
            f"Similarity: {source['similarity']:.4f}\n"
            f"Historical customer: {source['customer_message']}\n"
            f"Historical response: {source['response']}"
        )

    evidence_text = "\n\n".join(evidence)

    return f"""
{JUDGE_PROMPT}

CUSTOMER MESSAGE:
{customer_message}

DRAFT REPLY:
{draft_reply}

RETRIEVED HISTORICAL EVIDENCE:
{evidence_text}
"""


def main():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print(
            "OPENAI_API_KEY is not set.\n"
            "Set it before running the LLM judge."
        )
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    sys.path.insert(0, "src")

    from retrieve_resolutions import ResolutionRetriever
    from draft_reply import draft_reply

    print("Loading golden set...")

    with open(INPUT_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Golden examples: {len(rows)}")

    print("Loading retriever...")
    retriever = ResolutionRetriever()

    results = []

    for i, row in enumerate(rows, 1):
        customer_message = row["customer_message"]

        drafted = draft_reply(
            customer_message,
            retriever,
        )

        prompt = build_prompt(
            customer_message,
            drafted["reply"],
            drafted["sources"],
        )

        response = client.responses.create(
            model=MODEL,
            input=prompt,
        )

        text = response.output_text.strip()

        try:
            judgment = json.loads(text)
        except json.JSONDecodeError:
            print(f"Invalid JSON returned for example {i}")
            print(text)
            continue

        results.append({
            "customer_tweet_id": row["customer_tweet_id"],
            "customer_message": customer_message,
            "draft_reply": drafted["reply"],
            "best_similarity": (
                drafted["sources"][0]["similarity"]
                if drafted["sources"]
                else 0.0
            ),
            "correctness": judgment["correctness"],
            "helpfulness": judgment["helpfulness"],
            "relevance": judgment["relevance"],
            "groundedness": judgment["groundedness"],
            "overall_score": judgment["overall_score"],
            "pass": judgment["pass"],
            "short_reason": judgment["short_reason"],
        })

        if i % 10 == 0:
            print(f"Judged {i}/{len(rows)}")

    if not results:
        print("No judge results were produced.")
        sys.exit(1)

    with open(
        OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n=== LLM Judge Summary ===")

    for metric in [
        "correctness",
        "helpfulness",
        "relevance",
        "groundedness",
        "overall_score",
    ]:
        values = [float(row[metric]) for row in results]
        average = sum(values) / len(values)
        print(f"{metric}: {average:.2f}")

    pass_rate = sum(
        1 for row in results if row["pass"]
    ) / len(results)

    print(f"Pass rate: {pass_rate:.2%}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()