import re

from retrieve_resolutions import ResolutionRetriever


MIN_REPLY_SIMILARITY = 0.40


def clean_historical_response(response):
    """Clean Twitter-specific artifacts from a historical response."""
    response = str(response)

    # Remove Twitter mentions.
    response = re.sub(r"@\w+", "", response)

    # Remove URLs.
    response = re.sub(r"https?://\S+", "", response)

    # Remove common Twitter-specific wording.
    response = re.sub(r"\bDM\b", "direct message", response)

    # Normalize whitespace.
    response = re.sub(r"\s+", " ", response).strip()

    # Remove dangling whitespace before punctuation.
    response = re.sub(r"\s+([.,!?])", r"\1", response)

    return response.strip()


def draft_reply(customer_message, retriever):
    """
    Draft a customer-facing reply grounded in historical resolutions.

    A historical response is used only when the best retrieval match
    exceeds MIN_REPLY_SIMILARITY. Otherwise, the agent falls back to
    a safe support handoff rather than copying a potentially irrelevant
    historical response.
    """
    results = retriever.search(customer_message, top_k=3)

    if not results:
        return {
            "reply": (
                "Thanks for reaching out. We'd be happy to help. "
                "Please contact Apple Support so we can take a closer look "
                "at this issue."
            ),
            "sources": [],
        }

    best = results[0]
    similarity = float(best["similarity"])

    if similarity < MIN_REPLY_SIMILARITY:
        reply = (
            "Thanks for reaching out. We'd be happy to help. "
            "We'd like to take a closer look at this issue. "
            "Please contact Apple Support so we can help you further."
        )

        return {
            "reply": reply,
            "sources": results,
        }

    historical_response = clean_historical_response(
        best["response"]
    )

    if not historical_response:
        reply = (
            "Thanks for reaching out. We'd be happy to help. "
            "Please contact Apple Support so we can take a closer look "
            "at this issue."
        )
    else:
        reply = (
            "Thanks for reaching out. We'd be happy to help.\n\n"
            f"{historical_response}\n\n"
            "If this doesn't resolve the issue, please contact Apple Support "
            "so we can take a closer look."
        )

    return {
        "reply": reply,
        "sources": results,
    }


if __name__ == "__main__":
    retriever = ResolutionRetriever()

    customer_message = input("\nEnter a customer message: ")

    result = draft_reply(customer_message, retriever)

    print("\nDraft reply")
    print("===========")
    print(result["reply"])

    print("\nGrounding sources")
    print("=================")

    for i, source in enumerate(result["sources"], 1):
        print(f"\nSource {i}")
        print(f"Similarity: {source['similarity']:.4f}")
        print(f"Historical customer: {source['customer_message']}")
        print(f"Historical response: {source['response']}")