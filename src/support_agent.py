import joblib

from retrieve_resolutions import ResolutionRetriever
from draft_reply import clean_historical_response
from route_ticket import decide_route


MODEL_PATH = "results/intent_classifier.joblib"


class SupportAgent:
    def __init__(self):
        print("Loading intent classifier...")
        self.classifier = joblib.load(MODEL_PATH)

        print("Loading historical resolutions...")
        self.retriever = ResolutionRetriever()

        print("Support agent ready!")

    def handle(self, customer_message):
        # 1. Classify intent
        intent = self.classifier.predict([customer_message])[0]

        # 2. Retrieve historical resolutions
        sources = self.retriever.search(
            customer_message,
            top_k=3,
        )

        best_similarity = (
            sources[0]["similarity"]
            if sources
            else 0.0
        )

        # 3. Decide whether to auto-handle or escalate
        routing = decide_route(
            intent=intent,
            retrieval_similarity=best_similarity,
            customer_message=customer_message,
        )

        # 4. Draft grounded response
        if routing["decision"] == "escalate":
            reply = (
                "Thanks for reaching out. We'd like to make sure "
                "your issue gets the appropriate attention. "
                "We're escalating this to a support specialist "
                "who can take a closer look."
            )
        elif sources:
            historical_response = clean_historical_response(
                sources[0]["response"]
            )

            reply = (
                "Thanks for reaching out. We'd like to help get "
                "this resolved. Based on a similar case, "
                "we recommend the following:\n\n"
                f"{historical_response}\n\n"
                "If you're still having trouble after trying this, "
                "please contact Apple Support directly so we can "
                "take a closer look."
            )
        else:
            reply = (
                "Thanks for reaching out. We'd like to take a "
                "closer look at this issue. Please contact "
                "Apple Support so we can help further."
            )

        return {
            "intent": intent,
            "decision": routing["decision"],
            "reason": routing["reason"],
            "reply": reply,
            "retrieval_similarity": best_similarity,
            "sources": sources,
        }


if __name__ == "__main__":
    agent = SupportAgent()

    customer_message = input(
        "\nEnter a customer message: "
    )

    result = agent.handle(customer_message)

    print("\n" + "=" * 60)
    print("SUPPORT AGENT RESULT")
    print("=" * 60)

    print(f"\nIntent: {result['intent']}")
    print(f"Decision: {result['decision']}")
    print(f"Reason: {result['reason']}")
    print(
        f"Retrieval similarity: "
        f"{result['retrieval_similarity']:.4f}"
    )

    print("\nDraft reply")
    print("-----------")
    print(result["reply"])

    print("\nGrounding sources")
    print("-----------------")

    for i, source in enumerate(result["sources"], 1):
        print(f"\nSource {i}")
        print(f"Similarity: {source['similarity']:.4f}")
        print(f"Customer: {source['customer_message']}")
        print(f"Response: {source['response']}")