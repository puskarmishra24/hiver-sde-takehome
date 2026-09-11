import json
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


CONVERSATIONS_PATH = Path("data/processed/apple_conversations.jsonl")


def clean_text(text):
    """Basic text cleaning for retrieval."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class ResolutionRetriever:
    def __init__(self, conversations_path=CONVERSATIONS_PATH):
        self.conversations = []

        print("Loading historical conversations...")

        with conversations_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)

                customer_message = row.get("customer_message", "")
                brand_responses = row.get("brand_responses", [])

                if not customer_message or not brand_responses:
                    continue

                # Use the first AppleSupport response as the historical resolution.
                response = brand_responses[0].get("text", "")

                if not response:
                    continue

                self.conversations.append(
                    {
                        "customer_tweet_id": row.get("customer_tweet_id"),
                        "customer_message": customer_message,
                        "response": response,
                    }
                )

        print(f"Usable historical resolutions: {len(self.conversations):,}")

        self.texts = [
            clean_text(item["customer_message"])
            for item in self.conversations
        ]

        print("Building TF-IDF index...")

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=100_000,
            sublinear_tf=True,
        )

        self.matrix = self.vectorizer.fit_transform(self.texts)

        print("Retriever ready!")

    def search(self, query, top_k=3):
        """Return the most similar historical resolutions."""
        query_vector = self.vectorizer.transform([clean_text(query)])

        similarities = cosine_similarity(
            query_vector,
            self.matrix,
        ).flatten()

        top_indices = similarities.argsort()[::-1][:top_k]

        results = []

        for index in top_indices:
            item = self.conversations[index]

            results.append(
                {
                    "customer_tweet_id": item["customer_tweet_id"],
                    "customer_message": item["customer_message"],
                    "response": item["response"],
                    "similarity": float(similarities[index]),
                }
            )

        return results


if __name__ == "__main__":
    retriever = ResolutionRetriever()

    query = input("\nEnter a customer message: ")

    results = retriever.search(query, top_k=3)

    print("\nTop historical matches")
    print("======================")

    for i, result in enumerate(results, 1):
        print(f"\nMatch {i}")
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Customer: {result['customer_message']}")
        print(f"AppleSupport: {result['response']}")