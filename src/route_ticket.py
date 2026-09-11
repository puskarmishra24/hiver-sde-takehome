def decide_route(
    intent,
    retrieval_similarity,
    customer_message,
    intent_confidence=1.0,
):
    message = customer_message.lower().strip()
    words = message.split()

    # High-risk categories always require human review.
    if intent == "security_phishing_issue":
        return {
            "decision": "escalate",
            "reason": "Security or phishing issue requires human review.",
        }

    if intent == "payment_billing_issue":
        return {
            "decision": "escalate",
            "reason": "Payment or billing issue requires human review.",
        }

    if intent == "warranty_repair_recycling":
        return {
            "decision": "escalate",
            "reason": "Repair or warranty issue may require case-specific support.",
        }

    if intent == "support_contact_issue":
        return {
            "decision": "escalate",
            "reason": "Existing support case or contact request requires human follow-up.",
        }

    # Potentially severe situations.
    severe_terms = [
        "burn",
        "burned",
        "burnt",
        "fire",
        "smoke",
        "injury",
        "injured",
        "hurt",
        "stolen",
        "hacked",
        "lost everything",
        "disabled",
    ]

    if any(term in message for term in severe_terms):
        return {
            "decision": "escalate",
            "reason": "Message indicates a potentially severe or high-risk situation.",
        }

    # Account/authentication problems.
    account_risk_terms = [
        "can't log in",
        "cannot log in",
        "can't login",
        "cannot login",
        "can't access my account",
        "cannot access my account",
        "locked out",
        "password won't work",
        "password does not work",
        "password doesn't work",
        "security questions",
    ]

    if any(term in message for term in account_risk_terms):
        return {
            "decision": "escalate",
            "reason": "Account access or authentication issue requires human review.",
        }

    # Very short or vague requests.
    vague_phrases = [
        "it doesn't work",
        "it doesnt work",
        "doesn't work",
        "doesnt work",
        "please help",
        "help me",
        "not working",
        "something is wrong",
        "what is going on",
        "what's going on",
    ]

    if len(words) <= 6 or any(phrase in message for phrase in vague_phrases):
        return {
            "decision": "escalate",
            "reason": "Customer message is too vague to identify the issue with sufficient confidence.",
        }

    # Require a sufficiently similar historical resolution.
    if retrieval_similarity < 0.25:
        return {
            "decision": "escalate",
            "reason": "No sufficiently similar historical resolution was found.",
        }

    return {
        "decision": "auto_handle",
        "reason": (
            "Intent is suitable for automated handling and a sufficiently "
            "similar historical resolution was found."
        ),
    }


if __name__ == "__main__":
    examples = [
        (
            "software_update_issue",
            0.535,
            "My iPhone battery is draining really fast after the latest iOS update.",
        ),
        (
            "security_phishing_issue",
            0.38,
            "I received a suspicious Apple email asking for my password.",
        ),
        (
            "payment_billing_issue",
            0.41,
            "I was charged twice for the same purchase.",
        ),
        (
            "app_or_media_issue",
            0.66,
            "It doesn't work. Please help.",
        ),
    ]

    for intent, similarity, message in examples:
        result = decide_route(
            intent,
            similarity,
            message,
        )

        print("\nMessage:", message)
        print("Intent:", intent)
        print("Similarity:", similarity)
        print("Decision:", result["decision"])
        print("Reason:", result["reason"])