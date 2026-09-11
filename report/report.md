# AppleSupport AI Support Agent — Hiver SDE Intern Assignment

## 1. Executive Summary

This project builds a lightweight support agent for AppleSupport using the Customer Support on Twitter dataset. The agent performs three tasks: intent classification, historically grounded reply drafting, and auto-handle versus escalation routing.

The central evaluation principle is that the system should be judged by evidence rather than by a polished demo. On a 200-example evaluation set, the learned TF-IDF + Logistic Regression classifier achieves 30.5% accuracy, compared with 21.5% for a majority-class baseline and 26.0% for a keyword baseline. The routing component reaches 64.0% overall accuracy, but its escalation recall is only 24.1%, which means the headline routing accuracy substantially overstates safety.

The reply generator retrieves similar historical AppleSupport resolutions and uses the retrieved response as the grounding source. A 30-example human reply review gives an overall score of 3.73/5. The LLM judge was compared against the human ratings, with 63.33% pass/fail agreement, Cohen's kappa of 0.1081, and Spearman correlation of 0.2242; the low agreement means the judge is treated as a development aid rather than ground truth.

The most important conclusion is therefore not that the system is production-ready. It is that the prototype establishes a reproducible baseline, exposes its weaknesses, and identifies the next data and evaluation improvements needed.

---

## 2. Problem Framing

### Goal

For an incoming Apple customer-support message:

1. assign one intent from a compact taxonomy derived from the data;
2. retrieve relevant historical AppleSupport resolutions and draft a grounded response;
3. decide whether the issue can be auto-handled or should be escalated, with an explicit reason.

### What “good” means

For AppleSupport, a useful support agent should:

- recognize the customer's actual problem rather than overfitting to generic words;
- ground suggested actions in previous AppleSupport responses;
- avoid inventing account-specific facts, policies, or troubleshooting steps;
- escalate security, billing, account-access, severe, or otherwise ambiguous cases;
- remain concise and support-oriented.

### What I chose not to build

I deliberately did not attempt a full production support platform, recursive conversation-state reconstruction, account/tool integrations, autonomous actions, or a large generative fine-tuning pipeline. The assignment emphasizes proving the system works, so the implementation prioritizes an auditable retrieval/classification/routing pipeline.

---

## 3. Data and Scope

The primary dataset is Customer Support on Twitter (TWCS). The full CSV contains approximately 2.8 million tweets. AppleSupport is one of the largest support accounts in the dataset.

For this project:

- 204,772 AppleSupport-related tweets were identified.
- 78,406 customer/support-response pairs were extracted for historical resolution retrieval.
- The extraction pairs customer tweets with directly linked AppleSupport responses; it does not attempt to reconstruct every multi-turn thread.
- A 200-example evaluation set was sampled and held out from classifier training.

### Submission repository data note

The full TWCS raw dataset (`data/raw/twcs.csv`) and the generated historical conversation file (`data/processed/apple_conversations.jsonl`) are **not included in the submitted GitHub repository because of GitHub's web-upload file-size limit**. A small sample is included for inspection, and the repository retains the scripts needed to regenerate the processed conversation file when the full TWCS dataset is supplied locally. These files were excluded only for submission packaging; the implementation and evaluation artifacts remain in the repository.


The intent taxonomy contains 14 categories:

1. software_update_issue
2. device_performance_issue
3. keyboard_input_issue
4. connectivity_issue
5. app_or_media_issue
6. apple_id_icloud_issue
7. payment_billing_issue
8. order_sales_issue
9. hardware_accessory_issue
10. security_phishing_issue
11. support_contact_issue
12. warranty_repair_recycling
13. product_feature_question
14. general_or_unclear

The current 200-example evaluation set contains examples for all 14 taxonomy categories. `hardware_accessory_issue` has 2 examples, so the class is represented but conclusions about that minority class should be treated cautiously.

### Evaluation-set provenance

The 200-example golden evaluation set was personally reviewed and labelled by the candidate using the 14-intent taxonomy defined from the AppleSupport data. The set contains examples from all 14 intents and was held out from classifier training. The labels were used as the final evaluation reference set.

**Sampling and labelling:** I sampled 200 AppleSupport customer messages from the evaluation pool to cover all 14 intents, including ambiguous and difficult examples. I personally reviewed each example, assigned one intent from the taxonomy, and assigned an escalation label based on whether the issue could reasonably be handled from the available historical evidence or required human/account-specific intervention.

---

## 4. System Design

### Intent classifier

The final classifier uses:

- TF-IDF features;
- word unigrams and bigrams;
- Logistic Regression;
- balanced class weighting.

The training labels were generated using deterministic keyword rules. This is an important methodological limitation: the learned classifier is trained against automatically generated labels rather than a manually labelled training set. The keyword baseline and learned classifier should therefore be interpreted as a controlled prototype comparison, not as a fully supervised human-labelled benchmark.

### Historical retrieval

Historical customer messages and their AppleSupport responses are indexed with TF-IDF. At inference time, the incoming message is compared with historical customer messages and the top matching cases are retrieved.

A battery-drain smoke test produced a best similarity of approximately 0.535 and retrieved historical cases involving battery drain after iOS updates. This demonstrates that the retrieval layer can find semantically relevant historical support resolutions without requiring an external knowledge base.

### Reply drafting

The draft generator uses the best historical response as grounding material. It removes Twitter handles and URLs and adds a conservative fallback asking the customer to contact Apple Support when retrieval confidence is low.

The design intentionally avoids inventing detailed troubleshooting instructions that are not present in the historical evidence.

### Routing

The routing policy escalates:

- security/phishing;
- billing/payment;
- warranty/repair/recycling;
- support-contact cases;
- account-access/authentication cases;
- severe-risk language;
- very vague requests;
- low-retrieval-confidence cases.

Other sufficiently supported cases are eligible for auto-handling.

---

## 5. Evaluation Results

### Intent classification

| System | Accuracy |
|---|---:|
| Majority-class baseline | **21.5%** |
| Keyword baseline | **26.0%** |
| TF-IDF + Logistic Regression | **30.5%** |

The learned classifier improves over the majority baseline by **9.0 percentage points** (30.5% vs 21.5%) and over the keyword baseline by 4.5 points.

Additional final-model metrics:

- Macro precision: 0.44
- Macro recall: 0.27
- Macro F1: 0.29
- Weighted F1: 0.35

The gap between accuracy and macro recall/F1 is important. Several minority or ambiguous intents remain difficult, so the model should not be described as generally reliable merely because it beats the baselines.

### Routing

On the 200-example evaluation set:

- Accuracy: **64.0%**
- Escalation precision: **69.0%**
- Escalation recall: **24.1%**
- Escalation F1: **35.7%**
- True negatives: 108
- False positives: 9
- False negatives: 63
- True positives: 20

The low escalation recall is the critical safety result. A system that misses many cases that should be escalated should not be deployed as an unrestricted autonomous support agent.

---

## 6. Reply Quality Evaluation

A 30-example **human reply review** was completed using a five-dimension rubric:

- correctness;
- helpfulness;
- relevance;
- groundedness;
- overall quality.

The candidate personally rated all 30 examples. The human review scores were:

- correctness: **3.80/5**
- helpfulness: **3.60/5**
- relevance: **4.67/5**
- groundedness: **4.67/5**
- overall: **3.73/5**
- pass rate: **66.7%**

The LLM judge was evaluated against these human ratings using pass/fail agreement and rank correlation. The matched comparison produced **63.33% pass/fail agreement**, **Cohen's kappa of 0.1081**, and **Spearman correlation of 0.2242**. The low agreement indicates that the judge should not be treated as a substitute for human evaluation.


## 7. Top Five Failure Modes

### 1. General/unclear → app/media

This is a recurring confusion pattern in the classifier predictions. Examples include vague messages such as “what is going on?” and emotionally worded complaints where there is insufficient information to determine a precise intent.

**Hypothesis:** short, context-dependent Twitter messages contain too little lexical evidence for a fine-grained classifier.

**Improvement:** add an explicit ambiguity/insufficient-context class or confidence gate, and train on genuine human labels.

### 2. Product-feature questions → app/media

Questions about features, settings, maps, presentation formats, or future software changes are frequently classified as app/media issues.

**Hypothesis:** feature questions often mention a product/app surface without using distinctive intent vocabulary.

**Improvement:** introduce question-type features and more representative feature-question training examples.

### 3. Keyboard/input issues → app/media

Messages about individual letters, keyboard behavior, or input rendering are often mapped to generic app/media.

**Hypothesis:** short keyboard complaints overlap heavily with generic application/device vocabulary.

**Improvement:** increase labelled examples for keyboard/input and use character-level or spelling-robust features.

### 4. Device performance → software update

Performance complaints frequently mention a recent iOS update, causing the model to focus on “update” rather than the underlying symptom.

**Hypothesis:** lexical overlap with update language dominates the actual intent.

**Improvement:** explicitly separate root-cause language from symptom language and add hierarchical classification: symptom first, suspected cause second.

### 5. Apple ID/iCloud → app/media

Messages involving pictures, messages, login failures, and account data can look like application/media issues.

**Hypothesis:** account problems are often described through the affected content rather than the account mechanism.

**Improvement:** add entity-aware features for account/login/iCloud terms and collect more hard-negative examples.

---

## 8. What Is Misleading About My Headline Number?

The most tempting headline is the **64.0% routing accuracy**.

That number is misleading if interpreted as “64.0% of routing decisions are safe to automate.” The evaluation is small, the escalation class is imbalanced, and escalation recall is only 24.1%. In other words, the system can achieve a respectable overall accuracy while still missing a large fraction of cases that should have been escalated.

Similarly, the 30.5% intent accuracy is not evidence of production-grade intent classification. The evaluation set is only 200 examples, the smallest classes have very few examples, and the training labels were automatically generated.

The most defensible conclusion is therefore: **the prototype improves over simple baselines, but the current evidence is insufficient for autonomous deployment.**

---

## 9. One More Week

If given another week, I would prioritize the following in order:

1. **Expand the golden set to 500+ examples** with a written annotation guide, stratified sampling, and explicit edge-case rules.
2. **Expand the human reply-review sample**, ideally beyond the initial 30 examples, and keep the exact rubric consistent.
3. **Validate the LLM judge against independent human ratings** using a larger independently rated sample before relying on judge scores for headline evaluation.
4. Retrain the classifier on genuine human labels and compare against the same baselines.
5. Add confidence-aware routing and optimize specifically for escalation recall rather than raw accuracy.
6. Expand the retrieval evaluation with relevance labels and test whether retrieved cases actually support the drafted answer.
7. Fix response cleanup so removing historical URLs cannot leave dangling fragments.
8. Add more hard-negative examples for the five major confusion pairs.
9. Evaluate robustness to spelling mistakes, short tweets, sarcasm, and multi-intent messages.
10. Add automated regression tests for routing safety and unsupported reply claims.

---

## 10. Engineering Decision Log

1. **AppleSupport as the brand:** large enough to provide substantial historical coverage while keeping the scope focused.
2. **14-intent taxonomy:** broad enough to cover recurring support themes without creating dozens of brittle classes.
3. **Direct response pairing:** simpler and auditable than attempting perfect reconstruction of every Twitter conversation.
4. **TF-IDF retrieval:** inexpensive, deterministic, and easy to inspect for a take-home evaluation.
5. **TF-IDF + Logistic Regression:** strong classical baseline with interpretable feature behavior and fast inference.
6. **Keyword baseline:** establishes whether the learned model improves over simple lexical heuristics.
7. **Majority baseline:** establishes the minimum bar and exposes class imbalance.
8. **Balanced classifier weights:** reduce the tendency to ignore minority intents.
9. **Historical-response grounding:** reduces hallucination risk compared with unconstrained free-form generation.
10. **Conservative low-similarity fallback:** avoids presenting weakly matched historical cases as reliable solutions.
11. **Security/billing/account escalation:** these cases may require information or actions unavailable to a dataset-only agent.
12. **Vague-message escalation:** ambiguity is safer to route to a human than to force a fine-grained automated answer.
13. **Holdout by tweet ID:** prevents the evaluation examples from being used in classifier training.
14. **Human validation of the judge:** the LLM judge was compared with a 30-example human-rated sample, and the low agreement was reported rather than hidden.
15. **Explicit headline-number caveat:** routing accuracy is reported together with escalation recall because aggregate accuracy can hide unsafe misses.

---

## 11. Limitations

The most significant limitations are:

- the evaluation set contains 200 examples, with some minority intents represented by very few examples;
- classifier training labels are automatically generated;
- routing escalation recall is low;
- reply evaluation uses a 30-example human-rated comparison; human-vs-LLM judge agreement is low;
- historical retrieval uses directly linked responses rather than complete multi-turn state;
- the reply generator is conservative but not a full production LLM agent;
- no account, order, billing, or device APIs are available.

These limitations define exactly where the prototype needs additional work.

---

## 12. Reproducibility

### Fast verification of headline results

The submitted repository includes the 200-example golden set and saved prediction artifacts, so the headline intent and routing results can be verified without downloading the full TWCS dataset. The full raw dataset is only needed to rebuild the historical conversation index and retrain the prototype from scratch.

```powershell
python evaluation\evaluate_intent.py
python evaluation\evaluate_routing.py
```

Expected headline results:

```text
Intent accuracy: 0.3050
Routing accuracy: 0.6400
Escalation precision: 0.6897
Escalation recall: 0.2410
Escalation F1: 0.3571
```

This verification path is intended to stay within the assignment's 15-minute headline-result reproduction target on a normal development machine. A full rebuild from the TWCS raw dataset is a separate path and requires the dataset to be supplied locally.


The repository is designed so that a reviewer can reproduce the main experiments using the scripts under `src/` and `evaluation/`.

Key commands:

```powershell
python src/build_conversations.py
python src/create_golden_set.py
python src/prepare_golden_csv.py
python src/train_intent_classifier.py
python evaluation/baseline_intent.py
python evaluation/baseline_keyword.py
python evaluation/evaluate_intent.py
python evaluation/evaluate_routing.py
python evaluation/analyze_failures.py
python evaluation/llm_judge.py
python evaluation/compare_judge_review.py
```

The README contains the complete project structure, methodology, headline results, failure analysis, limitations, and decision log.

---

## 13. Repository Structure

```text
data/
  raw/
  processed/
  golden/

src/
  build_conversations.py
  create_golden_set.py
  draft_reply.py
  explore.py
  inspect_brand.py
  prepare_golden_csv.py
  retrieve_resolutions.py
  route_ticket.py
  sample_conversations.py
  support_agent.py
  train_intent_classifier.py

evaluation/
  analyze_failures.py
  baseline_intent.py
  baseline_keyword.py
  compare_judge_review.py
  evaluate_intent.py
  evaluate_routing.py
  llm_judge.py

results/
  failure_analysis.txt
  human_reply_review.csv
  intent_classifier.joblib
  intent_predictions.csv
  routing_predictions.csv
  top_failure_examples.csv
```

---

## 14. Conclusion

The project demonstrates an end-to-end, reproducible support-agent prototype over real customer-support data. The learned intent model beats both trivial and keyword baselines, historical retrieval provides concrete grounding evidence, and the router explicitly distinguishes cases that should not be automatically handled.

The evaluation also shows why the system is not production-ready: intent performance remains weak on ambiguous and minority classes, and routing recall for escalation is only 24.1%. The reply evaluation is based on a 30-example human-rated sample, and the low human-vs-LLM judge agreement limits how much the automated judge can be trusted.

The strongest result of this take-home is therefore not a single accuracy number. It is a transparent experimental pipeline that makes its assumptions, evidence, failure modes, and next improvements visible.
