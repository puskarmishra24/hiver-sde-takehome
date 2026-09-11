# Apple Support AI Agent

An AI customer-support agent built from the Customer Support on Twitter dataset. The system classifies incoming customer messages, retrieves historically similar Apple Support resolutions, drafts a grounded reply, and decides whether a ticket should be auto-handled or escalated to a human.

## 1. Problem Framing

The goal is not simply to maximize classification accuracy. A useful support agent needs to:

1. Identify the customer's likely issue.
2. Retrieve evidence from previously resolved support conversations.
3. Produce a useful response grounded in that evidence.
4. Avoid confidently auto-handling cases that require human intervention.

I selected **AppleSupport** as the target brand.

The system separates intent understanding, historical resolution retrieval, reply drafting, and automation/routing safety.

## 2. System Overview

```text
Customer message
       |
       v
+----------------------+
| Intent classifier    |
| TF-IDF + LogisticReg |
+----------------------+
       |
       +-----------------------+
       |                       |
       v                       v
 Predicted intent       Historical retrieval
                               |
                               v
                       Top similar resolutions
                               |
                               v
                        Grounded reply draft
                               |
                               v
                         Routing policy
                          /           \
                         /             \
                 Auto-handle         Escalate
```

## 3. Dataset

### Submission repository data note

The full TWCS raw dataset (`data/raw/twcs.csv`) and the generated historical conversation file (`data/processed/apple_conversations.jsonl`) are **not included in the submitted GitHub repository because of GitHub's web-upload file-size limit**. A small sample is included for inspection, and the repository retains the scripts needed to regenerate the processed conversation file when the full TWCS dataset is supplied locally. These files were excluded only for submission packaging; the implementation and evaluation artifacts remain in the repository.


### Evaluation-set provenance

The 200-example golden evaluation set was personally reviewed and labelled by the candidate using the 14-intent taxonomy defined from the AppleSupport data. The set contains examples from all 14 intents and was held out from classifier training. The labels were used as the final evaluation reference set.

**Sampling and labelling:** I sampled 200 AppleSupport customer messages from the evaluation pool to cover all 14 intents, including ambiguous and difficult examples. I personally reviewed each example, assigned one intent from the taxonomy, and assigned an escalation label based on whether the issue could reasonably be handled from the available historical evidence or required human/account-specific intervention.


Source: Customer Support on Twitter dataset (`thoughtvector/customer-support-on-twitter`).

The full dataset contains approximately 2.8M tweets.

For this project:

| Dataset component | Count |
|---|---:|
| AppleSupport-related tweets | 204,772 |
| Extracted customer/support conversations | 78,406 |
| Golden evaluation examples | 200 |

Historical conversations were constructed by pairing customer tweets with directly linked AppleSupport responses.

## 4. Intent Taxonomy

The system uses 14 intents:

```text
software_update_issue
device_performance_issue
keyboard_input_issue
connectivity_issue
app_or_media_issue
apple_id_icloud_issue
payment_billing_issue
order_sales_issue
hardware_accessory_issue
security_phishing_issue
support_contact_issue
warranty_repair_recycling
product_feature_question
general_or_unclear
```

The 200-example golden set contains examples for **all 14 intents**. `hardware_accessory_issue` has 2 examples, so the class is represented but conclusions about that minority class should be treated cautiously.

## 5. Evaluation

### 5.1 Intent Classification

The trivial baseline always predicts the majority intent.

| System | Accuracy |
|---|---:|
| Majority-class baseline | 21.5% |
| Keyword baseline | 26.0% |
| TF-IDF + Logistic Regression | **30.5%** |

Majority intent: `device_performance_issue` (32 / 200 examples).

Classifier metrics:

| Metric | Score |
|---|---:|
| Accuracy | **0.3050** |
| Macro precision | **0.44** |
| Macro recall | **0.27** |
| Macro F1 | **0.29** |
| Weighted F1 | **0.35** |

The classifier improves over the majority baseline by **9.0 percentage points** (30.5% vs 21.5%), or approximately **1.42x** the baseline accuracy. It also improves over the keyword baseline by 4.5 percentage points.

### 5.2 Routing

| Metric | Result |
|---|---:|
| Accuracy | **64.0%** |
| Escalation precision | **69.0%** |
| Escalation recall | **24.1%** |
| Escalation F1 | **35.7%** |

Confusion matrix:

```text
                 Predicted
              Auto    Escalate

Expected
Auto           108         9
Escalate        63        20
```

The 63 false negatives are important: the system sometimes predicts automated handling when the golden label expects escalation.

Therefore, **64.0% routing accuracy should not be interpreted as 64.0% safe automation**.

## 6. Reply Quality

A 30-example human reply review produced:

| Dimension | Score |
|---|---:|
| Correctness | **3.80 / 5** |
| Helpfulness | **3.60 / 5** |
| Relevance | **4.67 / 5** |
| Groundedness | **4.67 / 5** |
| Overall | **3.73 / 5** |
| Pass rate | **66.7%** |

The candidate personally rated the same 30 examples using the reply-quality rubric. The LLM judge was evaluated against those human ratings using pass/fail agreement and rank correlation. The matched comparison produced **63.33% pass/fail agreement**, **Cohen's kappa of 0.1081**, and **Spearman correlation of 0.2242**. The low agreement indicates that the judge should not be treated as a substitute for human evaluation.

## 7. Top Failure Modes

### 1. General/unclear -> app_or_media_issue

22 examples.

Example:

> "@AppleSupport what is going on?"

Hypothesis: a bag-of-words classifier cannot infer an intent when the message depends on missing conversation context.

### 2. Product feature -> app_or_media_issue

11 examples.

Example:

> "Can landscape support be enabled in future versions of software?"

Hypothesis: lexical features capture topic vocabulary but not the customer's speech act.

### 3. Keyboard/input -> app_or_media_issue

10 examples.

Example:

> "when will my phone let me type regular capital i's again?"

Hypothesis: unusual wording and malformed text reduce lexical signal.

### 4. Device performance -> software update

10 examples.

Example:

> "Why is my phone even worse after the last iOS update I have to charge twice a day now"

Hypothesis: the message contains both a trigger and a symptom, while the taxonomy requires one label.

### 5. Apple ID/iCloud -> app_or_media_issue

8 examples.

Example:

> "I just got a new iPhone but ... EVERYTHING GOT DELETED"

Hypothesis: downstream symptoms can dominate lexical similarity even when the underlying cause involves account, iCloud, migration, or synchronization.

Detailed examples are in `results/failure_analysis.txt`.

## 8. What Is Misleading About My Headline Number?

The most tempting headline number is **30.5% intent accuracy**.

It is useful but incomplete because:

- the golden set contains only 200 examples;
- some minority intents have very few golden examples;
- classifier development/training labels were automatically assigned; the 200-example golden evaluation set was personally reviewed and labelled;
- accuracy treats all mistakes equally;
- production safety depends more heavily on high-risk false negatives.

The routing accuracy has a similar limitation.

Although routing accuracy is **64.0%**, escalation recall is only **24.1%**, with 63 false negatives.

The appropriate conclusion is:

> The prototype demonstrates useful signal and a workable architecture, but it is not ready for unattended production automation.

## 9. One More Week

1. Expand the golden set to 500+ examples with a written annotation guide, stratified sampling, and explicit edge-case rules.
2. Measure inter-annotator agreement.
3. Compare TF-IDF with sentence embeddings or a lightweight transformer.
4. Add character-level features and stronger text normalization.
5. Separate issue family, customer intent, and root cause.
6. Evaluate retrieval with Recall@1, Recall@3, and MRR.
7. Synthesize replies from multiple retrieved resolutions instead of copying one.
8. Calibrate routing using cost-sensitive evaluation.
9. Add stratified difficulty/risk slices to the expanded golden set.
10. Add temporal holdout evaluation.
11. Expand the human-rated sample and revalidate the LLM judge on a larger set.
12. Add regression tests for high-impact failures.

## 10. Engineering Decision Log

1. Selected AppleSupport because it provides a large historical support corpus.
2. Used directly linked customer-to-brand response pairs.
3. Defined a compact operational intent taxonomy.
4. Held golden examples out of classifier training.
5. Started with TF-IDF + Logistic Regression for speed and interpretability.
6. Added bigram features for multi-word support phrases.
7. Used balanced class weights for uneven intent frequencies.
8. Separated retrieval from classification.
9. Removed Twitter mentions and URLs during retrieval.
10. Added a retrieval threshold to avoid weak historical matches.
11. Added a conservative fallback when retrieval evidence is weak.
12. Explicitly escalated high-risk categories.
13. Evaluated routing separately from classification.
14. Inspected concrete failure examples.
15. Validated the LLM judge against a 30-example human-rated sample and reported the low agreement rather than treating the judge as ground truth.

## 11. Limitations

This is a take-home prototype rather than a production support system.

Main limitations:

- small 200-example golden set;
- very small support for some minority intents;
- automatically generated classifier training labels (the 200-example golden set was personally labelled);
- lexical rather than semantic classification;
- simple TF-IDF retrieval;
- limited retrieval-specific evaluation;
- low escalation recall;
- historical-response copying rather than full synthesis;
- limited 30-example human reply evaluation and low human-vs-LLM judge agreement;
- no temporal generalization benchmark.

## 12. Reproduction

### Fast verification of headline results

The submitted repository includes the 200-example golden set and the saved prediction artifacts, so the headline intent and routing results can be verified without downloading the full TWCS dataset. The full raw dataset is only needed to rebuild the historical conversation index and retrain the prototype from scratch.

Run:

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

This verification path is designed to stay within the assignment's 15-minute reproduction target on a normal development machine. A full rebuild from the TWCS raw dataset is a separate path and requires the dataset to be supplied locally.


Install dependencies:

```powershell
pip install -r requirements.txt
```

Build conversations:

```powershell
python src\build_conversations.py
```

Create development sample:

```powershell
python src\sample_conversations.py
```

Train classifier:

```powershell
python src\train_intent_classifier.py
```

Run majority baseline:

```powershell
python evaluation\baseline_intent.py
```

Evaluate intent:

```powershell
python evaluation\evaluate_intent.py
```

Expected:

```text
Accuracy: 0.3050
```

Evaluate routing:

```powershell
python evaluation\evaluate_routing.py
```

Expected:

```text
Routing accuracy: 0.6400
Escalation precision: 0.6897
Escalation recall: 0.2410
Escalation F1: 0.3571
```

Run failure analysis:

```powershell
python evaluation\analyze_failures.py
```

Run the support agent:

```powershell
python src\support_agent.py
```

LLM judge:

```powershell
python evaluation\llm_judge.py
python evaluation\compare_judge_review.py
```

The LLM judge requires `OPENAI_API_KEY`.

## 13. Repository Structure

```text
hiver-sde-takehome/
├── data/
│   ├── raw/
│   ├── processed/
│   └── golden/
├── src/
├── evaluation/
├── results/
├── report/
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

## 14. Conclusion

The prototype demonstrates a complete support-agent workflow:

```text
Classify
   ↓
Retrieve
   ↓
Draft
   ↓
Route
```

The classifier substantially outperforms the trivial majority baseline, while the failure analysis shows where lexical modeling breaks down.

The next priority is improving **supervision and evaluation**, followed by semantic understanding and cost-sensitive routing.

The current results support the architecture as a strong prototype while making clear what is required before autonomous deployment.
