<div align="center">

# 🛡️ Explainable & Bias-Aware Phishing Website Detection

### Explainable Cybersecurity Intelligence using Machine Learning, SHAP, LIME, Fairness Analysis, and Trustworthy AI

<p align="center">
<img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge">
<img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?style=for-the-badge">
<img src="https://img.shields.io/badge/SHAP-Explainable%20AI-green?style=for-the-badge">
<img src="https://img.shields.io/badge/LIME-Interpretability-success?style=for-the-badge">
<img src="https://img.shields.io/badge/Cybersecurity-Phishing%20Detection-red?style=for-the-badge">
<img src="https://img.shields.io/badge/Fairness-Bias%20Analysis-purple?style=for-the-badge">
</p>

### Building Cybersecurity Systems That Security Analysts Can Understand, Trust, and Validate

Unlike traditional phishing detection systems that stop at prediction, this project investigates **why a website is classified as phishing, whether the decision is trustworthy, whether bias exists, and what hidden weaknesses remain inside the model.**

</div>

---

# 🚀 Project Highlights

| Category                         | Result                        |
| -------------------------------- | ----------------------------- |
| Dataset                          | PhiUSIIL Phishing URL Dataset |
| Total Samples                    | 235,795                       |
| Original Features                | 56                            |
| Final Deployment Features        | 56                            |
| Models Trained                   | 4                             |
| ROC-AUC                          | ~1.000                        |
| Accuracy                         | ~99.98–100%                   |
| F1 Score                         | ~99.98–100%                   |
| SHAP Features Explained          | 56                            |
| SHAP-LIME Agreement              | 52%                           |
| Feature Consistency Score        | 60%                           |
| Most Biased Dimension            | TLD                           |
| SHAP Rank Shifts Identified      | 540                           |
| Performance Investigation Charts | 7                             |

---

# 🎯 Project Goal

Phishing websites remain one of the most common cybersecurity threats.

Attackers continuously create fake websites designed to steal:

* User Credentials
* Banking Information
* Personal Data
* Corporate Assets
* Sensitive Organizational Information

Most machine learning systems focus on answering only one question:

> Is this website phishing?

This project goes much further.

It attempts to answer:

> Why was the website classified as phishing?

> Which features influenced the decision?

> Can the explanation be trusted?

> Is the model biased?

> Why does the model achieve near-perfect performance?

> Where could the model still fail?

The objective is to build a **Trustworthy Cybersecurity Intelligence Framework** rather than a simple phishing classifier.

---

# 🔍 What Makes This Project Different?

Traditional phishing detection projects typically follow:

```text
Website
    ↓
Machine Learning Model
    ↓
Prediction
```

This project extends the pipeline significantly:

```text
Website
    ↓
Feature Engineering
    ↓
Machine Learning
    ↓
Prediction
    ↓
SHAP Explainability
    ↓
LIME Explainability
    ↓
Bias Analysis
    ↓
Performance Investigation
    ↓
Blind Spot Discovery
    ↓
Cybersecurity Intelligence
```

The focus is not merely achieving high accuracy.

The focus is understanding, validating, and trusting every model decision.

---

# 🏗️ System Architecture

```text
┌──────────────────────┐
│   Website Features   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Feature Engineering  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Machine Learning     │
│ Models               │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Prediction Engine    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ SHAP Analysis        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ LIME Analysis        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Bias Investigation   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Performance          │
│ Investigation        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Security Intelligence│
└──────────────────────┘
```

---

# 📊 Dataset

## PhiUSIIL Phishing URL Dataset

A large-scale cybersecurity dataset containing phishing and legitimate websites represented through URL, domain, content, behavioral, and security-related indicators.

The dataset provides a rich collection of features suitable for phishing website detection, explainability analysis, and fairness evaluation.

### Dataset Summary

| Metric              | Value         |
| ------------------- | ------------- |
| Total Samples       | 235,795       |
| Legitimate Websites | 134,850       |
| Phishing Websites   | 100,945       |
| Original Features   | 56            |
| Missing Values      | 0             |
| Target Variable     | Label         |
| Domain              | Cybersecurity |

### Dataset Source

PhiUSIIL Phishing URL Dataset

https://www.kaggle.com/datasets/kaggleprollc/phishing-url-websites-dataset-phiusiil

### Dataset Distribution

<div align="center">

<img src="outputs/plots/eda/class_distribution.png" width="320">

</div>

<p align="center">
Class Distribution of Legitimate and Phishing Websites
</p>

---

# 🔬 Feature Audit & Leakage Investigation

Before any model training, a complete feature audit was performed.

The audit investigated:

* Feature Quality
* Data Leakage
* Multicollinearity
* Redundant Variables
* Dominant Predictors
* Feature Trustworthiness

### Key Findings

* URLSimilarityIndex showed extremely strong predictive influence.
* HTTPS was highly separative.
* Several features demonstrated dominance behavior.
* URLTitleMatchScore was removed due to redundancy.
* Separate experimental tracks were created to evaluate potential leakage effects.

This step ensured that performance results could be interpreted responsibly rather than blindly trusted.

---
# 📈 Exploratory Data Analysis

Before model development, a comprehensive exploratory analysis was conducted to understand the structure, distribution, and predictive characteristics of the dataset.

The analysis focused on:

* Class Distribution
* Feature Correlations
* Feature Dominance
* Data Leakage Indicators
* Statistical Characteristics
* Predictive Signals

---

### Dataset Distribution & Feature Relationships

<div align="center">

<img src="outputs/plots/eda/class_distribution.png" width="320">
<img src="outputs/plots/eda/target_correlation_full.png" width="450">

</div>

<p align="center">
Dataset Distribution and Feature Correlation Analysis
</p>

---

### Correlation Analysis

<div align="center">

<img src="outputs/plots/eda/correlation_heatmap_pearson.png" width="500">

</div>

<p align="center">
Pearson Correlation Heatmap
</p>

The analysis revealed several highly influential features and helped identify potential redundancy and leakage risks before model training.

---

# 🔬 Feature Leakage Investigation

One of the most important phases of this project involved determining whether exceptionally strong performance was caused by genuine learning or by highly dominant features.

Instead of blindly trusting accuracy metrics, the project investigated:

* Feature Leakage
* Artificial Performance Inflation
* Dominant Predictors
* Dataset Separability
* Trustworthiness of Results

---

### Key Discovery

The feature:

**URLSimilarityIndex**

demonstrated exceptionally strong predictive influence and required dedicated investigation.

To evaluate its impact responsibly:

| Track                | URLSimilarityIndex |
| -------------------- | ------------------ |
| Track A              | Included           |
| Track B              | Removed            |
| Deployment Benchmark | Track B            |

This approach allowed the project to compare performance under both conditions and better understand the influence of dominant features.

---

<div align="center">

<img src="outputs/plots/eda/leakage_urlsimilarity.png" width="380">
<img src="outputs/plots/eda/leakage_auroc_comparison.png" width="380">

</div>

<p align="center">
Leakage Investigation and Performance Impact Analysis
</p>

---

# ⚙️ Feature Engineering

To capture higher-level cybersecurity patterns, several engineered features were introduced.

### Engineered Features

| Feature                |
| ---------------------- |
| ContentComplexityScore |
| FormDangerIndex        |
| TrustBadgeScore        |
| RedirectActivity       |
| ExternalRefDensity     |
| TLD_is_gov_edu         |
| SubdomainRatio         |

These features provide richer representations of website behavior and improve the model's ability to distinguish phishing websites from legitimate ones.

---

### Final Feature Sets

| Track   | Features |
| ------- | -------- |
| Track A | 57       |
| Track B | 56       |

---

# 🤖 Machine Learning Pipeline

Multiple machine learning algorithms were trained and benchmarked to identify the most reliable phishing detection model.

### Models Evaluated

| Model               |
| ------------------- |
| Logistic Regression |
| Random Forest       |
| XGBoost             |
| LightGBM            |

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Cross Validation Performance

The goal was not simply selecting the highest-performing model, but identifying a model suitable for explainability, fairness analysis, and trustworthy deployment.

---

<div align="center">

<img src="outputs/plots/training/benchmark_roc_auc.png" width="380">
<img src="outputs/plots/training/benchmark_f1.png" width="380">

</div>

<p align="center">
ROC-AUC and F1 Score Benchmarking
</p>

---

<div align="center">

<img src="outputs/plots/training/benchmark_cv_roc_auc.png" width="450">

</div>

<p align="center">
Cross-Validation Performance Analysis
</p>

---

# 🧠 Explainable Artificial Intelligence

Most phishing detection systems behave as black boxes.

This project incorporates Explainable AI techniques to make model decisions transparent and interpretable.

Two complementary explainability frameworks were employed:

---

## SHAP

SHAP provides a global understanding of model behavior.

It answers:

> Which features influence predictions the most?

> How important is each feature across the entire dataset?

### SHAP Findings

| Metric                          | Result           |
| ------------------------------- | ---------------- |
| Features Explained              | 56               |
| Top Feature                     | LetterRatioInURL |
| Interaction Pairs Evaluated     | 1,540            |
| URLSimilarityIndex Contribution | 18.68%           |

---

<div align="center">

<img src="outputs/plots/shap/global_importance.png" width="380">
<img src="outputs/plots/shap/summary_beeswarm.png" width="380">

</div>

<p align="center">
SHAP Global Importance and Summary Analysis
</p>

---

<div align="center">

<img src="outputs/plots/shap/track_comparison_importance.png" width="450">

</div>

<p align="center">
Feature Importance Across Experimental Tracks
</p>

---

# 🔎 Local Explainability with LIME

While SHAP explains overall model behavior, LIME focuses on individual predictions.

LIME helps answer:

> Why was this specific website classified as phishing?

By generating local explanations, security analysts can understand the reasoning behind individual model decisions.

---

# 🔬 SHAP × LIME Validation

Interpretability itself should be validated.

To improve explanation reliability, SHAP and LIME explanations were compared across multiple prediction scenarios.

This evaluation focused on:

* Explanation Consistency
* Feature Agreement
* Reliability of Interpretations
* High-Disagreement Cases

---

### Results

| Metric                  | Value      |
| ----------------------- | ---------- |
| SHAP-LIME Agreement     | 0.52       |
| Feature Consistency     | 0.60       |
| Shared Top Features     | 12 / 20    |
| High Disagreement Cases | Identified |

---

<div align="center">

<img src="outputs/plots/lime/shap_lime_agreement.png" width="380">
<img src="outputs/plots/lime/feature_consistency.png" width="380">

</div>

<p align="center">
SHAP-LIME Agreement and Feature Consistency Analysis
</p>

---

# 🚀 Future Roadmap

### 🎯 M10 — Blind Spot Intelligence

The next phase focuses on discovering hidden model weaknesses through:

* Failure Archetype Discovery
* Blind Spot Severity Ranking
* Confidence Reliability Analysis
* SHAP–LIME Disagreement Investigation
* Hidden Weakness Detection

**Goal:** Understand where the model fails, why it fails, and how risky those failures are.

---

### 🛡️ M11 — Cybersecurity Intelligence Dashboard

The final phase transforms the project into an interactive intelligence platform featuring:

* Threat Detection Center
* SHAP & LIME Explainability Center
* Bias Intelligence Center
* Blind Spot Intelligence Center
* Performance Intelligence Center

**Goal:** Build a transparent, explainable, and trustworthy cybersecurity intelligence system.

---


### Why This Matters

Many phishing detection projects stop after generating predictions.

This project evaluates whether the explanations themselves can be trusted.

The result is a cybersecurity intelligence framework capable of providing both accurate predictions and transparent reasoning.

---
# 👨‍💻 Contributors

## Hifza Amir

**B.Tech CSE (Data Science)**  
IILM University, Greater Noida

**Areas of Interest**

- Data Science
- Machine Learning
- Explainable AI
- Predictive Analytics
- Trustworthy AI

GitHub:
https://github.com/hiifza

---

## Shihan Ahmad

**B.Tech CSE (Cybersecurity)**

**Areas of Interest**

- Cybersecurity
- Threat Intelligence
- Security Analytics
- Network Security

GitHub:
https://github.com/ShihanG9

<div align="center">

# 🌟 Vision

Building cybersecurity systems that are not only accurate but also transparent, interpretable, fair, and trustworthy.

### Explainable AI × Cybersecurity × Trustworthy Machine Learning

This project demonstrates how modern machine learning can move beyond black-box predictions and provide meaningful explanations, fairness insights, and actionable intelligence for security analysts.

⭐ If you found this project useful, consider starring the repository.

</div>
