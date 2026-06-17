import { useState } from "react";

const TABS = ["Implementation roadmap", "Experimental tracks", "Folder structure"];

const PHASES = [
  {
    id: "P1", label: "Phase 1", title: "Data preparation",
    color: "#185FA5", bg: "#E6F1FB",
    modules: [
      {
        id: "M1.1", title: "Raw data loading & deduplication",
        notebook: "01_data_audit.ipynb",
        src: "src/data/loader.py + cleaner.py",
        steps: [
          "Load CSV with pandas — verify 235,795 rows × 56 columns",
          "Confirm zero missing values across all columns",
          "Identify 425 duplicate URLs using URL as deduplication key",
          "Remove duplicates keeping first occurrence → 235,370 rows",
          "Confirm class balance post-dedup: phishing / legitimate ratio",
          "Save clean deduplicated frame to data/processed/clean_df.csv",
        ],
        inputs: "PhiUSIIL_Phishing_URL_Dataset.csv",
        output: "clean_df.csv (235,370 rows × 56 cols)",
      },
      {
        id: "M1.2", title: "Column removal & feature set finalization",
        notebook: "01_data_audit.ipynb",
        src: "src/data/cleaner.py",
        steps: [
          "Drop FILENAME — internal identifier, no predictive value",
          "Drop URL — raw string, identifier; duplicates already resolved",
          "Drop Domain — raw string derived directly from URL",
          "Drop Title — free multilingual text, not a structured feature",
          "Drop URLTitleMatchScore — r=0.961 with DomainTitleMatchScore (multicollinearity)",
          "Confirm remaining: 51 features + 1 label column",
          "Save final feature list to config/feature_config.yaml",
          "Document dropped columns + rationale in README",
        ],
        inputs: "clean_df.csv",
        output: "51 usable features + label; feature_config.yaml",
      },
    ],
  },
  {
    id: "P2", label: "Phase 2", title: "Exploratory data analysis",
    color: "#533AB7", bg: "#EEEDFE",
    modules: [
      {
        id: "M2.1", title: "Comprehensive EDA",
        notebook: "02_eda.ipynb",
        src: "src/utils/visualizer.py",
        steps: [
          "Class distribution bar chart — confirm label=0 phishing / label=1 legitimate",
          "Per-feature histogram + KDE split by class for all 51 features",
          "Full correlation heatmap (51×51) — flag pairs with |r| > 0.7",
          "URLSimilarityIndex isolation plot — confirm label=1 all exactly 100.0",
          "IsHTTPS cross-tab — confirm label=1 all HTTPS=1",
          "TLD frequency analysis: top 30 TLDs + phishing rate per TLD group",
          "Outlier boxplots for count features: LineOfCode, NoOfCSS, NoOfJS, NoOfImage, NoOfExternalRef",
          "Skewness table for all numeric features — identify log-transform candidates",
          "Binary feature rates by class — identify near-perfect separators",
          "Save all EDA plots to outputs/plots/eda/",
        ],
        inputs: "clean_df.csv (51 features)",
        output: "EDA plots + skewness table + leakage confirmation",
      },
    ],
  },
  {
    id: "P3", label: "Phase 3", title: "Preprocessing pipeline",
    color: "#0F6E56", bg: "#E1F5EE",
    modules: [
      {
        id: "M3.1", title: "Preprocessing pipeline construction",
        notebook: "03_preprocessing.ipynb",
        src: "src/features/preprocessing.py + encoding.py + engineering.py",
        steps: [
          "TLD frequency encoding: top 50 TLDs kept by name; remainder → 'rare_tld' bucket",
          "Ordinal-encode TLD using frequency rank (higher freq = lower rank integer)",
          "Log1p transform skewed count features: URLLength, DomainLength, LineOfCode, NoOfLettersInURL, NoOfDegitsInURL, NoOfOtherSpecialCharsInURL, NoOfImage, NoOfCSS, NoOfJS, NoOfSelfRef, NoOfEmptyRef, NoOfExternalRef, NoOfPopup, NoOfiFrame, LargestLineLength",
          "Cap each continuous feature at its own P99.9 before scaling",
          "Apply RobustScaler to all non-binary continuous features (IQR-robust; handles residual outliers)",
          "Binary features (19 cols): no scaling required — remain 0/1",
          "Engineered features (add 6 derived cols):",
          "  → ContentComplexityScore = log1p(NoOfCSS + NoOfJS + NoOfImage)",
          "  → FormDangerIndex = HasExternalFormSubmit + HasHiddenFields + HasPasswordField",
          "  → TrustBadgeScore = HasFavicon + Robots + HasCopyrightInfo + HasSocialNet",
          "  → RedirectActivity = NoOfURLRedirect + NoOfSelfRedirect",
          "  → ExternalRefDensity = NoOfExternalRef / (NoOfSelfRef + 1)",
          "  → TLD_is_gov_edu = binary flag for .gov/.edu/.mil TLDs",
          "Wrap all steps in sklearn Pipeline — single .fit_transform() call on train, .transform() on test",
          "Save fitted preprocessors: outputs/preprocessors/preprocessor_A.pkl and preprocessor_B.pkl",
        ],
        inputs: "clean_df.csv",
        output: "Fitted sklearn Pipeline; 57 final features (51 + 6 engineered)",
      },
    ],
  },
  {
    id: "P4", label: "Phase 4", title: "Experimental track setup",
    color: "#854F0B", bg: "#FAEEDA",
    modules: [
      {
        id: "M4.1", title: "Track A and Track B split definition",
        notebook: "03_preprocessing.ipynb",
        src: "src/data/splitter.py",
        steps: [
          "Define Track A feature set: all 57 features including URLSimilarityIndex",
          "Define Track B feature set: Track A minus URLSimilarityIndex → 56 features",
          "Stratified 80/20 train/test split — split ONCE before any model sees data",
          "Both tracks use identical row indices for train and test sets",
          "y_train and y_test are shared across tracks (same split, different X feature columns)",
          "Confirm class ratio preserved: ~57% legitimate in both train and test subsets",
          "Save to data/processed/track_A/ and data/processed/track_B/",
          "Save y_train.csv and y_test.csv to data/processed/y_split.csv",
          "Log split sizes and class balance to outputs/reports/split_summary.txt",
        ],
        inputs: "Preprocessed feature matrix",
        output: "X_train_A/B, X_test_A/B, y_train, y_test",
      },
    ],
  },
  {
    id: "P5", label: "Phase 5", title: "Model training (6 × 2 tracks)",
    color: "#993C1D", bg: "#FAECE7",
    modules: [
      {
        id: "M5.1", title: "Train all 6 models on Track A",
        notebook: "04_model_training.ipynb",
        src: "src/models/*.py + model_trainer.py",
        steps: [
          "LogisticRegression: C=1.0, max_iter=1000, class_weight='balanced', solver='lbfgs'",
          "DecisionTreeClassifier: max_depth=8, min_samples_leaf=50, class_weight='balanced'",
          "RandomForestClassifier: n_estimators=500, max_depth=None, n_jobs=-1, class_weight='balanced'",
          "XGBClassifier: n_estimators=500, learning_rate=0.05, max_depth=6, scale_pos_weight=ratio",
          "LGBMClassifier: n_estimators=500, learning_rate=0.05, max_depth=-1, class_weight='balanced'",
          "GradientBoostingClassifier: n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8",
          "Each model: 5-fold stratified CV on training set only — record CV mean and std per metric",
          "Final refit on full X_train_A after CV",
          "Save 6 model artifacts to outputs/models/track_A/",
        ],
        inputs: "X_train_A, y_train",
        output: "6 .pkl model files in outputs/models/track_A/",
      },
      {
        id: "M5.2", title: "Train all 6 models on Track B",
        notebook: "04_model_training.ipynb",
        src: "src/models/*.py + model_trainer.py",
        steps: [
          "Identical training procedure as M5.1 — same hyperparameters",
          "Input: X_train_B (URLSimilarityIndex excluded)",
          "This is the deployment-realistic scenario — leakage-free",
          "5-fold stratified CV on X_train_B for each of the 6 models",
          "Final refit on full X_train_B after CV",
          "Save 6 model artifacts to outputs/models/track_B/",
          "Note: expected performance drop vs Track A quantifies leakage contribution",
        ],
        inputs: "X_train_B, y_train",
        output: "6 .pkl model files in outputs/models/track_B/",
      },
    ],
  },
  {
    id: "P6", label: "Phase 6", title: "Evaluation & model selection",
    color: "#3B6D11", bg: "#EAF3DE",
    modules: [
      {
        id: "M6.1", title: "5-metric evaluation + best model selection",
        notebook: "05_evaluation.ipynb",
        src: "src/evaluation/metrics.py + selector.py",
        steps: [
          "Evaluate all 6 models × 2 tracks on held-out test set (X_test_A/B, y_test)",
          "Metric 1 — Accuracy: correct predictions / total predictions",
          "Metric 2 — Precision: per-class (phishing / legitimate) + weighted average",
          "Metric 3 — Recall: per-class + weighted average — critical for phishing miss rate",
          "Metric 4 — F1-Score: per-class + macro + weighted average",
          "Metric 5 — ROC AUC: primary ranking metric using predict_proba scores",
          "Bonus metrics: MCC (Matthew's Corr. Coeff.), Average Precision, Brier Score",
          "Generate full confusion matrix for each of the 12 model-track combinations",
          "Generate classification report (sklearn) for each combination",
          "Build unified comparison DataFrame: 12 rows × 8 metric columns",
          "Best model selection rule: highest ROC AUC on Track B (leakage-free); F1 as tiebreaker",
          "Save comparison table to outputs/reports/model_comparison.csv",
          "Save confusion matrices to outputs/plots/evaluation/",
        ],
        inputs: "All 12 trained models + X_test_A/B + y_test",
        output: "model_comparison.csv + best_model_A + best_model_B identified",
      },
    ],
  },
  {
    id: "P7", label: "Phase 7", title: "SHAP explainability",
    color: "#533AB7", bg: "#EEEDFE",
    modules: [
      {
        id: "M7.1", title: "SHAP global and local explanations",
        notebook: "06_shap.ipynb",
        src: "src/explainability/shap_explainer.py",
        steps: [
          "Initialize TreeExplainer for RF, XGBoost, LGBM, GB, DT (native tree SHAP)",
          "Initialize LinearExplainer for Logistic Regression",
          "GLOBAL — Track A: Compute SHAP values on full X_test_A",
          "  → Summary beeswarm plot (all features, all test samples)",
          "  → Bar plot of mean(|SHAP|) per feature — global ranking",
          "  → Dependence plot for top 5 features (marginal effect + interaction coloring)",
          "GLOBAL — Track B: Repeat above on X_test_B (without URLSimilarityIndex)",
          "  → Compare feature rankings Track A vs B — quantify leakage shift",
          "LOCAL — Select representative samples: 5 TP, 5 TN, 5 FP, 5 FN from test set",
          "  → Waterfall plot per sample — shows per-feature additive contribution",
          "  → Force plot per sample — alternative compact view",
          "INTERACTION — Compute SHAP interaction values for best model on Track B",
          "  → Top 5 feature-pair interaction heatmap",
          "  → Interaction dependence: IsHTTPS × SpacialCharRatioInURL",
          "Save all plots to outputs/plots/explainability/shap/track_A/ and track_B/",
        ],
        inputs: "Best model + X_test_A/B + y_test",
        output: "SHAP plots (global + local + interaction) for both tracks",
      },
    ],
  },
  {
    id: "P8", label: "Phase 8", title: "LIME explainability",
    color: "#993556", bg: "#FBEAF0",
    modules: [
      {
        id: "M8.1", title: "LIME local explanations + SHAP agreement",
        notebook: "07_lime.ipynb",
        src: "src/explainability/lime_explainer.py",
        steps: [
          "Initialize LimeTabularExplainer with X_train_B as background (Track B only)",
          "Set feature names, class names ['phishing', 'legitimate'], mode='classification'",
          "Select 40 representative test samples: 10 TP, 10 TN, 10 FP, 10 FN",
          "Generate LIME explanation for each sample (num_features=10, num_samples=3000)",
          "Plot per-sample top-10 feature contributions (positive = phishing, negative = legitimate)",
          "Compute SHAP vs LIME agreement rate:",
          "  → For each sample: rank top-5 features by |contribution| in SHAP and LIME",
          "  → Agreement = overlap in top-5 feature sets / 5",
          "  → Report mean agreement rate across all 40 samples",
          "Identify high-disagreement samples (SHAP-LIME agreement < 0.4) for blind spot analysis",
          "Save LIME plots to outputs/plots/explainability/lime/",
          "Save agreement rate summary to outputs/reports/shap_lime_agreement.csv",
        ],
        inputs: "Best model (Track B) + X_train_B + 40 selected test samples",
        output: "LIME plots + SHAP-LIME agreement rate report",
      },
    ],
  },
  {
    id: "P9", label: "Phase 9", title: "Bias analysis",
    color: "#854F0B", bg: "#FAEEDA",
    modules: [
      {
        id: "M9.1", title: "TLD group bias analysis",
        notebook: "08_bias_analysis.ipynb",
        src: "src/bias/tld_bias.py",
        steps: [
          "Define 5 TLD groups using tld_group_mapping.json:",
          "  → Commercial: .com .net .biz .info .mobi",
          "  → Gov / Edu: .gov .mil .edu .ac",
          "  → Country-code: .uk .de .au .ca .fr .jp .it .nl .br .ru .in (top 20 ccTLDs)",
          "  → New generic: .app .io .dev .tech .online .site .store .co",
          "  → Suspicious/free: .cf .gq .ga .ml .tk .top .icu .link .xyz .click",
          "For each TLD group: filter test set samples in that group",
          "Compute per-group: Accuracy, Precision, Recall, F1, FPR, FNR",
          "FPR = legitimate sites flagged as phishing (over-blocking harm)",
          "FNR = phishing sites missed (under-detection harm)",
          "Disparate impact ratio: min(group FPR) / max(group FPR) — flag if < 0.8",
          "Equalized odds check: compare FPR and FNR across groups",
          "Generate grouped bar chart: FPR and FNR by TLD group side-by-side",
          "Run on both Track A and Track B — compare bias levels",
          "Save to outputs/plots/bias/tld_bias/ and outputs/reports/bias_report.csv",
        ],
        inputs: "Best model predictions on test set + TLD group mapping",
        output: "Per-group metrics table + disparate impact scores + TLD bias plots",
      },
      {
        id: "M9.2", title: "HTTPS group bias analysis",
        notebook: "08_bias_analysis.ipynb",
        src: "src/bias/https_bias.py",
        steps: [
          "Split test set into two groups: IsHTTPS=1 (HTTPS sites) and IsHTTPS=0 (HTTP sites)",
          "Compute per-group: Accuracy, Precision, Recall, F1, FPR, FNR",
          "Critical subgroup: phishing sites WITH HTTPS (IsHTTPS=1 AND label=0)",
          "  → These are the hardest cases — attackers mimicking legitimate security signals",
          "  → Compute FNR within this subgroup specifically",
          "Critical subgroup: legitimate sites WITHOUT HTTPS (IsHTTPS=0 AND label=1)",
          "  → These may be unfairly flagged — older but real sites",
          "  → Compute FPR within this subgroup specifically",
          "Generate 2×2 confusion matrix breakdown: HTTPS vs HTTP × phishing vs legitimate",
          "Stacked bar chart: error type distribution by HTTPS group",
          "Run on both Track A (leaky) and Track B — compare over-reliance on IsHTTPS",
          "Save to outputs/plots/bias/https_bias/",
        ],
        inputs: "Best model predictions + IsHTTPS feature values",
        output: "HTTPS group metrics + subgroup FNR/FPR + HTTPS bias plots",
      },
      {
        id: "M9.3", title: "URL length group bias analysis",
        notebook: "08_bias_analysis.ipynb",
        src: "src/bias/url_length_bias.py",
        steps: [
          "Define URL length bins from dataset percentiles:",
          "  → Short: URLLength ≤ 23 (Q1)",
          "  → Medium: 23 < URLLength ≤ 27 (Q1–Q2)",
          "  → Long: 27 < URLLength ≤ 34 (Q2–Q3)",
          "  → Very long: URLLength > 34 (above Q3)",
          "Compute per-bin: Accuracy, Precision, Recall, F1, FPR, FNR",
          "Check if short/long URLs have systematically different error rates",
          "Generate line plot: FPR and FNR as a function of URL length bin",
          "Scatter plot: URLLength vs prediction confidence, colored by correct/wrong",
          "Identify if model under-performs on extremely long URLs (potential evasion vector)",
          "Save to outputs/plots/bias/url_length_bias/",
        ],
        inputs: "Best model predictions + URLLength feature values",
        output: "URL length group metrics + length-error correlation + plots",
      },
    ],
  },
  {
    id: "P10", label: "Phase 10", title: "Blind spot analysis",
    color: "#A32D2D", bg: "#FCEBEB",
    modules: [
      {
        id: "M10.1", title: "False positive (FP) deep-dive",
        notebook: "09_blindspot_analysis.ipynb",
        src: "src/blindspot/fp_analyzer.py",
        steps: [
          "Extract all FP cases from test set: legitimate sites misclassified as phishing",
          "Profile FP samples by TLD group — which TLD groups have highest FP concentration?",
          "Profile FP samples by HTTPS status — are HTTP legitimate sites over-flagged?",
          "Profile FP samples by URL length bin — do short URLs cause more FPs?",
          "Compute feature value distributions: FP samples vs correctly-classified positives (TP)",
          "Run SHAP analysis on FP subset: what features pushed prediction toward phishing?",
          "Rank top-5 features by mean |SHAP| within FP sample pool",
          "Identify recurring patterns: e.g., 'legitimate .top domains with no social net'",
          "Generate FP profile summary table: TLD group × HTTPS × avg confidence score",
          "Plot: feature value distribution FPs vs TPs for top-3 discriminating features",
          "Save to outputs/plots/blindspot/fp_analysis/",
        ],
        inputs: "Best model predictions + test set features + SHAP values",
        output: "FP profile table + FP SHAP analysis + FP pattern summary",
      },
      {
        id: "M10.2", title: "False negative (FN) deep-dive",
        notebook: "09_blindspot_analysis.ipynb",
        src: "src/blindspot/fn_analyzer.py",
        steps: [
          "Extract all FN cases from test set: phishing sites misclassified as legitimate",
          "Profile FN samples by TLD group — which TLDs produce the most undetected phishing?",
          "Profile FN samples by HTTPS status — HTTPS phishing sites that evaded detection",
          "Profile FN samples by URL length — short phishing URLs that look benign?",
          "Compute feature value distributions: FN samples vs correctly-classified negatives (TN)",
          "Run SHAP analysis on FN subset: what features made them appear legitimate?",
          "Rank top-5 features by contribution toward 'legitimate' class within FN pool",
          "Identify evasion patterns: e.g., 'phishing sites with social media links + HTTPS'",
          "Adversarial insight: list which 3–5 features a sophisticated attacker should mimic",
          "Generate FN profile summary table: TLD group × HTTPS × avg confidence score",
          "Plot: feature value distribution FNs vs TNs for top-3 evasion features",
          "Save to outputs/plots/blindspot/fn_analysis/",
          "Write blind spot report: most vulnerable subgroups + recommended detection gaps",
        ],
        inputs: "Best model predictions + test set features + SHAP values",
        output: "FN profile table + evasion pattern summary + blindspot_report.html",
      },
    ],
  },
  {
    id: "P11", label: "Phase 11", title: "Final report generation",
    color: "#444441", bg: "#F1EFE8",
    modules: [
      {
        id: "M11.1", title: "Report compilation",
        notebook: "10_final_report.ipynb",
        src: "src/reporting/report_generator.py",
        steps: [
          "Consolidate model_comparison.csv → formatted table Track A vs B × 6 models",
          "Best model identification section: winner on Track B + rationale",
          "Leakage quantification: AUC delta between Track A and Track B for best model",
          "SHAP summary: top-10 features by global importance (Track B)",
          "LIME summary: per-class top features + SHAP agreement rate",
          "Bias audit summary: flagged groups, disparate impact scores, recommendations",
          "Blind spot summary: FP patterns, FN patterns, adversarial vulnerabilities",
          "Embed all key plots inline as base64 PNG in HTML report",
          "Generate final_report.html to outputs/reports/",
          "Generate PDF version if required",
        ],
        inputs: "All outputs from phases 5–10",
        output: "final_report.html + bias_report.html + blindspot_report.html",
      },
    ],
  },
];

const TRACK_A = {
  label: "Track A",
  subtitle: "Full feature set",
  color: "#854F0B",
  bg: "#FAEEDA",
  features: 57,
  includes: "URLSimilarityIndex",
  purpose: "Establish ceiling performance and quantify leakage contribution",
  featureNote: "URLSimilarityIndex = 100 for ALL legitimate records",
  models: [
    { name: "Logistic Regression", note: "C=1.0, class_weight balanced" },
    { name: "Decision Tree", note: "max_depth=8, min_samples_leaf=50" },
    { name: "Random Forest", note: "n_estimators=500, class_weight balanced" },
    { name: "XGBoost", note: "n_estimators=500, lr=0.05, max_depth=6" },
    { name: "LightGBM", note: "n_estimators=500, lr=0.05, class_weight balanced" },
    { name: "Gradient Boosting", note: "n_estimators=300, lr=0.05, max_depth=5" },
  ],
  selection: "Best model = highest ROC AUC on test set",
  interpretation: "High AUC here is partially inflated by URLSimilarityIndex leakage",
};

const TRACK_B = {
  label: "Track B",
  subtitle: "Leakage-aware model",
  color: "#0F6E56",
  bg: "#E1F5EE",
  features: 56,
  includes: "—",
  purpose: "Deployment-realistic benchmark — honest performance without leaky feature",
  featureNote: "URLSimilarityIndex removed; all other 56 features retained",
  models: [
    { name: "Logistic Regression", note: "C=1.0, class_weight balanced" },
    { name: "Decision Tree", note: "max_depth=8, min_samples_leaf=50" },
    { name: "Random Forest", note: "n_estimators=500, class_weight balanced" },
    { name: "XGBoost", note: "n_estimators=500, lr=0.05, max_depth=6" },
    { name: "LightGBM", note: "n_estimators=500, lr=0.05, class_weight balanced" },
    { name: "Gradient Boosting", note: "n_estimators=300, lr=0.05, max_depth=5" },
  ],
  selection: "Best model = highest ROC AUC on test set (primary report benchmark)",
  interpretation: "SHAP + LIME + bias analysis all run primarily on Track B winner",
};

const FOLDER_TREE = [
  { type: "dir", name: "phishing_detection/", depth: 0, open: true },
  { type: "dir", name: "config/", depth: 1 },
  { type: "file", name: "experiment_config.yaml", depth: 2, note: "Track A/B definitions, split params" },
  { type: "file", name: "feature_config.yaml", depth: 2, note: "Feature sets, drop list, group maps" },
  { type: "file", name: "model_config.yaml", depth: 2, note: "Hyperparameters for all 6 models" },
  { type: "dir", name: "data/", depth: 1 },
  { type: "dir", name: "raw/", depth: 2 },
  { type: "file", name: "PhiUSIIL_Phishing_URL_Dataset.csv", depth: 3, note: "Original dataset — never modified" },
  { type: "dir", name: "processed/", depth: 2 },
  { type: "file", name: "clean_df.csv", depth: 3, note: "Deduped, identifiers dropped" },
  { type: "file", name: "y_split.csv", depth: 3, note: "y_train + y_test (shared across tracks)" },
  { type: "dir", name: "track_A/", depth: 3 },
  { type: "file", name: "X_train.csv", depth: 4, note: "57 features" },
  { type: "file", name: "X_test.csv", depth: 4, note: "57 features" },
  { type: "dir", name: "track_B/", depth: 3 },
  { type: "file", name: "X_train.csv", depth: 4, note: "56 features (no URLSimilarityIndex)" },
  { type: "file", name: "X_test.csv", depth: 4, note: "56 features" },
  { type: "dir", name: "metadata/", depth: 2 },
  { type: "file", name: "feature_groups.json", depth: 3, note: "Binary, count, ratio, categorical types" },
  { type: "file", name: "tld_group_mapping.json", depth: 3, note: "TLD → group (commercial/gov/country/suspicious)" },
  { type: "dir", name: "notebooks/", depth: 1 },
  { type: "file", name: "01_data_audit.ipynb", depth: 2, note: "M1.1 + M1.2" },
  { type: "file", name: "02_eda.ipynb", depth: 2, note: "M2.1" },
  { type: "file", name: "03_preprocessing.ipynb", depth: 2, note: "M3.1 + M4.1" },
  { type: "file", name: "04_model_training.ipynb", depth: 2, note: "M5.1 + M5.2" },
  { type: "file", name: "05_evaluation.ipynb", depth: 2, note: "M6.1" },
  { type: "file", name: "06_shap.ipynb", depth: 2, note: "M7.1" },
  { type: "file", name: "07_lime.ipynb", depth: 2, note: "M8.1" },
  { type: "file", name: "08_bias_analysis.ipynb", depth: 2, note: "M9.1 + M9.2 + M9.3" },
  { type: "file", name: "09_blindspot_analysis.ipynb", depth: 2, note: "M10.1 + M10.2" },
  { type: "file", name: "10_final_report.ipynb", depth: 2, note: "M11.1" },
  { type: "dir", name: "outputs/", depth: 1 },
  { type: "dir", name: "models/", depth: 2 },
  { type: "dir", name: "track_A/", depth: 3 },
  { type: "file", name: "logistic_regression.pkl", depth: 4 },
  { type: "file", name: "decision_tree.pkl", depth: 4 },
  { type: "file", name: "random_forest.pkl", depth: 4 },
  { type: "file", name: "xgboost.pkl", depth: 4 },
  { type: "file", name: "lightgbm.pkl", depth: 4 },
  { type: "file", name: "gradient_boosting.pkl", depth: 4 },
  { type: "dir", name: "track_B/", depth: 3, note: "Same 6 .pkl files" },
  { type: "dir", name: "preprocessors/", depth: 2 },
  { type: "file", name: "preprocessor_A.pkl", depth: 3, note: "Fitted sklearn Pipeline (57 features)" },
  { type: "file", name: "preprocessor_B.pkl", depth: 3, note: "Fitted sklearn Pipeline (56 features)" },
  { type: "dir", name: "plots/", depth: 2 },
  { type: "dir", name: "eda/", depth: 3, note: "Histograms, KDE, correlation heatmap" },
  { type: "dir", name: "evaluation/", depth: 3, note: "Confusion matrices, ROC curves" },
  { type: "dir", name: "explainability/", depth: 3 },
  { type: "dir", name: "shap/track_A/", depth: 4, note: "Beeswarm, waterfall, dependence plots" },
  { type: "dir", name: "shap/track_B/", depth: 4, note: "Primary SHAP output" },
  { type: "dir", name: "lime/", depth: 4, note: "Per-sample LIME explanations" },
  { type: "dir", name: "bias/", depth: 3 },
  { type: "dir", name: "tld_bias/", depth: 4, note: "Per-TLD-group FPR, FNR charts" },
  { type: "dir", name: "https_bias/", depth: 4, note: "HTTPS vs HTTP error breakdown" },
  { type: "dir", name: "url_length_bias/", depth: 4, note: "Length bin error distribution" },
  { type: "dir", name: "blindspot/", depth: 3 },
  { type: "dir", name: "fp_analysis/", depth: 4, note: "FP profiles and SHAP on FPs" },
  { type: "dir", name: "fn_analysis/", depth: 4, note: "FN profiles and evasion patterns" },
  { type: "dir", name: "reports/", depth: 2 },
  { type: "file", name: "model_comparison.csv", depth: 3, note: "12-row metric comparison table" },
  { type: "file", name: "bias_report.html", depth: 3, note: "TLD + HTTPS + URL length bias findings" },
  { type: "file", name: "blindspot_report.html", depth: 3, note: "FP + FN patterns + recommendations" },
  { type: "file", name: "final_report.html", depth: 3, note: "Complete project report" },
  { type: "dir", name: "src/", depth: 1 },
  { type: "dir", name: "data/", depth: 2 },
  { type: "file", name: "loader.py", depth: 3, note: "load_dataset(), validate_shape()" },
  { type: "file", name: "cleaner.py", depth: 3, note: "remove_duplicates(), drop_columns()" },
  { type: "file", name: "splitter.py", depth: 3, note: "stratified_split(), define_tracks()" },
  { type: "dir", name: "features/", depth: 2 },
  { type: "file", name: "preprocessing.py", depth: 3, note: "build_pipeline(), fit_transform()" },
  { type: "file", name: "encoding.py", depth: 3, note: "tld_frequency_encode(), log1p_transform()" },
  { type: "file", name: "engineering.py", depth: 3, note: "add_derived_features() — 6 engineered cols" },
  { type: "dir", name: "models/", depth: 2 },
  { type: "file", name: "logistic_regression.py", depth: 3 },
  { type: "file", name: "decision_tree.py", depth: 3 },
  { type: "file", name: "random_forest.py", depth: 3 },
  { type: "file", name: "xgboost_model.py", depth: 3 },
  { type: "file", name: "lightgbm_model.py", depth: 3 },
  { type: "file", name: "gradient_boosting.py", depth: 3 },
  { type: "dir", name: "evaluation/", depth: 2 },
  { type: "file", name: "metrics.py", depth: 3, note: "compute_all_metrics(), confusion_report()" },
  { type: "file", name: "selector.py", depth: 3, note: "select_best_model(), compare_tracks()" },
  { type: "dir", name: "explainability/", depth: 2 },
  { type: "file", name: "shap_explainer.py", depth: 3, note: "global_shap(), local_shap(), interaction_shap()" },
  { type: "file", name: "lime_explainer.py", depth: 3, note: "lime_local(), compute_agreement_rate()" },
  { type: "dir", name: "bias/", depth: 2 },
  { type: "file", name: "tld_bias.py", depth: 3, note: "tld_group_metrics(), disparate_impact()" },
  { type: "file", name: "https_bias.py", depth: 3, note: "https_group_metrics(), subgroup_fnr()" },
  { type: "file", name: "url_length_bias.py", depth: 3, note: "length_bin_metrics()" },
  { type: "dir", name: "blindspot/", depth: 2 },
  { type: "file", name: "fp_analyzer.py", depth: 3, note: "extract_fps(), profile_fps(), shap_fps()" },
  { type: "file", name: "fn_analyzer.py", depth: 3, note: "extract_fns(), profile_fns(), evasion_patterns()" },
  { type: "dir", name: "reporting/", depth: 2 },
  { type: "file", name: "report_generator.py", depth: 3, note: "compile_html_report(), embed_plots()" },
  { type: "dir", name: "utils/", depth: 2 },
  { type: "file", name: "config.py", depth: 3, note: "load_config(), get_feature_list()" },
  { type: "file", name: "logger.py", depth: 3, note: "setup_logger(), log_metrics()" },
  { type: "file", name: "visualizer.py", depth: 3, note: "save_fig(), styled_heatmap()" },
  { type: "dir", name: "tests/", depth: 1 },
  { type: "file", name: "test_features.py", depth: 2, note: "Pipeline output shape/type checks" },
  { type: "file", name: "test_models.py", depth: 2, note: "Model fit/predict smoke tests" },
  { type: "file", name: "test_evaluation.py", depth: 2, note: "Metric computation correctness" },
  { type: "file", name: "test_bias.py", depth: 2, note: "Group metric calculation checks" },
  { type: "file", name: "main.py", depth: 1, note: "End-to-end pipeline runner (all phases)" },
  { type: "file", name: "requirements.txt", depth: 1, note: "pandas, sklearn, xgboost, lightgbm, shap, lime, matplotlib, seaborn" },
  { type: "file", name: "README.md", depth: 1, note: "Setup, usage, project rationale" },
];

function ModuleCard({ m, color, bg }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ border: "0.5px solid var(--color-border-tertiary)", borderRadius: "var(--border-radius-lg)", marginBottom: 8, overflow: "hidden" }}>
      <button onClick={() => setOpen(!open)} style={{ width: "100%", background: open ? bg : "var(--color-background-primary)", border: "none", padding: "10px 14px", cursor: "pointer", display: "flex", alignItems: "center", gap: 10, textAlign: "left" }}>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, color, minWidth: 42, fontWeight: 500 }}>{m.id}</span>
        <span style={{ fontSize: 13, fontWeight: 500, flex: 1 }}>{m.title}</span>
        <span style={{ fontSize: 11, color: "var(--color-text-tertiary)", fontFamily: "var(--font-mono)", marginRight: 8 }}>{m.notebook}</span>
        <i className={`ti ${open ? "ti-chevron-up" : "ti-chevron-down"}`} style={{ fontSize: 13, color: "var(--color-text-tertiary)" }} aria-hidden />
      </button>
      {open && (
        <div style={{ borderTop: "0.5px solid var(--color-border-tertiary)", padding: "12px 14px" }}>
          <div style={{ display: "flex", gap: 8, marginBottom: 10, flexWrap: "wrap" }}>
            <span style={{ fontSize: 11, fontFamily: "var(--font-mono)", padding: "2px 7px", borderRadius: 4, background: bg, color }}>src: {m.src}</span>
          </div>
          <div style={{ marginBottom: 8 }}>
            {m.steps.map((s, i) => (
              <div key={i} style={{ display: "flex", gap: 8, padding: "3px 0", fontSize: 12, color: s.startsWith("  →") ? "var(--color-text-tertiary)" : "var(--color-text-secondary)" }}>
                {!s.startsWith("  →") && <span style={{ color, flexShrink: 0, marginTop: 1 }}>→</span>}
                <span style={{ paddingLeft: s.startsWith("  →") ? 16 : 0 }}>{s.startsWith("  →") ? s.slice(3) : s}</span>
              </div>
            ))}
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginTop: 10 }}>
            <div style={{ padding: "7px 10px", background: "var(--color-background-secondary)", borderRadius: "var(--border-radius-md)" }}>
              <div style={{ fontSize: 10, color: "var(--color-text-tertiary)", marginBottom: 2 }}>INPUT</div>
              <div style={{ fontSize: 11, color: "var(--color-text-secondary)" }}>{m.inputs}</div>
            </div>
            <div style={{ padding: "7px 10px", background: bg, borderRadius: "var(--border-radius-md)" }}>
              <div style={{ fontSize: 10, color, marginBottom: 2, fontWeight: 500 }}>OUTPUT</div>
              <div style={{ fontSize: 11, color: "var(--color-text-secondary)" }}>{m.output}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function TreeRow({ item }) {
  const indent = item.depth * 16;
  const isDir = item.type === "dir";
  return (
    <div style={{ display: "flex", alignItems: "baseline", gap: 6, padding: "2px 0", paddingLeft: indent }}>
      <i className={`ti ${isDir ? "ti-folder" : "ti-file"}`} style={{ fontSize: 13, color: isDir ? "#854F0B" : "var(--color-text-tertiary)", flexShrink: 0 }} aria-hidden />
      <span style={{ fontFamily: "var(--font-mono)", fontSize: 12, color: isDir ? "var(--color-text-primary)" : "var(--color-text-secondary)", fontWeight: isDir ? 500 : 400 }}>{item.name}</span>
      {item.note && <span style={{ fontSize: 11, color: "var(--color-text-tertiary)" }}>— {item.note}</span>}
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState(0);
  const [openPhase, setOpenPhase] = useState(0);

  return (
    <div style={{ fontFamily: "var(--font-sans)", color: "var(--color-text-primary)", maxWidth: 680, margin: "0 auto", paddingBottom: "2rem" }}>
      <h2 style={{ fontSize: 0, margin: 0 }}>Phishing detection implementation plan</h2>

      <div style={{ borderBottom: "0.5px solid var(--color-border-tertiary)", marginBottom: "1.25rem" }}>
        <div style={{ display: "flex", gap: 4, marginBottom: -1 }}>
          {TABS.map((t, i) => (
            <button key={i} onClick={() => setTab(i)} style={{ background: "none", border: "none", padding: "8px 12px", fontSize: 13, color: tab === i ? "var(--color-text-primary)" : "var(--color-text-secondary)", borderBottom: tab === i ? "2px solid var(--color-text-primary)" : "2px solid transparent", cursor: "pointer", fontFamily: "var(--font-sans)" }}>{t}</button>
          ))}
        </div>
      </div>

      {tab === 0 && (
        <div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8, marginBottom: "1.5rem" }}>
            {[["11 phases", "sequential"], ["15 modules", "total"], ["2 tracks", "A and B"], ["6 models", "per track"]].map(([v, l], i) => (
              <div key={i} style={{ background: "var(--color-background-secondary)", borderRadius: "var(--border-radius-md)", padding: "10px 12px" }}>
                <div style={{ fontSize: 18, fontWeight: 500 }}>{v}</div>
                <div style={{ fontSize: 11, color: "var(--color-text-secondary)" }}>{l}</div>
              </div>
            ))}
          </div>

          {PHASES.map((phase, pi) => (
            <div key={pi} style={{ marginBottom: 12 }}>
              <button onClick={() => setOpenPhase(openPhase === pi ? -1 : pi)} style={{ width: "100%", background: openPhase === pi ? phase.bg : "var(--color-background-secondary)", border: "0.5px solid var(--color-border-tertiary)", borderRadius: "var(--border-radius-md)", padding: "9px 14px", cursor: "pointer", display: "flex", alignItems: "center", gap: 10, textAlign: "left" }}>
                <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, fontWeight: 500, color: phase.color, minWidth: 28 }}>{phase.id}</span>
                <span style={{ fontSize: 13, fontWeight: 500, flex: 1 }}>{phase.title}</span>
                <span style={{ fontSize: 11, color: "var(--color-text-tertiary)" }}>{phase.modules.length} module{phase.modules.length > 1 ? "s" : ""}</span>
                <i className={`ti ${openPhase === pi ? "ti-chevron-up" : "ti-chevron-down"}`} style={{ fontSize: 13, color: "var(--color-text-tertiary)" }} aria-hidden />
              </button>
              {openPhase === pi && (
                <div style={{ padding: "10px 0 0" }}>
                  {phase.modules.map((m, mi) => (
                    <ModuleCard key={mi} m={m} color={phase.color} bg={phase.bg} />
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {tab === 1 && (
        <div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: "1.5rem" }}>
            {[TRACK_A, TRACK_B].map((track, ti) => (
              <div key={ti} style={{ border: ti === 1 ? `2px solid ${track.color}` : "0.5px solid var(--color-border-tertiary)", borderRadius: "var(--border-radius-lg)", overflow: "hidden" }}>
                <div style={{ background: track.bg, padding: "12px 14px", borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 3 }}>
                    <span style={{ fontSize: 15, fontWeight: 500 }}>{track.label}</span>
                    {ti === 1 && <span style={{ fontSize: 10, padding: "2px 7px", borderRadius: 3, background: track.bg, color: track.color, border: `0.5px solid ${track.color}` }}>primary benchmark</span>}
                  </div>
                  <div style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>{track.subtitle}</div>
                </div>
                <div style={{ padding: "12px 14px" }}>
                  <div style={{ fontSize: 11, color: "var(--color-text-secondary)", marginBottom: 10 }}>{track.purpose}</div>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10, fontSize: 12 }}>
                    <span style={{ color: "var(--color-text-tertiary)" }}>Feature count</span>
                    <span style={{ fontFamily: "var(--font-mono)", fontWeight: 500 }}>{track.features}</span>
                  </div>
                  <div style={{ fontSize: 11, padding: "6px 8px", background: ti === 0 ? "#FCEBEB" : "#E1F5EE", borderRadius: "var(--border-radius-md)", color: ti === 0 ? "#A32D2D" : "#0F6E56", marginBottom: 12 }}>{track.featureNote}</div>
                  <div style={{ fontSize: 11, fontWeight: 500, color: "var(--color-text-secondary)", marginBottom: 6 }}>Models</div>
                  {track.models.map((m, mi) => (
                    <div key={mi} style={{ fontSize: 12, padding: "4px 0", borderBottom: "0.5px solid var(--color-border-tertiary)", display: "flex", justifyContent: "space-between" }}>
                      <span>{m.name}</span>
                      <span style={{ fontSize: 10, color: "var(--color-text-tertiary)", fontFamily: "var(--font-mono)" }}>{m.note}</span>
                    </div>
                  ))}
                  <div style={{ marginTop: 10, padding: "7px 8px", background: "var(--color-background-secondary)", borderRadius: "var(--border-radius-md)", fontSize: 11, color: "var(--color-text-secondary)" }}>
                    <span style={{ fontWeight: 500, color: track.color }}>Selection: </span>{track.selection}
                  </div>
                  <div style={{ marginTop: 6, fontSize: 11, color: "var(--color-text-tertiary)", fontStyle: "italic" }}>{track.interpretation}</div>
                </div>
              </div>
            ))}
          </div>

          <div style={{ border: "0.5px solid var(--color-border-tertiary)", borderRadius: "var(--border-radius-lg)", padding: "14px 16px" }}>
            <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 10 }}>Shared infrastructure (both tracks)</div>
            {[
              ["Split", "Identical stratified 80/20 — same row indices for train and test"],
              ["CV", "5-fold stratified cross-validation — applied on training set only"],
              ["Metrics", "Accuracy, Precision, Recall, F1, ROC AUC — evaluated on test set"],
              ["Preprocessor", "Separate fitted Pipeline per track — prevents any data leakage"],
              ["XAI", "SHAP + LIME run on Track B winner (deployment-realistic model)"],
              ["Bias analysis", "All 3 bias dimensions applied to Track B winner"],
              ["Blind spot", "FP/FN analysis on Track B winner's test set predictions"],
            ].map(([k, v], i) => (
              <div key={i} style={{ display: "flex", gap: 12, padding: "5px 0", borderBottom: i < 6 ? "0.5px solid var(--color-border-tertiary)" : "none", fontSize: 12 }}>
                <span style={{ fontWeight: 500, minWidth: 96, color: "var(--color-text-primary)" }}>{k}</span>
                <span style={{ color: "var(--color-text-secondary)" }}>{v}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {tab === 2 && (
        <div>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: 12, border: "0.5px solid var(--color-border-tertiary)", borderRadius: "var(--border-radius-lg)", padding: "16px" }}>
            {FOLDER_TREE.map((item, i) => <TreeRow key={i} item={item} />)}
          </div>
        </div>
      )}
    </div>
  );
}
