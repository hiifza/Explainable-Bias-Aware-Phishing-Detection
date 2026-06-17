import { useState } from "react";

const TABS = ["Dataset Profile", "Feature Catalog", "Critical Flags", "Project Architecture"];

const FEATURE_CATALOG = [
  {
    category: "Identifiers / Metadata",
    color: "#888780",
    bg: "#F1EFE8",
    borderColor: "#B4B2A9",
    action: "DROP",
    actionColor: "#A32D2D",
    actionBg: "#FCEBEB",
    features: [
      { name: "FILENAME", type: "string", desc: "Source file reference (.txt ID)", note: "Identifier only" },
      { name: "URL", type: "string", desc: "Raw URL string", note: "425 duplicates detected" },
      { name: "Domain", type: "string", desc: "Extracted domain name", note: "Derived from URL" },
      { name: "Title", type: "string", desc: "Page title (multilingual)", note: "Free text, 197,874 unique" },
    ],
  },
  {
    category: "URL Structure",
    color: "#185FA5",
    bg: "#E6F1FB",
    borderColor: "#378ADD",
    action: "USE",
    actionColor: "#0F6E56",
    actionBg: "#E1F5EE",
    features: [
      { name: "URLLength", type: "int", desc: "Total URL character count", note: "Skewed — max 6097, P99=144. Log-transform" },
      { name: "DomainLength", type: "int", desc: "Domain name character count", note: "Range 4–110" },
      { name: "TLDLength", type: "int", desc: "TLD character count", note: "Range 2–13" },
      { name: "TLD", type: "string", desc: "Top-level domain", note: "695 unique — encode (frequency/target)" },
      { name: "IsDomainIP", type: "binary", desc: "Domain is raw IP address", note: "Only 638 positives (0.3%)" },
      { name: "NoOfSubDomain", type: "int", desc: "Number of subdomain levels", note: "Lowest label correlation (r=0.006)" },
      { name: "IsHTTPS", type: "binary", desc: "Uses HTTPS protocol", note: "⚠ ALL label=1 have IsHTTPS=1 — leakage risk" },
    ],
  },
  {
    category: "URL Statistical / Entropy",
    color: "#3B6D11",
    bg: "#EAF3DE",
    borderColor: "#639922",
    action: "USE",
    actionColor: "#0F6E56",
    actionBg: "#E1F5EE",
    features: [
      { name: "URLSimilarityIndex", type: "float", desc: "Similarity to known-legitimate URLs (0–100)", note: "🚨 ALL label=1 = 100.0 exactly — CRITICAL leakage" },
      { name: "CharContinuationRate", type: "float", desc: "Rate of sequential same-char runs", note: "r=0.47 with label" },
      { name: "TLDLegitimateProb", type: "float", desc: "Statistical legitimacy of TLD", note: "Max capped at 0.523" },
      { name: "URLCharProb", type: "float", desc: "Character probability in URL", note: "r=0.47 with label" },
    ],
  },
  {
    category: "URL Character Composition",
    color: "#3B6D11",
    bg: "#EAF3DE",
    borderColor: "#639922",
    action: "USE",
    actionColor: "#0F6E56",
    actionBg: "#E1F5EE",
    features: [
      { name: "NoOfLettersInURL", type: "int", desc: "Count of alphabetic characters", note: "Correlated with URLLength — consider ratio only" },
      { name: "LetterRatioInURL", type: "float", desc: "Fraction of letters in URL", note: "r=0.37 with label" },
      { name: "NoOfDegitsInURL", type: "int", desc: "Count of digit characters", note: "Typo in name (Degits)" },
      { name: "DegitRatioInURL", type: "float", desc: "Fraction of digits in URL", note: "r=0.43 with label — phishing uses more digits" },
      { name: "NoOfEqualsInURL", type: "int", desc: "Count of '=' characters", note: "Skewed — max 176" },
      { name: "NoOfQMarkInURL", type: "int", desc: "Count of '?' characters", note: "Max 4" },
      { name: "NoOfAmpersandInURL", type: "int", desc: "Count of '&' characters", note: "Low correlation (r=0.035)" },
      { name: "NoOfOtherSpecialCharsInURL", type: "int", desc: "Other special character count", note: "Max 499 — extreme outliers" },
      { name: "SpacialCharRatioInURL", type: "float", desc: "Special char fraction in URL", note: "r=0.53 with label — top predictor" },
    ],
  },
  {
    category: "Obfuscation",
    color: "#854F0B",
    bg: "#FAEEDA",
    borderColor: "#EF9F27",
    action: "USE",
    actionColor: "#0F6E56",
    actionBg: "#E1F5EE",
    features: [
      { name: "HasObfuscation", type: "binary", desc: "Binary: any obfuscated chars present", note: "Only 0.2% positives — highly imbalanced" },
      { name: "NoOfObfuscatedChar", type: "int", desc: "Count of obfuscated characters", note: "r=0.015 — very low signal" },
      { name: "ObfuscationRatio", type: "float", desc: "Obfuscated char fraction", note: "Max 0.348, very sparse" },
    ],
  },
  {
    category: "Page HTML Structure",
    color: "#533AB7",
    bg: "#EEEDFE",
    borderColor: "#7F77DD",
    action: "USE",
    actionColor: "#0F6E56",
    actionBg: "#E1F5EE",
    features: [
      { name: "LineOfCode", type: "int", desc: "Total HTML line count", note: "Extreme outliers — max 442,666. Log-transform + cap" },
      { name: "LargestLineLength", type: "int", desc: "Longest single HTML line (chars)", note: "Max 13.9M — severe outlier" },
      { name: "HasTitle", type: "binary", desc: "Page has <title> tag", note: "r=0.46" },
      { name: "DomainTitleMatchScore", type: "float", desc: "Domain–title text overlap score (0–100)", note: "r=0.58 — r=0.96 with URLTitleMatchScore" },
      { name: "URLTitleMatchScore", type: "float", desc: "URL–title text overlap score (0–100)", note: "⚠ r=0.96 with DomainTitleMatchScore — drop one" },
      { name: "HasFavicon", type: "binary", desc: "Has a favicon link", note: "r=0.49" },
      { name: "Robots", type: "binary", desc: "Has robots.txt", note: "r=0.39" },
      { name: "IsResponsive", type: "binary", desc: "Page uses responsive design", note: "r=0.55" },
    ],
  },
  {
    category: "Redirects & Popups",
    color: "#993C1D",
    bg: "#FAECE7",
    borderColor: "#D85A30",
    action: "USE",
    actionColor: "#0F6E56",
    actionBg: "#E1F5EE",
    features: [
      { name: "NoOfURLRedirect", type: "binary*", desc: "Has URL redirects (effectively 0/1)", note: "Only 2 unique values despite 'NoOf' name" },
      { name: "NoOfSelfRedirect", type: "binary*", desc: "Has self-redirects (effectively 0/1)", note: "Only 2 unique values" },
      { name: "NoOfPopup", type: "int", desc: "Count of popup triggers", note: "115 unique vals — max 602. Not truly binary" },
      { name: "NoOfiFrame", type: "int", desc: "Count of iframes", note: "119 unique vals — max 1602. Not truly binary" },
    ],
  },
  {
    category: "Forms & Interaction",
    color: "#993556",
    bg: "#FBEAF0",
    borderColor: "#D4537E",
    action: "USE",
    actionColor: "#0F6E56",
    actionBg: "#E1F5EE",
    features: [
      { name: "HasExternalFormSubmit", type: "binary", desc: "Form submits to external domain", note: "r=0.044 — low signal" },
      { name: "HasSubmitButton", type: "binary", desc: "Page has a submit button", note: "r=0.58 — strong signal" },
      { name: "HasHiddenFields", type: "binary", desc: "Page has hidden form fields", note: "r=0.51" },
      { name: "HasPasswordField", type: "binary", desc: "Page has password input field", note: "r=0.30" },
      { name: "HasDescription", type: "binary", desc: "Page has meta description", note: "r=0.69" },
    ],
  },
  {
    category: "Content & Trust Signals",
    color: "#0F6E56",
    bg: "#E1F5EE",
    borderColor: "#1D9E75",
    action: "USE",
    actionColor: "#0F6E56",
    actionBg: "#E1F5EE",
    features: [
      { name: "HasSocialNet", type: "binary", desc: "Links to social media present", note: "r=0.78 — 2nd highest predictor; label=1 = 79.5%" },
      { name: "HasCopyrightInfo", type: "binary", desc: "Page has copyright notice", note: "r=0.74 — label=1 = 80.8%" },
      { name: "Bank", type: "binary", desc: "Contains banking keywords", note: "r=0.13" },
      { name: "Pay", type: "binary", desc: "Contains payment keywords", note: "r=0.36" },
      { name: "Crypto", type: "binary", desc: "Contains cryptocurrency keywords", note: "r=0.11" },
    ],
  },
  {
    category: "External Resources",
    color: "#185FA5",
    bg: "#E6F1FB",
    borderColor: "#378ADD",
    action: "USE",
    actionColor: "#0F6E56",
    actionBg: "#E1F5EE",
    features: [
      { name: "NoOfImage", type: "int", desc: "Image element count", note: "Max 8,956 — heavy outliers. Log-transform" },
      { name: "NoOfCSS", type: "int", desc: "CSS file count", note: "Max 35,820 — extreme outliers. Skew=464" },
      { name: "NoOfJS", type: "int", desc: "JavaScript file count", note: "Max 6,957 — extreme outliers" },
      { name: "NoOfSelfRef", type: "int", desc: "Internal link count", note: "Max 27,397 — Log-transform" },
      { name: "NoOfEmptyRef", type: "int", desc: "Empty/null link count", note: "Max 4,887" },
      { name: "NoOfExternalRef", type: "int", desc: "External link count", note: "r=0.34 with label. Max 27,516" },
    ],
  },
];

const CRITICAL_FLAGS = [
  {
    level: "CRITICAL",
    color: "#A32D2D",
    bg: "#FCEBEB",
    border: "#E24B4A",
    icon: "ti-alert-triangle",
    title: "URLSimilarityIndex — Potential Data Leakage",
    body: "Every single label=1 (legitimate) record has URLSimilarityIndex = 100.000 exactly. No variance. This feature either encodes the label directly or was derived from a process that had access to the ground truth. If included in a model without investigation, it creates perfect or near-perfect separation and renders all other features meaningless. This must be isolated in a separate 'with-leakage' experiment and excluded from the primary evaluation.",
    action: "Run Experiment A (with) and Experiment B (without). Report both. Treat Experiment B as the honest performance benchmark.",
  },
  {
    level: "CRITICAL",
    color: "#A32D2D",
    bg: "#FCEBEB",
    border: "#E24B4A",
    icon: "ti-alert-triangle",
    title: "IsHTTPS — Perfect Separation for Legitimate Class",
    body: "IsHTTPS = 1 for 100% of label=1 (legitimate) records. 49.2% of phishing sites also use HTTPS. This feature is a strong but fragile predictor — it was historically valid, but modern phishing campaigns increasingly obtain SSL certificates. Including it uncritically in the model bakes in a stale assumption and contributes to IsHTTPS leakage inflating explainability SHAP values.",
    action: "Investigate temporal aspect of data collection. Include SHAP analysis of IsHTTPS contribution. Flag in bias report as a temporally-unstable feature.",
  },
  {
    level: "WARNING",
    color: "#854F0B",
    bg: "#FAEEDA",
    border: "#EF9F27",
    icon: "ti-alert-circle",
    title: "DomainTitleMatchScore ↔ URLTitleMatchScore — Severe Multicollinearity",
    body: "Pearson r = 0.961 between these two features. Including both in linear/regularized models degrades coefficient stability and inflates variance. SHAP values will split attribution between them unpredictably, harming explainability quality.",
    action: "Drop URLTitleMatchScore. Keep DomainTitleMatchScore. Or apply PCA to collapse both to a single 'Title Match' component. Document the decision.",
  },
  {
    level: "WARNING",
    color: "#854F0B",
    bg: "#FAEEDA",
    border: "#EF9F27",
    icon: "ti-alert-circle",
    title: "TLD Bias — Extreme Geographic and Organizational Skew",
    body: ".mil and .gov TLDs are 0% phishing in this dataset. .cf, .gq, .page, .icu are 99–100% phishing. A model will exploit TLD as a proxy for legitimacy in ways that may unfairly flag certain country-code TLDs (.ml = Mali, .cf = Central African Republic, .ga = Gabon) primarily because they were cheap/free during data collection, not because they are inherently malicious.",
    action: "Conduct fairness audit: compute FPR and FNR broken down by TLD group (commercial, education/gov, country-code, generic-new). Report disparate impact. Consider TLD-agnostic feature set as a sensitivity experiment.",
  },
  {
    level: "WARNING",
    color: "#854F0B",
    bg: "#FAEEDA",
    border: "#EF9F27",
    icon: "ti-alert-circle",
    title: "425 Duplicate URLs",
    body: "235,795 total rows vs 235,370 unique URLs = 425 duplicates. These may introduce data leakage across train/test splits if not handled before splitting.",
    action: "Deduplicate before any train/test split. Use URL as deduplication key, keeping the first occurrence. Re-check class balance after deduplication.",
  },
  {
    level: "INFO",
    color: "#185FA5",
    bg: "#E6F1FB",
    border: "#378ADD",
    icon: "ti-info-circle",
    title: "Extreme Outliers in Count Features",
    body: "Multiple count features have catastrophic max values far beyond P99.9: LineOfCode (max 442,666 vs P99=10,388), NoOfCSS (max 35,820 vs P99=51), LargestLineLength (max 13.9M vs P99=139K). These will dominate tree splits and destabilize scaling for distance-based models.",
    action: "Apply log1p transform to all NoOf* count features. Cap at P99.9 before scaling. Verify XGBoost/LGBM performance is unaffected (tree-based models are naturally robust, but caps help SHAP interpretation).",
  },
  {
    level: "INFO",
    color: "#185FA5",
    bg: "#E6F1FB",
    border: "#378ADD",
    icon: "ti-info-circle",
    title: "Mild Class Imbalance",
    body: "Label=1 (Legitimate): 57.19% (134,850). Label=0 (Phishing): 42.81% (100,945). This is mild imbalance — not severe enough to require SMOTE. Stratified K-fold is sufficient. Use class_weight='balanced' for Logistic Regression and SVM. Monitor F1-score and MCC in addition to accuracy.",
    action: "Use stratified 5-fold CV. Report F1 per class, macro-F1, MCC, and AUC-ROC for all models. Do not over-engineer the imbalance.",
  },
];

const ARCH_PHASES = [
  {
    phase: "01",
    title: "Data Audit & EDA",
    color: "#185FA5",
    bg: "#E6F1FB",
    modules: [
      "Duplicate URL detection and removal (425 rows)",
      "Class distribution and label meaning confirmation",
      "Per-feature distribution plots by class",
      "Correlation matrix — flag r > 0.85 pairs",
      "URLSimilarityIndex leakage investigation",
      "TLD distribution analysis — phishing rate by TLD",
      "Outlier quantification (P99, P99.9, max)",
    ],
    output: "EDA report + confirmed leakage inventory",
  },
  {
    phase: "02",
    title: "Preprocessing Pipeline",
    color: "#0F6E56",
    bg: "#E1F5EE",
    modules: [
      "Drop 4 identifiers: FILENAME, URL, Domain, Title",
      "Deduplicate on URL before any split",
      "TLD encoding: frequency encoding (rare < 100 → 'rare_tld')",
      "Drop URLTitleMatchScore (r=0.96 multicollinearity)",
      "Log1p transform: URLLength, LineOfCode, all NoOf* count features",
      "Outlier cap: P99.9 per feature before scaling",
      "RobustScaler for continuous features (IQR-robust)",
      "Stratified 80/20 train-test split (deduplicated first)",
    ],
    output: "Preprocessed feature matrix — 52 engineered features",
  },
  {
    phase: "03",
    title: "Feature Engineering",
    color: "#3B6D11",
    bg: "#EAF3DE",
    modules: [
      "ContentComplexityScore = log(NoOfCSS + NoOfJS + NoOfImage + 1)",
      "FormDangerIndex = HasExternalFormSubmit + HasHiddenFields + HasPasswordField",
      "TrustBadgeScore = HasFavicon + Robots + HasCopyrightInfo + HasSocialNet",
      "RedirectActivity = NoOfURLRedirect + NoOfSelfRedirect",
      "TLD category flags: is_gov_edu, is_com_org, is_country, is_new_generic",
      "ExternalRefDensity = NoOfExternalRef / (NoOfSelfRef + 1)",
      "SpecialCharDiversity = unique special char types (from URL string)",
    ],
    output: "Augmented feature set (~59 features)",
  },
  {
    phase: "04",
    title: "Leakage-Aware Experiments",
    color: "#854F0B",
    bg: "#FAEEDA",
    modules: [
      "Experiment A — Full feature set (URLSimilarityIndex + IsHTTPS included)",
      "Experiment B — Remove URLSimilarityIndex and IsHTTPS (deployment-realistic)",
      "Experiment C — Remove all features with r > 0.5 (adversarial stress test)",
      "Track AUC-ROC, F1, MCC per experiment across all models",
      "SHAP-based leakage quantification: contribution of leaky features vs rest",
    ],
    output: "Leakage impact quantification table",
  },
  {
    phase: "05",
    title: "Model Training (6 Models)",
    color: "#533AB7",
    bg: "#EEEDFE",
    models: [
      { name: "Logistic Regression", type: "Interpretable Baseline", note: "L2 regularization, class_weight='balanced'" },
      { name: "Decision Tree", type: "Rule Extraction", note: "Max depth 6–8, extract IF-THEN rule set" },
      { name: "Random Forest", type: "Ensemble Baseline", note: "500 trees, OOB score, permutation importance" },
      { name: "XGBoost / LightGBM", type: "Primary Performance Model", note: "Native SHAP support, early stopping, cross-val" },
      { name: "Explainable Boosting Machine (EBM)", type: "Inherently Interpretable", note: "InterpretML — additive model + interaction terms" },
      { name: "MLP Neural Network", type: "Deep Baseline", note: "3-layer, dropout, BatchNorm — SHAP DeepExplainer" },
    ],
    output: "Trained model artifacts + validation scores",
  },
  {
    phase: "06",
    title: "XAI Explainability Layer",
    color: "#993C1D",
    bg: "#FAECE7",
    xai: [
      { method: "SHAP (TreeExplainer / DeepExplainer)", scope: "Global + Local", details: "Summary beeswarm, waterfall per prediction, SHAP interaction values for top feature pairs" },
      { method: "LIME", scope: "Local", details: "Per-prediction explanations, compare agreement rate with SHAP to measure consistency" },
      { method: "Partial Dependence Plots (PDP/ICE)", scope: "Global", details: "Marginal effect of top 5 features — URLSimilarityIndex, HasSocialNet, SpacialCharRatioInURL, HasCopyrightInfo, DomainTitleMatchScore" },
      { method: "Decision Tree Rules", scope: "Global", details: "Human-readable IF-THEN rule extraction from DT, validated against XGBoost SHAP" },
      { method: "EBM Feature Graphs", scope: "Global", details: "Per-feature contribution curves from InterpretML — most readable for non-technical stakeholders" },
      { method: "SHAP Interaction Values", scope: "Global", details: "Detect feature synergies: IsHTTPS × URLSimilarityIndex, SpacialCharRatioInURL × HasPasswordField" },
    ],
    output: "Explanation dashboard + per-prediction explanation API",
  },
  {
    phase: "07",
    title: "Bias Analysis",
    color: "#993556",
    bg: "#FBEAF0",
    modules: [
      "TLD group fairness audit — FPR and FNR by TLD type",
      "Disparate impact index per TLD group (threshold 0.8)",
      "Equalized odds check: .mil/.gov vs .cf/.gq TLD groups",
      "Temporal feature stability — assess which features degrade over time",
      "Adversarial evasion analysis — which features can attackers easily manipulate?",
      "SHAP-based leakage attribution — what % of model decision comes from leaky features?",
      "Subgroup accuracy breakdown: HTTPS vs non-HTTPS phishing sites",
    ],
    output: "Bias report with fairness metrics + mitigation recommendations",
  },
  {
    phase: "08",
    title: "Evaluation & Comparison",
    color: "#888780",
    bg: "#F1EFE8",
    metrics: [
      { name: "AUC-ROC", desc: "Primary ranking metric" },
      { name: "Macro F1-Score", desc: "Balanced across both classes" },
      { name: "Matthews Correlation Coeff (MCC)", desc: "Robust to mild imbalance" },
      { name: "Average Precision (AP)", desc: "Precision-recall AUC" },
      { name: "False Negative Rate", desc: "Missed phishing (critical)" },
      { name: "False Positive Rate", desc: "Legitimate flagged as phishing" },
      { name: "Calibration (Brier Score)", desc: "Probability reliability" },
      { name: "SHAP Agreement (SHAP vs LIME)", desc: "Explanation stability" },
    ],
    output: "Full model comparison table across 3 experiments × 6 models",
  },
  {
    phase: "09",
    title: "Deployment Prototype",
    color: "#185FA5",
    bg: "#E6F1FB",
    modules: [
      "FastAPI REST endpoint: POST /predict { url: string }",
      "Real-time feature extraction from URL string (URL-based features only)",
      "Model inference + probability score + class label",
      "SHAP explanation JSON: top-5 contributing features per prediction",
      "Confidence level: High / Medium / Low based on probability margin",
      "Monitoring hook: log predictions for distribution drift detection",
    ],
    output: "Deployable API + interactive demo UI",
  },
];

const statCards = [
  { label: "Total records", value: "235,795", sub: "after dedup: 235,370" },
  { label: "Feature columns", value: "56", sub: "52 usable, 4 identifiers" },
  { label: "Target: phishing", value: "100,945", sub: "label = 0 → 42.81%" },
  { label: "Target: legitimate", value: "134,850", sub: "label = 1 → 57.19%" },
  { label: "Missing values", value: "0", sub: "complete dataset" },
  { label: "Binary features", value: "19", sub: "incl. target" },
  { label: "Count features", value: "17", sub: "NoOf* family" },
  { label: "Ratio / prob features", value: "14", sub: "continuous float" },
];

const topCorr = [
  { name: "URLSimilarityIndex", r: 0.86, note: "🚨 leakage" },
  { name: "HasSocialNet", r: 0.78, note: "" },
  { name: "HasCopyrightInfo", r: 0.74, note: "" },
  { name: "HasDescription", r: 0.69, note: "" },
  { name: "IsHTTPS", r: 0.61, note: "⚠ leakage" },
  { name: "DomainTitleMatchScore", r: 0.58, note: "" },
  { name: "HasSubmitButton", r: 0.58, note: "" },
  { name: "IsResponsive", r: 0.55, note: "" },
  { name: "URLTitleMatchScore", r: 0.54, note: "⚠ multicollinear" },
  { name: "SpacialCharRatioInURL", r: 0.53, note: "" },
];

export default function App() {
  const [tab, setTab] = useState(0);
  const [openCat, setOpenCat] = useState(null);
  const [openPhase, setOpenPhase] = useState(null);

  return (
    <div style={{ fontFamily: "var(--font-sans)", color: "var(--color-text-primary)", maxWidth: 680, margin: "0 auto", padding: "0 0 2rem" }}>
      <h2 style={{ sr: "only", fontSize: 0, margin: 0 }}>PhiUSIIL Phishing Dataset Analysis and Project Architecture</h2>

      {/* Header */}
      <div style={{ borderBottom: "0.5px solid var(--color-border-tertiary)", paddingBottom: "1rem", marginBottom: "1rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
          <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, background: "var(--color-background-info)", color: "var(--color-text-info)", padding: "2px 8px", borderRadius: 4 }}>DATASET ANALYSIS</span>
          <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, background: "var(--color-background-secondary)", color: "var(--color-text-secondary)", padding: "2px 8px", borderRadius: 4 }}>PhiUSIIL_Phishing_URL_Dataset.csv</span>
        </div>
        <p style={{ fontSize: 13, color: "var(--color-text-secondary)", margin: 0 }}>
          Explainable and Bias-Aware Machine Learning for Phishing Website Detection
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: "flex", gap: 4, marginBottom: "1.5rem", borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
        {TABS.map((t, i) => (
          <button key={i} onClick={() => setTab(i)} style={{
            background: "none", border: "none", padding: "8px 12px", fontSize: 13,
            color: tab === i ? "var(--color-text-primary)" : "var(--color-text-secondary)",
            borderBottom: tab === i ? "2px solid var(--color-text-primary)" : "2px solid transparent",
            cursor: "pointer", fontFamily: "var(--font-sans)", marginBottom: -1,
          }}>{t}</button>
        ))}
      </div>

      {/* ─── TAB 1: DATASET PROFILE ─── */}
      {tab === 0 && (
        <div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 10, marginBottom: "1.5rem" }}>
            {statCards.map((c, i) => (
              <div key={i} style={{ background: "var(--color-background-secondary)", borderRadius: "var(--border-radius-md)", padding: "12px 14px" }}>
                <div style={{ fontSize: 12, color: "var(--color-text-secondary)", marginBottom: 4 }}>{c.label}</div>
                <div style={{ fontSize: 20, fontWeight: 500 }}>{c.value}</div>
                <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginTop: 2 }}>{c.sub}</div>
              </div>
            ))}
          </div>

          {/* Class Distribution Bar */}
          <div style={{ background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-tertiary)", borderRadius: "var(--border-radius-lg)", padding: "1rem 1.25rem", marginBottom: "1.5rem" }}>
            <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 10 }}>Class distribution</div>
            <div style={{ display: "flex", borderRadius: 6, overflow: "hidden", height: 28, marginBottom: 10 }}>
              <div style={{ width: "42.81%", background: "#E24B4A", display: "flex", alignItems: "center", justifyContent: "center" }}>
                <span style={{ fontSize: 11, fontWeight: 500, color: "#fff" }}>Phishing 42.81%</span>
              </div>
              <div style={{ width: "57.19%", background: "#1D9E75", display: "flex", alignItems: "center", justifyContent: "center" }}>
                <span style={{ fontSize: 11, fontWeight: 500, color: "#fff" }}>Legitimate 57.19%</span>
              </div>
            </div>
            <div style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>
              Mild imbalance — stratified K-fold sufficient. label=0 → phishing, label=1 → legitimate.
            </div>
          </div>

          {/* Feature Type Breakdown */}
          <div style={{ background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-tertiary)", borderRadius: "var(--border-radius-lg)", padding: "1rem 1.25rem", marginBottom: "1.5rem" }}>
            <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 12 }}>Feature type breakdown (52 usable)</div>
            {[
              { label: "Binary flags (0/1)", count: 19, color: "#533AB7", pct: 36 },
              { label: "Count integers (NoOf*)", count: 17, color: "#185FA5", pct: 33 },
              { label: "Ratio / probability / score", count: 14, color: "#0F6E56", pct: 27 },
              { label: "Categorical (TLD)", count: 1, color: "#854F0B", pct: 2 },
              { label: "Drop — identifiers", count: 4, color: "#B4B2A9", pct: 7 },
            ].map((r, i) => (
              <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                <div style={{ width: 110, fontSize: 12, color: "var(--color-text-secondary)", flexShrink: 0 }}>{r.label}</div>
                <div style={{ flex: 1, height: 16, background: "var(--color-background-secondary)", borderRadius: 4, overflow: "hidden" }}>
                  <div style={{ width: `${r.pct * 2.5}%`, height: "100%", background: r.color, borderRadius: 4 }} />
                </div>
                <div style={{ fontSize: 12, fontFamily: "var(--font-mono)", minWidth: 20, textAlign: "right" }}>{r.count}</div>
              </div>
            ))}
          </div>

          {/* Top Correlations */}
          <div style={{ background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-tertiary)", borderRadius: "var(--border-radius-lg)", padding: "1rem 1.25rem" }}>
            <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 12 }}>Top 10 features by |correlation| with label</div>
            {topCorr.map((f, i) => (
              <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 7 }}>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, width: 200, color: "var(--color-text-primary)", flexShrink: 0 }}>{f.name}</div>
                <div style={{ flex: 1, height: 14, background: "var(--color-background-secondary)", borderRadius: 3, overflow: "hidden" }}>
                  <div style={{
                    width: `${f.r * 100}%`, height: "100%", borderRadius: 3,
                    background: f.note.includes("leakage") ? "#E24B4A" : f.note.includes("multi") ? "#EF9F27" : "#378ADD"
                  }} />
                </div>
                <div style={{ fontSize: 11, fontFamily: "var(--font-mono)", minWidth: 36, color: "var(--color-text-secondary)" }}>{f.r.toFixed(2)}</div>
                {f.note && <span style={{ fontSize: 10, padding: "1px 5px", borderRadius: 3, background: f.note.includes("leakage") ? "#FCEBEB" : "#FAEEDA", color: f.note.includes("leakage") ? "#A32D2D" : "#854F0B" }}>{f.note}</span>}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ─── TAB 2: FEATURE CATALOG ─── */}
      {tab === 1 && (
        <div>
          <p style={{ fontSize: 13, color: "var(--color-text-secondary)", marginTop: 0, marginBottom: "1rem" }}>
            All 56 columns across 10 categories. Click a category to expand features.
          </p>
          {FEATURE_CATALOG.map((cat, ci) => (
            <div key={ci} style={{ border: "0.5px solid var(--color-border-tertiary)", borderRadius: "var(--border-radius-lg)", marginBottom: 10, overflow: "hidden" }}>
              <button onClick={() => setOpenCat(openCat === ci ? null : ci)} style={{
                width: "100%", background: openCat === ci ? cat.bg : "var(--color-background-primary)",
                border: "none", padding: "10px 14px", cursor: "pointer", display: "flex", alignItems: "center", gap: 10,
                textAlign: "left", transition: "background 0.15s",
              }}>
                <div style={{ width: 10, height: 10, borderRadius: "50%", background: cat.color, flexShrink: 0 }} />
                <span style={{ fontSize: 13, fontWeight: 500, flex: 1 }}>{cat.category}</span>
                <span style={{ fontSize: 12, color: "var(--color-text-secondary)", marginRight: 8 }}>{cat.features.length} features</span>
                <span style={{ fontSize: 10, padding: "2px 7px", borderRadius: 3, background: cat.actionBg, color: cat.actionColor, fontWeight: 500 }}>{cat.action}</span>
                <i className={`ti ${openCat === ci ? "ti-chevron-up" : "ti-chevron-down"}`} style={{ fontSize: 14, color: "var(--color-text-tertiary)" }} aria-hidden />
              </button>
              {openCat === ci && (
                <div style={{ borderTop: "0.5px solid var(--color-border-tertiary)" }}>
                  {cat.features.map((f, fi) => (
                    <div key={fi} style={{ padding: "8px 14px", borderBottom: fi < cat.features.length - 1 ? "0.5px solid var(--color-border-tertiary)" : "none", display: "grid", gridTemplateColumns: "200px 50px 1fr", gap: 10, alignItems: "start" }}>
                      <div style={{ fontFamily: "var(--font-mono)", fontSize: 12, color: "var(--color-text-primary)" }}>{f.name}</div>
                      <div style={{ fontSize: 10, padding: "2px 5px", borderRadius: 3, background: "var(--color-background-secondary)", color: "var(--color-text-secondary)", fontFamily: "var(--font-mono)", alignSelf: "center", textAlign: "center" }}>{f.type}</div>
                      <div>
                        <div style={{ fontSize: 12, color: "var(--color-text-primary)", marginBottom: 2 }}>{f.desc}</div>
                        <div style={{ fontSize: 11, color: f.note.includes("🚨") ? "#A32D2D" : f.note.includes("⚠") ? "#854F0B" : "var(--color-text-tertiary)" }}>{f.note}</div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* ─── TAB 3: CRITICAL FLAGS ─── */}
      {tab === 2 && (
        <div>
          <p style={{ fontSize: 13, color: "var(--color-text-secondary)", marginTop: 0, marginBottom: "1rem" }}>
            Issues that directly affect model validity, explainability quality, and fairness.
          </p>
          {CRITICAL_FLAGS.map((f, i) => (
            <div key={i} style={{ border: `0.5px solid ${f.border}`, borderLeft: `3px solid ${f.border}`, borderRadius: "var(--border-radius-lg)", background: f.bg, padding: "14px 16px", marginBottom: 12 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                <i className={`ti ${f.icon}`} style={{ fontSize: 16, color: f.color }} aria-hidden />
                <span style={{ fontSize: 11, fontWeight: 500, padding: "2px 7px", borderRadius: 3, background: f.bg, color: f.color, border: `0.5px solid ${f.border}` }}>{f.level}</span>
                <span style={{ fontSize: 13, fontWeight: 500, color: "var(--color-text-primary)" }}>{f.title}</span>
              </div>
              <p style={{ fontSize: 13, color: "var(--color-text-secondary)", margin: "0 0 10px", lineHeight: 1.6 }}>{f.body}</p>
              <div style={{ display: "flex", alignItems: "flex-start", gap: 6 }}>
                <i className="ti ti-arrow-right" style={{ fontSize: 13, color: f.color, marginTop: 1, flexShrink: 0 }} aria-hidden />
                <span style={{ fontSize: 12, color: f.color, fontWeight: 500 }}>{f.action}</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ─── TAB 4: PROJECT ARCHITECTURE ─── */}
      {tab === 3 && (
        <div>
          <p style={{ fontSize: 13, color: "var(--color-text-secondary)", marginTop: 0, marginBottom: "1rem" }}>
            9-phase pipeline designed around this dataset's specific characteristics. Click a phase to expand.
          </p>
          {ARCH_PHASES.map((p, pi) => (
            <div key={pi} style={{ border: "0.5px solid var(--color-border-tertiary)", borderRadius: "var(--border-radius-lg)", marginBottom: 10, overflow: "hidden" }}>
              <button onClick={() => setOpenPhase(openPhase === pi ? null : pi)} style={{
                width: "100%", background: openPhase === pi ? p.bg : "var(--color-background-primary)",
                border: "none", padding: "10px 14px", cursor: "pointer", display: "flex", alignItems: "center", gap: 12, textAlign: "left",
              }}>
                <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, fontWeight: 500, color: p.color, minWidth: 24 }}>{p.phase}</span>
                <span style={{ fontSize: 13, fontWeight: 500, flex: 1 }}>{p.title}</span>
                <i className={`ti ${openPhase === pi ? "ti-chevron-up" : "ti-chevron-down"}`} style={{ fontSize: 14, color: "var(--color-text-tertiary)" }} aria-hidden />
              </button>
              {openPhase === pi && (
                <div style={{ borderTop: "0.5px solid var(--color-border-tertiary)", padding: "12px 14px" }}>
                  {/* Models phase special layout */}
                  {p.models ? (
                    <div>
                      {p.models.map((m, mi) => (
                        <div key={mi} style={{ display: "flex", gap: 10, alignItems: "flex-start", padding: "8px 0", borderBottom: mi < p.models.length - 1 ? "0.5px solid var(--color-border-tertiary)" : "none" }}>
                          <div style={{ width: 6, height: 6, borderRadius: "50%", background: p.color, marginTop: 5, flexShrink: 0 }} />
                          <div>
                            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 2 }}>
                              <span style={{ fontSize: 13, fontWeight: 500 }}>{m.name}</span>
                              <span style={{ fontSize: 10, padding: "1px 5px", borderRadius: 3, background: p.bg, color: p.color }}>{m.type}</span>
                            </div>
                            <div style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>{m.note}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : p.xai ? (
                    <div>
                      {p.xai.map((x, xi) => (
                        <div key={xi} style={{ padding: "8px 0", borderBottom: xi < p.xai.length - 1 ? "0.5px solid var(--color-border-tertiary)" : "none" }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 3 }}>
                            <div style={{ width: 6, height: 6, borderRadius: "50%", background: p.color, flexShrink: 0 }} />
                            <span style={{ fontSize: 13, fontWeight: 500 }}>{x.method}</span>
                            <span style={{ fontSize: 10, padding: "1px 5px", borderRadius: 3, background: p.bg, color: p.color }}>{x.scope}</span>
                          </div>
                          <div style={{ fontSize: 12, color: "var(--color-text-secondary)", paddingLeft: 14 }}>{x.details}</div>
                        </div>
                      ))}
                    </div>
                  ) : p.metrics ? (
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                      {p.metrics.map((m, mi) => (
                        <div key={mi} style={{ padding: "8px 10px", background: p.bg, borderRadius: "var(--border-radius-md)" }}>
                          <div style={{ fontSize: 12, fontWeight: 500, color: p.color, marginBottom: 2 }}>{m.name}</div>
                          <div style={{ fontSize: 11, color: "var(--color-text-secondary)" }}>{m.desc}</div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <ul style={{ margin: 0, padding: "0 0 0 16px", listStyle: "none" }}>
                      {p.modules.map((m, mi) => (
                        <li key={mi} style={{ display: "flex", gap: 8, alignItems: "flex-start", padding: "4px 0", fontSize: 13, color: "var(--color-text-secondary)" }}>
                          <span style={{ color: p.color, flexShrink: 0, marginTop: 2 }}>→</span>
                          <span>{m}</span>
                        </li>
                      ))}
                    </ul>
                  )}
                  {p.output && (
                    <div style={{ marginTop: 12, padding: "8px 10px", background: "var(--color-background-secondary)", borderRadius: "var(--border-radius-md)", display: "flex", alignItems: "center", gap: 8 }}>
                      <i className="ti ti-arrow-bear-right" style={{ fontSize: 14, color: p.color }} aria-hidden />
                      <span style={{ fontSize: 12, color: "var(--color-text-secondary)" }}><strong style={{ fontWeight: 500, color: "var(--color-text-primary)" }}>Output: </strong>{p.output}</span>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
