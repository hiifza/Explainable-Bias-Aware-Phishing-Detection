// Research.tsx
import { BiasSection } from '@/components/intelligence/IntelligenceSections'
import ModelPerformance from '@/components/intelligence/ModelPerformance'
import ShapSection from '@/components/intelligence/ShapSection'
import ShapLimeConflict from '@/components/intelligence/ShapLimeConflict'
import styles from './SubPage.module.css'

const FINDINGS = [
  {
    num: '01',
    title: 'URL composition is the strongest phishing signal',
    text: 'URL character patterns — specifically the ratio of letters, digits, and special characters — are the most predictive features, outweighing page-level HTML content. Phishing can often be detected before a page loads.',
  },
  {
    num: '02',
    title: 'Only 3 failures in 47,159 evaluated samples',
    text: 'Across the full test set, only three false negatives were identified. All three were discovered, documented, and analyzed as high-severity blind spots — demonstrating that exhaustive failure investigation is feasible at scale.',
  },
  {
    num: '03',
    title: 'SHAP and LIME exhibit complete local explanation disagreement',
    text: 'Despite identical predictions, SHAP and LIME identify different features as primary decision drivers. A single XAI method may be insufficient to reliably audit decisions in high-stakes cybersecurity deployments.',
  },
  {
    num: '04',
    title: 'The model is fair across all five audited dimensions',
    text: 'Fairness auditing across URL length, domain length, HTTPS status, TLD groups, and external resources revealed no significant performance disparities. The model maintains >99.98% for every subgroup.',
  },
  {
    num: '05',
    title: 'Explanation agreement predicts error risk better than confidence',
    text: 'In the lowest agreement bin (0.0–0.2), the error rate is 13.04% despite 97.31% mean model confidence. SHAP-LIME agreement is a stronger predictor of prediction failure than confidence scores alone.',
  },
]

const TIMELINE = [
  { id: 'M1',  name: 'Data Audit',                     desc: '235,795 rows · 56 features · 0 missing values · class distribution confirmed.' },
  { id: 'M2',  name: 'Feature Finalization',            desc: 'Feature audit, Track A/B definition, encoding strategy, URLSimilarityIndex leakage flagged.' },
  { id: 'M3',  name: 'Exploratory Data Analysis',       desc: 'Correlations, leakage detection, TLD analysis, skewness correction, mutual information screening.' },
  { id: 'M4',  name: 'Feature Engineering',             desc: '7 engineered features, pipeline construction, scaling, TLD encoding, train/test splits.' },
  { id: 'M5',  name: 'Model Training',                  desc: '4 models × 2 tracks, cross-validation, benchmarking, model registry.' },
  { id: 'M6',  name: 'Model Evaluation',                desc: 'Confusion matrices, ROC/PR curves, calibration, error analysis. Track B LightGBM selected.' },
  { id: 'M7',  name: 'SHAP Explainability',             desc: 'Global importance, beeswarm, dependence plots, 1,540 interaction pairs, local explanations.' },
  { id: 'M8',  name: 'LIME Explainability',             desc: 'Local surrogate explanations, SHAP-LIME agreement 52.0%, feature consistency 0.60.' },
  { id: 'M9',  name: 'Bias Intelligence',               desc: '5-dimension fairness audit. All dimensions pass. Most biased: TLD. Least biased: External Resources.' },
  { id: 'M10', name: 'Blind Spot Intelligence',         desc: 'Failure archetypes, severity scoring, top-20 blind spots, Green/Yellow/Red zones, PCA cluster maps.' },
  { id: 'M11', name: 'Cybersecurity Intelligence Platform', desc: 'This platform. Multi-layer architecture serving everyday users, analysts, and researchers.', current: true },
]

export function Research() {
  return (
    <div className={styles.page}>
      <div className={styles.hero}>
        <div className="section-inner">
          <div className="section-tag">Research Laboratory</div>
          <h1 className={styles.heroTitle}>Technical depth.<br />Five key findings.</h1>
          <p className={styles.heroSub}>
            The complete research pipeline across 11 modules, producing novel findings
            at the intersection of machine learning, explainable AI, and cybersecurity.
          </p>
        </div>
      </div>

      <section className={styles.section}>
        <div className="section-inner">
          <h2 className={styles.sectionTitle}>Research Findings</h2>
          <div className={styles.findingsList}>
            {FINDINGS.map((f) => (
              <div key={f.num} className={styles.findingCard}>
                <div className={styles.findingNum}>{f.num}</div>
                <div>
                  <div className={styles.findingTitle}>{f.title}</div>
                  <p className={styles.findingText}>{f.text}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className={`${styles.section} ${styles.sectionAlt}`}>
        <div className="section-inner">
          <h2 className={styles.sectionTitle}>Model Performance</h2>
          <ModelPerformance />
        </div>
      </section>

      <section className={styles.section}>
        <div className="section-inner">
          <h2 className={styles.sectionTitle}>SHAP Feature Importance</h2>
          <ShapSection />
        </div>
      </section>

      <section className={`${styles.section} ${styles.sectionAlt}`}>
        <div className="section-inner">
          <h2 className={styles.sectionTitle}>SHAP vs LIME Conflict</h2>
          <ShapLimeConflict />
        </div>
      </section>

      <section className={`${styles.section} ${styles.sectionAlt}`}>
        <div className="section-inner">
          <h2 className={styles.sectionTitle}>Bias & Fairness Audit</h2>
          <BiasSection />
        </div>
      </section>

      <section className={styles.section}>
        <div className="section-inner">
          <div className="section-tag">Research Pipeline</div>
          <h2 className={styles.sectionTitle}>11-Module Journey</h2>
          <div className={styles.timeline}>
            {TIMELINE.map((t) => (
              <div key={t.id} className={`${styles.tlItem} ${t.current ? styles.tlCurrent : ''}`}>
                <div className={styles.tlNode}>{t.id}</div>
                <div className={styles.tlContent}>
                  <div className={styles.tlName}>{t.name}</div>
                  <div className={styles.tlDesc}>{t.desc}</div>
                  {t.current && <div className={styles.tlLive}>⚡ Live</div>}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  )
}
export default Research
