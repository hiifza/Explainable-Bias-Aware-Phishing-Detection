import styles from './SubPage.module.css'

const TEAM = [
  {
    initials: 'HA', name: 'Hifza Amir', role: 'B.Tech CSE · Data Science',
    desc: 'Led data pipeline, feature engineering, SHAP explainability, and bias intelligence. Responsible for the dual-track evaluation framework and fairness auditing methodology.',
    github: 'https://github.com/hiifza',
    linkedin: 'https://www.linkedin.com/in/hiifzza/',
    modules: ['M1 Data Audit', 'M2 Features', 'M3 EDA', 'M7 SHAP', 'M9 Bias'],
  },
  {
    initials: 'SA', name: 'Shihan Ahmad', role: 'B.Tech CSE · Cybersecurity',
    desc: 'Led model training, LIME analysis, blind spot intelligence, failure archetype discovery, and reliability analysis. Responsible for cybersecurity threat framing.',
    github: 'https://github.com/ShihanG9',
    linkedin: 'https://www.linkedin.com/in/shihanahmad/',
    modules: ['M5 Training', 'M6 Evaluation', 'M8 LIME', 'M10 Blind Spots', 'M11 Platform'],
  },
]

export function About() {
  return (
    <div className={styles.page}>
      <div className={styles.hero}>
        <div className="section-inner">
          <div className="section-tag">About</div>
          <h1 className={styles.heroTitle}>Built by two<br />researchers.</h1>
          <p className={styles.heroSub}>
            PhishGuard represents months of original research into the intersection of
            machine learning, explainable AI, fairness auditing, and cybersecurity threat intelligence.
          </p>
        </div>
      </div>

      <section className={styles.section}>
        <div className="section-inner">
          <div className={styles.teamGrid}>
            {TEAM.map((m) => (
              <div key={m.name} className={styles.teamCard}>
                <div className={styles.avatar}>{m.initials}</div>
                <div className={styles.teamName}>{m.name}</div>
                <div className={styles.teamRole}>{m.role}</div>
                <p className={styles.teamDesc}>{m.desc}</p>
                <div className={styles.moduleList}>
                  {m.modules.map((mod) => (
                    <span key={mod} className={styles.modBadge}>{mod}</span>
                  ))}
                </div>
                <div className={styles.teamLinks}>
                  <a href={m.github} target="_blank" rel="noopener noreferrer" className={styles.teamLink}>
                    <span>⌁</span> GitHub
                  </a>
                  <a href={m.linkedin} target="_blank" rel="noopener noreferrer" className={styles.teamLink}>
                    <span>⌁</span> LinkedIn
                  </a>
                </div>
              </div>
            ))}
          </div>

          <div className={styles.datasetCard}>
            <div className={styles.datasetLabel}>Dataset</div>
            <div className={styles.datasetName}>PhiUSIIL Phishing URL Dataset</div>
            <div className={styles.datasetMeta}>
              235,795 samples · 56 features · 0 missing values ·
              57.19% legitimate / 42.81% phishing
            </div>
            <div className={styles.datasetModel}>
              Deployment model: <strong>LightGBM (Track B)</strong> ·
              ROC-AUC 1.00 · Accuracy 99.9936%
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}
export default About
