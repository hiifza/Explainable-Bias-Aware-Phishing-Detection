import styles from './IntelligenceSections.module.css'

/* ══════════════════════════════════════════════════════════════
   BLIND SPOT SECTION
══════════════════════════════════════════════════════════════ */
const BLINDSPOTS = [
  {
    rank: 1, id: 17372, type: 'False Negative',
    confidence: 62.61, severity: 0.712, risk: 'CRITICAL',
    insight: 'Lowest-confidence failure in the entire test set. Sits at the decision boundary where the model\'s uncertainty is highest, making it the most dangerous misclassification discovered.',
  },
  {
    rank: 2, id: 11301, type: 'False Negative',
    confidence: 87.46, severity: 0.638, risk: 'CRITICAL',
    insight: 'High-confidence false negative: the model classified a phishing URL as legitimate with 87.46% certainty. High confidence misclassifications are especially dangerous as they evade downstream thresholding.',
  },
  {
    rank: 3, id: 30588, type: 'False Negative',
    confidence: 88.35, severity: 0.635, risk: 'CRITICAL',
    insight: 'Highest-confidence failure at 88.35%. Indistinguishable from a correct prediction from the model\'s perspective. Represents the most challenging adversarial surface for any deployed phishing classifier.',
  },
]

export function BlindspotSection() {
  return (
    <div className={styles.bsWrap}>
      {BLINDSPOTS.map((b) => (
        <div key={b.id} className={styles.incidentCard}>
          <div className={styles.incidentLeft}>
            <div className={styles.incidentRank}>Rank {b.rank}</div>
            <div className={styles.incidentId}>#{b.id}</div>
            <div className={styles.incidentType}>{b.type}</div>
          </div>
          <div className={styles.incidentCenter}>
            <div className={styles.incidentStats}>
              <div className={styles.iStat}>
                <span className={styles.iStatVal}>{b.confidence}%</span>
                <span className={styles.iStatKey}>Confidence</span>
              </div>
              <div className={styles.iStat}>
                <span className={styles.iStatVal}>{b.severity}</span>
                <span className={styles.iStatKey}>Severity</span>
              </div>
              <div className={styles.iStat}>
                <span className={`${styles.iStatVal} ${styles.critical}`}>{b.risk}</span>
                <span className={styles.iStatKey}>Risk Level</span>
              </div>
            </div>
            <p className={styles.incidentInsight}>{b.insight}</p>
          </div>
        </div>
      ))}
    </div>
  )
}

/* ══════════════════════════════════════════════════════════════
   BIAS SECTION
══════════════════════════════════════════════════════════════ */
const BIAS_DIMS = [
  { name: 'URL Length Groups',        min: 99.98, note: 'No performance gap across short/medium/long URLs.',         biased: false },
  { name: 'Domain Length Groups',     min: 99.98, note: 'Domain length creates no measurable disparity.',            biased: false },
  { name: 'HTTPS Groups',             min: 99.99, note: 'HTTPS presence or absence does not bias performance.',      biased: false },
  { name: 'TLD Groups',               min: 99.99, note: 'Most scrutinized dimension. Still passes all thresholds.',  biased: true  },
  { name: 'External Resource Groups', min: 99.98, note: 'Least biased dimension of all five audited.',              biased: false },
]

export function BiasSection() {
  return (
    <div className={styles.biasWrap}>
      <div className={styles.biasGrid}>
        {BIAS_DIMS.map((d) => (
          <div key={d.name} className={`${styles.dimCard} ${d.biased ? styles.dimMostBiased : ''}`}>
            <div className={styles.dimHeader}>
              <span className={styles.dimName}>{d.name}</span>
              <span className={`${styles.dimStatus} ${styles.statusPass}`}>PASS</span>
            </div>
            <div className={styles.dimMin}>
              Min performance: <strong>{`> ${d.min}%`}</strong>
            </div>
            <p className={styles.dimNote}>{d.note}</p>
            {d.biased && (
              <div className={styles.mostBiasedTag}>Most scrutinized</div>
            )}
          </div>
        ))}
      </div>

      <div className={styles.fairnessVerdict}>
        <div className={styles.verdictCheck}>✓</div>
        <div>
          <div className={styles.verdictTitle}>No Significant Performance Drift Detected</div>
          <p className={styles.verdictText}>
            All five fairness dimensions pass. Performance exceeds 99.98% for every
            subgroup. Zero bias violations. The model is cleared for production deployment
            from a fairness and equity standpoint.
          </p>
        </div>
      </div>
    </div>
  )
}

/* ══════════════════════════════════════════════════════════════
   RELIABILITY SECTION
══════════════════════════════════════════════════════════════ */
const ZONES = [
  {
    zone: 'GREEN',  label: 'Green Zone',  agree: '0.8 – 1.0',
    samples: 'Majority', errorRate: '~0%',   confidence: 'High',
    desc: 'SHAP and LIME agree. Both explanation methods identify the same decision drivers. Predictions in this zone are maximally trustworthy.',
  },
  {
    zone: 'YELLOW', label: 'Yellow Zone', agree: '0.2 – 0.8',
    samples: 'Moderate', errorRate: '~5%',  confidence: 'Medium',
    desc: 'Partial explanation agreement. Some features conflict between methods. Proceed with elevated scrutiny before acting on prediction.',
  },
  {
    zone: 'RED',    label: 'Red Zone',    agree: '0.0 – 0.2',
    samples: '23',   errorRate: '13.04%', confidence: '97.31%',
    desc: 'Critical: SHAP and LIME fundamentally disagree. 13.04% error rate despite 97.31% mean model confidence. Highest-risk prediction zone.',
  },
]

export function ReliabilitySection() {
  return (
    <div className={styles.relWrap}>
      <div className={styles.zonesGrid}>
        {ZONES.map((z) => (
          <div key={z.zone} className={`${styles.zoneCard} ${styles[`zone_${z.zone}`]}`}>
            <div className={styles.zoneLabel}>{z.label}</div>
            <div className={`${styles.zoneErr} ${styles[`zoneErr_${z.zone}`]}`}>
              {z.errorRate}
            </div>
            <div className={styles.zoneErrLabel}>Error rate</div>
            <div className={styles.zoneMeta}>
              <span>Agreement: <strong>{z.agree}</strong></span>
              <span>Samples: <strong>{z.samples}</strong></span>
              <span>Confidence: <strong>{z.confidence}</strong></span>
            </div>
            <p className={styles.zoneDesc}>{z.desc}</p>
          </div>
        ))}
      </div>

      <div className={styles.relFinding}>
        <div className={styles.rfIcon}>⚠</div>
        <div>
          <div className={styles.rfTitle}>Key Finding: Confidence Does Not Equal Reliability</div>
          <p className={styles.rfText}>
            Red Zone samples carry 97.31% mean model confidence yet fail at 13.04%.
            This demonstrates that SHAP-LIME explanation agreement is a stronger
            predictor of prediction failure than model confidence scores — a novel
            deployment-time reliability indicator with direct implications for
            production cybersecurity systems.
          </p>
        </div>
      </div>
    </div>
  )
}
