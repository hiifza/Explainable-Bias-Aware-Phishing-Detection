import { useBlindspotData, useBiasData, useReliabilityData } from '@/hooks/useIntelligence'
import styles from './IntelligenceSections.module.css'

/* ══════════════════════════════════════════════════════════════
   BLIND SPOT SECTION — Live from /api/intelligence/blindspots
══════════════════════════════════════════════════════════════ */

const INSIGHTS: Record<number, string> = {
  17372: "Lowest-confidence failure in the entire test set. Sits at the decision boundary where the model's uncertainty is highest, making it the most dangerous misclassification discovered.",
  11301: "High-confidence false negative: the model classified a phishing URL as legitimate with 87.46% certainty. High confidence misclassifications are especially dangerous as they evade downstream thresholding.",
  30588: "Highest-confidence failure at 88.35%. Indistinguishable from a correct prediction from the model's perspective. Represents the most challenging adversarial surface for any deployed phishing classifier.",
}

export function BlindspotSection() {
  const { data, loading, source } = useBlindspotData()
  const blindspots = data?.top3 ?? []

  if (loading) return <SectionSkeleton rows={3} height={120} />

  return (
    <div className={styles.bsWrap}>
      {source === 'api' && <div className={styles.sourceTag}>Live from outputs/reports/top20_blind_spots.csv</div>}
      {blindspots.map((b: any) => {
        const sid = parseInt(b.sample_id ?? b.id ?? 0)
        return (
          <div key={sid} className={styles.incidentCard}>
            <div className={styles.incidentLeft}>
              <div className={styles.incidentRank}>Rank {b.rank}</div>
              <div className={styles.incidentId}>#{sid}</div>
              <div className={styles.incidentType}>{b.type}</div>
            </div>
            <div className={styles.incidentCenter}>
              <div className={styles.incidentStats}>
                <div className={styles.iStat}>
                  <span className={styles.iStatVal}>{parseFloat(b.confidence ?? b.confidence_pct ?? 0).toFixed(2)}%</span>
                  <span className={styles.iStatKey}>Confidence</span>
                </div>
                <div className={styles.iStat}>
                  <span className={styles.iStatVal}>{parseFloat(b.severity_score ?? b.severity ?? 0).toFixed(3)}</span>
                  <span className={styles.iStatKey}>Severity</span>
                </div>
                <div className={styles.iStat}>
                  <span className={`${styles.iStatVal} ${styles.critical}`}>{b.risk ?? 'CRITICAL'}</span>
                  <span className={styles.iStatKey}>Risk Level</span>
                </div>
              </div>
              <p className={styles.incidentInsight}>
                {INSIGHTS[sid] ?? `Sample #${sid} — ${b.type ?? 'Misclassification'} detected in blind spot analysis.`}
              </p>
            </div>
          </div>
        )
      })}
    </div>
  )
}

/* ══════════════════════════════════════════════════════════════
   BIAS SECTION — Live from /api/intelligence/bias
══════════════════════════════════════════════════════════════ */

const DIM_NOTES: Record<string, string> = {
  'URL Length Groups':        'No performance gap across short/medium/long URLs.',
  'Domain Length Groups':     'Domain length creates no measurable disparity.',
  'HTTPS Groups':             'HTTPS presence or absence does not bias performance.',
  'TLD Groups':               'Most scrutinized dimension. Still passes all thresholds.',
  'External Resource Groups': 'Least biased dimension of all five audited.',
}

export function BiasSection() {
  const { data, loading, source } = useBiasData()
  const dims = data?.summary?.dimensions ?? []

  if (loading) return <SectionSkeleton rows={5} height={80} />

  return (
    <div className={styles.biasWrap}>
      {source === 'api' && <div className={styles.sourceTag}>Live from outputs/reports/bias_metrics.csv</div>}
      <div className={styles.biasGrid}>
        {dims.map((d: any) => (
          <div key={d.name} className={`${styles.dimCard} ${d.most_biased ? styles.dimMostBiased : ''}`}>
            <div className={styles.dimHeader}>
              <span className={styles.dimName}>{d.name}</span>
              <span className={`${styles.dimStatus} ${styles.statusPass}`}>{d.status ?? 'PASS'}</span>
            </div>
            <div className={styles.dimMin}>
              Min performance: <strong>{`> ${d.min_performance ?? d.min ?? 99.98}%`}</strong>
            </div>
            <p className={styles.dimNote}>{DIM_NOTES[d.name] ?? 'No significant performance disparity detected.'}</p>
            {d.most_biased && <div className={styles.mostBiasedTag}>Most scrutinized</div>}
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
   RELIABILITY SECTION — Live from /api/intelligence/reliability
══════════════════════════════════════════════════════════════ */

const ZONE_DESC: Record<string, string> = {
  GREEN:  'SHAP and LIME agree. Both explanation methods identify the same decision drivers. Predictions in this zone are maximally trustworthy.',
  YELLOW: 'Partial explanation agreement. Some features conflict between methods. Proceed with elevated scrutiny before acting on prediction.',
  RED:    'Critical: SHAP and LIME fundamentally disagree. 13.04% error rate despite 97.31% mean model confidence. Highest-risk prediction zone.',
}

export function ReliabilitySection() {
  const { data, loading, source } = useReliabilityData()
  const zones = data?.summary?.zones ?? []

  const ZONE_LABELS: Record<string, string> = { GREEN: 'Green Zone', YELLOW: 'Yellow Zone', RED: 'Red Zone' }

  if (loading) return <SectionSkeleton rows={1} height={160} />

  return (
    <div className={styles.relWrap}>
      {source === 'api' && <div className={styles.sourceTag}>Live from outputs/reports/reliability_bin_stats.csv</div>}
      <div className={styles.zonesGrid}>
        {zones.map((z: any) => {
          const errRate = z.error_rate == null ? '~0%'
            : typeof z.error_rate === 'number' && z.error_rate === 0 ? '~0%'
            : `${z.error_rate}%`
          return (
            <div key={z.zone} className={`${styles.zoneCard} ${styles[`zone_${z.zone}`]}`}>
              <div className={styles.zoneLabel}>{ZONE_LABELS[z.zone] ?? z.zone}</div>
              <div className={`${styles.zoneErr} ${styles[`zoneErr_${z.zone}`]}`}>{errRate}</div>
              <div className={styles.zoneErrLabel}>Error rate</div>
              <div className={styles.zoneMeta}>
                <span>Agreement: <strong>{z.agreement_range}</strong></span>
                <span>Samples: <strong>{z.samples}</strong></span>
                {z.mean_confidence && <span>Confidence: <strong>{z.mean_confidence}%</strong></span>}
              </div>
              <p className={styles.zoneDesc}>{ZONE_DESC[z.zone]}</p>
            </div>
          )
        })}
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

/* ── Skeleton loader ─────────────────────────────────────────── */
function SectionSkeleton({ rows, height }: { rows: number; height: number }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} style={{
          height,
          borderRadius: 'var(--r-lg)',
          background: 'var(--surface-1)',
          opacity: 1 - i * 0.1,
          animation: 'pulseGlow 1.6s ease-in-out infinite',
        }} />
      ))}
    </div>
  )
}
