import type { ThreatSignal, AttackerSignal, ScanResult } from '@/store'
import styles from './SignalPanels.module.css'

/* ── Threat Signals ─────────────────────────────────────────── */
interface ThreatSignalsProps { signals: ThreatSignal[] }

export function ThreatSignals({ signals }: ThreatSignalsProps) {
  return (
    <div className={styles.panel}>
      <div className={styles.panelLabel}>Detected Signals</div>
      <div className={styles.signals}>
        {signals.map((sig) => (
          <div key={sig.id} className={`${styles.signal} ${styles[`impact_${sig.impact}`]}`}>
            <div className={styles.sigHeader}>
              <span className={styles.sigName}>{sig.human_label}</span>
              <span className={`${styles.sigBadge} ${styles[`badge_${sig.impact}`]}`}>
                {sig.impact}
              </span>
            </div>
            <p className={styles.sigDesc}>{sig.description}</p>
          </div>
        ))}
        {signals.length === 0 && (
          <p className={styles.empty}>No significant risk signals detected.</p>
        )}
      </div>
    </div>
  )
}

/* ── Attacker Simulation ─────────────────────────────────────── */
interface AttackerSimProps { signals: AttackerSignal[] }

export function AttackerSim({ signals }: AttackerSimProps) {
  if (!signals.length) return null
  return (
    <div className={styles.panel}>
      <div className={styles.panelLabel}>Why This Site May Fool People</div>
      <div className={styles.simIntro}>
        Attackers use these signals to build false trust:
      </div>
      <div className={styles.simList}>
        {signals.map((s, i) => (
          <div key={i} className={`${styles.simRow} ${styles[`simRow_${s.danger_level}`]}`}>
            <span className={styles.simCheck}>✓</span>
            <div>
              <div className={styles.simSignal}>{s.signal}</div>
              <div className={styles.simExpl}>{s.explanation}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

/* ── Action Guidance ─────────────────────────────────────────── */
interface ActionGuidanceProps { result: ScanResult }

export function ActionGuidance({ result }: ActionGuidanceProps) {
  const isRisky = ['HIGH_RISK', 'CRITICAL', 'SUSPICIOUS'].includes(result.risk_level)
  return (
    <div className={`${styles.panel} ${isRisky ? styles.panelDanger : styles.panelSafe}`}>
      <div className={styles.panelLabel}>
        {isRisky ? 'Recommended Actions' : 'You appear safe'}
      </div>
      <ul className={styles.actions}>
        {result.recommended_actions.map((a, i) => (
          <li key={i} className={styles.actionItem}>
            <span className={`${styles.actionBullet} ${isRisky ? styles.bulletDanger : styles.bulletSafe}`}>
              {isRisky ? '✕' : '✓'}
            </span>
            <span>{a}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

/* ── Expert Panel ────────────────────────────────────────────── */
interface ExpertPanelProps { result: ScanResult }

export function ExpertPanel({ result }: ExpertPanelProps) {
  return (
    <div className={styles.expertWrap}>
      <div className={styles.expertHeader}>
        <span className={styles.expertTag}>Expert Analysis</span>
        <span className={styles.expertSub}>
          SHAP · LIME · Reliability · Domain Analysis
        </span>
      </div>

      <div className={styles.expertGrid}>
        {/* SHAP Features */}
        <div className={styles.expertSection}>
          <div className={styles.expertSectionLabel}>SHAP Feature Contributions</div>
          {result.shap_features.slice(0, 8).map((f) => (
            <div key={f.feature} className={styles.featRow}>
              <span className={styles.featName}>{f.human_label}</span>
              <div className={styles.featBar}>
                <div
                  className={styles.featFill}
                  style={{
                    width: `${Math.min(Math.abs(f.contribution) * 400, 100)}%`,
                    background: f.direction === 'increases_risk'
                      ? 'var(--danger)' : 'var(--safe)',
                  }}
                />
              </div>
              <span className={styles.featVal} style={{
                color: f.direction === 'increases_risk' ? 'var(--danger)' : 'var(--safe)'
              }}>
                {f.contribution > 0 ? '+' : ''}{f.contribution.toFixed(3)}
              </span>
            </div>
          ))}
        </div>

        {/* LIME Features */}
        <div className={styles.expertSection}>
          <div className={styles.expertSectionLabel}>LIME Local Explanation</div>
          {result.lime_features.slice(0, 8).map((f) => (
            <div key={f.feature} className={styles.featRow}>
              <span className={styles.featName}>{f.human_label}</span>
              <div className={styles.featBar}>
                <div
                  className={styles.featFill}
                  style={{
                    width: `${Math.min(Math.abs(f.weight) * 200, 100)}%`,
                    background: f.weight > 0 ? 'var(--danger)' : 'var(--safe)',
                  }}
                />
              </div>
              <span className={styles.featVal}>{f.weight.toFixed(3)}</span>
            </div>
          ))}

          <div className={styles.agreementBadge}>
            <span>SHAP-LIME Agreement</span>
            <span className={`${styles.agreementVal} ${result.explanation_agreement < 0.3 ? styles.agreeRed : result.explanation_agreement < 0.6 ? styles.agreeYellow : styles.agreeGreen}`}>
              {Math.round(result.explanation_agreement * 100)}%
            </span>
          </div>
        </div>

        {/* Domain Analysis */}
        <div className={styles.expertSection}>
          <div className={styles.expertSectionLabel}>Domain Analysis</div>
          <div className={styles.domainGrid}>
            <DomainStat k="Domain"         v={result.domain_analysis.domain} />
            <DomainStat k="TLD"            v={`.${result.domain_analysis.tld}`} />
            <DomainStat k="HTTPS"          v={result.domain_analysis.is_https ? 'Yes' : 'No'}
              highlight={result.domain_analysis.is_https ? 'safe' : 'danger'} />
            <DomainStat k="Domain length"  v={`${result.domain_analysis.domain_length} chars`}
              highlight={result.domain_analysis.domain_length > 25 ? 'danger' : 'safe'} />
            <DomainStat k="Subdomains"     v={String(result.domain_analysis.subdomain_count)}
              highlight={result.domain_analysis.subdomain_count > 2 ? 'warn' : 'safe'} />
            <DomainStat k="IP in URL"      v={result.domain_analysis.has_ip ? 'Yes' : 'No'}
              highlight={result.domain_analysis.has_ip ? 'critical' : 'safe'} />
            <DomainStat k="Letter ratio"   v={result.domain_analysis.letter_ratio.toFixed(2)} />
            <DomainStat k="Digit count"    v={String(result.domain_analysis.digit_count)} />
          </div>
          {result.domain_analysis.suspicious_keywords.length > 0 && (
            <div className={styles.suspKeywords}>
              <span className={styles.suspLabel}>Suspicious keywords:</span>
              {result.domain_analysis.suspicious_keywords.map((kw) => (
                <span key={kw} className={styles.keyword}>{kw}</span>
              ))}
            </div>
          )}
        </div>

        {/* Scam Psychology */}
        <div className={styles.expertSection}>
          <div className={styles.expertSectionLabel}>Scam Psychology Analyzer</div>
          <div className={styles.psychList}>
            {result.scam_psychology.map((p) => (
              <div key={p.tactic} className={`${styles.psychRow} ${p.detected ? styles.psychDetected : styles.psychClear}`}>
                <span className={styles.psychDot}>{p.detected ? '●' : '○'}</span>
                <div>
                  <div className={styles.psychTactic}>{p.tactic}</div>
                  <div className={styles.psychDesc}>{p.description}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

function DomainStat({
  k, v, highlight,
}: { k: string; v: string; highlight?: 'safe'|'warn'|'danger'|'critical' }) {
  return (
    <div className={styles.domainStat}>
      <span className={styles.domainKey}>{k}</span>
      <span className={`${styles.domainVal} ${highlight ? styles[`hl_${highlight}`] : ''}`}>{v}</span>
    </div>
  )
}

export default ThreatSignals
