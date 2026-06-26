import { useEffect, useRef } from 'react'
import { useAppStore } from '@/store'
import { gsap } from 'gsap'
import TrustMeter from './TrustMeter'
import ThreatSignals from './ThreatSignals'
import ExpertPanel from './ExpertPanel'
import AttackerSim from './AttackerSim'
import ActionGuidance from './ActionGuidance'
import styles from './ScanResult.module.css'

const RISK_LABELS: Record<string, { label: string; color: string }> = {
  SAFE:       { label: '✓  Safe',              color: 'var(--safe)' },
  LOW_RISK:   { label: '◎  Low Risk',          color: 'var(--safe)' },
  SUSPICIOUS: { label: '⚠  Suspicious',        color: 'var(--warn)' },
  HIGH_RISK:  { label: '⚠  High Risk',         color: 'var(--danger)' },
  CRITICAL:   { label: '⊗  Critical Threat',   color: 'var(--critical)' },
}

export default function ScanResult() {
  const { scanResult, setShowResult, mode } = useAppStore()
  const overlayRef = useRef<HTMLDivElement>(null)
  const panelRef   = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // Entrance animation
    gsap.fromTo(overlayRef.current,
      { opacity: 0 },
      { opacity: 1, duration: 0.3, ease: 'power2.out' }
    )
    gsap.fromTo(panelRef.current,
      { opacity: 0, y: 30 },
      { opacity: 1, y: 0, duration: 0.5, ease: 'power3.out', delay: 0.1 }
    )
    document.body.style.overflow = 'hidden'
    return () => { document.body.style.overflow = '' }
  }, [])

  const close = () => {
    gsap.to(overlayRef.current, {
      opacity: 0, duration: 0.25, ease: 'power2.in',
      onComplete: () => setShowResult(false),
    })
  }

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') close() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  if (!scanResult) return null

  const risk = RISK_LABELS[scanResult.risk_level] ?? RISK_LABELS.SUSPICIOUS

  return (
    <div ref={overlayRef} className={styles.overlay} role="dialog" aria-modal="true">
      <div ref={panelRef} className={styles.panel}>

        {/* Header */}
        <div className={styles.header}>
          <div className={styles.headerLeft}>
            <div className={styles.headerTag}>Intelligence Briefing</div>
            <div className={styles.headerUrl} title={scanResult.url}>
              {scanResult.url.length > 70
                ? scanResult.url.slice(0, 70) + '…'
                : scanResult.url}
            </div>
          </div>
          <button className={styles.closeBtn} onClick={close} aria-label="Close">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M18 6L6 18M6 6l12 12"/>
            </svg>
          </button>
        </div>

        {/* Verdict banner */}
        <div className={styles.verdict} style={{ '--risk-color': risk.color } as React.CSSProperties}>
          <span className={styles.verdictLabel} style={{ color: risk.color }}>
            {risk.label}
          </span>
          <span className={styles.verdictConf}>
            {Math.round(scanResult.confidence * 100)}% confidence
          </span>
        </div>

        <div className={styles.body}>
          {/* Left column: Trust meter + human explanation */}
          <div className={styles.col}>
            <TrustMeter result={scanResult} />
            <div className={styles.narrative}>
              <div className={styles.narrativeLabel}>What this means</div>
              <p className={styles.narrativeText}>{scanResult.human_explanation}</p>
            </div>
            <ActionGuidance result={scanResult} />
          </div>

          {/* Right column */}
          <div className={styles.col}>
            <ThreatSignals signals={scanResult.threat_signals} />
            {(scanResult.brand_impersonation) && (
              <div className={styles.brandAlert}>
                <div className={styles.brandAlertIcon}>🎭</div>
                <div>
                  <div className={styles.brandAlertTitle}>
                    Brand Impersonation Detected
                  </div>
                  <div className={styles.brandAlertBody}>
                    Potential impersonation of <strong>{scanResult.brand_impersonation.target_brand}</strong>
                    {' '}({Math.round(scanResult.brand_impersonation.confidence * 100)}% confidence).
                    {' '}{scanResult.brand_impersonation.indicators.slice(0,2).join('; ')}.
                  </div>
                </div>
              </div>
            )}
            <AttackerSim signals={scanResult.attacker_simulation} />
          </div>
        </div>

        {/* Expert mode expansion */}
        {mode === 'expert' && (
          <ExpertPanel result={scanResult} />
        )}

      </div>
    </div>
  )
}
