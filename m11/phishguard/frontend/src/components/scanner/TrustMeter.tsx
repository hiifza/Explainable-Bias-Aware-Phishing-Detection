import { useEffect, useRef } from 'react'
import type { ScanResult } from '@/store'
import { gsap } from 'gsap'
import styles from './TrustMeter.module.css'

interface Props { result: ScanResult }

function getTrustColor(score: number): string {
  if (score >= 80) return 'var(--safe)'
  if (score >= 60) return 'var(--warn)'
  if (score >= 40) return 'var(--danger)'
  return 'var(--critical)'
}

function getTrustLabel(score: number): string {
  if (score >= 95) return 'Trusted'
  if (score >= 80) return 'Likely Safe'
  if (score >= 60) return 'Caution'
  if (score >= 40) return 'Suspicious'
  if (score >= 20) return 'Dangerous'
  return 'Critical Threat'
}

export default function TrustMeter({ result }: Props) {
  const arcRef      = useRef<SVGCircleElement>(null)
  const scoreRef    = useRef<HTMLDivElement>(null)
  const trustScore  = Math.round(result.trust_score)
  const color       = getTrustColor(trustScore)

  // Arc animation
  useEffect(() => {
    const circumference = 2 * Math.PI * 52
    const progress = trustScore / 100
    const dashOffset = circumference * (1 - progress)

    if (arcRef.current) {
      arcRef.current.style.strokeDasharray  = `${circumference}`
      arcRef.current.style.strokeDashoffset = `${circumference}`
      gsap.to(arcRef.current, {
        strokeDashoffset: dashOffset,
        duration: 1.6,
        ease: 'power3.out',
        delay: 0.2,
      })
    }

    // Counter
    if (scoreRef.current) {
      const obj = { val: 0 }
      gsap.to(obj, {
        val: trustScore,
        duration: 1.4,
        ease: 'power2.out',
        delay: 0.3,
        onUpdate: () => {
          if (scoreRef.current) {
            scoreRef.current.textContent = String(Math.round(obj.val))
          }
        },
      })
    }
  }, [trustScore])

  return (
    <div className={styles.wrap}>
      <div className={styles.arcWrap}>
        <svg viewBox="0 0 120 120" className={styles.arc}>
          {/* Track */}
          <circle
            cx="60" cy="60" r="52"
            fill="none"
            stroke="var(--border-subtle)"
            strokeWidth="6"
            strokeLinecap="round"
            transform="rotate(-90 60 60)"
          />
          {/* Progress */}
          <circle
            ref={arcRef}
            cx="60" cy="60" r="52"
            fill="none"
            stroke={color}
            strokeWidth="6"
            strokeLinecap="round"
            transform="rotate(-90 60 60)"
            style={{ filter: `drop-shadow(0 0 6px ${color}88)` }}
          />
        </svg>
        <div className={styles.inner}>
          <div className={styles.scoreNum} ref={scoreRef} style={{ color }}>
            0
          </div>
          <div className={styles.scoreLabel}>Trust Score</div>
        </div>
      </div>

      <div className={styles.verdict} style={{ color }}>
        {getTrustLabel(trustScore)}
      </div>

      <div className={styles.bars}>
        <ConfBar label="Phishing probability"   val={Math.round(result.phishing_probability * 100)}   color="var(--danger)" />
        <ConfBar label="Legitimate probability" val={Math.round(result.legitimate_probability * 100)} color="var(--safe)"   />
      </div>

      <div className={styles.zone}>
        <span className={styles.zoneLabel}>Reliability zone</span>
        <span className={`${styles.zoneBadge} ${styles[`zone${result.reliability_zone}`]}`}>
          {result.reliability_zone}
        </span>
      </div>
    </div>
  )
}

function ConfBar({ label, val, color }: { label: string; val: number; color: string }) {
  const fillRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (fillRef.current) {
      gsap.fromTo(fillRef.current,
        { width: '0%' },
        { width: `${val}%`, duration: 1.2, ease: 'power3.out', delay: 0.5 }
      )
    }
  }, [val])

  return (
    <div className={styles.barRow}>
      <span className={styles.barLabel}>{label}</span>
      <div className={styles.barBg}>
        <div ref={fillRef} className={styles.barFill} style={{ background: color, width: 0 }} />
      </div>
      <span className={styles.barVal} style={{ color }}>{val}%</span>
    </div>
  )
}
