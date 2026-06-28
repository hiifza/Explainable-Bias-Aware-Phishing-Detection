import { useRef, useEffect } from 'react'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import { useShapData } from '@/hooks/useIntelligence'
import styles from './ShapSection.module.css'

const HUMAN_LABELS: Record<string, string> = {
  LetterRatioInURL:           'Letter ratio in URL',
  LineOfCode:                 'Page line count',
  IsHTTPS:                    'HTTPS encryption',
  NoOfDegitsInURL:            'Digits in URL',
  DomainLength:               'Domain name length',
  NoOfSelfRef:                'Self-references',
  NoOfOtherSpecialCharsInURL: 'Special chars in URL',
  LargestLineLength:          'Largest line length',
  NoOfExternalRef:            'External references',
  SpacialCharRatioInURL:      'Special char ratio',
  URLLength:                  'URL total length',
  HasPasswordField:           'Password field present',
  HasTitle:                   'Page has title',
  ObfuscationRatio:           'Obfuscation ratio',
}

export default function ShapSection() {
  const { data, loading, source } = useShapData()
  const listRef = useRef<HTMLDivElement>(null)

  const features = data?.features ?? []
  const maxPct = features.length > 0
    ? Math.max(...features.map((f: any) => parseFloat(f.importance_pct ?? f.pct ?? 0)))
    : 10.51

  useEffect(() => {
    if (!listRef.current || loading || features.length === 0) return
    const bars = listRef.current.querySelectorAll<HTMLElement>('[data-bar-width]')

    ScrollTrigger.create({
      trigger: listRef.current,
      start: 'top 80%',
      once: true,
      onEnter: () => {
        bars.forEach((bar, i) => {
          gsap.fromTo(bar,
            { width: '0%' },
            { width: bar.dataset.barWidth + '%', duration: 1.1, ease: 'power3.out', delay: i * 0.06 }
          )
        })
        const rows = listRef.current!.querySelectorAll(`.${styles.row}`)
        gsap.fromTo(rows,
          { opacity: 0, x: -16 },
          { opacity: 1, x: 0, stagger: 0.05, duration: 0.5, ease: 'power3.out' }
        )
      },
    })
  }, [loading, features.length])

  if (loading) return <ShapSkeleton />

  return (
    <div className={styles.wrap}>
      {source === 'api' && (
        <div className={styles.sourceTag}>Live data from outputs/reports/shap_feature_ranking.csv</div>
      )}
      <div ref={listRef} className={styles.list}>
        {features.slice(0, 15).map((f: any) => {
          const pct   = parseFloat(f.importance_pct ?? f.pct ?? 0)
          const barW  = (pct / maxPct) * 100
          const name  = f.feature ?? f.name ?? ''
          const human = HUMAN_LABELS[name] ?? name
          const rank  = f.rank ?? (features.indexOf(f) + 1)
          return (
            <div key={name} className={styles.row}>
              <span className={styles.rank}>#{rank}</span>
              <div className={styles.info}>
                <div className={styles.names}>
                  <span className={styles.humanName}>{human}</span>
                  <span className={styles.techName}>{name}</span>
                </div>
                <div className={styles.barBg}>
                  <div
                    className={styles.barFill}
                    data-bar-width={barW.toFixed(1)}
                    style={{ width: 0 }}
                  />
                </div>
              </div>
              <span className={styles.pct}>{pct.toFixed(2)}%</span>
            </div>
          )
        })}
      </div>

      <div className={styles.note}>
        <span className={styles.noteIcon}>◈</span>
        <span>
          URLSimilarityIndex contributes <strong>18.68%</strong> to Track A predictions —
          classified as a critical data leakage signal and excluded from Track B deployment.
        </span>
      </div>
    </div>
  )
}

function ShapSkeleton() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      {Array.from({ length: 8 }).map((_, i) => (
        <div key={i} style={{
          height: 44, borderRadius: 'var(--r-md)',
          background: 'var(--surface-1)',
          opacity: 1 - i * 0.08,
          animation: 'pulseGlow 1.6s ease-in-out infinite',
        }} />
      ))}
    </div>
  )
}
