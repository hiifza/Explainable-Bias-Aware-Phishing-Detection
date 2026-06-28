import { useState, useRef, useEffect } from 'react'
import { gsap } from 'gsap'
import { useModelData } from '@/hooks/useIntelligence'
import styles from './ModelPerformance.module.css'

type Track = 'A' | 'B'

export default function ModelPerformance() {
  const [track, setTrack] = useState<Track>('B')
  const { data, loading, source } = useModelData()
  const gridRef = useRef<HTMLDivElement>(null)

  const models = track === 'A'
    ? (data?.models?.track_A ?? [])
    : (data?.models?.track_B ?? [])

  useEffect(() => {
    if (!gridRef.current || loading) return
    gsap.fromTo(
      gridRef.current.querySelectorAll(`.${styles.card}`),
      { opacity: 0, y: 18 },
      { opacity: 1, y: 0, stagger: 0.08, duration: 0.55, ease: 'power3.out' }
    )
    gridRef.current.querySelectorAll<HTMLElement>('[data-bar]').forEach((bar) => {
      const target = parseFloat(bar.dataset.bar ?? '0')
      gsap.fromTo(bar,
        { width: '0%' },
        { width: `${target}%`, duration: 1.2, ease: 'power3.out', delay: 0.2 }
      )
    })
  }, [track, loading])

  if (loading) return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
      {[1,2,3,4].map(i => (
        <div key={i} style={{ height: 160, borderRadius: 'var(--r-lg)', background: 'var(--surface-1)', animation: 'pulseGlow 1.6s ease-in-out infinite' }} />
      ))}
    </div>
  )

  return (
    <div className={styles.wrap}>
      <div className={styles.tabs}>
        {(['A', 'B'] as Track[]).map((t) => (
          <button
            key={t}
            className={`${styles.tab} ${track === t ? styles.tabActive : ''}`}
            onClick={() => setTrack(t)}
          >
            {t === 'B' ? 'Track B · Deployment' : 'Track A · Research'}
          </button>
        ))}
        {source === 'api' && <span className={styles.liveTag}>Live</span>}
      </div>

      {track === 'A' && (
        <div className={styles.notice}>
          Track A includes URLSimilarityIndex — a data leakage signal with 0.9961 AUROC.
          Excluded from production. Track B is the deployment-safe configuration.
        </div>
      )}

      <div ref={gridRef} className={styles.grid}>
        {models.map((m: any) => (
          <div key={m.name} className={`${styles.card} ${m.deploy ? styles.cardDeploy : ''}`}>
            {m.deploy && <span className={styles.deployBadge}>✦ Deployment Model</span>}
            <div className={styles.modelName}>{m.name}</div>
            <div className={styles.metrics}>
              {[
                { k: 'Accuracy', v: m.accuracy },
                { k: 'F1 Score', v: m.f1 },
                { k: 'ROC-AUC',  v: m.roc_auc },
              ].map(({ k, v }) => (
                <div key={k} className={styles.metric}>
                  <span className={styles.metricVal}>
                    {v === 1.00 || v === 1 ? '1.00' : `${parseFloat(v).toFixed(4)}%`}
                  </span>
                  <span className={styles.metricKey}>{k}</span>
                </div>
              ))}
            </div>
            <div className={styles.barWrap}>
              <div className={styles.barLabel}>
                <span>Accuracy</span>
                <span>{m.accuracy === 100 ? '100.00%' : `${parseFloat(m.accuracy).toFixed(4)}%`}</span>
              </div>
              <div className={styles.barBg}>
                <div
                  className={`${styles.barFill} ${m.deploy ? styles.barFillDeploy : ''}`}
                  data-bar={m.accuracy}
                  style={{ width: 0 }}
                />
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
