import { useState, useRef, useEffect } from 'react'
import { gsap } from 'gsap'
import styles from './ModelPerformance.module.css'

const TRACK_A = [
  { name: 'Logistic Regression', accuracy: 99.9958, f1: 99.9958, auc: 1.00, deploy: false },
  { name: 'Random Forest',       accuracy: 100.00,  f1: 100.00,  auc: 1.00, deploy: false },
  { name: 'XGBoost',             accuracy: 99.9958, f1: 99.9958, auc: 1.00, deploy: false },
  { name: 'LightGBM',            accuracy: 100.00,  f1: 100.00,  auc: 1.00, deploy: false },
]

const TRACK_B = [
  { name: 'LightGBM',            accuracy: 99.9936, f1: 99.9936, auc: 1.00, deploy: true  },
  { name: 'Logistic Regression', accuracy: 99.9936, f1: 99.9936, auc: 1.00, deploy: false },
  { name: 'XGBoost',             accuracy: 99.9894, f1: 99.9894, auc: 1.00, deploy: false },
  { name: 'Random Forest',       accuracy: 99.9851, f1: 99.9851, auc: 1.00, deploy: false },
]

type Track = 'A' | 'B'

export default function ModelPerformance() {
  const [track, setTrack] = useState<Track>('B')
  const gridRef = useRef<HTMLDivElement>(null)

  const models = track === 'A' ? TRACK_A : TRACK_B

  useEffect(() => {
    if (!gridRef.current) return
    gsap.fromTo(
      gridRef.current.querySelectorAll(`.${styles.card}`),
      { opacity: 0, y: 18 },
      { opacity: 1, y: 0, stagger: 0.08, duration: 0.55, ease: 'power3.out' }
    )
    // Animate bars
    gridRef.current.querySelectorAll<HTMLElement>('[data-bar]').forEach((bar) => {
      const target = parseFloat(bar.dataset.bar ?? '0')
      gsap.fromTo(bar,
        { width: '0%' },
        { width: `${target}%`, duration: 1.2, ease: 'power3.out', delay: 0.2 }
      )
    })
  }, [track])

  return (
    <div className={styles.wrap}>
      {/* Track switcher */}
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
      </div>

      {track === 'A' && (
        <div className={styles.notice}>
          Track A includes URLSimilarityIndex — a data leakage signal with 0.9961 AUROC.
          Excluded from production. Track B is the deployment-safe configuration.
        </div>
      )}

      <div ref={gridRef} className={styles.grid}>
        {models.map((m) => (
          <div key={m.name} className={`${styles.card} ${m.deploy ? styles.cardDeploy : ''}`}>
            {m.deploy && <span className={styles.deployBadge}>✦ Deployment Model</span>}

            <div className={styles.modelName}>{m.name}</div>

            <div className={styles.metrics}>
              {[
                { k: 'Accuracy', v: m.accuracy },
                { k: 'F1 Score', v: m.f1 },
                { k: 'ROC-AUC',  v: m.auc },
              ].map(({ k, v }) => (
                <div key={k} className={styles.metric}>
                  <span className={styles.metricVal}>
                    {v === 1.00 ? '1.00' : `${v.toFixed(4)}%`}
                  </span>
                  <span className={styles.metricKey}>{k}</span>
                </div>
              ))}
            </div>

            {/* Accuracy bar */}
            <div className={styles.barWrap}>
              <div className={styles.barLabel}>
                <span>Accuracy</span>
                <span>{m.accuracy === 100 ? '100.00%' : `${m.accuracy.toFixed(4)}%`}</span>
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
