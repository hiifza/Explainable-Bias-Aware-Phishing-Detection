import { useRef, useEffect } from 'react'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import { useShapData, useLimeData } from '@/hooks/useIntelligence'
import styles from './ShapLimeConflict.module.css'

const LIME_TOP = [
  'HasPasswordField', 'NoOfExternalRef', 'URLLength',
  'HasTitle', 'LargestLineLength', 'ObfuscationRatio',
]

export default function ShapLimeConflict() {
  const { data: shapData, loading: shapLoading } = useShapData()
  const { data: limeData, loading: limeLoading } = useLimeData()
  const scoreRef   = useRef<HTMLDivElement>(null)
  const sectionRef = useRef<HTMLDivElement>(null)

  const shapTop = shapData?.features?.slice(0, 6).map((f: any) => f.feature ?? f.name) ?? []
  const limeTop = limeData?.agreement_data?.slice(0, 6).map((r: any) => r.feature ?? r.lime_feature) ?? LIME_TOP
  const actualLime = limeTop.length > 0 ? limeTop : LIME_TOP

  const agree   = limeData?.summary?.mean_agreement ?? 0.52
  const localAgree = limeData?.summary?.local_agreement_pct ?? 0

  useEffect(() => {
    if (shapLoading || limeLoading || !sectionRef.current) return
    ScrollTrigger.create({
      trigger: sectionRef.current,
      start: 'top 75%',
      once: true,
      onEnter: () => {
        if (scoreRef.current) {
          gsap.fromTo(scoreRef.current,
            { scale: 0.6, opacity: 0 },
            { scale: 1, opacity: 1, duration: 0.8, ease: 'back.out(1.7)', delay: 0.2 }
          )
        }
        gsap.fromTo(`.${styles.shapRow}, .${styles.limeRow}`,
          { opacity: 0, y: 12 },
          { opacity: 1, y: 0, stagger: 0.07, duration: 0.5, ease: 'power3.out', delay: 0.4 }
        )
      },
    })
  }, [shapLoading, limeLoading])

  return (
    <div ref={sectionRef} className={styles.wrap}>
      <div className={styles.conflictRow}>

        {/* SHAP panel */}
        <div className={styles.methodPanel}>
          <div className={styles.methodHeader}>
            <span className={`${styles.methodChip} ${styles.chipShap}`}>SHAP</span>
            <span className={styles.methodTitle}>Global Importance</span>
          </div>
          <div className={styles.methodMeta}>
            Shapley values · Mathematically guaranteed · Model-consistent
          </div>
          <div className={styles.featureList}>
            {(shapTop.length > 0 ? shapTop : ['LetterRatioInURL','LineOfCode','IsHTTPS','NoOfDegitsInURL','DomainLength','NoOfSelfRef']).map((f: string, i: number) => (
              <div key={f} className={styles.shapRow}>
                <span className={styles.featRank}>{i + 1}</span>
                <span className={styles.featName}>{f}</span>
                <div className={styles.featBar}>
                  <div className={styles.featFillShap} style={{ width: `${100 - i * 14}%` }} />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Center */}
        <div className={styles.centerCol}>
          <div ref={scoreRef} className={styles.conflictScore}>{localAgree}%</div>
          <div className={styles.conflictLabel}>Agreement</div>
          <div className={styles.conflictArrows}>
            <span className={styles.arrow}>←</span>
            <span className={styles.arrowLabel}>vs</span>
            <span className={styles.arrow}>→</span>
          </div>
          <div className={styles.conflictSub}>
            Local feature<br />overlap across<br />top explanations
          </div>
        </div>

        {/* LIME panel */}
        <div className={styles.methodPanel}>
          <div className={styles.methodHeader}>
            <span className={`${styles.methodChip} ${styles.chipLime}`}>LIME</span>
            <span className={styles.methodTitle}>Local Explanation</span>
          </div>
          <div className={styles.methodMeta}>
            Surrogate model · Instance-level · Locally faithful
          </div>
          <div className={styles.featureList}>
            {actualLime.slice(0, 6).map((f: string, i: number) => (
              <div key={f} className={styles.limeRow}>
                <span className={styles.featRank}>{i + 1}</span>
                <span className={styles.featName}>{f}</span>
                <div className={styles.featBar}>
                  <div className={styles.featFillLime} style={{ width: `${100 - i * 14}%` }} />
                </div>
              </div>
            ))}
          </div>
        </div>

      </div>

      <div className={styles.statsStrip}>
        <Stat label="Mean SHAP-LIME Agreement" value={agree.toFixed(2)}         note="global"   color="warn"   />
        <Stat label="Feature Consistency"       value={(limeData?.summary?.feature_consistency ?? 0.60).toFixed(2)} note="score"    color="warn"   />
        <Stat label="Shared Top-20 Features"   value={String(limeData?.summary?.shared_top20 ?? 12)}              note="of 20"   color="warn"   />
        <Stat label="Local Agreement"           value={`${localAgree}%`}         note="critical" color="danger" />
      </div>

      <div className={styles.finding}>
        <div className={styles.findingIcon}>⊗</div>
        <div className={styles.findingBody}>
          <div className={styles.findingTitle}>Critical Explanation Conflict</div>
          <p className={styles.findingText}>
            SHAP and LIME provide completely different local explanations for identical predictions.
            In the Red Zone (agreement 0.0–0.2), this translates directly to a 13.04% error rate —
            despite the model showing 97.31% mean confidence. This finding challenges the assumption
            that confidence scores alone are sufficient for deployment-time reliability estimation.
          </p>
        </div>
      </div>
    </div>
  )
}

function Stat({ label, value, note, color }: { label: string; value: string; note: string; color: 'warn' | 'danger' }) {
  return (
    <div className={styles.stat}>
      <div className={`${styles.statVal} ${styles[`statVal_${color}`]}`}>{value}</div>
      <div className={styles.statNote}>{note}</div>
      <div className={styles.statLabel}>{label}</div>
    </div>
  )
}
