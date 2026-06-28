import { useEffect, useRef } from 'react'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import ThreatGlobe from '@/components/3d/ThreatGlobe'
import Scanner from '@/components/scanner/Scanner'
import MetricCards from '@/components/intelligence/MetricCards'
import ModelPerformance from '@/components/intelligence/ModelPerformance'
import ShapSection from '@/components/intelligence/ShapSection'
import ShapLimeConflict from '@/components/intelligence/ShapLimeConflict'
import BlindspotSection from '@/components/intelligence/BlindspotSection'
import BiasSection from '@/components/intelligence/BiasSection'
import ReliabilitySection from '@/components/intelligence/ReliabilitySection'
import { useAppStore } from '@/store'
import styles from './Home.module.css'

gsap.registerPlugin(ScrollTrigger)

export default function Home() {
  const ref  = useRef<HTMLDivElement>(null)
  const mode = useAppStore((s) => s.mode)

  useEffect(() => {
    const ctx = gsap.context(() => {
      // Staggered hero entrance
      const tl = gsap.timeline({ delay: 0.25 })
      tl.fromTo('.pg-eyebrow',   { opacity:0, y:14 }, { opacity:1, y:0, duration:0.6, ease:'power3.out' })
        .fromTo('.pg-title',     { opacity:0, y:28 }, { opacity:1, y:0, duration:0.8, ease:'power3.out' }, '-=0.3')
        .fromTo('.pg-sub',       { opacity:0, y:18 }, { opacity:1, y:0, duration:0.6, ease:'power3.out' }, '-=0.4')
        .fromTo('.pg-scanner',   { opacity:0, y:22, scale:0.98 }, { opacity:1, y:0, scale:1, duration:0.7, ease:'power3.out' }, '-=0.3')
        .fromTo('.pg-metrics',   { opacity:0, y:18 }, { opacity:1, y:0, duration:0.6, ease:'power3.out' }, '-=0.2')

      // Section reveals
      gsap.utils.toArray<HTMLElement>('.pg-reveal').forEach((el) => {
        gsap.fromTo(el,
          { opacity:0, y:36 },
          { opacity:1, y:0, duration:0.85, ease:'power3.out',
            scrollTrigger: { trigger: el, start:'top 88%', once:true } }
        )
      })
    }, ref)
    return () => ctx.revert()
  }, [])

  return (
    <div ref={ref} className={styles.home}>

      {/* ── HERO ─────────────────────────────────────────── */}
      <section className={styles.hero}>
        <div className={styles.gridBg} />
        <ThreatGlobe />

        <div className={styles.heroGrid}>
          <div className={`${styles.eyebrow} pg-eyebrow`}>
            <span className={styles.eyebrowDot} />
            Explainable · Bias-Aware · Phishing Detection
          </div>

          <h1 className={`${styles.heroTitle} pg-title`}>
            Is this website
            <span className={styles.accentWord}>safe to trust?</span>
          </h1>

          <p className={`${styles.heroSub} pg-sub`}>
            PhishGuard analyzes any URL through a machine learning system
            trained on 235,795 real phishing and legitimate sites — and explains
            exactly why it reached its verdict.
          </p>

          <div className={`pg-scanner ${styles.scannerWrap}`}>
            <Scanner />
          </div>
        </div>

        <div className={`pg-metrics ${styles.heroMetrics}`}>
          <MetricCards />
        </div>

        <div className={styles.scrollHint} aria-hidden="true">
          <span className={styles.scrollLine} />
          <span className={styles.scrollLabel}>Intelligence Below</span>
        </div>
      </section>

      {/* ── MODEL PERFORMANCE ──────────────────────────── */}
      <section className={styles.section} id="models">
        <div className="section-inner">
          <div className="pg-reveal">
            <div className="section-tag">Model Performance Laboratory</div>
            <h2 className={styles.sectionTitle}>Four models.<br />Two tracks. Near-perfect.</h2>
            <p className={styles.sectionSub}>
              Track A includes URLSimilarityIndex (data leakage). Track B is production-safe.
              Track B LightGBM is the deployment model.
            </p>
          </div>
          <div className="pg-reveal"><ModelPerformance /></div>
        </div>
      </section>

      {/* ── SHAP ────────────────────────────────────────── */}
      <section className={`${styles.section} ${styles.sectionAlt}`} id="shap">
        <div className="section-inner">
          <div className="pg-reveal">
            <div className="section-tag">SHAP Explainability</div>
            <h2 className={styles.sectionTitle}>What drives every decision?</h2>
            <p className={styles.sectionSub}>
              SHAP reveals the exact contribution of each feature — globally
              consistent across all 235,795 samples.
            </p>
          </div>
          <div className="pg-reveal"><ShapSection /></div>
        </div>
      </section>

      {/* ── CONFLICT ────────────────────────────────────── */}
      <section className={styles.section} id="conflict">
        <div className="section-inner">
          <div className="pg-reveal">
            <div className="section-tag">SHAP vs LIME Conflict Analyzer</div>
            <h2 className={styles.sectionTitle}>Two explanation methods.<br />Zero agreement.</h2>
            <p className={styles.sectionSub}>
              The most critical finding: despite near-perfect predictions, SHAP and LIME
              identify entirely different explanatory features for the same decisions.
            </p>
          </div>
          <div className="pg-reveal"><ShapLimeConflict /></div>
        </div>
      </section>

      {/* ── BLIND SPOTS (expert only) ──────────────────── */}
      {mode === 'expert' && (
        <section className={`${styles.section} ${styles.sectionAlt}`} id="blindspots">
          <div className="section-inner">
            <div className="pg-reveal">
              <div className="section-tag">Blind Spot Investigation Center</div>
              <h2 className={styles.sectionTitle}>3 failures in 47,159 samples.<br />All found.</h2>
            </div>
            <div className="pg-reveal"><BlindspotSection /></div>
          </div>
        </section>
      )}

      {/* ── BIAS ────────────────────────────────────────── */}
      <section className={`${styles.section} ${mode === 'expert' ? '' : styles.sectionAlt}`} id="bias">
        <div className="section-inner">
          <div className="pg-reveal">
            <div className="section-tag">Bias & Fairness Observatory</div>
            <h2 className={styles.sectionTitle}>Audited for fairness<br />across five dimensions.</h2>
            <p className={styles.sectionSub}>
              URL length, domain length, HTTPS status, TLD groups, external resources.
              Every dimension passes.
            </p>
          </div>
          <div className="pg-reveal"><BiasSection /></div>
        </div>
      </section>

      {/* ── RELIABILITY (expert only) ──────────────────── */}
      {mode === 'expert' && (
        <section className={styles.section} id="reliability">
          <div className="section-inner">
            <div className="pg-reveal">
              <div className="section-tag">Reliability Analysis Center</div>
              <h2 className={styles.sectionTitle}>13.04% error rate.<br />97.31% confidence.</h2>
              <p className={styles.sectionSub}>
                The Red Zone reveals that model confidence alone does not predict
                reliability — explanation agreement does.
              </p>
            </div>
            <div className="pg-reveal"><ReliabilitySection /></div>
          </div>
        </section>
      )}

    </div>
  )
}
