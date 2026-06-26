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
  const heroRef  = useRef<HTMLDivElement>(null)
  const titleRef = useRef<HTMLHeadingElement>(null)
  const mode     = useAppStore((s) => s.mode)

  useEffect(() => {
    const ctx = gsap.context(() => {
      // Hero entrance
      gsap.fromTo('.pg-hero-eyebrow',
        { opacity: 0, y: 16 },
        { opacity: 1, y: 0, duration: 0.7, ease: 'power3.out', delay: 0.2 }
      )
      gsap.fromTo('.pg-hero-title',
        { opacity: 0, y: 24 },
        { opacity: 1, y: 0, duration: 0.9, ease: 'power3.out', delay: 0.4 }
      )
      gsap.fromTo('.pg-hero-sub',
        { opacity: 0, y: 16 },
        { opacity: 1, y: 0, duration: 0.7, ease: 'power3.out', delay: 0.6 }
      )
      gsap.fromTo('.pg-hero-scanner',
        { opacity: 0, y: 20, scale: 0.98 },
        { opacity: 1, y: 0, scale: 1, duration: 0.8, ease: 'power3.out', delay: 0.8 }
      )
      gsap.fromTo('.pg-hero-metrics',
        { opacity: 0, y: 20 },
        { opacity: 1, y: 0, duration: 0.7, ease: 'power3.out', delay: 1.0 }
      )

      // Section reveals
      gsap.utils.toArray<HTMLElement>('.pg-reveal').forEach((el) => {
        gsap.fromTo(el,
          { opacity: 0, y: 32 },
          {
            opacity: 1, y: 0, duration: 0.8, ease: 'power3.out',
            scrollTrigger: { trigger: el, start: 'top 88%', once: true },
          }
        )
      })
    }, heroRef)

    return () => ctx.revert()
  }, [])

  return (
    <div ref={heroRef} className={styles.home}>

      {/* ── HERO ─────────────────────────────────────────────── */}
      <section className={styles.hero}>
        <ThreatGlobe />

        <div className={styles.heroGrid}>
          <div className={styles.heroContent}>
            <div className={`${styles.eyebrow} pg-hero-eyebrow`}>
              <span className={styles.eyebrowDot} />
              Explainable · Bias-Aware · Phishing Detection Intelligence
            </div>

            <h1 className={`${styles.heroTitle} pg-hero-title`} ref={titleRef}>
              Is this website
              <br />
              <span className={styles.accentWord}>safe to trust?</span>
            </h1>

            <p className={`${styles.heroSub} pg-hero-sub`}>
              PhishGuard analyzes any URL using a machine learning system
              trained on 235,795 real phishing and legitimate websites — and explains
              exactly why it reached its decision.
            </p>

            <div className={`pg-hero-scanner ${styles.scannerWrap}`}>
              <Scanner />
            </div>
          </div>
        </div>

        <div className={`pg-hero-metrics ${styles.heroMetrics}`}>
          <MetricCards />
        </div>

        <div className={styles.scrollHint} aria-hidden="true">
          <span className={styles.scrollLine} />
          <span className={styles.scrollLabel}>Intelligence Report Below</span>
        </div>
      </section>

      {/* ── MODEL PERFORMANCE ────────────────────────────────── */}
      <section className={styles.section} id="models">
        <div className="section-inner">
          <div className="pg-reveal">
            <div className="section-tag">Model Performance Laboratory</div>
            <h2 className={styles.sectionTitle}>
              Four models. Two tracks.
              <br />Near-perfect detection.
            </h2>
            <p className={styles.sectionSub}>
              All four architectures evaluated across Track A (with URLSimilarityIndex leakage signal)
              and Track B (production-safe). Track B LightGBM is the deployment model.
            </p>
          </div>
          <div className="pg-reveal">
            <ModelPerformance />
          </div>
        </div>
      </section>

      {/* ── SHAP EXPLAINABILITY ──────────────────────────────── */}
      <section className={`${styles.section} ${styles.sectionAlt}`} id="shap">
        <div className="section-inner">
          <div className="pg-reveal">
            <div className="section-tag">SHAP Explainability</div>
            <h2 className={styles.sectionTitle}>What drives every decision?</h2>
            <p className={styles.sectionSub}>
              SHAP (SHapley Additive exPlanations) reveals the precise contribution of each
              feature — globally consistent across all 235,795 samples.
            </p>
          </div>
          <div className="pg-reveal">
            <ShapSection />
          </div>
        </div>
      </section>

      {/* ── SHAP vs LIME ─────────────────────────────────────── */}
      <section className={styles.section} id="conflict">
        <div className="section-inner">
          <div className="pg-reveal">
            <div className="section-tag">SHAP vs LIME Conflict Analyzer</div>
            <h2 className={styles.sectionTitle}>
              Two explanation methods.
              <br />Zero agreement.
            </h2>
            <p className={styles.sectionSub}>
              The most critical research finding: despite near-perfect predictions,
              SHAP and LIME identify entirely different explanatory features.
            </p>
          </div>
          <div className="pg-reveal">
            <ShapLimeConflict />
          </div>
        </div>
      </section>

      {/* ── BLIND SPOTS ──────────────────────────────────────── */}
      {mode === 'expert' && (
        <section className={`${styles.section} ${styles.sectionAlt}`} id="blindspots">
          <div className="section-inner">
            <div className="pg-reveal">
              <div className="section-tag">Blind Spot Investigation Center</div>
              <h2 className={styles.sectionTitle}>
                3 failures in 47,159 samples.
                <br />We found all three.
              </h2>
            </div>
            <div className="pg-reveal">
              <BlindspotSection />
            </div>
          </div>
        </section>
      )}

      {/* ── BIAS ─────────────────────────────────────────────── */}
      <section className={`${styles.section} ${mode === 'expert' ? '' : styles.sectionAlt}`} id="bias">
        <div className="section-inner">
          <div className="pg-reveal">
            <div className="section-tag">Bias & Fairness Observatory</div>
            <h2 className={styles.sectionTitle}>Audited for fairness.</h2>
            <p className={styles.sectionSub}>
              Five dimensions of performance fairness — URL length, domain length,
              HTTPS status, TLD groups, and external resources.
              All five pass.
            </p>
          </div>
          <div className="pg-reveal">
            <BiasSection />
          </div>
        </div>
      </section>

      {/* ── RELIABILITY ──────────────────────────────────────── */}
      {mode === 'expert' && (
        <section className={styles.section} id="reliability">
          <div className="section-inner">
            <div className="pg-reveal">
              <div className="section-tag">Reliability Analysis Center</div>
              <h2 className={styles.sectionTitle}>
                When confidence misleads.
              </h2>
              <p className={styles.sectionSub}>
                The Red Zone: 13.04% error rate despite 97.31% mean model confidence.
                Explanation agreement is a better reliability signal than confidence alone.
              </p>
            </div>
            <div className="pg-reveal">
              <ReliabilitySection />
            </div>
          </div>
        </section>
      )}

    </div>
  )
}
