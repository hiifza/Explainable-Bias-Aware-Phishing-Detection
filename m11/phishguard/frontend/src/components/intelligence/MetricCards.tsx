import { useEffect, useRef } from 'react'
import { gsap } from 'gsap'
import styles from './MetricCards.module.css'

const METRICS = [
  { value: 100,    suffix: '%',  label: 'Best Accuracy',        color: 'safe',    decimals: 0 },
  { value: 235795, suffix: '',   label: 'URLs Analyzed',        color: 'accent',  decimals: 0 },
  { value: 56,     suffix: '',   label: 'Features Analyzed',    color: 'accent',  decimals: 0 },
  { value: 4,      suffix: '',   label: 'Models Evaluated',     color: 'accent',  decimals: 0 },
  { value: 3,      suffix: '',   label: 'Critical Failures',    color: 'danger',  decimals: 0 },
  { value: 0,      suffix: '%',  label: 'Bias Violations',      color: 'safe',    decimals: 0 },
  { value: 13.04,  suffix: '%',  label: 'Red Zone Error Rate',  color: 'warn',    decimals: 2 },
  { value: 1.0,    suffix: '',   label: 'Best ROC-AUC',         color: 'safe',    decimals: 2 },
]

export default function MetricCards() {
  const cardsRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!cardsRef.current) return
    const counters = cardsRef.current.querySelectorAll<HTMLElement>('[data-count]')

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return
          const el = entry.target as HTMLElement
          const target   = parseFloat(el.dataset.count ?? '0')
          const decimals = parseInt(el.dataset.decimals ?? '0')
          const obj = { val: 0 }
          gsap.to(obj, {
            val: target,
            duration: 1.8,
            ease: 'power3.out',
            delay: parseFloat(el.dataset.delay ?? '0'),
            onUpdate: () => {
              el.textContent = decimals > 0
                ? obj.val.toFixed(decimals)
                : Math.round(obj.val).toLocaleString()
            },
          })
          observer.unobserve(el)
        })
      },
      { threshold: 0.4 }
    )

    counters.forEach((c) => observer.observe(c))
    return () => observer.disconnect()
  }, [])

  return (
    <div ref={cardsRef} className={styles.grid}>
      {METRICS.map((m, i) => (
        <div key={m.label} className={`${styles.card} ${styles[`card_${m.color}`]}`}>
          <div className={`${styles.value} ${styles[`val_${m.color}`]}`}>
            <span
              data-count={m.value}
              data-decimals={m.decimals}
              data-delay={i * 0.07}
            >
              {m.decimals > 0 ? m.value.toFixed(m.decimals) : m.value.toLocaleString()}
            </span>
            {m.suffix && <span className={styles.suffix}>{m.suffix}</span>}
          </div>
          <div className={styles.label}>{m.label}</div>
        </div>
      ))}
    </div>
  )
}
