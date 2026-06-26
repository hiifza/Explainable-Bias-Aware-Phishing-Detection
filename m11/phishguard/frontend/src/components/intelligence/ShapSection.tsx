import { useRef, useEffect } from 'react'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import styles from './ShapSection.module.css'

const FEATURES = [
  { rank: 1,  name: 'LetterRatioInURL',             pct: 10.51, human: 'Letter ratio in URL' },
  { rank: 2,  name: 'LineOfCode',                   pct: 10.46, human: 'Page line count' },
  { rank: 3,  name: 'IsHTTPS',                      pct:  9.26, human: 'HTTPS encryption' },
  { rank: 4,  name: 'NoOfDegitsInURL',              pct:  8.65, human: 'Digits in URL' },
  { rank: 5,  name: 'DomainLength',                 pct:  7.09, human: 'Domain name length' },
  { rank: 6,  name: 'NoOfSelfRef',                  pct:  5.80, human: 'Self-references' },
  { rank: 7,  name: 'NoOfOtherSpecialCharsInURL',   pct:  4.90, human: 'Special chars in URL' },
  { rank: 8,  name: 'LargestLineLength',            pct:  4.40, human: 'Largest line length' },
  { rank: 9,  name: 'NoOfExternalRef',              pct:  4.00, human: 'External references' },
  { rank: 10, name: 'SpacialCharRatioInURL',        pct:  3.50, human: 'Special char ratio' },
]

const MAX = 10.51

export default function ShapSection() {
  const listRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!listRef.current) return
    const bars = listRef.current.querySelectorAll<HTMLElement>('[data-bar-width]')

    ScrollTrigger.create({
      trigger: listRef.current,
      start: 'top 80%',
      once: true,
      onEnter: () => {
        bars.forEach((bar, i) => {
          gsap.fromTo(bar,
            { width: '0%' },
            {
              width: bar.dataset.barWidth + '%',
              duration: 1.1,
              ease: 'power3.out',
              delay: i * 0.06,
            }
          )
        })
        const rows = listRef.current!.querySelectorAll(`.${styles.row}`)
        gsap.fromTo(rows,
          { opacity: 0, x: -16 },
          { opacity: 1, x: 0, stagger: 0.05, duration: 0.5, ease: 'power3.out' }
        )
      },
    })
  }, [])

  return (
    <div className={styles.wrap}>
      <div ref={listRef} className={styles.list}>
        {FEATURES.map((f) => {
          const barW = (f.pct / MAX) * 100
          return (
            <div key={f.rank} className={styles.row}>
              <span className={styles.rank}>#{f.rank}</span>
              <div className={styles.info}>
                <div className={styles.names}>
                  <span className={styles.humanName}>{f.human}</span>
                  <span className={styles.techName}>{f.name}</span>
                </div>
                <div className={styles.barBg}>
                  <div
                    className={styles.barFill}
                    data-bar-width={barW.toFixed(1)}
                    style={{ width: 0 }}
                  />
                </div>
              </div>
              <span className={styles.pct}>{f.pct.toFixed(2)}%</span>
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
