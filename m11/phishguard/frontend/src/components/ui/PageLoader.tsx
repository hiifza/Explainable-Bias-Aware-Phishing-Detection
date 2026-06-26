import { useEffect, useState } from 'react'
import { gsap } from 'gsap'
import styles from './PageLoader.module.css'

export default function PageLoader() {
  const [hidden, setHidden] = useState(false)

  useEffect(() => {
    const tl = gsap.timeline({
      onComplete: () => setHidden(true),
    })
    tl.to(`.${styles.fill}`, { width: '100%', duration: 1.1, ease: 'power2.inOut' })
    tl.to(`.${styles.loader}`, { opacity: 0, duration: 0.4, ease: 'power2.in' }, '+=0.15')
  }, [])

  if (hidden) return null

  return (
    <div className={styles.loader}>
      <div className={styles.inner}>
        <div className={styles.wordmark}>PhishGuard</div>
        <div className={styles.tagline}>Cyber Trust Intelligence Platform</div>
        <div className={styles.barBg}>
          <div className={styles.fill} />
        </div>
        <div className={styles.status}>Initializing intelligence systems…</div>
      </div>
    </div>
  )
}
