// Investigate.tsx
import { BlindspotSection } from '@/components/intelligence/IntelligenceSections'
import { ReliabilitySection } from '@/components/intelligence/IntelligenceSections'
import styles from './SubPage.module.css'

const ARCHETYPES = [
  {
    letter: 'A', name: 'Archetype Alpha',
    tags: ['Non-HTTPS Protocol', 'Gov / Edu Domain', 'Password Form Present'],
    icons: ['🔓', '🏛', '🔑'],
    insight: 'Sites mimicking government or educational institutions — without HTTPS — that include credential harvesting forms. These exploit institutional trust while deploying credential theft.',
  },
  {
    letter: 'B', name: 'Archetype Beta',
    tags: ['HTTPS Enabled', 'Gov / Edu Domain', 'Social Network Linked'],
    icons: ['🔒', '🏛', '📱'],
    insight: 'The most deceptive archetype. HTTPS creates false security, institutional domain lends authority, social links add social proof. A triple-layer trust manipulation.',
  },
  {
    letter: 'C', name: 'Archetype Gamma',
    tags: ['Non-HTTPS Protocol', 'Gov / Edu Domain', 'Social Network Linked'],
    icons: ['🔓', '🏛', '📱'],
    insight: 'Similar to Alpha but adds social network integration. Relies on domain trust and social proof to drive engagement before redirecting to malicious content.',
  },
]

export default function Investigate() {
  return (
    <div className={styles.page}>
      <div className={styles.hero}>
        <div className="section-inner">
          <div className="section-tag">Blind Spot Investigation Center</div>
          <h1 className={styles.heroTitle}>
            Forensic-grade failure<br />investigation.
          </h1>
          <p className={styles.heroSub}>
            Only 3 failures were discovered across 47,159 evaluated samples.
            Every one is documented as a high-priority security incident with full forensic detail.
          </p>
        </div>
      </div>

      <section className={styles.section}>
        <div className="section-inner">
          <h2 className={styles.sectionTitle}>Critical Blind Spots</h2>
          <BlindspotSection />
        </div>
      </section>

      <section className={`${styles.section} ${styles.sectionAlt}`}>
        <div className="section-inner">
          <div className="section-tag">Failure Archetype Discovery</div>
          <h2 className={styles.sectionTitle}>Not random errors. Recurring patterns.</h2>
          <p className={styles.sectionSub}>
            Failure archetype analysis revealed three distinct attack patterns
            that consistently challenge the model — all involving institutional domain trust.
          </p>
          <div className={styles.archetypeGrid}>
            {ARCHETYPES.map((a) => (
              <div key={a.letter} className={styles.archetypeCard}>
                <div className={styles.archetypeLetter}>{a.letter}</div>
                <div className={styles.archetypeName}>{a.name}</div>
                <div className={styles.archetypeTags}>
                  {a.tags.map((tag, i) => (
                    <div key={tag} className={styles.archTag}>
                      <span>{a.icons[i]}</span>
                      <span>{tag}</span>
                    </div>
                  ))}
                </div>
                <p className={styles.archetypeInsight}>{a.insight}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className={styles.section}>
        <div className="section-inner">
          <div className="section-tag">Reliability Analysis Center</div>
          <h2 className={styles.sectionTitle}>When confidence misleads.</h2>
          <ReliabilitySection />
        </div>
      </section>
    </div>
  )
}
