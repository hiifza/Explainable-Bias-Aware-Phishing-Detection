// Learn.tsx
import styles from './SubPage.module.css'

const TOPICS = [
  { icon: '🎣', title: 'Email Phishing',        desc: 'Emails impersonating banks, tech companies, or government agencies to steal credentials or install malware. Over 80% of all phishing attempts start here.' },
  { icon: '📱', title: 'Smishing (SMS)',          desc: '"Your package is delayed, click here." Short urgent messages bypass email filters entirely and exploit mobile users who click reflexively.' },
  { icon: '🏛', title: 'Brand Impersonation',    desc: 'Fake Amazon, PayPal, Microsoft, or banking portals designed to be visually indistinguishable from the real thing. Often combined with Gov/Edu domain mimicry.' },
  { icon: '🤖', title: 'AI-Generated Phishing',  desc: 'Large language models now generate flawless, personalized phishing content at scale. Grammar errors — once a reliable signal — are rapidly disappearing.' },
  { icon: '📷', title: 'QR Code Phishing',       desc: '"Quishing": malicious QR codes embedded in physical objects or PDFs that redirect to phishing sites. Cannot be pre-inspected by humans or most filters.' },
  { icon: '🧠', title: 'Social Engineering',     desc: 'Psychological manipulation: urgency ("Act now or lose access"), authority ("IT Security Team"), fear, and reward. PhishGuard detects these signals automatically.' },
  { icon: '🎭', title: 'Deepfake Scams',         desc: 'AI-generated voice or video impersonating executives or family members to authorize wire transfers or extract sensitive information.' },
  { icon: '💼', title: 'Fake Job Scams',         desc: 'Fraudulent job offers requiring "background check fees" or personal document submissions. Often targets recent graduates and job seekers.' },
]

const TIPS = [
  { tip: 'Check the exact domain',       detail: 'paypa1.com vs paypal.com — one character can be the difference between safe and compromised.' },
  { tip: 'HTTPS ≠ Safe',                 detail: 'Attackers also use HTTPS. It means the connection is encrypted, not that the site is trustworthy.' },
  { tip: 'Hover before clicking',        detail: 'On desktop, hover over links to see the real destination URL in your browser\'s status bar.' },
  { tip: 'Be suspicious of urgency',     detail: '"Act within 24 hours or your account closes" is a manipulation tactic, not a real deadline.' },
  { tip: 'Verify via official channels', detail: 'Never call numbers or click links in suspicious messages. Go directly to the official website.' },
  { tip: 'Use a password manager',       detail: 'Password managers recognize legitimate domains and refuse to autofill on fake sites.' },
]

export function Learn() {
  return (
    <div className={styles.page}>
      <div className={styles.hero}>
        <div className="section-inner">
          <div className="section-tag">Cyber Knowledge Hub</div>
          <h1 className={styles.heroTitle}>Understanding the<br />threat landscape.</h1>
          <p className={styles.heroSub}>
            Phishing attacks succeed because they exploit human psychology, not just technical
            vulnerabilities. Understanding the techniques is the first line of defense.
          </p>
        </div>
      </div>

      <section className={styles.section}>
        <div className="section-inner">
          <h2 className={styles.sectionTitle}>Attack Techniques</h2>
          <div className={styles.eduGrid}>
            {TOPICS.map((t) => (
              <div key={t.title} className={styles.eduCard}>
                <div className={styles.eduIcon}>{t.icon}</div>
                <div className={styles.eduTitle}>{t.title}</div>
                <p className={styles.eduDesc}>{t.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className={`${styles.section} ${styles.sectionAlt}`}>
        <div className="section-inner">
          <h2 className={styles.sectionTitle}>Six Rules for Safer Browsing</h2>
          <div className={styles.tipsList}>
            {TIPS.map((t, i) => (
              <div key={t.tip} className={styles.tipCard}>
                <div className={styles.tipNum}>{String(i + 1).padStart(2, '0')}</div>
                <div>
                  <div className={styles.tipTitle}>{t.tip}</div>
                  <div className={styles.tipDetail}>{t.detail}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  )
}
export default Learn
