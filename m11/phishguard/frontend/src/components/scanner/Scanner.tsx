import { useRef, useState, useCallback } from 'react'
import { useAppStore } from '@/store'
import { analyzeUrl } from '@/lib/api'
import { gsap } from 'gsap'
import styles from './Scanner.module.css'

const EXAMPLE_URLS = [
  'https://paypa1-secure-login.verification-portal.tk',
  'https://www.google.com',
  'http://192.168.1.1/admin/login',
  'https://microsoft-account-verify.xyz/update',
]

export default function Scanner() {
  const { scanUrl, setScanUrl, setIsScanning, setScanResult, setScanError, setShowResult, isScanning } = useAppStore()
  const inputRef  = useRef<HTMLInputElement>(null)
  const btnRef    = useRef<HTMLButtonElement>(null)
  const wrapRef   = useRef<HTMLDivElement>(null)
  const [focused, setFocused] = useState(false)

  const runScan = useCallback(async () => {
    const url = scanUrl.trim()
    if (!url || isScanning) return

    // Button pulse
    gsap.fromTo(btnRef.current,
      { scale: 0.95 },
      { scale: 1, duration: 0.4, ease: 'elastic.out(1, 0.5)' }
    )

    setIsScanning(true)
    setScanError(null)

    try {
      const result = await analyzeUrl(url)
      setScanResult(result)
      setShowResult(true)
    } catch (err: any) {
      const msg = err?.response?.data?.detail ?? 'Analysis failed. Check the URL and try again.'
      setScanError(msg)
    } finally {
      setIsScanning(false)
    }
  }, [scanUrl, isScanning, setIsScanning, setScanResult, setScanError, setShowResult])

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') runScan()
  }

  const handleExample = (url: string) => {
    setScanUrl(url)
    inputRef.current?.focus()
    gsap.fromTo(wrapRef.current,
      { borderColor: 'var(--border-subtle)' },
      { borderColor: 'var(--accent)', duration: 0.4, yoyo: true, repeat: 1 }
    )
  }

  return (
    <div className={styles.scanner}>
      <div
        ref={wrapRef}
        className={`${styles.inputWrap} ${focused ? styles.focused : ''}`}
      >
        <span className={styles.icon} aria-hidden="true">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
            <circle cx="11" cy="11" r="8"/>
            <path d="M21 21l-4.35-4.35"/>
          </svg>
        </span>
        <input
          ref={inputRef}
          type="url"
          className={styles.input}
          placeholder="Paste any URL to analyze — https://example.com"
          value={scanUrl}
          onChange={(e) => setScanUrl(e.target.value)}
          onKeyDown={handleKey}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          spellCheck={false}
          autoComplete="off"
          aria-label="URL to analyze"
        />
        <button
          ref={btnRef}
          className={styles.btn}
          onClick={runScan}
          disabled={isScanning || !scanUrl.trim()}
          aria-label="Analyze URL"
        >
          {isScanning ? (
            <span className={styles.spinner} aria-hidden="true" />
          ) : (
            <>
              <span>Analyze</span>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <path d="M5 12h14M12 5l7 7-7 7"/>
              </svg>
            </>
          )}
        </button>
      </div>

      {isScanning && (
        <div className={styles.scanStatus} role="status" aria-live="polite">
          <span className={styles.scanDot} />
          <span className={styles.scanText}>
            Analyzing URL through PhishGuard Intelligence Engine…
          </span>
        </div>
      )}

      <div className={styles.examples}>
        <span className={styles.examplesLabel}>Try an example:</span>
        {EXAMPLE_URLS.map((url) => (
          <button
            key={url}
            className={styles.exampleBtn}
            onClick={() => handleExample(url)}
            title={url}
          >
            {url.length > 42 ? url.slice(0, 42) + '…' : url}
          </button>
        ))}
      </div>
    </div>
  )
}
