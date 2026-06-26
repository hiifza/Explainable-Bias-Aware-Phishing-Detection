import { useRef, useEffect, useState } from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import { useAppStore } from '@/store'
import { gsap } from 'gsap'
import styles from './Nav.module.css'

const NAV_LINKS = [
  { to: '/',            label: 'Analyze' },
  { to: '/investigate', label: 'Investigate' },
  { to: '/research',    label: 'Research' },
  { to: '/learn',       label: 'Learn' },
  { to: '/about',       label: 'About' },
]

export default function Nav() {
  const { theme, toggleTheme, mode, toggleMode } = useAppStore()
  const navRef = useRef<HTMLElement>(null)
  const [scrolled, setScrolled] = useState(false)
  const location = useLocation()

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20)
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  // Magnetic button effect
  const handleMouseMove = (e: React.MouseEvent<HTMLButtonElement>) => {
    const btn = e.currentTarget
    const rect = btn.getBoundingClientRect()
    const cx = rect.left + rect.width / 2
    const cy = rect.top + rect.height / 2
    const dx = (e.clientX - cx) * 0.28
    const dy = (e.clientY - cy) * 0.28
    gsap.to(btn, { x: dx, y: dy, duration: 0.3, ease: 'power2.out' })
  }
  const handleMouseLeave = (e: React.MouseEvent<HTMLButtonElement>) => {
    gsap.to(e.currentTarget, { x: 0, y: 0, duration: 0.5, ease: 'elastic.out(1, 0.5)' })
  }

  return (
    <nav
      ref={navRef}
      className={`${styles.nav} ${scrolled ? styles.scrolled : ''}`}
    >
      <NavLink to="/" className={styles.logo}>
        <span className={styles.logoDot} />
        <span className={styles.logoWordmark}>PhishGuard</span>
        <span className={styles.logoBadge}>INTELLIGENCE</span>
      </NavLink>

      <div className={styles.links}>
        {NAV_LINKS.map(({ to, label }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `${styles.link} ${isActive ? styles.linkActive : ''}`
            }
            end={to === '/'}
          >
            {label}
          </NavLink>
        ))}
      </div>

      <div className={styles.actions}>
        <button
          className={`${styles.modeBtn} ${mode === 'expert' ? styles.modeBtnExpert : ''}`}
          onClick={toggleMode}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          aria-label="Toggle expert mode"
        >
          <span className={styles.modeDot} />
          {mode === 'beginner' ? 'Beginner' : 'Expert'}
        </button>

        <button
          className={styles.themeBtn}
          onClick={toggleTheme}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
        >
          {theme === 'dark' ? '◐' : '○'}
        </button>
      </div>
    </nav>
  )
}
