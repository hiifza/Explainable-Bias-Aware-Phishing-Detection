import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export type Theme = 'dark' | 'light'
export type Mode  = 'beginner' | 'expert'

export interface ScanResult {
  url: string
  trust_score: number
  risk_level: 'SAFE' | 'LOW_RISK' | 'SUSPICIOUS' | 'HIGH_RISK' | 'CRITICAL'
  prediction: 'legitimate' | 'phishing'
  confidence: number
  phishing_probability: number
  legitimate_probability: number
  threat_signals: ThreatSignal[]
  shap_features: ShapFeature[]
  lime_features: LimeFeature[]
  human_explanation: string
  attacker_simulation: AttackerSignal[]
  scam_psychology: ScamPsychology[]
  recommended_actions: string[]
  reliability_zone: 'GREEN' | 'YELLOW' | 'RED'
  explanation_agreement: number
  brand_impersonation: BrandImpersonation | null
  domain_analysis: DomainAnalysis
  timestamp: string
}

export interface ThreatSignal {
  id: string
  label: string
  human_label: string
  description: string
  impact: 'high' | 'medium' | 'low'
  value: string | number
  direction: 'phishing' | 'legitimate' | 'neutral'
}

export interface ShapFeature {
  feature: string
  value: number
  contribution: number
  direction: 'increases_risk' | 'decreases_risk'
  human_label: string
}

export interface LimeFeature {
  feature: string
  weight: number
  human_label: string
}

export interface AttackerSignal {
  signal: string
  explanation: string
  danger_level: 'high' | 'medium' | 'low'
}

export interface ScamPsychology {
  tactic: string
  description: string
  detected: boolean
}

export interface BrandImpersonation {
  target_brand: string
  confidence: number
  indicators: string[]
}

export interface DomainAnalysis {
  domain: string
  tld: string
  is_https: boolean
  domain_length: number
  subdomain_count: number
  has_ip: boolean
  suspicious_keywords: string[]
  letter_ratio: number
  digit_count: number
  special_chars: number
}

interface AppStore {
  // Theme
  theme: Theme
  setTheme: (t: Theme) => void
  toggleTheme: () => void

  // Mode
  mode: Mode
  setMode: (m: Mode) => void
  toggleMode: () => void

  // Nav
  activeSection: string
  setActiveSection: (s: string) => void

  // Scanner
  scanUrl: string
  setScanUrl: (url: string) => void
  scanResult: ScanResult | null
  setScanResult: (r: ScanResult | null) => void
  isScanning: boolean
  setIsScanning: (v: boolean) => void
  scanError: string | null
  setScanError: (e: string | null) => void
  showResult: boolean
  setShowResult: (v: boolean) => void
}

export const useAppStore = create<AppStore>()(
  persist(
    (set, get) => ({
      theme: 'dark',
      setTheme: (theme) => {
        document.documentElement.setAttribute('data-theme', theme)
        set({ theme })
      },
      toggleTheme: () => {
        const next = get().theme === 'dark' ? 'light' : 'dark'
        document.documentElement.setAttribute('data-theme', next)
        set({ theme: next })
      },

      mode: 'beginner',
      setMode: (mode) => set({ mode }),
      toggleMode: () =>
        set((s) => ({ mode: s.mode === 'beginner' ? 'expert' : 'beginner' })),

      activeSection: 'home',
      setActiveSection: (activeSection) => set({ activeSection }),

      scanUrl: '',
      setScanUrl: (scanUrl) => set({ scanUrl }),
      scanResult: null,
      setScanResult: (scanResult) => set({ scanResult }),
      isScanning: false,
      setIsScanning: (isScanning) => set({ isScanning }),
      scanError: null,
      setScanError: (scanError) => set({ scanError }),
      showResult: false,
      setShowResult: (showResult) => set({ showResult }),
    }),
    {
      name: 'phishguard-prefs',
      partialize: (s) => ({ theme: s.theme, mode: s.mode }),
      onRehydrateStorage: () => (state) => {
        if (state?.theme) {
          document.documentElement.setAttribute('data-theme', state.theme)
        }
      },
    }
  )
)
