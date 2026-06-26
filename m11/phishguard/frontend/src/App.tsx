import { Routes, Route } from 'react-router-dom'
import { useEffect } from 'react'
import { useAppStore } from '@/store'
import Nav from '@/components/layout/Nav'
import Home from '@/pages/Home'
import Investigate from '@/pages/Investigate'
import Research from '@/pages/Research'
import Learn from '@/pages/Learn'
import About from '@/pages/About'
import ScanResult from '@/components/scanner/ScanResult'
import PageLoader from '@/components/ui/PageLoader'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

gsap.registerPlugin(ScrollTrigger)

export default function App() {
  const theme = useAppStore((s) => s.theme)
  const showResult = useAppStore((s) => s.showResult)

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
  }, [theme])

  return (
    <>
      <PageLoader />
      <Nav />
      <main>
        <Routes>
          <Route path="/"            element={<Home />} />
          <Route path="/investigate" element={<Investigate />} />
          <Route path="/research"    element={<Research />} />
          <Route path="/learn"       element={<Learn />} />
          <Route path="/about"       element={<About />} />
        </Routes>
      </main>
      {showResult && <ScanResult />}
    </>
  )
}
